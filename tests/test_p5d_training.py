"""P5d objective, exact ordinary compatibility, and checkpoint rejection."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import copy
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.nn.functional as F

from experiments.p5a_training import GPUTrainer as P5aTrainer
from experiments.p5a_checkpoint import capture as p5a_capture
from experiments.p5d_objective import objective_loss
from experiments.p5d_training import GPUTrainer, _validated_config
from experiments.p5d_checkpoint import META, capture, load_checkpoint, save_checkpoint, versions


def config(objective="label_balanced", batch_size=31):
    return dict(seed=314500, epsilon=.1, lr=.001, batch_size=batch_size,
                chunk_size=1024, num_envs=1, mode="selfplay", objective=objective)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def rebind_checkpoint(path, mutate):
    payload = torch.load(path / "checkpoint.pt", weights_only=True)
    mutate(payload)
    torch.save(payload, path / "checkpoint.pt")
    manifest = {key: payload[key] for key in META}
    data = (path / "checkpoint.pt").read_bytes()
    manifest.update(sha256=sha256(data).hexdigest(), bytes=len(data))
    (path / "manifest.json").write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")


class ConfigTests(unittest.TestCase):
    def test_objective_required_and_strict(self):
        self.assertEqual(_validated_config(json.loads(json.dumps(config()))), config())
        self.assertEqual(_validated_config(config("ordinary"))["objective"], "ordinary")
        invalid = [config(True), config(False), config("balanced"),
                   {k: v for k, v in config().items() if k != "objective"},
                   {**config(), "extra": 1}]
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                _validated_config(candidate)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ObjectiveTests(unittest.TestCase):
    def test_imbalanced_formula_and_gradient(self):
        q = torch.tensor([-.25, .5, .75, -1.5], device="cuda", requires_grad=True)
        y = torch.tensor([1., 1., 1., -1.], device="cuda")
        loss, stats = objective_loss(q, y, "label_balanced")
        expected = .5 * ((q[:3] - 1).square().mean()) + .5 * ((q[3:] + 1).square().mean())
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        self.assertEqual(stats, dict(objective="label_balanced", n=4, positive=3, negative=1,
                                     positive_weight=4/6, negative_weight=2.,
                                     single_class=False, loss=float(loss.item())))
        loss.backward()
        expected_grad = torch.tensor([(-.25-1)/3, (.5-1)/3, (.75-1)/3, (-1.5+1)], device="cuda")
        torch.testing.assert_close(q.grad, expected_grad, rtol=0, atol=1e-7)

    def test_single_class_and_singleton_use_native_mse(self):
        for labels in ([1., 1., 1.], [-1., -1.], [1.], [-1.]):
            with self.subTest(labels=labels):
                q = torch.linspace(-.5, .5, len(labels), device="cuda", requires_grad=True)
                y = torch.tensor(labels, device="cuda")
                loss, stats = objective_loss(q, y, "label_balanced")
                expected = F.mse_loss(q, y)
                self.assertTrue(torch.equal(loss, expected))
                self.assertTrue(stats["single_class"])
                self.assertEqual(stats["positive_weight"], 1. if labels[0] == 1 else 0.)
                self.assertEqual(stats["negative_weight"], 1. if labels[0] == -1 else 0.)
                loss.backward()
                torch.testing.assert_close(q.grad, 2 * (q.detach() - y) / len(labels), rtol=0, atol=1e-7)

    def test_balanced_sign_swap(self):
        q = torch.tensor([.1, -.2, .3, .4, -.5], device="cuda", requires_grad=True)
        y = torch.tensor([1., -1., -1., -1., 1.], device="cuda")
        loss, stats = objective_loss(q, y, "label_balanced")
        loss.backward()
        other = (-q.detach()).requires_grad_(True)
        swapped, swapped_stats = objective_loss(other, -y, "label_balanced")
        swapped.backward()
        self.assertEqual(loss.item(), swapped.item())
        torch.testing.assert_close(q.grad, -other.grad, rtol=0, atol=0)
        self.assertEqual(stats["positive"], swapped_stats["negative"])
        self.assertEqual(stats["positive_weight"], swapped_stats["negative_weight"])

    def test_ordinary_exact_native_mse(self):
        q = torch.tensor([.1, -.2, .3], device="cuda", requires_grad=True)
        y = torch.tensor([1., -1., 1.], device="cuda")
        native = F.mse_loss(q, y)
        actual, stats = objective_loss(q, y, "ordinary")
        self.assertTrue(torch.equal(actual, native))
        self.assertEqual(stats["objective"], "ordinary")
        self.assertEqual((stats["positive_weight"], stats["negative_weight"]), (1., 1.))
        native.backward(retain_graph=True)
        gradient = q.grad.clone()
        q.grad = None
        actual.backward()
        self.assertTrue(torch.equal(q.grad, gradient))

    def test_invalid_tensors_and_objective(self):
        q = torch.tensor([0., .5], device="cuda", requires_grad=True)
        y = torch.tensor([1., -1.], device="cuda")
        cases = [(q, y, True), (q, y, "other"), (q.cpu(), y, "ordinary"),
                 (q.double(), y, "ordinary"), (q[:0], y[:0], "ordinary"),
                 (q.reshape(1, 2), y.reshape(1, 2), "ordinary"),
                 (q, y[:1], "ordinary"), (q, y.detach().clone().requires_grad_(True), "ordinary"),
                 (q, torch.tensor([1., 0.], device="cuda"), "ordinary"),
                 (q, torch.tensor([1., float("nan")], device="cuda"), "ordinary"),
                 (torch.tensor([0., float("inf")], device="cuda"), y.detach(), "ordinary")]
        for predictions, targets, objective in cases:
            with self.subTest(objective=objective, shape=tuple(predictions.shape)), self.assertRaises(ValueError):
                objective_loss(predictions, targets, objective)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TrainingTests(unittest.TestCase):
    def test_short_final_batch_and_mean_loss(self):
        trainer = GPUTrainer(config(batch_size=31))
        row = trainer.train_wave([(108900, 2, 0)])
        batches = row["batches"]
        self.assertGreater(len(batches), 1)
        self.assertLess(batches[-1]["n"], 31)
        self.assertEqual([b["start"] for b in batches], list(range(0, row["samples"], 31)))
        self.assertEqual(sum(b["n"] for b in batches), row["samples"])
        self.assertEqual(row["objective"], "label_balanced")
        self.assertEqual(row["mean_loss"], sum(b["loss"] * b["n"] for b in batches) / row["samples"])
        for batch in batches:
            self.assertEqual(batch["positive"] + batch["negative"], batch["n"])

    def test_ordinary_same_seed_deal_model_adam_rng_and_log(self):
        cfg = config("ordinary", batch_size=256)
        deal = [(108901, 3, 1)]
        original = P5aTrainer({k: v for k, v in cfg.items() if k != "objective"})
        old_row = original.train_wave(deal)
        old = p5a_capture(original)
        current = GPUTrainer(cfg)
        row = current.train_wave(deal)
        new = capture(current)
        for key in ("model", "optimizer", "policy_rng", "torch_rng", "cuda_rng",
                    "episodes", "updates", "waves", "used_deal_seeds", "frozen", "pool_sha256"):
            self.assertTrue(equal(old[key], new[key]), key)
        stripped = {k: v for k, v in row.items() if k not in ("objective", "batches")}
        self.assertTrue(equal(old_row, stripped))
        self.assertEqual(row["batches"][0]["start"], 0)

    def test_failed_adam_step_rejects_continue_and_save(self):
        deals = [(108902, 4, 2)]
        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / "initial"
            reference = GPUTrainer(config())
            save_checkpoint(reference, initial)
            expected_row = reference.train_wave(deals)
            expected_state = capture(reference)
            damaged = load_checkpoint(initial)
            original = damaged.optimizer.step
            def fail_after_step(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("injected after Adam")
            with patch.object(damaged.optimizer, "step", fail_after_step):
                with self.assertRaisesRegex(RuntimeError, "injected after Adam"):
                    damaged.train_wave(deals)
            self.assertFalse(damaged.ready_for_checkpoint)
            with self.assertRaisesRegex(RuntimeError, "successful wave boundary"):
                damaged.train_wave([(108903, 4, 2)])
            with self.assertRaises(ValueError):
                save_checkpoint(damaged, Path(directory) / "failed")
            restored = load_checkpoint(initial)
            self.assertTrue(equal(expected_row, restored.train_wave(deals)))
            self.assertTrue(equal(expected_state, capture(restored)))

    def test_checkpoint_objective_and_old_version_tamper_after_rehash(self):
        self.assertEqual(versions()["checkpoint"], "gd-p5d-checkpoint-v1")
        self.assertEqual(versions()["training"], "gd-p5d-wave-v1")
        self.assertEqual(versions()["pool"], "gd-p5a-pool-v1")
        trainer = GPUTrainer(config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good"
            manifest = save_checkpoint(trainer, good)
            self.assertTrue(equal(capture(trainer), capture(load_checkpoint(good, manifest["sha256"]))))
            for name, mutation in (
                ("objective", lambda p: p["config"].update(objective="ordinary")),
                ("old_version", lambda p: p["versions"].update(checkpoint="gd-p5a-checkpoint-v1")),
            ):
                path = root / name
                save_checkpoint(trainer, path)
                rebind_checkpoint(path, mutation)
                torch.manual_seed(888)
                torch.cuda.manual_seed_all(889)
                before = (torch.get_rng_state().clone(),
                          [state.clone() for state in torch.cuda.get_rng_state_all()])
                with self.subTest(name=name), self.assertRaises(ValueError):
                    load_checkpoint(path)
                self.assertTrue(equal(before, (torch.get_rng_state(), torch.cuda.get_rng_state_all())))


if __name__ == "__main__":
    unittest.main()
