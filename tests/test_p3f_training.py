"""Artifact digest contracts and explicitly opted-in tiny CUDA reset diagnostic."""
from hashlib import sha256
import json
import gzip
import os
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.p3f_training import _dmc_phase, model_digest, reset_phase2, save_phase1, train


class _TensorBytes:
    """Tiny NumPy-byte-compatible tensor stand-in for a torch-free digest test."""
    dtype = "torch.float32"

    def __init__(self, values):
        self.values = tuple(values)
        self.shape = (len(values),)

    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self
    def tobytes(self): return struct.pack("<" + "f" * len(self.values), *self.values)


class DigestTests(unittest.TestCase):
    def test_order_stable_and_parameter_change_changes_hash(self):
        a, b = _TensorBytes([1, 2]), _TensorBytes([3])
        value = model_digest({"b": b, "a": a})
        self.assertEqual(value, model_digest({"a": a, "b": b}))
        self.assertEqual(len(value), 64)
        self.assertNotEqual(value, model_digest({"a": _TensorBytes([1, 2.5]), "b": b}))
        self.assertNotEqual(value, model_digest({"different_name": a, "b": b}))
        alternate_dtype = _TensorBytes([1, 2])
        alternate_dtype.dtype = "different"
        self.assertNotEqual(value, model_digest({"a": alternate_dtype, "b": b}))
        # Independent framing computation pins metadata, shape and raw byte use.
        digest = sha256()
        for name, tensor in (("a", a), ("b", b)):
            meta = json.dumps([name, tensor.dtype, list(tensor.shape)],
                              ensure_ascii=False, allow_nan=False, sort_keys=True).encode()
            data = tensor.tobytes()
            digest.update(len(meta).to_bytes(8, "big") + meta)
            digest.update(len(data).to_bytes(8, "big") + data)
        self.assertEqual(value, digest.hexdigest())

    def test_invalid_job_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                train(Path(directory), "teacher-999")
            self.assertEqual(list(Path(directory).iterdir()), [])


@unittest.skipUnless(os.environ.get("RUN_P3F_CUDA") == "1", "CUDA diagnostic requires explicit opt-in")
class CUDAResetTests(unittest.TestCase):
    def test_dmc_instrumentation_is_bitwise_transparent(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from guandan.env import HandEnv
        from guandan.types import Action
        import guandan_gpu.training as training
        config = dict(seed=314389, epsilon=.1, lr=.001, batch_size=256,
                      chunk_size=1024, num_envs=4)
        original_env = training.HandEnv
        original = training.GPUTrainer(config)
        expected = original.train_wave([(108600 + i, 2 + (8600 + i) % 13,
                                         (8600 + i) % 4) for i in range(4)])
        instrumented = training.GPUTrainer(config)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            stats = _dmc_phase(instrumented, 8600, 8604, directory / "waves.jsonl",
                               directory / "replays.jsonl.gz")
            self.assertIs(training.HandEnv, original_env)
            wave = json.loads((directory / "waves.jsonl").read_text(encoding="utf-8"))
            wave.pop("elapsed_seconds")
            self.assertEqual(wave, expected)
            with gzip.open(directory / "replays.jsonl.gz", "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 4)
            independent_total = 0
            for i, row in enumerate(rows):
                self.assertEqual(row["seed"], 108600 + i)
                self.assertEqual(row["steps"], row["samples"])
                replay = row["replay"]
                env = HandEnv()
                env.reset(row["seed"], initial_level=row["level"],
                          starting_player=row["starting_player"])
                self.assertEqual([list(hand) for hand in env.state.initial_hands],
                                 replay["initial_hands"])
                candidates = 0
                for step in replay["steps"]:
                    candidates += len(env.legal_actions(step["player"]))
                    env.step(step["player"], Action.from_dict(step["action"]),
                             state_version=step["state_version"])
                    self.assertEqual(env.state_digest(), step["digest"])
                self.assertTrue(env.state.terminal)
                self.assertEqual(env.state_digest(), row["terminal_digest"])
                self.assertEqual(candidates, row["legal_candidates"])
                independent_total += candidates
            self.assertEqual(stats["legal_candidates"], independent_total)
            self.assertEqual(stats["samples"], expected["samples"])
        for key, value in original.model.state_dict().items():
            self.assertTrue(torch.equal(value, instrumented.model.state_dict()[key]), key)
        expected_optimizer = original.optimizer.state_dict()
        actual_optimizer = instrumented.optimizer.state_dict()
        self.assertEqual(expected_optimizer["param_groups"], actual_optimizer["param_groups"])
        self.assertEqual(set(expected_optimizer["state"]), set(actual_optimizer["state"]))
        for parameter, state in expected_optimizer["state"].items():
            self.assertEqual(set(state), set(actual_optimizer["state"][parameter]))
            for key, value in state.items():
                actual = actual_optimizer["state"][parameter][key]
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, actual), (parameter, key))
                else:
                    self.assertEqual(value, actual)

    def test_tiny_teacher_update_raw_save_fresh_reset(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from experiments.p3f_teacher import fit_batch
        from guandan.types import Action, PlayerObservation
        from guandan_gpu.training import GPUTrainer
        config = dict(seed=314380, epsilon=.1, lr=.001, batch_size=256,
                      chunk_size=1024, num_envs=4)
        trainer = GPUTrainer(config)
        initial_hash = model_digest(trainer.model)
        initial_rng = sha256(repr(trainer.rng.getstate()).encode()).hexdigest()
        obs = PlayerObservation(0, (0, 54), 2, 0, (2, 3, 3, 3), (),
                                None, None, (), (), 0, False, None)
        result = fit_batch(trainer.model, trainer.optimizer,
                           [(obs, [Action("single", (0,), 2), Action("single", (54,), 2),
                                   Action("pair", (0, 54), 2)])],
                           chunk_size=1)
        self.assertEqual(result["scored_candidates"], 3)
        trained_hash = model_digest(trainer.model)
        self.assertNotEqual(initial_hash, trained_hash)
        self.assertTrue(trainer.optimizer.state)
        # Perturb policy RNG to prove it is reset, rather than restored from phase1.
        trainer.rng.random()
        with tempfile.TemporaryDirectory() as directory:
            phase1 = Path(directory)
            manifest = save_phase1(trainer, phase1, "teacher", 314380, 1, hands=0)
            saved = torch.load(phase1 / "raw.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["phase1_updates"], 1)
            self.assertTrue(saved["optimizer"]["state"])
            fresh, reset = reset_phase2(phase1, config)
            self.assertEqual(model_digest(fresh.model), trained_hash)
            self.assertEqual(manifest["model_sha256"], trained_hash)
            self.assertEqual(reset["phase2_initial_model_sha256"], trained_hash)
            self.assertEqual(reset["policy_rng_repr_sha256"], initial_rng)
            self.assertTrue(reset["optimizer_empty"])
            self.assertEqual(reset["initial_counters"], dict(episodes=0, waves=0, updates=0))
            self.assertFalse(fresh.optimizer.state)
            self.assertEqual(fresh.used_deal_seeds, [])
            for key, value in trainer.model.state_dict().items():
                self.assertTrue(torch.equal(value, fresh.model.state_dict()[key]))
            data = (phase1 / "raw.pt").read_bytes()
            (phase1 / "raw.pt").write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "SHA/size"):
                reset_phase2(phase1, config)


if __name__ == "__main__":
    unittest.main()
