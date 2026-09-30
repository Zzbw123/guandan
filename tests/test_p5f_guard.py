"""P5f spawned guard CPU faults; optional real CUDA checkpoint integration."""
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest import TestCase, main, mock, skipUnless

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from guandan.types import Action, PlayerObservation
from experiments.p5f_guard import (GPUInferenceGuard, GuardError, GuardTimeout,
                                   _select_raw_logits)


PIN = "a" * 64


def observation():
    return PlayerObservation(0, (0,), 2, 0, (1, 1, 1, 1), (), None,
                             None, (), (), 0, False, None)


class GuardTests(TestCase):
    def make_guard(self, mode="echo", **kwargs):
        return GPUInferenceGuard(Path("unused-p5f-checkpoint"), PIN,
                                 startup_timeout=10, _test_mode=mode, **kwargs)

    def assert_reaped(self, guard):
        self.assertTrue(guard.closed)
        self.assertFalse(guard.is_alive())
        self.assertIsNotNone(guard.worker_pid)
        self.assertIsNotNone(guard.worker_exitcode)
        self.assertNotIn(guard.worker_pid, [p.pid for p in mp.active_children()])
        with self.assertRaises(GuardError):
            guard.act(observation(), [Action("pass")])
        guard.close()

    def test_8769_candidates_preserve_full_order_and_persistent_worker(self):
        with self.make_guard(decision_timeout=5) as guard:
            pid = guard.worker_pid
            legal = [Action("single", (i,), i) for i in range(8769)]
            chosen, meta = guard.act(observation(), legal)
            self.assertIs(chosen, legal[-1])
            self.assertEqual(meta["scored_candidates"], 8769)
            self.assertEqual(meta["device"], {"type": "test", "mode": "echo"})
            self.assertGreaterEqual(meta["roundtrip_ms"], meta["inference_ms"])
            chosen, meta = guard.act(observation(), legal[:2])
            self.assertIs(chosen, legal[1])
            self.assertEqual(meta["scored_candidates"], 2)
            self.assertEqual(pid, guard.worker_pid)
            self.assertTrue(guard.is_alive())
        self.assert_reaped(guard)

    def test_startup_deadline_reaps_spawned_worker(self):
        before = {p.pid for p in mp.active_children()}
        started = time.perf_counter()
        with self.assertRaises(GuardTimeout):
            GPUInferenceGuard(Path("unused"), PIN, startup_timeout=.15,
                              _test_mode="hang_start")
        self.assertLess(time.perf_counter() - started, 3)
        self.assertEqual(before, {p.pid for p in mp.active_children()})

    def test_decision_deadline_bounds_large_upload_and_reaps(self):
        guard = self.make_guard("hang_act", decision_timeout=.15)
        legal = [Action("single", (i,), i) for i in range(100000)]
        started = time.perf_counter()
        with self.assertRaises(GuardTimeout):
            guard.act(observation(), legal)
        self.assertLess(time.perf_counter() - started, 3)
        self.assert_reaped(guard)

    def test_worker_crash_is_fail_closed(self):
        guard = self.make_guard("crash_act", decision_timeout=3)
        with self.assertRaises(GuardError):
            guard.act(observation(), [Action("pass")])
        self.assert_reaped(guard)
        self.assertEqual(guard.worker_exitcode, 23)

    def test_invalid_index_is_fail_closed(self):
        guard = self.make_guard("bad_index")
        with self.assertRaisesRegex(GuardError, "illegal candidate index"):
            guard.act(observation(), [Action("pass")])
        self.assert_reaped(guard)

    def test_visible_input_contract_and_malformed_reply_fail_closed(self):
        for obs, legal in (({"deal_seed": 123}, [Action("pass")]),
                           (observation(), []),
                           (observation(), (Action("pass"),)),
                           (observation(), [object()])):
            with self.subTest(obs=type(obs).__name__, legal=type(legal).__name__):
                guard = self.make_guard()
                with self.assertRaises((TypeError, ValueError)):
                    guard.act(obs, legal)
                self.assert_reaped(guard)
        for response in (("error", "RuntimeError", "injected"), None,
                         ("result", True, 1, 0., {"type": "test", "mode": "echo"}),
                         ("result", 0, 2, 0., {"type": "test", "mode": "echo"}),
                         ("result", 0, 1, float("nan"), {"type": "test", "mode": "echo"}),
                         ("result", 0, 1, 0., {"type": "cuda"})):
            with self.subTest(response=response):
                guard = self.make_guard()
                with mock.patch.object(guard, "_receive", return_value=response):
                    with self.assertRaises(GuardError):
                        guard.act(observation(), [Action("pass")])
                self.assert_reaped(guard)

    def test_pins_timeouts_and_test_modes_fail_before_spawn(self):
        for pin in (None, "", "g" * 64, "a" * 63):
            with self.assertRaises(ValueError):
                GPUInferenceGuard(Path("unused"), pin, _test_mode="echo")
        for pin in ("", "g" * 64, "a" * 63):
            with self.assertRaises(ValueError):
                GPUInferenceGuard(Path("unused"), PIN,
                                  expected_manifest_sha256=pin, _test_mode="echo")
        for duration in (0, -1, True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                GPUInferenceGuard(Path("unused"), PIN, decision_timeout=duration,
                                  _test_mode="echo")
        with self.assertRaises(ValueError):
            GPUInferenceGuard(Path("unused"), PIN, _test_mode="unknown")

    def test_import_does_not_initialize_torch(self):
        result = subprocess.run([sys.executable, "-c", "import sys; "
            f"sys.path[:0] = {[str(ROOT), str(ROOT / 'src')]!r}; "
            "import experiments.p5f_guard; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_raw_logits_avoid_tanh_saturation_and_ties_use_first(self):
        import torch
        logits = torch.tensor([20., 30., 30.], dtype=torch.float32)
        self.assertEqual(torch.tanh(logits).tolist(), [1., 1., 1.])
        self.assertEqual(_select_raw_logits(logits, 3), 1)
        for bad, count in ((torch.tensor([1.]), 2),
                           (torch.tensor([1., float("nan")]), 2),
                           (torch.tensor([[1., 2.]]), 2)):
            with self.assertRaises(ValueError):
                _select_raw_logits(bad, count)


@skipUnless(os.environ.get("RUN_P5F_CUDA") == "1", "explicit P5f CUDA opt-in required")
class RealCheckpointTests(TestCase):
    def test_new_checkpoint_load_and_complete_scoring(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from hashlib import sha256
        from experiments.p5f_checkpoint import load_model, save_checkpoint
        from experiments.p5f_objective import score_requests
        from experiments.p5f_protocol import config
        from experiments.p5f_training import TeacherTrainer

        obs = observation()
        action = Action("single", (0,), 2)
        for objective in ("regression", "ranking"):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as tmp:
                trainer = TeacherTrainer(config(314560, objective), "0" * 64)
                trainer.update([(obs, [action])])
                checkpoint = Path(tmp) / objective
                receipt = save_checkpoint(trainer, checkpoint)
                manifest_digest = sha256((checkpoint / "manifest.json").read_bytes()).hexdigest()
                del trainer
                legal = [Action("single", (0,), 2) for _ in range(8769)]
                reference_model, _ = load_model(checkpoint, receipt["sha256"],
                                                expected_manifest_sha256=manifest_digest)
                reference_logits = score_requests(reference_model, [(obs, legal)],
                                                  chunk_size=1024)[0]
                self.assertEqual(reference_logits.numel(), len(legal))
                expected_index = int(torch.argmax(reference_logits).item())
                del reference_model
                with GPUInferenceGuard(checkpoint, receipt["sha256"],
                                       startup_timeout=60, decision_timeout=30,
                                       expected_manifest_sha256=manifest_digest) as guard:
                    chosen, meta = guard.act(obs, legal)
                    self.assertIs(chosen, legal[expected_index])
                    self.assertEqual(meta["scored_candidates"], 8769)
                    self.assertEqual(meta["device"]["type"], "cuda")


if __name__ == "__main__":
    main()
