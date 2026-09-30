"""Real spawn/IPC/termination tests requiring neither torch nor CUDA."""
import multiprocessing as mp
from pathlib import Path
import sys
import time
from unittest import TestCase, main, mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guandan.types import Action, PlayerObservation
from experiments.p3d.guard import GPUInferenceGuard, GuardError, GuardTimeout


PIN = "a" * 64


def observation():
    return PlayerObservation(0, (0,), 2, 0, (1, 1, 1, 1), (), None,
                             None, (), (), 0, False, None)


class GuardTests(TestCase):
    def make_guard(self, mode="echo", **kwargs):
        return GPUInferenceGuard(Path("unused-test-checkpoint"), PIN,
                                 startup_timeout=10, _test_mode=mode, **kwargs)

    def assert_reaped(self, guard):
        self.assertTrue(guard.closed)
        self.assertFalse(guard.is_alive())
        self.assertIsNotNone(guard.worker_pid)
        self.assertIsNotNone(guard.worker_exitcode)
        self.assertNotIn(guard.worker_pid, [p.pid for p in mp.active_children()])
        with self.assertRaises(GuardError):
            guard.act(observation(), [Action("pass")])
        guard.close()  # Idempotent cleanup.

    def test_echo_preserves_complete_order_and_persistent_worker(self):
        with self.make_guard(decision_timeout=5) as guard:
            pid = guard.worker_pid
            # Multiple chunks' worth of candidates, preserving the last index.
            legal = [Action("single", (i,), i) for i in range(4097)]
            chosen, meta = guard.act(observation(), legal)
            self.assertIs(chosen, legal[-1])
            self.assertEqual(meta["scored_candidates"], len(legal))
            self.assertEqual(meta["device"], {"type": "test", "mode": "echo"})
            self.assertGreaterEqual(meta["roundtrip_ms"], meta["inference_ms"])
            self.assertGreaterEqual(meta["inference_ms"], 0)
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
        self.assertLess(time.perf_counter() - started, 2)
        self.assertEqual(before, {p.pid for p in mp.active_children()})

    def test_inference_deadline_and_large_ipc_upload_are_bounded(self):
        guard = self.make_guard("hang_act", decision_timeout=.15)
        # This is far larger than an OS pipe buffer; sync Pipe.send can hang.
        legal = [Action("single", (i,), i) for i in range(100000)]
        started = time.perf_counter()
        with self.assertRaises(GuardTimeout):
            guard.act(observation(), legal)
        self.assertLess(time.perf_counter() - started, 2)
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

    def test_wrong_observation_type_closes_guard(self):
        guard = self.make_guard()
        with self.assertRaisesRegex(TypeError, "PlayerObservation"):
            guard.act({"hand": (0,), "deal_seed": 123}, [Action("pass")])
        self.assert_reaped(guard)

    def test_error_and_malformed_response_are_fail_closed(self):
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

    def test_invalid_candidate_contract_closes_guard(self):
        for legal in ([], (Action("pass"),), [object()]):
            with self.subTest(legal=legal):
                guard = self.make_guard()
                with self.assertRaises((ValueError, TypeError)):
                    guard.act(observation(), legal)
                self.assert_reaped(guard)

    def test_pin_and_timeout_contracts_fail_before_spawn(self):
        for pin in (None, "", "g" * 64, "a" * 63):
            with self.assertRaises(ValueError):
                GPUInferenceGuard(Path("unused"), pin, _test_mode="echo")
        for duration in (0, -1, True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                GPUInferenceGuard(Path("unused"), PIN, decision_timeout=duration,
                                  _test_mode="echo")

    def test_module_does_not_import_torch(self):
        # Run a fresh interpreter rather than depending on other test imports.
        import subprocess
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-c", "import sys; "
            f"sys.path[:0] = {[str(root), str(root / 'src')]!r}; "
            "import experiments.p3d.guard; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    main()
