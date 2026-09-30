"""Real spawn/IPC/termination tests requiring neither torch nor CUDA."""
import multiprocessing as mp
from pathlib import Path
import sys
import time
from unittest import TestCase, main, mock, skipUnless

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guandan.types import Action, PlayerObservation
from experiments.p3g_guard import GPUInferenceGuard, GuardError, GuardTimeout


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
            "import experiments.p3g_guard; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)



class RawIntegrityTests(TestCase):
    def test_raw_metadata_and_payload_integrity_before_cuda(self):
        import json
        import tempfile
        import types
        from hashlib import sha256
        from experiments.p3f_protocol import config
        from experiments.p3g_guard import verify_phase1
        from experiments.p3f_training import model_digest
        class Tensor:
            dtype = "torch.float32"
            shape = (1,)
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return self
            def tobytes(self): return b"\0\0\0\0"
        state = {"test": Tensor()}
        data = b"temporary fake torch payload"
        pin = sha256(data).hexdigest()
        manifest = dict(sha256=pin, bytes=len(data), model_sha256=model_digest(state),
                        config=config(314380), seed=314380, hands=200,
                        arm="teacher", phase1_updates=5)
        payload = dict(manifest, model=state, optimizer={"intentionally": "ignored"})
        fake_torch = types.SimpleNamespace(load=mock.Mock(return_value=payload))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"torch": fake_torch}):
            folder = Path(tmp)
            (folder / "raw.pt").write_bytes(data)
            def write_manifest(value):
                (folder / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
            write_manifest(manifest)
            verified, receipt = verify_phase1(folder, pin, 314380)
            self.assertIs(verified, payload)
            self.assertEqual(receipt["sha256"], pin)
            self.assertEqual(fake_torch.load.call_args.kwargs,
                             dict(map_location="cpu", weights_only=True))
            with self.assertRaisesRegex(ValueError, "manifest SHA"):
                verify_phase1(folder, pin, 314380, expected_manifest_sha256="a" * 64)
            for key, value in (("config", config(314381)), ("arm", "control"),
                               ("seed", 314381), ("hands", 199), ("hands", True),
                               ("phase1_updates", 0), ("phase1_updates", True),
                               ("bytes", len(data) + 1), ("sha256", "b" * 64),
                               ("model_sha256", "c" * 64)):
                write_manifest({**manifest, key: value})
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    verify_phase1(folder, pin, 314380)
            write_manifest(manifest)
            for key, value in (("seed", 314381), ("hands", True), ("phase1_updates", 6),
                               ("arm", "control"), ("config", config(314381))):
                fake_torch.load.return_value = {**payload, key: value}
                with self.subTest(payload=key), self.assertRaisesRegex(ValueError, "manifest mismatch"):
                    verify_phase1(folder, pin, 314380)
            fake_torch.load.return_value = payload
            (folder / "raw.pt").write_bytes(data + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA/size"):
                verify_phase1(folder, pin, 314380)

    def test_seed_is_bounded_before_spawn(self):
        for seed in (314379, 314383, True, "314380"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                GPUInferenceGuard(Path("unused"), PIN, seed=seed, _test_mode="echo")


class FrozenInputTests(TestCase):
    def test_source_and_input_drift_and_path_escape_fail_closed(self):
        import json
        import tempfile
        from hashlib import sha256
        root = Path(__file__).resolve().parents[1]
        with mock.patch.object(sys, "path", [str(root / "scripts"), *sys.path]):
            import p3g_evaluate as evaluation
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run = repo / "run"
            run.mkdir()
            source = repo / "source.py"
            raw = repo / "raw.pt"
            source.write_bytes(b"frozen source")
            raw.write_bytes(b"frozen input")
            pre = dict(source_sha256={"source.py": sha256(source.read_bytes()).hexdigest()},
                       input_sha256={"raw.pt": sha256(raw.read_bytes()).hexdigest()})
            def write_pre(value):
                (run / "preregistration.json").write_text(json.dumps(value), encoding="utf-8")
            write_pre(pre)
            with mock.patch.object(evaluation, "ROOT", repo):
                self.assertEqual(evaluation.verify_frozen(run),
                                 sha256((run / "preregistration.json").read_bytes()).hexdigest())
                for path, original in ((source, b"frozen source"), (raw, b"frozen input")):
                    path.write_bytes(original + b"tamper")
                    with self.assertRaisesRegex(ValueError, "frozen drift"):
                        evaluation.verify_frozen(run)
                    path.write_bytes(original)
                write_pre({**pre, "input_sha256": {"../escape.pt": "a" * 64}})
                with self.assertRaisesRegex(ValueError, "unsafe frozen path"):
                    evaluation.verify_frozen(run)
                write_pre({**pre, "input_sha256": {}})
                with self.assertRaisesRegex(ValueError, "empty input"):
                    evaluation.verify_frozen(run)


import os
@skipUnless(os.environ.get("RUN_P3G_CUDA") == "1", "CUDA diagnostic requires explicit opt-in")
class RawCUDATests(TestCase):
    def test_temporary_raw_loader_discards_optimizer_and_scores_complete_candidates(self):
        import tempfile
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from experiments.p3f_protocol import config
        from experiments.p3f_training import save_phase1, model_digest
        from experiments.p3g_guard import load_phase1
        from guandan_gpu.training import GPUTrainer, score_many
        trainer = GPUTrainer(config(314380))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = save_phase1(trainer, directory, "teacher", 314380, 1)
            loaded, receipt = load_phase1(directory, manifest["sha256"], 314380)
            self.assertFalse(hasattr(loaded, "optimizer"))
            self.assertFalse(loaded.model.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.model.parameters()))
            self.assertEqual(model_digest(loaded.model), manifest["model_sha256"])
            legal = [Action("pass")] * 1025
            scores = score_many(loaded.model, [(observation(), legal)], chunk_size=1024)[0]
            self.assertEqual(len(scores), len(legal))
            with GPUInferenceGuard(directory, manifest["sha256"], seed=314380,
                                   startup_timeout=60, decision_timeout=2) as guard:
                chosen, meta = guard.act(observation(), legal)
                self.assertIs(chosen, legal[0])
                self.assertEqual(meta["scored_candidates"], len(legal))
                self.assertEqual(meta["device"]["type"], "cuda")
            self.assertTrue(guard.closed)
            self.assertFalse(guard.is_alive())


if __name__ == "__main__":
    main()
