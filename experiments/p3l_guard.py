"""Fail-closed spawned CUDA policy; only visible observation and full actions cross IPC.

Test modes are private CPU fault injection and must never be exposed by a CLI.
Importing this module does not import torch or initialize CUDA.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import threading
import time

from guandan.types import Action, PlayerObservation


class GuardError(RuntimeError):
    """The inference worker failed; this guard can never be reused."""


class GuardTimeout(TimeoutError, GuardError):
    """A startup or complete decision deadline expired."""


def _worker(checkpoint, expected_sha256, incoming, outgoing, test_mode):
    try:
        if test_mode == "hang_start":
            time.sleep(3600)
        if test_mode is None:
            # Configure before load_checkpoint's strict runtime gate runs.
            import json
            from experiments.p3l_training import GPUTrainer, score_many
            from experiments.p3l_checkpoint import load_checkpoint
            import torch
            config = json.loads((Path(checkpoint) / "manifest.json").read_text(
                encoding="utf-8"))["config"]
            configured = GPUTrainer(config)
            del configured
            trainer = load_checkpoint(checkpoint, expected_sha256=expected_sha256)
            model = trainer.model
            model.eval()
            device_index = next(model.parameters()).device.index
            device = dict(type="cuda", index=device_index,
                          name=torch.cuda.get_device_name(device_index),
                          capability=list(torch.cuda.get_device_capability(device_index)),
                          torch=str(torch.__version__), cuda=torch.version.cuda)
        else:
            device = dict(type="test", mode=test_mode)
        outgoing.put(("ready", device))
        while True:
            # No environment, seed, replay, or policy-side hidden state is sent.
            obs, candidates = incoming.get()
            if type(obs) is not PlayerObservation or type(candidates) is not list:
                raise TypeError("invalid visible-information request")
            if not candidates or any(type(a) is not Action for a in candidates):
                raise TypeError("invalid full Action candidate list")
            started = time.perf_counter()
            if test_mode == "hang_act":
                time.sleep(3600)
            if test_mode == "crash_act":
                os._exit(23)
            if test_mode is None:
                torch.cuda.synchronize(device_index)
                started = time.perf_counter()
                scores = score_many(model, [(obs, candidates)], chunk_size=1024)[0]
                # torch.argmax chooses the first occurrence of a tied maximum.
                index = int(torch.argmax(scores).item())
                torch.cuda.synchronize(device_index)
            else:
                index = len(candidates) if test_mode == "bad_index" else len(candidates) - 1
            elapsed_ms = (time.perf_counter() - started) * 1000
            outgoing.put(("result", index, len(candidates), elapsed_ms, device))
    except BaseException as error:
        outgoing.put(("error", type(error).__name__, str(error)))


class GPUInferenceGuard:
    """Persistent worker with a deadline covering request upload and reply.

    The parent owns candidate completeness and the environment. A fault closes
    the guard permanently and raises; there is no fallback action.
    """
    def __init__(self, checkpoint: Path, expected_sha256: str,
                 startup_timeout=60., decision_timeout=2., *, _test_mode=None):
        if not isinstance(checkpoint, Path):
            raise TypeError("checkpoint must be pathlib.Path")
        if type(expected_sha256) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
            raise ValueError("expected_sha256 must be a pinned 64-digit SHA256")
        for value in (startup_timeout, decision_timeout):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("timeouts must be finite and positive")
        if _test_mode not in (None, "hang_start", "hang_act", "crash_act", "bad_index", "echo"):
            raise ValueError("unknown private test mode")
        self.decision_timeout = float(decision_timeout)
        self.closed = False
        self._lock = threading.Lock()
        context = mp.get_context("spawn")
        self._incoming = context.Queue(maxsize=1)
        self._outgoing = context.Queue(maxsize=1)
        self._process = context.Process(target=_worker, args=(str(checkpoint),
            expected_sha256.lower(), self._incoming, self._outgoing, _test_mode), daemon=True)
        self.worker_pid = None
        self.worker_exitcode = None
        deadline = time.perf_counter() + float(startup_timeout)
        try:
            self._process.start()
            self.worker_pid = self._process.pid
            reply = self._receive(deadline, "startup")
            if type(reply) is not tuple or len(reply) != 2 or reply[0] != "ready" or type(reply[1]) is not dict:
                raise GuardError(f"invalid startup response: {reply!r}")
            self.device = reply[1]
        except BaseException:
            self.close()
            raise

    def _receive(self, deadline, stage):
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise GuardTimeout(f"GPU {stage} deadline expired")
            try:
                response = self._outgoing.get(timeout=min(remaining, 0.025))
            except queue.Empty:
                if not self._process.is_alive():
                    raise GuardError(f"GPU worker exited during {stage}: {self._process.exitcode}")
                continue
            if time.perf_counter() > deadline:
                raise GuardTimeout(f"GPU {stage} deadline expired")
            return response

    def act(self, obs: PlayerObservation, legal: list[Action]):
        with self._lock:
            if self.closed:
                raise GuardError("GPU inference guard is closed and cannot be reused")
            started = time.perf_counter()
            deadline = started + self.decision_timeout
            try:
                if type(obs) is not PlayerObservation:
                    raise TypeError("observation must be PlayerObservation")
                if type(legal) is not list or not legal:
                    raise ValueError("legal must be a nonempty complete list")
                if any(type(action) is not Action for action in legal):
                    raise TypeError("legal candidates must be Action objects")
                candidates = list(legal)
                # Queue feeder serializes and uploads in the background. Never
                # synchronously send a potentially huge list through Pipe.send.
                self._incoming.put_nowait((obs, candidates))
                reply = self._receive(deadline, "decision")
                if type(reply) is not tuple or len(reply) != 5 or reply[0] != "result":
                    raise GuardError(f"invalid decision response: {reply!r}")
                _, index, count, inference_ms, device = reply
                if type(index) is not int or not 0 <= index < len(candidates):
                    raise GuardError("GPU worker returned an illegal candidate index")
                if type(count) is not int or count != len(candidates):
                    raise GuardError("GPU worker did not score every candidate")
                if (type(inference_ms) not in (int, float) or not math.isfinite(inference_ms)
                        or inference_ms < 0 or device != self.device):
                    raise GuardError("invalid GPU inference metadata")
                roundtrip_ms = (time.perf_counter() - started) * 1000
                if time.perf_counter() > deadline:
                    raise GuardTimeout("GPU decision deadline expired")
                return candidates[index], dict(inference_ms=inference_ms,
                    roundtrip_ms=roundtrip_ms, scored_candidates=count, device=device)
            except BaseException:
                self.close()
                raise

    def is_alive(self):
        return self._process.is_alive()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._process.pid is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.5)
            self.worker_exitcode = self._process.exitcode
        for channel in (self._incoming, self._outgoing):
            channel.cancel_join_thread()
            channel.close()
        if self._process.pid is not None and self._process.is_alive():
            raise GuardError("GPU worker could not be reaped within cleanup deadline")

    def __enter__(self):
        if self.closed:
            raise GuardError("GPU inference guard is closed and cannot be reused")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
