"""Bounded GPU-only DMC learning probe on 16 development hands."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import zipfile

# Set before importing torch or the shared benchmark module.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from _bootstrap import ROOT
import torch

import benchmark_learning_runtime as common
from guandan.learning.model import DMCNetwork, NETWORK_VERSION
from guandan.types import RULES_VERSION

SEED = 271828
DEALS = tuple((102200 + i, 2 + i % 13, i % 4, SEED + i) for i in range(16))
DEVICE = "cuda"
REPEATS = 20
BATCH_SIZE = 64


def source_paths() -> list[Path]:
    files = sorted((ROOT / "src").rglob("*.py"))
    files += [Path(__file__).resolve(), ROOT / "scripts/benchmark_learning_runtime.py",
              ROOT / "scripts/_bootstrap.py", ROOT / "docs/P3B_GPU_PROTOCOL.md",
              ROOT / "requirements-learning-gpu.txt",
              ROOT / "requirements-learning-gpu.lock"]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("GPU probe source, protocol, or requirements file missing")
    return files


def source_hashes(paths: list[Path]) -> dict:
    return {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _action_hash(actions) -> str:
    payload = [common.action_payload(action) for action in actions]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def high_branch_probe(model: DMCNetwork) -> dict:
    obs, actions = common.high_branch_fixture()
    chunk_lengths: list[int] = []
    def hook(_module, arguments, _output):
        if arguments[0].device.type != DEVICE:
            raise AssertionError("candidate features reached a non-CUDA action layer")
        chunk_lengths.append(len(arguments[0]))
    handle = model.action_fc.register_forward_hook(hook)
    try:
        common.score_complete(model, obs, actions, DEVICE)  # Warmup.
        chunk_lengths.clear()
        scores, _ = common.timed(DEVICE, lambda: common.score_complete(model, obs, actions, DEVICE))
    finally:
        handle.remove()
    if sum(chunk_lengths) != len(actions) or max(chunk_lengths) > common.CHUNK:
        raise AssertionError("high-branch candidate coverage incomplete")
    raw_ms = []
    for _ in range(REPEATS):
        repeated, elapsed = common.timed(
            DEVICE, lambda: common.score_complete(model, obs, actions, DEVICE))
        if not torch.equal(scores, repeated):
            raise AssertionError("fixed GPU high-branch scores changed between repetitions")
        raw_ms.append(elapsed)
    if not bool(torch.isfinite(scores).all()) or len(scores) != 8769:
        raise AssertionError("high-branch scores nonfinite or incomplete")
    return {"fixture": "p1_8769", "candidates": len(actions),
            "candidate_sha256": _action_hash(actions),
            "coverage": {"chunk_lengths": chunk_lengths,
                         "scored_candidates": sum(chunk_lengths),
                         "max_chunk": max(chunk_lengths)},
            "latency": common.stats(raw_ms), "scores": scores.tolist(),
            "all_scores_finite": True, "score_device": DEVICE,
            "timing_scope": "Python encoding, host-to-device copy, GPU score, CPU return"}


def cuda_batch_proof(model: DMCNetwork, samples: list) -> dict:
    """Check the exact batch shapes used by updates have CUDA prediction/loss."""
    states = torch.tensor([row[0] for row in samples], dtype=torch.float32, device=DEVICE)
    actions = torch.tensor([row[1] for row in samples], dtype=torch.float32, device=DEVICE)
    targets = torch.tensor([row[2] for row in samples], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        predictions = model(states, actions)
        loss = torch.nn.functional.mse_loss(predictions, targets)
    if (predictions.device.type != DEVICE or loss.device.type != DEVICE
            or not bool(torch.isfinite(loss))):
        raise AssertionError("CUDA batch prediction/loss proof failed")
    return {"states_device": states.device.type, "actions_device": actions.device.type,
            "targets_device": targets.device.type,
            "predictions_device": predictions.device.type, "loss_device": loss.device.type,
            "loss_finite": True}


def update_with_device_hooks(model: DMCNetwork, optimizer, samples: list) -> dict:
    forward_devices: list[str] = []
    gradient_devices: list[str] = []
    handles = [model.register_forward_hook(
        lambda _module, _arguments, output: forward_devices.append(output.device.type))]
    for parameter in model.parameters():
        handles.append(parameter.register_hook(
            lambda gradient: gradient_devices.append(gradient.device.type)))
    try:
        update = common.update_batch(model, optimizer, samples, DEVICE)
    finally:
        for handle in handles:
            handle.remove()
    expected_gradients = len(list(model.parameters()))
    if (forward_devices != [DEVICE] or len(gradient_devices) != expected_gradients
            or any(device != DEVICE for device in gradient_devices)):
        raise AssertionError("learning forward or gradient did not stay on CUDA")
    proof = cuda_batch_proof(model, samples)
    return {**update, "forward_hook_calls": len(forward_devices),
            "gradient_hook_calls": len(gradient_devices),
            "all_forward_cuda": True, "all_gradients_cuda": True,
            "batch_device_proof": proof}


def run_learning(model: DMCNetwork, progress_path: Path, started: float) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=common.LEARNING_RATE)
    initial_hash = common.state_dict_sha256(model.state_dict())
    hands = []
    all_losses = []
    total_start = time.perf_counter()
    for index, (seed, level, starter, policy_seed) in enumerate(DEALS):
        episode_start = time.perf_counter()
        score_forward_devices = []
        scored_chunk_lengths = []
        def score_hook(_module, arguments, _output):
            score_forward_devices.append(arguments[0].device.type)
            scored_chunk_lengths.append(len(arguments[0]))
        hook = model.action_fc.register_forward_hook(score_hook)
        try:
            hand, samples = common.sample_hand(model, DEVICE, seed, level, starter,
                                               policy_seed, collect=True)
        finally:
            hook.remove()
        if (not scored_chunk_lengths or any(device != DEVICE for device in score_forward_devices)
                or max(scored_chunk_lengths) > common.CHUNK):
            raise AssertionError("hand scoring did not stay in CUDA chunks")
        hand["score_cuda_proof"] = {"forward_hook_calls": len(score_forward_devices),
                                    "scored_candidates": sum(scored_chunk_lengths),
                                    "max_chunk": max(scored_chunk_lengths),
                                    "all_score_features_cuda": True}
        order = list(range(len(samples)))
        random.Random(SEED + seed).shuffle(order)
        updates = []
        for offset in range(0, len(order), BATCH_SIZE):
            batch = [samples[i] for i in order[offset:offset + BATCH_SIZE]]
            updates.append(update_with_device_hooks(model, optimizer, batch))
        hand["sampling_wall_ms"] = hand["total_ms"]
        hand["update_ms"] = sum(update["wall_ms"] for update in updates)
        hand["updates"] = len(updates)
        hand["losses"] = [update["loss"] for update in updates]
        hand["cuda_proofs"] = [{"forward_hook_calls": update["forward_hook_calls"],
                                "gradient_hook_calls": update["gradient_hook_calls"],
                                "all_forward_cuda": update["all_forward_cuda"],
                                "all_gradients_cuda": update["all_gradients_cuda"],
                                "parameters_changed": update["parameters_changed"],
                                "batch_device_proof": update["batch_device_proof"]}
                               for update in updates]
        hand["total_ms"] = (time.perf_counter() - episode_start) * 1000
        hand["other_ms"] = max(0.0, hand["total_ms"] - sum(hand[key] for key in
                                  ("enumeration_ms", "score_ms", "update_ms", "replay_ms")))
        all_losses.extend(hand["losses"])
        hands.append(hand)
        if (index + 1) % 4 == 0:
            common.progress(progress_path, "learning", "running", started,
                            completed_hands=index + 1, samples=sum(h["samples"] for h in hands),
                            updates=sum(h["updates"] for h in hands))
    final_hash = common.state_dict_sha256(model.state_dict())
    if initial_hash == final_hash:
        raise AssertionError("16-hand GPU learning did not change model parameters")
    total_ms = (time.perf_counter() - total_start) * 1000
    return {"hands": hands, "total_ms": total_ms,
            "loop_overhead_ms": max(0.0, total_ms - sum(hand["total_ms"] for hand in hands)),
            "component_ms": {key: sum(hand[key] for hand in hands) for key in
                             ("enumeration_ms", "score_ms", "update_ms", "replay_ms", "other_ms")},
            "samples": sum(hand["samples"] for hand in hands),
            "updates": sum(hand["updates"] for hand in hands),
            "losses": all_losses, "initial_model_state_sha256": initial_hash,
            "final_model_state_sha256": final_hash,
            "hands_per_hour_descriptive": len(DEALS) * 3600000 / total_ms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    paths = source_paths()
    before = source_hashes(paths)
    preregistration = {"kind": "p3b_gpu_preregistration", "status": "PREREGISTERED",
                       "utc": datetime.now(timezone.utc).isoformat(),
                       "output": str(output), "device": DEVICE, "model_seed": SEED,
                       "deals": [list(deal) for deal in DEALS], "hands": 16,
                       "epsilon": common.EPSILON, "learning_rate": common.LEARNING_RATE,
                       "batch_size": BATCH_SIZE, "chunk_size": common.CHUNK,
                       "high_branch_repeats": REPEATS, "source_sha256_before": before,
                       "scope": "bounded GPU learning runtime probe; no strength or recovery claim"}
    common.write_json(output / "preregistration.json", preregistration)
    with zipfile.ZipFile(output / "source-snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    zip_hash = sha256((output / "source-snapshot.zip").read_bytes()).hexdigest()
    report = {"kind": "p3b_gpu_learning_probe", "status": "RUNNING",
              "preregistration": "preregistration.json",
              "source_snapshot": "source-snapshot.zip", "source_snapshot_sha256": zip_hash,
              "source_sha256_before": before, "results": {}}
    report_path = output / "report.json"
    progress_path = output / "progress.jsonl"
    common.write_json(report_path, report)
    common.progress(progress_path, "setup", "complete", started)
    stage = "runtime"
    try:
        common.configure()
        report["runtime"] = common.runtime_info()
        report["network_version"] = NETWORK_VERSION
        report["rules_version"] = RULES_VERSION
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        model = DMCNetwork().float().to(DEVICE)
        if any(parameter.device.type != DEVICE or parameter.dtype != torch.float32
               for parameter in model.parameters()):
            raise AssertionError("network is not entirely CUDA float32")
        model.train()
        stage = "high_branch"
        common.progress(progress_path, stage, "started", started)
        report["results"][stage] = high_branch_probe(model)
        common.write_json(report_path, report)
        common.progress(progress_path, stage, "complete", started)
        stage = "learning"
        common.progress(progress_path, stage, "started", started)
        report["results"][stage] = run_learning(model, progress_path, started)
        common.write_json(report_path, report)
        common.progress(progress_path, stage, "complete", started)
        stage = "weights"
        weights_path = output / "runtime-weights.pt"
        torch.save(model.state_dict(), weights_path)
        report["runtime_weights"] = {"path": weights_path.name,
                                      "sha256": sha256(weights_path.read_bytes()).hexdigest(),
                                      "bytes": weights_path.stat().st_size,
                                      "scope": "GPU experiment weights only; no optimizer/RNG, no resume or candidate claim"}
        after = source_hashes(paths)
        report["source_sha256_after"] = after
        report["source_unchanged"] = before == after
        if before != after:
            raise RuntimeError("source changed during GPU probe")
        report["elapsed_s"] = time.perf_counter() - started
        report["status"] = "PASS"
        common.write_json(report_path, report)
        common.progress(progress_path, "final", "PASS", started)
        return 0
    except BaseException:
        report["status"] = "FAIL"
        report["failure"] = traceback.format_exc()
        report["failed_stage"] = stage
        report["source_sha256_after"] = source_hashes(paths)
        report["source_unchanged"] = before == report["source_sha256_after"]
        report["elapsed_s"] = time.perf_counter() - started
        common.write_json(report_path, report)
        common.write_json(output / "failure.json", {"status": "FAIL", "stage": stage,
                                                    "traceback": report["failure"],
                                                    "report": "report.json"})
        common.progress(progress_path, "final", "FAIL", started, failed_stage=stage,
                        error=report["failure"].splitlines()[-1])
        return 1


if __name__ == "__main__":
    sys.exit(main())
