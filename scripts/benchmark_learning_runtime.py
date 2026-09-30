"""P3b1 development-only DMC runtime benchmark; no checkpoint or candidate."""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
import zipfile

# CUDA determinism requires this before importing/initializing torch.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from _bootstrap import ROOT
import torch

from guandan.env import HandEnv
from guandan.learning.encoding import ACTION_DIM, STATE_DIM, encode_action, encode_observation
from guandan.learning.model import DMCNetwork, NETWORK_VERSION, score_actions
from guandan.rules.cards import card_id
from guandan.types import Action, PlayerObservation, RULES_VERSION

SEED = 271828
CHUNK = 256
DEALS = tuple(zip(range(102000, 102004), (2, 5, 10, 14), range(4)))
THROUGHPUT_DEALS = tuple((102100 + i, 2 + i % 13, i % 4, 900000 + i) for i in range(8))
TRAIN_BATCH_SIZES = (64, 256, 1024)
EPSILON = 0.1
LEARNING_RATE = 0.001
MAX_STEPS = 1000


def configure() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty latency sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = int(position)
    return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (position - left)


def stats(values: list[float]) -> dict:
    return {"count": len(values), "raw_ms": values, "p50_ms": percentile(values, .5),
            "p95_ms": percentile(values, .95), "min_ms": min(values), "max_ms": max(values)}


def timed(device: str, operation):
    synchronize(device)
    start = time.perf_counter()
    result = operation()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000


def action_payload(action: Action) -> dict:
    return {"kind": action.kind, "cards": action.cards, "main_rank": action.main_rank,
            "wildcards": action.wildcards}


def score_complete(model: DMCNetwork, obs: PlayerObservation,
                   actions: list[Action] | tuple[Action, ...], device: str) -> torch.Tensor:
    """Only player-visible observation/actions enter the policy; all scores return to CPU."""
    if type(obs) is not PlayerObservation or not isinstance(actions, (list, tuple)) or not actions:
        raise TypeError("score_complete requires PlayerObservation and nonempty actions")
    if any(type(action) is not Action for action in actions):
        raise TypeError("candidates must be Action objects")
    if next(model.parameters()).device.type != device:
        raise ValueError("model/device mismatch")
    state_cpu = torch.tensor((encode_observation(obs),), dtype=torch.float32)
    pieces = []
    with torch.no_grad():
        state_projection = model.state_fc(state_cpu.to(device))
        for start in range(0, len(actions), CHUNK):
            encoded = tuple(encode_action(a) for a in actions[start:start + CHUNK])
            action_cpu = torch.tensor(encoded, dtype=torch.float32)
            hidden = torch.relu(state_projection + model.action_fc(action_cpu.to(device)))
            hidden = torch.relu(model.hidden_fc(hidden))
            scores = torch.tanh(model.output_fc(hidden)).squeeze(-1)
            pieces.append(scores.cpu())
    result = torch.cat(pieces)
    if result.numel() != len(actions) or not bool(torch.isfinite(result).all()):
        raise ValueError("incomplete or nonfinite action scores")
    return result


def high_branch_fixture():
    hand = tuple(card_id(rank, suit, copy) for rank in (2, 3, 4)
                 for suit in range(4) for copy in range(2)) + (18, 72, 7)
    rest = [card for card in range(108) if card not in hand]
    env = HandEnv.from_hands((hand, tuple(rest[:27]), tuple(rest[27:54]), tuple(rest[54:])), 7)
    actions = env.legal_actions(0)
    if len(actions) != 8769:
        raise AssertionError(f"high-branch fixture changed: {len(actions)}")
    return env.observe(0), actions


def sample_hand(model: DMCNetwork, device: str, deal_seed: int, level: int,
                starting_player: int, policy_seed: int, collect: bool = False,
                verify_replay: bool = True) -> tuple[dict, list]:
    """One fixed-model hand; no update occurs before its terminal settlement."""
    if type(deal_seed) is not int or not 100000 <= deal_seed <= 109999:
        raise ValueError("runtime hand must use a development seed")
    hand_start = time.perf_counter()
    rng = random.Random(policy_seed)
    env = HandEnv()
    obs = env.reset(deal_seed, initial_level=level, starting_player=starting_player)
    durations = {"enumeration_ms": 0.0, "score_ms": 0.0, "replay_ms": 0.0,
                 "other_ms": 0.0}
    actions_chosen = []
    trajectory = []
    for step_number in range(1, MAX_STEPS + 1):
        legal, elapsed = timed(device, lambda: env.legal_actions(obs.player_id))
        durations["enumeration_ms"] += elapsed
        if not legal:
            raise RuntimeError("active player has no actions")
        draw = rng.random()
        if draw < EPSILON:
            selected = rng.randrange(len(legal))
        else:
            scores, elapsed = timed(device, lambda: score_complete(model, obs, legal, device))
            durations["score_ms"] += elapsed
            selected = int(torch.argmax(scores).item())
        action = legal[selected]
        if collect:
            trajectory.append((encode_observation(obs), encode_action(action), obs.player_id % 2))
        actions_chosen.append(action_payload(action))
        result = env.step(obs.player_id, action, state_version=obs.state_version)
        if result.terminal:
            break
        obs = env.observe(result.next_player)
    else:
        raise RuntimeError("1000-step hand guard exceeded")
    if result.settlement is None or result.settlement.team_rewards not in ((1, -1), (-1, 1)):
        raise RuntimeError("invalid terminal reward")
    terminal_digest = env.state_digest()
    if verify_replay:
        def replay_check():
            replay = env.serialize_replay()
            replayed = HandEnv.replay(replay)
            if replayed.state_digest() != terminal_digest or not replayed.state.terminal:
                raise RuntimeError("replay mismatch")
        _, elapsed = timed(device, replay_check)
        durations["replay_ms"] += elapsed
    total_ms = (time.perf_counter() - hand_start) * 1000
    durations["other_ms"] = max(0.0, total_ms - sum(durations.values()))
    action_digest = sha256(json.dumps(actions_chosen, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    row = {"seed": deal_seed, "level": level, "starting_player": starting_player,
           "policy_seed": policy_seed, "steps": step_number, "samples": len(actions_chosen),
           "team_rewards": list(result.settlement.team_rewards),
           "terminal_digest": terminal_digest, "action_digest": action_digest,
           "replay_verified": verify_replay, "total_ms": total_ms, **durations}
    if collect:
        trajectory = [(state, action, result.settlement.team_rewards[team])
                      for state, action, team in trajectory]
    return row, trajectory


def update_batch(model: DMCNetwork, optimizer, samples: list, device: str) -> dict:
    states_cpu = torch.tensor([row[0] for row in samples], dtype=torch.float32)
    actions_cpu = torch.tensor([row[1] for row in samples], dtype=torch.float32)
    targets_cpu = torch.tensor([row[2] for row in samples], dtype=torch.float32)
    before = [p.detach().clone() for p in model.parameters()]

    def operation():
        states = states_cpu.to(device)
        actions = actions_cpu.to(device)
        targets = targets_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(states, actions)
        if predictions.shape != targets.shape or not bool(torch.isfinite(predictions).all()):
            raise ValueError("nonfinite/mismatched predictions")
        loss = torch.nn.functional.mse_loss(predictions, targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite loss")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("nonfinite/missing gradient")
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("nonfinite parameters")
        return float(loss.item())

    loss, elapsed = timed(device, operation)
    changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    if not changed:
        raise ValueError("Adam did not update any parameter")
    return {"wall_ms": elapsed, "loss": loss, "parameters_changed": changed,
            "includes": "CPU tensor creation outside timer; host-to-device copy, forward, backward and Adam inside timer"}


def model_from_state(state: dict, device: str) -> DMCNetwork:
    model = DMCNetwork().to(device)
    model.load_state_dict(state)
    model.train()
    return model


def state_dict_sha256(state: dict) -> str:
    digest = sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        if tensor.dtype != torch.float32:
            raise TypeError(f"unexpected state dtype: {key}={tensor.dtype}")
        values = array("f", tensor.reshape(-1).tolist())
        if values.itemsize != 4:
            raise RuntimeError("platform float storage is not 32-bit")
        if sys.byteorder != "little":
            values.byteswap()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def compare_scores(state: dict, devices: list[str], repeats: int) -> dict:
    fixtures = []
    env = HandEnv()
    obs = env.reset(102000, initial_level=2)
    fixtures.append(("real_deal_102000", obs, env.legal_actions(0)))
    obs, actions = high_branch_fixture()
    fixtures.append(("p1_8769", obs, actions))
    models = {device: model_from_state(state, device).eval() for device in devices}
    output = []
    for label, obs, legal in fixtures:
        scores_by_device = {}
        latencies = {}
        coverage = {}
        for device in devices:
            chunk_lengths = []
            hook = models[device].action_fc.register_forward_hook(
                lambda _module, arguments, _output: chunk_lengths.append(len(arguments[0])))
            try:
                score_complete(models[device], obs, legal, device)  # Warmup and coverage proof.
            finally:
                hook.remove()
            if sum(chunk_lengths) != len(legal) or max(chunk_lengths) > CHUNK:
                raise AssertionError(f"candidate coverage failed: {label}/{device}")
            coverage[device] = {"chunk_lengths": chunk_lengths,
                                "scored_candidates": sum(chunk_lengths),
                                "max_chunk": max(chunk_lengths)}
            synchronize(device)
            samples = []
            for _ in range(repeats):
                scores, elapsed = timed(device, lambda: score_complete(models[device], obs, legal, device))
                samples.append(elapsed)
            scores_by_device[device] = scores
            latencies[device] = stats(samples)
        candidate_bytes = json.dumps([action_payload(action) for action in legal],
                                     sort_keys=True, separators=(",", ":")).encode()
        row = {"fixture": label, "candidates": len(legal), "latency": latencies,
               "coverage": coverage, "candidate_sha256": sha256(candidate_bytes).hexdigest(),
               "scores_by_device": {device: scores.tolist()
                                    for device, scores in scores_by_device.items()}}
        if "cuda" in devices:
            cpu, cuda = scores_by_device["cpu"], scores_by_device["cuda"]
            row.update(allclose=bool(torch.allclose(cpu, cuda, atol=1e-5, rtol=1e-4)),
                       max_abs_diff=float(torch.max(torch.abs(cpu - cuda))),
                       argmax_equal=int(torch.argmax(cpu)) == int(torch.argmax(cuda)),
                       atol=1e-5, rtol=1e-4)
            if not row["allclose"]:
                raise AssertionError(f"CPU/CUDA score mismatch: {label}")
        reference = score_actions(models["cpu"], obs, legal, CHUNK)
        row["cpu_reference_allclose"] = bool(torch.allclose(
            scores_by_device["cpu"], reference, atol=1e-5, rtol=1e-4))
        row["cpu_reference_max_abs_diff"] = float(torch.max(torch.abs(
            scores_by_device["cpu"] - reference)))
        if not row["cpu_reference_allclose"]:
            raise AssertionError(f"CPU benchmark scorer differs from P3a: {label}")
        output.append(row)
    return {"fixtures": output, "chunk_size": CHUNK,
            "argmax_equal_is_gate": False, "score_allclose_is_gate": "cuda" in devices}


def benchmark_minibatches(state: dict, devices: list[str], repeats: int,
                          collected: list) -> dict:
    if not collected:
        raise RuntimeError("no real hand samples collected")
    output = {}
    for size in TRAIN_BATCH_SIZES:
        rows = [collected[i % len(collected)] for i in range(size)]
        output[str(size)] = {}
        for device in devices:
            model = model_from_state(state, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            update_batch(model, optimizer, rows, device)  # Warmup.
            elapsed = []
            losses = []
            for _ in range(repeats):
                item = update_batch(model, optimizer, rows, device)
                elapsed.append(item["wall_ms"])
                losses.append(item["loss"])
            output[str(size)][device] = {"latency": stats(elapsed), "losses": losses,
                                          "samples": size, "finite": True,
                                          "parameters_updated_each_repeat": True}
    return {"actual_samples_collected": len(collected),
            "cyclic_fill_for_1024": len(collected) < 1024, "minibatches": output,
            "timing_scope": "Host-to-device transfer plus forward/backward/Adam; CPU tensor creation before timer"}


def benchmark_episodes(state: dict, devices: list[str]) -> dict:
    output = {}
    for device in devices:
        device_start = time.perf_counter()
        model = model_from_state(state, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        rows = []
        for index, (seed, level, starter) in enumerate(DEALS):
            episode_start = time.perf_counter()
            hand, samples = sample_hand(model, device, seed, level, starter,
                                        policy_seed=SEED + index, collect=True)
            updates = []
            order = list(range(len(samples)))
            random.Random(SEED + seed).shuffle(order)
            for offset in range(0, len(order), 64):
                batch = [samples[i] for i in order[offset:offset + 64]]
                updates.append(update_batch(model, optimizer, batch, device))
            hand["update_ms"] = sum(row["wall_ms"] for row in updates)
            hand["updates"] = len(updates)
            hand["sampling_wall_ms"] = hand["total_ms"]
            hand["total_ms"] = (time.perf_counter() - episode_start) * 1000
            hand["other_ms"] = max(0.0, hand["total_ms"] - sum(hand[key] for key in
                                      ("enumeration_ms", "score_ms", "update_ms", "replay_ms")))
            rows.append(hand)
        total = sum(row["total_ms"] for row in rows)
        output[device] = {"hands": rows, "total_ms": total,
                          "device_wall_ms": (time.perf_counter() - device_start) * 1000,
                          "hands_per_hour": 4 * 3600000 / total,
                          "samples": sum(row["samples"] for row in rows),
                          "updates": sum(row["updates"] for row in rows),
                          "component_ms": {key: sum(row[key] for row in rows) for key in
                                           ("enumeration_ms", "score_ms", "update_ms", "replay_ms", "other_ms")}}
    return {"fixed_development_deals": [list(row) for row in DEALS], "devices": output,
            "trajectory_equality_required": False,
            "scope": "single-process four-seat sample, learning and full replay"}


def _worker_hand(state: dict, deal: tuple) -> dict:
    configure()
    model = model_from_state(state, "cpu").eval()
    row, _ = sample_hand(model, "cpu", *deal)
    row["model_state_sha256"] = state_dict_sha256(model.state_dict())
    return row


def _signatures(rows: list[dict]) -> dict:
    return {row["seed"]: {key: row[key] for key in
                          ("terminal_digest", "action_digest", "samples")}
            for row in rows}


def benchmark_workers(state: dict) -> dict:
    output = {}
    baseline = None
    model_hash = state_dict_sha256(state)
    for workers in (1, 2, 4):
        start = time.perf_counter()
        if workers == 1:
            model = model_from_state(state, "cpu").eval()
            rows = [sample_hand(model, "cpu", *deal)[0] for deal in THROUGHPUT_DEALS]
            for row in rows:
                row["model_state_sha256"] = state_dict_sha256(model.state_dict())
        else:
            with ProcessPoolExecutor(max_workers=workers,
                                     mp_context=multiprocessing.get_context("spawn")) as executor:
                rows = list(executor.map(_worker_hand, (state,) * len(THROUGHPUT_DEALS),
                                         THROUGHPUT_DEALS))
        cold_ms = (time.perf_counter() - start) * 1000
        signatures = _signatures(rows)
        if any(row["model_state_sha256"] != model_hash for row in rows):
            raise AssertionError(f"spawn {workers} model state differs")
        if len(signatures) != 8:
            raise AssertionError("missing throughput hand")
        if baseline is None:
            baseline = signatures
        elif signatures != baseline:
            raise AssertionError(f"spawn {workers} results differ from single-process")
        output[str(workers)] = {"cold_wall_ms": cold_ms,
                                "worker_inner_ms": [row["total_ms"] for row in rows],
                                "worker_inner_sum_ms": sum(row["total_ms"] for row in rows),
                                "hands": rows, "signature_match": signatures == baseline}
    return {"fixed_hands": [list(row) for row in THROUGHPUT_DEALS],
            "model_state_sha256": model_hash,
            "workers": output, "scope": "CPU fixed-model sampling throughput probe; no model sync or recovery claim"}


def source_paths() -> list[Path]:
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [
        Path(__file__).resolve(),
        ROOT / "scripts/_bootstrap.py",
        ROOT / "tests/test_runtime_benchmark.py",
        ROOT / "requirements-learning-cu128.txt",
        ROOT / "docs/P3B_RUNTIME_PROTOCOL.md",
    ]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("benchmark source/protocol missing")
    return paths


def source_hashes(paths: list[Path]) -> dict:
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}


def write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def progress(path: Path, stage: str, status: str, started: float, **details) -> None:
    row = {"stage": stage, "status": status, "elapsed_s": round(time.perf_counter() - started, 3),
           "utc": datetime.now(timezone.utc).isoformat(), **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(row, ensure_ascii=False), flush=True)


def runtime_info() -> dict:
    cuda = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if cuda["available"]:
        cuda.update(device_name=torch.cuda.get_device_name(0),
                    capability=list(torch.cuda.get_device_capability(0)),
                    arch_list=torch.cuda.get_arch_list(), device_count=torch.cuda.device_count())
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "float_dtype": "float32", "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32, "cuda": cuda}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--hands", type=int, default=4,
                        help="must be 4 fixed episode deals; throughput probe always uses 8")
    args = parser.parse_args()
    if args.repeats < 1 or args.hands != 4 or len(set(args.devices)) != len(args.devices):
        parser.error("repeats must be positive, hands exactly 4, and devices unique")
    if "cuda" in args.devices and "cpu" not in args.devices:
        parser.error("CUDA comparison requires cpu and cuda devices")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    report_path = output / "report.json"
    progress_path = output / "progress.jsonl"
    paths = source_paths()
    before = source_hashes(paths)
    preregistration = {"kind": "p3b1_runtime_preregistration", "status": "PREREGISTERED",
                       "utc": datetime.now(timezone.utc).isoformat(),
                       "command": vars(args) | {"output": str(output)}, "seed": SEED,
                       "deals": [list(row) for row in DEALS],
                       "throughput_deals": [list(row) for row in THROUGHPUT_DEALS],
                       "minibatches": list(TRAIN_BATCH_SIZES), "chunk_size": CHUNK,
                       "epsilon": EPSILON, "learning_rate": LEARNING_RATE,
                       "score_tolerance": {"atol": 1e-5, "rtol": 1e-4},
                       "source_sha256_before": before,
                       "claims_excluded": ["policy strength", "formal learning training",
                                           "model sync", "recovery", "candidate promotion"]}
    write_json(output / "preregistration.json", preregistration)
    with zipfile.ZipFile(output / "source-snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    zip_hash = sha256((output / "source-snapshot.zip").read_bytes()).hexdigest()
    report = {"kind": "p3b1_runtime_benchmark", "status": "RUNNING", "preregistration":
              "preregistration.json", "source_snapshot": "source-snapshot.zip",
              "source_snapshot_sha256": zip_hash, "source_sha256_before": before,
              "results": {}}
    write_json(report_path, report)
    progress(progress_path, "setup", "complete", started)
    try:
        configure()
        report["runtime"] = runtime_info()
        if "cuda" in args.devices and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        base = DMCNetwork().float().cpu()
        initial_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        report["network_version"] = NETWORK_VERSION
        report["rules_version"] = RULES_VERSION
        phases = (
            ("score", lambda: compare_scores(initial_state, args.devices, args.repeats)),
            ("collect", None),
            ("minibatch", None),
            ("episodes", lambda: benchmark_episodes(initial_state, args.devices)),
            ("workers", lambda: benchmark_workers(initial_state)),
        )
        collected = None
        for stage, operation in phases:
            progress(progress_path, stage, "started", started)
            if stage == "collect":
                sample_model = model_from_state(initial_state, "cpu").eval()
                hands = [sample_hand(sample_model, "cpu", *deal, collect=True) for deal in THROUGHPUT_DEALS]
                collected = [sample for _, hand_samples in hands for sample in hand_samples]
                result = {"hands": [row for row, _ in hands], "samples": len(collected)}
            elif stage == "minibatch":
                result = benchmark_minibatches(initial_state, args.devices, args.repeats, collected)
            else:
                result = operation()
            report["results"][stage] = result
            write_json(report_path, report)
            progress(progress_path, stage, "complete", started)
        after = source_hashes(paths)
        report["source_sha256_after"] = after
        report["source_unchanged"] = before == after
        if before != after:
            raise RuntimeError("source changed during benchmark")
        report["elapsed_s"] = time.perf_counter() - started
        report["status"] = "PASS"
        write_json(report_path, report)
        progress(progress_path, "final", "PASS", started)
        return 0
    except BaseException:
        report["status"] = "FAIL"
        report["failure"] = traceback.format_exc()
        report["source_sha256_after"] = source_hashes(paths)
        report["source_unchanged"] = before == report["source_sha256_after"]
        report["elapsed_s"] = time.perf_counter() - started
        write_json(report_path, report)
        write_json(output / "failure.json", {"status": "FAIL", "traceback": report["failure"],
                                             "failed_stage": stage if "stage" in locals() else "setup",
                                             "report": "report.json"})
        progress(progress_path, "final", "FAIL", started, error=report["failure"].splitlines()[-1])
        return 1


if __name__ == "__main__":
    sys.exit(main())
