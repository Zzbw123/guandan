"""One frozen P3l paired-arm training or evaluation job; no orchestration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys
import time

from _bootstrap import ROOT

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/p3d"))

from p3l_protocol import config, schedule, specification
from p3e_common import behavior, check, digest, read, write, ratios
from p3e_metrics import sum_metrics
from guandan.env import HandEnv
from guandan.evaluation.statistics import summarize_matchup

VALIDATION_JOBS = ("validation-constant", "validation-normcap")
ARTIFACT_NAMES = ("results.jsonl", "measurements.jsonl.gz", "replays.jsonl.gz", "behavior.jsonl")


def verify_frozen(root):
    pre = read(root / "preregistration.json")
    check(pre["specification"] == specification(), "frozen P3l specification mismatch")
    check(bool(pre["source_sha256"]), "empty source snapshot")
    for name, expected in pre["source_sha256"].items():
        check(digest(ROOT / name) == expected, f"source drift: {name}")
    for name, expected in pre['input_sha256'].items():
        check(digest(ROOT/name) == expected, 'frozen input drift '+name)
    return digest(root / "preregistration.json")


def resolve_job(job):
    if job in VALIDATION_JOBS:
        return job.removeprefix("validation-"), 314380, True
    arm, seed = job.split("-")
    check(arm in ("constant", "normcap"), "unsupported P3l arm")
    return arm, int(seed), False


def artifact_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def verify_trial(row, measurement, replay, trial):
    """Environment-side integrity checks; guard receives no replay information."""
    tid = trial.trial_id
    check(row["trial_id"] == measurement["trial_id"] == tid, "trial metadata identity")
    check(row["status"] == "ok" and row["illegal_actions"] == row["timeouts"] == 0,
          f"failed trial {tid}: {row['error']}")
    check(replay is not None and measurement["replay_verified"], f"unverified replay: {tid}")
    env = HandEnv.replay(replay)
    settlement = env.state.settlement
    check(env.state.terminal and settlement is not None and
          env.state_digest() == row["terminal_digest"] and
          len(replay["steps"]) == row["steps"] and
          list(settlement.finish_order) == row["finish_order"] and
          int(settlement.winner_team == trial.focal_team) == row["win"] and
          settlement.team_rewards[trial.focal_team] == row["team_reward"],
          f"replay result mismatch: {tid}")
    n = row["steps"]
    check(all(len(measurement[key]) == n for key in
          ("enumeration_ms", "decision_ms", "candidate_counts")), "measurement coverage")
    check(all(type(count) is int and count > 0 for count in measurement["candidate_counts"]),
          "invalid complete candidate counts")
    check(all(math.isfinite(ms) and 0 <= ms <= 2000 for ms in measurement["decision_ms"]),
          "decision deadline")
    check(all(math.isfinite(ms) and ms >= 0 for ms in measurement["enumeration_ms"]),
          "enumeration measurements")
    neural_steps = [i for i, step in enumerate(replay["steps"])
                    if trial.policies[step["player"]] == "dmc"]
    check([item["step"] for item in measurement["neural"]] == neural_steps,
          "complete neural decision coverage")
    for item in measurement["neural"]:
        check(item["scored_candidates"] == measurement["candidate_counts"][item["step"]]
              and item["device"]["type"] == "cuda"
              and math.isfinite(item["inference_ms"]) and math.isfinite(item["roundtrip_ms"])
              and 0 <= item["inference_ms"] <= item["roundtrip_ms"] <= 2000,
              "complete CUDA candidate scoring and deadline")


def verify_saved(output, trials):
    expected = {trial.trial_id for trial in trials}
    check(len(expected) == len(trials), "duplicate scheduled trial IDs")
    for name in ARTIFACT_NAMES:
        seen = set()
        for item in artifact_lines(output / name):
            tid = item["trial_id"]
            check(tid in expected and tid not in seen, f"saved artifact coverage: {name}/{tid}")
            seen.add(tid)
        check(seen == expected, f"saved artifact missing trials: {name}")
    return {name: digest(output / name) for name in ARTIFACT_NAMES}


def evaluate(root, job):
    from experiments.p3l_guard import GPUInferenceGuard
    from p3d_evaluate import run_trial

    prereg_hash = verify_frozen(root)
    arm, seed, validation = resolve_job(job)
    trials = schedule(validation)
    check(len(trials) == (1560 if validation else 208), "fixed evaluation game budget")
    checkpoint = root / "training" / f"{arm}-{seed}" / "final"
    manifest = read(checkpoint / "manifest.json")
    expected = manifest["sha256"]
    check(manifest["config"] == config(seed, arm), "checkpoint config mismatch")
    check(digest(checkpoint / "checkpoint.pt") == expected, "pinned checkpoint hash mismatch")
    output = root / "evaluations" / job
    output.mkdir(parents=True, exist_ok=False)
    binding = dict(utc=datetime.now(timezone.utc).isoformat(), job=job, arm=arm, seed=seed,
        checkpoint=str(checkpoint), candidate_sha256=expected,
        manifest_sha256=digest(checkpoint / "manifest.json"),
        preregistration_sha256=prereg_hash, games=len(trials))
    write(output / "binding.json", binding)
    rows, behaviors = [], []
    guard = None
    started = time.perf_counter()
    try:
        with GPUInferenceGuard(checkpoint, expected, startup_timeout=60., decision_timeout=2.) as guard, \
             (output / "results.jsonl").open("x", encoding="utf-8") as results, \
             gzip.open(output / "measurements.jsonl.gz", "xt", encoding="utf-8") as measurements, \
             gzip.open(output / "replays.jsonl.gz", "xt", encoding="utf-8") as replays, \
             (output / "behavior.jsonl").open("x", encoding="utf-8") as counts_file:
            for trial in trials:
                row, measurement, replay = run_trial(guard, trial)
                # Preserve the original result and measurement even on failure.
                for stream, item in ((results, row), (measurements, measurement),
                        (replays, dict(trial_id=trial.trial_id, replay=replay))):
                    stream.write(json.dumps(item, allow_nan=False) + "\n")
                    stream.flush()
                verify_trial(row, measurement, replay, trial)
                counted = behavior(replay, trial)
                counts_file.write(json.dumps(counted, allow_nan=False) + "\n")
                counts_file.flush()
                rows.append(row)
                behaviors.append(counted)
                if len(rows) % 104 == 0:
                    print(json.dumps(dict(stage="evaluate", job=job, games=len(rows),
                          total=len(trials), elapsed_s=time.perf_counter()-started)), flush=True)
        check(guard.closed and not guard.is_alive(), "inference worker teardown")
        artifact_hashes = verify_saved(output, trials)
        check(digest(checkpoint / "checkpoint.pt") == expected and
              digest(checkpoint / "manifest.json") == binding["manifest_sha256"],
              "checkpoint changed during evaluation")
        check(verify_frozen(root) == prereg_hash, "preregistration changed during evaluation")
        counts = sum_metrics(item["counts"] for item in behaviors)
        summary = {matchup: summarize_matchup(
            [row for row in rows if row["matchup_id"] == matchup],
            [trial for trial in trials if trial.matchup_id == matchup], 5000, 314425)
            for matchup in sorted({trial.matchup_id for trial in trials})}
        if validation and arm == "normcap":
            primary = summary["dmc|dmc|greedy"]["metrics"]["win_rate"]
            passed = (primary["estimate"] >= .55 and primary["ci95"] is not None
                      and primary["ci95"][0] > .5 and not primary["degenerate"])
            gate = "VALIDATION_GATE_PASSED" if passed else "NOT_ESTABLISHED"
        else:
            gate = "SECONDARY_VALIDATION_ONLY" if validation else "DEVELOPMENT_ONLY"
        report = dict(status="PASS", job=job, arm=arm, seed=seed, candidate_sha256=expected,
            preregistration_sha256=prereg_hash, binding_sha256=digest(output / "binding.json"),
            artifact_sha256=artifact_hashes, games=len(rows), summary=summary, counts=counts,
            ratios=ratios(counts), validation_gate=gate, model_promoted=False,
            reserved_test_executed=False, worker_closed=guard.closed,
            worker_pid=guard.worker_pid, worker_exitcode=guard.worker_exitcode,
            elapsed_s=time.perf_counter()-started)
        write(output / "report.json", report)
        print(json.dumps(dict(stage="evaluate", job=job, status="PASS", games=len(rows),
                             validation_gate=gate)), flush=True)
        return report
    except BaseException as error:
        if guard is not None and not guard.closed:
            guard.close()
        write(output / "failure.json", dict(status="FAIL", job=job, completed_games=len(rows),
            error=type(error).__name__, message=str(error), elapsed_s=time.perf_counter()-started,
            worker_closed=guard.closed if guard is not None else None,
            worker_alive=guard.is_alive() if guard is not None else None,
            model_promoted=False, reserved_test_executed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("job")
    args = parser.parse_args()
    root = args.root.resolve()
    verify_frozen(root)
    spec = specification()
    if args.stage == "train":
        check(args.job in spec["training_order"], "training job outside frozen order")
        from experiments.p3l_job import train
        train(root, args.job)
        verify_frozen(root)
    else:
        check(args.job in list(spec["development_order"]) + list(VALIDATION_JOBS),
              "evaluation job outside frozen schedule")
        evaluate(root, args.job)


if __name__ == "__main__":
    main()
