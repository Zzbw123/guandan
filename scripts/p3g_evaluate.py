"""One frozen P3g teacher phase1 development-only evaluation; no training."""
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

from p3f_protocol import config, schedule
from p3e_common import behavior, check, digest, read, write, ratios
from p3e_metrics import sum_metrics
from guandan.env import HandEnv
from guandan.evaluation.statistics import summarize_matchup

def verify_frozen(root):
    """Verify both complete source and existing-input snapshots, fail closed."""
    pre = read(root / "preregistration.json")
    for key in ("source_sha256", "input_sha256"):
        pins = pre.get(key)
        check(type(pins) is dict and bool(pins), f"empty {key} snapshot")
        for name, expected in pins.items():
            check(type(name) is str and type(expected) is str, "invalid frozen pin")
            path = (ROOT / name).resolve()
            check(not Path(name).is_absolute() and path.is_relative_to(ROOT.resolve()),
                  f"unsafe frozen path: {name}")
            check(digest(path) == expected, f"frozen drift: {name}")
    return digest(root / "preregistration.json")


from p3f_worker import verify_trial, verify_saved


def evaluate(root, seed):
    from experiments.p3g_guard import GPUInferenceGuard, validate_seed
    from p3d_evaluate import run_trial

    validate_seed(seed)
    prereg_hash = verify_frozen(root)
    job, arm = f"teacher-phase1-{seed}", "teacher"
    trials = schedule(False)
    check(len(trials) == 208, "fixed development evaluation game budget")
    checkpoint = ROOT / "artifacts/evaluations/p3f-teacher-v1/training" / f"teacher-{seed}" / "phase1"
    pre = read(root / "preregistration.json")
    raw_name = (checkpoint / "raw.pt").relative_to(ROOT).as_posix()
    manifest_name = (checkpoint / "manifest.json").relative_to(ROOT).as_posix()
    check(raw_name in pre["input_sha256"] and manifest_name in pre["input_sha256"],
          "phase1 inputs must be explicitly externally pinned")
    expected = pre["input_sha256"][raw_name]
    manifest_pin = pre["input_sha256"][manifest_name]
    manifest = read(checkpoint / "manifest.json")
    check(manifest["config"] == config(seed) and manifest["sha256"] == expected,
          "phase1 manifest configuration/raw pin mismatch")
    output = root / "evaluations" / job
    output.mkdir(parents=True, exist_ok=False)
    binding = dict(utc=datetime.now(timezone.utc).isoformat(), job=job, arm=arm, seed=seed,
        checkpoint=str(checkpoint), candidate_sha256=expected, phase1_model_sha256=manifest["model_sha256"],
        manifest_sha256=manifest_pin,
        preregistration_sha256=prereg_hash, games=len(trials))
    write(output / "binding.json", binding)
    rows, behaviors = [], []
    guard = None
    started = time.perf_counter()
    try:
        with GPUInferenceGuard(checkpoint, expected, startup_timeout=60., decision_timeout=2.,
                               seed=seed, expected_manifest_sha256=manifest_pin) as guard, \
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
        check(digest(checkpoint / "raw.pt") == expected and
              digest(checkpoint / "manifest.json") == binding["manifest_sha256"],
              "checkpoint changed during evaluation")
        check(verify_frozen(root) == prereg_hash, "preregistration changed during evaluation")
        counts = sum_metrics(item["counts"] for item in behaviors)
        summary = {matchup: summarize_matchup(
            [row for row in rows if row["matchup_id"] == matchup],
            [trial for trial in trials if trial.matchup_id == matchup], 5000, 314395)
            for matchup in sorted({trial.matchup_id for trial in trials})}
        gate = "DEVELOPMENT_ONLY"
        report = dict(status="PASS", job=job, arm=arm, seed=seed, candidate_sha256=expected,
            preregistration_sha256=prereg_hash, binding_sha256=digest(output / "binding.json"),
            artifact_sha256=artifact_hashes, games=len(rows), summary=summary, counts=counts,
            ratios=ratios(counts), validation_gate=gate, model_promoted=False,
            reserved_test_executed=False, validation_executed=False, training_executed=False,
            phase1_model_sha256=manifest["model_sha256"], bootstrap_seed=314395,
            worker_closed=guard.closed,
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
    parser.add_argument("seed", type=int, choices=(314380, 314381, 314382))
    args = parser.parse_args()
    evaluate(args.root.resolve(), args.seed)


if __name__ == "__main__":
    main()
