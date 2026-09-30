"""One pinned P5f development evaluation job; no validation or promotion."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
# The accepted trial runner uses scripts/_bootstrap and the frozen P3d protocol.
# Its imports are intentionally isolated from the P5f protocol module name.
for directory in (ROOT / "scripts", ROOT, ROOT / "experiments", ROOT / "experiments/p3d"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.p5f_protocol import ARMS, INITS, config, schedule
from experiments.p5f_guard import GPUInferenceGuard
from p3e_common import behavior, check, digest, ratios, read, write
from p3e_metrics import sum_metrics
from p5e_worker import verify_saved, verify_trial
from p3d_evaluate import run_trial
from guandan.evaluation.statistics import summarize_matchup


def _job(job: str) -> tuple[str, int]:
    if type(job) is not str:
        raise ValueError("P5f job must be objective-initializer")
    parts = job.rsplit("-", 1)
    if (len(parts) != 2 or parts[0] not in ARMS
            or not parts[1].isdecimal() or int(parts[1]) not in INITS
            or job != f"{parts[0]}-{int(parts[1])}"):
        raise ValueError("unsupported P5f evaluation job")
    return parts[0], int(parts[1])


def _checkpoint(root: Path, job: str, arm: str, seed: int) -> tuple[Path, str, str]:
    checkpoint = root / "training" / job / "final"
    manifest_path = checkpoint / "manifest.json"
    manifest = read(manifest_path)
    expected = read(root / "candidates.json")["candidates"][job]
    check(type(expected) is str and len(expected) == 64, "invalid frozen candidate SHA256")
    check(manifest["sha256"] == expected, "frozen final candidate registry mismatch")
    check(manifest["config"] == config(seed, arm), "checkpoint config mismatch")
    check(type(manifest["updates"]) is int and manifest["updates"] > 0,
          "checkpoint has no successful updates")
    check(type(manifest["samples"]) is int and manifest["samples"] >= manifest["updates"],
          "checkpoint sample count mismatch")
    check(type(manifest["candidates"]) is int and manifest["candidates"] >= manifest["samples"],
          "checkpoint candidate count mismatch")
    check(type(manifest["corpus_sha256"]) is str and len(manifest["corpus_sha256"]) == 64,
          "checkpoint corpus pin missing")
    check(digest(checkpoint / "checkpoint.pt") == expected, "pinned checkpoint hash mismatch")
    return checkpoint, expected, digest(manifest_path)


def evaluate(root: Path, job: str) -> dict:
    """Run all frozen P5f development games for one final candidate.

    The guard receives only observations and complete legal actions from
    ``run_trial``. Every original trial artifact is flushed before validation.
    """
    from experiments.p5f_run_common import verify_frozen

    root = Path(root).resolve()
    arm, seed = _job(job)
    prereg_hash = verify_frozen(root)
    trials = schedule()
    check(len(trials) == 416 and
          {trial.matchup_id for trial in trials} == {"dmc|dmc|greedy", "dmc|dmc|random"},
          "fixed P5f development game budget")
    checkpoint, expected, manifest_sha = _checkpoint(root, job, arm, seed)
    output = root / "evaluations" / job
    output.mkdir(parents=True, exist_ok=False)
    rows, behaviors = [], []
    guard = None
    started = time.perf_counter()
    try:
        binding = dict(
            utc=datetime.now(timezone.utc).isoformat(), job=job, arm=arm, seed=seed,
            checkpoint=str(checkpoint), candidate_sha256=expected,
            manifest_sha256=manifest_sha,
            preregistration_sha256=prereg_hash, games=len(trials),
        )
        write(output / "binding.json", binding)
        with GPUInferenceGuard(
                checkpoint, expected, startup_timeout=60., decision_timeout=2.,
                expected_manifest_sha256=manifest_sha) as guard, \
             (output / "results.jsonl").open("x", encoding="utf-8") as results, \
             gzip.open(output / "measurements.jsonl.gz", "xt", encoding="utf-8") as measurements, \
             gzip.open(output / "replays.jsonl.gz", "xt", encoding="utf-8") as replays, \
             (output / "behavior.jsonl").open("x", encoding="utf-8") as behavior_file:
            for trial in trials:
                row, measurement, replay = run_trial(guard, trial)
                for stream, item in (
                        (results, row), (measurements, measurement),
                        (replays, dict(trial_id=trial.trial_id, replay=replay))):
                    stream.write(json.dumps(item, allow_nan=False) + "\n")
                    stream.flush()
                verify_trial(row, measurement, replay, trial)
                counted = behavior(replay, trial)
                behavior_file.write(json.dumps(counted, allow_nan=False) + "\n")
                behavior_file.flush()
                rows.append(row)
                behaviors.append(counted)
                if len(rows) % 104 == 0:
                    print(json.dumps(dict(stage="evaluate", job=job, games=len(rows),
                                          total=len(trials), elapsed_s=time.perf_counter()-started)),
                          flush=True)
        check(guard.closed and not guard.is_alive(), "inference worker teardown")
        artifact_hashes = verify_saved(output, trials)
        check(digest(checkpoint / "checkpoint.pt") == expected and
              digest(checkpoint / "manifest.json") == manifest_sha,
              "checkpoint changed during evaluation")
        check(verify_frozen(root) == prereg_hash, "preregistration changed during evaluation")
        check(read(root / "candidates.json")["candidates"][job] == expected,
              "candidate registry changed during evaluation")
        counts = sum_metrics(item["counts"] for item in behaviors)
        summary = {matchup: summarize_matchup(
            [row for row in rows if row["matchup_id"] == matchup],
            [trial for trial in trials if trial.matchup_id == matchup], 5000, 314565)
            for matchup in sorted({trial.matchup_id for trial in trials})}
        report = dict(
            status="PASS", job=job, arm=arm, seed=seed,
            candidate_sha256=expected, preregistration_sha256=prereg_hash,
            binding_sha256=digest(output / "binding.json"),
            artifact_sha256=artifact_hashes, games=len(rows), summary=summary,
            counts=counts, ratios=ratios(counts), validation_gate="DEVELOPMENT_ONLY",
            model_promoted=False, reserved_test_executed=False,
            worker_closed=guard.closed, worker_pid=guard.worker_pid,
            worker_exitcode=guard.worker_exitcode,
            elapsed_s=time.perf_counter()-started,
        )
        write(output / "report.json", report)
        print(json.dumps(dict(stage="evaluate", job=job, status="PASS",
                              games=len(rows), validation_gate="DEVELOPMENT_ONLY")),
              flush=True)
        return report
    except BaseException as error:
        if guard is not None and not guard.closed:
            guard.close()
        write(output / "failure.json", dict(
            status="FAIL", job=job, completed_games=len(rows),
            error=type(error).__name__, message=str(error),
            elapsed_s=time.perf_counter()-started,
            worker_closed=guard.closed if guard is not None else None,
            worker_alive=guard.is_alive() if guard is not None else None,
            model_promoted=False, reserved_test_executed=False,
        ))
        raise
