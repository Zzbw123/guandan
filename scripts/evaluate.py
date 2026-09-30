"""Run or independently recompute a preregistered paired baseline evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from _bootstrap import ROOT
from guandan.agents.baselines import make_agent
from guandan.evaluation.config import validate_config
from guandan.evaluation.contracts import EVALUATION_VERSION, POLICIES
from guandan.evaluation.reporting import markdown_report, sample_summary
from guandan.evaluation.runner import run_trial
from guandan.evaluation.schedule import build_trials
from guandan.evaluation.statistics import summarize_matchup
from guandan.types import ACTION_VERSION, OBSERVATION_VERSION, RULES_VERSION


def canonical(value):
    # JSON stringifies numeric mapping keys. Normalize before sorting so a
    # freshly computed summary and its disk roundtrip have identical bytes.
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_manifest():
    files = [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py"),
             *ROOT.joinpath("tests").glob("*.py"), *ROOT.joinpath("configs").rglob("*.json"),
             ROOT / "docs/rules.md", ROOT / "docs/P2_PROTOCOL.md", ROOT / "pyproject.toml"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}


def bootstrap_seed(base, matchup_id):
    value = hashlib.sha256(f"{base}:{matchup_id}".encode()).digest()
    return int.from_bytes(value[:8], "big")


def run_group(trials, config):
    rows = []
    measurements = {key: [] for key in ("enumeration_ms", "decision_ms", "candidate_counts")}
    for trial in trials:
        row, samples = run_trial(trial, config["policy_seed"], config["max_steps"], config["decision_timeout_ms"])
        rows.append(row)
        for key in measurements:
            measurements[key].extend(samples[key])
    return rows, measurements


def prepare(config, splits):
    seeds, matchups = validate_config(config, splits)
    expected = {m.key: build_trials(m, seeds) for m in matchups}
    jobs = []
    for key, trials in expected.items():
        groups = {}
        for trial in trials:
            groups.setdefault(trial.deal_index, []).append(trial)
        jobs.extend((key, index, group) for index, group in groups.items())
    return seeds, expected, jobs


def statistics_for(rows_by_matchup, expected, config):
    summaries, errors = {}, {}
    for key, trials in expected.items():
        try:
            summaries[key] = summarize_matchup(rows_by_matchup[key], trials, config["bootstrap_replicates"],
                                                bootstrap_seed(config["bootstrap_seed"], key))
        except ValueError as exc:
            errors[key] = str(exc)
    return summaries, errors


def outcome_digest(rows):
    return hashlib.sha256(canonical(sorted(rows, key=lambda r: r["trial_id"])).encode("utf-8")).hexdigest()


def primary_result(valid, summaries, config):
    primary = summaries.get(config["primary_matchup"])
    ci = primary["metrics"]["win_rate"]["ci95"] if primary else None
    conclusion = "ABOVE_50_PERCENT" if valid and ci and ci[0] > .5 else "NOT_ESTABLISHED" if valid else "INVALID"
    return {"matchup_id": config["primary_matchup"], "metric": "win_rate", "conclusion": conclusion,
            "scope": f"Prespecified baseline comparison on {config['split']}; secondary CIs are pointwise"}


def execute(config_path, output):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "configs/evaluation/seed-splits.json").read_text(encoding="utf-8"))
    seeds, expected, jobs = prepare(config, splits)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest()
    prereg = {"evaluation_version": EVALUATION_VERSION, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "config_file_sha256": digest(config_path), "seed_splits": splits,
              "seed_list": seeds, "expected_games": sum(map(len, expected.values())), "source_sha256": manifest,
              "rules_version": RULES_VERSION, "observation_version": OBSERVATION_VERSION, "action_version": ACTION_VERSION,
              "policy_versions": {name: make_agent(name).version for name in POLICIES},
              "model_hash": None, "feature_version": None, "python": sys.version, "platform": platform.platform(),
              "primary_matchup": config["primary_matchup"], "primary_metric": config["primary_metric"],
              "reserved_test_executed": False}
    write_json(output / "preregistration.json", prereg)
    rows_by_matchup = {key: [] for key in expected}
    timings = {key: {name: [] for name in ("enumeration_ms", "decision_ms", "candidate_counts")} for key in expected}
    started = time.perf_counter()
    completed, failed, worker_errors = 0, 0, []
    with (ProcessPoolExecutor(max_workers=config["workers"]) as pool,
          (output / "results.jsonl").open("x", encoding="utf-8") as results_file,
          gzip.open(output / "measurements.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as measurement_file,
          (output / "progress.jsonl").open("x", encoding="utf-8") as progress_file):
        futures = {pool.submit(run_group, group, config): (key, index, group) for key, index, group in jobs}
        for batch_number, future in enumerate(as_completed(futures), 1):
            key, index, group = futures[future]
            try:
                rows, samples = future.result()
                rows_by_matchup[key].extend(rows)
                completed += len(rows)
                failed += sum(row["status"] != "ok" for row in rows)
                for row in rows:
                    results_file.write(canonical(row) + "\n")
                measurement_file.write(canonical({"matchup_id": key, "deal_index": index,
                                                 "trial_ids": [trial.trial_id for trial in group], **samples}) + "\n")
                for name in samples:
                    timings[key][name].extend(samples[name])
            except Exception as exc:
                worker_errors.append({"matchup_id": key, "deal_index": index, "error": f"{type(exc).__name__}: {exc}"})
            progress = {"batches": batch_number, "total_batches": len(jobs), "recorded_games": completed,
                        "expected_games": prereg["expected_games"], "failed_games": failed,
                        "worker_errors": len(worker_errors), "elapsed_seconds": round(time.perf_counter() - started, 3)}
            progress_file.write(canonical(progress) + "\n")
            progress_file.flush()
            results_file.flush()
            if batch_number % 100 == 0 or batch_number == len(jobs):
                print(canonical(progress), flush=True)
    print("Computing preregistered clustered intervals...", flush=True)
    summaries, statistics_errors = statistics_for(rows_by_matchup, expected, config)
    rows = [row for group in rows_by_matchup.values() for row in group]
    unchanged = manifest == source_manifest()
    illegal = sum(row["illegal_actions"] for row in rows)
    timeouts = sum(row["timeouts"] for row in rows)
    valid = unchanged and not (failed or illegal or timeouts or worker_errors or statistics_errors) and completed == prereg["expected_games"]
    report = {"evaluation_version": EVALUATION_VERSION, "status": "PASS" if valid else "FAIL",
              "config": config, "output_directory": str(output), "expected_games": prereg["expected_games"],
              "recorded_games": completed, "failed_games": failed, "illegal_actions": illegal, "timeouts": timeouts,
              "elapsed_seconds": time.perf_counter() - started, "source_unchanged": unchanged,
              "statistics": summaries, "statistics_errors": statistics_errors, "worker_errors": worker_errors,
              "performance": {key: {name: sample_summary(values) for name, values in group.items()} for key, group in timings.items()},
              "primary": primary_result(valid, summaries, config),
              "promotion": {"eligible": False, "status": "NOT_APPLICABLE", "reason": "No learned candidate; reserved_test was not executed"},
              "outcome_sha256": outcome_digest(rows),
              "artifacts_sha256": {name: digest(output / name) for name in
                                  ("preregistration.json", "results.jsonl", "measurements.jsonl.gz", "progress.jsonl")}}
    write_json(output / "report.json", report)
    (output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(canonical({key: report[key] for key in ("status", "recorded_games", "failed_games", "elapsed_seconds", "primary", "outcome_sha256")}), flush=True)
    return 0 if valid else 1


def audit(directory):
    directory = directory.resolve()
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    prereg = json.loads((directory / "preregistration.json").read_text(encoding="utf-8"))
    for name, expected_hash in report["artifacts_sha256"].items():
        if name not in {"preregistration.json", "results.jsonl", "measurements.jsonl.gz", "progress.jsonl"} or digest(directory / name) != expected_hash:
            raise ValueError(f"Artifact hash mismatch: {name}")
    if len(report["artifacts_sha256"]) != 4 or prereg["source_sha256"] != source_manifest():
        raise ValueError("Missing artifacts or current code differs from preregistered source")
    if canonical(report["config"]) != canonical(prereg["config"]):
        raise ValueError("Config mismatch")
    if (prereg["reserved_test_executed"] is not False or prereg["model_hash"] is not None
            or prereg["feature_version"] is not None or prereg["evaluation_version"] != EVALUATION_VERSION
            or report["evaluation_version"] != EVALUATION_VERSION):
        raise ValueError("Invalid scope or experiment version")
    seeds, expected, jobs = prepare(prereg["config"], prereg["seed_splits"])
    if seeds != prereg["seed_list"] or sum(map(len, expected.values())) != prereg["expected_games"]:
        raise ValueError("Seed list or expected game count mismatch")
    rows_by_matchup = {key: [] for key in expected}
    with (directory / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows_by_matchup[row["matchup_id"]].append(row)
    all_rows = [row for group in rows_by_matchup.values() for row in group]
    if outcome_digest(all_rows) != report["outcome_sha256"]:
        raise ValueError("Outcome digest mismatch")
    summaries, errors = statistics_for(rows_by_matchup, expected, prereg["config"])
    if errors or canonical(summaries) != canonical(report["statistics"]):
        raise ValueError(f"Statistics recomputation mismatch: {errors}")
    job_map = {(key, index): group for key, index, group in jobs}
    timings = {key: {name: [] for name in ("enumeration_ms", "decision_ms", "candidate_counts")} for key in expected}
    seen = set()
    with gzip.open(directory / "measurements.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            key = (sample["matchup_id"], sample["deal_index"])
            if key in seen or key not in job_map or sample["trial_ids"] != [trial.trial_id for trial in job_map[key]]:
                raise ValueError("Invalid measurement group")
            seen.add(key)
            for name in timings[key[0]]:
                timings[key[0]][name].extend(sample[name])
    if seen != set(job_map):
        raise ValueError("Missing measurement group")
    performance = {key: {name: sample_summary(values) for name, values in group.items()} for key, group in timings.items()}
    if canonical(performance) != canonical(report["performance"]):
        raise ValueError("Performance recomputation mismatch")
    count = len(all_rows)
    if (report["status"] != "PASS" or report["source_unchanged"] is not True or count != prereg["expected_games"]
            or report["recorded_games"] != count or report["expected_games"] != count):
        raise ValueError("Report is not a complete accepted run")
    for name in ("failed_games", "illegal_actions", "timeouts"):
        if type(report[name]) is not int or report[name] != 0:
            raise ValueError(f"Invalid failure counter: {name}")
    if report["worker_errors"] or report["statistics_errors"]:
        raise ValueError("Run contains unresolved failures")
    if canonical(report["primary"]) != canonical(primary_result(True, summaries, prereg["config"])):
        raise ValueError("Primary conclusion mismatch")
    if report["promotion"] != {"eligible": False, "status": "NOT_APPLICABLE", "reason": "No learned candidate; reserved_test was not executed"}:
        raise ValueError("This baseline report cannot promote a learned model")
    for key, group in rows_by_matchup.items():
        total_steps = sum(row["steps"] for row in group)
        if any(len(values) != total_steps for values in timings[key].values()):
            raise ValueError("Timing sample counts do not match successful decision counts")
    result = {"status": "PASS", "recorded_games": count, "matchups_recomputed": len(summaries),
              "source_readback": True, "statistics_recomputed": True, "performance_recomputed": True,
              "outcome_sha256": report["outcome_sha256"], "report_sha256": digest(directory / "report.json")}
    target = directory / "audit.json"
    if target.exists():
        if canonical(json.loads(target.read_text(encoding="utf-8"))) != canonical(result):
            raise ValueError("Existing audit differs; preserving it")
    else:
        write_json(target, result)
    print(canonical(result), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/p2-validation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluations/p2-validation-v1")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    return audit(args.audit) if args.audit else execute(args.config, args.output)


if __name__ == "__main__":
    sys.exit(main())
