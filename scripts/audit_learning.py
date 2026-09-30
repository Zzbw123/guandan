"""Read-only independent arithmetic/source audit of the fixed P3a CPU receipt."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import zipfile

from _bootstrap import ROOT


def check(condition, label):
    if not condition:
        raise ValueError(label)


def audit(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory/name).read_text(encoding="utf-8"))
    prereg = read("preregistration.json")
    schedule = read("evaluation-preregistration.json")
    report = read("report.json")
    resume = read("resume-audit.json")
    manifest = json.loads((directory/"candidate/manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory/"evaluation.jsonl").read_text().splitlines()]
    training = [json.loads(line) for line in (directory/"training.jsonl").read_text().splitlines()]
    digest = sha256((directory/"candidate/checkpoint.pt").read_bytes()).hexdigest()
    check(digest == manifest["sha256"] == report["checkpoint_sha256"] == schedule["checkpoint_sha256"], "candidate hash")
    check(prereg["training_seeds"] == list(range(100500,100504)), "training schedule")
    check(prereg["evaluation_seeds"] == [101000,101001] and len(rows) == 48, "evaluation budget")
    check(prereg["reserved_test_executed"] is False and report["reserved_test_executed"] is False, "reserved test")
    check(len(training) == 4 and [r["seed"] for r in training] == prereg["training_seeds"], "training rows")
    check(all(r["steps"] == r["samples"] and r["replay_verified"] and math.isfinite(r["mean_loss"]) for r in training), "training integrity")
    check(sum(r["steps"] for r in training) == report["training_decisions"], "training decisions")
    check(training[-2:] == resume["replayed_rows"] and resume["model_optimizer_rng_counters_bitwise_equal"] is True, "resume trajectories")
    expected = {}
    for opponent in ("greedy", "random", "team"):
        for index, seed in enumerate((101000,101001)):
            for rotation in range(4):
                for swap in range(2):
                    key = f"dmc|dmc|{opponent}:{seed}:{rotation}:{swap}:0"
                    expected[key] = (opponent, seed, 2+index, rotation, swap, (rotation+swap)%2)
    seen = set()
    for row in rows:
        key = row["trial_id"]
        check(key in expected and key not in seen, "missing/duplicate/unexpected evaluation")
        seen.add(key)
        check(tuple(row[k] for k in ("opponent", "deal_seed", "level", "rotation", "swap", "focal_team")) == expected[key], "trial metadata")
        check(type(row["win"]) is int and row["win"] in (0,1) and row["reward"] == 2*row["win"]-1, "reward direction")
        check(row["replay_verified"] is True and 0 < row["steps"] <= 1000, "terminal/replay")
        check(len(row["decision_ms"]) == len(row["candidate_counts"]) > 0, "measurement completeness")
        check(all(math.isfinite(v) and v >= 0 for v in row["decision_ms"]), "finite measurements")
    computed = {}
    for opponent in ("greedy", "random", "team"):
        subset = [r for r in rows if r["opponent"] == opponent]
        wins = sum(r["win"] for r in subset)
        check(report["results"][opponent]["win_rate"] == wins/16, "independent arithmetic")
        check(report["results"][opponent]["confidence_interval"] is None, "no promotion interval")
        computed[opponent] = dict(wins=wins, games=16, win_rate=wins/16)
    registered = prereg["source_sha256"]
    for path, expected_hash in registered.items():
        check(sha256((ROOT/path).read_bytes()).hexdigest() == expected_hash, f"source drift: {path}")
    with zipfile.ZipFile(directory/"source-snapshot.zip") as archive:
        check(set(archive.namelist()) == set(registered), "snapshot member set")
        for path, expected_hash in registered.items():
            check(sha256(archive.read(path)).hexdigest() == expected_hash, f"snapshot hash: {path}")
    p2 = json.loads((ROOT/"artifacts/evaluations/p2-validation-v1/preregistration.json").read_text())["source_sha256"]
    for path, expected_hash in p2.items():
        check(sha256((ROOT/path).read_bytes()).hexdigest() == expected_hash, f"P2 drift: {path}")
    times = sorted(t for r in rows for t in r["decision_ms"])
    result = dict(status="PASS", candidate_sha256=digest, evaluation_games=48,
                  independent_deals=2, recomputed=computed, registered_sources=len(registered),
                  protected_p2_files=len(p2), snapshot_verified=True,
                  neural_decisions=len(times), decision_ms_p50=times[math.ceil(.5*len(times))-1],
                  decision_ms_p95=times[math.ceil(.95*len(times))-1], decision_ms_max=max(times),
                  max_candidates=max(n for r in rows for n in r["candidate_counts"]),
                  auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(text+"\n")
    print(text)
