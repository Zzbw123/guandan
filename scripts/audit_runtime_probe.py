"""Independent disk audit for P3b1: no benchmark implementation is imported."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

from _bootstrap import ROOT


def require(value, label):
    if not value:
        raise ValueError(label)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protected_sources():
    counts = {}
    for stage, path in (
        ("p2", "artifacts/evaluations/p2-validation-v1/preregistration.json"),
        ("p3a", "artifacts/evaluations/p3a-cpu-v1/preregistration.json"),
    ):
        sources = read(ROOT / path)["source_sha256"]
        for name, expected in sources.items():
            require(digest(ROOT / name) == expected, f"{stage} source changed: {name}")
        counts[stage] = len(sources)
    return counts


def check_snapshot(directory, sources):
    with zipfile.ZipFile(directory / "source-snapshot.zip") as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "duplicate ZIP members")
        require(set(archive.namelist()) == set(sources), "ZIP member set")
        for name, expected in sources.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, f"ZIP hash: {name}")
            require(digest(ROOT / name) == expected, f"current source: {name}")


def audit(directory):
    directory = Path(directory)
    prereg = read(directory / "preregistration.json")
    report = read(directory / "report.json")
    require(report["status"] == "PASS", "benchmark not PASS")
    require(not (directory / "failure.json").exists(), "failure receipt exists")
    require(prereg["command"]["devices"] == ["cpu", "cuda"], "both devices required")
    require(prereg["command"]["repeats"] == 20 and prereg["command"]["hands"] == 4, "fixed repeat budget")
    require(prereg["seed"] == 271828 and prereg["chunk_size"] == 256, "model/score configuration")
    require(prereg["deals"] == [[102000+i, level, i] for i, level in enumerate((2,5,10,14))], "episode schedule")
    require(prereg["throughput_deals"] == [[102100+i, 2+i, i%4, 900000+i] for i in range(8)], "worker schedule")
    require(prereg["minibatches"] == [64,256,1024], "batch budget")
    before = prereg["source_sha256_before"]
    require(before == report["source_sha256_before"] == report["source_sha256_after"], "source drift")
    require(report["source_unchanged"] is True, "source flag")
    check_snapshot(directory, before)
    require(digest(directory / "source-snapshot.zip") == report["source_snapshot_sha256"], "snapshot archive hash")
    protected = protected_sources()
    runtime = report["runtime"]
    require(runtime["torch"] == "2.9.1+cu128" and runtime["cuda"]["available"], "CUDA runtime")
    require(runtime["cuda"]["torch_cuda"] == "12.8" and runtime["cuda"]["capability"] == [12,0], "tested hardware")
    require(runtime["torch_threads"] == 1 and runtime["deterministic_algorithms"], "deterministic configuration")
    require(not runtime["tf32_matmul"] and not runtime["tf32_cudnn"], "TF32 disabled")
    require(runtime["cublas_workspace_config"] == ":4096:8", "CUBLAS configuration")
    result = report["results"]
    distributions = []

    def check_latency(item, label):
        values = item["raw_ms"]
        require(len(values) == item["count"] == 20, f"{label} raw budget")
        require(all(math.isfinite(x) and x >= 0 for x in values), f"{label} finite time")
        ordered = sorted(values)
        # Independently interpolate the two order statistics.
        for key, q in (("p50_ms", .5), ("p95_ms", .95)):
            pos = q * (len(ordered)-1)
            a, b = math.floor(pos), math.ceil(pos)
            expected = ordered[a]*(1-(pos-a)) + ordered[b]*(pos-a)
            require(math.isclose(item[key], expected, rel_tol=1e-12, abs_tol=1e-12), f"{label} {key}")
        require(item["min_ms"] == min(values) and item["max_ms"] == max(values), f"{label} extrema")
        distributions.append(label)

    fixtures = result["score"]["fixtures"]
    require(len(fixtures) == 2 and fixtures[1]["candidates"] == 8769, "full high-branch fixture")
    from guandan.env import HandEnv
    from guandan.rules.cards import card_id
    normal = HandEnv()
    normal.reset(102000, initial_level=2)
    hand = tuple(card_id(rank, suit, copy) for rank in (2,3,4) for suit in range(4) for copy in range(2)) + (18,72,7)
    rest = [c for c in range(108) if c not in hand]
    high = HandEnv.from_hands((hand,tuple(rest[:27]),tuple(rest[27:54]),tuple(rest[54:])),7)
    for fixture, env in zip(fixtures,(normal,high)):
        actions = env.legal_actions(0)
        encoded = json.dumps([a.to_dict() for a in actions], sort_keys=True, separators=(",", ":")).encode()
        require(hashlib.sha256(encoded).hexdigest() == fixture["candidate_sha256"], "regenerated full candidate hash")
        require(len(actions) == fixture["candidates"], "regenerated full candidate count")
    for row in fixtures:
        for device in ("cpu", "cuda"):
            check_latency(row["latency"][device], f"score/{row['fixture']}/{device}")
            scores = row["scores_by_device"][device]
            require(len(scores) == row["candidates"] and all(math.isfinite(x) for x in scores), "raw score completeness")
            coverage = row["coverage"][device]
            expected_chunks = [min(256, row["candidates"]-i) for i in range(0,row["candidates"],256)]
            require(coverage["chunk_lengths"] == expected_chunks, "candidate chunk coverage")
            require(coverage["scored_candidates"] == sum(expected_chunks) and coverage["max_chunk"] == max(expected_chunks), "coverage totals")
        require(row["allclose"] is True, "CPU/CUDA scores")
        a, b = row["scores_by_device"]["cpu"], row["scores_by_device"]["cuda"]
        require(all(abs(x-y) <= 1e-5 + 1e-4*abs(y) for x,y in zip(a,b)), "independent allclose")
        require(math.isclose(row["max_abs_diff"], max(abs(x-y) for x,y in zip(a,b)), rel_tol=1e-7, abs_tol=1e-12), "independent max score error")
        require(row["argmax_equal"] == (a.index(max(a)) == b.index(max(b))), "independent argmax")
        require(row["cpu_reference_allclose"] and row["cpu_reference_max_abs_diff"] <= 1e-5, "P3a reference scoring")

    collect = result["collect"]
    require(len(collect["hands"]) == 8, "sample collection budget")
    require([[r["seed"],r["level"],r["starting_player"],r["policy_seed"]] for r in collect["hands"]] == prereg["throughput_deals"], "collection schedule")
    require(collect["samples"] == sum(r["samples"] for r in collect["hands"]), "sample collection count")
    require(result["minibatch"]["actual_samples_collected"] == collect["samples"], "minibatch data origin")
    for size in (64,256,1024):
        for device in ("cpu", "cuda"):
            entry = result["minibatch"]["minibatches"][str(size)][device]
            check_latency(entry["latency"], f"update/{size}/{device}")
            require(entry["samples"] == size and entry["finite"] and entry["parameters_updated_each_repeat"], "update integrity")
            require(len(entry["losses"]) == 20 and all(math.isfinite(x) and x >= 0 for x in entry["losses"]), "loss history")

    def check_hand(row):
        require(100000 <= row["seed"] < 110000 and row["replay_verified"] is True, "development/replay")
        require(row["samples"] == row["steps"] and 0 < row["steps"] <= 1000, "decision budget")
        require(row["team_rewards"] in ([1,-1],[-1,1]), "team reward direction")
        require(len(row["terminal_digest"]) == len(row["action_digest"]) == 64, "trajectory digests")
        for key, value in row.items():
            if key.endswith("_ms"):
                require(math.isfinite(value) and value >= 0, f"finite {key}")
        components = sum(row.get(key,0) for key in ("enumeration_ms", "score_ms", "update_ms", "replay_ms", "other_ms"))
        require(math.isclose(row["total_ms"], components, rel_tol=1e-8, abs_tol=1e-5), "wall time decomposition")

    for row in collect["hands"]:
        check_hand(row)
    for device in ("cpu", "cuda"):
        entry = result["episodes"]["devices"][device]
        rows = entry["hands"]
        require([[r["seed"],r["level"],r["starting_player"]] for r in rows] == prereg["deals"], "episode metadata")
        for row in rows:
            check_hand(row)
            require(row["updates"] == math.ceil(row["samples"]/64), "episode update count")
        require(entry["samples"] == sum(r["samples"] for r in rows), "episode sample total")
        require(entry["updates"] == sum(r["updates"] for r in rows), "episode update total")
        require(math.isclose(entry["total_ms"], sum(r["total_ms"] for r in rows), rel_tol=1e-10), "episode time total")
        for key, value in entry["component_ms"].items():
            require(math.isclose(value, sum(r[key] for r in rows), rel_tol=1e-10), f"component sum {key}")

    baseline = None
    for workers in ("1", "2", "4"):
        entry = result["workers"]["workers"][workers]
        rows = entry["hands"]
        require([[r["seed"],r["level"],r["starting_player"],r["policy_seed"]] for r in rows] == prereg["throughput_deals"], "worker metadata")
        for row in rows:
            check_hand(row)
            require(row["model_state_sha256"] == result["workers"]["model_state_sha256"], "worker model identity")
        signature = [(r["seed"],r["terminal_digest"],r["action_digest"],r["samples"],r["team_rewards"]) for r in rows]
        if baseline is None:
            baseline = signature
        require(signature == baseline and entry["signature_match"], "worker trajectory equality")
        require(entry["worker_inner_ms"] == [r["total_ms"] for r in rows], "worker raw times")
        require(math.isclose(entry["worker_inner_sum_ms"], sum(entry["worker_inner_ms"]), rel_tol=1e-10), "worker time sum")
        require(entry["cold_wall_ms"] >= max(entry["worker_inner_ms"]), "cold timer covers tasks")
    return dict(status="PASS", registered_sources=len(before), protected_sources=protected,
                latency_distributions_recomputed=len(distributions),
                full_candidate_count=8769, sequential_learning_hands=8,
                static_sampling_unique_deals=8, static_sampling_runs=24,
                reserved_test_executed=False, promotion="NOT_ELIGIBLE",
                report_sha256=digest(directory / "report.json"),
                preregistration_sha256=digest(directory / "preregistration.json"),
                auditor_sha256=digest(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)
