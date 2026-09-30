"""Independent disk arithmetic, CUDA execution evidence and weight audit."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys

from _bootstrap import ROOT
from audit_runtime_probe import require, digest, read, check_snapshot, protected_sources


def audit(directory):
    directory = Path(directory)
    pre = read(directory / "preregistration.json")
    report = read(directory / "report.json")
    require(report["status"] == "PASS" and not (directory / "failure.json").exists(), "successful GPU execution")
    schedule = [[102200+i,2+i%13,i%4,271828+i] for i in range(16)]
    require(pre["deals"] == schedule and pre["hands"] == 16, "fixed development budget")
    require(pre["device"] == "cuda" and pre["model_seed"] == 271828, "GPU model configuration")
    require(pre["batch_size"] == 64 and pre["chunk_size"] == 256 and pre["high_branch_repeats"] == 20, "batch/candidate/repeat configuration")
    require(pre["epsilon"] == .1 and pre["learning_rate"] == .001, "learning configuration")
    hashes = pre["source_sha256_before"]
    require(hashes == report["source_sha256_before"] == report["source_sha256_after"] and report["source_unchanged"], "source identity")
    check_snapshot(directory, hashes)
    require(digest(directory / "source-snapshot.zip") == report["source_snapshot_sha256"], "ZIP identity")
    protected = protected_sources()
    runtime = report["runtime"]
    require(runtime["torch"] == "2.12.0+cu130" and runtime["cuda"]["torch_cuda"] == "13.0", "tested GPU runtime")
    require(runtime["cuda"]["available"] and runtime["cuda"]["capability"] == [12,0], "GPU hardware")
    require(runtime["torch_threads"] == 1 and runtime["deterministic_algorithms"], "deterministic mode")
    require(not runtime["tf32_matmul"] and not runtime["tf32_cudnn"], "TF32 disabled")
    high = report["results"]["high_branch"]
    require(high["candidates"] == len(high["scores"]) == 8769 and high["score_device"] == "cuda", "full CUDA candidate scoring")
    require(all(math.isfinite(x) and -1 <= x <= 1 for x in high["scores"]), "bounded finite scores")
    chunks = [256]*34 + [65]
    require(high["coverage"]["chunk_lengths"] == chunks and high["coverage"]["scored_candidates"] == 8769 and high["coverage"]["max_chunk"] == 256, "complete chunk coverage")
    from guandan.env import HandEnv
    from guandan.rules.cards import card_id
    hand = tuple(card_id(r,s,c) for r in (2,3,4) for s in range(4) for c in range(2))+(18,72,7)
    rest = [c for c in range(108) if c not in hand]
    env = HandEnv.from_hands((hand,tuple(rest[:27]),tuple(rest[27:54]),tuple(rest[54:])),7)
    actions = env.legal_actions(0)
    raw = json.dumps([a.to_dict() for a in actions],sort_keys=True,separators=(",", ":")).encode()
    require(hashlib.sha256(raw).hexdigest() == high["candidate_sha256"], "independent candidate regeneration")
    latency = high["latency"]
    values = sorted(latency["raw_ms"])
    require(len(values) == latency["count"] == 20 and all(math.isfinite(x) and x >= 0 for x in values), "raw latency budget")
    for key,q in (("p50_ms",.5),("p95_ms",.95)):
        index = q*(len(values)-1)
        lo,hi = math.floor(index),math.ceil(index)
        expected = values[lo]*(1-index+lo)+values[hi]*(index-lo)
        require(math.isclose(latency[key],expected,rel_tol=1e-12), "independent "+key)
    require(latency["min_ms"] == min(values) and latency["max_ms"] == max(values), "latency extrema")
    learning = report["results"]["learning"]
    rows = learning["hands"]
    require([[r["seed"],r["level"],r["starting_player"],r["policy_seed"]] for r in rows] == schedule, "actual hand schedule")
    gradient_calls = 0
    for row in rows:
        require(row["steps"] == row["samples"] and 0 < row["samples"] <= 1000, "complete trajectories")
        require(row["replay_verified"] and row["team_rewards"] in ([1,-1],[-1,1]), "terminal replay/rewards")
        require(len(row["terminal_digest"]) == len(row["action_digest"]) == 64, "trajectory identities")
        require(row["updates"] == math.ceil(row["samples"]/64) == len(row["losses"]) == len(row["cuda_proofs"]), "update budget")
        require(all(math.isfinite(x) and x >= 0 for x in row["losses"]), "finite losses")
        scoring = row["score_cuda_proof"]
        require(scoring["all_score_features_cuda"] and scoring["forward_hook_calls"] > 0 and scoring["scored_candidates"] >= scoring["forward_hook_calls"] and 0 < scoring["max_chunk"] <= 256, "CUDA policy scoring hooks")
        for proof in row["cuda_proofs"]:
            require(proof["forward_hook_calls"] == 1 and proof["gradient_hook_calls"] == 7, "CUDA hook counts")
            require(proof["all_forward_cuda"] and proof["all_gradients_cuda"], "CUDA forward/backward")
            require(proof["parameters_changed"], "actual optimizer update")
            device_proof = proof["batch_device_proof"]
            require(all(v == "cuda" for k,v in device_proof.items() if k.endswith("_device")) and device_proof["loss_finite"], "CUDA batch and loss")
            gradient_calls += proof["gradient_hook_calls"]
        components = sum(row[k] for k in ("enumeration_ms","score_ms","update_ms","replay_ms","other_ms"))
        require(math.isclose(components,row["total_ms"],rel_tol=1e-9,abs_tol=1e-6), "episode wall time accounting")
        require(all(math.isfinite(v) and v >= 0 for k,v in row.items() if k.endswith("_ms")), "nonnegative timings")
    require(learning["losses"] == [x for r in rows for x in r["losses"]], "all loss history")
    require(learning["samples"] == sum(r["samples"] for r in rows) and learning["updates"] == sum(r["updates"] for r in rows), "training totals")
    for key,total in learning["component_ms"].items():
        require(math.isclose(total,sum(r[key] for r in rows),rel_tol=1e-9), "component total "+key)
    require(math.isclose(learning["total_ms"],sum(r["total_ms"] for r in rows)+learning["loop_overhead_ms"],rel_tol=1e-9), "whole loop timing")
    require(learning["loop_overhead_ms"] >= 0, "loop overhead")
    weights = report["runtime_weights"]
    require(weights["path"] == "runtime-weights.pt", "weight file identity")
    path = directory / weights["path"]
    require(digest(path) == weights["sha256"] and path.stat().st_size == weights["bytes"], "weight file SHA256")
    import torch
    state = torch.load(path,map_location="cpu",weights_only=True)
    computed = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "finite weight values")
        computed.update(key.encode())
        computed.update(str(tuple(value.shape)).encode())
        computed.update(str(value.dtype).encode())
        data = array('f',value.flatten().tolist())
        if sys.byteorder != "little": data.byteswap()
        computed.update(data.tobytes())
    require(computed.hexdigest() == learning["final_model_state_sha256"] != learning["initial_model_state_sha256"], "persisted changed model state")
    return dict(status="PASS",gpu_learning_hands=16,samples=learning["samples"],updates=learning["updates"],
                verified_cuda_gradient_hooks=gradient_calls,complete_candidates=8769,
                registered_sources=len(hashes),protected_sources=protected,
                runtime_weights_sha256=weights["sha256"],report_sha256=digest(directory/"report.json"),
                audit_source_sha256={name:digest(ROOT/name) for name in ("scripts/audit_gpu_probe.py","scripts/audit_runtime_probe.py")},
                promotion="NOT_ELIGIBLE",resume_verified=False,reserved_test_executed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    text = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        with args.output.open('x',encoding='utf-8') as file: file.write(text+'\n')
    print(text)
