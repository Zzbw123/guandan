"""Controller disk audit, independent of GPUTrainer's report calculations."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import zipfile

from _bootstrap import ROOT
import torch


def check(value, message):
    if not value:
        raise ValueError(message)


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b


def audit(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory/name).read_text(encoding="utf-8"))
    pre, report = read("preregistration.json"), read("report.json")
    rows = [json.loads(line) for line in (directory/"training.jsonl").read_text().splitlines()]
    expected = [[103000+i, 2+i%13, i%4] for i in range(16)]
    config = dict(seed=314260, epsilon=.1, lr=.001, batch_size=256, chunk_size=1024, num_envs=4)
    check(pre["deals"] == expected and pre["config"] == config, "preregistered schedule/config")
    check(report["status"] == "PASS" and len(rows) == 4, "complete run")
    total_samples = updates = 0
    for w, row in enumerate(rows, 1):
        check(row["wave"] == row["learning_version"] == w and row["behavior_version"] == w-1, "model sync version")
        hands = row["hands"]
        check(len(hands) == 4 and [[h[k] for k in ("seed","level","starting_player")] for h in hands] == expected[4*(w-1):4*w], "hand schedule")
        check(row["episodes"] == w*4, "episode counter")
        for h in hands:
            check(type(h["steps"]) is int and h["steps"] == h["samples"] and 0 < h["steps"] <= 1000
                  and h["team_rewards"] in ([1,-1],[-1,1]) and h["replay_verified"] is True
                  and len(h["terminal_digest"]) == 64, "hand integrity")
        samples = sum(h["samples"] for h in hands)
        local_updates = math.ceil(samples/config["batch_size"])
        updates += local_updates; total_samples += samples
        check(row["samples"] == samples and row["updates"] == updates and math.isfinite(row["mean_loss"]), "sample/update/loss")
        proof = row["device_proof"]
        check(proof["device"] == "cuda:0" and proof["dtype"] == "torch.float32"
              and proof["forward_checks"] == local_updates
              and proof["gradient_checks"] == 7*local_updates
              and proof["adam_tensor_checks"] == 21*local_updates, "CUDA proof counts")
        check(0 < row["score_requests"] <= samples and row["scored_candidates"] >= row["score_requests"]
              and row["score_batches"] >= math.ceil(row["scored_candidates"]/config["chunk_size"]), "scoring counts")
    check(report["samples"] == total_samples and report["updates"] == updates and report["hands"] == 16 and report["waves"] == 4, "report arithmetic")
    check(len(report["wave_seconds"]) == 4 and all(math.isfinite(t) and t > 0 for t in report["wave_seconds"])
          and report["training_seconds"] == sum(report["wave_seconds"]), "training timing")
    check(report["promotion"] == pre["promotion"] == "NOT_ELIGIBLE"
          and report["reserved_test_executed"] is pre["reserved_test_executed"] is False, "scope")
    high = read("high-branch.json")
    check(high["candidates"][0] == 8769 and len(high["candidates"]) == 2
          and sum(high["chunk_lengths"]) == sum(high["candidates"])
          and max(high["chunk_lengths"]) <= 1024 and set(high["devices"]) == {"cuda"}, "high candidate coverage")
    check(len(high["scores"]) == len(high["reference_scores"]) == 2, "score request count")
    for i, (a,b) in enumerate(zip(high["scores"],high["reference_scores"])):
        check(len(a) == len(b) == high["candidates"][i], "score lengths")
        check(all(math.isfinite(x) and math.isfinite(y) and abs(x-y) <= 1e-5+1e-4*abs(y) for x,y in zip(a,b)), "score reference tolerance")
        check(high["max_abs_diff"][i] == max(abs(x-y) for x,y in zip(a,b)), "score difference")
    for label in ("old_latency", "new_latency"):
        values = sorted(high[label]["raw_ms"])
        check(len(values) == high[label]["count"] == 5 and all(math.isfinite(v) and v>=0 for v in values), "raw timing")
        check(high[label]["p50_ms"] == values[2] and math.isclose(high[label]["p95_ms"], values[3]+.8*(values[4]-values[3])), "timing quantiles")
    resume = read("resume-audit.json")
    rerows = [json.loads(line) for line in (directory/"fresh-process-resume/training.jsonl").read_text().splitlines()]
    check(rows[2:] == rerows == resume["replayed_rows"] and resume["fresh_process"] is True, "fresh process trajectories")
    payloads = []
    for prefix in ("", "fresh-process-resume/"):
        for wave in ((1,2,3,4) if not prefix else (3,4)):
            manifest = read(f"{prefix}wave-{wave}/manifest.json")
            path = directory/f"{prefix}wave-{wave}/checkpoint.pt"
            data = path.read_bytes()
            check(sha256(data).hexdigest() == manifest["sha256"] and len(data) == manifest["bytes"], "checkpoint hash")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for key in ("versions","runtime","sources","boundary","config","episodes","updates","waves","used_deal_seeds"):
                check(payload[key] == manifest[key], f"checkpoint metadata {key}")
            check(payload["waves"] == wave and payload["episodes"] == 4*wave and payload["updates"] == rows[wave-1]["updates"]
                  and payload["used_deal_seeds"] == list(range(103000,103000+4*wave)), "checkpoint history")
            check(set(payload["optimizer"]["state"]) == set(range(7)), "Adam completeness")
            check(all(v["step"].item() == payload["updates"] for v in payload["optimizer"]["state"].values()), "Adam steps")
            check(all(torch.isfinite(v).all() for v in payload["model"].values()), "finite model")
            if wave == 4: payloads.append(payload)
            if not prefix and wave == 4: check(manifest["sha256"] == report["checkpoint_sha256"], "final hash")
    check(same(*payloads), "independent bitwise model/Adam/Python/CPU/CUDA RNG/counter comparison")
    negative = read("negative-audit.json")
    cases = {"version","counter","seed","duplicate_seed","adam","cuda_rng","nonfinite","truncated",
             "overwrite","commit_interruption","incomplete_load","sample_failure","failed_wave_save","failed_wave_continue"}
    check(set(negative["cases"]) == cases and all(c["rejected"] is True for c in negative["cases"].values()), "negative coverage")
    check(negative["replayed_row"] == rows[2] and negative["recovery_after_sampling_failure_equal"] is True, "failure recovery")
    registered = pre["source_sha256"]
    with zipfile.ZipFile(directory/"source-snapshot.zip") as z:
        check(set(z.namelist()) == set(registered), "source snapshot names")
        for name, digest in registered.items():
            check(sha256((ROOT/name).read_bytes()).hexdigest() == digest == sha256(z.read(name)).hexdigest(), f"source {name}")
    protected = {}
    for name, relative, field in (("p2","p2-validation-v1","source_sha256"), ("p3a","p3a-cpu-v1","source_sha256"), ("p3b","p3b-gpu-v1","source_sha256_before")):
        historical = json.loads((ROOT/f"artifacts/evaluations/{relative}/preregistration.json").read_text())[field]
        for path, digest in historical.items():
            check(sha256((ROOT/path).read_bytes()).hexdigest() == digest, f"protected {name}: {path}")
        protected[name] = len(historical)
    return dict(status="PASS", hands=16, samples=total_samples, updates=updates,
                bitwise_fresh_process_recovery=True, negative_cases=len(cases),
                high_candidates=high["candidates"], protected_sources=protected,
                registered_sources=len(registered), checkpoint_sha256=report["checkpoint_sha256"],
                report_sha256=sha256((directory/"report.json").read_bytes()).hexdigest(),
                auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), promotion="NOT_ELIGIBLE")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.directory)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as f: f.write(text+"\n")
    print(text)
