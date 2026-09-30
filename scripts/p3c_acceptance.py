"""Fixed GPU engineering experiment; fresh-process deterministic recovery proof."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from _bootstrap import ROOT
import torch
import benchmark_learning_runtime as old
from guandan.env import HandEnv
from guandan_gpu.training import GPUTrainer, score_many
from guandan_gpu.checkpoint import capture, load_checkpoint, save_checkpoint
from train_gpu import CONFIG, register, write_json

DEALS = [(103000+i, 2+i%13, i%4) for i in range(16)]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def high_probe(trainer):
    obs, actions = old.high_branch_fixture()
    env = HandEnv()
    other = env.reset(103090, initial_level=9, starting_player=2)
    requests = [(obs, actions), (other, env.legal_actions(2))]
    counts, devices = [], []
    def hook(_module, args, _output):
        counts.append(len(args[0])); devices.append(args[0].device.type)
    h = trainer.model.action_fc.register_forward_hook(hook)
    try:
        values = score_many(trainer.model, requests, CONFIG["chunk_size"])
    finally:
        h.remove()
    refs = [old.score_complete(trainer.model, ob, ac, "cuda") for ob, ac in requests]
    if sum(counts) != sum(len(ac) for _, ac in requests) or set(devices) != {"cuda"}:
        raise AssertionError("candidate coverage/device")
    if not all(torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(values, refs)):
        raise AssertionError("batched reference score mismatch")
    raw_old, raw_new = [], []
    for i in range(5):
        methods = ("old", "new") if i%2 == 0 else ("new", "old")
        for method in methods:
            fn = (lambda: old.score_complete(trainer.model, obs, actions, "cuda")) if method == "old" else (
                lambda: score_many(trainer.model, [(obs, actions)], CONFIG["chunk_size"])[0])
            scores, ms = old.timed("cuda", fn)
            if not torch.allclose(scores, refs[0], atol=1e-5, rtol=1e-4):
                raise AssertionError("repeated score drift")
            (raw_old if method == "old" else raw_new).append(ms)
    return dict(candidates=[len(ac) for _, ac in requests], chunk_lengths=counts,
                devices=devices, max_abs_diff=[float((a-b).abs().max()) for a,b in zip(values,refs)],
                scores=[v.tolist() for v in values], reference_scores=[v.tolist() for v in refs],
                old_latency=old.stats(raw_old), new_latency=old.stats(raw_new),
                scope="encoding+upload+CUDA scoring+CPU return; five interleaved repeats")


def negative_checks(output):
    base = output / "wave-2"
    original = torch.load(base / "checkpoint.pt", weights_only=True)
    receipt = json.loads((base / "manifest.json").read_text())
    negatives = output / "negative-cases"
    negatives.mkdir()
    result = {}
    def must_reject(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError, FileExistsError, FileNotFoundError, KeyError, TypeError) as e:
            result[name] = dict(rejected=True, exception=type(e).__name__, message=str(e))
        else:
            raise AssertionError(f"accepted invalid case {name}")
    mutations = {
        "version": lambda p: p["versions"].update(training="wrong"),
        "counter": lambda p: p.update(waves=p["waves"]+1),
        "seed": lambda p: p["used_deal_seeds"].__setitem__(0, 9000000),
        "duplicate_seed": lambda p: p["used_deal_seeds"].__setitem__(0, p["used_deal_seeds"][1]),
        "adam": lambda p: p["optimizer"]["state"][0].update(step=torch.tensor(-1.)),
        "cuda_rng": lambda p: p.update(cuda_rng=[torch.zeros(2, dtype=torch.uint8)]),
        "nonfinite": lambda p: next(iter(p["model"].values())).fill_(float("nan")),
    }
    for name, change in mutations.items():
        payload, manifest = deepcopy(original), deepcopy(receipt)
        change(payload)
        for key in list(manifest):
            if key in payload: manifest[key] = payload[key]
        target = negatives / name
        target.mkdir()
        torch.save(payload, target / "checkpoint.pt")
        data = (target / "checkpoint.pt").read_bytes()
        manifest.update(sha256=sha256(data).hexdigest(), bytes=len(data))
        write_json(target / "manifest.json", manifest)
        before = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        must_reject(name, lambda: load_checkpoint(target))
        if not equal(before, (torch.get_rng_state(), torch.cuda.get_rng_state_all())):
            raise AssertionError("rejected checkpoint changed RNG")
    target = negatives / "truncated"; target.mkdir()
    (target / "checkpoint.pt").write_bytes((base / "checkpoint.pt").read_bytes()[:128])
    write_json(target / "manifest.json", receipt)
    must_reject("truncated", lambda: load_checkpoint(target))
    trainer = load_checkpoint(base)
    before_hash = sha256((base / "checkpoint.pt").read_bytes()).hexdigest()
    must_reject("overwrite", lambda: save_checkpoint(trainer, base))
    assert before_hash == sha256((base / "checkpoint.pt").read_bytes()).hexdigest()
    with patch("guandan_gpu.checkpoint.os.rename", side_effect=RuntimeError("injected before atomic commit")):
        must_reject("commit_interruption", lambda: save_checkpoint(trainer, negatives / "not-committed"))
    assert not (negatives / "not-committed").exists()
    stage = next(negatives.glob("not-committed.incomplete-*"))
    must_reject("incomplete_load", lambda: load_checkpoint(stage))
    with patch("guandan_gpu.training.HandEnv.step", side_effect=RuntimeError("injected sampling failure")):
        must_reject("sample_failure", lambda: trainer.train_wave(DEALS[8:12]))
    assert trainer.ready_for_checkpoint is False
    must_reject("failed_wave_save", lambda: save_checkpoint(trainer, negatives / "failed-wave"))
    must_reject("failed_wave_continue", lambda: trainer.train_wave(DEALS[8:12]))
    recovered = load_checkpoint(base)
    replay = recovered.train_wave(DEALS[8:12])
    return result, replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists(): p.error("output already exists")
    output.mkdir(parents=True)
    pre = register(output, CONFIG, DEALS, scope="GPU engineering acceptance; no strength claim")
    started = time.perf_counter()
    rows = []
    try:
        trainer = GPUTrainer(CONFIG)
        initial = capture(trainer)
        high = high_probe(trainer)
        write_json(output / "high-branch.json", high)
        times = []
        for i in range(4):
            torch.cuda.synchronize(); start = time.perf_counter()
            row = trainer.train_wave(DEALS[4*i:4*i+4])
            torch.cuda.synchronize(); times.append(time.perf_counter()-start)
            rows.append(row)
            save_checkpoint(trainer, output / f"wave-{i+1}")
            with (output / "training.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
            print(json.dumps(dict(stage="training", wave=i+1, episodes=trainer.episodes,
                                  samples=row["samples"], elapsed_s=time.perf_counter()-started)), flush=True)
        final = capture(trainer)
        assert not equal(initial["model"], final["model"])
        child = subprocess.run([sys.executable, str(ROOT / "scripts/train_gpu.py"),
            "--resume", str(output / "wave-2"), "--waves", "2", "--seed-start", "103008",
            "--output", str(output / "fresh-process-resume")], capture_output=True, text=True, timeout=600)
        (output / "resume-process.log").write_text(child.stdout + child.stderr, encoding="utf-8")
        if child.returncode: raise RuntimeError(f"resume process failed {child.returncode}: {child.stderr}")
        rerows = [json.loads(line) for line in (output / "fresh-process-resume/training.jsonl").read_text().splitlines()]
        restored = load_checkpoint(output / "fresh-process-resume/wave-4")
        compare = capture(restored)
        if not equal(final, compare) or rows[2:] != rerows:
            raise AssertionError("fresh-process recovery not bitwise equal")
        write_json(output / "resume-audit.json", dict(status="PASS", fresh_process=True,
                   model_optimizer_python_cpu_cuda_rng_counters_bitwise_equal=True, replayed_rows=rerows))
        print(json.dumps(dict(stage="resume", status="PASS")), flush=True)
        negative, replay = negative_checks(output)
        assert rows[2] == replay
        write_json(output / "negative-audit.json", dict(status="PASS", cases=negative,
                   recovery_after_sampling_failure_equal=True, replayed_row=replay))
        current = {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in pre["source_sha256"]}
        assert current == pre["source_sha256"]
        manifest = json.loads((output / "wave-4/manifest.json").read_text())
        write_json(output / "report.json", dict(status="PASS", hands=16, waves=4,
                   samples=sum(r["samples"] for r in rows), updates=trainer.updates,
                   wave_seconds=times, training_seconds=sum(times), elapsed_s=time.perf_counter()-started,
                   checkpoint_sha256=manifest["sha256"], source_unchanged=True,
                   promotion="NOT_ELIGIBLE", reserved_test_executed=False))
        print((output / "report.json").read_text(), flush=True)
    except BaseException:
        write_json(output / "failure.json", dict(status="FAIL", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
