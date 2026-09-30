"""Bounded development-only synchronous GPU training and wave-boundary resume."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from _bootstrap import ROOT
from guandan_gpu.training import GPUTrainer
from guandan_gpu.checkpoint import load_checkpoint, save_checkpoint, runtime

CONFIG = dict(seed=314260, epsilon=.1, lr=.001, batch_size=256, chunk_size=1024, num_envs=4)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def source_files():
    return sorted((ROOT / "src").rglob("*.py")) + [
        ROOT / name for name in ("scripts/train_gpu.py", "scripts/p3c_acceptance.py",
        "scripts/audit_gpu_training.py", "scripts/_bootstrap.py",
        "scripts/benchmark_learning_runtime.py", "docs/P3C_GPU_TRAINING_PROTOCOL.md",
        "requirements-learning-gpu.txt", "requirements-learning-gpu.lock")]


def register(output, config, deals, **extra):
    paths = source_files()
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    pre = dict(kind="gd-gpu-wave-preregistration-v1", utc=datetime.now(timezone.utc).isoformat(),
               config=config, deals=deals, source_sha256=hashes,
               reserved_test_executed=False, promotion="NOT_ELIGIBLE", **extra)
    write_json(output / "preregistration.json", pre)
    with zipfile.ZipFile(output / "source-snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT).as_posix())
    return pre


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--waves", type=int, default=4)
    p.add_argument("--seed-start", type=int, default=103000)
    p.add_argument("--resume", type=Path)
    p.add_argument("--expected-sha256")
    args = p.parse_args()
    if args.output.exists():
        p.error("output already exists")
    if not 1 <= args.waves <= 25:
        p.error("waves must be 1..25")
    trainer = GPUTrainer(CONFIG)
    if args.resume:
        trainer = load_checkpoint(args.resume, args.expected_sha256)
    count = args.waves * trainer.config["num_envs"]
    if args.seed_start < 100000 or args.seed_start + count > 110000:
        p.error("only development seeds 100000..109999")
    if set(range(args.seed_start, args.seed_start + count)) & set(trainer.used_deal_seeds):
        p.error("training seed already used")
    offset = trainer.episodes
    deals = [[args.seed_start+i, 2+(offset+i)%13, (offset+i)%4] for i in range(count)]
    args.output.mkdir(parents=True)
    register(args.output, trainer.config, deals,
             resume_sha256=None if not args.resume else sha256((args.resume / "checkpoint.pt").read_bytes()).hexdigest())
    start = time.perf_counter()
    try:
        for i in range(args.waves):
            n = trainer.config["num_envs"]
            row = trainer.train_wave([tuple(d) for d in deals[i*n:(i+1)*n]])
            receipt = save_checkpoint(trainer, args.output / f"wave-{trainer.waves}")
            with (args.output / "training.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(json.dumps(dict(wave=trainer.waves, episodes=trainer.episodes,
                                  updates=trainer.updates, elapsed_s=time.perf_counter()-start)), flush=True)
        write_json(args.output / "report.json", dict(status="PASS", runtime=runtime(),
                   episodes=trainer.episodes, updates=trainer.updates, waves=trainer.waves,
                   checkpoint_sha256=receipt["sha256"], elapsed_s=time.perf_counter()-start,
                   promotion="NOT_ELIGIBLE", reserved_test_executed=False))
    except BaseException:
        write_json(args.output / "failure.json", dict(status="FAIL", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
