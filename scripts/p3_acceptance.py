"""Controller's fixed P3a experiment with exact continuation verification."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import zipfile

from _bootstrap import ROOT
import torch
from guandan.learning.checkpoint import load_checkpoint, save_checkpoint, runtime
from guandan.learning.training import Trainer
from guandan.learning.evaluation import smoke_schedule, evaluate_trial, summarize


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def source_hashes():
    files = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"),
             *ROOT.glob("tests/test_*.py"), ROOT / "docs/P3_PROTOCOL.md"]
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def exact_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact_equal(x, y) for x, y in zip(a, b))
    return a == b


def trainer_state(trainer):
    return deepcopy(dict(model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                         rng=trainer.rng.getstate(), torch_rng=torch.get_rng_state(),
                         episodes=trainer.episodes, updates=trainer.updates,
                         used_deal_seeds=trainer.used_deal_seeds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    old = json.loads((ROOT / "artifacts/evaluations/p2-validation-v1/preregistration.json").read_text(encoding="utf-8"))["source_sha256"]
    mismatches = [p for p, digest in old.items() if sha256((ROOT/p).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError(f"P2 protected sources changed: {mismatches}")
    config = dict(seed=314159, epsilon=.1, lr=.001, batch_size=64, chunk_size=256)
    write_json(output / "preregistration.json", dict(config=config, runtime=runtime(),
               training_seeds=list(range(100500, 100504)), evaluation_seeds=[101000, 101001],
               expected_games=48, primary_opponent="greedy", split="development",
               promotion="NOT_ELIGIBLE", reserved_test_executed=False, source_sha256=sources))
    with zipfile.ZipFile(output / "source-snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path, digest in sources.items():
            data = (ROOT/path).read_bytes()
            if sha256(data).hexdigest() != digest:
                raise RuntimeError("Source changed during snapshot")
            z.writestr(path, data)
    started = perf_counter()
    trainer = Trainer(config)
    initial = deepcopy(trainer.model.state_dict())
    training_rows = []
    with (output / "training.jsonl").open("x", encoding="utf-8") as f:
        for i in range(2):
            row = trainer.train_episode(100500+i, 2+i, i%4)
            training_rows.append(row)
            f.write(json.dumps(row, allow_nan=False)+"\n"); f.flush()
            print(f"training {i+1}/4: {row}", flush=True)
        middle = save_checkpoint(trainer, output / "episode-2")
        continuous = []
        for i in range(2, 4):
            row = trainer.train_episode(100500+i, 2+i, i%4)
            training_rows.append(row); continuous.append(row)
            f.write(json.dumps(row, allow_nan=False)+"\n"); f.flush()
            print(f"training {i+1}/4: {row}", flush=True)
    expected = trainer_state(trainer)
    if exact_equal(initial, expected["model"]):
        raise RuntimeError("Training did not update parameters")
    restored = load_checkpoint(output / "episode-2", middle["sha256"])
    replayed = [restored.train_episode(100500+i, 2+i, i%4) for i in range(2, 4)]
    if not exact_equal(expected, trainer_state(restored)) or continuous != replayed:
        raise RuntimeError("Continuous and restored training diverged")
    write_json(output / "resume-audit.json", dict(status="PASS", episodes_replayed=2,
                model_optimizer_rng_counters_bitwise_equal=True, trajectory_rows_equal=True,
                parameters_changed=True, replayed_rows=replayed))
    final = save_checkpoint(restored, output / "candidate")
    training_seconds = perf_counter()-started
    # Evaluate a fresh disk load, bound to the candidate hash before any outcomes.
    candidate = load_checkpoint(output / "candidate", final["sha256"])
    trials = smoke_schedule()
    write_json(output / "evaluation-preregistration.json", dict(checkpoint_sha256=final["sha256"],
                expected_games=len(trials), schedule=[asdict(t) for t in trials],
                epsilon=0, primary_opponent="greedy", promotion="NOT_ELIGIBLE"))
    eval_started = perf_counter()
    rows = []
    with (output / "evaluation.jsonl").open("x", encoding="utf-8") as f:
        for i, trial in enumerate(trials):
            row = evaluate_trial(candidate.model, trial)
            rows.append(row)
            f.write(json.dumps(row, allow_nan=False)+"\n"); f.flush()
            if (i+1)%8 == 0:
                print(f"evaluation {i+1}/{len(trials)} complete", flush=True)
    if source_hashes() != sources:
        raise RuntimeError("Experiment sources changed during execution")
    report = dict(status="CPU_LOOP_VERIFIED", p3_status="PARTIALLY_ACCEPTED",
                  runtime=runtime(), checkpoint_sha256=final["sha256"],
                  training_episodes=4, training_decisions=sum(r["steps"] for r in training_rows),
                  training_updates=restored.updates, training_and_resume_seconds=training_seconds,
                  evaluation_games=len(rows), evaluation_seconds=perf_counter()-eval_started,
                  results=summarize(rows), source_hashes_unchanged=True,
                  p2_protected_files_unchanged=len(old), promotion="NOT_ELIGIBLE",
                  reserved_test_executed=False, exact_resume=True)
    write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
