"""Bounded CPU DMC training. Existing output/checkpoint directories are refused."""
import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from guandan.learning.checkpoint import load_checkpoint, save_checkpoint
from guandan.learning.training import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=100500)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100 or not 100000 <= args.seed_start < args.seed_start+args.episodes <= 110000:
        parser.error("Require 1..100 episodes entirely within development seeds")
    trainer = load_checkpoint(args.resume) if args.resume else Trainer(
        dict(seed=314159, epsilon=.1, lr=.001, batch_size=64, chunk_size=256))
    if set(range(args.seed_start, args.seed_start+args.episodes)) & set(trainer.used_deal_seeds):
        parser.error("Explicit seed range overlaps this checkpoint's previous training")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as f:
        for seed in range(args.seed_start, args.seed_start+args.episodes):
            row = trainer.train_episode(seed, 2+trainer.episodes%13, trainer.episodes%4)
            f.write(json.dumps(row, allow_nan=False)+"\n")
            f.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
    receipt = save_checkpoint(trainer, args.output / "checkpoint")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
