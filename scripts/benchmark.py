"""Headless invariant/replay stress gate, not a policy-strength evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

from _bootstrap import ROOT
from guandan.agents.baselines import make_agent
from guandan.env import HandEnv
from guandan.types import ACTION_VERSION, OBSERVATION_VERSION, RULES_VERSION


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)] if ordered else None


def run_batch(indices, start_seed, policy_seed, replay_every):
    times, decisions, counts, step_counts, failures = [], [], [], [], []
    wins = [0, 0]
    replays = 0
    successes = 0
    for i in indices:
        env = HandEnv()
        player = i % 4
        seed = start_seed + i
        try:
            env.reset(seed, initial_level=2 + i % 13, starting_player=player)
            # Rotating mixtures exercise pass and cooperative behavior. Not fair matchups.
            agents = [make_agent(("random", "greedy", "team")[(i + p) % 3], policy_seed + 4*i + p) for p in range(4)]
            for step in range(1000):
                observation = env.observe(player)
                start = time.perf_counter_ns()
                legal = env.legal_actions(player)
                times.append((time.perf_counter_ns() - start) / 1e6)
                counts.append(len(legal))
                action = agents[player].act(observation, legal)
                decisions.append((time.perf_counter_ns() - start) / 1e6)
                result = env.step(player, action, state_version=observation.state_version)
                env.assert_invariants()
                if result.terminal:
                    break
                player = result.next_player
            else:
                raise RuntimeError("1000-step guard exceeded")
            if i % replay_every == 0:
                replayed = HandEnv.replay(env.serialize_replay())
                if replayed.state_digest() != env.state_digest():
                    raise AssertionError("Replay terminal digest mismatch")
                replays += 1
            successes += 1
            wins[result.settlement.winner_team] += 1
            step_counts.append(step + 1)
        except Exception:
            failures.append({"seed": seed, "level": 2 + i % 13, "starting_player": i % 4,
                             "error": traceback.format_exc()})
    return {"successes": successes, "replays": replays, "failures": failures,
            "wins": wins, "enumeration_ms": times, "decision_ms": decisions, "action_counts": counts, "steps": step_counts}


def source_manifest():
    files = sorted([*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py"),
                    ROOT / "docs/rules.md", ROOT / "configs/rules.json"])
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--policy-seed", type=int, default=700000)
    parser.add_argument("--replay-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluations/environment-benchmark.json")
    args = parser.parse_args()
    if min(args.hands, args.workers, args.replay_every, args.batch_size) < 1:
        parser.error("hands/workers/replay-every/batch-size must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing receipt: {args.output}")
    manifest = source_manifest()
    start = time.perf_counter()
    aggregate = {"successes": 0, "replays": 0, "failures": [], "wins": [0, 0],
                 "enumeration_ms": [], "decision_ms": [], "action_counts": [], "steps": []}
    log_path = args.output.with_suffix(".progress.jsonl")
    with ProcessPoolExecutor(max_workers=args.workers) as executor, log_path.open("w", encoding="utf-8") as log:
        futures = [executor.submit(run_batch, list(range(i, min(i + args.batch_size, args.hands))),
                                   args.start_seed, args.policy_seed, args.replay_every)
                   for i in range(0, args.hands, args.batch_size)]
        for future in as_completed(futures):
            batch = future.result()
            for key in ("successes", "replays"):
                aggregate[key] += batch[key]
            for key in ("failures", "enumeration_ms", "decision_ms", "action_counts", "steps"):
                aggregate[key].extend(batch[key])
            aggregate["wins"] = [x + y for x, y in zip(aggregate["wins"], batch["wins"])]
            progress = {"completed": aggregate["successes"], "failed": len(aggregate["failures"]),
                        "elapsed_seconds": round(time.perf_counter() - start, 3)}
            line = json.dumps(progress)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
    elapsed = time.perf_counter() - start
    source_unchanged = manifest == source_manifest()
    report = {"kind": "environment_stress_gate", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS" if aggregate["successes"] == args.hands and not aggregate["failures"] and source_unchanged else "FAIL",
              "scope": "Environment stability; no policy superiority or learned model claim",
              "python": sys.version, "platform": platform.platform(),
              "rules_version": RULES_VERSION, "action_version": ACTION_VERSION, "observation_version": OBSERVATION_VERSION,
              "baseline_versions": [make_agent(n).version for n in ("random", "greedy", "team")],
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "seed_list": list(range(args.start_seed, args.start_seed + args.hands)),
              "successful_hands": aggregate["successes"], "replays_verified": aggregate["replays"],
              "failures": aggregate["failures"], "elapsed_seconds": elapsed,
              "hands_per_hour": aggregate["successes"] / elapsed * 3600,
              "decisions": len(aggregate["action_counts"]),
              "enumeration_ms": {"p50": percentile(aggregate["enumeration_ms"], .5), "p95": percentile(aggregate["enumeration_ms"], .95), "max": max(aggregate["enumeration_ms"], default=0)},
              "decision_ms": {"p50": percentile(aggregate["decision_ms"], .5), "p95": percentile(aggregate["decision_ms"], .95), "max": max(aggregate["decision_ms"], default=0)},
              "action_counts": {"p50": percentile(aggregate["action_counts"], .5), "p95": percentile(aggregate["action_counts"], .95), "max": max(aggregate["action_counts"], default=0)},
              "max_steps": max(aggregate["steps"], default=0), "source_unchanged": source_unchanged,
              "source_sha256": manifest,
              "reproduce": f'python scripts/benchmark.py --hands {args.hands} --workers {args.workers} --start-seed {args.start_seed} --policy-seed {args.policy_seed} --replay-every {args.replay_every} --batch-size {args.batch_size} --output artifacts/evaluations/rerun.json'}
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in ("seed_list", "source_sha256")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
