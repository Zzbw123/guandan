"""Reproducible headless baseline game; full replay is a research artifact."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from guandan.agents.baselines import make_agent
from guandan.env import HandEnv
from guandan.rules.cards import label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--agent", choices=("random", "greedy", "team"), default="team")
    parser.add_argument("--policy-seed", type=int, default=12000)
    parser.add_argument("--replay", type=Path, default=Path("artifacts/replays/hand-42.json"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    env = HandEnv()
    env.reset(args.seed, initial_level=args.level)
    agents = [make_agent(args.agent, args.policy_seed + p) for p in range(4)]
    player = 0
    for _ in range(1000):
        observation = env.observe(player)
        actions = env.legal_actions(player)
        action = agents[player].act(observation, actions)
        if args.verbose:
            print(f"P{player} {action.kind}: {' '.join(label(c) for c in action.cards)}")
        result = env.step(player, action, state_version=observation.state_version)
        if result.terminal:
            break
        player = result.next_player
    else:
        raise RuntimeError("Exceeded game length guard")
    replay = env.serialize_replay()
    recovered = HandEnv.replay(replay)
    if recovered.state_digest() != env.state_digest():
        raise RuntimeError("Replay mismatch")
    args.replay.parent.mkdir(parents=True, exist_ok=True)
    args.replay.write_text(json.dumps(replay, indent=2, ensure_ascii=False), encoding="utf-8")
    from dataclasses import asdict
    print(json.dumps({"settlement": asdict(result.settlement), "steps": result.state_version,
                      "replay_verified": True, "replay": str(args.replay.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
