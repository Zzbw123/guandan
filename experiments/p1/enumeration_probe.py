"""Constructed high-branching hand; this is not a proven global worst case."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from guandan.agents import TeamHeuristicAgent
from guandan.env import HandEnv
from guandan.rules import validate_action
from guandan.rules.cards import card_id


def main():
    hand = [card_id(r, s, copy) for r in (2, 3, 4) for s in range(4) for copy in range(2)]
    hand += [card_id(7, 1), card_id(7, 1, 1), card_id(9)]
    rest = [c for c in range(108) if c not in hand]
    env = HandEnv.from_hands((hand, rest[:27], rest[27:54], rest[54:]), initial_level=7)
    observation = env.observe(0)
    agent = TeamHeuristicAgent()
    legal = env.legal_actions(0)
    assert all(validate_action(a, hand, 7) for a in legal)
    enum_ms, decision_ms = [], []
    for _ in range(30):
        start = time.perf_counter_ns()
        actions = env.legal_actions(0)
        enum_ms.append((time.perf_counter_ns() - start) / 1e6)
        selected = agent.act(observation, actions)
        assert selected in legal
        decision_ms.append((time.perf_counter_ns() - start) / 1e6)
    report = {
        "kind": "constructed_high_branching_probe", "not_proven_global_worst_case": True,
        "hand": hand, "level": 7, "repetitions": 30, "candidates": len(legal),
        "all_candidates_valid": True,
        "enumeration_ms": {"p50": sorted(enum_ms)[14], "p95": sorted(enum_ms)[28], "max": max(enum_ms)},
        "team_decision_ms": {"p50": sorted(decision_ms)[14], "p95": sorted(decision_ms)[28], "max": max(decision_ms)},
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*sorted((ROOT / "src").rglob("*.py")), Path(__file__)]},
        "reproduce": "python experiments/p1/enumeration_probe.py"
    }
    output = ROOT / "artifacts/evaluations/p1-high-branching.json"
    if output.exists():
        raise FileExistsError("Choose a new output path to preserve the original receipt")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
