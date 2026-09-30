"""Independent arithmetic/symmetry audit of P2 recorded outcomes.

Uses only stdlib and the written data, not the evaluator's summary/schedule code.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts/evaluations/p2-validation-v1"
    directory = directory.resolve()
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    prereg = json.loads((directory / "preregistration.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert report["status"] == "PASS"
    assert prereg["reserved_test_executed"] is False
    assert len(rows) == report["expected_games"]
    by_scenario = defaultdict(list)
    index = {}
    for row in rows:
        assert row["status"] == "ok" and row["illegal_actions"] == row["timeouts"] == 0
        key = (row["matchup_id"], row["deal_seed"], row["rotation"], row["swap"], row["flip"])
        assert key not in index
        index[key] = row
        by_scenario[row["matchup_id"]].append(row)
    complementary, role_flips = 0, 0
    for name, scenario in by_scenario.items():
        own, mate, opponent = name.split("|")
        clusters = defaultdict(list)
        for row in scenario:
            clusters[row["deal_seed"]].append(row)
            if own == mate:
                other_name = f"{opponent}|{opponent}|{own}"
                other_key = (other_name, row["deal_seed"], row["rotation"], 1 - row["swap"], 0)
                other = index[other_key]
                assert row["win"] + other["win"] == 1
                complementary += 1
            else:
                other_name = f"{mate}|{own}|{opponent}"
                other_key = (other_name, row["deal_seed"], row["rotation"], row["swap"], 1 - row["flip"])
                other = index[other_key]
                assert row["win"] == other["win"]
                role_flips += 1
            assert row["policies"] == other["policies"]
            assert row["finish_order"] == other["finish_order"]
            assert row["steps"] == other["steps"]
            assert row["terminal_digest"] == other["terminal_digest"]
        assert set(clusters) == set(prereg["seed_list"])
        summary = report["statistics"][name]
        assert summary["independent_deals"] == len(clusters)
        per_seed = {item["deal_seed"]: item for item in summary["clusters"]}
        expected_size = 8 if own == mate else 16
        for seed, group in clusters.items():
            assert len(group) == expected_size
            assert len({row["level"] for row in group}) == 1
            for metric, field in (("win_rate", "win"), ("team_reward", "team_reward"),
                                  ("focal_level_gain", "focal_level_gain"), ("opponent_level_gain", "opponent_level_gain")):
                direct = math.fsum(row[field] for row in group) / len(group)
                assert math.isclose(direct, per_seed[seed][metric], abs_tol=1e-12)
        for metric, field in (("win_rate", "win"), ("team_reward", "team_reward"),
                              ("focal_level_gain", "focal_level_gain"), ("opponent_level_gain", "opponent_level_gain")):
            direct = math.fsum(math.fsum(row[field] for row in group) / len(group) for group in clusters.values()) / len(clusters)
            assert math.isclose(direct, summary["metrics"][metric]["estimate"], abs_tol=1e-12)
        win_ci = summary["metrics"]["win_rate"]["ci95"]
        reward_ci = summary["metrics"]["team_reward"]["ci95"]
        assert win_ci is not None and reward_ci is not None
        for win, reward in zip(win_ci, reward_ci):
            assert math.isclose(2 * win - 1, reward, abs_tol=1e-12)
        if own == mate == opponent:
            assert summary["metrics"]["win_rate"]["estimate"] == .5
    receipt = {"status": "PASS", "scenarios": len(by_scenario), "rows": len(rows),
               "independent_deals_per_scenario": prereg["config"]["deal_count"],
               "complementary_pair_links_checked": complementary, "teammate_role_flip_links_checked": role_flips,
               "note": "Pair links count both directions; they are not additional samples",
               "four_metrics_independently_recomputed": True, "win_reward_ci_linear_relation_checked": True,
               "report_sha256": hashlib.sha256((directory / "report.json").read_bytes()).hexdigest(),
               "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = directory / "controller-arithmetic-audit.json"
    if output.exists():
        assert json.loads(output.read_text(encoding="utf-8")) == receipt
    else:
        output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
