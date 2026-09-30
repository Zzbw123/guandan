"""Independent checks for deal-level P2 statistics and completeness gates."""

import math
import unittest

from guandan.evaluation.contracts import Trial
from guandan.evaluation.statistics import cluster_bootstrap, summarize_matchup


def make_scenario(wins_per_deal=(0, 2, 6, 8), levels=(2, 2, 3, 3), hetero=False):
    trials = []
    rows = []
    for deal_index, (wins, level) in enumerate(zip(wins_per_deal, levels, strict=True)):
        for rotation in range(4):
            for swap in range(2):
                for flip in range(2 if hetero else 1):
                    focal = (rotation + swap + 2 * flip) % 4
                    if hetero:
                        policies = ["random"] * 4
                        policies[focal] = "team"
                        policies[(focal + 2) % 4] = "greedy"
                        matchup_id = "team|greedy|random"
                    else:
                        policies = ["team" if seat % 2 == focal % 2 else "greedy" for seat in range(4)]
                        matchup_id = "team|team|greedy"
                    trial = Trial(matchup_id, deal_index, 200000 + deal_index,
                                  level, rotation, swap, flip, focal, tuple(policies))
                    trials.append(trial)
                    win = rotation * 2 + swap < wins
                    winner = focal if win else (focal + 1) % 4
                    rows.append({
                        "trial_id": trial.trial_id, "matchup_id": trial.matchup_id,
                        "deal_index": trial.deal_index, "deal_seed": trial.deal_seed,
                        "level": trial.level, "rotation": trial.rotation,
                        "swap": trial.swap, "flip": trial.flip,
                        "focal_seat": trial.focal_seat, "focal_team": trial.focal_team,
                        "policies": list(trial.policies), "starting_player": 0,
                        "status": "ok", "win": int(win),
                        "team_reward": 1 if win else -1,
                        "focal_level_gain": 3 if win else 0,
                        "opponent_level_gain": 0 if win else 3,
                        "finish_order": [winner, (winner + 2) % 4],
                        "terminal_digest": "a" * 64, "steps": 100,
                        "illegal_actions": 0, "timeouts": 0, "error": None,
                    })
    return trials, rows


class ClusterBootstrapTests(unittest.TestCase):
    def test_stratified_determinism_and_extreme_degeneracy(self):
        args = ([0.0, 0.25, 0.75, 1.0], [2, 2, 3, 3], 5000, 7)
        first = cluster_bootstrap(*args)
        self.assertEqual(first, cluster_bootstrap(*args))
        self.assertEqual(first["estimate"], 0.5)
        self.assertTrue(0 <= first["ci95"][0] <= 0.5 <= first["ci95"][1] <= 1)
        constant = cluster_bootstrap([1, 1, 1, 1], [2, 2, 3, 3], 5000, 7)
        self.assertEqual(constant["ci95"], [1.0, 1.0])
        self.assertTrue(constant["degenerate"])
        self.assertEqual(constant["reason"], "bootstrap_distribution_degenerate")
        fixed_strata = cluster_bootstrap([0, 0, 1, 1], [2, 2, 3, 3], 100, 7)
        self.assertEqual(fixed_strata["ci95"], [0.5, 0.5])
        self.assertTrue(fixed_strata["degenerate"])

    def test_singleton_stratum_suppresses_interval(self):
        value = cluster_bootstrap([0.0, 1.0, 0.5], [2, 2, 3], 100, 1)
        self.assertIsNone(value["ci95"])
        self.assertEqual(value["reason"], "fewer_than_two_clusters_in_stratum")

    def test_invalid_numeric_inputs_are_rejected(self):
        invalids = [True, float("nan"), float("inf")]
        for invalid in invalids:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                cluster_bootstrap([0.0, invalid], [2, 2], 100, 1)
        with self.assertRaises(ValueError):
            cluster_bootstrap([0, 1], [2, True], 100, 1)
        with self.assertRaises(ValueError):
            cluster_bootstrap([0, 1], [2, 2], True, 1)


class MatchupSummaryTests(unittest.TestCase):
    def test_hand_computed_cluster_summary_and_valid_n(self):
        trials, rows = make_scenario()
        summary = summarize_matchup(rows, trials, 5000, 19)
        self.assertEqual(summary["games"], 32)
        self.assertEqual(summary["independent_deals"], 4)
        self.assertEqual(summary["strata_counts"], {2: 2, 3: 2})
        self.assertEqual([c["win_rate"] for c in summary["clusters"]], [0, 0.25, 0.75, 1])
        self.assertEqual(summary["metrics"]["win_rate"]["estimate"], 0.5)
        self.assertEqual(summary["metrics"]["team_reward"]["estimate"], 0)
        self.assertEqual(summary["metrics"]["focal_level_gain"]["estimate"], 1.5)
        self.assertEqual(summary["metrics"]["opponent_level_gain"]["estimate"], 1.5)
        self.assertEqual(summary["effect_vs_50pp"], 0)
        quality = summary["quality"]["cluster_win_rate"]
        self.assertEqual(quality["median"], 0.5)
        self.assertEqual(quality["min"], 0)
        self.assertEqual(quality["max"], 1)
        self.assertEqual(quality["distribution"], {0: 1, 0.25: 1, 0.75: 1, 1: 1})
        self.assertTrue(math.isclose(quality["sd"], math.sqrt(0.625 / 3)))
        self.assertEqual(summary, summarize_matchup(list(reversed(rows)), trials, 5000, 19))

    def test_duplicate_missing_and_failed_games_cannot_be_dropped(self):
        trials, rows = make_scenario()
        variants = [rows[:-1], rows + [rows[0]],
                    [{**rows[0], "status": "error"}, *rows[1:]],
                    [{**rows[0], "timeouts": 1}, *rows[1:]],
                    [{**rows[0], "illegal_actions": 1}, *rows[1:]]]
        for variant in variants:
            with self.subTest(length=len(variant), row=variant[0]), self.assertRaises(ValueError):
                summarize_matchup(variant, trials, 10, 1)

    def test_metadata_result_types_and_settlement_are_strict(self):
        trials, rows = make_scenario()
        changes = [
            {"deal_index": True}, {"policies": tuple(rows[0]["policies"])},
            {"policies": [True, *rows[0]["policies"][1:]]},
            {"win": True}, {"win": float("nan")},
            {"team_reward": True}, {"steps": True},
            {"steps": 1001}, {"focal_level_gain": float("nan")},
            {"finish_order": [rows[0]["finish_order"][0], rows[0]["finish_order"][0]]},
            {"finish_order": tuple(rows[0]["finish_order"])},
        ]
        for change in changes:
            variant = [{**rows[0], **change}, *rows[1:]]
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize_matchup(variant, trials, 10, 1)

    def test_row_requires_exact_protocol_field_set(self):
        trials, rows = make_scenario()
        missing_error = dict(rows[0])
        del missing_error["error"]
        missing_digest = dict(rows[0])
        del missing_digest["terminal_digest"]
        extra_field = {**rows[0], "unexpected": 1}
        for changed in (missing_error, missing_digest, extra_field):
            with self.subTest(fields=set(changed)), self.assertRaisesRegex(ValueError, "row fields mismatch"):
                summarize_matchup([changed, *rows[1:]], trials, 10, 1)

    def test_more_rotations_do_not_create_independent_deals(self):
        trials, rows = make_scenario()
        hetero_trials, hetero_rows = make_scenario(hetero=True)
        homogeneous = summarize_matchup(rows, trials, 10, 1)
        heterogeneous = summarize_matchup(hetero_rows, hetero_trials, 10, 1)
        self.assertEqual(heterogeneous["games"], 2 * homogeneous["games"])
        self.assertEqual(heterogeneous["independent_deals"], homogeneous["independent_deals"])
        self.assertEqual(heterogeneous["metrics"]["win_rate"], homogeneous["metrics"]["win_rate"])
        # An extra rotation for the same deal is a duplicate, not a fifth cluster.
        with self.assertRaises(ValueError):
            summarize_matchup(rows + rows[:1], trials, 10, 1)


if __name__ == "__main__":
    unittest.main()
