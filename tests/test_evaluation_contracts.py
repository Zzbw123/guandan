"""Black-box acceptance checks for the frozen gd-evaluation-v1 protocol."""

from __future__ import annotations

from dataclasses import fields, replace
from itertools import product
from unittest import TestCase
from unittest.mock import patch

from guandan.agents import GreedyAgent, RandomAgent, TeamHeuristicAgent
from guandan.env import HandEnv
from guandan.evaluation.contracts import Matchup
from guandan.rules.cards import DECK
from guandan.types import Action, PlayerObservation


def _variants(matchup: Matchup, seeds: list[int]):
    from guandan.evaluation.schedule import build_trials

    return build_trials(matchup, seeds)


def _completed_row(trial, win: bool) -> dict:
    """A possible double-down settlement, with all schedule metadata present."""
    winner_team = trial.focal_team if win else 1 - trial.focal_team
    first = winner_team
    return {
        "trial_id": trial.trial_id,
        "matchup_id": trial.matchup_id,
        "deal_index": trial.deal_index,
        "deal_seed": trial.deal_seed,
        "level": trial.level,
        "rotation": trial.rotation,
        "swap": trial.swap,
        "flip": trial.flip,
        "focal_seat": trial.focal_seat,
        "focal_team": trial.focal_team,
        "policies": list(trial.policies),
        "starting_player": trial.starting_player,
        "status": "ok",
        "win": int(win),
        "team_reward": 1 if win else -1,
        "focal_level_gain": 3 if win else 0,
        "opponent_level_gain": 0 if win else 3,
        "finish_order": [first, first + 2],
        "terminal_digest": "a" * 64,
        "steps": 2,
        "illegal_actions": 0,
        "timeouts": 0,
        "error": None,
    }


class EvaluationScheduleAcceptance(TestCase):
    def test_full_matrix_and_hand_rotation(self):
        from guandan.evaluation.schedule import deal_hands

        seed = 200000
        original = HandEnv()
        original.reset(seed, initial_level=2)
        original_hands = original.state.initial_hands
        all_ids = set()
        total = 0
        for focal, teammate, opponent in product(("random", "greedy", "team"), repeat=3):
            matchup = Matchup(focal, teammate, opponent)
            trials = _variants(matchup, [seed])
            expected_count = 8 if focal == teammate else 16
            self.assertEqual(len(trials), expected_count, matchup.key)
            self.assertEqual(len({t.trial_id for t in trials}), expected_count)
            self.assertEqual({(t.rotation, t.swap, t.flip) for t in trials},
                             {(r, b, f) for r in range(4) for b in range(2)
                              for f in range(1 if focal == teammate else 2)})
            for trial in trials:
                self.assertEqual(trial.matchup_id, matchup.key)
                self.assertEqual((trial.deal_index, trial.deal_seed, trial.level), (0, seed, 2))
                self.assertEqual(trial.starting_player, 0)
                self.assertEqual(trial.focal_seat, (trial.rotation + trial.swap + 2 * trial.flip) % 4)
                self.assertEqual(trial.focal_team, trial.focal_seat % 2)
                self.assertEqual(trial.policies[trial.focal_seat], focal)
                self.assertEqual(trial.policies[(trial.focal_seat + 2) % 4], teammate)
                self.assertEqual([trial.policies[s] for s in range(4)
                                  if s % 2 != trial.focal_team], [opponent, opponent])
                hands = deal_hands(trial)
                self.assertEqual(tuple(map(len, hands)), (27, 27, 27, 27))
                self.assertEqual(set(card for hand in hands for card in hand), set(DECK))
                for old_seat in range(4):
                    self.assertEqual(tuple(hands[(old_seat + trial.rotation) % 4]),
                                     original_hands[old_seat])
                    self.assertEqual(((old_seat + trial.rotation) % 4 + 2) % 4,
                                     (old_seat + 2 + trial.rotation) % 4)
                self.assertEqual(hands[0], original_hands[(-trial.rotation) % 4])
                all_ids.add(trial.trial_id)
            self.assertEqual({t.focal_seat for t in trials}, set(range(4)))
            self.assertEqual({t.focal_team for t in trials}, {0, 1})
            total += len(trials)
        self.assertEqual(total, 360)
        self.assertEqual(len(all_ids), 360)

    def test_level_cycle_and_policy_rng_is_independent_of_deal_seed(self):
        from guandan.evaluation.schedule import policy_rng_seed

        matchup = Matchup("random", "team", "greedy")
        seeds = list(range(200000, 200027))
        trials = _variants(matchup, seeds)
        self.assertEqual(len(trials), 27 * 16)
        self.assertEqual({(t.deal_index, t.level) for t in trials},
                         {(i, 2 + i % 13) for i in range(27)})
        trial = trials[0]
        changed_deal_seed = replace(trial, deal_seed=7777777)
        changed_policy_seed = 98765
        for seat in range(4):
            first = policy_rng_seed(12345, trial, seat)
            self.assertIs(type(first), int)
            self.assertEqual(first, policy_rng_seed(12345, trial, seat))
            self.assertEqual(first, policy_rng_seed(12345, changed_deal_seed, seat))
            self.assertNotEqual(first, policy_rng_seed(changed_policy_seed, trial, seat))
        same_hand_same_rotation = next(t for t in trials if t.deal_index == 0
                                       and t.rotation == trial.rotation
                                       and t.swap != trial.swap)
        for seat in range(4):
            self.assertEqual(policy_rng_seed(12345, trial, seat),
                             policy_rng_seed(12345, same_hand_same_rotation, seat))
        next_deal = next(t for t in trials if t.deal_index == 1
                         and (t.rotation, t.swap, t.flip) ==
                         (trial.rotation, trial.swap, trial.flip))
        self.assertNotEqual(policy_rng_seed(12345, trial, 0),
                            policy_rng_seed(12345, next_deal, 0))


class EvaluationRunnerAcceptance(TestCase):
    def test_real_trial_repeats_and_policy_receives_only_observation_and_actions(self):
        from guandan.agents.baselines import make_agent
        from guandan.evaluation import runner

        trial = _variants(Matchup("random", "greedy", "team"), [200000])[0]
        calls = []

        class SpyAgent:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def act(self, observation, legal_actions):
                self_outer.assertIs(type(observation), PlayerObservation)
                self_outer.assertTrue(legal_actions)
                self_outer.assertTrue(all(type(a) is Action for a in legal_actions))
                names = {field.name for field in fields(observation)}
                self_outer.assertFalse(names & {"deal_seed", "policy_seed", "initial_hands",
                                                "hands", "trial", "game_state", "state"})
                self_outer.assertEqual(len(observation.hand),
                                       observation.remaining_counts[observation.player_id])
                self_outer.assertTrue(set(observation.hand) <= set(DECK))
                calls.append((observation.player_id, observation.state_version))
                return self.wrapped.act(observation, legal_actions)

        self_outer = self

        def spy_factory(name, seed=0):
            self.assertIs(type(seed), int)
            self.assertIn(name, ("random", "greedy", "team"))
            return SpyAgent(make_agent(name, seed))

        with patch.object(runner, "make_agent", side_effect=spy_factory):
            first, measurements = runner.run_trial(trial, policy_seed=817)
        second, _ = runner.run_trial(trial, policy_seed=817)
        self.assertTrue(calls)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["illegal_actions"], 0)
        self.assertEqual(first["timeouts"], 0)
        self.assertEqual({"enumeration_ms", "decision_ms", "candidate_counts"},
                         set(measurements))
        self.assertEqual(len(measurements["decision_ms"]), first["steps"])
        self.assertEqual(first, second)

    def test_hidden_hand_exchange_preserves_initial_policy_decision(self):
        first = HandEnv()
        first.reset(200000)
        hands = [list(hand) for hand in first.state.hands]
        hands[1][0], hands[3][0] = hands[3][0], hands[1][0]
        second = HandEnv.from_hands(hands)
        self.assertNotEqual(first.state_digest(), second.state_digest())
        self.assertEqual(first.observe(0), second.observe(0))
        self.assertEqual(first.legal_actions(0), second.legal_actions(0))
        for policy in (RandomAgent, GreedyAgent, TeamHeuristicAgent):
            a, b = (policy(817), policy(817)) if policy is RandomAgent else (policy(), policy())
            self.assertEqual(a.act(first.observe(0), first.legal_actions(0)),
                             b.act(second.observe(0), second.legal_actions(0)))

    def test_same_policy_swap_and_heterogeneous_role_flip_pairing(self):
        from guandan.evaluation.runner import run_trial

        homogeneous = _variants(Matchup("greedy", "greedy", "greedy"), [200001])
        a = next(t for t in homogeneous if (t.rotation, t.swap, t.flip) == (0, 0, 0))
        b = next(t for t in homogeneous if (t.rotation, t.swap, t.flip) == (0, 1, 0))
        row_a, _ = run_trial(a, policy_seed=411)
        row_b, _ = run_trial(b, policy_seed=411)
        self.assertEqual((row_a["status"], row_b["status"]), ("ok", "ok"))
        self.assertEqual(row_a["terminal_digest"], row_b["terminal_digest"])
        self.assertEqual(row_a["win"], not row_b["win"])

        ab = _variants(Matchup("random", "greedy", "team"), [200002])
        ba = _variants(Matchup("greedy", "random", "team"), [200002])
        left = next(t for t in ab if (t.rotation, t.swap, t.flip) == (0, 0, 0))
        right = next(t for t in ba if (t.rotation, t.swap, t.flip) == (0, 0, 1))
        self.assertEqual(left.policies, right.policies)
        row_left, _ = run_trial(left, policy_seed=412)
        row_right, _ = run_trial(right, policy_seed=412)
        self.assertEqual((row_left["status"], row_right["status"]), ("ok", "ok"))
        self.assertEqual(row_left["terminal_digest"], row_right["terminal_digest"])
        self.assertEqual(row_left["win"], row_right["win"])


class EvaluationStatisticsAcceptance(TestCase):
    def test_group_means_and_corrupted_inputs_are_rejected(self):
        from guandan.evaluation.statistics import summarize_matchup

        matchup = Matchup("team", "team", "greedy")
        expected = _variants(matchup, [200000, 200001])
        rows = [_completed_row(t, (t.deal_index == 0 and t.rotation == 0)
                               or (t.deal_index == 1 and t.rotation != 0))
                for t in expected]
        summary = summarize_matchup(rows, expected, bootstrap_replicates=32,
                                    bootstrap_seed=123)
        self.assertEqual(summary["matchup_id"], matchup.key)
        self.assertEqual(summary["games"], 16)
        self.assertEqual(summary["independent_deals"], 2)
        self.assertAlmostEqual(summary["metrics"]["win_rate"]["estimate"], 0.5)
        self.assertAlmostEqual(summary["effect_vs_50pp"], 0.0)
        self.assertEqual({cluster["deal_index"]: cluster["win_rate"]
                          for cluster in summary["clusters"]}, {0: 0.25, 1: 0.75})
        self.assertEqual(summary["metrics"]["team_reward"]["estimate"], 0.0)
        self.assertEqual(summary["metrics"]["focal_level_gain"]["estimate"], 1.5)
        self.assertEqual(summary["metrics"]["opponent_level_gain"]["estimate"], 1.5)
        self.assertEqual(summary["metrics"]["win_rate"]["ci95"], None)

        mutations = {
            "missing variant": rows[:-1],
            "duplicate trial": rows + [rows[0]],
            "wrong metadata": [dict(rows[0], level=14), *rows[1:]],
            "timeout": [dict(rows[0], status="error", timeouts=1, error="timeout"), *rows[1:]],
            "illegal action": [dict(rows[0], status="error", illegal_actions=1,
                                    error="illegal action"), *rows[1:]],
            "inconsistent win": [dict(rows[0], win=1 - rows[0]["win"]), *rows[1:]],
        }
        for label, invalid in mutations.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                summarize_matchup(invalid, expected, bootstrap_replicates=32,
                                  bootstrap_seed=123)

    def test_stratified_bootstrap_uses_deal_groups_and_marks_degeneracy(self):
        from guandan.evaluation.statistics import cluster_bootstrap

        constant = cluster_bootstrap([0.0, 0.0, 1.0, 1.0],
                                     [2, 2, 3, 3], repetitions=32, seed=19)
        self.assertEqual(constant["estimate"], 0.5)
        self.assertEqual(constant["ci95"], [0.5, 0.5])
        self.assertTrue(constant["degenerate"])
        insufficient = cluster_bootstrap([0.0, 1.0], [2, 3], repetitions=32, seed=19)
        self.assertIsNone(insufficient["ci95"])
        self.assertTrue(insufficient["reason"])
