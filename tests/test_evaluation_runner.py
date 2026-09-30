"""Paired-trial schedule, information boundary, and failure accounting."""

import unittest
from unittest.mock import patch

from guandan.env import HandEnv
from guandan.evaluation.contracts import Matchup
from guandan.evaluation.runner import run_trial
from guandan.evaluation.schedule import build_trials, deal_hands, policy_rng_seed
from guandan.rules.cards import card_id
from guandan.types import Action, PASS, PlayerObservation


def small_hands():
    return tuple((card_id(rank, 0),) for rank in (2, 3, 4, 5))


class FirstLegalAgent:
    def act(self, observation, legal_actions):
        assert type(observation) is PlayerObservation
        assert type(legal_actions) is tuple
        return legal_actions[0]


class BadAgent:
    def act(self, observation, legal_actions):
        return PASS  # Illegal on the free lead.


class CrashingAgent:
    def act(self, observation, legal_actions):
        raise RuntimeError("policy failed")


class EvaluationRunnerTests(unittest.TestCase):
    def test_schedule_rotates_complete_hands_and_keeps_teams_opposite(self):
        matchup = Matchup("random", "team", "greedy")
        trials = build_trials(matchup, [200000, 200001])
        self.assertEqual(len(trials), 32)
        self.assertEqual(len({t.trial_id for t in trials}), 32)
        original_env = HandEnv()
        original_env.reset(200000)
        original = original_env.state.initial_hands
        for trial in trials[:16]:
            with self.subTest(trial=trial.trial_id):
                self.assertEqual(trial.level, 2)
                self.assertEqual(trial.starting_player, 0)
                self.assertEqual(trial.focal_seat,
                                 (trial.rotation + trial.swap + 2 * trial.flip) % 4)
                self.assertEqual(trial.policies[trial.focal_seat], "random")
                self.assertEqual(trial.policies[(trial.focal_seat + 2) % 4], "team")
                self.assertTrue(all(trial.policies[seat] == "greedy"
                                    for seat in range(4)
                                    if seat % 2 != trial.focal_team))
                hands = deal_hands(trial)
                self.assertEqual(hands, tuple(original[(seat - trial.rotation) % 4]
                                              for seat in range(4)))
        homogeneous = build_trials(Matchup("team", "team", "greedy"), [200000])
        self.assertEqual(len(homogeneous), 8)
        self.assertTrue(all(t.flip == 0 for t in homogeneous))
        with self.assertRaises(ValueError):
            build_trials(matchup, [3, 3])
        with self.assertRaises(ValueError):
            build_trials(matchup, [True])
        with self.assertRaises(ValueError):
            build_trials(matchup, [-1])

    def test_policy_seeds_use_original_hand_seat_but_not_deal_seed_or_swap(self):
        matchup = Matchup("random", "team", "greedy")
        a = build_trials(matchup, [200000, 200001])
        same_deal = a[:16]
        for trial in same_deal:
            for original_seat in range(4):
                physical = (original_seat + trial.rotation) % 4
                seed = policy_rng_seed(51, trial, physical)
                counterpart = next(t for t in same_deal
                                   if t.rotation == trial.rotation and
                                   t.swap != trial.swap and t.flip == trial.flip)
                self.assertEqual(seed, policy_rng_seed(51, counterpart, physical))
                changed_deal_seed = a[16 + same_deal.index(trial)]
                self.assertNotEqual(trial.deal_index, changed_deal_seed.deal_index)
                self.assertNotEqual(seed, policy_rng_seed(51, changed_deal_seed, physical))
        self.assertEqual(len({policy_rng_seed(51, a[0], seat) for seat in range(4)}), 4)
        self.assertNotEqual(policy_rng_seed(51, a[0], 0), policy_rng_seed(52, a[0], 0))

    def test_success_is_deterministic_except_wall_clock_arrays(self):
        trial = build_trials(Matchup("greedy", "greedy", "greedy"), [200000])[0]
        with patch("guandan.evaluation.runner.deal_hands", return_value=small_hands()):
            first, first_measures = run_trial(trial, policy_seed=77)
            second, second_measures = run_trial(trial, policy_seed=77)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "ok")
        self.assertIn(first["win"], (0, 1))
        self.assertEqual(first["team_reward"], 2 * first["win"] - 1)
        self.assertEqual(first["focal_level_gain"] * first["opponent_level_gain"], 0)
        self.assertEqual(len(first["terminal_digest"]), 64)
        self.assertEqual(first["steps"], len(first_measures["enumeration_ms"]))
        self.assertEqual(first["steps"], len(first_measures["decision_ms"]))
        self.assertEqual(first["steps"], len(first_measures["candidate_counts"]))
        self.assertEqual(first_measures["candidate_counts"], second_measures["candidate_counts"])

    def test_illegal_action_and_step_guard_leave_no_outcome(self):
        trial = build_trials(Matchup("random", "random", "random"), [200000])[0]
        with (patch("guandan.evaluation.runner.deal_hands", return_value=small_hands()),
              patch("guandan.evaluation.runner.make_agent", return_value=BadAgent())):
            row, measurements = run_trial(trial, 9)
        self.assertEqual((row["status"], row["illegal_actions"], row["timeouts"]),
                         ("error", 1, 0))
        self.assertEqual(row["error"]["type"], "IllegalActionError")
        self.assertEqual(row["steps"], 0)
        self.assertEqual(len(measurements["decision_ms"]), 1)
        for name in ("win", "team_reward", "focal_level_gain", "opponent_level_gain",
                     "finish_order", "terminal_digest"):
            self.assertIsNone(row[name])

        with (patch("guandan.evaluation.runner.deal_hands", return_value=small_hands()),
              patch("guandan.evaluation.runner.make_agent", return_value=FirstLegalAgent())):
            row, measurements = run_trial(trial, 9, max_steps=1)
        self.assertEqual(row["error"]["type"], "StepLimitExceeded")
        self.assertEqual(row["steps"], 1)
        self.assertEqual(len(measurements["enumeration_ms"]), 1)

    def test_returned_action_is_subject_to_soft_timeout(self):
        trial = build_trials(Matchup("random", "random", "random"), [200000])[0]
        with (patch("guandan.evaluation.runner.deal_hands", return_value=small_hands()),
              patch("guandan.evaluation.runner.make_agent", return_value=FirstLegalAgent()),
              patch("guandan.evaluation.runner.perf_counter_ns",
                    side_effect=[0, 100_000, 200_000, 1_300_000])):
            row, measurements = run_trial(trial, 9, decision_timeout_ms=1.0)
        self.assertEqual(row["error"]["type"], "DecisionTimeoutError")
        self.assertEqual((row["steps"], row["illegal_actions"], row["timeouts"]),
                         (0, 0, 1))
        self.assertEqual(measurements["enumeration_ms"], [0.1])
        self.assertEqual(measurements["decision_ms"], [1.1])
        self.assertEqual(measurements["candidate_counts"], [1])

    def test_slow_crashing_policy_still_counts_timeout(self):
        trial = build_trials(Matchup("random", "random", "random"), [200000])[0]
        with (patch("guandan.evaluation.runner.deal_hands", return_value=small_hands()),
              patch("guandan.evaluation.runner.make_agent", return_value=CrashingAgent()),
              patch("guandan.evaluation.runner.perf_counter_ns",
                    side_effect=[0, 100_000, 200_000, 1_300_000])):
            row, measurements = run_trial(trial, 9, decision_timeout_ms=1.0)
        self.assertEqual(row["error"], {"type": "RuntimeError", "message": "policy failed"})
        self.assertEqual((row["steps"], row["timeouts"], row["illegal_actions"]),
                         (0, 1, 0))
        self.assertEqual(measurements["decision_ms"], [1.1])


if __name__ == "__main__":
    unittest.main()
