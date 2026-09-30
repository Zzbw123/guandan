"""Diagnostic numerators and denominators at the policy observation boundary."""
from dataclasses import replace
import unittest

from experiments.p3e_metrics import decision_metrics, sum_metrics
from guandan.types import Action, PASS, PlayerObservation


def observation(hand=(0, 1, 2), **changes):
    obs = PlayerObservation(
        player_id=0, hand=hand, level=2, current_player=0,
        remaining_counts=(len(hand), 10, 10, 10), finish_order=(),
        last_action=Action("single", (8,)), last_player=1,
        passed_players=(), history=(), state_version=1,
        terminal=False, settlement=None,
    )
    return replace(obs, **changes)


class DecisionMetricsTests(unittest.TestCase):
    def test_forced_pass_is_not_optional_or_teammate_response(self):
        row = decision_metrics(observation(last_player=2), [PASS], PASS)
        self.assertEqual(row["forced_pass"], 1)
        for key in ("choices", "optional_pass_opportunities", "optional_passes",
                    "teammate_response_opportunities", "teammate_overtakes",
                    "endgame_optional_pass_opportunities", "endgame_optional_passes"):
            self.assertEqual(row[key], 0, key)
        self.assertEqual(row["action_pass"], 1)
        self.assertEqual(row["cards_played"], 0)

    def test_optional_pass_denominator_exists_when_nonpass_is_chosen(self):
        play = Action("single", (0,))
        for chosen in (play, PASS):
            row = decision_metrics(observation(), [play, PASS], chosen)
            self.assertEqual(row["optional_pass_opportunities"], 1)
            self.assertEqual(row["optional_passes"], int(chosen == PASS))
            self.assertEqual(row["choices"], 1)
            self.assertEqual(row["forced_pass"], 0)
            self.assertEqual(row["endgame_optional_pass_opportunities"], 1)
            self.assertEqual(row["endgame_optional_passes"], int(chosen == PASS))

    def test_teammate_response_and_overtake_use_last_player_parity(self):
        play = Action("single", (0,))
        for player in range(4):
            obs = observation(player_id=player, current_player=player,
                              last_player=(player + 2) % 4)
            for chosen in (PASS, play):
                row = decision_metrics(obs, [PASS, play], chosen)
                self.assertEqual(row["teammate_response_opportunities"], 1)
                self.assertEqual(row["teammate_overtakes"], int(chosen != PASS))
        for obs in (observation(last_player=1), observation(last_player=None),
                    observation(last_player=2, last_action=None)):
            row = decision_metrics(obs, [PASS, play], play)
            self.assertEqual(row["teammate_response_opportunities"], 0)
            self.assertEqual(row["teammate_overtakes"], 0)

    def test_immediate_finish_omissions_include_partial_play_and_pass(self):
        finish = Action("triple", (0, 1, 2))
        partial = Action("single", (0,))
        for chosen in (finish, partial, PASS):
            row = decision_metrics(observation(), [partial, PASS, finish], chosen)
            self.assertEqual(row["finish_opportunities"], 1)
            self.assertEqual(row["missed_finishes"], int(chosen != finish))
            self.assertEqual(row["cards_played"], len(chosen.cards))
            self.assertEqual(row["legal_candidates"], 3)
            self.assertEqual([key for key in row if key.startswith("action_")],
                             [f"action_{chosen.kind}"])
        row = decision_metrics(observation(), [partial, PASS], PASS)
        self.assertEqual(row["finish_opportunities"], 0)
        self.assertEqual(row["missed_finishes"], 0)

    def test_endgame_boundary_and_lead_denominators(self):
        play = Action("single", (0,))
        for count in (5, 6):
            row = decision_metrics(observation(hand=tuple(range(count))), [PASS, play], PASS)
            self.assertEqual(row["optional_pass_opportunities"], 1)
            self.assertEqual(row["optional_passes"], 1)
            for key in ("endgame_decisions", "endgame_optional_pass_opportunities",
                        "endgame_optional_passes"):
                self.assertEqual(row[key], int(count <= 5), key)
        lead = decision_metrics(observation(last_action=None, last_player=None), [play], play)
        self.assertEqual(lead["lead_decisions"], 1)
        self.assertEqual(lead["endgame_decisions"], 1)
        self.assertEqual(lead["endgame_optional_pass_opportunities"], 0)

    def test_invalid_decision_inputs_are_rejected(self):
        play = Action("single", (0,))
        for obs in (observation(terminal=True), observation(current_player=1),
                    observation(current_player=None),
                    observation(player_id=4, current_player=4)):
            with self.assertRaises(ValueError):
                decision_metrics(obs, [play], play)
        with self.assertRaises(ValueError):
            decision_metrics(observation(), [], PASS)
        with self.assertRaises(ValueError):
            decision_metrics(observation(), [play], PASS)

    def test_every_output_is_a_nonnegative_integer_count(self):
        play = Action("single", (0,))
        row = decision_metrics(observation(), [play, PASS], play)
        self.assertEqual(row["decisions"], 1)
        self.assertTrue(all(type(value) is int and value >= 0 for value in row.values()))


class SumMetricsTests(unittest.TestCase):
    def test_dynamic_keys_zero_values_and_generator_rows(self):
        rows = [{"decisions": 1, "action_pass": 1, "zero": 0},
                {"decisions": 1, "action_single": 1}]
        self.assertEqual(sum_metrics(iter(rows)),
                         {"decisions": 2, "action_pass": 1, "action_single": 1, "zero": 0})
        self.assertEqual(sum_metrics([]), {})

    def test_invalid_count_values_are_rejected(self):
        for value in (-1, True, False, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sum_metrics([{"valid": 1}, {"invalid": value}])


if __name__ == "__main__":
    unittest.main()
