"""CPU-only edge cases for the diagnostic score accounting."""
import math
import unittest

from scripts import _bootstrap  # Make src importable without importing Torch.
from experiments.p3f_score_diagnostic import compare_scores, summarize_scores
from guandan.types import Action, PASS


class ScoreSummaryTests(unittest.TestCase):
    def setUp(self):
        self.partial = Action("single", (0,))
        self.finish = Action("pair", (0, 1))
        self.legal = [PASS, self.partial, self.finish]

    def test_finish_score_gap_preserves_chosen_and_teacher_indices(self):
        row = summarize_scores([.7, .8, .6], self.legal, 2, 1, 2)
        self.assertEqual(row["selected_index"], 1)
        self.assertEqual(row["teacher_index"], 2)
        self.assertEqual(row["finish_indexes"], [2])
        self.assertEqual(row["argmax_index"], 1)
        self.assertAlmostEqual(row["finish_score_gap"], .2)
        self.assertFalse(row["chosen_finishes"])

    def test_exact_ties_use_first_original_candidate(self):
        row = summarize_scores([.8, .8, .8], self.legal, 2, 2, 2)
        self.assertEqual(row["argmax_index"], 0)
        self.assertEqual(row["near_top_indexes"], [0, 1, 2])
        self.assertTrue(row["chosen_finishes"])
        self.assertEqual(row["chosen_score_gap"], 0)
        self.assertEqual(row["spread"], 0)

    def test_physical_copies_are_not_collapsed(self):
        other_copy = Action("single", (54,))
        row = summarize_scores([.2, .2], [self.partial, other_copy], 2, 1, 0)
        self.assertEqual(row["candidate_count"], 2)
        self.assertEqual(row["selected_index"], 1)
        self.assertEqual(row["argmax_index"], 0)
        self.assertEqual(row["finish_indexes"], [])
        self.assertIsNone(row["finish_score_gap"])

    def test_saturation_threshold_is_absolute_inclusive(self):
        row = summarize_scores([-.99, .989, 1.0], self.legal, 2, 2, 2)
        self.assertEqual(row["saturated_count"], 2)
        self.assertEqual(row["min"], -.99)
        self.assertEqual(row["max"], 1.)
        self.assertAlmostEqual(row["spread"], 1.99)

    def test_pass_never_finishes_even_for_invalid_empty_cards_semantics(self):
        row = summarize_scores([0], [PASS], 1, 0, 0)
        self.assertEqual(row["finish_indexes"], [])
        self.assertFalse(row["chosen_finishes"])

    def test_malformed_scores_and_indices_rejected(self):
        for scores, chosen, teacher in (([], 0, 0), ([0], 0, 0),
                ([0, math.nan, 1], 0, 0), ([0, 1, math.inf], 0, 0),
                ([0, 1, 2], -1, 0), ([0, 1, 2], 0, 3),
                ([0, 1, 2], True, 0)):
            with self.subTest(scores=scores, chosen=chosen, teacher=teacher):
                with self.assertRaises(ValueError):
                    summarize_scores(scores, self.legal, 2, chosen, teacher)
        for hand_size in (0, -1, True):
            with self.assertRaises(ValueError):
                summarize_scores([0, 1, 2], self.legal, hand_size, 0, 0)


class FidelityTests(unittest.TestCase):
    def test_numerical_tie_rank_change_is_recorded_without_false_failure(self):
        row = compare_scores([.5, .5], [.5, .5000005])
        self.assertTrue(row["argmax_changed"])
        self.assertTrue(row["rank_sensitive"])
        self.assertTrue(row["near_tie"])
        self.assertFalse(row["failed"])
        self.assertEqual(row["reference_argmax"], 0)
        self.assertEqual(row["alternative_argmax"], 1)

    def test_change_within_two_error_bands_is_rank_sensitive(self):
        row = compare_scores([.5, .5000015], [.5000008, .5000007])
        self.assertTrue(row["rank_sensitive"])
        self.assertFalse(row["near_tie"])
        self.assertFalse(row["failed"])

    def test_numeric_failure_even_when_argmax_unchanged(self):
        row = compare_scores([0, 1], [.000002, 1])
        self.assertFalse(row["argmax_changed"])
        self.assertTrue(row["failed"])

    def test_genuine_choice_gap_rejected(self):
        row = compare_scores([0, .01], [.01, 0])
        self.assertFalse(row["rank_sensitive"])
        self.assertGreater(row["choice_gap"], 2e-6)
        self.assertTrue(row["failed"])

    def test_empty_mismatching_nonfinite_comparisons_rejected(self):
        for a, b in (([], []), ([0], [0, 1]), ([0], [math.nan])):
            with self.assertRaises(ValueError):
                compare_scores(a, b)


if __name__ == "__main__":
    unittest.main()
