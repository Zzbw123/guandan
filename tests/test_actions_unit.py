import random
import unittest

from guandan.rules import beats, enumerate_actions, validate_action
from guandan.rules.cards import card_id
from guandan.types import Action, PASS


def ids(*specs):
    return tuple(card_id(*spec) for spec in specs)


class ActionRulesTests(unittest.TestCase):
    def test_copy_equivalence_and_suit_consumption(self):
        s2, s2_copy, c2 = ids((2, 0, 0), (2, 0, 1), (2, 2, 0))
        actions = enumerate_actions((s2, s2_copy, c2), 7)
        singles = [a for a in actions if a.kind == "single"]
        pairs = [a for a in actions if a.kind == "pair"]
        self.assertEqual({a.cards for a in singles}, {(s2,), (c2,)})
        self.assertEqual({a.cards for a in pairs}, {(s2, s2_copy), (s2, c2)})
        self.assertTrue(validate_action(Action("single", (s2_copy,), 2),
                                        (s2, s2_copy, c2), 7))
        self.assertTrue(validate_action(Action("pair", (s2_copy, c2), 2),
                                        (s2, s2_copy, c2), 7))

    def test_wildcard_rank_choices_and_required_declarations(self):
        w1, w2, s8 = ids((7, 1, 0), (7, 1, 1), (8, 0, 0))
        hand = (w1, w2, s8)
        actions = enumerate_actions(hand, 7)
        self.assertIn(Action("pair", (w1, w2), 11,
                             ((w1, 11, -1), (w2, 11, -1))), actions)
        self.assertIn(Action("triple", tuple(sorted(hand)), 8,
                             ((w1, 8, -1), (w2, 8, -1))), actions)
        self.assertIn(Action("single", (w1,), 7), actions)
        self.assertFalse(any(a.kind == "single" and a.main_rank != 7
                             and w1 in a.cards for a in actions))
        self.assertFalse(validate_action(Action("pair", (w1, s8), 8), hand, 7))
        self.assertFalse(validate_action(Action("single", (w1,), 8,
                                                ((w1, 8, -1),)), hand, 7))
        self.assertTrue(validate_action(Action("single", (w2,), 7,
                                               ((w2, 7, -1),)), hand, 7))
        self.assertFalse(validate_action(Action("pair", (w1, w2), 16,
                                                ((w1, 16, -1), (w2, 16, -1))),
                                         hand, 7))

    def test_low_ace_runs_and_no_wrap(self):
        a, two, three, four, five, king = ids(
            (14, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0),
            (5, 0, 0), (13, 0, 0))
        hand = (a, two, three, four, five, king)
        actions = enumerate_actions(hand, 7)
        self.assertIn(Action("straight", tuple(sorted((a, two, three, four, five))), 5),
                      actions)
        self.assertIn(Action("straight_flush", tuple(sorted((a, two, three, four, five))), 5),
                      actions)
        self.assertFalse(validate_action(Action("straight", (king, a, two, three, four), 4),
                                         hand, 7))
        self.assertFalse(validate_action(Action("straight", (a, two, three, four, five), 14),
                                         hand, 7))
        self.assertTrue(validate_action(Action("pair_chain", ids(
            (14, 0, 0), (14, 0, 1), (2, 0, 0), (2, 0, 1),
            (3, 0, 0), (3, 0, 1)), 3),
            ids((14, 0, 0), (14, 0, 1), (2, 0, 0), (2, 0, 1),
                (3, 0, 0), (3, 0, 1)), 7))

    def test_flush_wildcard_suit_and_normal_straight(self):
        w, s2, s3, s4, s5 = ids((7, 1, 0), (2, 0, 0), (3, 0, 0),
                               (4, 0, 0), (5, 0, 0))
        hand = (w, s2, s3, s4, s5)
        cards = tuple(sorted(hand))
        flush = Action("straight_flush", cards, 5, ((w, 14, 0),))
        ordinary = Action("straight", cards, 5, ((w, 14, -1),))
        actions = enumerate_actions(hand, 7)
        self.assertIn(flush, actions)
        self.assertIn(ordinary, actions)
        self.assertTrue(validate_action(flush, hand, 7))
        self.assertFalse(validate_action(Action("straight_flush", cards, 5,
                                                ((w, 14, 1),)), hand, 7))
        self.assertFalse(validate_action(Action("straight", cards, 5,
                                                ((w, 14, 0),)), hand, 7))

    def test_joker_pair_full_house_and_bomb_order(self):
        w, s4, c4, d4, h4, small1, small2, big1, big2 = ids(
            (7, 1, 0), (4, 0, 0), (4, 2, 0), (4, 3, 0),
            (4, 1, 0), (16, 0, 0), (16, 0, 1),
            (17, 0, 0), (17, 0, 1))
        hand = (w, s4, c4, d4, h4, small1, small2, big1, big2)
        actions = enumerate_actions(hand, 7)
        self.assertIn(Action("full_house", tuple(sorted((s4, c4, d4, small1, small2))), 4),
                      actions)
        self.assertIn(Action("joker_bomb", tuple(sorted((small1, small2, big1, big2))), 17),
                      actions)
        self.assertFalse(any(any(declared == 16 for _, declared, _ in a.wildcards)
                             for a in actions))
        four = Action("bomb", (s4, c4, d4, h4), 4)
        five = Action("bomb", (s4, c4, d4, h4, w), 4, ((w, 4, -1),))
        flush = Action("straight_flush", (1, 2, 3, 4, 5), 7)
        six = Action("bomb", (1, 2, 3, 4, 5, 6), 4)
        joker = Action("joker_bomb", (small1, small2, big1, big2), 17)
        self.assertTrue(beats(five, four, 7))
        self.assertTrue(beats(flush, five, 7))
        self.assertTrue(beats(six, flush, 7))
        self.assertTrue(beats(joker, six, 7))
        self.assertFalse(beats(four, joker, 7))
        self.assertFalse(beats(joker, joker, 7))
        self.assertFalse(beats(PASS, four, 7))
        self.assertTrue(beats(Action("single", (1,), 7),
                              Action("single", (2,), 14), 7))
        self.assertFalse(beats(Action("straight", (1, 2, 3, 4, 5), 5),
                               Action("pair", (1, 2), 2), 7))

    def test_random_hands_all_candidates_validate(self):
        for seed in range(12):
            hand = random.Random(seed).sample(range(108), 27)
            actions = enumerate_actions(hand, 7)
            self.assertEqual(len(actions), len(set(actions)))
            self.assertTrue(actions)
            for action in actions:
                self.assertTrue(validate_action(action, hand, 7),
                                f"seed={seed}: {action}")


if __name__ == "__main__":
    unittest.main()
