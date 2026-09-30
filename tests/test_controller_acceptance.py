"""Controller-owned cross-contract checks, independent of worker assertions."""
import unittest

from guandan.agents import GreedyAgent, RandomAgent, TeamHeuristicAgent
from guandan.env import HandEnv
from guandan.rules import beats, enumerate_actions
from guandan.rules.cards import DECK, card_id, label, rank, suit
from guandan.types import Action


class ControllerAcceptanceTests(unittest.TestCase):
    def test_deck_faces_and_roundtrip(self):
        self.assertEqual(len(DECK), 108)
        self.assertEqual(len({label(c) for c in DECK}), 108)
        for c in DECK:
            self.assertEqual(c, card_id(rank(c), max(0, suit(c)), c // 54))

    def test_all_level_single_order_and_equal_values(self):
        for level in range(2, 15):
            ordered = [r for r in range(2, 15) if r != level] + [level, 16, 17]
            for i, lo in enumerate(ordered):
                low = Action("single", (card_id(lo),), lo)
                self.assertFalse(beats(low, low, level))
                for hi in ordered[i + 1:]:
                    high = Action("single", (card_id(hi),), hi)
                    self.assertTrue(beats(high, low, level))
                    self.assertFalse(beats(low, high, level))

    def test_sequence_comparison_does_not_promote_level(self):
        for kind, count in (("straight", 5), ("straight_flush", 5), ("pair_chain", 6), ("triple_chain", 6)):
            # Comparator receives already legal actions; use equal lengths here.
            for level in range(2, 14):
                low = Action(kind, tuple(range(count)), level)
                high = Action(kind, tuple(range(count)), 14)
                self.assertTrue(beats(high, low, level))
                self.assertFalse(beats(low, high, level))
                self.assertFalse(beats(high, high, level))

    def test_hidden_permutation_does_not_change_policy_outputs(self):
        first = HandEnv()
        first.reset(901)
        hands = list(first.state.hands)
        second_hand, third_hand = list(hands[1]), list(hands[3])
        second_hand[0], third_hand[0] = third_hand[0], second_hand[0]
        hands[1], hands[3] = tuple(second_hand), tuple(third_hand)
        second = HandEnv.from_hands(tuple(hands))
        self.assertNotEqual(first.state_digest(), second.state_digest())
        self.assertEqual(first.observe(0), second.observe(0))
        for agent_type in (RandomAgent, GreedyAgent, TeamHeuristicAgent):
            a, b = agent_type(), agent_type()
            for _ in range(5):
                self.assertEqual(a.act(first.observe(0), first.legal_actions(0)),
                                 b.act(second.observe(0), second.legal_actions(0)))

    def test_full_108_card_deal_and_observation_are_immutable(self):
        env = HandEnv()
        observation = env.reset(321)
        self.assertEqual(tuple(map(len, env.state.hands)), (27, 27, 27, 27))
        self.assertEqual(set(sum(env.state.hands, ())), set(DECK))
        initial = observation.hand
        actions = env.legal_actions(0)
        action = actions[0]
        actions.clear()
        self.assertTrue(env.legal_actions(0))
        env.step(0, action, 0)
        self.assertEqual(observation.hand, initial)
        self.assertEqual(observation.state_version, 0)


if __name__ == "__main__":
    unittest.main()
