"""State transition and research replay checks for HandEnv."""

import copy
import json
import unittest

from guandan.env import HandEnv
from guandan.rules.cards import DECK, card_id
from guandan.types import Action, PASS, RulesConfig


def single(card):
    return Action("single", (card,), card % 13 + 2)


class HandEnvTests(unittest.TestCase):
    def setUp(self):
        self.cards = [card_id(2, 0), card_id(3, 0),
                      card_id(4, 0), card_id(5, 0)]

    def fixture(self):
        return HandEnv.from_hands([[c] for c in self.cards], initial_level=10)

    def test_reset_is_deterministic_and_observation_is_private(self):
        a, b = HandEnv(), HandEnv()
        observation = a.reset(9124, RulesConfig(), starting_player=1)
        b.reset(9124, starting_player=1)
        self.assertEqual(a.state.initial_hands, b.state.initial_hands)
        self.assertEqual(tuple(len(h) for h in a.state.hands), (27, 27, 27, 27))
        self.assertEqual(observation.hand, a.state.hands[1])
        self.assertFalse(hasattr(observation, "initial_hands"))
        self.assertFalse(hasattr(observation, "seed"))
        self.assertFalse(hasattr(observation, "state_digest"))
        self.assertEqual(len(observation.remaining_counts), 4)
        with self.assertRaises(ValueError):
            a.legal_actions(0)
        with self.assertRaises(TypeError):
            a.reset(True)
        with self.assertRaises(ValueError):
            a.reset(1, initial_level=15)

    def test_finished_incumbent_pass_cycle_gives_partner_lead(self):
        env = self.fixture()
        env.step(0, single(self.cards[0]))
        self.assertEqual(env.state.finish_order, (0,))
        self.assertIn(PASS, env.legal_actions(1))
        env.step(1, PASS)
        env.step(2, PASS)
        result = env.step(3, PASS)
        self.assertTrue(result.events[0].trick_closed)
        self.assertEqual(result.next_player, 2)
        self.assertIsNone(env.state.last_action)
        self.assertEqual(env.state.passed_players, ())
        self.assertNotIn(PASS, env.legal_actions(2))
        env.assert_invariants()

    def test_double_down_ends_without_fake_loser_ranks(self):
        env = self.fixture()
        env.step(0, single(self.cards[0]))
        env.step(1, PASS)
        result = env.step(2, single(self.cards[2]))
        self.assertTrue(result.terminal)
        self.assertEqual(result.settlement.finish_order, (0, 2))
        self.assertEqual(result.settlement.remaining_players, (1, 3))
        self.assertEqual(result.settlement.level_gain, 3)
        self.assertEqual(result.settlement.team_rewards, (1, -1))
        self.assertEqual(env.observe(3).finish_order, (0, 2))
        self.assertEqual(tuple(map(len, env.state.hands)), (0, 1, 0, 1))
        with self.assertRaises(ValueError):
            env.legal_actions(1)
        with self.assertRaises(ValueError):
            env.step(1, PASS)

    def test_third_finish_assigns_fourth_without_consuming_hand(self):
        env = self.fixture()
        env.step(0, single(self.cards[0]))
        env.step(1, single(self.cards[1]))
        result = env.step(2, single(self.cards[2]))
        self.assertEqual(env.state.finish_order, (0, 1, 2))
        self.assertEqual(result.settlement.finish_order, (0, 1, 2, 3))
        self.assertEqual(result.settlement.remaining_players, (3,))
        self.assertEqual(result.settlement.level_gain, 2)
        self.assertEqual(env.state.hands[3], (self.cards[3],))
        self.assertEqual(len(env.state.played) + sum(map(len, env.state.hands)), 4)
        env.assert_invariants()

    def test_teammate_fourth_gets_one_level(self):
        env = self.fixture()
        env.step(0, single(self.cards[0]))
        env.step(1, single(self.cards[1]))
        env.step(2, PASS)
        result = env.step(3, single(self.cards[3]))
        self.assertEqual(result.settlement.finish_order, (0, 1, 3, 2))
        self.assertEqual(result.settlement.level_gain, 1)
        self.assertEqual(result.settlement.remaining_players, (2,))

    def test_cover_resets_pass_set_and_earlier_passer_can_return(self):
        hands = [
            [card_id(2, 0), card_id(6, 0)],
            [card_id(3, 0), card_id(7, 0)],
            [card_id(4, 0), card_id(8, 0)],
            [card_id(5, 0), card_id(9, 0)],
        ]
        env = HandEnv.from_hands(hands, initial_level=10)
        env.step(0, single(card_id(2, 0)))
        env.step(1, PASS)
        self.assertEqual(env.state.passed_players, (1,))
        env.step(2, single(card_id(4, 0)))
        self.assertEqual(env.state.passed_players, ())
        env.step(3, PASS)
        env.step(0, PASS)
        self.assertEqual(env.state.current_player, 1)
        self.assertIn(single(card_id(7, 0)), env.legal_actions(1))
        env.step(1, single(card_id(7, 0)))
        env.assert_invariants()

    def test_active_incumbent_retains_next_free_lead(self):
        env = HandEnv.from_hands([
            [card_id(2, 0), card_id(6, 0)],
            [card_id(3, 0)], [card_id(4, 0)], [card_id(5, 0)],
        ], initial_level=10)
        env.step(0, single(card_id(2, 0)))
        env.step(1, PASS)
        env.step(2, PASS)
        result = env.step(3, PASS)
        self.assertTrue(result.events[0].trick_closed)
        self.assertEqual(result.next_player, 0)
        self.assertEqual(env.state.passed_players, ())
        self.assertEqual(env.legal_actions(0), [single(card_id(6, 0))])

    def test_rejected_actions_leave_state_unchanged(self):
        env = self.fixture()
        before = env.state_digest()
        with self.assertRaises(ValueError):
            env.step(0, PASS)
        with self.assertRaises(ValueError):
            env.step(0, Action("pass", (), False))
        with self.assertRaises(ValueError):
            env.step(1, single(self.cards[1]))
        with self.assertRaises(ValueError):
            env.step(0, Action("single", (self.cards[1],), 3))
        self.assertEqual(env.state_digest(), before)
        env.step(0, single(self.cards[0]), state_version=0)
        before = env.state_digest()
        with self.assertRaises(ValueError):
            env.step(1, Action("pass", (), False), state_version=1)
        with self.assertRaises(ValueError):
            env.step(1, single(self.cards[1]), state_version=0)
        with self.assertRaises(TypeError):
            env.step(1, single(self.cards[1]), state_version=True)
        with self.assertRaises(ValueError):
            env.step(1, Action("single", (self.cards[1],), 2), state_version=1)
        self.assertEqual(env.state_digest(), before)

    def test_replay_checks_every_event_digest_and_terminal_metadata(self):
        env = self.fixture()
        env.step(0, single(self.cards[0]))
        env.step(1, single(self.cards[1]))
        env.step(2, single(self.cards[2]))
        replay = json.loads(json.dumps(env.serialize_replay()))
        self.assertEqual(replay["visibility"], "research_full")
        self.assertTrue(replay["research_fixture"])
        self.assertNotIn("seed", replay)
        self.assertEqual(HandEnv.replay(replay).state_digest(), env.state_digest())
        for change in (
            lambda r: r["steps"][0].update(digest="0" * 64),
            lambda r: r["steps"][1]["event"].update(trick_closed=True),
            lambda r: r["steps"][0]["event"].update(version=True),
            lambda r: r["steps"][2].update(terminal=False),
            lambda r: r["final_settlement"].update(level_gain=3),
            lambda r: r["final_settlement"].update(level_gain=True),
            lambda r: r["steps"][2]["settlement"].update(winner_team=False),
            lambda r: r.update(action_version="bad"),
            lambda r: r["initial_hands"][0].append(self.cards[1]),
        ):
            broken = copy.deepcopy(replay)
            change(broken)
            with self.assertRaises((ValueError, AssertionError)):
                HandEnv.replay(broken)

    def test_uneven_108_card_distribution_is_research_fixture(self):
        env = HandEnv.from_hands([
            DECK[:26], DECK[26:54], DECK[54:81], DECK[81:]
        ])
        self.assertEqual(tuple(map(len, env.state.hands)), (26, 28, 27, 27))
        self.assertTrue(env.state.research_fixture)
        replay = env.serialize_replay()
        self.assertTrue(replay["research_fixture"])
        self.assertEqual(HandEnv.replay(replay).state_digest(), env.state_digest())


if __name__ == "__main__":
    unittest.main()
