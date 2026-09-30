"""Black-box information-boundary and replay/state contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import unittest

from guandan.env.hand_env import HandEnv
from guandan.rules.cards import card_id, rank
from guandan.types import Action, PASS


def _small_worlds() -> tuple[HandEnv, HandEnv]:
    c = card_id
    visible = (c(3, 0), c(4, 0), c(5, 0))
    next_hand = (c(6, 2), c(7, 2), c(8, 2))
    hidden_two = (c(9, 3), c(10, 3), c(11, 3))
    hidden_three = (c(12, 1), c(13, 1), c(14, 1))
    first = HandEnv.from_hands((visible, next_hand, hidden_two, hidden_three))
    second = HandEnv.from_hands((
        visible, next_hand,
        (hidden_three[0], *hidden_two[1:]),
        (hidden_two[0], *hidden_three[1:]),
    ))
    return first, second


class ContractTests(unittest.TestCase):
    def test_hidden_hand_exchange_preserves_observation_and_legal_semantics(self):
        a, b = _small_worlds()
        self.assertNotEqual(a.state_digest(), b.state_digest())
        for player in (0, 1):
            with self.subTest(player=player):
                self.assertEqual(a.observe(player), b.observe(player))
        observation_fields = asdict(a.observe(0))
        for forbidden in ("seed", "state_digest", "initial_hands", "research_fixture"):
            self.assertNotIn(forbidden, observation_fields)
        self.assertEqual(set(a.legal_actions(0)), set(b.legal_actions(0)))
        # A second exchange includes the next player, so every opponent's
        # hidden hand is tested against seat 0's strategy inputs.
        hands = a.state.initial_hands
        changed_next_player = HandEnv.from_hands((
            hands[0], (hands[2][0], *hands[1][1:]),
            (hands[1][0], *hands[2][1:]), hands[3],
        ))
        self.assertNotEqual(a.state_digest(), changed_next_player.state_digest())
        self.assertEqual(a.observe(0), changed_next_player.observe(0))
        self.assertEqual(set(a.legal_actions(0)),
                         set(changed_next_player.legal_actions(0)))

        first_card = a.observe(0).hand[0]
        lead = Action("single", (first_card,), rank(first_card))
        a.step(0, lead, state_version=0)
        b.step(0, lead, state_version=0)
        self.assertNotEqual(a.state_digest(), b.state_digest())
        self.assertEqual(a.observe(1), b.observe(1))
        self.assertEqual(set(a.legal_actions(1)), set(b.legal_actions(1)))

    def test_seed_replay_and_step_digests_are_deterministic(self):
        a, b = HandEnv(), HandEnv()
        self.assertEqual(a.reset(seed=20260924), b.reset(seed=20260924))
        self.assertEqual(a.state_digest(), b.state_digest())
        initial = a.observe(0)
        first_card = initial.hand[0]
        actions = ((0, Action("single", (first_card,), rank(first_card))),
                   (1, PASS), (2, PASS))
        for version, (seat, action) in enumerate(actions):
            with self.subTest(version=version):
                self.assertEqual(a.step(seat, action, state_version=version),
                                 b.step(seat, action, state_version=version))
                self.assertEqual(a.state_digest(), b.state_digest())
        replay = a.serialize_replay()
        self.assertEqual("research_full", replay["visibility"])
        self.assertEqual(3, len(replay["steps"]))
        self.assertEqual(a.state_digest(), replay["final_digest"])
        self.assertEqual(a.state_digest(), HandEnv.replay(replay).state_digest())

    def test_replay_tampering_is_rejected(self):
        env, _ = _small_worlds()
        first_card = env.observe(0).hand[0]
        env.step(0, Action("single", (first_card,), rank(first_card)), state_version=0)
        original = env.serialize_replay()
        changes = {
            "visibility": lambda r: r.__setitem__("visibility", "player"),
            "rules_version": lambda r: r.__setitem__("rules_version", "gd-hand-fake"),
            "action_version": lambda r: r.__setitem__("action_version", "gd-action-fake"),
            "initial_hand": lambda r: r["initial_hands"][0].__setitem__(0, 107),
            "step_digest": lambda r: r["steps"][0].__setitem__("digest", "0" * 64),
            "step_action": lambda r: r["steps"][0]["action"].__setitem__("main_rank", 14),
            "step_version": lambda r: r["steps"][0].__setitem__("state_version", 4),
            "final_digest": lambda r: r.__setitem__("final_digest", "0" * 64),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name):
                replay = deepcopy(original)
                mutate(replay)
                with self.assertRaises((ValueError, TypeError)):
                    HandEnv.replay(replay)

    def test_wrong_seat_and_stale_version_leave_state_unchanged(self):
        env, _ = _small_worlds()
        first_card = env.observe(0).hand[0]
        lead = Action("single", (first_card,), rank(first_card))
        before = env.state_digest()
        with self.assertRaises(ValueError):
            env.step(1, Action("single", (env.observe(1).hand[0],),
                               rank(env.observe(1).hand[0])), state_version=0)
        self.assertEqual(before, env.state_digest())
        with self.assertRaises(ValueError):
            env.step(0, lead, state_version=1)
        self.assertEqual(before, env.state_digest())
        env.step(0, lead, state_version=0)
        accepted = env.state_digest()
        with self.assertRaises(ValueError):
            env.step(1, PASS, state_version=0)
        self.assertEqual(accepted, env.state_digest())


if __name__ == "__main__":
    unittest.main()
