"""Observable-only, fixed-layout feature encoding contracts."""

from dataclasses import replace
import unittest

from guandan.env.hand_env import HandEnv
from guandan.learning.encoding import (
    ACTION_DIM, FEATURE_VERSION, STATE_DIM, encode_action, encode_observation,
)
from guandan.rules.cards import card_id
from guandan.types import Action, PASS


def _copy_swap(card: int) -> int:
    return (card + 54) % 108


def _swap_action(action: Action) -> Action:
    return replace(
        action,
        cards=tuple(_copy_swap(card) for card in action.cards),
        wildcards=tuple((_copy_swap(card), value, color)
                        for card, value, color in action.wildcards),
    )


class EncodingTests(unittest.TestCase):
    def test_fixed_dimensions_and_public_layout(self):
        env = HandEnv()
        obs = env.reset(seed=100500, initial_level=7, starting_player=0)
        features = encode_observation(obs)
        self.assertEqual("gd-features-v1", FEATURE_VERSION)
        self.assertEqual(2993, STATE_DIM)
        self.assertEqual(149, ACTION_DIM)
        self.assertEqual(STATE_DIM, len(features))
        self.assertTrue(all(type(value) is float for value in features))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), features[270:274])
        self.assertEqual((0.0,) * 4, features[274:278])
        self.assertEqual((0.0,) * 4, features[278:282])
        self.assertEqual(1.0, features[282 + 7 - 2])
        self.assertEqual(1.0, features[299])  # No previous player.
        self.assertEqual((0.0,) * 149, features[300:449])
        self.assertEqual((0.0,) * (16 * 159), features[449:])

    def test_action_copy_swap_and_wildcard_declaration(self):
        wild = card_id(7, 1)
        natural = card_id(9, 0)
        first = Action("pair", (wild, natural), 9, ((wild, 9, -1),))
        changed = Action("pair", (wild, natural), 9, ((wild, 10, -1),))
        encoded = encode_action(first)
        self.assertEqual(ACTION_DIM, len(encoded))
        self.assertEqual(encoded, encode_action(_swap_action(first)))
        self.assertNotEqual(encoded, encode_action(changed))
        self.assertEqual(0.5, encoded[(wild % 54)])
        self.assertEqual(0.5, encoded[54 + 11 + 18 + (9 - 2) * 5])
        self.assertEqual(2 / 27.0, encoded[-1])
        self.assertEqual(1.0, encode_action(PASS)[54])

    def test_relative_public_event_and_physical_copy_invariance(self):
        c = card_id
        hands = ((c(3), c(4)), (c(5), c(6)), (c(7), c(8)), (c(9), c(10)))
        env = HandEnv.from_hands(hands, initial_level=7)
        lead = Action("single", (c(3),), 3)
        env.step(0, lead)
        obs = env.observe(1)
        encoded = encode_observation(obs)
        self.assertEqual(0.5, encoded[54 + 3 * 54 + c(3) % 54])
        self.assertEqual(1.0, encoded[295 + 3])
        self.assertEqual((0.0,) * (15 * 159), encoded[449:449 + 15 * 159])
        recent = encoded[449 + 15 * 159:]
        self.assertEqual((0.0, 0.0, 0.0, 1.0), recent[:4])
        self.assertEqual(1.0, recent[-1])

        swapped = replace(
            obs,
            hand=tuple(_copy_swap(card) for card in obs.hand),
            last_action=_swap_action(obs.last_action),
            history=tuple(replace(event, action=_swap_action(event.action))
                          for event in obs.history),
        )
        self.assertEqual(encoded, encode_observation(swapped))

    def test_rejects_wrong_boundary_and_malformed_action(self):
        obs = HandEnv().reset(seed=100500)
        with self.assertRaises(TypeError):
            encode_observation({"hand": obs.hand})
        with self.assertRaises(ValueError):
            encode_observation(replace(obs, current_player=1))
        with self.assertRaises(ValueError):
            encode_observation(replace(obs, terminal=True))
        with self.assertRaises(ValueError):
            encode_observation(replace(obs, observation_version="wrong"))
        with self.assertRaises(ValueError):
            encode_action(Action("single", (108,), 3))
        with self.assertRaises(ValueError):
            encode_action(Action("single", (card_id(3), card_id(3)), 3))
        with self.assertRaises(ValueError):
            encode_action(Action("pair", (card_id(3), card_id(4)), 3,
                                 ((card_id(7, 1), 3, -1),)))


if __name__ == "__main__":
    unittest.main()
