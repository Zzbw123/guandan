"""Independent, small-hand semantic oracle for the action enumerator.

This deliberately does not import action validation or any engine classifier.
It enumerates physical subsets and wildcard rank assignments.  The comparison
uses the public R15 semantic identity: kind, main rank, and consumed face
multiset (a face is a physical ID modulo the deck size).
"""

from __future__ import annotations

from itertools import combinations, product
from random import Random
import unittest

from guandan.rules.cards import card_id
from guandan.rules.actions import enumerate_actions


KINDS = {
    "single", "pair", "triple", "full_house", "straight", "pair_chain",
    "triple_chain", "bomb", "straight_flush", "joker_bomb",
}


def _face(card: int) -> int:
    return card % 54


def _rank(card: int) -> int:
    face = _face(card)
    return face % 13 + 2 if face < 52 else face - 36


def _suit(card: int) -> int:
    face = _face(card)
    return face // 13 if face < 52 else -1


def _is_wild(card: int, level: int) -> bool:
    return _rank(card) == level and _suit(card) == 1


def _sequence_top(ranks: tuple[int, ...], length: int) -> int | None:
    """Return natural top rank, including the single legal low-A sequence."""
    if len(ranks) != length or len(set(ranks)) != length:
        return None
    values = set(ranks)
    low_ace = {14, *range(2, length + 1)}
    if values == low_ace:
        return length
    start = min(values)
    if start >= 2 and values == set(range(start, start + length)) and max(values) <= 14:
        return max(values)
    return None


def _classify(ranks: tuple[int, ...], suits: tuple[int, ...], wild_positions: tuple[int, ...]):
    """Classify one *independently assigned* subset; emit all declarations."""
    n = len(ranks)
    counts = {rank: ranks.count(rank) for rank in set(ranks)}
    if n == 1:
        yield "single", ranks[0]
    if n == 2 and len(counts) == 1:
        yield "pair", ranks[0]
    if n == 3 and len(counts) == 1:
        yield "triple", ranks[0]
    if n == 5 and sorted(counts.values()) == [2, 3]:
        yield "full_house", next(rank for rank, count in counts.items() if count == 3)

    if n == 5:
        top = _sequence_top(ranks, 5)
        if top is not None:
            yield "straight", top
            # Wildcards may choose any suit.  All natural cards must already
            # share one suit; at most two wildcards exist in a 5-card hand.
            natural_suits = {suits[i] for i in range(n) if i not in wild_positions}
            if len(natural_suits) == 1 and -1 not in natural_suits:
                yield "straight_flush", top

    if n == 6 and len(counts) == 3 and set(counts.values()) == {2}:
        top = _sequence_top(tuple(counts), 3)
        if top is not None:
            yield "pair_chain", top
    if n == 6 and len(counts) == 2 and set(counts.values()) == {3}:
        top = _sequence_top(tuple(counts), 2)
        if top is not None:
            yield "triple_chain", top
    if 4 <= n <= 10 and len(counts) == 1 and ranks[0] <= 14:
        yield "bomb", ranks[0]
    if n == 4 and counts.get(16) == 2 and counts.get(17) == 2:
        yield "joker_bomb", 17


def oracle_semantics(hand: tuple[int, ...], level: int) -> set[tuple[str, int, tuple[int, ...]]]:
    """Enumerate subsets and the 13^w rank choices for w <= 2 wildcards."""
    answer: set[tuple[str, int, tuple[int, ...]]] = set()
    for size in range(1, min(len(hand), 10) + 1):
        for subset in combinations(hand, size):
            wild_positions = tuple(i for i, card in enumerate(subset) if _is_wild(card, level))
            if size == 1 or not wild_positions:
                assignments = ((),)
            else:
                assignments = product(range(2, 15), repeat=len(wild_positions))
            faces = tuple(sorted(_face(card) for card in subset))
            natural = tuple(_rank(card) for card in subset)
            suits = tuple(_suit(card) for card in subset)
            for assigned in assignments:
                ranks = list(natural)
                for position, value in zip(wild_positions, assigned):
                    ranks[position] = value
                for kind, main_rank in _classify(tuple(ranks), suits, wild_positions):
                    answer.add((kind, main_rank, faces))
    return answer


def engine_semantics(hand: tuple[int, ...], level: int) -> set[tuple[str, int, tuple[int, ...]]]:
    actions = enumerate_actions(hand, level)
    result = set()
    for action in actions:
        if action.kind == "pass":
            raise AssertionError("enumerate_actions returned pass")
        if action.kind not in KINDS:
            raise AssertionError(f"unexpected action kind: {action.kind}")
        if not set(action.cards).issubset(hand) or len(action.cards) != len(set(action.cards)):
            raise AssertionError(f"action uses absent or duplicate physical cards: {action}")
        wild_cards = {card for card in action.cards if _is_wild(card, level)}
        declarations = {card: (rank, suit) for card, rank, suit in action.wildcards}
        if len(declarations) != len(action.wildcards):
            raise AssertionError(f"duplicate wildcard declarations: {action}")
        if set(declarations) != (set() if action.kind == "single" else wild_cards):
            raise AssertionError(f"missing or extra wildcard declaration: {action}")
        ranks = []
        suits = []
        wild_positions = []
        for position, card in enumerate(action.cards):
            physical_suit = _suit(card)
            if card in declarations:
                declared_rank, declared_suit = declarations[card]
                if declared_rank not in range(2, 15):
                    raise AssertionError(f"wildcard declared an invalid rank: {action}")
                if action.kind == "straight_flush":
                    if declared_suit not in range(4):
                        raise AssertionError(f"flush wildcard lacks a suit: {action}")
                elif declared_suit != -1:
                    raise AssertionError(f"nonflush wildcard declares a suit: {action}")
                ranks.append(declared_rank)
                suits.append(declared_suit)
                wild_positions.append(position)
            else:
                ranks.append(_rank(card))
                suits.append(physical_suit)
        if action.kind == "straight_flush" and len(set(suits)) != 1:
            raise AssertionError(f"straight flush has mismatched declared suits: {action}")
        classifications = set(_classify(tuple(ranks), tuple(suits), tuple(wild_positions)))
        if (action.kind, action.main_rank) not in classifications:
            raise AssertionError(f"incorrect declared kind or main rank: {action}")
        result.add((action.kind, action.main_rank, tuple(sorted(_face(c) for c in action.cards))))
    return result


class EnumerationOracleTests(unittest.TestCase):
    def test_seeded_six_card_oracle_all_levels_and_wildcard_counts(self):
        rng = Random(20260924)
        for level in range(2, 15):
            wildcards = (card_id(level, 1), card_id(level, 1, 1))
            ordinary_pool = tuple(card for card in range(108) if card not in wildcards)
            for wild_count in range(3):
                with self.subTest(level=level, wild_count=wild_count):
                    hand = tuple(sorted((*wildcards[:wild_count],
                                         *rng.sample(ordinary_pool, 6 - wild_count))))
                    expected = oracle_semantics(hand, level)
                    actual = engine_semantics(hand, level)
                    self.assertEqual(expected, actual,
                                     f"missing={expected - actual}; extra={actual - expected}")

    def test_independent_small_hand_oracle(self):
        c = card_id
        cases = {
            "ordinary_and_flush_straight": (2, (c(3, 0), c(4, 0), c(5, 0), c(6, 0), c(7, 0))),
            "two_wildcards": (5, (c(5, 1), c(5, 1, 1), c(7, 0), c(8, 0), c(9, 0))),
            "joker_pair_in_full_house": (2, (c(9, 0), c(9, 2), c(9, 3), c(16), c(16, copy=1))),
            "ace_low_pair_chain": (5, (c(14, 0), c(14, 2), c(2, 0), c(2, 2), c(3, 0), c(3, 2))),
            "ace_low_triple_chain": (5, (c(14, 0), c(14, 2), c(14, 3), c(2, 0), c(2, 2), c(2, 3))),
            "four_jokers": (2, (c(16), c(16, copy=1), c(17), c(17, copy=1))),
            "level_bomb": (7, (c(7, 0), c(7, 2), c(7, 3), c(7, 0, 1), c(7, 1))),
            "ace_low_straight": (7, (c(14, 0), c(2, 1), c(3, 2), c(4, 3), c(5, 0))),
            "ace_high_straight": (7, (c(10, 0), c(11, 1), c(12, 2), c(13, 3), c(14, 0))),
        }
        seen = set()
        for name, (level, hand) in cases.items():
            with self.subTest(name=name):
                expected = oracle_semantics(hand, level)
                actual = engine_semantics(hand, level)
                self.assertEqual(expected, actual, f"missing={expected - actual}; extra={actual - expected}")
                seen.update(kind for kind, _, _ in expected)
        self.assertEqual(KINDS, seen)

    def test_ten_card_bomb_targeted_without_large_hand_oracle(self):
        naturals = tuple(card_id(9, suit, copy) for copy in range(2) for suit in range(4))
        wildcards = (card_id(5, 1), card_id(5, 1, 1))
        hand = naturals + wildcards
        action_keys = engine_semantics(hand, 5)
        for length in range(4, 11):
            with self.subTest(length=length):
                consumed = naturals[:min(length, 8)] + wildcards[:max(0, length - 8)]
                self.assertIn(("bomb", 9, tuple(sorted(_face(card) for card in consumed))),
                              action_keys)

    def test_same_physical_straight_has_two_declarations(self):
        hand = tuple(card_id(rank, 0) for rank in range(3, 8))
        faces = tuple(sorted(_face(card) for card in hand))
        keys = engine_semantics(hand, 2)
        self.assertIn(("straight", 7, faces), keys)
        self.assertIn(("straight_flush", 7, faces), keys)


if __name__ == "__main__":
    unittest.main()
