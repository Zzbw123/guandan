"""Complete, copy-canonical action generation for gd-hand-v1.

Only interchangeable physical copies of the same face are collapsed.  In
particular, spending a heart-level wildcard and declaring a different main
rank are observable action differences.
"""

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product

from guandan.types import Action
from guandan.rules.cards import is_wild, rank, rank_strength, suit, validate_card


_ORDINARY = tuple(range(2, 15))
_RANKS = _ORDINARY + (16, 17)
_KINDS = frozenset(("single", "pair", "triple", "full_house", "straight",
                    "pair_chain", "triple_chain", "bomb", "straight_flush",
                    "joker_bomb"))


def _runs(length: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    # A is represented by 14 even in the low run; main_rank is the high
    # natural point of that run, never its level-adjusted strength.
    low = tuple(sorted((14, *range(2, length + 1))))
    return ((low, length),) + tuple(
        (tuple(range(start, start + length)), start + length - 1)
        for start in range(2, 16 - length)
    )


_RUNS = {5: _runs(5), 3: _runs(3), 2: _runs(2)}
_RUN_MAX = {length: dict(runs) for length, runs in _RUNS.items()}


def enumerate_actions(hand, level: int) -> list[Action]:
    """Return every non-pass semantic action, canonicalizing copy swaps only.

    The hand is expected to be a valid collection of distinct physical IDs.
    Candidate construction never enumerates arbitrary subsets of the hand.
    """
    if type(level) is not int or level not in _ORDINARY:
        raise ValueError("level must be 2..14")
    hand = tuple(sorted(hand))
    if len(set(hand)) != len(hand):
        raise ValueError("hand contains duplicate physical cards")
    for card in hand:
        validate_card(card)

    wild = tuple(card for card in hand if is_wild(card, level))
    faces: dict[int, dict[int, tuple[int, ...]]] = defaultdict(dict)
    groups: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for card in hand:
        if not is_wild(card, level):
            groups[rank(card)][suit(card)].append(card)
    for value, suits in groups.items():
        faces[value] = {color: tuple(ids) for color, ids in suits.items()}

    found: set[Action] = set()

    @lru_cache(maxsize=None)
    def choices(value: int, count: int, color: int = -1):
        """(natural cards, wild count), retaining each suit consumption."""
        by_suit = faces.get(value, {})
        colors = (color,) if color >= 0 else tuple(sorted(by_suit))
        buckets = tuple(by_suit.get(c, ()) for c in colors)
        result = []
        for amounts in product(*(range(len(bucket) + 1) for bucket in buckets)):
            natural_count = sum(amounts)
            missing = count - natural_count
            if 0 <= missing <= len(wild) and (missing == 0 or value in _ORDINARY):
                cards = tuple(card for bucket, amount in zip(buckets, amounts)
                              for card in bucket[:amount])
                result.append((cards, missing))
        return tuple(result)

    def emit(kind: str, demand: tuple[tuple[int, int], ...], main: int,
             color: int = -1):
        option_sets = tuple(choices(value, count, color) for value, count in demand)
        if not option_sets or any(not options for options in option_sets):
            return
        for selected in product(*option_sets):
            used = sum(missing for _, missing in selected)
            if used > len(wild):
                continue
            cards = tuple(sorted((*wild[:used],
                                  *(card for natural, _ in selected for card in natural))))
            declarations = []
            next_wild = 0
            for (value, _), (_, missing) in zip(demand, selected):
                for card in wild[next_wild:next_wild + missing]:
                    declarations.append((card, value, color))
                next_wild += missing
            found.add(Action(kind, cards, main, tuple(declarations)))

    # A single heart-level card has its fixed natural interpretation.
    for ids in (tuple(ids) for suits in groups.values() for ids in suits.values()):
        found.add(Action("single", (ids[0],), rank(ids[0])))
    if wild:
        found.add(Action("single", (wild[0],), level))

    for value in _RANKS:
        emit("pair", ((value, 2),), value)
        emit("triple", ((value, 3),), value)
    for triple in _ORDINARY:
        for pair in _RANKS:
            if pair != triple:
                emit("full_house", ((triple, 3), (pair, 2)), triple)
    for length, kind, multiplicity in ((5, "straight", 1),
                                       (3, "pair_chain", 2),
                                       (2, "triple_chain", 3)):
        for values, main in _RUNS[length]:
            emit(kind, tuple((value, multiplicity) for value in values), main)
    for color in range(4):
        for values, main in _RUNS[5]:
            emit("straight_flush", tuple((value, 1) for value in values),
                 main, color)
    for value in _ORDINARY:
        for size in range(4, min(10, sum(map(len, faces.get(value, {}).values()))
                                      + len(wild)) + 1):
            emit("bomb", ((value, size),), value)

    if all(len(faces.get(value, {}).get(-1, ())) == 2 for value in (16, 17)):
        cards = tuple(sorted((*faces[16][-1], *faces[17][-1])))
        found.add(Action("joker_bomb", cards, 17))
    return sorted(found, key=lambda a: (a.kind, len(a.cards), a.main_rank,
                                        a.cards, a.wildcards))


def _tier(action: Action) -> int:
    if action.kind == "joker_bomb":
        return 9
    if action.kind == "bomb":
        size = len(action.cards)
        if size == 4:
            return 1
        if size == 5:
            return 2
        if 6 <= size <= 10:
            return size - 2
        return -1
    if action.kind == "straight_flush":
        return 3
    return 0 if action.kind in _KINDS else -1


def beats(candidate: Action, incumbent: Action, level: int) -> bool:
    """Compare already-legal actions under the fixed bomb hierarchy."""
    if not isinstance(candidate, Action) or not isinstance(incumbent, Action):
        return False
    if type(level) is not int or level not in _ORDINARY:
        return False
    candidate_tier, incumbent_tier = _tier(candidate), _tier(incumbent)
    if candidate_tier < 0 or incumbent_tier < 0:
        return False
    if candidate_tier != incumbent_tier:
        return candidate_tier > incumbent_tier
    if candidate_tier == 9:
        return False
    if candidate_tier == 0 and (candidate.kind != incumbent.kind or
                                len(candidate.cards) != len(incumbent.cards)):
        return False
    if candidate_tier in (3, 0) and candidate.kind in (
            "straight", "pair_chain", "triple_chain", "straight_flush"):
        return candidate.main_rank > incumbent.main_rank
    return rank_strength(candidate.main_rank, level) > rank_strength(incumbent.main_rank, level)


def validate_action(action: Action, hand, level: int) -> bool:
    """Validate physical IDs, wildcard interpretation, and declared shape.

    A different physical copy of the same face is accepted, even though the
    enumerator emits one canonical representative.
    """
    try:
        if not isinstance(action, Action) or action.kind not in _KINDS:
            return False
        if type(level) is not int or level not in _ORDINARY:
            return False
        if type(action.main_rank) is not int:
            return False
        cards = action.cards
        if not isinstance(cards, tuple) or not cards or len(set(cards)) != len(cards):
            return False
        for card in cards:
            validate_card(card)
        if not set(cards).issubset(set(hand)):
            return False
        if not isinstance(action.wildcards, tuple):
            return False
        wild_ids = {card for card in cards if is_wild(card, level)}
        declarations = {}
        for declaration in action.wildcards:
            if not isinstance(declaration, tuple) or len(declaration) != 3:
                return False
            card, value, color = declaration
            if card not in wild_ids or card in declarations:
                return False
            if type(value) is not int or value not in _ORDINARY:
                return False
            if type(color) is not int or (color not in range(4) if action.kind == "straight_flush"
                                                 else color != -1):
                return False
            declarations[card] = (value, color)
        if wild_ids != set(declarations):
            if not (action.kind == "single" and len(cards) == 1 and wild_ids and
                    not declarations):
                return False
        if action.kind == "single" and wild_ids and declarations:
            if declarations[cards[0]] != (level, -1):
                return False

        effective_ranks = [declarations[card][0] if card in declarations else rank(card)
                           for card in cards]
        counts = Counter(effective_ranks)
        n = len(cards)
        kind = action.kind
        if kind == "single":
            return n == 1 and action.main_rank == effective_ranks[0]
        if kind in ("pair", "triple"):
            return n == (2 if kind == "pair" else 3) and len(counts) == 1 and \
                action.main_rank == effective_ranks[0]
        if kind == "full_house":
            return n == 5 and sorted(counts.values()) == [2, 3] and \
                action.main_rank == next(value for value, count in counts.items() if count == 3)
        if kind == "bomb":
            return 4 <= n <= 10 and len(counts) == 1 and \
                effective_ranks[0] in _ORDINARY and action.main_rank == effective_ranks[0]
        if kind == "joker_bomb":
            return n == 4 and counts == {16: 2, 17: 2} and \
                not declarations and action.main_rank == 17
        if kind in ("straight", "pair_chain", "triple_chain", "straight_flush"):
            length, multiplicity = {
                "straight": (5, 1), "pair_chain": (3, 2),
                "triple_chain": (2, 3), "straight_flush": (5, 1),
            }[kind]
            if n != length * multiplicity or len(counts) != length or \
                    any(count != multiplicity for count in counts.values()):
                return False
            if action.main_rank != _RUN_MAX[length].get(tuple(sorted(counts))):
                return False
            if kind == "straight_flush":
                effective_suits = [declarations[card][1] if card in declarations
                                   else suit(card) for card in cards]
                return len(set(effective_suits)) == 1
            return True
        return False
    except (TypeError, ValueError, KeyError):
        return False
