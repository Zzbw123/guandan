"""Stable two-deck physical card IDs, independent of level and player."""

SUITS = ("S", "H", "C", "D")
RANK_NAMES = {**{r: str(r) for r in range(2, 11)}, 11: "J", 12: "Q", 13: "K", 14: "A", 16: "BJ", 17: "RJ"}
DECK = tuple(range(108))


def validate_card(card: int) -> None:
    if type(card) is not int or not 0 <= card < 108:
        raise ValueError(f"Invalid physical card ID: {card!r}")


def rank(card: int) -> int:
    validate_card(card)
    face = card % 54
    return face % 13 + 2 if face < 52 else face - 36


def suit(card: int) -> int:
    validate_card(card)
    face = card % 54
    return face // 13 if face < 52 else -1


def card_id(value: int, color: int = 0, copy: int = 0) -> int:
    if type(copy) is not int or copy not in (0, 1):
        raise ValueError("copy must be 0 or 1")
    if type(value) is not int or value not in (*range(2, 15), 16, 17):
        raise ValueError("rank must be 2..14, 16 (small joker), or 17 (big joker)")
    if value >= 16:
        return copy * 54 + value + 36
    if type(color) is not int or color not in range(4):
        raise ValueError("suit must be 0..3")
    return copy * 54 + color * 13 + value - 2


def is_wild(card: int, level: int) -> bool:
    return rank(card) == level and suit(card) == 1


def rank_strength(value: int, level: int) -> int:
    return 15 if value == level else value


def label(card: int) -> str:
    r, s = rank(card), suit(card)
    return f"{SUITS[s] if s >= 0 else ''}{RANK_NAMES[r]}#{card // 54 + 1}"
