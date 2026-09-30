"""Fixed gd-features-v1 encodings of player-visible information only."""

from __future__ import annotations

from guandan.rules.cards import rank, suit, validate_card
from guandan.types import Action, GameEvent, OBSERVATION_VERSION, PlayerObservation, RULES_VERSION

FEATURE_VERSION = "gd-features-v1"
ACTION_DIM = 149
STATE_DIM = 2993
_KINDS = (
    "pass", "single", "pair", "triple", "full_house", "straight", "pair_chain",
    "triple_chain", "bomb", "straight_flush", "joker_bomb",
)
_SEATS = range(4)
_ZERO_ACTION = (0.0,) * ACTION_DIM
_ZERO_EVENT = (0.0,) * 159


def _seat(value: int) -> bool:
    return type(value) is int and value in _SEATS


def _action_valid(action: Action) -> bool:
    if type(action) is not Action or action.kind not in _KINDS:
        return False
    if action.kind == "pass":
        return action.cards == () and action.main_rank == 0 and action.wildcards == ()
    if (type(action.cards) is not tuple or not action.cards or len(action.cards) > 27
            or any(type(card) is not int for card in action.cards)
            or len(set(action.cards)) != len(action.cards)
            or type(action.main_rank) is not int
            or action.main_rank not in (*range(2, 15), 16, 17)
            or type(action.wildcards) is not tuple):
        return False
    try:
        for card in action.cards:
            validate_card(card)
        declared_cards: set[int] = set()
        for declaration in action.wildcards:
            if type(declaration) is not tuple or len(declaration) != 3:
                return False
            card, declared_rank, declared_suit = declaration
            if (type(card) is not int or card not in action.cards or card in declared_cards
                    or suit(card) != 1 or rank(card) not in range(2, 15)
                    or type(declared_rank) is not int or declared_rank not in range(2, 15)
                    or type(declared_suit) is not int or declared_suit not in range(-1, 4)):
                return False
            declared_cards.add(card)
        return True
    except (TypeError, ValueError):
        return False


def _encode_action(action: Action) -> tuple[float, ...]:
    faces = [0.0] * 54
    for card in action.cards:
        faces[card % 54] += 0.5
    kinds = [0.0] * len(_KINDS)
    kinds[_KINDS.index(action.kind)] = 1.0
    ranks = [0.0] * 18
    ranks[action.main_rank] = 1.0
    declarations = [0.0] * 65
    for _, declared_rank, declared_suit in action.wildcards:
        declarations[(declared_rank - 2) * 5 + declared_suit + 1] += 0.5
    return tuple((*faces, *kinds, *ranks, *declarations, len(action.cards) / 27.0))


def encode_action(action: Action) -> tuple[float, ...]:
    """Encode a structurally valid action without physical copy identifiers."""
    if not _action_valid(action):
        raise ValueError("action must be a structurally valid Action")
    return _encode_action(action)


def _validate_observation(obs: PlayerObservation) -> None:
    if type(obs) is not PlayerObservation:
        raise TypeError("observation must be PlayerObservation")
    if (obs.rules_version != RULES_VERSION
            or obs.observation_version != OBSERVATION_VERSION):
        raise ValueError("unsupported rules or observation version")
    if obs.terminal or obs.current_player != obs.player_id or not _seat(obs.player_id):
        raise ValueError("observation must belong to the active player")
    if type(obs.level) is not int or obs.level not in range(2, 15):
        raise ValueError("invalid level")
    if (type(obs.hand) is not tuple or not obs.hand
            or len(set(obs.hand)) != len(obs.hand)):
        raise ValueError("invalid player hand")
    for card in obs.hand:
        validate_card(card)
    if (type(obs.remaining_counts) is not tuple or len(obs.remaining_counts) != 4
            or any(type(n) is not int or not 0 <= n <= 27 for n in obs.remaining_counts)
            or obs.remaining_counts[obs.player_id] != len(obs.hand)):
        raise ValueError("invalid remaining counts")
    if (type(obs.finish_order) is not tuple or len(set(obs.finish_order)) != len(obs.finish_order)
            or any(not _seat(seat) for seat in obs.finish_order)
            or any(obs.remaining_counts[seat] != 0 for seat in obs.finish_order)):
        raise ValueError("invalid finish order")
    if (type(obs.passed_players) is not tuple
            or len(set(obs.passed_players)) != len(obs.passed_players)
            or any(not _seat(seat) for seat in obs.passed_players)):
        raise ValueError("invalid passed players")
    if (obs.last_action is None) != (obs.last_player is None):
        raise ValueError("last action and player must appear together")
    if obs.last_action is not None:
        if not _seat(obs.last_player) or not _action_valid(obs.last_action):
            raise ValueError("invalid last action or player")
        if obs.last_action.kind == "pass":
            raise ValueError("last action cannot be pass")
    if type(obs.history) is not tuple:
        raise ValueError("history must be a tuple")
    for event in obs.history:
        if (type(event) is not GameEvent or not _seat(event.player)
                or type(event.trick_closed) is not bool
                or type(event.finished) is not tuple
                or len(set(event.finished)) != len(event.finished)
                or any(not _seat(seat) for seat in event.finished)
                or not _action_valid(event.action)):
            raise ValueError("invalid public event")


def encode_observation(obs: PlayerObservation) -> tuple[float, ...]:
    """Encode a current player's observation; no research state is accepted."""
    _validate_observation(obs)
    player = obs.player_id
    result: list[float] = []

    hand = [0.0] * 54
    for card in obs.hand:
        hand[card % 54] += 0.5
    result.extend(hand)

    played = [0.0] * (4 * 54)
    for event in obs.history:
        seat = (event.player - player) % 4
        for card in event.action.cards:
            played[seat * 54 + card % 54] += 0.5
    result.extend(played)

    result.extend(obs.remaining_counts[(player + seat) % 4] / 27.0 for seat in _SEATS)
    finish_places = [0.0] * 4
    for place, seat in enumerate(obs.finish_order, start=1):
        finish_places[(seat - player) % 4] = place / 4.0
    result.extend(finish_places)
    result.extend(float((player + seat) % 4 in obs.passed_players) for seat in _SEATS)
    result.extend(float(obs.level == value) for value in range(2, 15))
    last_player = [0.0] * 5
    last_player[4 if obs.last_player is None else (obs.last_player - player) % 4] = 1.0
    result.extend(last_player)
    result.extend(_ZERO_ACTION if obs.last_action is None else _encode_action(obs.last_action))
    assert len(result) == 449

    recent = obs.history[-16:]
    result.extend(_ZERO_EVENT * (16 - len(recent)))
    for event in recent:
        seat = (event.player - player) % 4
        result.extend(float(seat == position) for position in _SEATS)
        result.extend(_encode_action(event.action))
        result.append(float(event.trick_closed))
        result.extend(float((player + position) % 4 in event.finished) for position in _SEATS)
        result.append(1.0)
    assert len(result) == STATE_DIM
    return tuple(result)
