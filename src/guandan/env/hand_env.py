"""Auditable, single-hand Guandan state machine.

Only ``observe`` and ``legal_actions`` are intended as strategy inputs. The
state, digest and replay contain hidden information for research tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from random import Random

from guandan.rules.actions import beats, enumerate_actions, validate_action
from guandan.rules.cards import DECK, validate_card
from guandan.types import (
    ACTION_VERSION, RULES_VERSION, Action, GameEvent, PASS,
    PlayerObservation, RulesConfig, Settlement, StepResult,
)


@dataclass(frozen=True, slots=True)
class GameState:
    """Full research state.

    ``finish_order`` contains only players who emptied their hands. A normal
    settlement additionally assigns fourth place to the sole remaining hand;
    a double-down leaves both remaining players unranked.
    """

    initial_hands: tuple[tuple[int, ...], ...]
    hands: tuple[tuple[int, ...], ...]
    played: tuple[int, ...]
    initial_level: int
    starting_player: int
    current_player: int | None
    last_action: Action | None
    last_player: int | None
    passed_players: tuple[int, ...]
    finish_order: tuple[int, ...]
    history: tuple[GameEvent, ...]
    version: int
    terminal: bool
    settlement: Settlement | None
    research_fixture: bool


def _seat(value: int, label: str) -> None:
    if type(value) is not int or value not in range(4):
        raise ValueError(f"{label} must be a seat in 0..3")


def _level(value: int) -> None:
    if type(value) is not int or value not in range(2, 15):
        raise ValueError("initial_level must be an integer in 2..14")


def _hands(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("hands must contain exactly four hands")
    result = []
    seen: set[int] = set()
    for hand in value:
        if not isinstance(hand, (list, tuple)) or not hand:
            raise ValueError("every hand must be a nonempty sequence")
        for card in hand:
            validate_card(card)
            if card in seen:
                raise ValueError("physical card IDs must be disjoint")
            seen.add(card)
        result.append(tuple(sorted(hand)))
    return tuple(result)


def _action_dict(action: Action | None) -> dict | None:
    if action is None:
        return None
    return {
        "kind": action.kind,
        "cards": list(action.cards),
        "main_rank": action.main_rank,
        "wildcards": [list(w) for w in action.wildcards],
    }


def _event_dict(event: GameEvent) -> dict:
    return {
        "version": event.version,
        "player": event.player,
        "action": _action_dict(event.action),
        "trick_closed": event.trick_closed,
        "finished": list(event.finished),
    }


def _settlement_dict(value: Settlement | None) -> dict | None:
    if value is None:
        return None
    return {
        "winner_team": value.winner_team,
        "finish_order": list(value.finish_order),
        "remaining_players": list(value.remaining_players),
        "level_gain": value.level_gain,
        "team_rewards": list(value.team_rewards),
    }


def _parse_action(value: object) -> Action:
    if type(value) is not dict or set(value) != {"kind", "cards", "main_rank", "wildcards"}:
        raise ValueError("invalid replay action fields")
    if type(value["kind"]) is not str or type(value["main_rank"]) is not int:
        raise ValueError("invalid replay action types")
    cards, wildcards = value["cards"], value["wildcards"]
    if type(cards) is not list or any(type(c) is not int for c in cards):
        raise ValueError("invalid replay cards")
    if type(wildcards) is not list or any(
        type(w) is not list or len(w) != 3 or any(type(v) is not int for v in w)
        for w in wildcards
    ):
        raise ValueError("invalid replay wildcard declarations")
    return Action(value["kind"], tuple(cards), value["main_rank"],
                  tuple(tuple(w) for w in wildcards))


def _same_json(actual: object, expected: object) -> bool:
    """Compare replay JSON without Python's bool == int equivalence."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (actual.keys() == expected.keys()
                and all(_same_json(actual[key], value)
                        for key, value in expected.items()))
    if type(expected) is list:
        return (len(actual) == len(expected)
                and all(_same_json(a, b) for a, b in zip(actual, expected)))
    return actual == expected


class HandEnv:
    def __init__(self) -> None:
        self._state: GameState | None = None

    @property
    def state(self) -> GameState:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    @classmethod
    def from_hands(
        cls, hands: list[list[int]] | tuple[tuple[int, ...], ...],
        initial_level: int = 2, starting_player: int = 0,
    ) -> HandEnv:
        """Create a research fixture with four disjoint, nonempty hands."""
        _level(initial_level)
        _seat(starting_player, "starting_player")
        initial = _hands(hands)
        env = cls()
        env._state = GameState(
            initial, initial, (), initial_level, starting_player,
            starting_player, None, None, (), (), (), 0, False, None,
            sum(map(len, initial)) != 108 or any(len(h) != 27 for h in initial),
        )
        env.assert_invariants()
        return env

    def reset(
        self, seed: int, rules_config: RulesConfig | None = None,
        initial_level: int = 2, starting_player: int = 0,
    ) -> PlayerObservation:
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        if rules_config is not None and type(rules_config) is not RulesConfig:
            raise TypeError("rules_config must be RulesConfig or None")
        _level(initial_level)
        _seat(starting_player, "starting_player")
        deck = list(DECK)
        Random(seed).shuffle(deck)
        initial = _hands([deck[i * 27:(i + 1) * 27] for i in range(4)])
        candidate = GameState(
            initial, initial, (), initial_level, starting_player,
            starting_player, None, None, (), (), (), 0, False, None, False,
        )
        self._assert_state(candidate)
        self._state = candidate
        return self.observe(starting_player)

    def observe(self, player_id: int) -> PlayerObservation:
        _seat(player_id, "player_id")
        s = self.state
        return PlayerObservation(
            player_id=player_id, hand=s.hands[player_id], level=s.initial_level,
            current_player=s.current_player,
            remaining_counts=tuple(len(h) for h in s.hands),
            finish_order=s.finish_order, last_action=s.last_action,
            last_player=s.last_player, passed_players=s.passed_players,
            history=s.history, state_version=s.version, terminal=s.terminal,
            settlement=s.settlement,
        )

    def legal_actions(self, player_id: int) -> list[Action]:
        _seat(player_id, "player_id")
        s = self.state
        if s.terminal or player_id != s.current_player:
            raise ValueError("legal actions are available only to the current player")
        actions = enumerate_actions(s.hands[player_id], s.initial_level)
        if s.last_action is None:
            return list(actions)
        return [PASS, *(action for action in actions
                        if beats(action, s.last_action, s.initial_level))]

    def step(
        self, player_id: int, action: Action,
        state_version: int | None = None,
    ) -> StepResult:
        _seat(player_id, "player_id")
        s = self.state
        if state_version is not None:
            if type(state_version) is not int:
                raise TypeError("state_version must be an integer or None")
            if state_version != s.version:
                raise ValueError("stale state version")
        if s.terminal or player_id != s.current_player:
            raise ValueError("action is not from the current active player")
        if type(action) is not Action:
            raise TypeError("action must be an Action")
        self._check_action_shape(action)
        if action == PASS:
            if s.last_action is None:
                raise ValueError("cannot pass on a free lead")
            passed = tuple(sorted((*s.passed_players, player_id)))
            active = {p for p in range(4) if s.hands[p]}
            waiting = active - ({s.last_player} if s.last_player in active else set())
            closed = waiting.issubset(passed)
            if closed:
                leader = s.last_player if s.last_player in active else (
                    (s.last_player + 2) % 4
                    if (s.last_player + 2) % 4 in active else
                    self._next_active(s.hands, s.last_player)
                )
                candidate = replace(s, current_player=leader, last_action=None,
                                    last_player=None, passed_players=())
            else:
                candidate = replace(s, current_player=self._next_active(s.hands, player_id),
                                    passed_players=passed)
            finished = ()
        else:
            if not validate_action(action, s.hands[player_id], s.initial_level):
                raise ValueError("invalid declared action or physical cards")
            if s.last_action is not None and not beats(action, s.last_action, s.initial_level):
                raise ValueError("action does not beat incumbent")
            new_hand = tuple(c for c in s.hands[player_id] if c not in action.cards)
            hands = list(s.hands)
            hands[player_id] = new_hand
            finish_order = s.finish_order + ((player_id,) if not new_hand else ())
            finished = (player_id,) if not new_hand else ()
            settlement = self._settle(finish_order, tuple(hands))
            candidate = replace(
                s, hands=tuple(hands), played=s.played + tuple(action.cards),
                current_player=None if settlement else self._next_active(tuple(hands), player_id),
                last_action=action, last_player=player_id, passed_players=(),
                finish_order=finish_order, terminal=settlement is not None,
                settlement=settlement,
            )
            closed = False
        event = GameEvent(s.version + 1, player_id, action, closed, finished)
        candidate = replace(candidate, version=s.version + 1, history=s.history + (event,))
        self._assert_state(candidate)  # Commit only after all checks have succeeded.
        self._state = candidate
        return StepResult(candidate.current_player, (event,), candidate.terminal,
                          candidate.settlement, candidate.version)

    @staticmethod
    def _check_action_shape(action: Action) -> None:
        if type(action.kind) is not str or type(action.main_rank) is not int:
            raise ValueError("invalid action fields")
        if type(action.cards) is not tuple or any(type(c) is not int for c in action.cards):
            raise ValueError("action cards must be integer tuple")
        if type(action.wildcards) is not tuple or any(
            type(w) is not tuple or len(w) != 3 or any(type(v) is not int for v in w)
            for w in action.wildcards
        ):
            raise ValueError("invalid wildcard declarations")

    @staticmethod
    def _next_active(hands: tuple[tuple[int, ...], ...], after: int) -> int:
        for distance in range(1, 5):
            seat = (after + distance) % 4
            if hands[seat]:
                return seat
        raise AssertionError("no active player")

    @staticmethod
    def _settle(
        finished: tuple[int, ...], hands: tuple[tuple[int, ...], ...],
    ) -> Settlement | None:
        if len(finished) < 2:
            return None
        double_down = finished[0] % 2 == finished[1] % 2
        if not double_down and len(finished) < 3:
            return None
        winner = finished[0] % 2
        remaining = tuple(p for p in range(4) if hands[p])
        final_order = finished if double_down else finished + remaining
        teammate = (finished[0] + 2) % 4
        teammate_place = final_order.index(teammate) + 1
        gain = {2: 3, 3: 2, 4: 1}[teammate_place]
        rewards = (1, -1) if winner == 0 else (-1, 1)
        return Settlement(winner, final_order, remaining, gain, rewards)

    def state_digest(self) -> str:
        return self._digest(self.state)

    @staticmethod
    def _digest(s: GameState) -> str:
        payload = {
            "rules_version": RULES_VERSION, "action_version": ACTION_VERSION,
            "initial_hands": s.initial_hands, "hands": s.hands,
            "played": s.played, "initial_level": s.initial_level,
            "starting_player": s.starting_player,
            "current_player": s.current_player,
            "last_action": _action_dict(s.last_action), "last_player": s.last_player,
            "passed_players": s.passed_players, "finish_order": s.finish_order,
            "history": [_event_dict(e) for e in s.history], "version": s.version,
            "terminal": s.terminal, "settlement": _settlement_dict(s.settlement),
            "research_fixture": s.research_fixture,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        return sha256(encoded).hexdigest()

    def assert_invariants(self) -> None:
        self._assert_state(self.state)

    @staticmethod
    def _assert_state(s: GameState) -> None:
        assert len(s.hands) == len(s.initial_hands) == 4
        original = [c for hand in s.initial_hands for c in hand]
        current = [c for hand in s.hands for c in hand] + list(s.played)
        assert len(original) == len(set(original))
        assert len(current) == len(set(current))
        assert set(original) == set(current), "physical card conservation"
        assert all(0 <= c < 108 for c in current)
        assert s.research_fixture == (len(original) != 108 or
                                      any(len(h) != 27 for h in s.initial_hands))
        assert s.version == len(s.history)
        assert all(e.version == i + 1 for i, e in enumerate(s.history))
        assert len(set(s.finish_order)) == len(s.finish_order)
        assert all(not s.hands[p] for p in s.finish_order)
        assert set(s.passed_players) <= {p for p in range(4) if s.hands[p]}
        assert len(set(s.passed_players)) == len(s.passed_players)
        assert (s.last_action is None) == (s.last_player is None)
        if s.last_action is None:
            assert not s.passed_players
        if s.terminal:
            assert s.current_player is None and s.settlement is not None
            assert s.settlement == HandEnv._settle(s.finish_order, s.hands)
        else:
            assert s.settlement is None and s.current_player in range(4)
            assert s.hands[s.current_player]
            assert HandEnv._settle(s.finish_order, s.hands) is None

    def serialize_replay(self) -> dict:
        s = self.state
        replay = {
            "visibility": "research_full", "rules_version": RULES_VERSION,
            "action_version": ACTION_VERSION,
            "initial_hands": [list(h) for h in s.initial_hands],
            "initial_level": s.initial_level,
            "starting_player": s.starting_player,
            "research_fixture": s.research_fixture,
            "initial_digest": "",
            "steps": [],
            "final_digest": self.state_digest(),
            "final_settlement": _settlement_dict(s.settlement),
            "final_terminal": s.terminal,
        }
        clone = HandEnv.from_hands(s.initial_hands, s.initial_level, s.starting_player)
        replay["initial_digest"] = clone.state_digest()
        for event in s.history:
            clone.step(event.player, event.action, state_version=clone.state.version)
            replay["steps"].append({
                "player": event.player, "action": _action_dict(event.action),
                "state_version": event.version - 1,
                "event": _event_dict(event), "digest": clone.state_digest(),
                "terminal": clone.state.terminal,
                "settlement": _settlement_dict(clone.state.settlement),
            })
        assert replay["final_digest"] == clone.state_digest()
        return replay

    @classmethod
    def replay(cls, replay_dict: dict) -> HandEnv:
        if type(replay_dict) is not dict or set(replay_dict) != {
            "visibility", "rules_version", "action_version", "initial_hands",
            "initial_level", "starting_player", "research_fixture",
            "initial_digest", "steps", "final_digest", "final_settlement",
            "final_terminal",
        }:
            raise ValueError("invalid replay fields")
        if (replay_dict["visibility"] != "research_full"
                or replay_dict["rules_version"] != RULES_VERSION
                or replay_dict["action_version"] != ACTION_VERSION):
            raise ValueError("unsupported replay visibility or version")
        env = cls.from_hands(replay_dict["initial_hands"],
                             replay_dict["initial_level"],
                             replay_dict["starting_player"])
        if (type(replay_dict["research_fixture"]) is not bool
                or replay_dict["research_fixture"] != env.state.research_fixture
                or replay_dict["initial_digest"] != env.state_digest()):
            raise ValueError("replay initial state mismatch")
        steps = replay_dict["steps"]
        if type(steps) is not list:
            raise ValueError("replay steps must be a list")
        for row in steps:
            if type(row) is not dict or set(row) != {
                "player", "action", "state_version", "event", "digest",
                "terminal", "settlement",
            }:
                raise ValueError("invalid replay step fields")
            if type(row["player"]) is not int or type(row["state_version"]) is not int:
                raise ValueError("invalid replay player or version")
            action = _parse_action(row["action"])
            result = env.step(row["player"], action, row["state_version"])
            if (not _same_json(row["event"], _event_dict(result.events[0]))
                    or row["digest"] != env.state_digest()
                    or type(row["terminal"]) is not bool
                    or row["terminal"] != result.terminal
                    or not _same_json(row["settlement"],
                                      _settlement_dict(result.settlement))):
                raise ValueError("replay step verification failed")
        if (replay_dict["final_digest"] != env.state_digest()
                or type(replay_dict["final_terminal"]) is not bool
                or replay_dict["final_terminal"] != env.state.terminal
                or not _same_json(replay_dict["final_settlement"],
                                  _settlement_dict(env.state.settlement))):
            raise ValueError("replay final state mismatch")
        return env
