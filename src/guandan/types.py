"""Versioned contracts owned by the controller.

Only PlayerObservation and legal Action objects may cross into an agent.
"""
from dataclasses import asdict, dataclass

RULES_VERSION = "gd-hand-v1"
OBSERVATION_VERSION = "gd-observation-v1"
ACTION_VERSION = "gd-action-v1"


@dataclass(frozen=True, slots=True)
class RulesConfig:
    version: str = RULES_VERSION

    def __post_init__(self):
        if self.version != RULES_VERSION:
            raise ValueError(f"Unsupported rules version: {self.version}")


@dataclass(frozen=True, slots=True)
class Action:
    kind: str
    cards: tuple[int, ...] = ()
    main_rank: int = 0
    # (physical wildcard ID, declared natural rank, declared suit).
    # suit = -1 for rank-only combinations; sequence flushes use 0..3.
    wildcards: tuple[tuple[int, int, int], ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Action":
        return cls(value["kind"], tuple(value["cards"]), value["main_rank"],
                   tuple(tuple(x) for x in value.get("wildcards", ())))


PASS = Action("pass")


@dataclass(frozen=True, slots=True)
class GameEvent:
    version: int
    player: int
    action: Action
    trick_closed: bool
    finished: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Settlement:
    winner_team: int
    # Actual finishers only. In a double-down, losers remain unranked.
    finish_order: tuple[int, ...]
    remaining_players: tuple[int, ...]
    level_gain: int
    team_rewards: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    player_id: int
    hand: tuple[int, ...]
    level: int
    current_player: int | None
    remaining_counts: tuple[int, ...]
    finish_order: tuple[int, ...]
    last_action: Action | None
    last_player: int | None
    passed_players: tuple[int, ...]
    history: tuple[GameEvent, ...]
    state_version: int
    terminal: bool
    settlement: Settlement | None
    rules_version: str = RULES_VERSION
    observation_version: str = OBSERVATION_VERSION


@dataclass(frozen=True, slots=True)
class StepResult:
    next_player: int | None
    events: tuple[GameEvent, ...]
    terminal: bool
    settlement: Settlement | None
    state_version: int
