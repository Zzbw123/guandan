"""Controller-owned evaluation contracts, frozen before implementation."""
from dataclasses import dataclass

EVALUATION_VERSION = "gd-evaluation-v1"
POLICIES = ("random", "greedy", "team")


@dataclass(frozen=True, slots=True)
class Matchup:
    focal: str
    teammate: str
    opponent: str

    def __post_init__(self):
        if any(p not in POLICIES for p in (self.focal, self.teammate, self.opponent)):
            raise ValueError("Unknown baseline policy")

    @property
    def key(self) -> str:
        return f"{self.focal}|{self.teammate}|{self.opponent}"


@dataclass(frozen=True, slots=True)
class Trial:
    matchup_id: str
    deal_index: int
    deal_seed: int
    level: int
    rotation: int
    swap: int
    flip: int
    focal_seat: int
    policies: tuple[str, ...]
    starting_player: int = 0

    @property
    def trial_id(self) -> str:
        return f"{self.matchup_id}:{self.deal_seed}:{self.rotation}:{self.swap}:{self.flip}"

    @property
    def focal_team(self) -> int:
        return self.focal_seat % 2
