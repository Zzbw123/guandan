from collections import Counter
import random
from typing import Protocol, Sequence

from guandan.rules.cards import rank, rank_strength
from guandan.types import Action, PlayerObservation


class Agent(Protocol):
    version: str

    def act(self, observation: PlayerObservation, legal_actions: Sequence[Action]) -> Action: ...


class RandomAgent:
    version = "random-v1"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, observation: PlayerObservation, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("Agent requires at least one legal action")
        return self._rng.choice(legal_actions)


class GreedyAgent:
    version = "greedy-v1"

    def score(self, observation: PlayerObservation, action: Action) -> float:
        if action.kind == "pass":
            return -5.0
        if len(action.cards) == len(observation.hand):
            return 1000.0
        counts = Counter(rank(c) for c in observation.hand if c not in action.cards)
        structure = sum(n // 2 for n in counts.values()) * 0.25
        bomb_cost = 12.0 if action.kind in ("bomb", "straight_flush", "joker_bomb") else 0.0
        return 5.0 * len(action.cards) + structure - bomb_cost - rank_strength(action.main_rank, observation.level) / 100

    def act(self, observation: PlayerObservation, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("Agent requires at least one legal action")
        return max(legal_actions, key=lambda a: self.score(observation, a))


class TeamHeuristicAgent(GreedyAgent):
    version = "team-heuristic-v1"

    def score(self, observation: PlayerObservation, action: Action) -> float:
        base = super().score(observation, action)
        mate = (observation.player_id + 2) % 4
        mate_leads = observation.last_player == mate
        opponents = [(observation.player_id + 1) % 4, (observation.player_id + 3) % 4]
        threatened = any(0 < observation.remaining_counts[p] <= 2 for p in opponents)
        if action.kind == "pass":
            return 20.0 if mate_leads else (-20.0 if threatened else base)
        if len(action.cards) == len(observation.hand):
            return base
        if mate_leads:
            base -= 18.0
        if threatened:
            base += rank_strength(action.main_rank, observation.level) / 2
        return base


def make_agent(name: str, seed: int = 0) -> Agent:
    factories = {"random": lambda: RandomAgent(seed), "greedy": GreedyAgent, "team": TeamHeuristicAgent}
    if name not in factories:
        raise ValueError(f"Unknown baseline: {name}")
    return factories[name]()
