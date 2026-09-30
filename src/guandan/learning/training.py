"""CPU DMC self-play over complete hands and terminal team rewards.

The policy sees only player observations and their complete legal action lists.
The environment's full state and replay are used after play for research audit.
"""

from __future__ import annotations

import math
import random

import torch

from guandan.env import HandEnv
from guandan.learning.encoding import (
    ACTION_DIM, STATE_DIM, encode_action, encode_observation,
)
from guandan.learning.model import DMCNetwork, score_actions


_CONFIG_KEYS = frozenset({"seed", "epsilon", "lr", "batch_size", "chunk_size"})
_MAX_STEPS = 1000


def _validated_config(config: dict) -> dict:
    if type(config) is not dict or set(config) != _CONFIG_KEYS:
        raise ValueError(f"config must contain exactly {sorted(_CONFIG_KEYS)}")
    for key in ("seed", "batch_size", "chunk_size"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if config["seed"] < 0:
        raise ValueError("seed must be nonnegative")
    for key in ("batch_size", "chunk_size"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("epsilon", "lr"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= config["epsilon"] <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if config["lr"] <= 0:
        raise ValueError("lr must be positive")
    return dict(config)


def _finite_features(values: tuple[float, ...], size: int, label: str) -> tuple[float, ...]:
    if len(values) != size or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must have {size} finite values")
    return values


class Trainer:
    """One network for all four seats; no parameter update during a hand."""

    def __init__(self, config: dict) -> None:
        self.config = _validated_config(config)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(self.config["seed"])
        self.model = DMCNetwork().cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.rng = random.Random(self.config["seed"])
        self.episodes = 0
        self.updates = 0
        self.used_deal_seeds: list[int] = []
        self.ready_for_checkpoint = True

    def train_episode(self, deal_seed: int, level: int, starting_player: int) -> dict:
        if not self.ready_for_checkpoint:
            raise RuntimeError("trainer is not at an episode boundary")
        if type(deal_seed) is not int or not 100000 <= deal_seed <= 109999:
            raise ValueError("deal_seed must be in development range 100000..109999")
        if type(level) is not int or not 2 <= level <= 14:
            raise ValueError("level must be an integer in 2..14")
        if type(starting_player) is not int or not 0 <= starting_player <= 3:
            raise ValueError("starting_player must be an integer in 0..3")

        self.ready_for_checkpoint = False
        env = HandEnv()
        observation = env.reset(deal_seed, initial_level=level,
                                starting_player=starting_player)
        trajectory: list[tuple[tuple[float, ...], tuple[float, ...], int]] = []
        settlement = None
        for steps in range(1, _MAX_STEPS + 1):
            player = observation.player_id
            actions = env.legal_actions(player)
            if not actions:
                raise RuntimeError("current player has no legal actions")
            state_features = _finite_features(
                encode_observation(observation), STATE_DIM, "state features",
            )
            if self.rng.random() < self.config["epsilon"]:
                selected = self.rng.randrange(len(actions))
            else:
                with torch.no_grad():
                    scores = score_actions(
                        self.model, observation, actions,
                        chunk_size=self.config["chunk_size"],
                    )
                if scores.ndim != 1 or len(scores) != len(actions) or not torch.isfinite(scores).all().item():
                    raise ValueError("action scores must be finite and cover every legal action")
                selected = int(torch.argmax(scores).item())  # First listed action wins ties.
            action = actions[selected]
            action_features = _finite_features(
                encode_action(action), ACTION_DIM, "action features",
            )
            trajectory.append((state_features, action_features, player % 2))
            result = env.step(player, action, state_version=observation.state_version)
            if result.terminal:
                settlement = result.settlement
                break
            if result.next_player is None:
                raise RuntimeError("nonterminal step did not name the next player")
            observation = env.observe(result.next_player)
        else:
            raise RuntimeError(f"hand exceeded {_MAX_STEPS} steps")

        if settlement is None or settlement.team_rewards not in ((1, -1), (-1, 1)):
            raise RuntimeError("terminal hand has invalid team rewards")
        terminal_digest = env.state_digest()
        replay = env.serialize_replay()
        verified = HandEnv.replay(replay)
        if (not verified.state.terminal or verified.state.settlement != settlement
                or verified.state_digest() != terminal_digest
                or replay["final_digest"] != terminal_digest
                or len(replay["steps"]) != steps):
            raise RuntimeError("terminal research replay verification failed")

        # Targets are assigned from each decision maker's own team. Shuffle with
        # the dedicated policy RNG, independently of the deal RNG in HandEnv.
        order = list(range(len(trajectory)))
        self.rng.shuffle(order)
        weighted_loss = 0.0
        for offset in range(0, len(order), self.config["batch_size"]):
            batch = [trajectory[index] for index in order[offset:offset + self.config["batch_size"]]]
            states = torch.tensor([item[0] for item in batch], dtype=torch.float32)
            actions = torch.tensor([item[1] for item in batch], dtype=torch.float32)
            targets = torch.tensor(
                [settlement.team_rewards[item[2]] for item in batch], dtype=torch.float32,
            )
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(states, actions)
            if predictions.shape != targets.shape or not torch.isfinite(predictions).all().item():
                raise ValueError("model predictions must be finite and match targets")
            loss = torch.nn.functional.mse_loss(predictions, targets)
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite training loss")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
                   for parameter in self.model.parameters()):
                raise ValueError("nonfinite or missing training gradient")
            self.optimizer.step()
            if any(not torch.isfinite(parameter).all().item()
                   for parameter in self.model.parameters()):
                raise ValueError("nonfinite model parameter after update")
            weighted_loss += float(loss.item()) * len(batch)
            self.updates += 1

        mean_loss = weighted_loss / len(trajectory)
        if not math.isfinite(mean_loss):
            raise ValueError("nonfinite mean training loss")
        self.episodes += 1
        self.used_deal_seeds.append(deal_seed)
        self.ready_for_checkpoint = True
        return {
            "episode": self.episodes,
            "seed": deal_seed,
            "level": level,
            "steps": steps,
            "samples": len(trajectory),
            "mean_loss": mean_loss,
            "team_rewards": list(settlement.team_rewards),
            "terminal_digest": terminal_digest,
            "updates": self.updates,
            "replay_verified": True,
        }
