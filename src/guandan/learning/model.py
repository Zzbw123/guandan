"""CPU DMC action-value network and complete-candidate scoring."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from guandan.learning.encoding import ACTION_DIM, STATE_DIM, encode_action, encode_observation
from guandan.types import Action, PlayerObservation

NETWORK_VERSION = "gd-dmc-network-v1"


class DMCNetwork(nn.Module):
    """Shared-seat state/action value network with a bounded scalar output."""

    def __init__(self) -> None:
        super().__init__()
        self.state_fc = nn.Linear(STATE_DIM, 128)
        self.action_fc = nn.Linear(ACTION_DIM, 128, bias=False)
        self.hidden_fc = nn.Linear(128, 64)
        self.output_fc = nn.Linear(64, 1)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        if (states.ndim != 2 or states.shape[1] != STATE_DIM
                or actions.ndim != 2 or actions.shape[1] != ACTION_DIM
                or states.shape[0] != actions.shape[0]):
            raise ValueError("states/actions must be aligned [B, STATE_DIM/ACTION_DIM]")
        hidden = torch.relu(self.state_fc(states) + self.action_fc(actions))
        hidden = torch.relu(self.hidden_fc(hidden))
        return torch.tanh(self.output_fc(hidden)).squeeze(-1)


def score_actions(
    model: DMCNetwork,
    obs: PlayerObservation,
    actions: list[Action] | tuple[Action, ...],
    chunk_size: int = 256,
) -> Tensor:
    """Score every supplied candidate, projecting the observation once."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(model, DMCNetwork):
        raise TypeError("model must be DMCNetwork")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("score_actions requires a CPU model")
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("actions must be a nonempty complete candidate sequence")
    if any(type(action) is not Action for action in actions):
        raise TypeError("all candidates must be Action objects")
    state = torch.tensor((encode_observation(obs),), dtype=torch.float32)
    pieces: list[Tensor] = []
    with torch.no_grad():
        state_projection = model.state_fc(state)
        for start in range(0, len(actions), chunk_size):
            encoded = tuple(encode_action(action) for action in actions[start:start + chunk_size])
            batch = torch.tensor(encoded, dtype=torch.float32)
            hidden = torch.relu(state_projection + model.action_fc(batch))
            hidden = torch.relu(model.hidden_fc(hidden))
            values = torch.tanh(model.output_fc(hidden)).squeeze(-1)
            if not torch.isfinite(values).all().item():
                raise ValueError("non-finite action score")
            pieces.append(values)
    return torch.cat(pieces).cpu()


class DMCAgent:
    """Greedy inference agent; exploration is owned by the trainer."""

    def __init__(self, model: DMCNetwork, chunk_size: int = 256) -> None:
        if not isinstance(model, DMCNetwork):
            raise TypeError("model must be DMCNetwork")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.model = model
        self.chunk_size = chunk_size

    def act(self, obs: PlayerObservation, actions: list[Action] | tuple[Action, ...]) -> Action:
        scores = score_actions(self.model, obs, actions, self.chunk_size)
        return actions[int(torch.argmax(scores).item())]
