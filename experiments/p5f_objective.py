"""Complete-candidate teacher pretraining objectives for the existing DMC net.

Only player-visible observations and their supplied legal actions enter here.
The caller is responsible for supplying the complete legal list; this module
keeps every candidate, including distinct actions with identical encodings.
"""
from __future__ import annotations

import math

import torch

from guandan.agents import GreedyAgent
from guandan.learning.encoding import (
    _action_valid, _validate_observation, encode_action, encode_observation,
)
from guandan.learning.model import DMCNetwork
from guandan.types import Action, PlayerObservation


def raw_scores(model: DMCNetwork, states: torch.Tensor,
               actions: torch.Tensor) -> torch.Tensor:
    """DMC hidden layers with unbounded output logits (no final tanh)."""
    if not isinstance(model, DMCNetwork):
        raise TypeError("model must be DMCNetwork")
    if (states.ndim != 2 or states.shape[1] != model.state_fc.in_features
            or actions.ndim != 2 or actions.shape[1] != model.action_fc.in_features
            or states.shape[0] != actions.shape[0]):
        raise ValueError("states/actions must be aligned feature matrices")
    hidden = torch.relu(model.state_fc(states) + model.action_fc(actions))
    return model.output_fc(torch.relu(model.hidden_fc(hidden))).squeeze(-1)


def teacher_labels(obs: PlayerObservation, legal: list[Action]) -> dict:
    """Return exact GreedyAgent top ties and min-max regression labels."""
    _validate_observation(obs)
    if type(legal) is not list or not legal:
        raise ValueError("legal must be a nonempty candidate list")
    for action in legal:
        if type(action) is not Action:
            raise TypeError("every candidate must be Action")
        if not _action_valid(action) or not set(action.cards).issubset(obs.hand):
            raise ValueError("invalid candidate or cards outside player's hand")
    teacher = GreedyAgent()
    scores = [float(teacher.score(obs, action)) for action in legal]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("nonfinite teacher score")
    by_encoding = {}
    for action, score in zip(legal, scores):
        encoded = encode_action(action)
        previous = by_encoding.setdefault(encoded, score)
        if previous != score:
            raise ValueError("identically encoded candidates have different teacher scores")
    low, high = min(scores), max(scores)
    targets = ([0.0] * len(scores) if low == high else
               [1.6 * ((value - low) / (high - low)) - 0.8 for value in scores])
    if not all(math.isfinite(value) for value in targets):
        raise ValueError("nonfinite teacher target")
    return {"scores": scores, "targets": targets,
            "optimal": [value == high for value in scores]}


def _check_tensor(tensor: torch.Tensor, label: str, device: torch.device,
                  *, cuda: bool = True) -> None:
    if tensor.device != device or (cuda and tensor.device.type != "cuda"):
        raise RuntimeError(f"{label} must be on the model CUDA device")
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"{label} must be float32")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"nonfinite {label}")


def _check_model(model: DMCNetwork, *, cuda: bool,
                 require_grad: bool = True) -> tuple[list, torch.device]:
    if not isinstance(model, DMCNetwork):
        raise TypeError("model must be DMCNetwork")
    parameters = list(model.parameters())
    device = parameters[0].device
    for parameter in parameters:
        _check_tensor(parameter, "model parameter", device, cuda=cuda)
        if require_grad and not parameter.requires_grad:
            raise ValueError("all model parameters must require gradients")
    return parameters, device


def _check_optimizer(optimizer, parameters, device):
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError("optimizer must be Adam or AdamW")
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len(parameters) or {id(p) for p in owned} != {id(p) for p in parameters}:
        raise ValueError("optimizer must own exactly the model parameters once")
    if any(not (group.get("capturable") or group.get("fused"))
           for group in optimizer.param_groups):
        raise ValueError("Adam requires capturable=True or fused=True")

    def check_states():
        count = 0
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    _check_tensor(value, "optimizer tensor state", device)
                    count += 1
        return count
    check_states()
    return check_states


def _validate_requests(requests, *, with_labels=True):
    if type(requests) is not list or not requests:
        raise ValueError("requests must be a nonempty list")
    prepared = []
    for request in requests:
        if type(request) is not tuple or len(request) != 2:
            raise ValueError("request must be (PlayerObservation, legal list)")
        obs, legal = request
        if with_labels:
            labels = teacher_labels(obs, legal)
        else:
            _validate_observation(obs)
            if type(legal) is not list or not legal:
                raise ValueError("legal must be a nonempty candidate list")
            for action in legal:
                if type(action) is not Action:
                    raise TypeError("every candidate must be Action")
                if not _action_valid(action) or not set(action.cards).issubset(obs.hand):
                    raise ValueError("invalid candidate or cards outside player's hand")
            labels = None
        state = encode_observation(obs)
        actions = [encode_action(action) for action in legal]
        prepared.append((state, actions, labels))
    return prepared


def _chunks(actions, size):
    for start in range(0, len(actions), size):
        yield start, min(start + size, len(actions))


def _forward_chunk(model, state, encoded, start, end, device):
    actions = torch.tensor(encoded[start:end], dtype=torch.float32, device=device)
    states = state.expand(end - start, -1)
    result = raw_scores(model, states, actions)
    _check_tensor(result, "raw scores", device, cuda=False)
    if result.shape != (end - start,):
        raise ValueError("forward output must have one score per candidate")
    return result


def score_requests(model: DMCNetwork, requests, chunk_size: int = 1024):
    """Return one complete CPU logit tensor per observation, in input order."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    _, device = _check_model(model, cuda=False, require_grad=False)
    prepared = _validate_requests(requests, with_labels=False)
    output = []
    with torch.no_grad():
        for state, encoded, _ in prepared:
            state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
            parts = [_forward_chunk(model, state_tensor, encoded, start, end, device).cpu()
                     for start, end in _chunks(encoded, chunk_size)]
            output.append(torch.cat(parts))
    return output


def fit_batch(model: DMCNetwork, optimizer, requests,
              objective: str, chunk_size: int = 1024) -> dict:
    """One CUDA Adam update with equal observation weights and bounded graphs."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if objective not in ("regression", "ranking"):
        raise ValueError("objective must be regression or ranking")
    parameters, device = _check_model(model, cuda=True)
    check_states = _check_optimizer(optimizer, parameters, device)
    prepared = _validate_requests(requests)
    states = torch.tensor([state for state, _, _ in prepared],
                          dtype=torch.float32, device=device)
    sizes = [len(encoded) for _, encoded, _ in prepared]
    flat_actions = [action for _, actions, _ in prepared for action in actions]
    indices = torch.tensor([i for i, size in enumerate(sizes) for _ in range(size)],
                           dtype=torch.long, device=device)
    _check_tensor(states, "states", device)

    def forward(start, end):
        actions = torch.tensor(flat_actions[start:end], dtype=torch.float32, device=device)
        _check_tensor(actions, "actions", device)
        output = raw_scores(model, states.index_select(0, indices[start:end]), actions)
        _check_tensor(output, "raw scores", device)
        if output.shape != (end - start,):
            raise ValueError("forward output must have one score per candidate")
        return output

    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    chunks = 0
    top1_matches = 0
    if objective == "ranking":
        with torch.no_grad():
            logits = torch.cat([forward(start, end)
                                for start, end in _chunks(flat_actions, chunk_size)])
            coefficients = torch.zeros_like(logits)
            offset = 0
            for size, (_, _, labels) in zip(sizes, prepared):
                view = logits[offset:offset + size]
                optimal = torch.tensor(labels["optimal"], dtype=torch.bool, device=device)
                top1_matches += int(bool(optimal[int(torch.argmax(view).item())].item()))
                if not bool(optimal.all().item()):
                    all_log = torch.logsumexp(view, 0)
                    optimal_log = torch.logsumexp(view[optimal], 0)
                    loss = all_log - optimal_log
                    _check_tensor(loss, "ranking loss", device)
                    total_loss += loss / len(prepared)
                    coefficient_view = coefficients[offset:offset + size]
                    coefficient_view.copy_(torch.exp(view - all_log))
                    coefficient_view[optimal] -= torch.exp(view[optimal] - optimal_log)
                offset += size
            coefficients /= len(prepared)
            _check_tensor(coefficients, "ranking coefficients", device)
        for start, end in _chunks(flat_actions, chunk_size):
            forward(start, end).backward(coefficients[start:end])
            chunks += 1
    else:
        targets = torch.tensor([target for _, _, labels in prepared
                                for target in labels["targets"]],
                               dtype=torch.float32, device=device)
        weights = torch.tensor([1.0 / (len(prepared) * size)
                                for size in sizes for _ in range(size)],
                               dtype=torch.float32, device=device)
        _check_tensor(targets, "targets", device)
        _check_tensor(weights, "weights", device)
        for start, end in _chunks(flat_actions, chunk_size):
            output = forward(start, end)
            loss = ((torch.tanh(output) - targets[start:end]).square()
                    * weights[start:end]).sum()
            _check_tensor(loss, "regression loss", device)
            loss.backward()
            total_loss += loss.detach()
            chunks += 1
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("missing model gradient")
        _check_tensor(parameter.grad, "model gradient", device)
    _check_tensor(total_loss, "batch loss", device)
    check_states()
    optimizer.step()
    for parameter in parameters:
        _check_tensor(parameter, "updated model parameter", device)
    state_checks = check_states()
    return {"loss": float(total_loss.item()), "requests": len(prepared),
            "scored_candidates": len(flat_actions), "chunks": chunks,
            "per_request_sizes": sizes, "top1_matches": top1_matches if objective == "ranking" else None,
            "multi_requests": sum(size > 1 for size in sizes),
            "device_proof": {"device": str(device), "dtype": "float32",
                             "forward_checks": chunks, "gradient_checks": len(parameters),
                             "parameter_checks": len(parameters),
                             "adam_tensor_checks": state_checks,
                             "optimizer_tensor_checks": state_checks}}
