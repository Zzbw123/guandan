"""Complete-candidate heuristic-score regression; targets are not probabilities.

Only player-visible observations and supplied legal actions enter this module.
Candidate completeness remains the caller's obligation; none are pruned here.
"""
from __future__ import annotations

import math
from guandan.agents import GreedyAgent
from guandan.learning.encoding import (
    _action_valid, _validate_observation, encode_action, encode_observation,
)
from guandan.types import Action, PlayerObservation


def teacher_targets(obs: PlayerObservation, legal: list[Action]) -> list[float]:
    """Normalize every GreedyAgent score in supplied order to [-0.8, 0.8]."""
    _validate_observation(obs)
    if type(legal) is not list or not legal:
        raise ValueError("legal must be a nonempty complete candidate list")
    for action in legal:
        if type(action) is not Action:
            raise TypeError("every candidate must be Action")
        if not _action_valid(action) or not set(action.cards).issubset(obs.hand):
            raise ValueError("candidate must be structurally valid and use the player's cards")
    teacher = GreedyAgent()
    scores = [float(teacher.score(obs, action)) for action in legal]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("nonfinite teacher score")
    low, high = min(scores), max(scores)
    if low == high:
        return [0.0] * len(scores)
    return [1.6 * ((score - low) / (high - low)) - 0.8 for score in scores]


def fit_batch(model, optimizer, requests: list[tuple[PlayerObservation, list[Action]]],
              chunk_size: int = 1024) -> dict:
    """One CUDA update minimizing the mean of per-observation candidate MSEs.

Independent forward graphs bound activation memory while gradients accumulate
over every supplied candidate. There is no sampling, shuffle, or candidate cap.
PyTorch is imported only here so CPU target generation needs no torch runtime.
"""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if type(requests) is not list or not requests:
        raise ValueError("requests must be a nonempty list")
    import torch
    from guandan.learning.model import DMCNetwork
    if not isinstance(model, DMCNetwork):
        raise TypeError("model must be DMCNetwork")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer")
    parameters = list(model.parameters())
    device = parameters[0].device

    def checked(tensor, label, float32=True):
        if tensor.device.type != "cuda" or tensor.device != device:
            raise RuntimeError(f"{label} must be on the model CUDA device")
        if float32 and tensor.dtype != torch.float32:
            raise RuntimeError(f"{label} must be float32")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"nonfinite {label}")

    for parameter in parameters:
        checked(parameter, "model parameter")
        if not parameter.requires_grad:
            raise ValueError("all model parameters must require gradients")
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len(parameters) or {id(p) for p in owned} != {id(p) for p in parameters}:
        raise ValueError("optimizer must own exactly the model parameters once")
    # Adam's default scalar step is CPU; require its explicit CUDA-state mode.
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        if any(not (group.get("capturable") or group.get("fused"))
               for group in optimizer.param_groups):
            raise ValueError("Adam requires capturable=True or fused=True for CUDA tensor states")

    def check_optimizer_states():
        count = 0
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    checked(value, "optimizer tensor state")
                    count += 1
        return count

    check_optimizer_states()
    states, actions, indices, targets, weights, sizes = [], [], [], [], [], []
    for index, request in enumerate(requests):
        if type(request) is not tuple or len(request) != 2:
            raise ValueError("request must be (PlayerObservation, legal list)")
        obs, legal = request
        values = teacher_targets(obs, legal)
        states.append(encode_observation(obs))
        actions.extend(encode_action(action) for action in legal)
        indices.extend([index] * len(legal))
        targets.extend(values)
        weights.extend([1.0 / (len(requests) * len(legal))] * len(legal))
        sizes.append(len(legal))
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.float32, device=device)
    indices = torch.tensor(indices, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.float32, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    for tensor, label in ((states, "states"), (actions, "actions"),
                          (targets, "targets"), (weights, "weights")):
        checked(tensor, label)
    checked(indices, "request indices", float32=False)
    optimizer.zero_grad(set_to_none=True)
    detached_loss = torch.zeros((), dtype=torch.float32, device=device)
    forward_checks = 0
    for start in range(0, len(actions), chunk_size):
        end = min(start + chunk_size, len(actions))
        output = model(states.index_select(0, indices[start:end]), actions[start:end])
        checked(output, "forward output")
        if output.shape != targets[start:end].shape:
            raise ValueError("forward output must have one score per candidate")
        loss = ((output - targets[start:end]).square() * weights[start:end]).sum()
        checked(loss, "chunk loss")
        loss.backward()
        detached_loss += loss.detach()
        forward_checks += 1
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("missing model gradient")
        checked(parameter.grad, "model gradient")
        checked(parameter, "model parameter before update")
    check_optimizer_states()
    optimizer.step()
    for parameter in parameters:
        checked(parameter, "updated model parameter")
    state_checks = check_optimizer_states()
    checked(detached_loss, "batch loss")
    return {
        "loss": float(detached_loss.item()), "requests": len(requests),
        "scored_candidates": len(actions), "chunks": forward_checks,
        "per_request_sizes": sizes,
        "device_proof": {"device": str(device), "dtype": "float32",
                         "forward_checks": forward_checks,
                         "gradient_checks": len(parameters),
                         "parameter_checks": len(parameters),
                         "adam_tensor_checks": state_checks,
                         "optimizer_tensor_checks": state_checks},
    }
