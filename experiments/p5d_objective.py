"""Terminal-label objective for the P5d GPU training intervention."""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch


OBJECTIVES = ("ordinary", "label_balanced")


def objective_loss(predictions: torch.Tensor, targets: torch.Tensor, objective: str):
    """Return a differentiable scalar loss and JSON-ready minibatch statistics.

    The caller supplies exactly the observed terminal labels. No examples or
    rewards are synthesized, and a one-class minibatch uses ordinary MSE.
    """
    if type(objective) is not str or objective not in OBJECTIVES:
        raise ValueError("objective must be ordinary or label_balanced")
    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError("predictions and targets must be tensors")
    for name, value in (("predictions", predictions), ("targets", targets)):
        if (value.device.type != "cuda" or value.dtype != torch.float32
                or value.ndim != 1 or value.numel() == 0
                or not torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be nonempty finite 1-D CUDA float32")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must use the same CUDA device")
    if targets.requires_grad:
        raise ValueError("targets must not require gradients")
    positive_mask = targets == 1.0
    negative_mask = targets == -1.0
    positive = int(positive_mask.sum().item())
    negative = int(negative_mask.sum().item())
    n = targets.numel()
    if positive + negative != n:
        raise ValueError("targets must contain only terminal labels +1 and -1")
    single_class = positive == 0 or negative == 0
    if objective == "ordinary" or single_class:
        # Keep this exact PyTorch call for bitwise compatibility with P5a.
        loss = torch.nn.functional.mse_loss(predictions, targets)
    else:
        loss = (0.5 * torch.mean((predictions[positive_mask] - 1.0).square())
                + 0.5 * torch.mean((predictions[negative_mask] + 1.0).square()))
    if not torch.isfinite(loss).item():
        raise ValueError("nonfinite objective loss")
    if objective == "ordinary" or single_class:
        positive_weight = 1.0 if positive else 0.0
        negative_weight = 1.0 if negative else 0.0
    else:
        positive_weight = n / (2 * positive)
        negative_weight = n / (2 * negative)
    stats = dict(objective=objective, n=n, positive=positive, negative=negative,
                 positive_weight=positive_weight, negative_weight=negative_weight,
                 single_class=single_class, loss=float(loss.item()))
    return loss, stats


# A concise alias for callers that treat the module itself as the objective API.
compute_objective = objective_loss
