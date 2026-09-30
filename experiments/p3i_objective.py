"""Centered complete-candidate auxiliary gradient prototype; never steps Adam.

Only observations and complete Action lists enter this API. The environment
owns completeness. On any failure discard gradients/reload the batch boundary.
"""
import math

from experiments.p3f_teacher import teacher_targets
from guandan.learning.encoding import encode_action, encode_observation


def decompose(scores, targets):
    """Float64/Python diagnostic; candidates equally weighted within a state."""
    q, t = list(scores), list(targets)
    if not q or len(q) != len(t) or not all(math.isfinite(x) for x in q + t):
        raise ValueError('nonempty matching finite score arrays required')
    n = len(q)
    error = [a-b for a, b in zip(q, t)]
    shift = math.fsum(error)/n
    mse = math.fsum(e*e for e in error)/n
    centered = math.fsum((e-shift)**2 for e in error)/n
    return dict(candidates=n, mean_error=shift, mse=mse, offset_mse=shift**2,
                centered_mse=centered, identity_residual=mse-shift**2-centered)


def centered_backward(model, requests, weight=.1, chunk_size=1024):
    """Two forwards, global per-state centering, bounded activation memory.

The no-grad first pass supplies a detached global residual mean. Its missing
derivative cancels because the centered residuals sum to zero. This is valid
for the deterministic DMCNetwork only, without intervening parameter updates.
"""
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError('finite nonnegative weight required')
    if type(chunk_size) is not int or not 1 <= chunk_size <= 65536:
        raise ValueError('integer chunk_size in 1..65536 required')
    if type(requests) is not list or not requests:
        raise ValueError('nonempty request list required')
    import torch
    from guandan.learning.model import DMCNetwork
    if type(model) is not DMCNetwork:
        raise TypeError('deterministic DMCNetwork required')
    device = next(model.parameters()).device
    for p in model.parameters():
        if p.device != device or device.type != 'cuda' or p.dtype != torch.float32:
            raise ValueError('CUDA float32 parameters required')
        if not p.requires_grad or not torch.isfinite(p).all().item():
            raise ValueError('finite trainable parameters required')

    # Encode and validate the entire request list before accumulating gradients.
    encoded = []
    for request in requests:
        if type(request) is not tuple or len(request) != 2:
            raise ValueError('request must be (observation, complete legal list)')
        obs, legal = request
        target = teacher_targets(obs, legal)
        encoded.append((encode_observation(obs), [encode_action(a) for a in legal], target))
    total, count = 0., 0
    for state, actions, targets in encoded:
        n = len(actions)
        count += n
        st = torch.tensor([state], dtype=torch.float32, device=device)
        ac = torch.tensor(actions, dtype=torch.float32, device=device)
        ta = torch.tensor(targets, dtype=torch.float32, device=device)
        with torch.no_grad():
            pieces = [model(st.expand(len(ac[start:start+chunk_size]), -1),
                            ac[start:start+chunk_size]) - ta[start:start+chunk_size]
                      for start in range(0, n, chunk_size)]
            residual = torch.cat(pieces)
            if not torch.isfinite(residual).all().item():
                raise ValueError('nonfinite candidate residual')
            mean = residual.mean()
            total += float((residual-mean).square().mean().item())/len(requests)
        if weight == 0:
            continue
        for start in range(0, n, chunk_size):
            part = ac[start:start+chunk_size]
            residual = model(st.expand(len(part), -1), part)-ta[start:start+chunk_size]-mean
            loss = residual.square().sum()/(len(requests)*n)
            if not torch.isfinite(loss).item():
                raise ValueError('nonfinite centered auxiliary loss')
            (weight*loss).backward()
    return dict(loss=total, candidates=count, requests=len(requests))
