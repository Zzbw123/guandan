"""P3l detached, globally reduced gradient weighting of full centered batches."""
import math
import torch
from experiments.p3j_objective import centered_backward

ZERO_THRESHOLD = 1e-12
TARGET_RATIO = .1
ALPHA_CAP = 1.


def gradient_metrics(dmc, centered, objective='normcap'):
    """Pure detached all-parameter reduction, also used by independent vector tests."""
    if objective not in ('constant', 'normcap'):
        raise ValueError('P3l objective must be constant or normcap')
    if not dmc or len(dmc) != len(centered):
        raise ValueError('complete paired gradient lists required')
    ds = cs = dot = None
    for gd, gc in zip(dmc, centered):
        if (not isinstance(gd, torch.Tensor) or not isinstance(gc, torch.Tensor)
                or gd.shape != gc.shape or gd.device != gc.device or gd.dtype != gc.dtype
                or not torch.isfinite(gd).all().item() or not torch.isfinite(gc).all().item()):
            raise ValueError('finite matching gradients required')
        a, b = gd.detach().to(torch.float64), gc.detach().to(torch.float64)
        x, y, z = a.square().sum(), b.square().sum(), (a*b).sum()
        ds = x if ds is None else ds+x
        cs = y if cs is None else cs+y
        dot = z if dot is None else dot+z
    d, c, product = ds.sqrt().item(), cs.sqrt().item(), dot.item()
    if not all(math.isfinite(x) for x in (d, c, product)):
        raise ValueError('nonfinite gradient reduction')
    zero_d, zero_c = d <= ZERO_THRESHOLD, c <= ZERO_THRESHOLD
    raw = None if zero_d or zero_c else TARGET_RATIO*d/c
    alpha = TARGET_RATIO if objective == 'constant' else (0. if raw is None else min(ALPHA_CAP, raw))
    return dict(d=d, c=c, alpha=alpha,
                auxiliary_to_dmc_ratio=None if zero_d else alpha*c/d,
                cosine=None if zero_d or zero_c else product/(d*c),
                cap_triggered=objective == 'normcap' and raw is not None and raw >= ALPHA_CAP,
                joint_descent_factor=None if zero_d else 1.+alpha*product/(d*d),
                alpha_branch=('constant' if objective == 'constant' else
                              'zero_dmc' if zero_d else 'zero_centered' if zero_c else
                              'cap' if raw >= ALPHA_CAP else 'ratio'),
                zero_dmc=zero_d, zero_centered=zero_c)


def _snapshot(parameters):
    result = []
    for p in parameters:
        if p.grad is None or not torch.isfinite(p.grad).all().item():
            raise ValueError('missing or nonfinite gradient')
        result.append(p.grad.detach().clone())
    return result


def joint_backward(model, requests, objective='normcap', chunk_size=1024):
    """Consume existing DMC gradients and leave a single joint gradient for Adam.

    Constant restores the original DMC buffers and calls the exact historical
    weighted accumulation. The separate unweighted pass exists only for logging
    and consumes no RNG. Normcap combines detached full parameter gradients.
    """
    if objective not in ('constant', 'normcap'):
        raise ValueError('P3l objective must be constant or normcap')
    parameters = list(model.parameters())
    dmc = _snapshot(parameters)
    for p in parameters:
        p.grad = None
    auxiliary = centered_backward(model, requests, 1., chunk_size)
    centered = _snapshot(parameters)
    metrics = gradient_metrics(dmc, centered, objective)
    if objective == 'constant':
        for p, gd in zip(parameters, dmc):
            p.grad = gd
        centered_backward(model, requests, .1, chunk_size)
    else:
        alpha = metrics['alpha']  # Python float obtained only from detached reductions.
        for p, gd, gc in zip(parameters, dmc, centered):
            p.grad = gd + alpha*gc
    _snapshot(parameters)
    return {**auxiliary, **metrics,
            'candidate_sizes': [len(legal) for _, legal in requests],
            'state_denominator': len(requests),
            'single_candidate_requests': sum(len(legal) == 1 for _, legal in requests)}
