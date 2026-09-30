"""Add lambda times observation-mean full-candidate MSE gradients; no Adam step."""
import math
from experiments.p3f_teacher import teacher_targets
from guandan.learning.encoding import encode_observation, encode_action

def auxiliary_backward(model, requests, weight=.1, chunk_size=1024):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError('finite nonnegative auxiliary weight required')
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('positive integer chunk_size required')
    if type(requests) is not list or not requests:
        raise ValueError('nonempty observation/candidate requests required')
    import torch
    device = next(model.parameters()).device
    if device.type != 'cuda': raise ValueError('CUDA required')
    states, actions, indices, targets, weights = [], [], [], [], []
    for i, (obs, legal) in enumerate(requests):
        target = teacher_targets(obs, legal)
        states.append(encode_observation(obs))
        actions.extend(encode_action(a) for a in legal)
        indices.extend([i]*len(legal)); targets.extend(target)
        weights.extend([1/(len(requests)*len(legal))]*len(legal))
    st = torch.tensor(states, dtype=torch.float32, device=device)
    ac = torch.tensor(actions, dtype=torch.float32, device=device)
    ix = torch.tensor(indices, dtype=torch.long, device=device)
    ta = torch.tensor(targets, dtype=torch.float32, device=device)
    wt = torch.tensor(weights, dtype=torch.float32, device=device)
    total = torch.zeros((), device=device)
    for start in range(0,len(actions),chunk_size):
        sl=slice(start,start+chunk_size)
        output=model(st.index_select(0,ix[sl]),ac[sl])
        loss=((output-ta[sl]).square()*wt[sl]).sum()
        if not torch.isfinite(loss).item(): raise ValueError('nonfinite auxiliary loss')
        (weight*loss).backward()
        total += loss.detach()
    return dict(loss=float(total.item()),candidates=len(actions),requests=len(requests))
