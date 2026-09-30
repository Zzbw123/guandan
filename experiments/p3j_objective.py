"""Batched deterministic two-pass centered complete-candidate gradients."""
import math
from experiments.p3f_teacher import teacher_targets
from guandan.learning.encoding import encode_action,encode_observation

def centered_backward(model,requests,weight=.1,chunk_size=1024):
    if type(weight) not in (int,float) or not math.isfinite(weight) or weight<0:raise ValueError('finite nonnegative weight')
    if type(chunk_size) is not int or not 1<=chunk_size<=65536:raise ValueError('integer chunk in 1..65536')
    if type(requests) is not list or not requests:raise ValueError('nonempty requests')
    import torch
    from guandan.learning.model import DMCNetwork
    if type(model) is not DMCNetwork:raise TypeError('deterministic DMCNetwork only')
    device=next(model.parameters()).device
    for p in model.parameters():
        if device.type!='cuda' or p.device!=device or p.dtype!=torch.float32 or not p.requires_grad or not torch.isfinite(p).all().item():
            raise ValueError('finite trainable CUDA float32 parameters')
    states,actions,targets,indices,sizes=[],[],[],[],[]
    for index,request in enumerate(requests):
        if type(request) is not tuple or len(request)!=2:raise ValueError('observation/legal request')
        obs,legal=request;target=teacher_targets(obs,legal)
        states.append(encode_observation(obs));actions.extend(encode_action(a) for a in legal)
        targets.extend(target);indices.extend([index]*len(legal));sizes.append(len(legal))
    st=torch.tensor(states,dtype=torch.float32,device=device)
    ac=torch.tensor(actions,dtype=torch.float32,device=device)
    ta=torch.tensor(targets,dtype=torch.float32,device=device)
    ix=torch.tensor(indices,dtype=torch.long,device=device)
    # Complete per-state segments, never per-chunk means or atomic scatter sums.
    with torch.no_grad():
        parts=[]
        for start in range(0,len(actions),chunk_size):
            sl=slice(start,start+chunk_size)
            parts.append(model(st.index_select(0,ix[sl]),ac[sl])-ta[sl])
        residual=torch.cat(parts)
        if not torch.isfinite(residual).all().item():raise ValueError('nonfinite full residual')
        segments=torch.split(residual,sizes)
        means=torch.stack([x.mean() for x in segments])
        weights=torch.tensor([1/(len(requests)*n) for n in sizes],device=device).index_select(0,ix)
        centered=residual-means.index_select(0,ix)
        total=(centered.square()*weights).sum().item()
    if weight:
        for start in range(0,len(actions),chunk_size):
            sl=slice(start,start+chunk_size)
            r=model(st.index_select(0,ix[sl]),ac[sl])-ta[sl]-means.index_select(0,ix[sl])
            loss=(r.square()*weights[sl]).sum()
            if not torch.isfinite(loss).item():raise ValueError('nonfinite centered loss')
            (weight*loss).backward()
    return dict(loss=total,candidates=len(actions),requests=len(requests))
