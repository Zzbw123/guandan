"""Read-only, full-candidate fixed-weight gradient probes. No optimizer exists here."""
import math
import torch
from guandan.learning.encoding import encode_observation, encode_action
from experiments.p3f_teacher import teacher_targets
from experiments.p3h_objective import auxiliary_backward
from experiments.p3j_objective import centered_backward

WEIGHT = .1
ZERO = 1e-12

def flat_grad(model):
    values = [p.grad.detach().clone().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
              for p in model.parameters()]
    result = torch.cat(values)
    if not torch.isfinite(result).all().item(): raise ValueError('nonfinite gradient')
    return result

def tensors(model, obs, legal, action, reward):
    if type(reward) not in (int, float) or reward not in (-1, 1): raise ValueError('terminal team reward must be +/-1')
    if not legal or action not in legal: raise ValueError('executed action must be legal')
    device = next(model.parameters()).device
    if device.type != 'cuda': raise ValueError('CUDA required')
    state = torch.tensor([encode_observation(obs)], device=device, dtype=torch.float32)
    actions = torch.tensor([encode_action(a) for a in legal], device=device, dtype=torch.float32)
    target = torch.tensor(teacher_targets(obs, legal), device=device, dtype=torch.float32)
    return state, actions, target, legal.index(action)

def production(model, obs, legal, action, reward, chunk_size=1024):
    st, ac, target, ix = tensors(model, obs, legal, action, reward)
    model.zero_grad(set_to_none=True)
    loss = (model(st, ac[ix:ix+1])[0] - reward).square()
    loss.backward(); grads = [flat_grad(model)]; losses = [loss.item()]
    for fn in (auxiliary_backward, centered_backward):
        model.zero_grad(set_to_none=True)
        result = fn(model, [(obs, legal)], weight=1., chunk_size=chunk_size)
        if result['candidates'] != len(legal): raise ValueError('candidate denominator')
        grads.append(flat_grad(model)); losses.append(result['loss'])
    with torch.no_grad():
        q = torch.cat([model(st.expand(len(ac[s:s+chunk_size]), -1), ac[s:s+chunk_size])
                       for s in range(0, len(ac), chunk_size)])
    model.zero_grad(set_to_none=True)
    return grads, losses, q, target

def oracle(model, obs, legal, action, reward):
    """Independent dense autograd expression, including differentiable global mean."""
    st, ac, target, ix = tensors(model, obs, legal, action, reward)
    # Spell out layers to cross-check the model.forward and chunked loss paths.
    hidden = torch.relu(torch.nn.functional.linear(st, model.state_fc.weight, model.state_fc.bias)
                        + torch.nn.functional.linear(ac, model.action_fc.weight))
    hidden = torch.relu(torch.nn.functional.linear(hidden, model.hidden_fc.weight, model.hidden_fc.bias))
    q = torch.tanh(torch.nn.functional.linear(hidden, model.output_fc.weight, model.output_fc.bias)).flatten()
    residual = q-target
    losses = [(q[ix]-reward).square(), residual.square().mean(),
              ((q-q.mean())-(target-target.mean())).square().mean()]
    grads = []
    for i, loss in enumerate(losses):
        g = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=i<2)
        grads.append(torch.cat([v.reshape(-1) for v in g]).detach())
    return grads, [v.item() for v in losses], q.detach(), target

def geometry(gram):
    """Rows/columns: DMC, absolute, centered; dot products before weight 0.1."""
    if len(gram)!=3 or any(len(r)!=3 for r in gram) or not all(math.isfinite(v) for r in gram for v in r):
        raise ValueError('finite 3x3 Gram matrix')
    norms = [math.sqrt(max(0., gram[i][i])) for i in range(3)]
    out = dict(dmc_norm=norms[0])
    for i, name in ((1,'absolute'), (2,'centered')):
        denom=norms[0]*norms[i]
        cos = None if min(norms[0],norms[i])<=ZERO else max(-1., min(1.,gram[0][i]/denom))
        ratio = None if norms[0]<=ZERO else WEIGHT*norms[i]/norms[0]
        out[name] = dict(norm=norms[i],cosine=cos,weighted_norm_ratio=ratio,
            conflict=None if cos is None else cos<0,
            # Dot(g_D, g_D + .1 g_A) / ||g_D||^2: negative reverses local DMC descent.
            dmc_descent_factor=None if norms[0]<=ZERO else 1+WEIGHT*gram[0][i]/gram[0][0])
    return out

def gram_matrix(grads):
    doubles=[g.double() for g in grads]
    return [[torch.dot(a,b).item() for b in doubles] for a in doubles]

def summarize_probe(model, grads, losses, q, target, executed):
    gram=gram_matrix(grads); blocks={};cursor=0
    for name,p in model.named_parameters():
        n=p.numel();blocks[name]=gram_matrix([g[cursor:cursor+n] for g in grads]);cursor+=n
    values=q.double().cpu().tolist(); targets=target.double().cpu().tolist()
    chosen=max(range(len(values)),key=values.__getitem__);ordered=sorted(targets,reverse=True)
    span=max(values)-min(values)
    return dict(losses=dict(zip(('dmc','absolute','centered'),losses)),gram=gram,
        parameter_grams=blocks,geometry=geometry(gram),
        candidate_count=len(values),singleton=len(values)==1,
        teacher_top_gap=None if len(values)==1 else ordered[0]-ordered[1],
        teacher_top_ties=sum(t==max(targets) for t in targets),
        model_top_gap=None if len(values)==1 else sorted(values,reverse=True)[0]-sorted(values,reverse=True)[1],
        model_span=span,top1=int(targets[chosen]==max(targets)),
        teacher_regret=max(targets)-targets[chosen],
        executed_q=values[executed],executed_teacher_target=targets[executed],
        saturation_fraction=sum(abs(qv)>=.95 for qv in values)/len(values),
        tanh_slope_mean=sum(1-qv*qv for qv in values)/len(values))
