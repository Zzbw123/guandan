"""Controller dense autograd oracle, independently normalized teacher targets.

Before the study: mixed complete public states and all three inherited weights.
After the study: the first actual shuffled training batch of each of six jobs.
No environment/seed/terminal data are passed to the model or teacher.
"""
from pathlib import Path
import sys, json, math, copy, gzip
from random import Random
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from guandan.env import HandEnv
from guandan.types import Action
from guandan.agents import GreedyAgent
from guandan.learning.encoding import encode_observation,encode_action
from experiments.p3l_training import GPUTrainer
from experiments.p3l_objective import joint_backward
from experiments.p3e_common import read,write,digest,check

def dense_losses(model, batch):
    req=[x[0] for x in batch]; device='cuda'
    st=torch.tensor([encode_observation(o) for o,l in req],device=device)
    ac=torch.tensor([encode_action(x[1]) for x in batch],device=device)
    y=torch.tensor([float(x[2]) for x in batch],device=device)
    dmc=(model(st,ac)-y).square().mean()
    sizes=[len(l) for o,l in req]
    all_actions=torch.tensor([encode_action(a) for o,l in req for a in l],device=device)
    indices=torch.tensor([i for i,n in enumerate(sizes) for _ in range(n)],device=device)
    scores=model(st.index_select(0,indices),all_actions)
    terms=[]
    for (obs,legal),values in zip(req,scores.split(sizes)):
        raw=[float(GreedyAgent().score(obs,a)) for a in legal]
        lo,hi=min(raw),max(raw)
        targets=torch.tensor([0. if hi==lo else 1.6*((v-lo)/(hi-lo))-.8 for v in raw],device=device)
        # Differentiating both means is independent of the production detached
        # global-mean/two-forward accumulation.
        residual=(values-values.mean())-(targets-targets.mean())
        terms.append(residual.square().mean())
    return dmc,torch.stack(terms).mean()

def compare(seed,arm,batch,chunk=1024,logged=None):
    cfg=dict(seed=seed,epsilon=.1,lr=.001,batch_size=256,chunk_size=chunk,num_envs=4,objective=arm,auxiliary_weight=.1)
    t=GPUTrainer(cfg)
    source=ROOT/f'artifacts/evaluations/p3f-teacher-v1/training/teacher-{seed}/phase1/raw.pt'
    t.model.load_state_dict(torch.load(source,map_location='cpu',weights_only=True)['model'])
    oracle=copy.deepcopy(t.model)
    ld,lc=dense_losses(oracle,batch);params=list(oracle.parameters())
    gd=torch.autograd.grad(ld,params,retain_graph=True)
    gc=torch.autograd.grad(lc,params,retain_graph=True)
    d=math.sqrt(math.fsum(float(g.double().square().sum().item()) for g in gd))
    c=math.sqrt(math.fsum(float(g.double().square().sum().item()) for g in gc))
    dot=math.fsum(float((a.double()*b.double()).sum().item()) for a,b in zip(gd,gc))
    alpha=.1 if arm=='constant' else 0. if min(d,c)<=1e-12 else min(1.,.1*d/c)
    (ld+alpha*lc).backward()
    states=torch.tensor([encode_observation(x[0][0]) for x in batch],device='cuda')
    actions=torch.tensor([encode_action(x[1]) for x in batch],device='cuda')
    targets=torch.tensor([float(x[2]) for x in batch],device='cuda')
    loss=(t.model(states,actions)-targets).square().mean();loss.backward()
    record=joint_backward(t.model,[x[0] for x in batch],arm,chunk)
    check(record['candidate_sizes']==[len(x[0][1]) for x in batch], 'complete per-state candidates')
    for key,expected in [('d',d),('c',c),('alpha',alpha),('loss',lc.item())]:
        check(math.isclose(record[key],expected,rel_tol=3e-4,abs_tol=3e-6), 'dense metric '+key)
    if min(d,c)>1e-12:
        check(math.isclose(record['cosine'],dot/(d*c),rel_tol=3e-4,abs_tol=3e-6),'dense cosine')
    error=0.
    for a,b in zip(t.model.parameters(),params):
        error=max(error,float((a.grad-b.grad).abs().max().item()))
        torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-6)
    oa=torch.optim.Adam(oracle.parameters(),lr=.001,capturable=True)
    t.optimizer.step();oa.step()
    adam_error=0.
    for a,b in zip(t.model.parameters(),params):
        adam_error=max(adam_error,float((a-b).abs().max().item()))
        torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-6)
        for key in ('exp_avg','exp_avg_sq'):
            torch.testing.assert_close(t.optimizer.state[a][key],oa.state[b][key],rtol=3e-4,atol=3e-6)
    if logged is not None:
        for key in record:
            expected=record[key]
            field={'loss':'auxiliary_loss','candidates':'auxiliary_candidates','requests':'auxiliary_requests'}.get(key,key)
            value=logged[field]
            if type(expected) is float:check(math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-12),'real first batch metric '+key)
            else:check(value==expected,'real first batch denominator '+key)
        check(math.isclose(logged['dmc_loss'],loss.item(),abs_tol=1e-12),'real first batch DMC loss')
    return dict(seed=seed,arm=arm,states=len(batch),candidates=sum(len(x[0][1]) for x in batch),
        chunk=chunk,max_gradient_abs_error=error,max_adam_parameter_abs_error=adam_error,
        dense_d=d,dense_c=c,dense_alpha=alpha,teacher_sha256=digest(source))

def actual_batch(root,job):
    folder=root/'training'/job
    with gzip.open(folder/'replays.jsonl.gz','rt',encoding='utf-8') as f:
        rows=[json.loads(next(f)) for _ in range(4)]
    trajectories=[]
    for row in rows:
        env=HandEnv();env.reset(row['seed'],initial_level=row['level'],starting_player=row['starting_player'])
        path=[]
        for s in row['replay']['steps']:
            obs=env.observe(s['player']);legal=env.legal_actions(obs.player_id);a=Action.from_dict(s['action'])
            path.append(((obs,legal),a,s['player']%2))
            env.step(obs.player_id,a,state_version=s['state_version'])
        reward=env.state.settlement.team_rewards
        trajectories.append([(req,a,reward[team]) for req,a,team in path])
    rng=Random(int(job.split('-')[1]))
    for i in range(max(map(len,trajectories))):
        for path in trajectories:
            if i<len(path) and rng.random()<.1:rng.randrange(len(path[i][0][1]))
    samples=[r for path in trajectories for r in path];rng.shuffle(samples)
    wave=json.loads((folder/'waves.jsonl').read_text('utf-8').splitlines()[0])
    return samples[:256],wave['batches'][0]

def main():
    out=Path(sys.argv[1]);check(not out.exists(),'fresh controller receipt required')
    records=[]
    if len(sys.argv)>2:
        root=Path(sys.argv[2])
        from experiments.p3l_protocol import specification
        for job in specification()['training_order']:
            arm,seed=job.split('-');batch,logged=actual_batch(root,job)
            records.append(compare(int(seed),arm,batch,logged=logged))
            print('dense actual first batch',job,flush=True)
        scope='six actual first shuffled 256-observation batches; all legal candidates; not every historical update'
    else:
        env=HandEnv();env.reset(109180);batch=[]
        while not env.state.terminal and len(batch)<12:
            obs=env.observe(env.state.current_player);legal=env.legal_actions(obs.player_id)
            action=GreedyAgent().act(obs,legal)
            batch.append(((obs,legal),action,1 if len(batch)%2 else -1))
            env.step(obs.player_id,action,state_version=obs.state_version)
        for seed in (314380,314381,314382):
            for arm in ('constant','normcap'):
                for chunk in (7,1024):records.append(compare(seed,arm,batch,chunk))
        scope='12 real development observations, complete candidates, three inherited teachers, two arms, two chunk sizes; synthetic +/-1 DMC labels for numerical engineering'
    write(out,dict(status='PASS',scope=scope,records=records,script_sha256=digest(Path(__file__)),
        tolerances=dict(gradient_rtol=3e-4,gradient_atol=3e-6,adam_rtol=3e-4,adam_atol=3e-6)))
    print(json.dumps(dict(status='PASS',cases=len(records))))
if __name__=='__main__':main()
