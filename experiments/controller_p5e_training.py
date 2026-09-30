"""Independent full training replay and sample/RNG audit; GPU first-wave numerics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
import gzip,json,math,random,copy
from itertools import islice
from experiments.controller_p5d_audit import audit_batch_rows
from collections import Counter
from hashlib import sha256
from guandan.env import HandEnv
from guandan.types import Action
from guandan.agents.baselines import GreedyAgent,TeamHeuristicAgent
from experiments.p5e_protocol import config
from experiments.p3e_common import check,read,digest

def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def payload(path):
    import torch
    m=read(path/'manifest.json');check(digest(path/'checkpoint.pt')==m['sha256'],'checkpoint bytes')
    p=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
    for k,v in m.items():
        if k not in ('sha256','bytes'):check(equal(p[k],v),'checkpoint metadata '+k)
    return p

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def audit_wave(w,index,cfg,rng,encode=False):
    from guandan.learning.encoding import encode_observation,encode_action
    check(w['objective']==cfg['objective'],'wave objective')
    check(w['wave']==w['learning_version']==index+1 and w['behavior_version']==index,'wave versions')
    check(w['episodes']==4*(index+1) and len(w['hands'])==4,'wave hand count')
    patterns=[['current','greedy','current','greedy'],['current','team','greedy','team'],
              ['current','frozen','team','frozen'],['current','greedy','frozen','team']]
    envs=[];observations=[];trajectories=[[] for _ in range(4)];counts=Counter()
    for i,h in enumerate(w['hands']):
        n=index*4+i;env=HandEnv();obs=env.reset(100000+n,initial_level=2+n%13,starting_player=n%4)
        expected=['current']*4 if cfg['mode']=='selfplay' else [patterns[(n//4)%4][(s-n%4)%4] for s in range(4)]
        check(h['roles']==expected,'fixed role schedule')
        check((h['seed'],h['level'],h['starting_player'])==(100000+n,2+n%13,n%4),'training deal schedule')
        replay=h['replay'];other=HandEnv.replay(replay)
        check(other.state.terminal and other.state_digest()==h['terminal_digest'],'authoritative full replay')
        check(replay['initial_hands']==[list(x) for x in env.state.initial_hands] and replay['initial_digest']==env.state_digest(),'seed to initial hands')
        check(h['steps']==len(replay['steps'])==len(h['decisions']) and h['replay_verified'] is True,'complete decision trace')
        envs.append(env);observations.append(obs)
    indices=[0]*4;score_batches=0
    while any(not e.state.terminal for e in envs):
        cycle=0
        for i,env in enumerate(envs):
            if env.state.terminal:continue
            step_index=indices[i];h=w['hands'][i];d=h['decisions'][step_index];s=h['replay']['steps'][step_index]
            obs=observations[i];legal=env.legal_actions(obs.player_id);role=h['roles'][obs.player_id]
            check(d['player']==s['player']==obs.player_id and d['role']==role,'actor routing')
            check(d['candidates']==len(legal),'complete candidates')
            selected=d['selected'];check(type(selected) is int and 0<=selected<len(legal),'selected index')
            action=Action.from_dict(s['action']);check(action==legal[selected],'selected action')
            check(type(d['sample']) is bool and d['sample']==(role=='current'),'sample ownership')
            explore=False
            if role=='current':
                explore=rng.random()<cfg['epsilon']
                if explore:check(rng.randrange(len(legal))==selected,'exploration index RNG')
                else:
                    counts['scored_candidates']+=len(legal);counts['score_requests']+=1;cycle+=len(legal)
                # Features are only required for independently recomputing the first-wave update.
                item=(encode_observation(obs),encode_action(action),obs.player_id%2) if encode else (i,step_index,obs.player_id%2)
                trajectories[i].append(item)
                counts['exploration']+=int(explore)
            elif role=='frozen':
                counts['frozen_candidates']+=len(legal);counts['frozen_requests']+=1
            else:
                agent=GreedyAgent() if role=='greedy' else TeamHeuristicAgent()
                check(agent.act(obs,tuple(legal))==action,'fixed baseline action')
            check(type(d['explored']) is bool and d['explored']==explore,'exploration flag')
            counts['decisions']+=1;counts[role+'_decisions']+=1;counts['all_candidates']+=len(legal)
            result=env.step(obs.player_id,action,state_version=s['state_version']);indices[i]+=1
            check(env.state_digest()==s['digest'],'step digest')
            if not result.terminal:observations[i]=env.observe(result.next_player)
        score_batches+=math.ceil(cycle/cfg['chunk_size'])
    samples=[]
    for i,env in enumerate(envs):
        h=w['hands'][i];rewards=env.state.settlement.team_rewards
        check(list(rewards)==h['team_rewards'] and env.state_digest()==h['terminal_digest'],'terminal rewards')
        check(len(trajectories[i])==h['samples'],'hand learner denominator')
        samples.extend((a,b,rewards[t]) for a,b,t in trajectories[i])
    check(len(samples)==w['samples'] and len(samples)>0,'wave sample denominator')
    rng.shuffle(samples)
    audit_batch_rows(samples,w,cfg)
    counts['positive']=sum(x[2]==1 for x in samples)
    counts['negative']=len(samples)-counts['positive']
    counts['single_class_batches']=sum(b['single_class'] for b in w['batches'])
    counts['dual_class_batches']=sum(not b['single_class'] for b in w['batches'])
    counts['samples']=len(samples);counts['updates']=math.ceil(len(samples)/cfg['batch_size'])
    for key in ('scored_candidates','score_requests','frozen_candidates','frozen_requests'):
        check(w[key]==counts[key],'scoring count '+key)
    check(w['score_batches']==score_batches,'CUDA scoring batches')
    u=counts['updates']
    check(w['device_proof']==dict(device='cuda:0',dtype='torch.float32',forward_checks=u,gradient_checks=7*u,adam_tensor_checks=21*u),'CUDA training counts')
    check(math.isfinite(w['mean_loss']) and w['mean_loss']>=0,'finite loss')
    return counts,samples

def audit_training(root,job):
    import torch
    torch.set_num_threads(1)
    from guandan.learning.model import DMCNetwork
    from experiments.p5d_checkpoint import sources,versions
    from experiments.review_p3f import verify_optimizer
    folder=root/'training'/job;report=read(folder/'report.json');arm,seed=job.split('-');seed=int(seed);cfg=config(seed,arm)
    check(report['status']=='PASS' and report['config']==cfg and report['job']==job,'training metadata')
    initial=payload(folder/'initial');final=payload(folder/'final')
    check(initial['config']==cfg and initial['sources']==sources() and initial['versions']==versions() and initial['objective']==cfg['objective'],'initial source/config/version/objective')
    torch.manual_seed(seed);expected=DMCNetwork().state_dict()
    check(equal(initial['model'],expected) and equal(initial['frozen'],expected),'paired from-scratch initialization')
    check(initial['waves']==initial['episodes']==initial['updates']==0 and initial['optimizer']['state']=={} and initial['used_deal_seeds']==[],'fresh optimizer and counters')
    rng=random.Random(seed);check(equal(rng.getstate(),initial['policy_rng']),'initial policy RNG')
    snapshots={w:payload(folder/f'wave-{w}') for w in (1,2,3,4,200)};snapshots[400]=final
    totals=Counter();wave_count=0
    for i,w in enumerate(rows(folder/'waves.jsonl.gz')):
        check(i<400,'no excess waves');counts,_=audit_wave(w,i,cfg,rng)
        totals.update(counts);wave_count+=1
        check(w['updates']==totals['updates'],'cumulative Adam updates')
        check(w['pool_sha256']==initial['pool_sha256'] and w['pool_version']=='gd-p5a-pool-v1','fixed pool identity')
        if wave_count in snapshots:
            p=snapshots[wave_count]
            check(equal(p['policy_rng'],rng.getstate()),'checkpoint exploration and shuffle RNG')
            check(equal(p['frozen'],initial['frozen']) and p['pool_sha256']==initial['pool_sha256'],'immutable frozen pool')
            check(p['episodes']==wave_count*4 and p['waves']==wave_count and p['updates']==totals['updates'],'checkpoint counters')
            check(p['used_deal_seeds']==list(range(100000,100000+wave_count*4)),'checkpoint seed lineage')
            check(p['objective']==cfg['objective'] and p['versions']==versions(),'checkpoint objective/version')
            check(p['config']==cfg and p['sources']==sources() and p['runtime']==initial['runtime'],'checkpoint config/source/runtime')
            check(equal(p['torch_rng'],initial['torch_rng']) and equal(p['cuda_rng'],initial['cuda_rng']),'unused tensor RNG unchanged')
            verify_optimizer(p,totals['updates'])
    check(wave_count==400 and report['hands']==1600 and report['waves']==400,'fixed training budget')
    for key in ('samples','decisions','updates','scored_candidates','frozen_candidates'):
        check(report[key]==totals[key],'reported training total '+key)
    check(report['final_manifest']==read(folder/'final/manifest.json') and report['initial_manifest']==read(folder/'initial/manifest.json'),'report checkpoint binding')
    check(report['runtime']==final['runtime'] and report['model_promoted'] is False,'training scope')
    if arm=='ordinary':
        ordinary_compatibility(root,job)
    return dict(job=job,hands=1600,waves=400,**dict(totals),seconds=report['seconds'],
        candidate_sha256=report['final_manifest']['sha256'],pool_sha256=initial['pool_sha256'])

def wave_numerics(root,job,index):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from guandan.learning.model import DMCNetwork
    from experiments.controller_p5a_audit import explicit_forward,full_scores
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=root/'training'/job;arm,seed=job.split('-');cfg=config(int(seed),arm)
    before=payload(folder/('initial' if index==0 else f'wave-{index}'))
    after=payload(folder/f'wave-{index+1}');w=next(islice(rows(folder/'waves.jsonl.gz'),index,None))
    rng=random.Random();rng.setstate(before['policy_rng'])
    _,samples=audit_wave(w,index,cfg,rng,encode=True)
    model=DMCNetwork().cuda();model.load_state_dict(before['model'])
    fixed=DMCNetwork().cuda();fixed.load_state_dict(before['frozen'])
    # Check every non-exploratory neural action of this real first wave directly.
    requests=candidates=0
    for h in w['hands']:
        env=HandEnv();env.reset(h['seed'],initial_level=h['level'],starting_player=h['starting_player'])
        for d,s in zip(h['decisions'],h['replay']['steps'],strict=True):
            obs=env.observe(s['player']);legal=env.legal_actions(s['player'])
            if d['role']=='frozen' or (d['role']=='current' and not d['explored']):
                scorer=fixed if d['role']=='frozen' else model
                check(int(full_scores(scorer,obs,legal).argmax())==d['selected'],'first-wave direct CUDA policy action')
                requests+=1;candidates+=len(legal)
            env.step(s['player'],Action.from_dict(s['action']),state_version=s['state_version'])
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],capturable=True)
    opt.load_state_dict(copy.deepcopy(before['optimizer']));loss_total=0
    for start in range(0,len(samples),256):
        batch=samples[start:start+256]
        state=torch.tensor([s[0] for s in batch],device='cuda');actions=torch.tensor([s[1] for s in batch],device='cuda')
        target=torch.tensor([s[2] for s in batch],device='cuda',dtype=torch.float32)
        opt.zero_grad(set_to_none=True);residual=explicit_forward(model,state,actions)-target
        positive=sum(s[2]==1 for s in batch);negative=len(batch)-positive
        balanced=cfg['objective']=='label_balanced' and positive and negative
        wp=len(batch)/(2*positive) if balanced else 1.
        wn=len(batch)/(2*negative) if balanced else 1.
        weights=torch.tensor([wp if s[2]==1 else wn for s in batch],device='cuda')
        loss=(weights*residual.square()).sum()/len(batch)
        check(abs(float(loss.detach())-w['batches'][start//256]['loss'])<3e-6,'independent weighted batch loss')
        loss.backward();opt.step();loss_total+=float(loss.detach())*len(batch)
    max_model=max_adam=0.
    for k,t in model.state_dict().items():
        ref=after['model'][k].cuda();torch.testing.assert_close(t,ref,rtol=3e-4,atol=3e-6)
        max_model=max(max_model,float((t-ref).abs().max()))
    for k,state in opt.state_dict()['state'].items():
        for name,t in state.items():
            ref=after['optimizer']['state'][k][name].cuda();torch.testing.assert_close(t,ref,rtol=3e-4,atol=3e-6)
            max_adam=max(max_adam,float((t-ref).abs().max()))
    check(abs(loss_total/len(samples)-w['mean_loss'])<3e-6,'first-wave independent mean loss')
    return dict(job=job,wave=index+1,samples=len(samples),updates=len(w['batches']),
        dual_class_batches=sum(not b['single_class'] for b in w['batches']),single_class_batches=sum(b['single_class'] for b in w['batches']),direct_policy_requests=requests,
        direct_policy_candidates=candidates,max_model_abs_error=max_model,max_adam_abs_error=max_adam,
        rtol=3e-4,atol=3e-6)


def first_wave_numerics(root,job):
    checks=[wave_numerics(root,job,i) for i in range(4)]
    return dict(job=job,waves=checks,
        **{k:sum(c[k] for c in checks) for k in ('samples','updates','direct_policy_requests','direct_policy_candidates','dual_class_batches','single_class_batches')},
        max_model_abs_error=max(c['max_model_abs_error'] for c in checks),
        max_adam_abs_error=max(c['max_adam_abs_error'] for c in checks))


def ordinary_compatibility(root,job):
    seed=job.split('-')[1]
    current=root/'training'/job;old=ROOT/'artifacts/evaluations/p5b-pool-v1/training'/f'mixed-{seed}'
    for name in ('initial','wave-1','wave-200','final'):
        a=payload(current/name);b=payload(old/name)
        for key in ('model','frozen','optimizer','policy_rng','torch_rng','cuda_rng','episodes','waves','updates','used_deal_seeds','pool_sha256'):
            check(equal(a[key],b[key]),'P5b ordinary core bitwise '+name+' '+key)
    count=0
    for a,b in zip(rows(current/'waves.jsonl.gz'),rows(old/'waves.jsonl.gz'),strict=True):
        check({k:v for k,v in a.items() if k not in ('objective','batches')}==b,'P5b ordinary full wave log')
        count+=1
    check(count==400,'ordinary full history 400 waves')
    return dict(status='BITWISE_EQUAL',waves=400,checkpoint_count=4)
