"""Independent P5a replay, actor routing, RNG, numerical and recovery audit.

Does not call train_wave, lineup, or score_many. All decisions are reconstructed
from scheduled deals, player projections and complete legal lists. Trusted local
checkpoints only. Results are engineering evidence, never a promotion test.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'experiments')]
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from collections import Counter
from hashlib import sha256
from io import BytesIO
import argparse
import copy
import json
import math
import random
import zipfile
import torch
import torch.nn.functional as F
from guandan.env import HandEnv
from guandan.types import Action
from guandan.agents.baselines import GreedyAgent, TeamHeuristicAgent
from guandan.learning.encoding import encode_observation, encode_action
from guandan.learning.model import DMCNetwork


def check(ok, label):
    if not ok: raise ValueError(label)


def read(p): return json.loads(Path(p).read_text('utf-8'))
def digest(p): return sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def load_payload(directory):
    data=(directory/'checkpoint.pt').read_bytes(); m=read(directory/'manifest.json')
    check(sha256(data).hexdigest()==m['sha256'] and len(data)==m['bytes'],'checkpoint file binding')
    p=torch.load(BytesIO(data),map_location='cpu',weights_only=True)
    for k in ('versions','runtime','sources','boundary','config','episodes','updates','waves','used_deal_seeds','pool_sha256'):
        check(p[k]==m[k],'checkpoint metadata '+k)
    return p


def explicit_forward(model, states, actions):
    w=dict(model.named_parameters())
    h=F.relu(F.linear(states,w['state_fc.weight'],w['state_fc.bias'])+F.linear(actions,w['action_fc.weight']))
    h=F.relu(F.linear(h,w['hidden_fc.weight'],w['hidden_fc.bias']))
    return torch.tanh(F.linear(h,w['output_fc.weight'],w['output_fc.bias'])).flatten()


def full_scores(model,obs,legal):
    # Independent dense per-state expression; chunking only bounds memory.
    result=[]
    with torch.no_grad():
        for start in range(0,len(legal),257):
            part=legal[start:start+257]
            s=torch.tensor([encode_observation(obs)]*len(part),device='cuda')
            a=torch.tensor([encode_action(x) for x in part],device='cuda')
            result.append(explicit_forward(model,s,a).cpu())
    result=torch.cat(result)
    check(len(result)==len(legal) and torch.isfinite(result).all().item(),'direct candidate coverage')
    return result


def audit_wave(row, before, after, wave_index, numerical=True):
    cfg=before['config']; learner=DMCNetwork().cuda(); learner.load_state_dict(before['model'])
    fixed=DMCNetwork().cuda(); fixed.load_state_dict(before['frozen'])
    check(equal(before['frozen'],after['frozen']),'frozen weights immutable')
    check(before['pool_sha256']==after['pool_sha256']==row['pool_sha256'],'pool identity')
    check(row['pool_version']=='gd-p5a-pool-v1','pool version')
    check(row['wave']==wave_index+1 and before['waves']==wave_index and after['waves']==wave_index+1,'wave number')
    check(row['behavior_version']==wave_index and row['learning_version']==wave_index+1,'behavior version')
    check(before['episodes']==4*wave_index and after['episodes']==row['episodes']==4*(wave_index+1),'episode counts')
    check(after['used_deal_seeds']==list(range(108800,108804+4*wave_index)),'scheduled seed history')
    check(len(row['hands'])==4,'four hand wave')
    patterns=[['current','greedy','current','greedy'],['current','team','greedy','team'],
              ['current','frozen','team','frozen'],['current','greedy','frozen','team']]
    envs=[]; observations=[]; trajectories=[[] for _ in range(4)]
    for i,h in enumerate(row['hands']):
        n=wave_index*4+i; env=HandEnv(); obs=env.reset(108800+n,initial_level=2+n%13,starting_player=n%4)
        roles=[patterns[wave_index][(s-i)%4] for s in range(4)]
        check((h['seed'],h['level'],h['starting_player'])==(108800+n,2+n%13,n%4),'deal schedule')
        check(h['roles']==roles,'role schedule')
        check(h['replay']['initial_hands']==[list(x) for x in env.state.initial_hands],'seed-to-hands binding')
        check(h['replay']['initial_digest']==env.state_digest(),'initial digest')
        replayed=HandEnv.replay(h['replay'])
        check(replayed.state.terminal and replayed.state_digest()==h['terminal_digest'],'full replay terminal')
        check(h['steps']==len(h['decisions'])==len(h['replay']['steps']) and h['replay_verified'] is True,'step count')
        check(h['samples']==sum(d['sample'] for d in h['decisions']),'hand sample count')
        envs.append(env); observations.append(obs)
    rng=random.Random(); rng.setstate(before['policy_rng'])
    counts=Counter(); cursor=[0]*4; score_batches=0
    while any(not e.state.terminal for e in envs):
        cycle_candidates=0
        for i,env in enumerate(envs):
            if env.state.terminal: continue
            j=cursor[i]; h=row['hands'][i]; d=h['decisions'][j]; step=h['replay']['steps'][j]
            obs=observations[i]; legal=env.legal_actions(obs.player_id); role=h['roles'][obs.player_id]
            check(d['player']==step['player']==obs.player_id and d['role']==role,'actor routing')
            check(d['candidates']==len(legal) and type(d['selected']) is int and 0<=d['selected']<len(legal),'candidate denominator')
            check(type(d['sample']) is bool and d['sample']==(role=='current'),'sample ownership')
            explored=False
            if role=='current':
                explored=rng.random()<cfg['epsilon']
                if explored:
                    selected=rng.randrange(len(legal)); counts['exploration']+=1
                else:
                    selected=int(full_scores(learner,obs,legal).argmax())
                    counts['current_requests']+=1; counts['current_candidates']+=len(legal)
                    cycle_candidates+=len(legal)
            elif role=='frozen':
                selected=int(full_scores(fixed,obs,legal).argmax())
                counts['frozen_requests']+=1; counts['frozen_candidates']+=len(legal)
            else:
                baseline=GreedyAgent() if role=='greedy' else TeamHeuristicAgent()
                selected=legal.index(baseline.act(obs,tuple(legal)))
            check(type(d['explored']) is bool and d['explored']==explored,'exploration RNG')
            check(selected==d['selected'] and legal[selected]==Action.from_dict(step['action']),'policy action selection')
            action=legal[selected]
            if role=='current':
                trajectories[i].append((encode_observation(obs),encode_action(action),obs.player_id%2))
            counts[role+'_decisions']+=1; counts['decisions']+=1; counts['all_candidates']+=len(legal)
            result=env.step(obs.player_id,action,state_version=step['state_version'])
            check(env.state_digest()==step['digest'],'per-step digest')
            cursor[i]+=1
            if not result.terminal: observations[i]=env.observe(result.next_player)
        score_batches+=math.ceil(cycle_candidates/cfg['chunk_size'])
    samples=[]
    for i,env in enumerate(envs):
        rewards=env.state.settlement.team_rewards; h=row['hands'][i]
        check(list(rewards)==h['team_rewards'] and env.state_digest()==h['terminal_digest'],'terminal label')
        samples.extend((s,a,rewards[t]) for s,a,t in trajectories[i])
        check(len(trajectories[i])==h['samples'],'learner trajectory size')
    check(len(samples)==row['samples'],'wave sample denominator')
    rng.shuffle(samples)
    check(equal(rng.getstate(),after['policy_rng']),'exploration and shuffle final RNG')
    for key,expected in [('scored_candidates',counts['current_candidates']),('score_requests',counts['current_requests']),
                         ('score_batches',score_batches),('frozen_candidates',counts['frozen_candidates']),
                         ('frozen_requests',counts['frozen_requests'])]:
        check(row[key]==expected,'scoring counter '+key)
    batches=math.ceil(len(samples)/cfg['batch_size'])
    check(row['updates']==after['updates']==before['updates']+batches,'update count')
    check(row['device_proof']['forward_checks']==batches,'CUDA forward count')
    numerical_result={}
    if numerical:
        optimizer=torch.optim.Adam(learner.parameters(),lr=cfg['lr'],capturable=True)
        optimizer.load_state_dict(copy.deepcopy(before['optimizer']))
        loss_sum=0.0; max_gradient_error=0.0; gradient_elements=0
        for start in range(0,len(samples),cfg['batch_size']):
            batch=samples[start:start+cfg['batch_size']]
            states=torch.tensor([x[0] for x in batch],device='cuda')
            actions=torch.tensor([x[1] for x in batch],device='cuda')
            targets=torch.tensor([x[2] for x in batch],device='cuda',dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            residual=explicit_forward(learner,states,actions)-targets
            loss=torch.sum(residual.square())/len(batch)
            loss.backward()
            if batches==1:
                # With one update between committed checkpoints, Adam's first
                # moment independently reveals the gradient actually applied.
                beta1=optimizer.param_groups[0]['betas'][0]
                for index,parameter in enumerate(learner.parameters()):
                    final_moment=after['optimizer']['state'][index]['exp_avg'].cuda()
                    previous=before['optimizer']['state'].get(index)
                    initial_moment=previous['exp_avg'].cuda() if previous else torch.zeros_like(final_moment)
                    applied_gradient=(final_moment-beta1*initial_moment)/(1-beta1)
                    torch.testing.assert_close(parameter.grad,applied_gradient,rtol=3e-4,atol=3e-6)
                    max_gradient_error=max(max_gradient_error,float((parameter.grad-applied_gradient).abs().max()))
                    gradient_elements+=parameter.numel()
            optimizer.step()
            loss_sum+=float(loss.detach())*len(batch)
        max_model_error=0.0; max_adam_error=0.0
        for key,t in learner.state_dict().items():
            ref=after['model'][key].cuda()
            torch.testing.assert_close(t,ref,rtol=3e-4,atol=3e-6)
            max_model_error=max(max_model_error,float((t-ref).abs().max()))
        for key,state in optimizer.state_dict()['state'].items():
            for name,t in state.items():
                ref=after['optimizer']['state'][key][name].cuda()
                torch.testing.assert_close(t,ref,rtol=3e-4,atol=3e-6)
                max_adam_error=max(max_adam_error,float((t-ref).abs().max()))
        check(abs(loss_sum/len(samples)-row['mean_loss'])<=3e-6,'independent mean loss')
        numerical_result=dict(batches=batches,samples=len(samples),max_model_abs_error=max_model_error,
                              max_adam_abs_error=max_adam_error,mean_loss=loss_sum/len(samples),rtol=3e-4,atol=3e-6,
                              gradient_elements_checked=gradient_elements,max_gradient_abs_error=max_gradient_error)
    return dict(wave=wave_index+1,hands=4,samples=len(samples),counts=dict(counts),numerical=numerical_result)


def negatives(row,before,after):
    def role(r): r['hands'][0]['roles'][0]='greedy'
    def candidate(r): r['hands'][0]['decisions'][0]['candidates']+=1
    def sample(r): r['hands'][0]['decisions'][0]['sample']=False
    def action(r): r['hands'][0]['decisions'][0]['selected']=-1
    def exploration(r): r['hands'][0]['decisions'][0]['explored']=not r['hands'][0]['decisions'][0]['explored']
    def label(r): r['hands'][0]['team_rewards'].reverse()
    def seed(r): r['hands'][0]['seed']+=1
    cases=dict(role=role,candidates=candidate,sample_ownership=sample,action=action,exploration=exploration,
               team_label=label,seed=seed,sample_count=lambda r:r.update(samples=r['samples']+1),
               scoring_count=lambda r:r.update(scored_candidates=r['scored_candidates']+1),
               updates=lambda r:r.update(updates=r['updates']+1),pool=lambda r:r.update(pool_sha256='0'*64))
    results={}
    for name,mutation in cases.items():
        value=copy.deepcopy(row); mutation(value)
        try: audit_wave(value,before,after,0,numerical=False)
        except (ValueError,AssertionError) as exc: results[name]=str(exc)
        else: raise ValueError('negative accepted: '+name)
    bad=copy.deepcopy(after); bad['policy_rng']=random.Random(17).getstate()
    try: audit_wave(row,before,bad,0,numerical=False)
    except ValueError as exc: results['rng']=str(exc)
    else: raise ValueError('negative accepted: rng')
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path); parser.add_argument('resume',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.run=args.run.resolve(); args.resume=args.resume.resolve(); args.output=args.output.resolve()
    check(not args.output.exists(),'fresh audit output')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    from experiments.p5a_checkpoint import sources,load_checkpoint
    source_before=read(args.run/'source-before.json')
    check(source_before==sources()==read(args.resume/'source-before.json'),'frozen source identity')
    expected_config=dict(seed=314500,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4,mode='mixed')
    checkpoints=[args.run/'initial']+[args.run/f'wave-{w}' for w in range(1,5)]
    payloads=[]
    for path in checkpoints:
        load_checkpoint(path)  # strict on-disk production compatibility checks, separate from replay proof
        p=load_payload(path); check(p['config']==expected_config,'fixed engineering config')
        check(p['sources']==source_before,'checkpoint source binding'); payloads.append(p)
    torch.manual_seed(314500); expected=DMCNetwork()
    check(equal(expected.state_dict(),payloads[0]['model']) and equal(payloads[0]['model'],payloads[0]['frozen']),
          'initial weights and fixed pool origin')
    check(payloads[0]['updates']==0 and payloads[0]['episodes']==0 and payloads[0]['used_deal_seeds']==[], 'fresh start')
    for folder in (args.run,args.resume):
        completion=read(folder/'complete.json')
        check(completion['status']=='PASS' and completion['episodes']==16 and completion['waves']==4
              and completion['updates']==payloads[4]['updates'] and completion['source_sha256']==source_before
              and completion['scope']=='ENGINEERING_ONLY' and completion['model_promoted'] is False
              and completion['reserved_test_executed'] is False,'completion receipt')
    rows=[read(args.run/f'wave-{w}.json') for w in range(1,5)]
    wave_audits=[]
    for i,row in enumerate(rows):
        wave_audits.append(audit_wave(row,payloads[i],payloads[i+1],i))
        print('independent wave',i+1,'PASS',flush=True)
    for w in (3,4):
        check(read(args.resume/f'wave-{w}.json')==rows[w-1],'fresh-process exact wave log')
        check(equal(load_payload(args.resume/f'wave-{w}'),payloads[w]),'fresh-process exact complete state')
    negative=negatives(rows[0],payloads[0],payloads[1])
    print('negative mutations',len(negative),'PASS',flush=True)
    from experiments.controller_p3_closeout import audit_history
    history=audit_history()
    # Include P3l, which was itself the producer of the earlier closure receipt.
    p3l=ROOT/'artifacts/evaluations/p3l-normcap-v1'
    pre=read(p3l/'preregistration.json')
    registry=pre.get('source_sha256',pre.get('source_sha256_before'))
    check(bool(registry),'P3l source registry')
    for path,h in registry.items(): check(digest(ROOT/path)==h,'P3l source '+path)
    closure=read(p3l/'p3-completion-audit.json')
    checked=0
    for path,h in closure['artifact_sha256'].items():
        check(digest(ROOT/path)==h,'P3 closure evidence '+path); checked+=1
    check(closure['status']=='PASS','P3 closure status')
    check(sources()==source_before,'post-audit source identity')
    artifact_paths=[]
    for folder in (args.run,args.resume):
        artifact_paths.extend(p for p in folder.rglob('*') if p.is_file())
    result=dict(status='PASS',scope='ACCEPTED_FIXED_POOL_SAMPLING_V1',training_hands=16,
        resume_repeated_hands=8,unique_deals=16,samples=sum(r['samples'] for r in rows),
        updates=payloads[4]['updates'],waves=wave_audits,negative_rejections=negative,
        fresh_process_resume='BITWISE_EQUAL',pool_immutable=True,source_sha256=source_before,
        historical=history,p3l_registered_sources=len(registry),p3_closure_artifacts_checked=checked,
        artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in artifact_paths},
        controller_sha256=digest(Path(__file__)),model_promoted=False,strength='NOT_ESTABLISHED',
        validation_games=0,reserved_test_executed=False)
    write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','scope','training_hands','samples','updates','fresh_process_resume')}))


if __name__=='__main__': main()
