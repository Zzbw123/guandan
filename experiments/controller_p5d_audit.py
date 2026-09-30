"""Independent P5d replay, actor routing, RNG, numerical and recovery audit.

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
    cfg=before['config']; check(row['objective']==cfg['objective'],'objective binding'); learner=DMCNetwork().cuda(); learner.load_state_dict(before['model'])
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
    audit_batch_rows(samples,row,cfg)
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
            positive=sum(x[2]==1 for x in batch); negative=len(batch)-positive
            # Independent per-sample weighted sum, not the production class means.
            wp=len(batch)/(2*positive) if cfg['objective']=='label_balanced' and positive and negative else 1.0
            wn=len(batch)/(2*negative) if cfg['objective']=='label_balanced' and positive and negative else 1.0
            weights=torch.tensor([wp if x[2]==1 else wn for x in batch],device='cuda')
            loss=torch.sum(weights*residual.square())/len(batch)
            check(abs(float(loss.detach())-row['batches'][start//cfg['batch_size']]['loss'])<=3e-6,'batch numerical loss')
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



def audit_batch_rows(samples,row,cfg):
    expected_batches=math.ceil(len(samples)/cfg['batch_size'])
    check(len(row['batches'])==expected_batches,'batch count')
    weighted_sum=0.0
    for bi,start in enumerate(range(0,len(samples),cfg['batch_size'])):
        batch=samples[start:start+cfg['batch_size']]; n=len(batch)
        positive=sum(x[2]==1 for x in batch);negative=n-positive
        balanced=cfg['objective']=='label_balanced' and positive>0 and negative>0
        wp=n/(2*positive) if balanced else float(positive>0)
        wn=n/(2*negative) if balanced else float(negative>0)
        expected=dict(objective=cfg['objective'],start=start,n=n,positive=positive,negative=negative,
                      positive_weight=wp,negative_weight=wn,single_class=not(positive and negative))
        actual=row['batches'][bi]
        check(set(actual)==set(expected)|{'loss'},'batch schema')
        for key,value in expected.items():
            check(type(actual[key]) is type(value) and actual[key]==value,'batch '+key)
        check(type(actual['loss']) is float and math.isfinite(actual['loss']) and actual['loss']>=0,'finite batch loss')
        weighted_sum+=actual['loss']*n
    check(abs(weighted_sum/len(samples)-row['mean_loss'])<1e-12,'batch aggregate loss')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();out=args.run.resolve()
    output=args.output.resolve() if args.output else out/'controller-audit.json'
    check(not output.exists(),'fresh controller report')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from experiments.p5d_checkpoint import sources,load_checkpoint
    from experiments.p5d_package import protected
    pre=read(out/'preregistration.json');receipt=read(out/'run-receipt.json')
    check(receipt['status']=='PASS','run completion')
    for name,h in {**pre['sources'],**pre['inputs']}.items():check(digest(ROOT/name)==h,'frozen source/input '+name)
    for name,h in receipt['artifacts'].items():check(digest(out/name)==h,'run binding '+name)
    check(sources()==pre['checkpoint_sources'],'registered checkpoint sources')
    with zipfile.ZipFile(out/'source-snapshot.zip') as archive:
        check(set(archive.namelist())==set(pre['sources']),'archive paths')
        for name,h in pre['sources'].items():check(sha256(archive.read(name)).hexdigest()==h,'archive source')
    audits={};all_payloads={};all_rows={}
    for objective in ('ordinary','label_balanced'):
        folder=out/objective
        cfg=dict(seed=314500,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4,mode='mixed',objective=objective)
        paths=[folder/'initial']+[folder/f'wave-{w}' for w in range(1,5)]
        payloads=[]
        for path in paths:
            load_checkpoint(path)
            value=load_payload(path)
            check(value['config']==cfg and value['sources']==sources(),'fixed config/source')
            payloads.append(value)
        torch.manual_seed(314500);expected=DMCNetwork()
        check(equal(expected.state_dict(),payloads[0]['model']) and equal(payloads[0]['model'],payloads[0]['frozen']),'initial weights')
        check(payloads[0]['episodes']==payloads[0]['updates']==payloads[0]['waves']==0 and payloads[0]['used_deal_seeds']==[],'fresh initial state')
        complete=read(folder/'complete.json')
        check(complete==dict(status='PASS',objective=objective,episodes=16,waves=4,updates=payloads[-1]['updates'],
            source_sha256=sources(),scope='ENGINEERING_ONLY',model_promoted=False,validation_games=0,reserved_test_executed=False),'completion receipt')
        rows=[read(folder/f'wave-{w}.json') for w in range(1,5)]
        checks=[]
        for wi,row in enumerate(rows):
            checks.append(audit_wave(row,payloads[wi],payloads[wi+1],wi))
            print(objective,'independent wave',wi+1,'PASS',flush=True)
        audits[objective]=checks;all_payloads[objective]=payloads;all_rows[objective]=rows
    core=('model','optimizer','policy_rng','torch_rng','cuda_rng','episodes','updates','waves','used_deal_seeds','frozen','pool_sha256')
    check(all(equal(all_payloads['ordinary'][0][k],all_payloads['label_balanced'][0][k]) for k in core),'paired initial state')
    historical=ROOT/'artifacts/evaluations/p5a-pool-v2'
    for wi,payload in enumerate(all_payloads['ordinary']):
        old=load_payload(historical/('initial' if wi==0 else f'wave-{wi}'))
        for key in core:check(equal(payload[key],old[key]),'ordinary P5a bitwise '+key)
        if wi:
            new={k:v for k,v in all_rows['ordinary'][wi-1].items() if k not in ('objective','batches')}
            check(new==read(historical/f'wave-{wi}.json'),'ordinary historical wave log')
    for wi in (3,4):
        load_checkpoint(out/'resume'/f'wave-{wi}')
        check(equal(load_payload(out/'resume'/f'wave-{wi}'),all_payloads['label_balanced'][wi]),'fresh-process exact checkpoint')
        check(read(out/'resume'/f'wave-{wi}.json')==all_rows['label_balanced'][wi-1],'fresh-process exact log')
    check(read(out/'resume/complete.json')==read(out/'label_balanced/complete.json'),'resume completion')
    negative=negatives(all_rows['label_balanced'][0],all_payloads['label_balanced'][0],all_payloads['label_balanced'][1])
    row=all_rows['label_balanced'][0];before=all_payloads['label_balanced'][0];after=all_payloads['label_balanced'][1]
    mutations={
        'objective':lambda r:r.update(objective='ordinary'),
        'batch_count':lambda r:r['batches'].pop(),
        'batch_positive':lambda r:r['batches'][0].update(positive=r['batches'][0]['positive']+1),
        'batch_denominator':lambda r:r['batches'][0].update(n=r['batches'][0]['n']+1),
        'batch_weight':lambda r:r['batches'][0].update(positive_weight=r['batches'][0]['positive_weight']+.25),
        'batch_single_class':lambda r:r['batches'][0].update(single_class=not r['batches'][0]['single_class']),
        'batch_start':lambda r:r['batches'][0].update(start=1),
        'batch_loss':lambda r:r['batches'][0].update(loss=r['batches'][0]['loss']+.1)}
    for name,mutate in mutations.items():
        bad=copy.deepcopy(row);mutate(bad)
        try:audit_wave(bad,before,after,0,numerical=False)
        except (ValueError,AssertionError) as exc:negative[name]=str(exc)
        else:raise ValueError('negative accepted '+name)
    # Coordinated edits to both loss aggregates must still fail numerical recomputation.
    bad=copy.deepcopy(row);bad['batches'][0]['loss']+=.125
    bad['mean_loss']+=.125*bad['batches'][0]['n']/bad['samples']
    try:audit_wave(bad,before,after,0,numerical=True)
    except (ValueError,AssertionError) as exc:negative['coordinated_loss']=str(exc)
    else:raise ValueError('coordinated loss mutation accepted')
    history=protected()
    for name,h in {**pre['sources'],**pre['inputs']}.items():check(digest(ROOT/name)==h,'post-audit binding '+name)
    for name,h in receipt['artifacts'].items():check(digest(out/name)==h,'post-audit artifact '+name)
    result=dict(status='PASS',scope='ACCEPTED_LABEL_BALANCED_ENGINEERING_V1',training_hands=32,
        unique_deals=16,resume_repeated_hands=8,validation_games=0,reserved_test_executed=False,
        model_promoted=False,strength='NOT_ESTABLISHED',fresh_process_resume='BITWISE_EQUAL',
        ordinary_p5a_compatibility='BITWISE_EQUAL',waves=audits,negative_rejections=negative,
        historical=history,source_sha256=pre['sources'],controller_sha256=digest(Path(__file__)),
        artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in out.rglob('*') if p.is_file()})
    write(output,result)
    print(json.dumps(dict(status=result['status'],negative_rejections=len(negative),
        samples={k:sum(x['samples'] for x in v) for k,v in audits.items()},fresh_process_resume=result['fresh_process_resume'])),flush=True)


if __name__=='__main__':main()
