"""Controller-owned independent numerical, corpus, and evaluation checks."""
from collections import Counter
from dataclasses import asdict
import gzip,json,math
from pathlib import Path
from random import Random
from unittest.mock import patch
import torch
import torch.nn.functional as F
from experiments.p5f_run_common import check,digest,read,write,verify_frozen
from experiments.p5f_protocol import config,schedule,TRAIN,DEV,INITS,ARMS
from experiments.p5f_training import TeacherTrainer
from experiments.p5f_checkpoint import restore,read_payload
from guandan.env import HandEnv
from guandan.types import Action
from guandan.rules.cards import rank,rank_strength
from guandan.learning.encoding import encode_observation,encode_action


def normalize(value):return json.loads(json.dumps(value,allow_nan=False))


def lines(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def teacher_score(obs,a):
    if a.kind=='pass':return -5.
    if len(a.cards)==len(obs.hand):return 1000.
    remaining=Counter(rank(c) for c in obs.hand if c not in a.cards)
    return 5.*len(a.cards)+sum(n//2 for n in remaining.values())*.25-(
        12. if a.kind in ('bomb','straight_flush','joker_bomb') else 0.)-rank_strength(a.main_rank,obs.level)/100


def direct(model,obs,legal):
    state=torch.tensor([encode_observation(obs)],device='cuda',dtype=torch.float32)
    actions=torch.tensor([encode_action(a) for a in legal],device='cuda',dtype=torch.float32)
    weights=model.state_dict(keep_vars=True)
    hidden=F.relu(F.linear(state,weights['state_fc.weight'],weights['state_fc.bias'])+
                  F.linear(actions,weights['action_fc.weight']))
    hidden=F.relu(F.linear(hidden,weights['hidden_fc.weight'],weights['hidden_fc.bias']))
    return F.linear(hidden,weights['output_fc.weight'],weights['output_fc.bias']).flatten()


def independent_step(trainer,requests):
    terms=[]
    for obs,legal in requests:
        raw=direct(trainer.model,obs,legal)
        scores=[teacher_score(obs,a) for a in legal];lo,hi=min(scores),max(scores)
        if trainer.config['objective']=='ranking':
            top=torch.tensor([x==hi for x in scores],device='cuda')
            terms.append(torch.logsumexp(raw,0)-torch.logsumexp(raw[top],0))
        else:
            targets=torch.tensor([0.]*len(scores) if lo==hi else
                                 [1.6*(x-lo)/(hi-lo)-.8 for x in scores],device='cuda')
            terms.append((raw.tanh()-targets).square().mean())
    loss=torch.stack(terms).mean()
    trainer.optimizer.zero_grad(set_to_none=True);loss.backward()
    grads=[p.grad.detach().clone() for p in trainer.model.parameters()]
    trainer.optimizer.step()
    return float(loss),grads


def audit_corpus(path,deals):
    manifest=read(path/'manifest.json');saved=lines(path/'replays.jsonl.gz')
    check(manifest['deals']==[list(d) for d in deals] and len(saved)==len(deals),'corpus frozen schedule')
    check(digest(path/'replays.jsonl.gz')==manifest['replays_sha256'],'corpus bytes')
    all_sizes=[];first=[];probes=[];development=[]
    for index,(row,deal) in enumerate(zip(saved,deals,strict=True)):
        check(row['index']==index and [row['seed'],row['level'],row['starting_player']]==list(deal),'deal metadata')
        env=HandEnv();env.reset(deal[0],initial_level=deal[1],starting_player=deal[2])
        replay=row['replay'];check([list(h) for h in env.state.initial_hands]==replay['initial_hands'],'scheduled hands')
        sizes=[];chosen_probe=0
        for step,item in enumerate(replay['steps']):
            obs=env.observe(env.state.current_player);legal=env.legal_actions(obs.player_id)
            check(item['player']==obs.player_id and item['state_version']==obs.state_version,'turn state')
            scores=[teacher_score(obs,a) for a in legal]
            chosen=max(range(len(legal)),key=scores.__getitem__)
            action=Action.from_dict(item['action'])
            check(action==legal[chosen],'independent teacher action')
            by_encoding={}
            for a,score in zip(legal,scores):
                encoding=encode_action(a)
                check(by_encoding.get(encoding,score)==score,'encoded candidate teacher collision')
                by_encoding[encoding]=score
            if len(first)<128:first.append((obs,legal))
            if index<13 and len(legal)>1 and chosen_probe<2:
                probes.append((index,step,obs,legal));chosen_probe+=1
            if len(deals)==26:development.append((index,step,obs,legal))
            result=env.step(obs.player_id,action,state_version=obs.state_version)
            check(env.state_digest()==item['digest'] and result.terminal==item['terminal'],'step replay')
            check(normalize(asdict(result.events[0]))==item['event'],'event replay')
            settlement=normalize(asdict(result.settlement)) if result.settlement else None
            check(settlement==item['settlement'],'step settlement')
            sizes.append(len(legal))
        check(env.state.terminal and env.state_digest()==replay['final_digest'],'corpus terminal')
        check(len(sizes)==row['steps'] and sum(sizes)==row['legal_candidates'],'per-hand counts')
        all_sizes.extend(sizes)
    check(len(all_sizes)==manifest['observations'] and sum(all_sizes)==manifest['candidates'],'corpus totals')
    return all_sizes,first,probes,development


def audit_training(root,job,sizes,first):
    objective,seed=job.split('-');out=root/'training'/job;report=read(out/'report.json')
    check(report['config']==config(int(seed),objective) and report['hands']==1600,'training report config/budget')
    corpus_pin=digest(root/'corpus/train/manifest.json')
    logs=lines(out/'batches.jsonl')
    check(digest(out/'batches.jsonl')==report['batches_sha256'],'batch file SHA')
    check(len(logs)==math.ceil(len(sizes)/64)==report['updates'],'update budget')
    for i,row in enumerate(logs):
        group=sizes[i*64:(i+1)*64]
        check(row['update']==i+1 and row['observation_start']==i*64 and
              row['observation_stop']==min((i+1)*64,len(sizes)),'FIFO position')
        check(row['per_request_sizes']==group and row['requests']==len(group) and
              row['scored_candidates']==sum(group) and row['total_candidates']==sum(sizes[:(i+1)*64]),'complete candidate denominators')
        check(math.isfinite(row['loss']) and row['device_proof']['device'].startswith('cuda'),'finite CUDA objective')
    check(report['samples']==len(sizes) and report['candidates']==sum(sizes),'sample totals')
    initial=read(out/'initial/manifest.json')
    control,_=restore(out/'initial',initial['sha256'],corpus_pin)
    production,_=restore(out/'initial',initial['sha256'],corpus_pin)
    max_grad=max_model=0.
    for i in range(2):
        requests=first[i*64:(i+1)*64]
        loss,grads=independent_step(control,requests)
        row=production.update(requests)
        check(row==logs[i],'first numerical batch log exact reproduction')
        check(math.isclose(loss,row['loss'],rel_tol=3e-4,abs_tol=3e-6),'dense objective')
        for p,g in zip(production.model.parameters(),grads):
            torch.testing.assert_close(p.grad,g,rtol=3e-4,atol=3e-6)
            max_grad=max(max_grad,float((p.grad-g).abs().max()))
        manifest=read(out/f'batch-{i+1}/manifest.json')
        payload,_=read_payload(out/f'batch-{i+1}',manifest['sha256'])
        check(payload['corpus_sha256']==corpus_pin and payload['updates']==i+1,'numerical checkpoint lineage')
        for name,p in control.model.state_dict().items():
            torch.testing.assert_close(p.cpu(),payload['model'][name],rtol=3e-4,atol=3e-6)
            max_model=max(max_model,float((p.cpu()-payload['model'][name]).abs().max()))
        for j,values in control.optimizer.state_dict()['state'].items():
            for name,p in values.items():torch.testing.assert_close(p.cpu(),payload['optimizer']['state'][j][name],rtol=3e-4,atol=3e-6)
    final=read(out/'final/manifest.json')
    payload,_=read_payload(out/'final',final['sha256'])
    check(payload['samples']==len(sizes) and payload['updates']==len(logs) and
          payload['corpus_sha256']==corpus_pin and final==report['final'],'final counters/binding')
    return dict(job=job,samples=len(sizes),candidates=sum(sizes),updates=len(logs),
                numerical_batches=2,max_gradient_error=max_grad,max_model_error=max_model)


def audit_fit(root,job,probes,development):
    from experiments.p5f_checkpoint import load_model
    from experiments.p5f_job import fit_summary
    out=root/'fit'/job;report=read(out/'report.json')
    checkpoint=root/'training'/job/'final';manifest=read(checkpoint/'manifest.json')
    model,_=load_model(checkpoint,manifest['sha256'])
    max_error=0.
    for source,selected in [('train_probe',probes),('development',development)]:
        path=out/f'{source}.jsonl.gz';check(digest(path)==report['artifact_sha256'][path.name],'fit SHA')
        records=lines(path);check(len(records)==len(selected),'fit coverage')
        for row,(deal,step,obs,legal) in zip(records,selected,strict=True):
            teacher=[teacher_score(obs,a) for a in legal]
            with torch.no_grad():values=direct(model,obs,legal).cpu()
            original=torch.tensor(row['logits'])
            torch.testing.assert_close(original,values,rtol=3e-4,atol=3e-6)
            max_error=max(max_error,float((values-original).abs().max()))
            check(row['teacher_scores']==teacher and row['candidates']==len(legal) and
                  row['deal_index']==deal and row['step']==step,'fit source/labels')
            chosen=int(original.argmax());lo,hi=min(teacher),max(teacher)
            check(row['index']==chosen and row['match']==(teacher[chosen]==hi),'fit choice')
            check(math.isclose(row['regret'],0. if hi==lo else (hi-teacher[chosen])/(hi-lo),abs_tol=1e-14),'regret')
            check(row['multi']==(len(legal)>1) and row['long_candidates']==(len(legal)>1024) and
                  row['endgame']==(len(obs.hand)<=5) and row['finish_opportunity']==any(
                      a.kind!='pass' and len(a.cards)==len(obs.hand) for a in legal),'fit strata')
        check(fit_summary(records)==report['summary'][source],'fit summaries')
        selected_multi=[r for r in records if r['multi']]
        check(report['summary'][source]['multi']['matches']==sum(r['match'] for r in selected_multi),'independent fit counts')
    return dict(job=job,max_score_error=max_error,summary=report['summary'])


def interval(values,levels):
    groups=[[v for v,l in zip(values,levels) if l==level] for level in range(2,15)]
    check(all(len(g)==2 for g in groups),'two original development deals per level')
    rng=Random(314565);draws=sorted(sum(g[rng.randrange(len(g))] for g in groups for _ in g)/len(values) for _ in range(5000))
    def q(p):
        position=(len(draws)-1)*p;i=int(position);f=position-i
        return draws[i]*(1-f)+draws[min(i+1,len(draws)-1)]*f
    return [q(.025),q(.975)]


def audit_evaluation(root,job):
    # Reuse accepted independent replay/behavior audit with explicitly bound
    # P5f schedule and independently recomputed P5f-seeded intervals.
    import experiments.controller_p5e_evaluation as previous
    from guandan.evaluation.statistics import summarize_matchup
    def summary(rows,trials):
        return {key:summarize_matchup([r for r in rows if r['matchup_id']==key],
            [t for t in trials if t.matchup_id==key],5000,314565) for key in sorted({t.matchup_id for t in trials})}
    with patch.object(previous,'schedule',lambda validation=False:schedule()), \
         patch.object(previous,'summarize',summary),patch.object(previous,'independent_interval',interval):
        return previous.audit_evaluation(root,job)


def audit(root):
    verify_frozen(root)
    sizes,first,probes,_=audit_corpus(root/'corpus/train',TRAIN)
    _,_,_,development=audit_corpus(root/'corpus/development',DEV)
    reports={};fit_reports={};evaluations={}
    for initializer in INITS:
        for arm in ARMS:
            job=f'{arm}-{initializer}'
            reports[job]=audit_training(root,job,sizes,first)
            fit_reports[job]=audit_fit(root,job,probes,development)
            evaluations[job]=audit_evaluation(root,job)
            print(json.dumps(dict(audit=job,status='PASS')),flush=True)
        a=read(root/'training'/f'regression-{initializer}'/'initial/manifest.json')
        b=read(root/'training'/f'ranking-{initializer}'/'initial/manifest.json')
        pa,_=read_payload(root/'training'/f'regression-{initializer}'/'initial',a['sha256'])
        pb,_=read_payload(root/'training'/f'ranking-{initializer}'/'initial',b['sha256'])
        for k in pa['model']:check(torch.equal(pa['model'][k],pb['model'][k]),'paired initialization')
    paired={};gate=True
    for initializer in INITS:
        rank_report=fit_reports[f'ranking-{initializer}']['summary']
        gate &= rank_report['train_probe']['multi']['accuracy']>=.99 and rank_report['development']['multi']['accuracy']>=.95
        for opponent,threshold in [('greedy',.45),('random',.80)]:
            key='dmc|dmc|'+opponent;groups=[]
            for arm in ARMS:
                rows=lines(root/'evaluations'/f'{arm}-{initializer}'/'results.jsonl')
                groups.append([sum(r['win'] for r in rows if r['matchup_id']==key and r['deal_seed']==d[0])/8 for d in DEV])
            delta=[b-a for a,b in zip(*groups)]
            paired[f'{initializer}-{opponent}']=dict(difference=sum(delta)/26,ci95=interval(delta,[d[1] for d in DEV]))
            gate &= sum(groups[1])/26>=threshold
    verify_frozen(root)
    result=dict(status='PASS',scope='ACCEPTED_PAIRED_RANKING_PRETRAINING_STUDY_V1',
        training=reports,teacher_fit=fit_reports,evaluations=evaluations,paired=paired,
        development_gate='READY_FOR_RL_STUDY_DESIGN' if gate else 'PRETRAINING_GATE_NOT_MET',
        validation_gate='NOT_ESTABLISHED',model_promoted=False,reserved_test_executed=False,
        training_unique_deals=1600,development_unique_deals=26,evaluation_games=2496,
        numerical_audit_batches=12,gradient_scope='first two batches per job, not all historical updates')
    write(root/'controller-audit.json',result)
    return result
