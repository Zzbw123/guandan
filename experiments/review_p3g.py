"""Controller independent disk/replay/arithmetic and CUDA spot audit for P3g."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from p3e_common import check,digest,read,write,normalized,ratios
from p3f_protocol import schedule,config
from guandan.env import HandEnv
from guandan.evaluation.schedule import deal_hands
from guandan.evaluation.statistics import summarize_matchup
from guandan.types import Action
from guandan.agents import GreedyAgent
BASE=ROOT/'artifacts/evaluations/p3f-teacher-v1'
SEEDS=(314380,314381,314382)
MODELS=[f'{phase}-{seed}' for seed in SEEDS for phase in ('phase1','final')]

def lines(p):
    with (gzip.open(p,'rt',encoding='utf-8') if str(p).endswith('.gz') else open(p,encoding='utf-8')) as f:
        for line in f:
            if line.strip():yield json.loads(line)

def canonical(x):
    return sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def close(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(float,int)) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    return a==b

def interval(values):
    rng=Random(314395);groups=[values[i::13] for i in range(13)];draws=[]
    check(len(values)==26 and all(len(g)==2 for g in groups),'26 grouped deals')
    for _ in range(5000):draws.append(sum(g[rng.randrange(2)] for g in groups for _ in range(2))/26)
    draws.sort()
    def pct(p):
        x=4999*p;i=int(x);f=x-i;return draws[i]*(1-f)+draws[min(i+1,4999)]*f
    return [pct(.025),pct(.975)]

def validate_rows(rows):
    trials=schedule();check(len(rows)==len(trials),'full evaluation rows')
    by={r['trial_id']:r for r in rows};check(len(by)==len(trials),'duplicate trial')
    for t in trials:
        check(t.trial_id in by,'missing scheduled trial');r=by[t.trial_id]
        check(all(r[k]==normalized(v) for k,v in asdict(t).items()),'scheduled row metadata')
        check(r['status']=='ok' and r['illegal_actions']==r['timeouts']==0 and r['win'] in (0,1),'successful result')
    return by

def scope(report):
    check(report['model_promoted'] is False and report['reserved_test_executed'] is False,'scope flags')

def audit_eval(root,seed):
    path=root/'evaluations'/f'teacher-phase1-{seed}';r=read(path/'report.json');b=read(path/'binding.json')
    scope(r);check(r['validation_executed'] is r['training_executed'] is False,'no new validation/training')
    check(r['status']=='PASS' and r['games']==208 and r['worker_closed'] is True and r['worker_exitcode'] in (0,-15),'eval completion')
    check(r['binding_sha256']==digest(path/'binding.json'),'binding hash')
    check(b['preregistration_sha256']==r['preregistration_sha256']==digest(root/'preregistration.json'),'prereg binding')
    expected=BASE/'training'/f'teacher-{seed}'/'phase1'
    check(Path(b['checkpoint']).resolve()==expected.resolve(),'fixed raw source')
    check(digest(expected/'raw.pt')==b['candidate_sha256']==r['candidate_sha256'],'raw pin')
    check(digest(expected/'manifest.json')==b['manifest_sha256'],'manifest pin')
    for name,h in r['artifact_sha256'].items():check(digest(path/name)==h,'evaluation artifact hash')
    rows=list(lines(path/'results.jsonl'));by=validate_rows(rows);trials={t.trial_id:t for t in schedule()}
    summary=normalized(summarize_matchup(rows,list(trials.values()),5000,314395))
    check(close(summary,r['summary']['dmc|dmc|greedy']),'summary recomputation')
    values=[sum(x['win'] for x in rows if x['deal_seed']==s)/8 for s in range(108100,108126)]
    check(close(interval(values),summary['metrics']['win_rate']['ci95']),'independent grouped bootstrap')
    mm=list(lines(path/'measurements.jsonl.gz'));bb=list(lines(path/'behavior.jsonl'));rr=list(lines(path/'replays.jsonl.gz'))
    for ls in (mm,bb,rr):check(len(ls)==208 and {v['trial_id'] for v in ls}==set(trials),'artifact full coverage')
    measurements={x['trial_id']:x for x in mm};behaviors={x['trial_id']:x for x in bb};total=Counter()
    for item in rr:
        tid=item['trial_id'];t=trials[tid];row=by[tid];m=measurements[tid];replay=item['replay'];local=Counter();examples=[]
        env=HandEnv.from_hands(deal_hands(t),t.level,t.starting_player)
        check([list(h) for h in env.state.initial_hands]==replay['initial_hands'] and env.state_digest()==replay['initial_digest'],'scheduled initial replay')
        n=len(replay['steps']);check(n==row['steps'] and m['replay_verified'],'replay steps')
        check(all(len(m[k])==n for k in ('candidate_counts','decision_ms','enumeration_ms')),'measurement coverage')
        check(all(math.isfinite(v) and 0<=v<=2000 for v in m['decision_ms']),'decision deadline')
        neural={v['step']:v for v in m['neural']};expected_neural=[]
        for i,st in enumerate(replay['steps']):
            seat=st['player'];obs=env.observe(seat);legal=env.legal_actions(seat);a=Action.from_dict(st['action'])
            check(a in legal and len(legal)==m['candidate_counts'][i],'complete candidate enumeration')
            if seat%2==t.focal_team:
                expected_neural.append(i);v=neural[i]
                check(v['scored_candidates']==len(legal) and v['device']['type']=='cuda' and 0<=v['inference_ms']<=v['roundtrip_ms']<=2000,'CUDA scoring coverage')
                nonpass=[x for x in legal if x.kind!='pass'];passing=a.kind=='pass';optional=bool(nonpass) and any(x.kind=='pass' for x in legal)
                finish=any(x.kind!='pass' and len(x.cards)==len(obs.hand) for x in legal)
                mate=obs.last_player is not None and obs.last_player%2==seat%2 and bool(nonpass) and obs.last_action is not None
                end=len(obs.hand)<=5
                local.update(dict(decisions=1,choices=int(len(legal)>1),forced_pass=int(not nonpass),optional_pass_opportunities=int(optional),optional_passes=int(optional and passing),teammate_response_opportunities=int(mate),teammate_overtakes=int(mate and not passing),finish_opportunities=int(finish),missed_finishes=int(finish and len(a.cards)!=len(obs.hand)),endgame_decisions=int(end),endgame_optional_pass_opportunities=int(end and optional),endgame_optional_passes=int(end and optional and passing),lead_decisions=int(obs.last_action is None),cards_played=len(a.cards),legal_candidates=len(legal)))
                local['action_'+a.kind]+=1
                if finish and len(a.cards)!=len(obs.hand) and len(examples)<3:examples.append(dict(step=i,player=seat,hand=list(obs.hand),chosen=a.to_dict(),finishing_action=next(x.to_dict() for x in legal if len(x.cards)==len(obs.hand))))
            env.step(seat,a,state_version=st['state_version']);check(env.state_digest()==st['digest'],'step digest')
        check(list(neural)==expected_neural and len(m['neural'])==len(expected_neural),'neural step identities')
        s=env.state.settlement;check(env.state.terminal and s is not None and env.state_digest()==row['terminal_digest']==replay['final_digest'],'final replay')
        check(row['win']==int(s.winner_team==t.focal_team) and row['finish_order']==list(s.finish_order) and row['team_reward']==s.team_rewards[t.focal_team],'settlement')
        check(normalized(dict(trial_id=tid,counts=dict(local),missed_finish_examples=examples))==behaviors[tid],'independent behavior')
        total.update(local)
    check(dict(total)==r['counts'] and ratios(total)==r['ratios'],'aggregate behavior')
    return dict(games=208,values=values,estimate=sum(values)/26,ci95=interval(values),counts=dict(total))

def independent_metrics(scores,teachers,finishes):
    check(len(scores)==len(teachers)>0 and all(math.isfinite(x) for x in scores+teachers),'finite full scores')
    check(all(type(x) is int and 0<=x<len(scores) for x in finishes) and len(set(finishes))==len(finishes),'finish indices')
    low=min(teachers);high=max(teachers);targets=[0. if high==low else (x-low)*1.6/(high-low)-.8 for x in teachers]
    index=max(range(len(scores)),key=lambda i:scores[i]);n=len(scores)
    return dict(candidate_count=n,argmax_index=index,teacher_top1=teachers[index]==high,normalized_regret=0. if high==low else (high-teachers[index])/(high-low),candidate_mse=sum((a-b)**2 for a,b in zip(scores,targets))/n,min=min(scores),max=max(scores),spread=max(scores)-min(scores),saturated_count=sum(abs(x)>=.99 for x in scores),finish_opportunity=bool(finishes),finish_miss=index not in finishes if finishes else None)

def validate_score(row,source,gid,step,obs,legal):
    check((row['source'],row['game_id'],row['step'],row['player'],row['hand_size'])==(source,gid,step,obs.player_id,len(obs.hand)),'score state identity')
    check(row['candidate_sha256']==canonical([a.to_dict() for a in legal]) and row['observation_sha256']==canonical(asdict(obs)),'candidate/observation binding')
    teachers=[GreedyAgent().score(obs,a) for a in legal];lo=min(teachers);hi=max(teachers)
    targets=[0. if hi==lo else 1.6*(x-lo)/(hi-lo)-.8 for x in teachers]
    finishes=[i for i,a in enumerate(legal) if a.kind!='pass' and len(a.cards)==len(obs.hand)]
    check(close(teachers,row['teacher_scores']) and close(targets,row['targets']) and finishes==row['finish_indexes'],'teacher targets/finish set')
    check(set(row['models'])==set(MODELS),'all six fixed models')
    for name,entry in row['models'].items():check(close(independent_metrics(entry['scores'],teachers,finishes),entry['metrics']),'independent metric equations')

def consume_score(stream,context):
    row=next(stream,None);check(row is not None,'missing score state');validate_score(row,*context);return row

def corpus(root,ledger):
    trials={t.trial_id:t for t in schedule()}
    specs=[('training',BASE/'training/teacher-314380/phase1/replays.jsonl.gz')]
    specs += [(f'final-{s}',BASE/'evaluations'/f'teacher-{s}'/'replays.jsonl.gz') for s in SEEDS]
    specs += [(f'phase1-{s}',root/'evaluations'/f'teacher-phase1-{s}'/'replays.jsonl.gz') for s in SEEDS]
    for source,path in specs:
        seen=set();counts=Counter();per_game=[];ledger[source]=counts
        for item in lines(path):
            replay=item['replay'];training=source=='training';gid=str(item['seed']) if training else item['trial_id']
            check(gid not in seen,'source duplicate game');seen.add(gid)
            if training:
                seed=item['seed'];check(100000<=seed<100200,'training corpus seed')
                env=HandEnv();env.reset(seed,initial_level=item['level'],starting_player=item['starting_player']);focal=None
            else:
                t=trials[gid];env=HandEnv.from_hands(deal_hands(t),t.level,t.starting_player);focal=t.focal_team
            check([list(h) for h in env.state.initial_hands]==replay['initial_hands'] and env.state_digest()==replay['initial_digest'],'corpus initial state')
            other=0;selected=0;opportunities=0
            for i,st in enumerate(replay['steps']):
                seat=st['player'];obs=env.observe(seat);legal=env.legal_actions(seat);a=Action.from_dict(st['action'])
                check(a in legal,'corpus legal action')
                counts['all_decisions_reenumerated']+=1
                if training or seat%2==focal:
                    counts['eligible_decisions']+=1
                    finish=any(x.kind!='pass' and len(x.cards)==len(obs.hand) for x in legal)
                    counts['finish_opportunities']+=int(finish);opportunities+=int(finish)
                    counts['other_choice_opportunities']+=int(not finish and len(legal)>1)
                    if finish or (len(legal)>1 and other<4):
                        if not finish:other+=1
                        selected+=1;counts['selected_states']+=1;counts['selected_candidates']+=len(legal)
                        counts['finish_states_selected']+=int(finish);counts['other_choices_selected']+=int(not finish)
                        yield source,gid,i,obs,legal
                env.step(seat,a,state_version=st['state_version']);check(env.state_digest()==st['digest'],'corpus step digest')
            check(env.state.terminal and env.state_digest()==replay['final_digest'],'corpus final')
            counts['games']+=1;per_game.append(dict(game_id=gid,selected_states=selected,finish_opportunities=opportunities,other_choices_selected=other))
        check(len(seen)==(200 if source=='training' else 208),'corpus game coverage')
        counts['per_game']=per_game

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--write',action='store_true');args=p.parse_args();root=args.root.resolve()
    pre=read(root/'preregistration.json')
    for group in ('source_sha256','input_sha256'):
        for name,h in pre[group].items():check(digest(ROOT/name)==h,f'frozen file changed {name}')
    check(digest(root/'source.zip')==pre['source_zip_sha256'],'source archive hash')
    with zipfile.ZipFile(root/'source.zip') as z:
        check(set(z.namelist())==set(pre['source_sha256']),'source archive coverage')
        for name,h in pre['source_sha256'].items():check(sha256(z.read(name)).hexdigest()==h,'source archived bytes')
    evaluations={};paired={}
    for seed in SEEDS:
        evaluations[str(seed)]=audit_eval(root,seed)
        old=list(lines(BASE/'evaluations'/f'teacher-{seed}'/'results.jsonl'));validate_rows(old)
        final=[sum(r['win'] for r in old if r['deal_seed']==s)/8 for s in range(108100,108126)]
        first=evaluations[str(seed)]['values'];delta=[b-a for a,b in zip(first,final)]
        paired[str(seed)]=dict(phase1=sum(first)/26,final=sum(final)/26,delta=sum(delta)/26,ci95=interval(delta),phase1_values=first,final_values=final)
        print(json.dumps(dict(stage='eval_audit',seed=seed)),flush=True)
    audit_scores(root,pre,evaluations,paired,args.write)

# audit_scores is defined below to keep its numeric sampler separate from replay checks.

def aggregate_metrics(ms):
    n=len(ms);c=sum(x['candidate_count'] for x in ms);sat=sum(x['saturated_count'] for x in ms)
    f=sum(x['finish_opportunity'] for x in ms);miss=sum(x['finish_miss'] is True for x in ms)
    mean=lambda k:sum(x[k] for x in ms)/n if n else None
    return dict(observations=n,mean_teacher_top1=mean('teacher_top1'),mean_normalized_regret=mean('normalized_regret'),mean_candidate_mse=mean('candidate_mse'),mean_spread=mean('spread'),candidates=c,saturated_candidates=sat,saturation_rate=sat/c if c else None,finish_opportunities=f,finish_misses=miss,finish_miss_rate=miss/f if f else None)

def reject(name,fn):
    try:fn()
    except (ValueError,AssertionError,KeyError,IndexError,StopIteration):return name
    raise ValueError('negative case not rejected: '+name)

def audit_scores(root,pre,evaluations,paired,save):
    import torch
    from guandan_gpu.training import GPUTrainer
    from guandan.learning.encoding import encode_action,encode_observation
    from p3f_training import model_digest
    report=read(root/'scores/report.json');freeze=read(root/'scores/input-freeze.json')
    check(report['status']=='PASS' and report['model_promoted'] is report['training_executed'] is report['validation_read'] is False,'score scope')
    check(report['scores_sha256']==digest(root/'scores/scores.jsonl.gz') and report['input_freeze_sha256']==digest(root/'scores/input-freeze.json'),'score output hashes')
    check(report['preregistration_sha256']==freeze['preregistration_sha256']==digest(root/'preregistration.json'),'score prereg')
    for name,h in freeze['new_input_sha256'].items():check(digest(ROOT/name)==h,'new eval frozen')
    models={}
    for name in MODELS:
        phase,seed=name.split('-');folder=BASE/'training'/f'teacher-{seed}'/phase
        path=folder/('raw.pt' if phase=='phase1' else 'checkpoint.pt');rel=path.relative_to(ROOT).as_posix();metadata=report['models'][name]
        check(metadata['file']==rel and metadata['file_sha256']==pre['input_sha256'][rel]==digest(path),'model identity')
        # Independently extract only state tensors from hash-pinned payload; no optimizer or counter restoration.
        payload=torch.load(path,map_location='cpu',weights_only=True)
        trainer=GPUTrainer(config(int(seed)));trainer.model.load_state_dict(payload['model'],strict=True);trainer.model.eval()
        check(model_digest(trainer.model)==metadata['model_sha256'] and metadata['device']=='cuda:0','model tensor binding')
        models[name]=trainer.model;del trainer
    scores=iter(lines(root/'scores/scores.jsonl.gz'));ledger={};gather=defaultdict(lambda:defaultdict(list));sampled=Counter();numeric=[];first=None;second=None;total=0
    for source,gid,step,obs,legal in corpus(root,ledger):
        context=(source,gid,step,obs,legal);row=consume_score(scores,context);total+=1
        if first is None:first=(deepcopy(row),source,gid,step,obs,legal)
        elif second is None:second=(deepcopy(row),source,gid,step,obs,legal)
        for name in MODELS:gather[source][name].append(independent_metrics(row['models'][name]['scores'],row['teacher_scores'],row['finish_indexes']))
        if sampled[source]<16:
            for name,model in models.items():
                direct=[]
                with torch.no_grad():
                    state=encode_observation(obs)
                    for start in range(0,len(legal),37):
                        aa=legal[start:start+37]
                        ss=torch.tensor([state for _ in aa],dtype=torch.float32,device='cuda')
                        ac=torch.tensor([encode_action(a) for a in aa],dtype=torch.float32,device='cuda')
                        direct.extend(model.forward(ss,ac).cpu().tolist())
                recorded=row['models'][name]['scores'];err=max(abs(a-b) for a,b in zip(direct,recorded))
                i=max(range(len(direct)),key=direct.__getitem__);j=max(range(len(recorded)),key=recorded.__getitem__)
                gap=max(direct[i]-direct[j],recorded[j]-recorded[i]);check(err<=1e-6 and (i==j or gap<=2e-6),'independent direct forward')
                numeric.append(dict(source=source,game_id=gid,step=step,model=name,scores=direct,max_error=err,argmax_changed=i!=j,choice_gap=gap))
            sampled[source]+=1
    check(next(scores,None) is None,'extra/duplicate score states')
    check(close(ledger,report['sources']),'independent corpus counts')
    aggs={s:{m:{'all':aggregate_metrics(ms),'multi_choice':aggregate_metrics([x for x in ms if x['candidate_count']>1])} for m,ms in group.items()} for s,group in gather.items()}
    check(close(aggs,report['aggregates']),'independent aggregate arithmetic')
    negatives=[];row,*context=first
    for name,change in [('state_identity',lambda x:x.update(step=x['step']+1)),('candidate_hash',lambda x:x.update(candidate_sha256='0'*64)),('teacher_target',lambda x:x['targets'].__setitem__(0,x['targets'][0]+.1)),('metric',lambda x:x['models'][MODELS[0]]['metrics'].update(candidate_mse=-1)),('nonfinite_score',lambda x:x['models'][MODELS[0]]['scores'].__setitem__(0,float('nan')))]:
        bad=deepcopy(row);change(bad);negatives.append(reject(name,lambda:validate_score(bad,*context)))
    rows=list(lines(root/'evaluations/teacher-phase1-314380/results.jsonl'))
    negatives.append(reject('missing_trial',lambda:validate_rows(rows[:-1])))
    negatives.append(reject('duplicate_trial',lambda:validate_rows(rows[:-1]+[rows[0]])))
    bad=deepcopy(rows);bad[0]['deal_seed']=9000000;negatives.append(reject('wrong_schedule',lambda:validate_rows(bad)))
    negatives.append(reject('wrong_promotion',lambda:scope(dict(model_promoted=True,reserved_test_executed=False))))
    # The streaming one-to-one corpus check is what rejects missing, duplicate or reordered score rows.
    def sequence_check(values):
        stream=iter(values)
        for record in (first,second):consume_score(stream,record[1:])
        check(next(stream,None) is None,'extra score state')
    negatives.append(reject('missing_score_state',lambda:sequence_check([first[0]])))
    negatives.append(reject('duplicate_score_state',lambda:sequence_check([first[0],first[0]])))
    for group in ('source_sha256','input_sha256'):
        for name,h in pre[group].items():check(digest(ROOT/name)==h,'post audit frozen files')
    result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),preregistration_sha256=digest(root/'preregistration.json'),new_evaluation_games=624,reused_evaluation_games=624,source_counts=ledger,selected_states=total,candidate_sets=sum(v['selected_candidates'] for v in ledger.values()),model_score_values=6*sum(v['selected_candidates'] for v in ledger.values()),evaluations=evaluations,paired=paired,aggregates=aggs,direct_forward_states=dict(sampled),direct_forward_comparisons=len(numeric),max_forward_error=max(x['max_error'] for x in numeric),negative_cases=negatives,model_promoted=False,reserved_test_executed=False,training_executed=False,overall_P3='PARTIALLY_ACCEPTED')
    if save:
        with gzip.open(root/'controller-forward.jsonl.gz','xt',encoding='utf-8') as f:
            for item in numeric:f.write(json.dumps(item,allow_nan=False)+'\n')
        result['direct_forward_sha256']=digest(root/'controller-forward.jsonl.gz')
        write(root/'controller-audit.json',result)
    print(json.dumps(dict(status=result['status'],selected_states=total,candidate_sets=result['candidate_sets'],paired=paired,negative_cases=negatives)),flush=True)

if __name__=='__main__':main()
