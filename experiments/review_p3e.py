"""Controller disk audit: frozen schedules, replay, paired arithmetic and provenance."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
from random import Random
import statistics
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from p3e_protocol import specification,schedule,INITS,OLD_HASH
from p3e_common import check,digest,read,write,normalized,summarize,behavior,ratios
from p3e_metrics import sum_metrics
from guandan.env import HandEnv
from guandan.evaluation.statistics import cluster_bootstrap
from guandan.evaluation.schedule import deal_hands
from guandan.types import Action

def lines(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]

def independent_interval(values,levels):
    groups=[[v for v,l in zip(values,levels) if l==level] for level in range(2,15)]
    check(all(len(g)>=2 for g in groups),'bootstrap group sizes')
    rng=Random(314375);draws=[]
    for _ in range(5000):draws.append(sum(g[rng.randrange(len(g))] for g in groups for _ in g)/len(values))
    draws.sort()
    def percentile(p):
        x=(len(draws)-1)*p;i=int(x);f=x-i
        return draws[i]*(1-f)+draws[min(i+1,len(draws)-1)]*f
    return [percentile(.025),percentile(.975)]

def verify_stats(rows,trials,reported):
    computed=normalized(summarize(rows,trials));check(computed==reported,'summary recomputation')
    for matchup,summary in reported.items():
        selected=[r for r in rows if r['matchup_id']==matchup]
        seeds=sorted({t.deal_seed for t in trials if t.matchup_id==matchup})
        vals=[sum(r['win'] for r in selected if r['deal_seed']==s)/8 for s in seeds]
        levels=[next(t.level for t in trials if t.deal_seed==s) for s in seeds]
        metric=summary['metrics']['win_rate']
        check(math.isclose(sum(vals)/len(vals),metric['estimate'],abs_tol=1e-14),'independent win arithmetic')
        check(all(math.isclose(a,b,abs_tol=1e-14) for a,b in zip(independent_interval(vals,levels),metric['ci95'])),'independent interval')
    return computed

def audit_evaluation(root,job):
    path=root/'evaluations'/job;report=read(path/'report.json');binding=read(path/'binding.json')
    check(report['model_promoted'] is report['reserved_test_executed'] is False,'scope flags')
    check(ratios(report['counts'])==report['ratios'],'reported denominator integrity')
    trials=schedule(job=='validation',job=='baseline');expected={t.trial_id:t for t in trials}
    check(binding['preregistration_sha256']==digest(root/'preregistration.json') and binding['games']==len(trials),'evaluation prereg binding')
    if binding['checkpoint']:
        checkpoint=ROOT/binding['checkpoint']
        check(digest(checkpoint/'checkpoint.pt')==binding['candidate_sha256']==report['candidate_sha256'],'evaluation candidate binding')
        if job=='old':check(binding['candidate_sha256']==OLD_HASH,'old fixed candidate')
        else:
            seed,hands=('314370','1600') if job=='validation' else job.split('-')
            check(checkpoint.resolve()==(root/'training'/seed/f'wave-{int(hands)//4}').resolve(),'fixed checkpoint selection')
    else:check(job=='baseline' and report['candidate_sha256'] is None,'baseline binding')
    rows=lines(path/'results.jsonl');verify_stats(rows,trials,report['summary'])
    measures=lines(path/'measurements.jsonl.gz');replays=lines(path/'replays.jsonl.gz');counts=lines(path/'behavior.jsonl')
    for values in (measures,replays,counts):
        check(len(values)==len(expected) and {v['trial_id'] for v in values}==set(expected),'complete artifact schedule')
    byrow={r['trial_id']:r for r in rows};bym={m['trial_id']:m for m in measures};byb={b['trial_id']:b for b in counts}
    recomputed=[];decision_times=[];neural_candidates=0;independent=Counter()
    for item in replays:
        tid=item['trial_id'];trial=expected[tid];r=byrow[tid];m=bym[tid];replay=item['replay']
        env=HandEnv.replay(replay);settlement=env.state.settlement
        check(env.state.terminal and env.state_digest()==r['terminal_digest'],'terminal digest')
        check(int(settlement.winner_team==trial.focal_team)==r['win'] and list(settlement.finish_order)==r['finish_order'],'replay settlement')
        n=r['steps'];check(n==len(replay['steps']),'replay step count')
        check(m['replay_verified'] and all(len(m[k])==n for k in ('enumeration_ms','decision_ms','candidate_counts')),'measurement step coverage')
        check(all(math.isfinite(v) and 0<=v<=2000 for v in m['decision_ms']),'decision deadline')
        check(all(math.isfinite(v) and v>=0 for v in m['enumeration_ms']),'enumeration times')
        expected_neural=[i for i,s in enumerate(replay['steps']) if trial.policies[s['player']]=='dmc']
        check([v['step'] for v in m['neural']]==expected_neural,'neural step coverage')
        for v in m['neural']:
            check(v['scored_candidates']==m['candidate_counts'][v['step']] and v['device']['type']=='cuda'
                and 0<=v['inference_ms']<=v['roundtrip_ms']<=2000,'complete CUDA candidate scoring')
            neural_candidates+=v['scored_candidates'];decision_times.append(v['roundtrip_ms'])
        # Independent implementation of every behavior count, from a new replay traversal.
        state=HandEnv.from_hands(deal_hands(trial),trial.level,trial.starting_player)
        check([list(h) for h in state.state.initial_hands]==replay['initial_hands'],'scheduled initial hands')
        local=Counter();examples=[]
        for i,s in enumerate(replay['steps']):
            seat=s['player'];legal=state.legal_actions(seat);obs=state.observe(seat);action=Action.from_dict(s['action'])
            check(len(legal)==m['candidate_counts'][i],'full legal enumeration count')
            if seat%2==trial.focal_team:
                independent['decisions']+=1
                optional=any(a.kind=='pass' for a in legal) and any(a.kind!='pass' for a in legal)
                finish=any(a.kind!='pass' and len(a.cards)==len(obs.hand) for a in legal)
                independent['optional_pass_opportunities']+=int(optional)
                independent['optional_passes']+=int(optional and action.kind=='pass')
                independent['finish_opportunities']+=int(finish)
                independent['missed_finishes']+=int(finish and len(action.cards)!=len(obs.hand))
                nonpass=[v for v in legal if v.kind!='pass'];passing=action.kind=='pass'
                teammate=(obs.last_player is not None and obs.last_player%2==seat%2
                          and bool(nonpass) and obs.last_action is not None)
                endgame=len(obs.hand)<=5
                local.update(dict(decisions=1,choices=int(len(legal)>1),forced_pass=int(not nonpass),
                    optional_pass_opportunities=int(optional),optional_passes=int(optional and passing),
                    teammate_response_opportunities=int(teammate),teammate_overtakes=int(teammate and not passing),
                    finish_opportunities=int(finish),missed_finishes=int(finish and len(action.cards)!=len(obs.hand)),
                    endgame_decisions=int(endgame),endgame_optional_pass_opportunities=int(endgame and optional),
                    endgame_optional_passes=int(endgame and optional and passing),lead_decisions=int(obs.last_action is None),
                    cards_played=len(action.cards),legal_candidates=len(legal)))
                local['action_'+action.kind]+=1
                if finish and len(action.cards)!=len(obs.hand) and len(examples)<3:
                    examples.append(dict(step=i,player=seat,hand=list(obs.hand),chosen=action.to_dict(),
                        finishing_action=next(v.to_dict() for v in legal if len(v.cards)==len(obs.hand))))
            state.step(seat,action,state_version=s['state_version'])
        check(normalized(dict(trial_id=tid,counts=dict(local),missed_finish_examples=examples))==byb[tid],
              'independent behavior replay counts/examples')
        recomputed.append(dict(local))
    total=sum_metrics(recomputed)
    check(total==report['counts'] and ratios(total)==report['ratios'],'behavior aggregates')
    check(all(total[k]==v for k,v in independent.items()),'independent behavioral arithmetic')
    check(report['status']=='PASS' and report['games']==len(trials) and report['worker_closed'],'successful complete job')
    if job=='validation':
        p=report['summary']['dmc|dmc|greedy']['metrics']['win_rate']
        gate='VALIDATION_GATE_PASSED' if p['estimate']>=.55 and p['ci95'][0]>.5 and not p['degenerate'] else 'NOT_ESTABLISHED'
        check(report['validation_gate']==gate,'validation gate')
    else:check(report['validation_gate']=='DEVELOPMENT_ONLY','development scope')
    return dict(job=job,games=len(rows),counts=total,ratios=ratios(total),summary=report['summary'],
                scored_candidates=neural_candidates,neural_decisions=len(decision_times),
                independent_behavior=dict(independent),candidate_sha256=report['candidate_sha256'])

def audit_training(root,seed):
    import torch
    path=root/'training'/str(seed);waves=lines(path/'waves.jsonl');report=read(path/'report.json')
    check(len(waves)==400,'400 waves');updates=samples=0;segments=[]
    for i,w in enumerate(waves):
        check(w['wave']==w['learning_version']==i+1 and w['behavior_version']==i and w['episodes']==4*(i+1),'wave synchronization')
        check([(h['seed'],h['level'],h['starting_player']) for h in w['hands']]==[(105000+j,2+j%13,j%4) for j in range(4*i,4*i+4)],'fixed training deals')
        check(all(h['replay_verified'] and 0<h['steps']==h['samples']<=1000 for h in w['hands']),'training replay/counters')
        n=sum(h['samples'] for h in w['hands']);u=math.ceil(n/256);samples+=n;updates+=u
        check(w['samples']==n and w['updates']==updates and math.isfinite(w['mean_loss']),'learning samples/updates')
        d=w['device_proof'];check(d==dict(device='cuda:0',dtype='torch.float32',forward_checks=u,gradient_checks=7*u,adam_tensor_checks=21*u),'CUDA proof')
        if (i+1)%100==0:
            cp=path/f'wave-{i+1}';manifest=read(cp/'manifest.json')
            check(digest(cp/'checkpoint.pt')==manifest['sha256'],'training checkpoint hash')
            payload=torch.load(cp/'checkpoint.pt',map_location='cpu',weights_only=True)
            for key in ('config','episodes','updates','waves','used_deal_seeds','versions','runtime','sources','boundary'):
                check(payload[key]==manifest[key],f'checkpoint metadata {key}')
            check(payload['config']==next(c for c in specification()['configs'] if c['seed']==seed),'frozen config')
            source_hashes=read(root/'preregistration.json')['source_sha256']
            check(payload['sources']=={name[4:]:value for name,value in source_hashes.items() if name.startswith('src/')},'checkpoint frozen sources')
            check(payload['runtime']==report['runtime'] and payload['runtime']['deterministic'] is True
                and payload['runtime']['threads']==1 and payload['runtime']['tf32_matmul'] is False
                and payload['runtime']['tf32_cudnn'] is False,'deterministic checkpoint runtime')
            check(payload['episodes']==4*(i+1) and payload['waves']==i+1 and payload['updates']==updates and payload['used_deal_seeds']==list(range(105000,105000+4*(i+1))),'checkpoint history')
            check(all(torch.isfinite(v).all() for v in payload['model'].values()),'finite model')
            check(set(payload['optimizer']['state'])==set(range(7)),'complete Adam')
            check(all(v['step'].item()==updates and all(torch.isfinite(t).all() for t in v.values()) for v in payload['optimizer']['state'].values()),'Adam state')
            recent=waves[i-99:i+1]
            segments.append(dict(hands=4*(i+1),samples=sum(r['samples'] for r in recent),
                mean_online_loss=sum(r['mean_loss']*r['samples'] for r in recent)/sum(r['samples'] for r in recent)))
    check(report['status']=='PASS' and report['samples']==samples and report['updates']==updates and report['hands']==1600,'training report')
    return dict(seed=seed,hands=1600,samples=samples,updates=updates,elapsed_s=report['elapsed_s'],online_loss_segments=segments)

def audit(root):
    pre=read(root/'preregistration.json');receipt=read(root/'receipt.json')
    check(pre['specification']==specification(),'frozen protocol')
    check(receipt['status']=='RUN_COMPLETE_PENDING_AUDIT' and receipt['model_promoted'] is receipt['reserved_test_executed'] is False,'run receipt scope')
    for name,value in receipt['artifact_sha256'].items():check(digest(root/name)==value,f'artifact hash {name}')
    check(pre['seed_scan_sha256']==digest(ROOT/'artifacts/evaluations/p3e-seed-availability.json'),'unseen seed receipt')
    with zipfile.ZipFile(root/'source-snapshot.zip') as z:
        check(set(z.namelist())==set(pre['source_sha256']),'snapshot members')
        from hashlib import sha256
        for name,value in pre['source_sha256'].items():check(digest(ROOT/name)==value==sha256(z.read(name)).hexdigest(),f'source {name}')
    protected={}
    for name,key in [('p2-validation-v1','source_sha256'),('p3a-cpu-v1','source_sha256'),('p3b-gpu-v1','source_sha256_before'),('p3c-gpu-wave-v1','source_sha256'),('p3d-validation-v2','source_sha256')]:
        hashes=read(ROOT/f'artifacts/evaluations/{name}/preregistration.json')[key]
        for path,value in hashes.items():check(digest(ROOT/path)==value,f'history {name}/{path}')
        protected[name]=len(hashes)
    training=[audit_training(root,s) for s in INITS]
    evaluations={}
    for job in ['old','baseline']+specification()['development_order']+['validation']:
        evaluations[job]=audit_evaluation(root,job);print(f'audited {job}',flush=True)
    paired=[]
    for seed in INITS:
        before=evaluations[f'{seed}-400']['summary']['dmc|dmc|greedy']
        after=evaluations[f'{seed}-1600']['summary']['dmc|dmc|greedy']
        values=[b['win_rate']-a['win_rate'] for a,b in zip(before['clusters'],after['clusters'])]
        levels=[g['level'] for g in before['clusters']]
        interval=independent_interval(values,levels);other=cluster_bootstrap(values,levels,5000,314375)
        check(all(math.isclose(a,b,abs_tol=1e-14) for a,b in zip(interval,other['ci95'])),'paired interval')
        paired.append(dict(seed=seed,win_rate_400=before['metrics']['win_rate']['estimate'],
            win_rate_1600=after['metrics']['win_rate']['estimate'],delta_pp=100*sum(values)/26,
            ci95_delta_pp=[100*x for x in interval],group_delta=values,
            inference_scope='descriptive conditional on this initializer, 26 original deals'))
    stability={}
    for budget in (400,1600):
        values=[r[f'win_rate_{budget}'] for r in paired]
        stability[str(budget)]=dict(n_initializers=3,mean=statistics.mean(values),
            sd=statistics.stdev(values),min=min(values),max=max(values),values=values)
    final=read(root/'evaluations/validation/report.json')
    return dict(status='PASS',engineering='ACCEPTED_DIAGNOSTIC_SCALE_STUDY_V1',
        model_promoted=False,reserved_test_executed=False,validation_gate=final['validation_gate'],
        candidate_sha256=final['candidate_sha256'],training=training,evaluations=evaluations,
        paired=paired,stability=stability,protected_sources=protected,
        review_sha256=digest(Path(__file__)),receipt_sha256=digest(root/'receipt.json'))

def negative_checks(root):
    path=root/'evaluations/314370-400';rows=lines(path/'results.jsonl');trials=schedule();report=read(path/'report.json')
    outcomes={}
    cases={'missing':rows[:-1],'duplicate':rows[:-1]+[rows[0]],'wrong_win':deepcopy(rows)}
    cases['wrong_win'][0]['win']=1-cases['wrong_win'][0]['win']
    for name,data in cases.items():
        try:verify_stats(data,trials,report['summary'])
        except ValueError:outcomes[name]='REJECTED'
        else:raise AssertionError(name)
    wrong=deepcopy(report['summary']);wrong['dmc|dmc|greedy']['metrics']['win_rate']['ci95'][0]=-.1
    try:verify_stats(rows,trials,wrong)
    except ValueError:outcomes['wrong_interval']='REJECTED'
    else:raise AssertionError('wrong_interval')
    from unittest.mock import patch
    # Inject corrupted disk reads without rewriting any frozen experiment files.
    actual_read=read
    for name in ('fake_promotion','behavior_denominator','candidate_selection'):
        def altered(p):
            value=actual_read(p)
            if Path(p)==path/'report.json':
                if name=='fake_promotion':value['model_promoted']=True
                elif name=='behavior_denominator':value['counts']['finish_opportunities']+=1
            if Path(p)==path/'binding.json' and name=='candidate_selection':
                value['checkpoint']=value['checkpoint'].replace('wave-100','wave-400')
            return value
        with patch(__name__+'.read',altered):
            try:audit_evaluation(root,'314370-400')
            except ValueError:outcomes[name]='REJECTED'
            else:raise AssertionError(name)
    return dict(status='PASS',cases=outcomes,review_sha256=digest(Path(__file__)))

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--write',action='store_true');a=p.parse_args()
    root=a.directory.resolve();result=audit(root)
    if a.write:
        write(root/'controller-audit.json',result);write(root/'controller-negative.json',negative_checks(root))
    print(json.dumps({k:v for k,v in result.items() if k not in ('evaluations',)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
