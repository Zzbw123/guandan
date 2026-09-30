"""Controller disk audit: frozen schedules, replay, paired arithmetic and provenance."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
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
from p3l_protocol import specification,schedule,config,INITS,ARMS
from p3e_common import check,digest,read,write,normalized,ratios
from p3e_metrics import sum_metrics
from guandan.env import HandEnv
from guandan.evaluation.statistics import cluster_bootstrap,summarize_matchup
from guandan.evaluation.schedule import deal_hands
from guandan.types import Action

def lines(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]

def independent_interval(values,levels):
    groups=[[v for v,l in zip(values,levels) if l==level] for level in range(2,15)]
    check(all(len(g)>=2 for g in groups),'bootstrap group sizes')
    rng=Random(314425);draws=[]
    for _ in range(5000):draws.append(sum(g[rng.randrange(len(g))] for g in groups for _ in g)/len(values))
    draws.sort()
    def percentile(p):
        x=(len(draws)-1)*p;i=int(x);f=x-i
        return draws[i]*(1-f)+draws[min(i+1,len(draws)-1)]*f
    return [percentile(.025),percentile(.975)]

def summarize(rows,trials):
    return {key:summarize_matchup([r for r in rows if r["matchup_id"]==key], [t for t in trials if t.matchup_id==key],5000,314425) for key in sorted({t.matchup_id for t in trials})}

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
    validation=job.startswith('validation-');trials=schedule(validation);expected={t.trial_id:t for t in trials}
    check(binding['preregistration_sha256']==digest(root/'preregistration.json') and binding['games']==len(trials),'evaluation prereg binding')
    arm=job.removeprefix('validation-') if validation else job.split('-')[0]
    seed=314380 if validation else int(job.split('-')[1])
    checkpoint=Path(binding['checkpoint'])
    check(checkpoint.resolve()==(root/'training'/f'{arm}-{seed}'/'final').resolve(),'fixed checkpoint selection')
    check(digest(checkpoint/'checkpoint.pt')==binding['candidate_sha256']==report['candidate_sha256'],'candidate SHA')
    check(digest(checkpoint/'manifest.json')==binding['manifest_sha256'],'manifest SHA')
    check(report['binding_sha256']==digest(path/'binding.json'),'binding SHA')
    for name,value in report['artifact_sha256'].items():check(digest(path/name)==value,'evaluation file hash')
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
    # Frozen GPUInferenceGuard.close() deliberately terminates then joins the
    # worker. Windows multiprocessing maps its TERMINATE code to -SIGTERM (-15).
    check(report['status']=='PASS' and report['games']==len(trials) and report['worker_closed'] is True
          and type(report['worker_exitcode']) is int and report['worker_exitcode'] in (0,-15),
          'successful complete job and bounded guard teardown')
    if validation and arm=='normcap':
        p=report['summary']['dmc|dmc|greedy']['metrics']['win_rate']
        gate='VALIDATION_GATE_PASSED' if p['estimate']>=.55 and p['ci95'][0]>.5 and not p['degenerate'] else 'NOT_ESTABLISHED'
        check(report['validation_gate']==gate,'validation gate')
    else:check(report['validation_gate']==('SECONDARY_VALIDATION_ONLY' if validation else 'DEVELOPMENT_ONLY'),'development/secondary scope')
    return dict(job=job,games=len(rows),counts=total,ratios=ratios(total),summary=report['summary'],
                scored_candidates=neural_candidates,neural_decisions=len(decision_times),
                independent_behavior=dict(independent),candidate_sha256=report['candidate_sha256'])
