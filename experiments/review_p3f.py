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
from p3f_protocol import specification,schedule,config,INITS,ARMS
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
    rng=Random(314385);draws=[]
    for _ in range(5000):draws.append(sum(g[rng.randrange(len(g))] for g in groups for _ in g)/len(values))
    draws.sort()
    def percentile(p):
        x=(len(draws)-1)*p;i=int(x);f=x-i
        return draws[i]*(1-f)+draws[min(i+1,len(draws)-1)]*f
    return [percentile(.025),percentile(.975)]

def summarize(rows,trials):
    return {key:summarize_matchup([r for r in rows if r["matchup_id"]==key], [t for t in trials if t.matchup_id==key],5000,314385) for key in sorted({t.matchup_id for t in trials})}

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
    if validation:
        p=report['summary']['dmc|dmc|greedy']['metrics']['win_rate']
        gate='VALIDATION_GATE_PASSED' if p['estimate']>=.55 and p['ci95'][0]>.5 and not p['degenerate'] else 'NOT_ESTABLISHED'
        check(report['validation_gate']==gate,'validation gate')
    else:check(report['validation_gate']=='DEVELOPMENT_ONLY','development scope')
    return dict(job=job,games=len(rows),counts=total,ratios=ratios(total),summary=report['summary'],
                scored_candidates=neural_candidates,neural_decisions=len(decision_times),
                independent_behavior=dict(independent),candidate_sha256=report['candidate_sha256'])

def tensor_digest(state):
    from hashlib import sha256
    h=sha256()
    for name,tensor in sorted(state.items()):
        raw=tensor.detach().cpu().contiguous().numpy().tobytes()
        meta=json.dumps([name,str(tensor.dtype),list(tensor.shape)],ensure_ascii=False,allow_nan=False,sort_keys=True).encode()
        h.update(len(meta).to_bytes(8,'big')+meta+len(raw).to_bytes(8,'big')+raw)
    return h.hexdigest()

def verify_optimizer(payload,updates):
    import torch
    check(set(payload['optimizer']['state'])==set(range(7)),'all Adam parameter states')
    for value in payload['optimizer']['state'].values():
        check(value['step'].item()==updates,'Adam step count')
        check(all(torch.isfinite(t).all() for t in value.values()),'finite Adam tensors')
    check(all(torch.isfinite(t).all() for t in payload['model'].values()),'finite model tensors')

def verify_reset(reset,manifest,model_hash,seed):
    from hashlib import sha256
    check(reset['phase1_file_sha256']==manifest['sha256'],'reset raw checkpoint binding')
    check(reset['phase1_model_sha256']==reset['phase2_initial_model_sha256']==manifest['model_sha256']==model_hash,'exact boundary parameter transfer')
    check(reset['optimizer_empty'] is True and reset['initial_counters']==dict(episodes=0,waves=0,updates=0)
          and reset['used_deal_seeds']==[],'reset counter and Adam boundary')
    wanted=sha256(repr(Random(seed).getstate()).encode()).hexdigest()
    check(reset['policy_rng_repr_sha256']==wanted,'independent policy RNG reset')

def verify_training_replays(path,first,last,teacher=False):
    from guandan.agents import GreedyAgent
    records=lines(path);check(len(records)==last-first,'complete training replay count')
    sizes=[];digests=[];rows=[]
    for i,r in enumerate(records,first):
        check((r['seed'],r['level'],r['starting_player'])==(100000+i,2+i%13,i%4),'training scheduled seed/level/start')
        replay=r['replay'];verified=HandEnv.replay(replay)
        check(verified.state.terminal and verified.state_digest()==r['terminal_digest'],'training authoritative replay')
        env=HandEnv();env.reset(100000+i,initial_level=2+i%13,starting_player=i%4)
        check([list(h) for h in env.state.initial_hands]==replay['initial_hands'],'scheduled training initial hands')
        local=[]
        for s in replay['steps']:
            legal=env.legal_actions(s['player']);action=Action.from_dict(s['action']);local.append(len(legal))
            check(action in legal,'training action in complete legal set')
            if teacher:
                obs=env.observe(s['player'])
                # act implementation not reused: pin first max of pure observed-action scores.
                selected=max(legal,key=lambda a:GreedyAgent().score(obs,a))
                check(action==selected,'all teacher actions use observed heuristic')
            env.step(s['player'],action,state_version=s['state_version'])
            check(env.state_digest()==s['digest'],'training step digest')
        check(env.state.terminal and r['replay_verified'] is True and 0<r['steps']==r['samples']==len(local)<=1000,'training hand counts')
        check(r['legal_candidates']==sum(local),'independent complete training candidate count')
        sizes.extend(local);digests.append(r['terminal_digest']);rows.append({k:v for k,v in r.items() if k!='replay'})
    return dict(hands=len(records),samples=len(sizes),legal_candidates=sum(sizes),hand_rows=rows),sizes,digests

def verify_waves(path,first,last,phase):
    waves=lines(path);check(len(waves)==(last-first)//4,'DMC wave count');updates=samples=0
    for n,w in enumerate(waves):
        check(w['wave']==w['learning_version']==n+1 and w['behavior_version']==n and w['episodes']==4*(n+1),'synchronous wave counter')
        expected=phase['hand_rows'][4*n:4*n+4]
        for hand,record in zip(w['hands'],expected):
            check(all(hand[k]==record[k] for k in ('seed','level','starting_player','steps','samples','terminal_digest','replay_verified')),'wave-to-replay match')
        count=sum(h['samples'] for h in expected);u=math.ceil(count/256);updates+=u;samples+=count
        check(w['samples']==count and w['updates']==updates and math.isfinite(w['mean_loss']),'DMC batch/update counts')
        check(w['device_proof']==dict(device='cuda:0',dtype='torch.float32',forward_checks=u,gradient_checks=7*u,adam_tensor_checks=21*u),'actual CUDA training evidence')
    check(samples==phase['samples'] and updates==phase['updates'],'phase learning totals')
    return dict(samples=samples,updates=updates,online_loss=sum(w['mean_loss']*w['samples'] for w in waves)/samples,
                scored_candidates=sum(w['scored_candidates'] for w in waves))

def verify_teacher_batches(path,sizes,phase):
    batches=lines(path);check(len(batches)==math.ceil(len(sizes)/64),'single-pass teacher update count')
    for i,b in enumerate(batches):
        part=sizes[64*i:64*i+64];n=len(part);count=sum(part);chunks=math.ceil(count/1024)
        check(b['update']==i+1 and b['observation_start']==64*i and b['observation_stop']==64*i+n,'FIFO no repeats/skips')
        check(b['requests']==n and b['per_request_sizes']==part and b['scored_candidates']==count and b['chunks']==chunks,'all candidate objective coverage')
        check(math.isfinite(b['loss']) and b['loss']>=0,'finite teacher loss')
        check(b['device_proof']==dict(device='cuda:0',dtype='float32',forward_checks=chunks,gradient_checks=7,
              parameter_checks=7,adam_tensor_checks=21,optimizer_tensor_checks=21),'teacher CUDA update evidence')
    check(phase['updates']==len(batches),'teacher update total')
    return dict(samples=len(sizes),updates=len(batches),scored_candidates=sum(sizes),
        online_loss=sum(b['loss']*b['requests'] for b in batches)/len(sizes))

def audit_training(root,job):
    import torch
    torch.set_num_threads(1)
    path=root/'training'/job;r=read(path/'report.json');arm,seedtext=job.split('-');seed=int(seedtext)
    check(r['status']=='PASS' and r['job']==job and r['config']==config(seed),'training job contract')
    phase1,sizes,digests=verify_training_replays(path/'phase1/replays.jsonl.gz',0,200,arm=='teacher')
    phase2,_,_=verify_training_replays(path/'phase2-replays.jsonl.gz',200,800)
    for key,computed in (('phase1',phase1),('phase2',phase2)):
        check(all(r[key][k]==v for k,v in computed.items()),'training report independently recomputed')
    p1=(verify_teacher_batches(path/'phase1/batches.jsonl',sizes,r['phase1']) if arm=='teacher'
        else verify_waves(path/'phase1/waves.jsonl',0,200,r['phase1']))
    p2=verify_waves(path/'phase2-waves.jsonl',200,800,r['phase2'])
    for key in ('samples','legal_candidates','updates'):
        check(r['total_'+key]==r['phase1'][key]+r['phase2'][key],'two-phase total '+key)
    check(r['total_hands']==800 and r['phase1']['hands']==200 and r['phase2']['hands']==600,'matched hand budgets')
    manifest=read(path/'phase1/manifest.json');check(manifest==r['phase1_manifest'],'phase1 manifest report')
    check(digest(path/'phase1/raw.pt')==manifest['sha256'],'phase1 raw hash')
    raw=torch.load(path/'phase1/raw.pt',map_location='cpu',weights_only=True)
    for key in ('config','hands','arm','seed','phase1_updates'):check(raw[key]==manifest[key],'raw metadata '+key)
    check(raw['config']==config(seed) and raw['hands']==200 and raw['arm']==arm and raw['phase1_updates']==p1['updates'],'raw phase1 lineage')
    verify_optimizer(raw,p1['updates']);reset=read(path/'reset.json')
    verify_reset(reset,manifest,tensor_digest(raw['model']),seed)
    check(Path(reset['phase1_file']).resolve()==(path/'phase1/raw.pt').resolve(),'exact raw input path')
    final=read(path/'final/manifest.json');check(final==r['final_manifest'],'final manifest report')
    check(digest(path/'final/checkpoint.pt')==final['sha256'],'final checkpoint hash')
    payload=torch.load(path/'final/checkpoint.pt',map_location='cpu',weights_only=True)
    for key in ('config','episodes','updates','waves','used_deal_seeds','versions','runtime','sources','boundary'):
        check(payload[key]==final[key],'final metadata '+key)
    check(payload['episodes']==600 and payload['waves']==150 and payload['updates']==p2['updates'] and
          payload['used_deal_seeds']==list(range(100200,100800)) and payload['config']==config(seed),'honest phase2 counters')
    source=read(root/'preregistration.json')['source_sha256']
    check(payload['sources']=={k[4:]:v for k,v in source.items() if k.startswith('src/')},'final protected engine sources')
    check(payload['runtime']==r['runtime'] and r['runtime']['deterministic'] is True and r['runtime']['threads']==1
          and r['runtime']['tf32_matmul'] is r['runtime']['tf32_cudnn'] is False,'fixed runtime')
    verify_optimizer(payload,p2['updates'])
    return dict(job=job,arm=arm,seed=seed,phase1={**p1,**{k:v for k,v in phase1.items() if k!='hand_rows'},'seconds':r['phase1']['seconds']},
        phase2={**p2,**{k:v for k,v in phase2.items() if k!='hand_rows'},'seconds':r['phase2']['seconds']},
        total_hands=800,total_samples=r['total_samples'],total_updates=r['total_updates'],seconds=r['seconds'],
        initial_model_sha256=r['phase1_initial_model_sha256'],phase1_model_sha256=manifest['model_sha256'],
        final_sha256=final['sha256'],reset=reset,teacher_digests=digests if arm=='teacher' else None)

def reviewed_job(root,kind,job,cache=None):
    """Optional controller work reuse: exact source and immutable input hashes only.

    Default review never uses a cache. This option overlaps completed-job review
    with the still-running final evaluation without repeating expensive replays.
    """
    folder=root/('training' if kind=='training' else 'evaluations')/job
    manifest={p.relative_to(folder).as_posix():digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    provenance=dict(review_sha256=digest(Path(__file__)),preregistration_sha256=digest(root/'preregistration.json'),input_sha256=manifest)
    if cache:
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);path=cache/f'{kind}-{job}.json'
        if path.exists():
            saved=read(path);check(saved['provenance']==provenance,'controller work provenance changed')
            return saved['result']
    pre=read(root/'preregistration.json')
    for name,value in pre['source_sha256'].items():check(digest(ROOT/name)==value,'source during independent review')
    result=(audit_training if kind=='training' else audit_evaluation)(root,job)
    check(manifest=={p.relative_to(folder).as_posix():digest(p) for p in sorted(folder.rglob('*')) if p.is_file()},'immutable reviewed job')
    if cache:write(path,dict(status='PASS',provenance=provenance,result=result))
    return result

def audit(root,cache=None):
    from hashlib import sha256
    pre=read(root/'preregistration.json');receipt=read(root/'receipt.json')
    check(pre['specification']==specification(),'frozen P3f specification')
    check(receipt['status']=='RUN_COMPLETE_PENDING_AUDIT' and receipt['model_promoted'] is receipt['reserved_test_executed'] is False,'receipt scope')
    for name,value in receipt['artifact_sha256'].items():check(digest(root/name)==value,'experiment hash '+name)
    for name,value in pre['prerequisite_sha256'].items():check(digest(ROOT/name)==value,'prerequisite hash')
    with zipfile.ZipFile(root/'source-snapshot.zip') as z:
        check(set(z.namelist())==set(pre['source_sha256']),'snapshot exact members')
        for name,value in pre['source_sha256'].items():check(digest(ROOT/name)==value==sha256(z.read(name)).hexdigest(),'snapshot source '+name)
    tests=read(ROOT/'artifacts/evaluations/p3f-test-receipt.json')
    for name,value in tests['sha256'].items():check(digest(ROOT/name)==value,'tests log provenance')
    protected={}
    for name,key in [('p2-validation-v1','source_sha256'),('p3a-cpu-v1','source_sha256'),('p3b-gpu-v1','source_sha256_before'),('p3c-gpu-wave-v1','source_sha256'),('p3d-validation-v2','source_sha256'),('p3e-scale-v2','source_sha256')]:
        hashes=read(ROOT/f'artifacts/evaluations/{name}/preregistration.json')[key]
        for path,value in hashes.items():check(digest(ROOT/path)==value,'historical source '+path)
        protected[name]=len(hashes)
    training={}
    # Independent immutable job directories; two CPU reviewers bound peak memory.
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(reviewed_job,root,'training',job,cache):job for job in specification()['training_order']}
        for future in as_completed(pending):
            job=pending[future];training[job]=future.result();print('audited training '+job,flush=True)
    for seed in INITS:
        a,b=training[f'control-{seed}'],training[f'teacher-{seed}']
        check(a['initial_model_sha256']==b['initial_model_sha256'],'paired network initialization')
        for key in ('policy_rng_repr_sha256','torch_rng_sha256','cuda_rng_sha256'):
            check(a['reset'][key]==b['reset'][key],'paired random state reset')
    check(training['teacher-314380']['teacher_digests']==training['teacher-314381']['teacher_digests']==training['teacher-314382']['teacher_digests'],'identical greedy teacher demonstrations across initializers')
    evaluations={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(reviewed_job,root,'evaluation',job,cache):job for job in specification()['development_order']+specification()['validation_order']}
        for future in as_completed(pending):
            job=pending[future];evaluations[job]=future.result();print('audited evaluation '+job,flush=True)
    paired=[]
    for validation,seed in [(False,s) for s in INITS]+[(True,314380)]:
        names=[f'validation-{arm}' if validation else f'{arm}-{seed}' for arm in ARMS]
        for matchup in evaluations[names[0]]['summary']:
            control=evaluations[names[0]]['summary'][matchup];teacher=evaluations[names[1]]['summary'][matchup]
            ac,bc=control['clusters'],teacher['clusters']
            check([(r['deal_seed'],r['level']) for r in ac]==[(r['deal_seed'],r['level']) for r in bc],'paired original deals')
            values=[b['win_rate']-a['win_rate'] for a,b in zip(ac,bc)];levels=[r['level'] for r in ac]
            interval=independent_interval(values,levels);other=cluster_bootstrap(values,levels,5000,314385)
            check(all(math.isclose(a,b,abs_tol=1e-14) for a,b in zip(interval,other['ci95'])),'independent paired interval')
            paired.append(dict(validation=validation,seed=seed,matchup=matchup,groups=len(values),
                control=control['metrics']['win_rate']['estimate'],teacher=teacher['metrics']['win_rate']['estimate'],
                delta_pp=100*sum(values)/len(values),ci95_delta_pp=[100*x for x in interval],group_delta=values,
                scope='pointwise descriptive difference conditional on initializer; no across-initializer confidence claim'))
    stability={}
    for arm in ARMS:
        values=[r[arm] for r in paired if not r['validation']]
        stability[arm]=dict(n=3,values=values,mean=statistics.mean(values),sd=statistics.stdev(values),min=min(values),max=max(values))
    final=read(root/'evaluations/validation-teacher/report.json')
    return dict(status='PASS',engineering='ACCEPTED_PAIRED_TEACHER_STUDY_V1',training=training,evaluations=evaluations,
        paired=paired,stability=stability,validation_gate=final['validation_gate'],candidate_sha256=final['candidate_sha256'],
        model_promoted=False,reserved_test_executed=False,protected_sources=protected,
        review_sha256=digest(Path(__file__)),receipt_sha256=digest(root/'receipt.json'),
        controller_work_directory=str(cache) if cache else None)

def negative_checks(root):
    from unittest.mock import patch
    path=root/'evaluations/control-314380';rows=lines(path/'results.jsonl');trials=schedule();report=read(path/'report.json');outcomes={}
    cases={'missing':rows[:-1],'duplicate':rows[:-1]+[rows[0]],'wrong_win':deepcopy(rows)}
    cases['wrong_win'][0]['win']=1-cases['wrong_win'][0]['win']
    for name,data in cases.items():
        try:verify_stats(data,trials,report['summary'])
        except ValueError:outcomes[name]='REJECTED'
        else:raise AssertionError(name)
    wrong=deepcopy(report['summary']);wrong['dmc|dmc|greedy']['metrics']['win_rate']['ci95'][0]=-.1
    try:verify_stats(rows,trials,wrong)
    except ValueError:outcomes['wrong_interval']='REJECTED'
    else:raise AssertionError('wrong interval')
    actual=read
    for name in ('fake_promotion','behavior_denominator','candidate_selection','worker_not_reaped'):
        def altered(p):
            v=actual(p)
            if Path(p)==path/'report.json':
                if name=='fake_promotion':v['model_promoted']=True
                elif name=='behavior_denominator':v['counts']['finish_opportunities']+=1
                elif name=='worker_not_reaped':v['worker_closed']=False
            if Path(p)==path/'binding.json' and name=='candidate_selection':v['checkpoint']=v['checkpoint'].replace('control-314380','teacher-314380')
            return v
        with patch(__name__+'.read',altered):
            try:audit_evaluation(root,'control-314380')
            except ValueError:outcomes[name]='REJECTED'
            else:raise AssertionError(name)
    base=root/'training/teacher-314380';manifest=read(base/'phase1/manifest.json');reset=read(base/'reset.json')
    for name,key,value in [('wrong_transfer','phase2_initial_model_sha256','0'*64),('optimizer_not_reset','optimizer_empty',False),('wrong_rng','policy_rng_repr_sha256','0'*64)]:
        changed=deepcopy(reset);changed[key]=value
        try:verify_reset(changed,manifest,manifest['model_sha256'],314380)
        except ValueError:outcomes[name]='REJECTED'
        else:raise AssertionError(name)
    return dict(status='PASS',cases=outcomes,review_sha256=digest(Path(__file__)))

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--write',action='store_true');p.add_argument('--cache',type=Path);a=p.parse_args()
    root=a.directory.resolve();result=audit(root,a.cache.resolve() if a.cache else None)
    if a.write:
        write(root/'controller-audit.json',result);write(root/'controller-negative.json',negative_checks(root))
    print(json.dumps({k:v for k,v in result.items() if k not in ('training','evaluations')},ensure_ascii=False))

if __name__=='__main__':main()

