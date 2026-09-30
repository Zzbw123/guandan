"""Controller independent P3l disk audit; no training or result-dependent changes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
import json
import math
import zipfile
from hashlib import sha256
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from experiments.p3l_protocol import specification,config,INITS,ARMS,schedule
from experiments.p3e_common import read,write,digest,check
from experiments.review_p3l_evaluation import audit_evaluation,independent_interval,verify_stats,lines
from review_p3f import verify_training_replays,verify_waves,verify_optimizer,verify_reset,tensor_digest
from guandan.evaluation.statistics import cluster_bootstrap

def equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
    return a==b

def validate_batches(waves,phase,weight,objective):
    for i,w in enumerate(waves):
        check(w['objective']==objective,'frozen objective')
        n=w['samples'];batches=w['batches'];expected=phase['hand_rows'][4*i:4*i+4]
        check([b['samples'] for b in batches]==[min(256,n-start) for start in range(0,n,256)],'all batch samples including tail')
        wanted=sum(h['legal_candidates'] for h in expected) if weight else 0
        check(w['auxiliary_candidates']==wanted==sum(b['auxiliary_candidates'] for b in batches),'full training candidate coverage')
        check(sum(b['auxiliary_requests'] for b in batches)==(n if weight else 0),'all batch requests')
        check(w['auxiliary_weight']==weight,'frozen auxiliary weight')
        for b in batches:
            check(all(math.isfinite(b[k]) and b[k]>=0 for k in ('dmc_loss','auxiliary_loss')),'finite batch losses')
            check(b['auxiliary_candidates']>=b['auxiliary_requests'],'candidate/request counts')
            validate_norm_record(b, objective)
        for out,field in (('mean_loss','dmc_loss'),('auxiliary_loss','auxiliary_loss')):
            value=sum(b[field]*b['samples'] for b in batches)/n
            check(math.isclose(value,w[out],abs_tol=1e-12),'weighted wave loss')

def validate_norm_record(b, objective):
    d,c,a=b['d'],b['c'],b['alpha']
    check(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (d,c,a)), 'finite norms/alpha')
    zd,zc=d<=1e-12,c<=1e-12
    raw=None if zd or zc else .1*d/c
    expected=.1 if objective=='constant' else (0. if raw is None else min(1.,raw))
    branch='constant' if objective=='constant' else 'zero_dmc' if zd else 'zero_centered' if zc else 'cap' if raw>=1 else 'ratio'
    check(a==expected and b['alpha_branch']==branch, 'independent alpha formula/branch')
    check(b['zero_dmc']==zd and b['zero_centered']==zc, 'zero norm flags')
    check(b['cap_triggered']==(objective=='normcap' and raw is not None and raw>=1), 'alpha cap flag')
    ratio=None if zd else a*c/d
    check(b['auxiliary_to_dmc_ratio']==ratio, 'norm ratio arithmetic')
    cosine=b['cosine'];factor=b['joint_descent_factor']
    if zd or zc: check(cosine is None,'undefined cosine')
    else: check(math.isfinite(cosine) and abs(cosine)<=1+1e-12,'cosine range')
    if zd: check(factor is None,'undefined descent factor')
    else:
        expected_factor=1. if zc else 1.+a*c*cosine/d
        check(math.isclose(factor,expected_factor,rel_tol=1e-12,abs_tol=1e-12),'descent factor arithmetic')
    if objective=='normcap':
        check(ratio is None or ratio<=.1+1e-12,'bounded auxiliary norm')
        check(factor is None or factor>=.9-1e-12,'raw-gradient directional bound')
    sizes=b['candidate_sizes']
    check(len(sizes)==b['samples']==b['state_denominator']==b['auxiliary_requests'],'state denominator')
    check(all(type(n) is int and n>=1 for n in sizes) and sum(sizes)==b['auxiliary_candidates'],'complete candidate sizes')
    check(sum(n==1 for n in sizes)==b['single_candidate_requests'],'singleton denominator')

def audit_training(root,job):
    import torch
    torch.set_num_threads(1)
    arm,seedtext=job.split('-');seed=int(seedtext);cfg=config(seed,arm)
    path=root/'training'/job;r=read(path/'report.json');reset=read(path/'reset.json')
    check(r['status']=='PASS' and r['job']==job and r['config']==cfg and r['reset']==reset,'training metadata')
    phase,_,_=verify_training_replays(path/'replays.jsonl.gz',200,800)
    check(all(r['phase'][k]==v for k,v in phase.items()),'training replay/report totals')
    counters=verify_waves(path/'waves.jsonl',200,800,r['phase'])
    waves=lines(path/'waves.jsonl');validate_batches(waves,phase,cfg['auxiliary_weight'],arm)
    inherited=ROOT/f'artifacts/evaluations/p3f-teacher-v1/training/teacher-{seed}/phase1'
    manifest=read(inherited/'manifest.json')
    raw=torch.load(inherited/'raw.pt',map_location='cpu',weights_only=True)
    verify_reset(reset,manifest,tensor_digest(raw['model']),seed)
    check(Path(reset['phase1_file']).resolve()==(inherited/'raw.pt').resolve(),'exact inherited teacher path')
    check(reset['auxiliary_weight']==cfg['auxiliary_weight'] and reset['objective']==arm,'reset auxiliary objective/weight')
    final=read(path/'final/manifest.json');data=torch.load(path/'final/checkpoint.pt',map_location='cpu',weights_only=True)
    check(final==r['final_manifest'] and digest(path/'final/checkpoint.pt')==final['sha256'],'final checkpoint provenance')
    for k in ('config','episodes','updates','waves','used_deal_seeds','versions','runtime','sources','boundary'):
        check(data[k]==final[k],'checkpoint metadata '+k)
    check(data['config']==cfg and data['episodes']==600 and data['waves']==150 and data['updates']==counters['updates']
        and data['used_deal_seeds']==list(range(100200,100800)),'honest phase counters and seed history')
    check(data['versions']['training']=='gd-p3l-wave-v1' and data['versions']['checkpoint']=='gd-p3l-checkpoint-v1','algorithm-specific checkpoint')
    check(tensor_digest(data['model'])==r['model_sha256'],'final tensor digest')
    pre=read(root/'preregistration.json')
    pins={k:v for k,v in pre['source_sha256'].items() if k.startswith('src/') or (k.startswith('experiments/p3l') and k.endswith('.py')) or k in ('experiments/p3j_objective.py','experiments/p3i_objective.py','experiments/p3h_objective.py','experiments/p3f_teacher.py')}
    check(data['sources']==pins,'checkpoint complete source identity')
    runtime=r['runtime'];check(runtime==data['runtime'] and runtime['deterministic'] is True and runtime['threads']==1
        and runtime['tf32_matmul'] is runtime['tf32_cudnn'] is False,'deterministic CUDA runtime')
    verify_optimizer(data,counters['updates'])
    control_equivalent=None
    if arm=='constant':
        previous=torch.load(ROOT/f'artifacts/evaluations/p3j-centered-v1/training/centered-{seed}/final/checkpoint.pt',map_location='cpu',weights_only=True)
        for key in ('model','optimizer','policy_rng','torch_rng','cuda_rng','episodes','waves','updates','used_deal_seeds'):
            check(equal(data[key],previous[key]),'full 600-hand old control equivalence '+key)
        control_equivalent=True
    return dict(job=job,arm=arm,seed=seed,hands=600,samples=phase['samples'],updates=counters['updates'],
        legal_candidates=phase['legal_candidates'],auxiliary_candidates=sum(w['auxiliary_candidates'] for w in waves),
        seconds=r['seconds'],model_sha256=r['model_sha256'],candidate_sha256=final['sha256'],reset=reset,
        old_control_bitwise_equal=control_equivalent)

def reviewed(root,kind,job,cache):
    folder=root/('training' if kind=='training' else 'evaluations')/job
    pins={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}
    provenance=dict(preregistration_sha256=digest(root/'preregistration.json'),input_sha256=pins,
        review_sha256={p.name:digest(p) for p in (Path(__file__),ROOT/'experiments/review_p3l_evaluation.py',ROOT/'experiments/review_p3f.py')})
    file=cache/f'{kind}-{job}.json' if cache else None
    if file and file.exists():
        saved=read(file);check(saved['provenance']==provenance,'cached review provenance drift');return saved['result']
    result=(audit_training if kind=='training' else audit_evaluation)(root,job)
    check(pins=={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()},'review inputs immutable')
    if file:write(file,dict(status='PASS',provenance=provenance,result=result))
    return result

def verify_freeze(root,complete=True):
    pre=read(root/'preregistration.json');check(pre['specification']==specification(),'exact predeclared specification')
    for key in ('source_sha256','input_sha256'):
        for name,h in pre[key].items():check(digest(ROOT/name)==h,'frozen pin '+name)
    with zipfile.ZipFile(root/'source-snapshot.zip') as z:
        check(set(z.namelist())==set(pre['source_sha256']),'exact source ZIP members')
        for name,h in pre['source_sha256'].items():check(sha256(z.read(name)).hexdigest()==h,'source ZIP bytes')
    if complete:
        receipt=read(root/'receipt.json')
        check(receipt['status']=='RUN_COMPLETE_PENDING_AUDIT' and receipt['model_promoted'] is receipt['reserved_test_executed'] is False,'receipt state')
        for name,h in receipt['artifact_sha256'].items():check(digest(root/name)==h,'completed run artifact '+name)

def audit(root,cache=None,completed_only=False):
    verify_freeze(root,not completed_only)
    if cache:cache.mkdir(parents=True,exist_ok=True)
    spec=specification();jobs=[]
    for kind,folder,names in [('training','training',spec['training_order']),('evaluation','evaluations',spec['development_order']+spec['validation_order'])]:
        for job in names:
            if not completed_only or (root/folder/job/'report.json').exists():jobs.append((kind,job))
    training={};evaluations={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(reviewed,root,kind,job,cache):(kind,job) for kind,job in jobs}
        for f in as_completed(pending):
            kind,job=pending[f];(training if kind=='training' else evaluations)[job]=f.result()
            print('independently audited',kind,job,flush=True)
    if completed_only:return dict(status='PARTIAL_REVIEW',training=list(training),evaluations=list(evaluations))
    for seed in INITS:
        a,b=[training[f'{arm}-{seed}']['reset'] for arm in ARMS]
        for key in ('phase1_model_sha256','phase2_initial_model_sha256','policy_rng_repr_sha256','torch_rng_sha256','cuda_rng_sha256'):
            check(a[key]==b[key],'paired reset '+key)
    paired=[]
    for validation,seed in [(False,s) for s in INITS]+[(True,314380)]:
        names=[f'validation-{a}' if validation else f'{a}-{seed}' for a in ARMS]
        for matchup in evaluations[names[0]]['summary']:
            ca,cb=[evaluations[name]['summary'][matchup] for name in names]
            ac,bc=ca['clusters'],cb['clusters']
            check([(r['deal_seed'],r['level']) for r in ac]==[(r['deal_seed'],r['level']) for r in bc],'paired original deals')
            values=[b['win_rate']-a['win_rate'] for a,b in zip(ac,bc)];levels=[r['level'] for r in ac]
            interval=independent_interval(values,levels);standard=cluster_bootstrap(values,levels,5000,314425)
            check(all(math.isclose(a,b,abs_tol=1e-14) for a,b in zip(interval,standard['ci95'])),'independent paired bootstrap')
            paired.append(dict(validation=validation,seed=seed,matchup=matchup,groups=len(values),
                constant=ca['metrics']['win_rate']['estimate'],normcap=cb['metrics']['win_rate']['estimate'],
                delta_pp=100*sum(values)/len(values),ci95_delta_pp=[100*x for x in interval],
                group_delta=values,levels=levels,deal_seeds=[r['deal_seed'] for r in ac],
                min_delta=min(values),max_delta=max(values),zero_deltas=values.count(0),missing_groups=0))
    final=read(root/'evaluations/validation-normcap/report.json')
    return dict(status='PASS',engineering='ACCEPTED_NORMCAP_AUXILIARY_STUDY_V1',overall_P3='PARTIALLY_ACCEPTED',
        training=training,evaluations=evaluations,paired=paired,validation_gate=final['validation_gate'],
        candidate_sha256=final['candidate_sha256'],model_promoted=False,reserved_test_executed=False,
        review_sha256=digest(Path(__file__)),receipt_sha256=digest(root/'receipt.json'))

def negative(root):
    from unittest.mock import patch
    import experiments.review_p3l_evaluation as er
    folder=root/'evaluations/constant-314380';rows=lines(folder/'results.jsonl');report=read(folder/'report.json')
    cases={};datasets={'missing':rows[:-1],'duplicate':rows[:-1]+[rows[0]],'wrong_win':deepcopy(rows)}
    datasets['wrong_win'][0]['win']=1-datasets['wrong_win'][0]['win']
    for name,data in datasets.items():
        try:verify_stats(data,schedule(),report['summary'])
        except ValueError:cases[name]='REJECTED'
        else:raise AssertionError(name)
    incorrect=deepcopy(report['summary']);incorrect['dmc|dmc|greedy']['metrics']['win_rate']['ci95'][0]=-.1
    try:verify_stats(rows,schedule(),incorrect)
    except ValueError:cases['wrong_interval']='REJECTED'
    else:raise AssertionError('wrong_interval')
    for name in ('fake_promotion','behavior_denominator','candidate_selection'):
        def altered(p):
            v=read(p)
            if Path(p)==folder/'report.json':
                if name=='fake_promotion':v['model_promoted']=True
                if name=='behavior_denominator':v['counts']['finish_opportunities']+=1
            if Path(p)==folder/'binding.json' and name=='candidate_selection':v['checkpoint']=v['checkpoint'].replace('constant-','normcap-')
            return v
        with patch.object(er,'read',altered):
            try:er.audit_evaluation(root,'constant-314380')
            except ValueError:cases[name]='REJECTED'
            else:raise AssertionError(name)
    path=root/'training/normcap-314380';phase=read(path/'report.json')['phase'];waves=lines(path/'waves.jsonl')
    for name,key,value in [('dropped_candidates','auxiliary_candidates',0),('changed_weight','auxiliary_weight',.2),('wrong_objective','objective','constant')]:
        bad=deepcopy(waves);bad[0][key]=value
        try:validate_batches(bad,phase,.1,'normcap')
        except ValueError:cases[name]='REJECTED'
        else:raise AssertionError(name)
    for name,key,value in [('alpha_tamper','alpha',42.),('ratio_tamper','auxiliary_to_dmc_ratio',.2),
            ('norm_tamper','d',-1.),('denominator_tamper','state_denominator',0),('branch_tamper','alpha_branch','invented')]:
        bad=deepcopy(waves);bad[0]['batches'][0][key]=value
        try:validate_batches(bad,phase,.1,'normcap')
        except ValueError:cases[name]='REJECTED'
        else:raise AssertionError(name)
    return dict(status='PASS',cases=cases)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--cache',type=Path)
    p.add_argument('--completed',action='store_true');p.add_argument('--write',action='store_true');a=p.parse_args()
    root=a.root.resolve();r=audit(root,a.cache.resolve() if a.cache else None,a.completed)
    if a.write and not a.completed:
        write(root/'controller-audit.json',r);write(root/'controller-negative.json',negative(root))
    print(json.dumps({k:v for k,v in r.items() if k not in ('training','evaluations','paired')},ensure_ascii=False))
if __name__=='__main__':main()
