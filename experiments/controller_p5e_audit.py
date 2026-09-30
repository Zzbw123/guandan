"""Independent controller acceptance for the preregistered P5e experiment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
import argparse,copy,json,math,random,zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from hashlib import sha256
from experiments.p3e_common import check,read,write,digest
from experiments.p5e_protocol import specification,INITS,ARMS,schedule
from experiments.controller_p5e_training import audit_training,audit_wave,first_wave_numerics,rows,payload,equal
from experiments.controller_p5e_evaluation import audit_evaluation,independent_interval,verify_stats,lines
from guandan.evaluation.statistics import cluster_bootstrap

CONTROLLERS=['experiments/controller_p5e_audit.py','experiments/controller_p5e_training.py',
             'experiments/controller_p5e_evaluation.py','experiments/controller_p5a_audit.py','experiments/controller_p5d_audit.py']

def freeze(root,complete=True):
    pre=read(root/'preregistration.json');check(pre['specification']==specification(),'frozen specification')
    for key in ('source_sha256','input_sha256','historical_seed_files'):
        for name,h in pre[key].items():check(digest(ROOT/name)==h,'frozen '+key+' '+name)
    for archive,key in [('source-snapshot.zip','source_sha256'),('inputs-snapshot.zip','input_sha256')]:
        with zipfile.ZipFile(root/archive) as z:
            check(set(z.namelist())==set(pre[key]),'snapshot file set')
            for name,h in pre[key].items():check(sha256(z.read(name)).hexdigest()==h,'snapshot bytes '+name)
    if complete:
        receipt=read(root/'receipt.json');check(receipt['status']=='RUN_COMPLETE_PENDING_AUDIT','completed run receipt')
        for name,h in receipt['artifact_sha256'].items():check(digest(root/name)==h,'immutable run artifact '+name)
    return pre

def audited_job(root,kind,job):
    folder=root/('training' if kind=='training' else 'evaluations')/job
    inputs={p.relative_to(root).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}
    controllers={p:digest(ROOT/p) for p in CONTROLLERS}
    cache=root/'controller-cache';cache.mkdir(exist_ok=True)
    target=cache/f'{kind}-{job}.json'
    provenance=dict(inputs=inputs,controllers=controllers,preregistration_sha256=digest(root/'preregistration.json'))
    if target.exists():
        saved=read(target);check(saved['provenance']==provenance,'cached audit provenance');return saved['result']
    result=(audit_training if kind=='training' else audit_evaluation)(root,job)
    check(inputs=={p.relative_to(root).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()},'audit inputs changed')
    write(target,dict(status='PASS',provenance=provenance,result=result));return result

def paired(left,right):
    output={}
    for key in left:
        a=left[key]['clusters'];b=right[key]['clusters']
        check(len(a)==len(b),'paired cluster count')
        deltas=[];levels=[]
        for x,y in zip(a,b,strict=True):
            check((x['deal_seed'],x['level'],x['games'])==(y['deal_seed'],y['level'],y['games']) and x['games']==8,'paired cluster identity')
            deltas.append(y['win_rate']-x['win_rate']);levels.append(x['level'])
        summary=cluster_bootstrap(deltas,levels,5000,314553)
        check(math.isclose(sum(deltas)/len(deltas),summary['estimate'],abs_tol=1e-14),'independent paired arithmetic')
        check(all(math.isclose(x,y,abs_tol=1e-14) for x,y in zip(summary['ci95'],independent_interval(deltas,levels))),'independent paired CI')
        output[key]=dict(label_balanced_minus_ordinary=summary,original_deals=len(deltas),
            clusters=[dict(deal_seed=x['deal_seed'],level=x['level'],difference=d) for x,d in zip(a,deltas)],
            scope='SECONDARY_POINTWISE_DESCRIPTION')
    return output

def negative_checks(root):
    from experiments.p5e_protocol import config
    job='label_balanced-314510';w=next(rows(root/'training'/job/'waves.jsonl.gz'));cfg=config(314510,'label_balanced')
    def decision(r):return r['hands'][0]['decisions'][0]
    cases=dict(role=lambda r:r['hands'][0]['roles'].__setitem__(0,'greedy'),
        seed=lambda r:r['hands'][0].update(seed=100001),
        candidate=lambda r:decision(r).update(candidates=decision(r)['candidates']+1),
        selected=lambda r:decision(r).update(selected=-1),
        sample=lambda r:decision(r).update(sample=False),
        exploration=lambda r:decision(r).update(explored=not decision(r)['explored']),
        sample_count=lambda r:r.update(samples=r['samples']+1),
        scoring_count=lambda r:r.update(scored_candidates=r['scored_candidates']+1),
        rewards=lambda r:r['hands'][0]['team_rewards'].reverse(),
        objective=lambda r:r.update(objective='ordinary'),
        batch_count=lambda r:r['batches'].pop(),
        batch_positive=lambda r:r['batches'][0].update(positive=r['batches'][0]['positive']+1),
        batch_denominator=lambda r:r['batches'][0].update(n=r['batches'][0]['n']+1),
        batch_weight=lambda r:r['batches'][0].update(negative_weight=r['batches'][0]['negative_weight']+.5),
        batch_single_class=lambda r:r['batches'][0].update(single_class=not r['batches'][0]['single_class']),
        batch_start=lambda r:r['batches'][0].update(start=1),
        batch_loss=lambda r:r['batches'][0].update(loss=r['batches'][0]['loss']+.1))
    rejections={}
    for name,mutate in cases.items():
        bad=copy.deepcopy(w);mutate(bad)
        try:audit_wave(bad,0,cfg,random.Random(314510))
        except (ValueError,AssertionError) as exc:rejections[name]=str(exc)
        else:raise ValueError('training mutation accepted '+name)
    folder=root/'evaluations/validation-label_balanced';observed=lines(folder/'results.jsonl');report=read(folder/'report.json')
    checks={
        'statistic':(observed,copy.deepcopy(report['summary'])),
        'missing_trial':(observed[:-1],report['summary']),
        'duplicate_trial':(observed+[observed[0]],report['summary'])}
    checks['statistic'][1]['dmc|dmc|greedy']['metrics']['win_rate']['estimate']+=.1
    for name,(data,summary) in checks.items():
        try:verify_stats(data,schedule(True),summary)
        except (ValueError,AssertionError) as exc:rejections[name]=str(exc)
        else:raise ValueError('statistics mutation accepted '+name)
    return rejections

def historical():
    from experiments.p5d_package import protected
    result=protected()
    path=ROOT/'artifacts/evaluations/p5d-balanced-v1/delivery-receipt.json'
    receipt=read(path);check(receipt['status']=='PASS','P5d delivery')
    count=0
    for name,h in receipt['artifact_sha256'].items():
        if name=='docs/STATUS.md':continue
        check(digest(ROOT/name)==h,'P5d historical '+name);count+=1
    pre=read(ROOT/'artifacts/evaluations/p5d-balanced-v1/preregistration.json')
    for key in ('sources','inputs'):
        for name,h in pre[key].items():check(digest(ROOT/name)==h,'P5d frozen '+name)
    return dict(previous=result,p5d_bindings=count)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--training-only',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();root=args.root.resolve()
    output=args.output.resolve() if args.output else root/'controller-audit.json'
    check(args.training_only or not output.exists(),'fresh controller output')
    freeze(root,complete=not args.training_only)
    spec=specification();training={};evaluations={}
    work=[('training',j) for j in spec['training_order']]
    if not args.training_only:work += [('evaluations',j) for j in spec['development_order']+spec['validation_order']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(audited_job,root,k,j):(k,j) for k,j in work}
        for future in as_completed(futures):
            kind,job=futures[future];result=future.result()
            (training if kind=='training' else evaluations)[job]=result
            print(json.dumps(dict(stage='independent-audit',kind=kind,job=job,status='PASS')),flush=True)
    if args.training_only:return
    for seed in INITS:
        left=payload(root/'training'/f'ordinary-{seed}'/'initial')
        right=payload(root/'training'/f'label_balanced-{seed}'/'initial')
        for key in ('model','frozen','optimizer','policy_rng','torch_rng','cuda_rng','episodes','updates','used_deal_seeds'):
            check(equal(left[key],right[key]),'paired initial state '+key)
    numerical=[first_wave_numerics(root,j) for j in spec['training_order']]
    development={str(seed):paired(evaluations[f'ordinary-{seed}']['summary'],evaluations[f'label_balanced-{seed}']['summary']) for seed in INITS}
    validation=paired(evaluations['validation-ordinary']['summary'],evaluations['validation-label_balanced']['summary'])
    candidates=read(root/'candidates.json')
    check(candidates['primary']=='label_balanced-314510' and candidates['selection']=='final wave 400 only','fixed primary candidate')
    for job,h in candidates['candidates'].items():check(training[job]['candidate_sha256']==h,'frozen candidate registry')
    primary=read(root/'evaluations/validation-label_balanced/report.json')
    negative=negative_checks(root);history=historical();freeze(root)
    report=dict(status='PASS',scope='ACCEPTED_PAIRED_LABEL_BALANCED_STUDY_V1',training=training,evaluations=evaluations,
        first_four_wave_numerics=numerical,ordinary_p5b_compatibility='BITWISE_EQUAL_ALL_THREE',paired_development=development,paired_validation=validation,
        negative_rejections=negative,historical=history,validation_gate=primary['validation_gate'],
        candidate_sha256=candidates['candidates']['label_balanced-314510'],model_promoted=False,reserved_test_executed=False,
        controllers={p:digest(ROOT/p) for p in CONTROLLERS},preregistration_sha256=digest(root/'preregistration.json'))
    write(output,report)
    print(json.dumps(dict(status='PASS',validation_gate=report['validation_gate'],training_hands=sum(r['hands'] for r in training.values()),
        evaluation_games=sum(r['games'] for r in evaluations.values()),negative_rejections=len(negative))),flush=True)

if __name__=='__main__':main()
