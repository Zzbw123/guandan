"""Construct isolated controller sources without modifying frozen predecessors."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]

def put(name,text):
    with (root/name).open('x',encoding='utf-8') as f:f.write(text)

training=(root/'experiments/controller_p5b_training.py').read_text('utf-8')
training=training.replace('experiments.p5b_protocol','experiments.p5e_protocol')
training=training.replace('from experiments.p5a_checkpoint import sources','from experiments.p5d_checkpoint import sources,versions')
training=training.replace('import gzip,json,math,random','import gzip,json,math,random,copy\nfrom itertools import islice\nfrom experiments.controller_p5d_audit import audit_batch_rows')
training=training.replace("check(w['wave']==", "check(w['objective']==cfg['objective'],'wave objective')\n    check(w['wave']==",1)
training=training.replace("rng.shuffle(samples)\n", "rng.shuffle(samples)\n    audit_batch_rows(samples,w,cfg)\n    counts['positive']=sum(x[2]==1 for x in samples)\n    counts['negative']=len(samples)-counts['positive']\n    counts['single_class_batches']=sum(b['single_class'] for b in w['batches'])\n    counts['dual_class_batches']=sum(not b['single_class'] for b in w['batches'])\n",1)
training=training.replace("initial=payload(folder/'initial');final=payload(folder/'final')", "initial=payload(folder/'initial');final=payload(folder/'final')\n    check(initial['config']==cfg and initial['sources']==sources() and initial['versions']==versions() and initial['objective']==cfg['objective'],'initial source/config/version/objective')")
training=training.replace("snapshots={1:payload(folder/'wave-1'),200:payload(folder/'wave-200'),400:final}","snapshots={w:payload(folder/f'wave-{w}') for w in (1,2,3,4,200)};snapshots[400]=final")
training=training.replace("check(p['config']==cfg and p['sources']==sources()", "check(p['objective']==cfg['objective'] and p['versions']==versions(),'checkpoint objective/version')\n            check(p['config']==cfg and p['sources']==sources()")
training=training.replace("return dict(job=job,hands=1600", "if arm=='ordinary':\n        ordinary_compatibility(root,job)\n    return dict(job=job,hands=1600")
training=training.replace('def first_wave_numerics(root,job):','def wave_numerics(root,job,index):')
training=training.replace("before=payload(folder/'initial');after=payload(folder/'wave-1');w=next(rows(folder/'waves.jsonl.gz'))\n    _,samples=audit_wave(w,0,cfg,random.Random(int(seed)),encode=True)","before=payload(folder/('initial' if index==0 else f'wave-{index}'))\n    after=payload(folder/f'wave-{index+1}');w=next(islice(rows(folder/'waves.jsonl.gz'),index,None))\n    rng=random.Random();rng.setstate(before['policy_rng'])\n    _,samples=audit_wave(w,index,cfg,rng,encode=True)")
training=training.replace("model=DMCNetwork().cuda();model.load_state_dict(before['model'])", "model=DMCNetwork().cuda();model.load_state_dict(before['model'])\n    fixed=DMCNetwork().cuda();fixed.load_state_dict(before['frozen'])")
training=training.replace("if d['role']=='current' and not d['explored']:\n                check(int(full_scores(model,obs,legal).argmax())==d['selected']", "if d['role']=='frozen' or (d['role']=='current' and not d['explored']):\n                scorer=fixed if d['role']=='frozen' else model\n                check(int(full_scores(scorer,obs,legal).argmax())==d['selected']")
training=training.replace("opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],capturable=True);loss_total=0", "opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],capturable=True)\n    opt.load_state_dict(copy.deepcopy(before['optimizer']));loss_total=0")
training=training.replace("loss=residual.square().sum()/len(batch);loss.backward();opt.step();loss_total+=float(loss.detach())*len(batch)","positive=sum(s[2]==1 for s in batch);negative=len(batch)-positive\n        balanced=cfg['objective']=='label_balanced' and positive and negative\n        wp=len(batch)/(2*positive) if balanced else 1.\n        wn=len(batch)/(2*negative) if balanced else 1.\n        weights=torch.tensor([wp if s[2]==1 else wn for s in batch],device='cuda')\n        loss=(weights*residual.square()).sum()/len(batch)\n        check(abs(float(loss.detach())-w['batches'][start//256]['loss'])<3e-6,'independent weighted batch loss')\n        loss.backward();opt.step();loss_total+=float(loss.detach())*len(batch)")
training=training.replace("return dict(job=job,samples=len(samples),updates=w['updates']", "return dict(job=job,wave=index+1,samples=len(samples),updates=len(w['batches']),\n        dual_class_batches=sum(not b['single_class'] for b in w['batches']),single_class_batches=sum(b['single_class'] for b in w['batches'])")
training+='''

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
'''
put('experiments/controller_p5e_training.py',training)

evaluation=(root/'experiments/controller_p5b_evaluation.py').read_text('utf-8')
evaluation=evaluation.replace('p5b_protocol','p5e_protocol').replace('314523','314553').replace("arm=='mixed'","arm=='label_balanced'")
evaluation=evaluation.replace("rows=lines(path/'results.jsonl');verify_stats", "rows=lines(path/'results.jsonl')\n    check(len(rows)==len(expected) and {r['trial_id'] for r in rows}==set(expected),'complete result schedule')\n    check(all(r['status']=='ok' and r['illegal_actions']==r['timeouts']==0 for r in rows),'all evaluation games valid')\n    verify_stats")
put('experiments/controller_p5e_evaluation.py',evaluation)

audit=(root/'experiments/controller_p5b_audit.py').read_text('utf-8')
audit=audit.replace('p5b','p5e').replace('P5b','P5e').replace('314523','314553')
audit=audit.replace('selfplay','ordinary').replace('mixed','label_balanced')
# Previous sources/receipts must remain bound to actual historical paths, not the new run.
start=audit.index('def historical():');end=audit.index('\n\ndef main():',start)
audit=audit[:start]+'''def historical():
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
'''+audit[end:]
audit=audit.replace("'experiments/controller_p5a_audit.py']", "'experiments/controller_p5a_audit.py','experiments/controller_p5d_audit.py']")
audit=audit.replace("scope='ACCEPTED_PAIRED_POOL_STUDY_V1'", "scope='ACCEPTED_PAIRED_LABEL_BALANCED_STUDY_V1'")
audit=audit.replace("job='ordinary-314510'", "job='label_balanced-314510'").replace("cfg=config(314510,'ordinary')", "cfg=config(314510,'label_balanced')")
audit=audit.replace("rewards=lambda r:r['hands'][0]['team_rewards'].reverse())", """rewards=lambda r:r['hands'][0]['team_rewards'].reverse(),
        objective=lambda r:r.update(objective='ordinary'),
        batch_count=lambda r:r['batches'].pop(),
        batch_positive=lambda r:r['batches'][0].update(positive=r['batches'][0]['positive']+1),
        batch_denominator=lambda r:r['batches'][0].update(n=r['batches'][0]['n']+1),
        batch_weight=lambda r:r['batches'][0].update(negative_weight=r['batches'][0]['negative_weight']+.5),
        batch_single_class=lambda r:r['batches'][0].update(single_class=not r['batches'][0]['single_class']),
        batch_start=lambda r:r['batches'][0].update(start=1),
        batch_loss=lambda r:r['batches'][0].update(loss=r['batches'][0]['loss']+.1))""")
audit=audit.replace("first_wave_numerics=numerical,", "first_four_wave_numerics=numerical,ordinary_p5b_compatibility='BITWISE_EQUAL_ALL_THREE',")
audit=audit.replace("p.add_argument('--training-only',action='store_true')", "p.add_argument('--training-only',action='store_true');p.add_argument('--output',type=Path)")
audit=audit.replace("args=p.parse_args();root=args.root.resolve();freeze", "args=p.parse_args();root=args.root.resolve()\n    output=args.output.resolve() if args.output else root/'controller-audit.json'\n    check(args.training_only or not output.exists(),'fresh controller output')\n    freeze")
audit=audit.replace("write(root/'controller-audit.json',report)", "write(output,report)")
put('experiments/controller_p5e_audit.py',audit)
