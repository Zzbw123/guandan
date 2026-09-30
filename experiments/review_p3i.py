"""Independent disk oracle: variance/error decomposition, provenance, and negatives.

Does not import the P3i loss/decomposition implementation.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from copy import deepcopy
import csv
import gzip
from hashlib import sha256
import json
import math
from statistics import fmean, pvariance
import zipfile

BASE=ROOT/'artifacts/evaluations/p3h-aux-v1'
JOBS=[f'{a}-{s}' for s in (314380,314381,314382) for a in ('control','aux')]
def read(p):return json.loads(Path(p).read_text('utf-8'))
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def check(ok,msg):
    if not ok:raise ValueError(msg)
def close(a,b):check(math.isfinite(a) and math.isclose(a,b,rel_tol=0.,abs_tol=1e-12),'independent arithmetic')

def verify_rows(data, report, original):
    check(len(data)==768 and len(original)==128,'full rows')
    check((report['states'],report['candidates'],report['rows'])==(128,4174,768),'report denominators')
    check(sum(x['candidate_count'] for x in original)==4174,'raw candidates')
    check(report['formal_training_hands']==report['new_evaluation_games']==0 and
          report['model_promoted'] is report['reserved_test_executed'] is False,'scope gate')
    check(set(report['summary'])==set(JOBS),'exact models')
    sums={job:[] for job in JOBS};max_residual=0.
    for idx, row in enumerate(data):
        i,j=divmod(idx,6);job=JOBS[j];old=original[i]
        check(row['state_index']==i and row['job']==job and row['identity']==old['identity'],'exact row identity/order')
        check(row['candidate_sha256']==old['candidate_sha256'] and row['candidates']==old['candidate_count'],'full candidates')
        q=old['scores'][job];raw=old['teacher_scores'];n=len(raw)
        check(len(q)==n==row['candidates'] and all(math.isfinite(v) for v in q+raw),'raw finite coverage')
        lo,hi=min(raw),max(raw)
        t=[0. if hi==lo else (v-lo)/(hi-lo)*1.6-.8 for v in raw]
        errors=[v-w for v,w in zip(q,t)]
        offset=fmean(q)-fmean(t)
        # Independent centered computation uses the population variance of errors.
        expected=dict(mse=fmean(x*x for x in errors),mean_error=offset,
                      offset_mse=offset*offset,centered_mse=pvariance(errors))
        selected=q.index(max(q))
        expected.update(top1=int(raw[selected]==hi),regret=0. if hi==lo else (hi-raw[selected])/(hi-lo))
        for k,v in expected.items():close(row[k],v)
        residual=row['mse']-row['offset_mse']-row['centered_mse']
        close(row['identity_residual'],residual);check(abs(residual)<=1e-12,'decomposition identity')
        max_residual=max(max_residual,abs(residual));sums[job].append(expected)
    for job,values in sums.items():
        for key in values[0]:close(report['summary'][job][key],fmean(v[key] for v in values))
        close(report['summary'][job]['offset_fraction'],fmean(v['offset_mse'] for v in values)/fmean(v['mse'] for v in values))
    check(len(report['paired'])==3,'paired rows')
    for i,pair in enumerate(report['paired']):
        seed=314380+i;check(pair['seed']==seed,'paired initializer')
        a,b=[report['summary'][f'{arm}-{seed}'] for arm in ('control','aux')]
        for key in ('mse','offset_mse','centered_mse','top1','regret'):close(pair['delta_'+key],b[key]-a[key])
    close(report['max_identity_residual'],max_residual)

def verify_pins(pre):
    for field in ('source_sha256','input_sha256'):
        for path,h in pre[field].items():check(digest(ROOT/path)==h,'pin '+path)

def audit(out):
    pre=read(out/'preregistration.json');verify_pins(pre)
    check(pre['version']=='gd-p3i-diagnostic-v1' and pre['jobs']==JOBS and pre['states']==128 and pre['candidates']==4174,'fixed design')
    check(pre['formal_training_hands']==pre['new_evaluation_games']==0 and
          pre['model_promoted'] is pre['reserved_test_executed'] is False,'preregistration scope')
    with zipfile.ZipFile(out/'source-snapshot.zip') as z:
        check(set(z.namelist())==set(pre['source_sha256']),'source archive members')
        for name,h in pre['source_sha256'].items():check(sha256(z.read(name)).hexdigest()==h,'archive bytes')
    from experiments.review_p3h import verify_freeze
    verify_freeze(BASE,False)
    oldreport=read(BASE/'fidelity/report.json');oldaudit=read(BASE/'fidelity/controller-audit.json')
    check(oldreport['scores_sha256']==digest(BASE/'fidelity/scores.jsonl.gz') and
          oldaudit['report_sha256']==digest(BASE/'fidelity/report.json') and oldaudit['status']=='PASS','prior score bindings')
    for job in JOBS:
        folder=BASE/f'training/{job}/final';manifest=read(folder/'manifest.json')
        check(manifest['sha256']==digest(folder/'checkpoint.pt')==oldreport['model_sha256'][job],'model binding')
    with gzip.open(BASE/'fidelity/scores.jsonl.gz','rt',encoding='utf-8') as f:original=[json.loads(s) for s in f]
    data=[json.loads(s) for s in (out/'decomposition.jsonl').read_text('utf-8').splitlines()]
    report=read(out/'report.json')
    for key,name in [('preregistration_sha256','preregistration.json'),('decomposition_sha256','decomposition.jsonl'),('csv_sha256','summary.csv')]:
        check(report[key]==digest(out/name),'report file binding')
    verify_rows(data,report,original)
    with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
    check([r['job'] for r in csvrows]==JOBS,'CSV coverage')
    for row in csvrows:
        for k,v in report['summary'][row['job']].items():close(float(row[k]),v)
    tests=read(ROOT/'artifacts/evaluations/p3i-tests-v1/receipt.json')
    check(tests['status']=='PASS' and tests['total']==10 and tests['skipped']==0,'CUDA contract receipt')
    for p,h in tests['sources'].items():check(digest(ROOT/p)==h,'tested source')
    check(tests['log_sha256']==digest(ROOT/'artifacts/evaluations/p3i-tests-v1/tests.log'),'tested log')
    negatives={}
    for name in ('missing','duplicate','denominator','decomposition','fake_promotion','summary','input_hash'):
        d,r,p=deepcopy(data),deepcopy(report),deepcopy(pre)
        if name=='missing':d.pop()
        elif name=='duplicate':d[1]=deepcopy(d[0])
        elif name=='denominator':d[0]['candidates']-=1
        elif name=='decomposition':d[0]['offset_mse']+=.01
        elif name=='fake_promotion':r['model_promoted']=True
        elif name=='summary':r['summary'][JOBS[0]]['centered_mse']+=.01
        else:p['input_sha256'][next(iter(p['input_sha256']))]='0'*64
        try:
            if name=='input_hash':verify_pins(p)
            else:verify_rows(d,r,original)
        except ValueError as e:negatives[name]=str(e)
        else:raise ValueError('negative accepted '+name)
    return dict(status='PASS',rows=768,states=128,candidates=4174,negative_cases=negatives,
                old_P3h_sources_unchanged=True,source_files=len(pre['source_sha256']),
                report_sha256=digest(out/'report.json'),review_sha256=digest(Path(__file__)),
                scope='independent arithmetic and provenance; direct-forward audit separate',model_promoted=False)

if __name__=='__main__':
    out=Path(sys.argv[1]).resolve();result=audit(out)
    target=out/'controller-audit.json'
    if not target.exists():target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    else:check(read(target)==result,'repeat audit equality')
    print(json.dumps(result))
