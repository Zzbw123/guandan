"""Freeze and run P3l only after independent engineering acceptance."""
from _bootstrap import ROOT
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.p3l_protocol import specification
from experiments.p3l_scan import scan
from experiments.p3e_common import check,digest,read,write
from pathlib import Path
from datetime import datetime,timezone
import subprocess,time,zipfile,traceback,json

HISTORICAL=('p2-validation-v1','p3a-cpu-v1','p3b-gpu-v1','p3c-gpu-wave-v1','p3d-validation-v2',
    'p3e-scale-v2','p3f-teacher-v1','p3g-fit-v1','p3h-aux-v1','p3i-centered-v1','p3j-centered-v1','p3k-gradients-v1')


def historical_sources():
    registrations=[]
    for directory in HISTORICAL:
        path=ROOT/f'artifacts/evaluations/{directory}/preregistration.json'
        pre=read(path);hashes=pre.get('source_sha256',pre.get('source_sha256_before'))
        check(isinstance(hashes,dict) and hashes,'historical registry '+directory)
        for name,h in hashes.items():check(digest(ROOT/name)==h,'historical source '+name)
        registrations.append(path)
    return registrations


def p3l_sources():
    return sorted([p for d,pattern in [('experiments','p3l*.py'),('scripts','p3l*.py'),('tests','test_p3l*.py')]
        for p in (ROOT/d).glob(pattern)])


def verify_prerequisites():
    registrations=historical_sources()
    base=ROOT/'artifacts/evaluations'
    paths=[base/name for name in ('p3l-seed-availability.json','p3l-test-receipt.json',
        'p3l-resume-test.json','p3l-engineering-acceptance.json')]
    for p in paths:check(read(p)['status']=='PASS','prerequisite '+p.name)
    tests=read(paths[1]);check(tests['total']>0 and tests['skipped']==0,'complete engineering tests')
    check(tests['log_sha256']==digest(base/'p3l-tests.log'),'test log binding')
    required={p.relative_to(ROOT).as_posix() for p in p3l_sources()}
    for receipt in (tests,read(paths[2]),read(paths[3])):
        check(required <= set(receipt['source_sha256']),'complete tested/accepted P3l source inventory')
        for name,h in receipt['source_sha256'].items():check(digest(ROOT/name)==h,'tested/accepted source '+name)
    acceptance=read(paths[3])
    check(acceptance['resume_receipt_sha256']==digest(paths[2]),'accepted recovery receipt binding')
    check(acceptance['test_receipt_sha256']==digest(paths[1]),'accepted test receipt binding')
    previous=read(paths[0]);current=scan()
    check(current['status']=='PASS' and not current['hits'] and not current['errors'],'unused validation schedule')
    check(previous==current,'scan stale or historical artifact inventory drift')
    return paths+registrations+[base/'p3l-tests.log'],current



def paired_validation(out):
    """Pair original deals before stratified bootstrap; a secondary description."""
    from guandan.evaluation.statistics import cluster_bootstrap
    reports={arm:read(out/f'evaluations/validation-{arm}/report.json') for arm in ('constant','normcap')}
    result={}
    for opponent in ('greedy','random','team'):
        key=f'dmc|dmc|{opponent}'
        left=reports['constant']['summary'][key]['clusters']
        right=reports['normcap']['summary'][key]['clusters']
        check(len(left)==len(right)==65,'paired validation cluster coverage')
        differences=[];strata=[]
        for a,b in zip(left,right,strict=True):
            check((a['deal_seed'],a['level'],a['games'])==(b['deal_seed'],b['level'],b['games']) and a['games']==8,'paired original deal identity')
            differences.append(b['win_rate']-a['win_rate']);strata.append(a['level'])
        result[opponent]=dict(normcap_minus_constant=cluster_bootstrap(differences,strata,5000,314425),
            clusters=[dict(deal_seed=a['deal_seed'],level=a['level'],difference=d) for a,d in zip(left,differences,strict=True)])
    write(out/'paired-validation.json',dict(status='PASS',role='SECONDARY_POINTWISE_DESCRIPTION',
        initializer=314380,comparisons=result,does_not_establish_equivalence=True,model_promoted=False))

def main():
    check(len(sys.argv)==2,'one fresh output path required')
    out=Path(sys.argv[1]).resolve()
    check(not out.exists(),'output exists')
    check(out.is_relative_to(ROOT/'artifacts/evaluations') and out.name.startswith('p3l-'),'isolated p3l artifact output required')
    prerequisites,scan_receipt=verify_prerequisites()
    spec=specification();check(json.loads(json.dumps(spec,allow_nan=False))==spec,'specification JSON roundtrip')
    sources=sorted(set([p for directory in ('src','experiments','scripts','tests') for p in (ROOT/directory).rglob('*.py')]
        +[ROOT/'docs/P3L_PROTOCOL.md',ROOT/'scripts/run_gpu.ps1']))
    inputs=list(prerequisites)
    for seed in (314380,314381,314382):
        base=ROOT/f'artifacts/evaluations/p3f-teacher-v1/training/teacher-{seed}/phase1'
        inputs += [base/name for name in ('raw.pt','manifest.json','replays.jsonl.gz')]
        base=ROOT/f'artifacts/evaluations/p3j-centered-v1/training/centered-{seed}/final'
        inputs += [base/'checkpoint.pt',base/'manifest.json']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    input_hashes=dict(scan_receipt['files'])
    input_hashes.update({p.relative_to(ROOT).as_posix():digest(p) for p in inputs})
    out.mkdir(parents=True)
    write(out/'preregistration.json',dict(utc=datetime.now(timezone.utc).isoformat(),specification=spec,
        source_sha256=hashes,input_sha256=input_hashes,historical_seed_files=scan_receipt['files']))
    check(read(out/'preregistration.json')['specification']==spec,'written frozen specification roundtrip')
    for name,paths in [('source-snapshot.zip',sources),('inputs-snapshot.zip',sorted(set(inputs)))]:
        with zipfile.ZipFile(out/name,'x',zipfile.ZIP_DEFLATED) as z:
            for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
    start=time.perf_counter()
    jobs=[('train',s) for s in spec['training_order']]+[('evaluate',s) for s in spec['development_order']+spec['validation_order']]
    try:
        for stage,job in jobs:
            with (out/f'{stage}-{job}.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts/p3l_worker.py'),str(out),stage,job],stdout=log,stderr=subprocess.STDOUT)
                try:code=child.wait(timeout=spec['process_timeout_s'])
                except BaseException:
                    if sys.platform=='win32':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=10)
                    else:child.kill()
                    child.wait(timeout=10);raise
            check(code==0,f'{stage} {job} exit {code}')
            print(json.dumps(dict(stage=stage,job=job,status='PASS',elapsed_s=time.perf_counter()-start)),flush=True)
        paired_validation(out)
        historical_sources()
        for name,h in hashes.items():check(digest(ROOT/name)==h,'source drift')
        write(out/'receipt.json',dict(status='RUN_COMPLETE_PENDING_AUDIT',elapsed_s=time.perf_counter()-start,
            model_promoted=False,reserved_test_executed=False,
            artifact_sha256={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}))
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
