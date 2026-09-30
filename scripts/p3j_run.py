"""Freeze and execute the predeclared P3j budget; fail closed on any job failure."""
from _bootstrap import ROOT
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.p3j_protocol import specification
from experiments.p3e_common import check,digest,read,write
from pathlib import Path
from datetime import datetime,timezone
import subprocess,time,zipfile,traceback,json

def main():
    out=Path(sys.argv[1]).resolve();check(not out.exists(),'output exists')
    subprocess.run([sys.executable,str(ROOT/'experiments/deliver_p3i.py'),'--verify'],check=True,timeout=120)
    prerequisites=[ROOT/'artifacts/evaluations/p3j-seed-availability.json',ROOT/'artifacts/evaluations/p3j-test-receipt.json']
    for p in prerequisites:check(read(p)['status']=='PASS','prerequisite '+p.name)
    tests=read(prerequisites[1]);check(tests['total']==15 and tests['skipped']==0,'all P3j tests')
    check(tests['log_sha256']==digest(ROOT/'artifacts/evaluations/p3j-tests.log'),'test log binding')
    for name,h in tests['source_sha256'].items():check(digest(ROOT/name)==h,'tested source '+name)
    scan=read(prerequisites[0]);check(not scan['hits'] and not scan['errors'],'unused validation schedule')
    for name,h in scan['files'].items():check(digest(ROOT/name)==h,'historical scan input drift')
    paths=sorted(set([p for directory in ('src','experiments','scripts','tests') for p in (ROOT/directory).rglob('*.py')]
        +[ROOT/'docs/P3J_PROTOCOL.md',ROOT/'scripts/run_gpu.ps1']))
    inputs=prerequisites+[ROOT/'artifacts/evaluations/p3i-centered-v1/delivery.json']
    for seed in (314380,314381,314382):
        base=ROOT/f'artifacts/evaluations/p3f-teacher-v1/training/teacher-{seed}'
        inputs += [base/'phase1/raw.pt',base/'phase1/manifest.json',base/'phase1/replays.jsonl.gz']
        base=ROOT/f'artifacts/evaluations/p3h-aux-v1/training/aux-{seed}/final'
        inputs += [base/'checkpoint.pt',base/'manifest.json']
    out.mkdir(parents=True)
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    write(out/'preregistration.json',dict(utc=datetime.now(timezone.utc).isoformat(),specification=specification(),
        source_sha256=hashes,input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}))
    with zipfile.ZipFile(out/'source-snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
    spec=specification();start=time.perf_counter()
    jobs=[('train',s) for s in spec['training_order']]+[('evaluate',s) for s in spec['development_order']+spec['validation_order']]
    try:
        for stage,job in jobs:
            with (out/f'{stage}-{job}.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts/p3j_worker.py'),str(out),stage,job],stdout=log,stderr=subprocess.STDOUT)
                try:code=child.wait(timeout=3600)
                except BaseException:
                    if sys.platform=='win32':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=10)
                    else:child.kill()
                    child.wait(timeout=10);raise
            check(code==0,f'{stage} {job} exit {code}')
            print(json.dumps(dict(stage=stage,job=job,status='PASS',elapsed_s=time.perf_counter()-start)),flush=True)
        for name,h in hashes.items():check(digest(ROOT/name)==h,'source drift')
        write(out/'receipt.json',dict(status='RUN_COMPLETE_PENDING_AUDIT',elapsed_s=time.perf_counter()-start,
            model_promoted=False,reserved_test_executed=False,
            artifact_sha256={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}))
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
