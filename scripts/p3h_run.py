"""Freeze and run the fixed P3h experiment, without result-dependent choices."""
from _bootstrap import ROOT
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.p3h_protocol import specification
from experiments.p3e_common import check,digest,read,write
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import time
import zipfile
import traceback
import json

def main():
    out=Path(sys.argv[1]).resolve()
    check(not out.exists(),'output already exists')
    previous=read(ROOT/'artifacts/evaluations/p3g-fit-v1/delivery-receipt.json')
    for name,h in previous['artifact_sha256'].items():check(digest(ROOT/name)==h,'P3g delivery drift '+name)
    prerequisites=[ROOT/'artifacts/evaluations/p3h-seed-availability.json',ROOT/'artifacts/evaluations/p3h-test-receipt.json']
    for p in prerequisites:check(read(p)['status']=='PASS','prerequisite '+p.name)
    tests=read(prerequisites[1])
    for name,h in tests['source_sha256'].items():check(digest(ROOT/name)==h,'tested source drift')
    paths=sorted(set(list((ROOT/'src').rglob('*.py'))+list((ROOT/'experiments').glob('*.py'))+
        list((ROOT/'experiments/p3d').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+
        list((ROOT/'tests').glob('test_p3h*.py'))+[ROOT/'docs/P3H_PROTOCOL.md',ROOT/'scripts/run_gpu.ps1']))
    inputs=prerequisites+[ROOT/'artifacts/evaluations/p3g-fit-v1/delivery-receipt.json']
    for s in (314380,314381,314382):
        base=ROOT/f'artifacts/evaluations/p3f-teacher-v1/training/teacher-{s}'
        inputs += [base/'phase1/raw.pt',base/'phase1/manifest.json',base/'phase1/replays.jsonl.gz',base/'final/checkpoint.pt']
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
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts/p3h_worker.py'),str(out),stage,job],stdout=log,stderr=subprocess.STDOUT)
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
