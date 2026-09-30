"""Freeze P3f and execute fixed paired training/development/validation jobs."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
from _bootstrap import ROOT
sys.path.insert(0,str(ROOT/'experiments'))
from p3f_protocol import specification
from p3e_common import check,digest,read,write

def sources():
    return (sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'experiments/p3d').glob('*.py'))+
        sorted((ROOT/'experiments').glob('p3f*.py'))+sorted((ROOT/'scripts').glob('p3f*.py'))+
        sorted((ROOT/'tests').glob('test_p3f*.py'))+[ROOT/'scripts/p3d_evaluate.py',ROOT/'scripts/run_gpu.ps1',
        ROOT/'scripts/_bootstrap.py',ROOT/'docs/P3F_PROTOCOL.md',ROOT/'docs/P3F_DIAGNOSTIC_PROTOCOL.md',
        ROOT/'requirements-learning-gpu.lock',ROOT/'experiments/p3e_protocol.py',
        ROOT/'experiments/p3e_common.py',ROOT/'experiments/p3e_metrics.py'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=a.output.resolve()
    if out.exists():p.error('output already exists')
    required=[ROOT/'artifacts/evaluations/p3f-seed-availability.json',ROOT/'artifacts/evaluations/p3f-diagnostic-controller.json',
              ROOT/'artifacts/evaluations/p3f-test-receipt.json']
    for path in required:check(read(path)['status']=='PASS',f'prerequisite {path.name}')
    for name,value in read(required[1])['input_sha256'].items():check(digest(ROOT/name)==value,'diagnostic evidence drift')
    out.mkdir(parents=True);paths=sources();hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    write(out/'preregistration.json',dict(utc=datetime.now(timezone.utc).isoformat(),specification=specification(),
        source_sha256=hashes,prerequisite_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in required}))
    with zipfile.ZipFile(out/'source-snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,path.relative_to(ROOT).as_posix())
    start=time.perf_counter();spec=specification()
    jobs=([('train',s) for s in spec['training_order']]+[('evaluate',s) for s in spec['development_order']+spec['validation_order']])
    try:
        for stage,job in jobs:
            with (out/f'{stage}-{job}.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts/p3f_worker.py'),str(out),stage,job],stdout=log,stderr=subprocess.STDOUT)
                try:code=child.wait(timeout=2400)
                except subprocess.TimeoutExpired:
                    if sys.platform=='win32':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=10)
                    else:child.kill()
                    child.wait(timeout=10);raise TimeoutError(f'{stage} {job} exceeded 2400s')
            check(code==0,f'{stage} {job} exit {code}; retained log')
            print(json.dumps(dict(stage=stage,job=job,status='PASS',elapsed_s=time.perf_counter()-start)),flush=True)
        for name,value in hashes.items():check(digest(ROOT/name)==value,f'source drift {name}')
        files=sorted(p for p in out.rglob('*') if p.is_file())
        write(out/'receipt.json',dict(status='RUN_COMPLETE_PENDING_AUDIT',elapsed_s=time.perf_counter()-start,
            model_promoted=False,reserved_test_executed=False,artifact_sha256={p.relative_to(out).as_posix():digest(p) for p in files}))
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
