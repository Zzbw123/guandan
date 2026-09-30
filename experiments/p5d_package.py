"""Freeze, run, and preserve the fixed engineering package. No evaluation."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,subprocess,zipfile
from hashlib import sha256


def read(path):return json.loads(Path(path).read_text('utf-8'))
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def protected():
    from experiments.p5c_diagnostic import protected as historical
    result=historical()
    path=ROOT/'artifacts/evaluations/p5c-diagnostic-v1/delivery-receipt.json'
    receipt=read(path)
    assert receipt['status']=='PASS'
    count=0
    for name,h in receipt['artifact_sha256'].items():
        if name=='docs/STATUS.md':continue
        assert digest(ROOT/name)==h, name
        count+=1
    result['p5c_bindings']=count
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    history=protected()
    from experiments.p5d_checkpoint import sources
    checkpoint_sources=sources()
    source=dict(checkpoint_sources)
    for name in ['experiments/controller_p5d_audit.py','experiments/controller_p5a_audit.py',
                 'experiments/p5d_package.py','experiments/p5c_diagnostic.py','scripts/run_gpu.ps1']:
        source[name]=digest(ROOT/name)
    inputs={}
    for folder in ['p5a-pool-v2','p5c-diagnostic-v1']:
        for path in (ROOT/'artifacts/evaluations'/folder).rglob('*'):
            if path.is_file():inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    pre=dict(version='gd-p5d-label-balanced-engineering-v1',checkpoint_sources=checkpoint_sources,
        sources=source,inputs=inputs,historical=history,objectives=['ordinary','label_balanced'],
        training_hands=32,unique_deals=16,resume_repeated_hands=8,validation_games=0,
        reserved_test_executed=False,model_promoted=False,timeout_seconds=600)
    assert json.loads(json.dumps(pre,allow_nan=False))==pre
    out.mkdir(parents=True)
    write(out/'preregistration.json',pre)
    with zipfile.ZipFile(out/'source-snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in source:z.write(ROOT/name,name)
    try:
        for objective in pre['objectives']:
            command=[sys.executable,str(ROOT/'scripts/p5d_run.py'),str(out/objective),
                     '--objective',objective,'--registration',str(out/'preregistration.json')]
            with (out/f'{objective}.log').open('x',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
            print(objective,'16 hands complete',flush=True)
        with (out/'resume.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/p5d_run.py'),str(out/'resume'),
                '--objective','label_balanced','--registration',str(out/'preregistration.json'),
                '--resume',str(out/'label_balanced/wave-2')],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
        for name,h in {**source,**inputs}.items():assert digest(ROOT/name)==h,name
        assert sources()==checkpoint_sources
        protected()
        artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}
        write(out/'run-receipt.json',dict(status='PASS',scope='ENGINEERING_RUN_ONLY',artifacts=artifacts,
            training_hands=32,unique_deals=16,resume_repeated_hands=8,validation_games=0,model_promoted=False))
        print('package run complete; independent acceptance pending',flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':main()
