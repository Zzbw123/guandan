"""Execute a fresh, source-frozen, six-job paired teacher pretraining study."""
from _bootstrap import ROOT
import sys
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'scripts')]
from pathlib import Path
import json,subprocess,time,traceback
from experiments.p5f_run_common import freeze,verify_frozen,check,read,write,digest
from experiments.p5f_protocol import specification,TRAIN,DEV
from experiments.p5f_data import build_corpus


def child(root,stage,job):
    with (root/f'{stage}-{job}.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'scripts/p5f_worker.py'),str(root),stage,job],
                                 stdout=log,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=7200)
        except BaseException:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=15)
            process.wait(timeout=15);raise
    check(code==0,f'{stage} {job} exit {code}')
    print(json.dumps(dict(stage=stage,job=job,status='PASS')),flush=True)


def main(root,engineering):
    e=read(engineering/'receipt.json')
    check(e['status']=='PASS' and e['fit_gate']=='PASS' and e['probe_updates']==512,'engineering/fit admission gate')
    verify_frozen(engineering)
    for name,h in e['artifact_sha256'].items():check(digest(engineering/name)==h,'engineering artifact drift')
    freeze(root,'paired-study',[engineering/'receipt.json'])
    started=time.perf_counter()
    try:
        for name,deals in [('train',TRAIN),('development',DEV)]:
            manifest=build_corpus(root/'corpus'/name,deals)
            print(json.dumps(dict(stage='corpus',name=name,hands=manifest['hands'],observations=manifest['observations'])),flush=True)
        spec=specification()
        for job in spec['training_order']:child(root,'train',job)
        candidates={job:read(root/'training'/job/'final/manifest.json')['sha256'] for job in spec['training_order']}
        write(root/'candidates.json',dict(selection='final one-pass only; all six fixed before development scoring',candidates=candidates))
        for job in spec['development_order']:
            child(root,'fit',job);child(root,'evaluate',job)
        verify_frozen(root)
        write(root/'receipt.json',dict(status='RUN_COMPLETE_PENDING_AUDIT',seconds=time.perf_counter()-started,
            model_promoted=False,reserved_test_executed=False,
            artifact_sha256={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}))
    except BaseException:
        write(root/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
