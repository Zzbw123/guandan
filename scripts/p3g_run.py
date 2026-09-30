"""Freeze and run the bounded, read-only P3g development diagnostic."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import subprocess
import sys
import zipfile
from _bootstrap import ROOT
sys.path.insert(0,str(ROOT/'experiments'))
from p3e_common import check,digest,read,write

BASE=ROOT/'artifacts/evaluations/p3f-teacher-v1'

def freeze(output):
    receipt=read(BASE/'delivery-receipt.json')
    for name,h in receipt['artifact_sha256'].items():
        check(digest(ROOT/name)==h,f'P3f delivery changed: {name}')
    output.mkdir(parents=True,exist_ok=False)
    sources=sorted((ROOT/'src').rglob('*.py'))
    sources+=sorted((ROOT/'experiments').glob('*.py'))
    sources+=sorted((ROOT/'experiments/p3d').glob('*.py'))
    sources+=sorted((ROOT/'scripts').glob('*.py'))
    sources+=sorted((ROOT/'tests').glob('test_p3g*.py'))
    sources += [ROOT/'scripts/run_gpu.ps1',ROOT/'docs/P3G_PROTOCOL.md',ROOT/'AGENTS.md']
    inputs=[BASE/'delivery-receipt.json',BASE/'controller-audit.json',BASE/'preregistration.json']
    for seed in (314380,314381,314382):
        tr=BASE/'training'/f'teacher-{seed}'
        inputs += [tr/'phase1/raw.pt',tr/'phase1/manifest.json',tr/'final/checkpoint.pt',tr/'final/manifest.json']
        inputs += sorted(p for p in (BASE/'evaluations'/f'teacher-{seed}').iterdir() if p.is_file())
    inputs += [BASE/'training/teacher-314380/phase1/replays.jsonl.gz']
    source_sha={p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    input_sha={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}
    with zipfile.ZipFile(output/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in source_sha:z.write(ROOT/name,name)
    order=[314380,314381,314382];Random(314394).shuffle(order)
    pre=dict(version='gd-p3g-fit-v1',utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_sha,input_sha256=input_sha,source_zip_sha256=digest(output/'source.zip'),
        previous_delivery_verified=len(receipt['artifact_sha256']),evaluation_order=order,
        new_games=624,reused_games=624,development_seeds=list(range(108100,108126)),
        bootstrap_seed=314395,bootstrap_replicates=5000,teacher_training_replays=200,
        selection='all finish opportunities plus first four other multi-choice states per game',
        numeric_audit='first 16 selected states per source, all six models, complete candidates',
        model_promoted=False,training_executed=False,new_validation_executed=False,reserved_test_executed=False)
    write(output/'preregistration.json',pre)
    print(json.dumps(dict(stage='frozen',order=order,source_files=len(source_sha),input_files=len(input_sha))),flush=True)
    return pre

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--freeze-only',action='store_true');p.add_argument('--run-frozen',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    pre=read(out/'preregistration.json') if a.run_frozen else freeze(out)
    if a.freeze_only:return
    commands=[['scripts/p3g_evaluate.py',str(out),str(seed)] for seed in pre['evaluation_order']]
    commands += [['experiments/p3g_scores.py',str(out)]]
    for index,command in enumerate(commands):
        with (out/f'job-{index}.log').open('x',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        check(result.returncode==0,f'P3g job {index} failed; preserve log')
        print(json.dumps(dict(stage='job_complete',index=index,command=command)),flush=True)
    write(out/'run-receipt.json',dict(status='PASS',preregistration_sha256=digest(out/'preregistration.json'),
        report_sha256={p.relative_to(out).as_posix():digest(p) for p in out.rglob('report.json')},
        new_games=624,model_promoted=False,training_executed=False,reserved_test_executed=False))

if __name__=='__main__':main()
