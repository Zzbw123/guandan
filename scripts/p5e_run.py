"""Freeze and execute six equal-hand jobs followed by final-only evaluation."""
from _bootstrap import ROOT
import sys
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,time,traceback,zipfile
from experiments.p5e_protocol import specification,INITS,ARMS
from experiments.p3e_common import check,read,write,digest

def prerequisites():
    folder=ROOT/'artifacts/evaluations/p5d-balanced-v1'
    receipt=read(folder/'delivery-receipt.json')
    check(receipt['status']=='PASS' and receipt['scope']=='ACCEPTED_LABEL_BALANCED_ENGINEERING_V1'
          and receipt['full_gpu_tests']>=262 and receipt['skipped']==0,'P5d acceptance')
    for name,h in receipt['artifact_sha256'].items():
        if name=='docs/STATUS.md':continue
        check(digest(ROOT/name)==h,'P5d delivery artifact '+name)
    previous=read(folder/'preregistration.json')
    check(previous['version']=='gd-p5d-label-balanced-engineering-v1'
          and bool(previous['historical']), 'P5d frozen preregistration/history')
    for group in ('checkpoint_sources','sources','inputs'):
        check(bool(previous[group]), 'P5d empty '+group)
        for name,h in previous[group].items():
            check(digest(ROOT/name)==h,'P5d prereg '+group+' '+name)
    from experiments.p5d_package import protected as verify_history
    check(verify_history()==previous['historical'],'P5d registered history drift')
    scan=read(ROOT/'artifacts/evaluations/p5e-seed-availability.json')
    check(scan['status']=='PASS' and not scan['hits'] and not scan['errors'],'fresh validation seed availability')
    check(scan['validation_seeds']==list(range(208000,208065)),'validation seed scan range')
    for name,h in scan['files'].items():check(digest(ROOT/name)==h,'historical seed file changed '+name)
    log=ROOT/'artifacts/evaluations/p5e-tests-gpu-v1.log'
    text=log.read_text('utf-8-sig')
    import re
    match=re.search(r'Ran (\d+) tests?',text)
    check(match is not None and int(match.group(1))>=7 and text.rstrip().endswith('OK')
          and 'skipped=' not in text,'P5e engineering tests')
    return [folder/'delivery-receipt.json',folder/'preregistration.json',
            folder/'controller-audit.json',ROOT/'artifacts/evaluations/p5e-seed-availability.json',log],scan

def child(out,stage,job):
    with (out/f'{stage}-{job}.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'scripts/p5e_worker.py'),str(out),stage,job],stdout=log,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=3600)
        except BaseException:
            if sys.platform=='win32':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=10)
            else:process.kill()
            process.wait(timeout=10);raise
    check(code==0,f'{stage} {job} exit {code}')
    print(json.dumps(dict(stage=stage,job=job,status='PASS')),flush=True)

def main():
    check(len(sys.argv)==2,'one fresh output path')
    out=Path(sys.argv[1]).resolve()
    check(not out.exists(),'output exists')
    check(out.is_relative_to(ROOT/'artifacts/evaluations') and out.name.startswith('p5e-'),'isolated output')
    inputs,scan=prerequisites();spec=specification()
    check(json.loads(json.dumps(spec,allow_nan=False))==spec,'specification JSON roundtrip')
    from experiments.p5d_checkpoint import sources as old_sources
    paths=set(ROOT/name for name in old_sources())
    # Include every controller and transitive historical Python dependency.
    paths.update((ROOT/'experiments').glob('*.py'))
    paths.update((ROOT/'experiments/p3d').rglob('*.py'))
    paths.update((ROOT/'scripts').glob('*.py'))
    paths.update((ROOT/'experiments').glob('p5e_*.py'))
    paths.update((ROOT/'experiments').glob('controller_p5e_*.py'))
    paths.update((ROOT/'scripts').glob('p5e_*.py'))
    paths.update((ROOT/'tests').glob('test_p5e*.py'))
    paths.update(ROOT/name for name in ['scripts/_bootstrap.py','scripts/run_gpu.ps1','scripts/p3d_evaluate.py',
        'experiments/p3d/protocol.py','experiments/p3d/guard.py','experiments/p3e_common.py',
        'experiments/p3e_metrics.py','docs/P5E_PROTOCOL.md',
        'experiments/p5b_protocol.py','experiments/p5b_job.py','experiments/p5b_guard.py',
        'scripts/p5b_worker.py'])
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}
    input_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}
    out.mkdir(parents=True)
    write(out/'preregistration.json',dict(utc=datetime.now(timezone.utc).isoformat(),specification=spec,
        source_sha256=hashes,input_sha256=input_hashes,historical_seed_files=scan['files']))
    check(read(out/'preregistration.json')['specification']==spec,'written specification roundtrip')
    for name,items in [('source-snapshot.zip',paths),('inputs-snapshot.zip',inputs)]:
        with zipfile.ZipFile(out/name,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(items):z.write(p,p.relative_to(ROOT).as_posix())
    started=time.perf_counter()
    try:
        for job in spec['training_order']:child(out,'train',job)
        candidates={}
        for job in spec['training_order']:
            report=read(out/'training'/job/'report.json')
            manifest=read(out/'training'/job/'final/manifest.json')
            check(report['status']=='PASS' and report['hands']==1600 and report['waves']==400,
                  'incomplete training '+job)
            check(manifest['config']==spec['configs'][job] and manifest['episodes']==1600
                  and manifest['waves']==400 and report['final_manifest']==manifest,
                  'final checkpoint metadata '+job)
            check(digest(out/'training'/job/'final/checkpoint.pt')==manifest['sha256'],
                  'final checkpoint bytes '+job)
            candidates[job]=manifest['sha256']
        write(out/'candidates.json',dict(primary=spec['primary_candidate'],selection='final wave 400 only',
            candidates=candidates,preregistration_sha256=digest(out/'preregistration.json')))
        for job in spec['development_order']+spec['validation_order']:child(out,'evaluate',job)
        for path,h in hashes.items():check(digest(ROOT/path)==h,'source drift '+path)
        for path,h in input_hashes.items():check(digest(ROOT/path)==h,'input drift '+path)
        prerequisites()
        for job,h in candidates.items():
            check(digest(out/'training'/job/'final/checkpoint.pt')==h,
                  'candidate changed after evaluation '+job)
        write(out/'receipt.json',dict(status='RUN_COMPLETE_PENDING_AUDIT',seconds=time.perf_counter()-started,
            model_promoted=False,reserved_test_executed=False,
            artifact_sha256={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}))
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
