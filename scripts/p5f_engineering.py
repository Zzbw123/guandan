"""Frozen engineering, cross-process resume, and fixed 512-step fit gate."""
from _bootstrap import ROOT
import sys
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'scripts')]
from pathlib import Path
import json,subprocess,traceback,time
import torch
from experiments.p5f_protocol import TRAIN,config
from experiments.p5f_run_common import freeze,verify_frozen,write,read,digest,check
from experiments.p5f_data import build_corpus,iter_batches
from experiments.p5f_training import TeacherTrainer
from experiments.p5f_checkpoint import save_checkpoint,restore,capture
from experiments.p5f_job import probe_requests,score_rows,fit_summary
from experiments.controller_p5f_audit import independent_step


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def resumed(root,objective):
    verify_frozen(root)
    cp=root/objective/'batch-2';m=read(cp/'manifest.json')
    trainer,_=restore(cp,m['sha256'],digest(root/'corpus/manifest.json'))
    logs=[]
    for i,batch in enumerate(iter_batches(root/'corpus',64,start_batch=2),start=3):
        logs.append(trainer.update(batch))
        if i==4:break
    save_checkpoint(trainer,root/objective/'resumed-4')
    write(root/objective/'resumed-logs.json',logs)


def main(root):
    log=ROOT/'artifacts/evaluations/p5f-tests-gpu-v2.log'
    text=log.read_text(encoding='utf-8-sig')
    check(text.rstrip().endswith('OK') and 'skipped=' not in text,'P5f GPU tests passed without skips')
    freeze(root,'engineering',[log])
    started=time.perf_counter()
    try:
        build_corpus(root/'corpus',TRAIN[:13])
        batches=[]
        for batch in iter_batches(root/'corpus',64):
            batches.append(batch)
            if len(batches)==4:break
        numerical=[]
        for objective in ('regression','ranking'):
            trainer=TeacherTrainer(config(314560,objective),digest(root/'corpus/manifest.json'))
            independent=TeacherTrainer(config(314560,objective),trainer.corpus_sha256)
            logs=[];max_error=0.
            for index,batch in enumerate(batches,start=1):
                reference,grads=independent_step(independent,batch)
                row=trainer.update(batch);logs.append(row)
                check(abs(reference-row['loss'])<=3e-6+3e-4*abs(reference),'engineering independent loss')
                for p,q,g in zip(trainer.model.parameters(),independent.model.parameters(),grads):
                    torch.testing.assert_close(p.grad,g,rtol=3e-4,atol=3e-6)
                    torch.testing.assert_close(p,q,rtol=3e-4,atol=3e-6)
                    max_error=max(max_error,float((p-q).abs().max()))
                if index==2:save_checkpoint(trainer,root/objective/'batch-2')
            save_checkpoint(trainer,root/objective/'continuous-4')
            write(root/objective/'continuous-logs.json',logs)
            with (root/objective/'resume-process.log').open('x',encoding='utf-8') as output:
                child=subprocess.run([sys.executable,__file__,'--resume',str(root),objective],
                    stdout=output,stderr=subprocess.STDOUT,timeout=180)
            check(child.returncode==0,'cross-process resume failed')
            a=torch.load(root/objective/'continuous-4/checkpoint.pt',weights_only=True)
            b=torch.load(root/objective/'resumed-4/checkpoint.pt',weights_only=True)
            check(equal(a,b),'cross-process entire payload exact equality')
            check(logs[2:]==read(root/objective/'resumed-logs.json'),'cross-process logs equality')
            numerical.append(dict(objective=objective,batches=4,max_model_error=max_error,resume_exact=True))
        selected=probe_requests(root/'corpus');requests=[(o,l) for _,_,o,l in selected]
        check(0<len(requests)<=26,'fixed fit probe selection')
        trainer=TeacherTrainer(config(314560,'ranking'),digest(root/'corpus/manifest.json'))
        save_checkpoint(trainer,root/'overfit/initial')
        write(root/'overfit/before.json',score_rows(trainer.model,selected))
        with (root/'overfit/batches.jsonl').open('x',encoding='utf-8') as output:
            for i in range(512):
                row=trainer.update(requests);output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
                if (i+1)%64==0:print(json.dumps(dict(stage='fit-probe',updates=i+1,loss=row['loss'])),flush=True)
        save_checkpoint(trainer,root/'overfit/final')
        rows=score_rows(trainer.model,selected);summary=fit_summary(rows)
        write(root/'overfit/after.json',rows)
        gate=summary['multi']['accuracy']>=.99
        verify_frozen(root)
        receipt=dict(status='PASS',scope='ACCEPTED_RANKING_PRETRAINING_ENGINEERING_V1',
            numerical=numerical,probe_summary=summary,probe_updates=512,
            fit_gate='PASS' if gate else 'PRETRAINING_FIT_GATE_NOT_MET',
            seconds=time.perf_counter()-started,model_promoted=False,reserved_test_executed=False,
            artifact_sha256={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()})
        write(root/'receipt.json',receipt)
        print(json.dumps(receipt['probe_summary']),flush=True)
    except BaseException:
        write(root/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise


if __name__=='__main__':
    if sys.argv[1]=='--resume':resumed(Path(sys.argv[2]),sys.argv[3])
    else:main(Path(sys.argv[1]).resolve())
