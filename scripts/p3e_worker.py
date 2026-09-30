"""One bounded P3e training or evaluation job. Caller supplies frozen root."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import gzip
import json
import os
import sys
import time
from pathlib import Path
from _bootstrap import ROOT
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'experiments/p3d'))
from p3e_protocol import config,schedule,specification,OLD,OLD_HASH
from p3e_common import check,digest,read,write,summarize,behavior,ratios
from p3e_metrics import sum_metrics

def train(root,seed):
    from guandan_gpu.training import GPUTrainer
    from guandan_gpu.checkpoint import save_checkpoint,runtime
    output=root/'training'/str(seed); output.mkdir(parents=True)
    trainer=GPUTrainer(config(seed)); started=time.perf_counter(); samples=0
    with (output/'waves.jsonl').open('x',encoding='utf-8') as log:
        for w in range(400):
            row=trainer.train_wave([(105000+i,2+i%13,i%4) for i in range(4*w,4*w+4)])
            samples+=row['samples'];row['elapsed_s']=time.perf_counter()-started
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if (w+1)%100==0:
                os.fsync(log.fileno());save_checkpoint(trainer,output/f'wave-{w+1}')
                print(json.dumps(dict(stage='training',seed=seed,hands=trainer.episodes,
                      updates=trainer.updates,elapsed_s=row['elapsed_s'])),flush=True)
    write(output/'report.json',dict(status='PASS',seed=seed,hands=1600,waves=400,samples=samples,
        updates=trainer.updates,elapsed_s=time.perf_counter()-started,runtime=runtime()))

def evaluate(root,job):
    from guard import GPUInferenceGuard
    from p3d_evaluate import run_trial
    baseline=job=='baseline'; validation=job=='validation'
    trials=schedule(validation,baseline)
    if job=='old': checkpoint=ROOT/OLD; expected=OLD_HASH
    elif baseline: checkpoint=None;expected=None
    else:
        seed,hands=('314370','1600') if validation else job.split('-')
        checkpoint=root/'training'/seed/f'wave-{int(hands)//4}'
        expected=read(checkpoint/'manifest.json')['sha256']
    if checkpoint: check(digest(checkpoint/'checkpoint.pt')==expected,'candidate hash')
    output=root/'evaluations'/job;output.mkdir(parents=True)
    write(output/'binding.json',dict(utc=datetime.now(timezone.utc).isoformat(),job=job,
        checkpoint=str(checkpoint.relative_to(ROOT)).replace('\\','/') if checkpoint else None,
        candidate_sha256=expected,preregistration_sha256=digest(root/'preregistration.json'),games=len(trials)))
    started=time.perf_counter();rows=[];behaviors=[]
    context=nullcontext(None) if baseline else GPUInferenceGuard(checkpoint,expected)
    with context as guard, (output/'results.jsonl').open('x',encoding='utf-8') as result_file, \
         gzip.open(output/'measurements.jsonl.gz','xt',encoding='utf-8') as measurements, \
         gzip.open(output/'replays.jsonl.gz','xt',encoding='utf-8') as replays, \
         (output/'behavior.jsonl').open('x',encoding='utf-8') as counts_file:
        for trial in trials:
            row,measurement,replay=run_trial(guard,trial)
            result_file.write(json.dumps(row,allow_nan=False)+'\n');result_file.flush()
            measurements.write(json.dumps(measurement,allow_nan=False)+'\n');measurements.flush()
            replays.write(json.dumps(dict(trial_id=trial.trial_id,replay=replay),allow_nan=False)+'\n');replays.flush()
            check(row['status']=='ok',f'failed trial {trial.trial_id}: {row["error"]}')
            b=behavior(replay,trial);behaviors.append(b)
            counts_file.write(json.dumps(b,allow_nan=False)+'\n');counts_file.flush();rows.append(row)
            if len(rows)%104==0: print(json.dumps(dict(stage=job,games=len(rows),total=len(trials))),flush=True)
    if guard is not None: check(not guard.is_alive() and guard.closed,'worker teardown')
    counts=sum_metrics(b['counts'] for b in behaviors)
    summary=summarize(rows,trials)
    primary=summary['dmc|dmc|greedy']['metrics']['win_rate'] if validation else None
    gate=('VALIDATION_GATE_PASSED' if primary['estimate']>=.55 and primary['ci95'] and
           primary['ci95'][0]>.5 and not primary['degenerate'] else 'NOT_ESTABLISHED') if validation else 'DEVELOPMENT_ONLY'
    write(output/'report.json',dict(status='PASS',job=job,candidate_sha256=expected,games=len(rows),
        summary=summary,counts=counts,ratios=ratios(counts),validation_gate=gate,
        model_promoted=False,reserved_test_executed=False,worker_closed=True,
        elapsed_s=time.perf_counter()-started))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('stage',choices=['train','evaluate']);p.add_argument('job')
    a=p.parse_args();root=a.root.resolve()
    pre=read(root/'preregistration.json');check(pre['specification']==specification(),'frozen specification')
    for name,value in pre['source_sha256'].items():check(digest(ROOT/name)==value,f'source drift {name}')
    if a.stage=='train':
        check(int(a.job) in (314370,314371,314372),'training initializer');train(root,int(a.job))
    else:
        check(a.job in ['old','baseline','validation']+specification()['development_order'],'evaluation job');evaluate(root,a.job)
if __name__=='__main__':main()
