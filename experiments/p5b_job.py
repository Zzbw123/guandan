"""Fixed-budget training using unchanged, accepted P5a trainer/checkpoint."""
from pathlib import Path
import gzip,json,time
from experiments.p5a_training import GPUTrainer,weight_digest
from experiments.p5a_checkpoint import save_checkpoint,runtime
from experiments.p5b_protocol import config,INITS,ARMS
from experiments.p3e_common import check,write

def train(root,job):
    check(job in [f'{a}-{s}' for s in INITS for a in ARMS],'fixed training job')
    arm,seed=job.split('-');seed=int(seed);cfg=config(seed,arm)
    out=root/'training'/job;out.mkdir(parents=True,exist_ok=False)
    trainer=GPUTrainer(cfg);initial=save_checkpoint(trainer,out/'initial')
    started=time.perf_counter();totals=dict(samples=0,decisions=0,scored_candidates=0,frozen_candidates=0)
    with gzip.open(out/'waves.jsonl.gz','xt',encoding='utf-8') as f:
        for wave in range(400):
            deals=[(100000+n,2+n%13,n%4) for n in range(wave*4,wave*4+4)]
            row=trainer.train_wave(deals)
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            totals['decisions']+=sum(h['steps'] for h in row['hands'])
            for key in ('samples','scored_candidates','frozen_candidates'):totals[key]+=row[key]
            if wave+1 in (1,200):save_checkpoint(trainer,out/f'wave-{wave+1}')
            if (wave+1)%25==0:print(json.dumps(dict(job=job,wave=wave+1,episodes=trainer.episodes,
                updates=trainer.updates,**totals,seconds=time.perf_counter()-started)),flush=True)
    check(trainer.episodes==1600 and trainer.waves==400,'fixed training budget')
    final=save_checkpoint(trainer,out/'final')
    write(out/'report.json',dict(status='PASS',job=job,config=cfg,initial_manifest=initial,final_manifest=final,
        hands=trainer.episodes,waves=trainer.waves,updates=trainer.updates,**totals,
        pool_sha256=trainer.pool_sha256,model_sha256=weight_digest(trainer.model),runtime=runtime(),
        seconds=time.perf_counter()-started,model_promoted=False,reserved_test_executed=False))
