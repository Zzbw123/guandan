"""Fixed P5d engineering worker; orchestrator supplies a frozen registration."""
from _bootstrap import ROOT
import sys, os
sys.path[:0] = [str(ROOT), str(ROOT/'experiments')]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, json
from pathlib import Path
import torch
from experiments.p5d_training import GPUTrainer
from experiments.p5d_checkpoint import save_checkpoint, load_checkpoint, sources


def deals(wave):
    return [(108800+n, 2+n%13, n%4) for n in range(wave*4, wave*4+4)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--objective',choices=['ordinary','label_balanced'],required=True)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    pre=json.loads(a.registration.read_text('utf-8'))
    assert sources()==pre['checkpoint_sources'], 'frozen source mismatch'
    a.output.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (a.output/name).open('x',encoding='utf-8') as f:
            json.dump(value,f,indent=2,allow_nan=False)
    write('source-before.json',sources())
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        cfg=dict(seed=314500,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,
                 num_envs=4,mode='mixed',objective=a.objective)
        trainer=load_checkpoint(a.resume) if a.resume else GPUTrainer(cfg)
        assert trainer.config==cfg
        if a.resume:
            assert a.objective=='label_balanced' and trainer.waves==2
        else:
            save_checkpoint(trainer,a.output/'initial')
        for wave in range(trainer.waves,4):
            row=trainer.train_wave(deals(wave))
            write(f'wave-{wave+1}.json',row)
            save_checkpoint(trainer,a.output/f'wave-{wave+1}')
            print(json.dumps(dict(objective=a.objective,wave=wave+1,samples=row['samples'],
                                  updates=row['updates'],batches=row['batches'])),flush=True)
        assert sources()==pre['checkpoint_sources']
        write('complete.json',dict(status='PASS',objective=a.objective,episodes=trainer.episodes,
            waves=trainer.waves,updates=trainer.updates,source_sha256=sources(),
            scope='ENGINEERING_ONLY',model_promoted=False,validation_games=0,reserved_test_executed=False))
    except BaseException as exc:
        write('failure.json',dict(type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':main()
