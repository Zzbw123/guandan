"""Bounded P5a engineering run; a fresh process may resume at wave 2."""
from _bootstrap import ROOT
import sys
sys.path.insert(0, str(ROOT))
import argparse
import json
import torch
from pathlib import Path
from experiments.p5a_training import GPUTrainer
from experiments.p5a_checkpoint import save_checkpoint, load_checkpoint, sources


def deals(wave):
    return [(108800+n, 2+n % 13, n % 4) for n in range(wave*4, wave*4+4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    (a.output/'source-before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    try:
        # A fresh resume process has not run GPUTrainer.__init__ yet. Establish
        # the deterministic execution mode before the loader verifies runtime.
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        trainer = load_checkpoint(a.resume) if a.resume else GPUTrainer(dict(
            seed=314500, epsilon=.1, lr=.001, batch_size=256, chunk_size=1024, num_envs=4, mode='mixed'))
        if a.resume and trainer.waves != 2:
            raise ValueError('only wave-2 engineering resume permitted')
        if not a.resume:
            save_checkpoint(trainer, a.output/'initial')
        for wave in range(trainer.waves, 4):
            result = trainer.train_wave(deals(wave))
            (a.output/f'wave-{wave+1}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            save_checkpoint(trainer, a.output/f'wave-{wave+1}')
            print(json.dumps({k:result[k] for k in ('wave', 'samples', 'updates', 'scored_candidates', 'frozen_candidates')}), flush=True)
        if sources() != before:
            raise ValueError('source changed during run')
        (a.output/'complete.json').write_text(json.dumps(dict(status='PASS', episodes=trainer.episodes,
            waves=trainer.waves, updates=trainer.updates, source_sha256=before, scope='ENGINEERING_ONLY',
            model_promoted=False, reserved_test_executed=False), indent=2), encoding='utf-8')
    except BaseException as exc:
        (a.output/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__, error=str(exc))), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
