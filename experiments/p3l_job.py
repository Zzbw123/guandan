"""One fixed P3l training job, preserving inherited teacher model lineage."""
from hashlib import sha256
from pathlib import Path
import time
import torch
from experiments.p3f_training import _dmc_phase, model_digest, reset_phase2
from experiments.p3l_protocol import config, INITS, ARMS
from experiments.p3l_training import GPUTrainer
from experiments.p3l_checkpoint import save_checkpoint, runtime
from experiments.p3e_common import check, read, write, digest

ROOT=Path(__file__).resolve().parents[1]
def train(root,job):
    check(job in [f'{a}-{s}' for s in INITS for a in ARMS], 'unknown P3l job')
    arm,seedtext=job.split('-');seed=int(seedtext);cfg=config(seed,arm)
    source=ROOT/'artifacts/evaluations/p3f-teacher-v1/training'/f'teacher-{seed}'/'phase1'
    pre=read(root/'preregistration.json')
    for name in ('raw.pt','manifest.json'):
        check(digest(source/name)==pre['input_sha256'][(source/name).relative_to(ROOT).as_posix()], 'pinned teacher input')
    basecfg={k:v for k,v in cfg.items() if k not in ('auxiliary_weight','objective')}
    previous,reset=reset_phase2(source,basecfg)
    trainer=GPUTrainer(cfg);trainer.model.load_state_dict(previous.model.state_dict());del previous
    check(model_digest(trainer.model)==reset['phase1_model_sha256'], 'paired teacher starting model')
    reset.update(auxiliary_weight=cfg['auxiliary_weight'], objective=arm,
        policy_rng_repr_sha256=sha256(repr(trainer.rng.getstate()).encode()).hexdigest(),
        torch_rng_sha256=sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        cuda_rng_sha256=[sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()])
    output=root/'training'/job;output.mkdir(parents=True,exist_ok=False)
    write(output/'reset.json',reset)
    start=time.perf_counter()
    phase=_dmc_phase(trainer,200,800,output/'waves.jsonl',output/'replays.jsonl.gz')
    check(trainer.episodes==600 and trainer.waves==150, 'fixed training budget')
    manifest=save_checkpoint(trainer,output/'final')
    write(output/'report.json',dict(status='PASS',job=job,arm=arm,seed=seed,config=cfg,
        phase=phase,runtime=runtime(),final_manifest=manifest,reset=reset,
        model_sha256=model_digest(trainer.model),seconds=time.perf_counter()-start,
        inherited_hands=200,new_hands=600,model_promoted=False))
