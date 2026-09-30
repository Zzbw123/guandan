"""Strict P5f successful-batch checkpoints bound to teacher corpus and source."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from uuid import uuid4
import torch
from guandan.learning.checkpoint import _finite
from guandan.learning.encoding import FEATURE_VERSION
from guandan.learning.model import NETWORK_VERSION
from guandan.types import RULES_VERSION, ACTION_VERSION, OBSERVATION_VERSION
from experiments.p5f_training import TeacherTrainer, configure_runtime
from experiments.p5d_checkpoint import runtime, _cpu

ROOT = Path(__file__).resolve().parents[1]
META = ('versions','runtime','sources','config','corpus_sha256','updates','samples','candidates','boundary')


def versions():
    return dict(checkpoint='gd-p5f-checkpoint-v1', rules=RULES_VERSION, action=ACTION_VERSION,
                observation=OBSERVATION_VERSION, feature=FEATURE_VERSION, network=NETWORK_VERSION,
                output='raw-logits-v1', training='gd-p5f-fifo-v1')


def sources():
    paths = list((ROOT/'src').rglob('*.py'))
    paths += [ROOT/'experiments'/f'p5f_{name}.py' for name in
              ('protocol','objective','data','training','checkpoint')]
    paths += [ROOT/'experiments/p5d_checkpoint.py', ROOT/'docs/P5F_PROTOCOL.md']
    return {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def capture(trainer):
    if type(trainer) is not TeacherTrainer or not trainer.ready:
        raise ValueError('successful P5f batch boundary required')
    value = dict(versions=versions(), runtime=runtime(), sources=sources(),
                 config=trainer.config, corpus_sha256=trainer.corpus_sha256,
                 updates=trainer.updates, samples=trainer.samples, candidates=trainer.candidates,
                 boundary='successful_batch', model=_cpu(trainer.model.state_dict()),
                 optimizer=_cpu(trainer.optimizer.state_dict()), torch_rng=torch.get_rng_state(),
                 cuda_rng=[r.cpu().clone() for r in torch.cuda.get_rng_state_all()])
    _finite(value)
    return value


def save_checkpoint(trainer, directory):
    target = Path(directory)
    if target.exists():
        raise FileExistsError(target)
    payload = capture(trainer)
    buffer = BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    manifest = {k:payload[k] for k in META}
    manifest.update(sha256=sha256(data).hexdigest(), bytes=len(data))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(target.name+'.incomplete-'+uuid4().hex)
    stage.mkdir()
    for name, content in [('checkpoint.pt',data), ('manifest.json',
            (json.dumps(manifest,allow_nan=False,indent=2)+'\n').encode())]:
        with (stage/name).open('xb') as f:
            f.write(content); f.flush(); os.fsync(f.fileno())
    if target.exists():
        raise FileExistsError(target)
    os.rename(stage,target)
    return manifest


def read_payload(directory, expected_sha256, expected_manifest_sha256=None):
    if type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None:
        raise ValueError('pinned checkpoint SHA256 required')
    directory = Path(directory)
    manifest_bytes = (directory/'manifest.json').read_bytes()
    if expected_manifest_sha256 is not None and sha256(manifest_bytes).hexdigest()!=expected_manifest_sha256:
        raise ValueError('manifest hash mismatch')
    manifest = json.loads(manifest_bytes)
    data = (directory/'checkpoint.pt').read_bytes()
    if (sha256(data).hexdigest()!=expected_sha256 or manifest.get('sha256')!=expected_sha256 or
            manifest.get('bytes')!=len(data)):
        raise ValueError('checkpoint hash/size mismatch')
    payload = torch.load(BytesIO(data),map_location='cpu',weights_only=True)
    if type(payload) is not dict or set(payload)!=set(META)|{'model','optimizer','torch_rng','cuda_rng'}:
        raise ValueError('payload keys')
    if set(manifest)!=set(META)|{'sha256','bytes'}:
        raise ValueError('manifest keys')
    if any(payload[k]!=manifest[k] for k in META):
        raise ValueError('manifest metadata mismatch')
    if payload['versions']!=versions() or payload['sources']!=sources() or payload['boundary']!='successful_batch':
        raise ValueError('version/source/boundary mismatch')
    for name in ('updates','samples','candidates'):
        if type(payload[name]) is not int or payload[name]<0:
            raise ValueError('invalid counter')
    n,u=payload['samples'],payload['updates']
    if (u==0)!=(n==0) or (u and not u<=n<=u*64) or payload['candidates']<n:
        raise ValueError('inconsistent counters')
    _finite(payload)
    return payload,manifest


def restore(directory,expected_sha256,expected_corpus_sha256,expected_manifest_sha256=None):
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=[r.clone() for r in torch.cuda.get_rng_state_all()]
    try:
        configure_runtime(0)
        p,m=read_payload(directory,expected_sha256,expected_manifest_sha256)
        if p['corpus_sha256']!=expected_corpus_sha256 or p['runtime']!=runtime():
            raise ValueError('corpus/runtime mismatch')
        trainer=TeacherTrainer(p['config'],p['corpus_sha256'])
        ref=trainer.model.state_dict()
        if set(p['model'])!=set(ref) or any(not torch.is_tensor(p['model'][k]) or
                p['model'][k].shape!=v.shape or p['model'][k].dtype!=v.dtype for k,v in ref.items()):
            raise ValueError('model shape/dtype')
        state=p['optimizer']
        if (set(state)!={'state','param_groups'} or
                state['param_groups']!=trainer.optimizer.state_dict()['param_groups']):
            raise ValueError('optimizer configuration')
        params=list(trainer.model.parameters())
        if set(state['state'])!=(set(range(len(params))) if p['updates'] else set()):
            raise ValueError('optimizer coverage')
        for i, row in state['state'].items():
            if set(row)!={'step','exp_avg','exp_avg_sq'}:
                raise ValueError('Adam keys')
            step=row['step']
            if not torch.is_tensor(step) or step.shape!=torch.Size([]) or step.dtype!=torch.float32 or step.item()!=p['updates']:
                raise ValueError('Adam step')
            for key in ('exp_avg','exp_avg_sq'):
                if not torch.is_tensor(row[key]) or row[key].shape!=params[i].shape or row[key].dtype!=torch.float32:
                    raise ValueError('Adam tensor shape/dtype')
            if (row['exp_avg_sq']<0).any():
                raise ValueError('negative Adam second moment')
        if (not torch.is_tensor(p['torch_rng']) or p['torch_rng'].dtype!=torch.uint8 or
                p['torch_rng'].shape!=cpu_rng.shape or type(p['cuda_rng']) is not list or
                len(p['cuda_rng'])!=len(cuda_rng)):
            raise ValueError('RNG format')
        for r,ref_rng in zip(p['cuda_rng'],cuda_rng):
            if not torch.is_tensor(r) or r.dtype!=torch.uint8 or r.shape!=ref_rng.shape:
                raise ValueError('CUDA RNG format')
        trainer.model.load_state_dict(p['model'],strict=True)
        trainer.optimizer.load_state_dict(state)
        trainer.updates,trainer.samples,trainer.candidates=p['updates'],p['samples'],p['candidates']
        torch.set_rng_state(p['torch_rng'])
        torch.cuda.set_rng_state_all(p['cuda_rng'])
        return trainer,m
    except BaseException:
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        raise


def load_model(directory,expected_sha256,expected_manifest_sha256=None):
    # Corpus is pinned by the manifest hash in production guard calls.
    _,manifest=read_payload(directory,expected_sha256,expected_manifest_sha256)
    trainer,manifest=restore(directory,expected_sha256,manifest['corpus_sha256'],expected_manifest_sha256)
    model=trainer.model.eval()
    model.requires_grad_(False)
    return model,manifest
