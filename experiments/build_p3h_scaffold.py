"""One-time local construction of isolated P3h adapters from immutable P3f code.

Not an experiment entrypoint. Existing source files are never overwritten.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def put(name, text):
    with (ROOT/name).open('x', encoding='utf-8', newline='\n') as f: f.write(text)

source=(ROOT/'src/guandan_gpu/training.py').read_text('utf-8')
body=source[source.index('    def train_wave(self, deals):'):]
body=body.replace('env = HandEnv()', 'env = base.HandEnv()')
body=body.replace('obs.player_id % 2))', 'obs.player_id % 2, (obs, candidates)))')
body=body.replace('samples.extend((state, action, settlement.team_rewards[team])\n                           for state, action, team in run["trajectory"])', 'samples.extend((state, action, settlement.team_rewards[team], request)\n                           for state, action, team, request in run["trajectory"])')
body=body.replace('weighted_loss = 0.0', 'weighted_loss = 0.0\n        auxiliary_candidates = 0\n        auxiliary_weighted_loss = 0.0\n        batch_records = []')
body=body.replace('            loss.backward()\n', '''            loss.backward()
            auxiliary = dict(loss=0., candidates=0, requests=0)
            if self.config['auxiliary_weight']:
                auxiliary = auxiliary_backward(self.model, [x[3] for x in batch],
                    self.config['auxiliary_weight'], self.config['chunk_size'])
            auxiliary_candidates += auxiliary['candidates']
            auxiliary_weighted_loss += auxiliary['loss'] * len(batch)
            batch_records.append(dict(samples=len(batch), dmc_loss=float(loss.item()),
                auxiliary_loss=auxiliary['loss'], auxiliary_candidates=auxiliary['candidates'],
                auxiliary_requests=auxiliary['requests']))
''')
body=body.replace('device_proof=dict(device=str(device)', '''auxiliary_candidates=auxiliary_candidates,
                    auxiliary_loss=auxiliary_weighted_loss / len(samples),
                    auxiliary_weight=self.config['auxiliary_weight'], batches=batch_records,
                    device_proof=dict(device=str(device)''')
put('experiments/p3h_training.py', '''"""P3h isolated DMC plus complete-candidate auxiliary regression.

Sampling/update order derives from frozen GPUTrainer; zero weight is tested
bitwise against it. Only observation and complete legal actions enter targets.
"""
import math
import torch
import guandan_gpu.training as base
from guandan_gpu.training import (GPUTrainer as BaseTrainer, _validated_deals,
    _model_device, _cuda_float32, _MAX_STEPS, score_many)
from guandan.env import HandEnv
from guandan.learning.encoding import encode_observation, encode_action
from experiments.p3h_objective import auxiliary_backward

class GPUTrainer(BaseTrainer):
    def __init__(self, config):
        config = dict(config)
        weight = config.pop('auxiliary_weight', None)
        if type(weight) not in (int, float) or weight not in (0., .1):
            raise ValueError('P3h auxiliary_weight must be exactly 0 or 0.1')
        super().__init__(config)
        self.config['auxiliary_weight'] = float(weight)

'''+body)

source=(ROOT/'src/guandan_gpu/checkpoint.py').read_text('utf-8')
source=source.replace('from guandan_gpu.training import GPUTrainer','from experiments.p3h_training import GPUTrainer')
source=source.replace('checkpoint="gd-gpu-checkpoint-v1", training="gd-gpu-wave-v1"','checkpoint="gd-p3h-checkpoint-v1", training="gd-p3h-wave-v1"')
start=source.index('def sources():'); end=source.index('\n\ndef runtime():',start)
source=source[:start]+'''def sources():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root/'src').rglob('*.py')) + sorted((root/'experiments').glob('p3h*.py'))
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
'''+source[end:]
put('experiments/p3h_checkpoint.py',source)
source=(ROOT/'experiments/p3d/guard.py').read_text('utf-8')
source=source.replace('from guandan_gpu.training import GPUTrainer, score_many','from experiments.p3h_training import GPUTrainer, score_many')
source=source.replace('from guandan_gpu.checkpoint import load_checkpoint','from experiments.p3h_checkpoint import load_checkpoint')
put('experiments/p3h_guard.py',source)
source=(ROOT/'scripts/p3f_worker.py').read_text('utf-8').replace('P3f','P3h').replace('p3f_protocol','p3h_protocol').replace('validation-teacher','validation-aux').replace('("control", "teacher")','("control", "aux")')
source=source.replace('from guard import GPUInferenceGuard','from experiments.p3h_guard import GPUInferenceGuard')
source=source.replace('manifest["config"] == config(seed)','manifest["config"] == config(seed, arm)')
source=source.replace('314385','314405').replace('from p3f_training import train','from experiments.p3h_job import train')
source=source.replace('    return digest(root / "preregistration.json")','''    for name, expected in pre['input_sha256'].items():
        check(digest(ROOT/name) == expected, 'frozen input drift '+name)
    return digest(root / "preregistration.json")''')
put('scripts/p3h_worker.py',source)
source=(ROOT/'experiments/p3f_scan.py').read_text('utf-8').replace('p3f','p3h').replace('P3f','P3h')
put('experiments/p3h_scan.py',source)

# Independent replay/statistical implementation is retained in the new review;
# only experiment-specific schedule/checkpoint schema changes.
source=(ROOT/'experiments/review_p3f.py').read_text('utf-8')
part=source[:source.index('\ndef tensor_digest')]
part=part.replace('p3f_protocol','p3h_protocol').replace('314385','314405')
put('experiments/p3h_review_evaluation.py',part)
