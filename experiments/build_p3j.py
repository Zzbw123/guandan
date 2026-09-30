"""One-shot isolated copies of frozen P3h infrastructure; never changes history."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def copy(old,new,changes=()):
    text=(ROOT/old).read_text('utf-8').replace('p3h','p3j').replace('P3h','P3j').replace('314405','314415')
    for a,b in changes:
        if a not in text:raise ValueError(f'missing replacement {new}: {a[:90]}')
        text=text.replace(a,b)
    with (ROOT/new).open('x',encoding='utf-8',newline='\n') as f:f.write(text)

copy('experiments/p3h_training.py','experiments/p3j_training.py',[
 ('from experiments.p3j_objective import auxiliary_backward',
  'from experiments.p3h_objective import auxiliary_backward\nfrom experiments.p3j_objective import centered_backward'),
 ("weight = config.pop('auxiliary_weight', None)",
  "objective = config.pop('objective', None)\n        if objective not in ('absolute', 'centered'):\n            raise ValueError('P3j objective must be absolute or centered')\n        weight = config.pop('auxiliary_weight', None)"),
 ("weight not in (0., .1)","weight != .1"),
 ("exactly 0 or 0.1","exactly 0.1"),
 ("self.config['auxiliary_weight'] = float(weight)","self.config['auxiliary_weight'] = float(weight)\n        self.config['objective'] = objective"),
 ("auxiliary = auxiliary_backward(self.model, [x[3] for x in batch],", "objective_fn = auxiliary_backward if self.config['objective']=='absolute' else centered_backward\n                auxiliary = objective_fn(self.model, [x[3] for x in batch],"),
 ("auxiliary_weight=self.config['auxiliary_weight'], batches=batch_records,","auxiliary_weight=self.config['auxiliary_weight'], objective=self.config['objective'], batches=batch_records,")])
copy('experiments/p3h_checkpoint.py','experiments/p3j_checkpoint.py',[
 ("paths = sorted((root/'src').rglob('*.py')) + sorted((root/'experiments').glob('p3j*.py'))",
  "paths = sorted((root/'src').rglob('*.py')) + sorted((root/'experiments').glob('p3j*.py'))\n    paths += [root/'experiments/p3h_objective.py', root/'experiments/p3f_teacher.py']")])
copy('experiments/p3h_guard.py','experiments/p3j_guard.py')
copy('experiments/p3h_job.py','experiments/p3j_job.py',[
 ("if k!='auxiliary_weight'","if k not in ('auxiliary_weight','objective')"),
 ("reset.update(auxiliary_weight=cfg['auxiliary_weight'],","reset.update(auxiliary_weight=cfg['auxiliary_weight'], objective=arm,")])
copy('scripts/p3h_worker.py','scripts/p3j_worker.py',[
 ('"validation-control", "validation-aux"','"validation-absolute", "validation-centered"'),
 ('("control", "aux")','("absolute", "centered")')])
copy('experiments/p3h_review_evaluation.py','experiments/p3j_review_evaluation.py')
copy('experiments/verify_p3h_batch_order.py','experiments/verify_p3j_batch_order.py',[
 ("if arm=='aux' else 0","if arm in ('absolute','centered') else 0")])
copy('experiments/inspect_p3h.py','experiments/inspect_p3j.py')

copy('experiments/review_p3h.py','experiments/review_p3j.py',[
 ("def validate_batches(waves,phase,weight):","def validate_batches(waves,phase,weight,objective):"),
 ("n=w['samples'];batches=w['batches'];expected=phase['hand_rows'][4*i:4*i+4]","check(w['objective']==objective,'frozen objective')\n        n=w['samples'];batches=w['batches'];expected=phase['hand_rows'][4*i:4*i+4]"),
 ("validate_batches(waves,phase,cfg['auxiliary_weight'])","validate_batches(waves,phase,cfg['auxiliary_weight'],arm)"),
 ("pins={k:v for k,v in pre['source_sha256'].items() if k.startswith('src/') or (k.startswith('experiments/p3j') and k.endswith('.py'))}",
  "pins={k:v for k,v in pre['source_sha256'].items() if k.startswith('src/') or (k.startswith('experiments/p3j') and k.endswith('.py')) or k in ('experiments/p3h_objective.py','experiments/p3f_teacher.py')}"),
 ("if arm=='control':","if arm=='absolute':"),
 ("previous=torch.load(inherited.parent/'final/checkpoint.pt',map_location='cpu',weights_only=True)",
  "previous=torch.load(ROOT/f'artifacts/evaluations/p3h-aux-v1/training/aux-{seed}/final/checkpoint.pt',map_location='cpu',weights_only=True)"),
 ("control=ca['metrics']['win_rate']['estimate'],aux=cb['metrics']['win_rate']['estimate']", "absolute=ca['metrics']['win_rate']['estimate'],centered=cb['metrics']['win_rate']['estimate']"),
 ('validation-aux','validation-centered'),
 ('ACCEPTED_AUXILIARY_LOSS_STUDY_V1','ACCEPTED_CENTERED_AUXILIARY_STUDY_V1'),
 ('control-314380','absolute-314380'),('aux-314380','centered-314380'),
 ("replace('control-','aux-')","replace('absolute-','centered-')"),
 ("validate_batches(bad,phase,.1)","validate_batches(bad,phase,.1,'centered')"),
 ("('changed_weight','auxiliary_weight',.2)","('changed_weight','auxiliary_weight',.2),('wrong_objective','objective','absolute')")
])
print('P3j isolated modules created')
