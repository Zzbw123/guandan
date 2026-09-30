"""Frozen P3i disk-score decomposition; no training and no new game schedule."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'experiments'))
import csv
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import zipfile
from experiments.p3i_objective import decompose

BASE = ROOT/'artifacts/evaluations/p3h-aux-v1'
JOBS = [f'{arm}-{seed}' for seed in (314380,314381,314382) for arm in ('control','aux')]

def digest(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text('utf-8'))
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)

def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def pins(paths): return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}

def run(out):
    out.mkdir(parents=True, exist_ok=False)
    sources = [p for folder in ('src','experiments','scripts','tests')
               for p in (ROOT/folder).rglob('*.py')]
    sources += [ROOT/'docs/P3I_PROTOCOL.md',ROOT/'scripts/run_gpu.ps1']
    inputs = [BASE/f'fidelity/{name}' for name in ('scores.jsonl.gz','report.json','controller-audit.json')]
    inputs += [BASE/'preregistration.json', ROOT/'artifacts/evaluations/p3i-tests-v1/receipt.json',
               ROOT/'artifacts/evaluations/p3i-tests-v1/tests.log',
               ROOT/'artifacts/evaluations/p3f-teacher-v1/training/teacher-314380/phase1/replays.jsonl.gz']
    inputs += [BASE/f'training/{job}/final/{file}' for job in JOBS for file in ('manifest.json','checkpoint.pt')]
    pre = dict(version='gd-p3i-diagnostic-v1', utc=datetime.now(timezone.utc).isoformat(),
               selection='unchanged P3h first 128 multicandidate teacher-314380 states',
               states=128,candidates=4174,jobs=JOBS, aggregation='state equal weight',
               source_sha256=pins(sources),input_sha256=pins(inputs),
               formal_training_hands=0,new_evaluation_games=0,model_promoted=False,reserved_test_executed=False)
    write(out/'preregistration.json', pre)
    with zipfile.ZipFile(out/'source-snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sources: z.write(path,path.relative_to(ROOT).as_posix())
    original = read(BASE/'fidelity/report.json')
    assert original['scores_sha256']==digest(BASE/'fidelity/scores.jsonl.gz')
    data = rows(BASE/'fidelity/scores.jsonl.gz')
    assert len(data)==128 and sum(r['candidate_count'] for r in data)==4174
    output=[]
    for index,row in enumerate(data):
        raw=row['teacher_scores'];lo,hi=min(raw),max(raw)
        targets=[0. if hi==lo else 1.6*((v-lo)/(hi-lo))-.8 for v in raw]
        assert set(row['scores'])==set(JOBS)
        for job in JOBS:
            item=decompose(row['scores'][job], targets)
            assert item['candidates']==row['candidate_count'] and abs(item['identity_residual'])<=1e-12
            output.append(dict(state_index=index,identity=row['identity'],job=job,
                               candidate_sha256=row['candidate_sha256'], **item,
                               top1=row['metrics'][job]['top1'],regret=row['metrics'][job]['regret']))
    with (out/'decomposition.jsonl').open('x',encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row,allow_nan=False)+'\n')
    summary={}
    for job in JOBS:
        selected=[r for r in output if r['job']==job]
        summary[job]={key:sum(r[key] for r in selected)/128
                      for key in ('mse','mean_error','offset_mse','centered_mse','top1','regret')}
        summary[job]['offset_fraction']=summary[job]['offset_mse']/summary[job]['mse']
    paired=[]
    for seed in (314380,314381,314382):
        a,b=[summary[f'{arm}-{seed}'] for arm in ('control','aux')]
        paired.append(dict(seed=seed,**{f'delta_{key}':b[key]-a[key]
                                     for key in ('mse','offset_mse','centered_mse','top1','regret')}))
    with (out/'summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['job']+list(summary[JOBS[0]]))
        writer.writeheader();writer.writerows(dict(job=job,**values) for job,values in summary.items())
    write(out/'report.json',dict(status='COMPLETE_PENDING_AUDIT',states=128,candidates=4174,
          rows=len(output),summary=summary,paired=paired,
          max_identity_residual=max(abs(r['identity_residual']) for r in output),
          formal_training_hands=0,new_evaluation_games=0,model_promoted=False,reserved_test_executed=False,
          preregistration_sha256=digest(out/'preregistration.json'),
          decomposition_sha256=digest(out/'decomposition.jsonl'),csv_sha256=digest(out/'summary.csv')))
    print(json.dumps(dict(output=str(out),summary=summary,paired=paired),ensure_ascii=False))

if __name__=='__main__': run(Path(sys.argv[1]).resolve())
