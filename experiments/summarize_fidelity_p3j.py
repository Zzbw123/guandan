"""Decompose the predeclared same-state P3j scores and independently recheck arithmetic."""
from pathlib import Path
import sys,gzip,json,math,csv
from statistics import fmean,pvariance
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.p3i_objective import decompose

def main():
    root=Path(sys.argv[1]);path=root/'fidelity';report=json.loads((path/'report.json').read_text('utf-8'))
    audit=json.loads((path/'controller-audit.json').read_text('utf-8'))
    digest=lambda p:sha256(p.read_bytes()).hexdigest()
    assert audit['status']=='PASS' and audit['report_sha256']==digest(path/'report.json')
    assert report['scores_sha256']==digest(path/'scores.jsonl.gz')
    with gzip.open(path/'scores.jsonl.gz','rt',encoding='utf-8') as f:source=[json.loads(s) for s in f]
    assert len(source)==128 and sum(r['candidate_count'] for r in source)==4174
    rows=[]
    for i,row in enumerate(source):
        raw=row['teacher_scores'];lo,hi=min(raw),max(raw)
        target=[0. if hi==lo else 1.6*(v-lo)/(hi-lo)-.8 for v in raw]
        for job,q in row['scores'].items():
            calc=decompose(q,target);error=[x-y for x,y in zip(q,target)]
            assert math.isclose(calc['centered_mse'],pvariance(error),abs_tol=1e-12)
            assert math.isclose(calc['offset_mse'],(fmean(q)-fmean(target))**2,abs_tol=1e-12)
            assert abs(calc['identity_residual'])<=1e-12
            assert math.isclose(calc['mse'],row['metrics'][job]['mse'],abs_tol=1e-12)
            rows.append(dict(state_index=i,job=job,**row['identity'],**calc,
                             top1=row['metrics'][job]['top1'],regret=row['metrics'][job]['regret']))
    summary={job:{k:fmean(r[k] for r in rows if r['job']==job) for k in
                 ('mse','mean_error','offset_mse','centered_mse','top1','regret')}
             for job in report['summary']}
    for job in summary:
        for k in ('mse','top1','regret'):assert math.isclose(summary[job][k],audit['summary'][job][k],abs_tol=1e-12)
    with (path/'decomposition.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(status='PASS',states=128,candidates=4174,rows=len(rows),summary=summary,
        max_identity_residual=max(abs(r['identity_residual']) for r in rows),
        input_sha256=digest(path/'scores.jsonl.gz'),script_sha256=digest(Path(__file__)),
        csv_sha256=digest(path/'decomposition.csv'))
    with (path/'decomposition-audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
