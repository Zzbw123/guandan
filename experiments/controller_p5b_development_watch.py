"""Audit completed development jobs during the ongoing local run; never change jobs."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.controller_p5b_audit import audited_job
from experiments.p5b_protocol import specification

def main():
    root=Path(sys.argv[1]).resolve();deadline=time.monotonic()+7200
    spec=specification();jobs=sys.argv[2:] or spec['development_order']
    if any(j not in spec['development_order']+spec['validation_order'] for j in jobs):
        raise ValueError('audit job outside frozen schedule')
    for job in jobs:
        report=root/'evaluations'/job/'report.json'
        while True:
            if time.monotonic()>deadline:raise TimeoutError('development audit wait budget')
            if (root/'failure.json').exists():raise RuntimeError('formal run failed; stopping audit watcher')
            try:
                value=json.loads(report.read_text('utf-8'))
                if value['status']=='PASS':break
            except (FileNotFoundError,json.JSONDecodeError):pass
            time.sleep(5)
        result=audited_job(root,'evaluations',job)
        print(json.dumps(dict(job=job,games=result['games'],status='PASS')),flush=True)

if __name__=='__main__':main()
