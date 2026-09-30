"""Overlap controller review of completed immutable jobs with final GPU evaluation."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys
from review_p3f import reviewed_job,specification

def main():
    root=Path(sys.argv[1]).resolve();cache=Path(sys.argv[2]).resolve();jobs=[]
    for kind,folder,names in [('training','training',specification()['training_order']),
        ('evaluation','evaluations',specification()['development_order']+specification()['validation_order'])]:
        for job in names:
            if (root/folder/job/'report.json').exists():jobs.append((kind,job))
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(reviewed_job,root,kind,job,cache):(kind,job) for kind,job in jobs}
        for future in as_completed(pending):
            result=future.result();print('independently reviewed',*pending[future],flush=True)

if __name__=='__main__':main()
