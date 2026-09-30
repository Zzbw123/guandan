"""Bounded current-run disk reviewer; one CPU review at a time, never trains."""
from pathlib import Path
import sys
import time
from review_p3h import specification,reviewed,check

if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();cache=Path(sys.argv[2]).resolve();cache.mkdir(parents=True,exist_ok=True)
    spec=specification();pending=[('training',j) for j in spec['training_order']]+[('evaluation',j) for j in spec['development_order']+spec['validation_order']]
    deadline=time.monotonic()+14400
    while pending:
        check(not (root/'failure.json').exists(),'experiment stopped with failure; retain completed reviews')
        check(time.monotonic()<deadline,'bounded review watcher deadline')
        progressed=False
        for kind,job in pending[:]:
            folder=root/('training' if kind=='training' else 'evaluations')/job
            if (folder/'report.json').exists():
                reviewed(root,kind,job,cache);pending.remove((kind,job));progressed=True
                print('independently reviewed',kind,job,'remaining',len(pending),flush=True)
        if not progressed:time.sleep(15)
    print('all fixed jobs independently reviewed',flush=True)
