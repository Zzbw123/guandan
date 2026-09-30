from _bootstrap import ROOT
import sys
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'scripts')]
from pathlib import Path
from experiments.p5f_run_common import verify_frozen,check
from experiments.p5f_protocol import specification

if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();stage=sys.argv[2];job=sys.argv[3]
    verify_frozen(root)
    check(job in specification()['training_order'],'fixed job required')
    if stage=='train':
        from experiments.p5f_job import train
        train(root,job)
    elif stage=='fit':
        from experiments.p5f_job import fit
        fit(root,job)
    elif stage=='evaluate':
        from experiments.p5f_evaluation import evaluate
        evaluate(root,job)
    else:raise ValueError('unknown stage')
