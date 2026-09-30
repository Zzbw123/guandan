"""Repeat full independent P5b audit to a fresh receipt without replacing evidence."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from concurrent.futures import ProcessPoolExecutor,as_completed
from experiments.controller_p5b_audit import freeze,negative_checks,historical,paired
from experiments.controller_p5b_training import audit_training,first_wave_numerics
from experiments.controller_p5b_evaluation import audit_evaluation
from experiments.p5b_protocol import specification,INITS
from experiments.p3e_common import check,write,digest

def main():
    check(len(sys.argv)==3,'run directory and fresh output JSON required')
    root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();check(not output.exists(),'fresh output')
    freeze(root);spec=specification();training={};evaluation={}
    work=[('training',j) for j in spec['training_order']]+[('evaluation',j) for j in spec['development_order']+spec['validation_order']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(audit_training if kind=='training' else audit_evaluation,root,job):(kind,job) for kind,job in work}
        for future in as_completed(futures):
            kind,job=futures[future];(training if kind=='training' else evaluation)[job]=future.result()
            print(kind,job,'PASS',flush=True)
    numerics=[first_wave_numerics(root,j) for j in spec['training_order']]
    development={str(s):paired(evaluation[f'selfplay-{s}']['summary'],evaluation[f'mixed-{s}']['summary']) for s in INITS}
    validation=paired(evaluation['validation-selfplay']['summary'],evaluation['validation-mixed']['summary'])
    negative=negative_checks(root);history=historical();freeze(root)
    write(output,dict(status='PASS',training=training,evaluations=evaluation,numerics=numerics,
        paired_development=development,paired_validation=validation,negative_rejections=negative,historical=history,
        script_sha256=digest(Path(__file__)),model_promoted=False,reserved_test_executed=False))

if __name__=='__main__':main()
