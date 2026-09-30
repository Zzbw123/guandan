"""Bounded current-turn post-run checks; never trains or changes the frozen design."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=Path(sys.argv[1]).resolve();deadline=time.monotonic()+14400
    def waiting():
        if (out/'failure.json').exists():raise RuntimeError('formal run failed')
        if time.monotonic()>deadline:raise TimeoutError('postrun deadline')
        time.sleep(10)
    while not (out/'receipt.json').exists():waiting()
    env=os.environ.copy();env.update(RUN_P3F_CUDA='1',RUN_P3G_CUDA='1',PYTHONPATH=os.pathsep.join(map(str,(ROOT,ROOT/'src',ROOT/'experiments'))))
    def run(script,args,logname,timeout=1800):
        with (out/logname).open('x',encoding='utf-8') as log:
            r=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,
                             env=env,timeout=timeout,cwd=ROOT)
        if r.returncode:raise RuntimeError(f'{script}: {r.returncode}; {logname}')
        print(json.dumps(dict(script=script,status='PASS')),flush=True)
    run('experiments/inspect_p3j.py',[out],'fidelity-run.log')
    run('scripts/validate.py',[],'full-tests-gpu.log')
    run('experiments/summarize_fidelity_p3j.py',[out],'fidelity-decomposition.log')
    watch=ROOT/'artifacts/evaluations/p3j-review-watch.log'
    while 'all fixed jobs independently reviewed' not in watch.read_text('utf-8'):
        if 'Traceback (most recent call last)' in watch.read_text('utf-8'):raise RuntimeError('independent watcher failed')
        waiting()
    run('experiments/review_p3j.py',[out,'--cache',ROOT/'artifacts/evaluations/p3j-controller-cache','--write'],'final-controller.log')
    run('experiments/verify_p3j_batch_order.py',[out],'batch-order-final.log')
    run('experiments/figures_p3j.py',[out],'figures.log')
    run('experiments/report_p3j.py',[out],'report-build.log')
    print('postrun complete; visual QA, scientific interpretation, and delivery remain',flush=True)

if __name__=='__main__':main()
