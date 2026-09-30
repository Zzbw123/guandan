"""Full GPU regression, with explicit opt-ins and immutable test receipt."""
from _bootstrap import ROOT
import sys,os,json,unittest
from hashlib import sha256
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
os.environ['RUN_P3F_CUDA']='1';os.environ['RUN_P3G_CUDA']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

if __name__=='__main__':
    paths=sorted(p for folder in ('src','experiments','scripts','tests') for p in (ROOT/folder).rglob('*.py'))
    hashes={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    target=ROOT/'artifacts/evaluations/p5c-full-tests-gpu-v1.log'
    with target.open('x',encoding='utf-8') as f:
        r=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    print(target.read_text('utf-8'))
    assert r.wasSuccessful() and not r.skipped
    assert hashes=={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    receipt=dict(status='PASS',tests=r.testsRun,skipped=len(r.skipped),sources=hashes,log_sha256=sha256(target.read_bytes()).hexdigest())
    with target.with_suffix('.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
