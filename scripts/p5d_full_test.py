"""Complete GPU regression with a new immutable output path."""
from _bootstrap import ROOT
import sys,os,json,unittest,argparse
from pathlib import Path
from hashlib import sha256
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
os.environ['RUN_P3F_CUDA']='1';os.environ['RUN_P3G_CUDA']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    paths=sorted(p for folder in ('src','experiments','scripts','tests') for p in (ROOT/folder).rglob('*.py'))
    hashes={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    with a.output.open('x',encoding='utf-8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    print(a.output.read_text('utf-8'))
    assert result.wasSuccessful() and not result.skipped
    assert hashes=={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    receipt=dict(status='PASS',tests=result.testsRun,skipped=len(result.skipped),sources=hashes,
                 log_sha256=sha256(a.output.read_bytes()).hexdigest())
    with a.output.with_suffix('.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
