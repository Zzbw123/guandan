"""Run new P3h contracts and write an exact tested-source receipt."""
from _bootstrap import ROOT
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import unittest
import json
from hashlib import sha256
from pathlib import Path

if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_p3h*.py')
    path=ROOT/'artifacts/evaluations/p3h-tests.log'
    with path.open('w',encoding='utf-8') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    print(path.read_text('utf-8'))
    if not r.wasSuccessful() or r.skipped:raise SystemExit(1)
    paths=list((ROOT/'experiments').glob('p3h*.py'))+list((ROOT/'scripts').glob('p3h*.py'))+list((ROOT/'tests').glob('test_p3h*.py'))
    receipt=dict(status='PASS',total=r.testsRun,skipped=len(r.skipped),log_sha256=sha256(path.read_bytes()).hexdigest(),
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths})
    with (ROOT/'artifacts/evaluations/p3h-test-receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
