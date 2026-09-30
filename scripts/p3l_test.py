"""Run new P3l contracts and write an exact tested-source receipt."""
from _bootstrap import ROOT
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import unittest
import json
from hashlib import sha256
from pathlib import Path

if __name__=='__main__':
    paths=sorted(list((ROOT/'experiments').glob('p3l*.py'))+list((ROOT/'scripts').glob('p3l*.py'))+list((ROOT/'tests').glob('test_p3l*.py')))
    before={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_p3l*.py')
    path=ROOT/'artifacts/evaluations/p3l-tests.log'
    if path.exists() or (path.parent/'p3l-test-receipt.json').exists():raise SystemExit('fresh test log/receipt required')
    with path.open('w',encoding='utf-8') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    print(path.read_text('utf-8'))
    if not r.wasSuccessful() or r.skipped:raise SystemExit(1)
    paths=sorted(list((ROOT/'experiments').glob('p3l*.py'))+list((ROOT/'scripts').glob('p3l*.py'))+list((ROOT/'tests').glob('test_p3l*.py')))
    after={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    if before != after:raise SystemExit('source changed during tests; no receipt written')
    receipt=dict(status='PASS',total=r.testsRun,skipped=len(r.skipped),log_sha256=sha256(path.read_bytes()).hexdigest(),
        source_sha256=after)
    with (ROOT/'artifacts/evaluations/p3l-test-receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
