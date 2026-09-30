"""Run P3i CUDA contracts, preserving logs even when tests fail."""
from _bootstrap import ROOT
import sys
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
import json
from hashlib import sha256
from pathlib import Path
import unittest

if __name__ == '__main__':
    folder = Path(sys.argv[1]); folder.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_p3i*.py')
    with (folder/'tests.log').open('x', encoding='utf-8') as f:
        result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
    print((folder/'tests.log').read_text('utf-8'))
    sources = [ROOT/'experiments/p3i_objective.py', ROOT/'tests/test_p3i_objective.py', Path(__file__)]
    receipt = dict(status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
                   total=result.testsRun, skipped=len(result.skipped),
                   log_sha256=sha256((folder/'tests.log').read_bytes()).hexdigest(),
                   sources={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in sources})
    (folder/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    raise SystemExit(0 if receipt['status']=='PASS' else 1)
