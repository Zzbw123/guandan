"""GPU receipt negative checks; copies only, no inference or training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_gpu_probe import audit


def run(source):
    original=json.loads((source/'report.json').read_text(encoding='utf-8'))
    cases={
        'missing_candidate':lambda r:r['results']['high_branch']['scores'].pop(),
        'percentile':lambda r:r['results']['high_branch']['latency'].__setitem__('p95_ms',99999),
        'duplicate_hand':lambda r:r['results']['learning']['hands'].__setitem__(1,copy.deepcopy(r['results']['learning']['hands'][0])),
        'cpu_gradient_claim':lambda r:r['results']['learning']['hands'][0]['cuda_proofs'][0].__setitem__('all_gradients_cuda',False),
        'cpu_loss_device':lambda r:r['results']['learning']['hands'][0]['cuda_proofs'][0]['batch_device_proof'].__setitem__('loss_device','cpu'),
        'wrong_weight_hash':lambda r:r['runtime_weights'].__setitem__('sha256','0'*64),
    }
    rejected={}
    with tempfile.TemporaryDirectory(prefix='gd-gpu-negative-') as temp:
        directory=Path(temp)
        for name in ('preregistration.json','source-snapshot.zip','runtime-weights.pt'):
            shutil.copyfile(source/name,directory/name)
        for name,mutation in cases.items():
            changed=copy.deepcopy(original)
            mutation(changed)
            (directory/'report.json').write_text(json.dumps(changed),encoding='utf-8')
            try:
                audit(directory)
            except ValueError as error:
                rejected[name]=str(error)
            else:
                raise AssertionError('Accepted invalid receipt: '+name)
    return dict(status='PASS',rejected=rejected,
                source_report_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
                test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    text=json.dumps(run(args.source),ensure_ascii=False,indent=2)
    with args.output.open('x',encoding='utf-8') as file:file.write(text+'\n')
    print(text)
