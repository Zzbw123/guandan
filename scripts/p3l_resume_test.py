"""Separate-process engineering continuation; --verify keeps durable recovery evidence."""
from _bootstrap import ROOT
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.p3l_training import GPUTrainer
from experiments.p3l_checkpoint import load_checkpoint,save_checkpoint,capture,sources


def equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def verify(directory):
    import torch
    from hashlib import sha256
    out=Path(directory).resolve()
    receipt=out.with_suffix('.json')
    if out.exists() or receipt.exists():raise FileExistsError('fresh recovery output/receipt required')
    out.mkdir(parents=True)
    cfg=dict(seed=314381,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4,
             auxiliary_weight=.1,objective='normcap')
    t=GPUTrainer(cfg)
    first=t.train_wave([(109110+i,2+i,i) for i in range(4)])
    initial=save_checkpoint(t,out/'before')
    expected=t.train_wave([(109120+i,6+i,i) for i in range(4)])
    final=save_checkpoint(t,out/'expected')
    (out/'expected-wave.json').write_text(json.dumps(expected,indent=2,allow_nan=False),encoding='utf-8')
    (out/'first-wave.json').write_text(json.dumps(first,indent=2,allow_nan=False),encoding='utf-8')
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),str(out/'before'),
                           initial['sha256'],str(out/'child')],capture_output=True,text=True,timeout=240)
    (out/'child-process.json').write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,
                            stderr=result.stderr,interpreter=sys.executable),indent=2),encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    child=load_checkpoint(out/'child')
    expected_payload=torch.load(out/'expected/checkpoint.pt',map_location='cpu',weights_only=True)
    actual_payload=capture(child)
    keys=('model','optimizer','policy_rng','torch_rng','cuda_rng','episodes','waves','updates','used_deal_seeds')
    checks={k:equal(expected_payload[k],actual_payload[k]) for k in keys}
    checks['wave_record']=expected==json.loads((out/'wave.json').read_text('utf-8'))
    if not all(checks.values()):raise RuntimeError(f'recovery mismatch: {checks}')
    value=dict(status='PASS',scope='normcap engineering two-wave cross-process recovery only',
               interpreter=sys.executable,checks=checks,initial_checkpoint_sha256=initial['sha256'],
               expected_checkpoint_sha256=final['sha256'],source_sha256=sources(),
               child_checkpoint_sha256=sha256((out/'child/checkpoint.pt').read_bytes()).hexdigest(),
               evidence={p.relative_to(out).as_posix():sha256(p.read_bytes()).hexdigest()
                         for p in out.rglob('*') if p.is_file()})
    receipt.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(receipt=str(receipt),checks=checks,interpreter=sys.executable),indent=2))


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--verify':
        verify(sys.argv[2])
    else:
        p=Path(sys.argv[1]);GPUTrainer(json.loads((p/'manifest.json').read_text('utf-8'))['config'])
        t=load_checkpoint(p,sys.argv[2]);row=t.train_wave([(109120+i,6+i,i) for i in range(4)])
        out=Path(sys.argv[3]);save_checkpoint(t,out)
        (out.parent/'wave.json').write_text(json.dumps(row),encoding='utf-8')
