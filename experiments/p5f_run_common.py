"""P5f immutable source registration and run bindings."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile
from datetime import datetime,timezone
from experiments.p5f_protocol import specification
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,allow_nan=False,ensure_ascii=False,indent=2);f.write('\n')


def check(value,message):
    if not value:raise ValueError(message)


def historical():
    path=ROOT/'artifacts/evaluations/p5e-balanced-v1/delivery-receipt.json'
    receipt=read(path)
    check(receipt['status']=='PASS' and receipt['full_gpu_tests']==274,'P5e delivery prerequisite')
    items={k:v for k,v in receipt['artifact_sha256'].items() if k!='docs/STATUS.md'}
    # Registered historical sources include every former experiment dependency.
    old=read(path.parent/'preregistration.json')
    items.update(old['source_sha256'])
    items.update(old['input_sha256'])
    items.pop('docs/STATUS.md',None)
    for name,h in items.items():check(digest(ROOT/name)==h,'historical drift '+name)
    return items


def freeze(root,phase,inputs=()):
    root=Path(root).resolve()
    check(not root.exists() and root.is_relative_to(ROOT/'artifacts/evaluations') and root.name.startswith('p5f-'),
          'fresh isolated P5f output required')
    hist=historical()
    paths=list((ROOT/'src').rglob('*.py'))
    paths+=list((ROOT/'experiments').rglob('*.py'))
    paths+=list((ROOT/'scripts').glob('*.py'))
    paths+=list((ROOT/'tests').glob('test_p5f*.py'))
    paths += [ROOT/'docs/P5F_PROTOCOL.md',ROOT/'scripts/run_gpu.ps1']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))}
    input_hashes={Path(p).resolve().relative_to(ROOT).as_posix():digest(p) for p in inputs}
    spec=specification();check(json.loads(json.dumps(spec))==spec,'spec roundtrip')
    root.mkdir(parents=True)
    write(root/'preregistration.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),
        specification=spec,source_sha256=hashes,input_sha256=input_hashes,historical_sha256=hist))
    with zipfile.ZipFile(root/'source-snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(ROOT/name,name)
    return verify_frozen(root)


def verify_frozen(root):
    pre=read(Path(root)/'preregistration.json')
    check(pre['specification']==specification(),'P5f specification mismatch')
    for group in ('source_sha256','input_sha256','historical_sha256'):
        check(bool(pre[group]) or group=='input_sha256','empty source/history')
        for name,h in pre[group].items():check(digest(ROOT/name)==h,'frozen drift '+name)
    return digest(Path(root)/'preregistration.json')
