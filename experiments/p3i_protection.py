"""Read-only historical source checks and prospective seed registration scan."""
from pathlib import Path
import json
from hashlib import sha256
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return sha256(p.read_bytes()).hexdigest()

def main():
    base=ROOT/'artifacts/evaluations';out=base/'p3i-centered-v1'
    accepted=['p2-validation-v1','p3a-cpu-v1','p3b-gpu-v1','p3c-gpu-wave-v1',
              'p3d-validation-v2','p3e-scale-v2','p3f-teacher-v1','p3g-fit-v1','p3h-aux-v1']
    protection={}
    for name in accepted:
        pre=json.loads((base/name/'preregistration.json').read_text('utf-8'))
        hashes=pre.get('source_sha256',pre.get('source_sha256_before'))
        assert hashes
        bad=[p for p,h in hashes.items() if digest(ROOT/p)!=h]
        protection[name]=dict(files=len(hashes),mismatches=bad)
        if bad:raise ValueError(f'historical source changed: {name} {bad}')
    # Scan explicit integers in all existing preregistrations, including nested
    # trial lists; this is an inventory check, not proof against unregistered runs.
    registered=list(base.glob('*/preregistration.json'))
    hits=[]
    def walk(obj,path):
        if type(obj) is int and 205000<=obj<=205064:hits.append(dict(path=path,value=obj))
        elif isinstance(obj,dict):
            for k,v in obj.items():walk(v,path+'/'+k)
        elif isinstance(obj,list):
            for i,v in enumerate(obj):walk(v,path+'/'+str(i))
    for p in registered:walk(json.loads(p.read_text('utf-8')),p.relative_to(ROOT).as_posix())
    result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),historical_sources=protection,
                proposed_validation=[205000,205064],explicit_registration_hits=hits,
                scan_scope='existing top-level experiment preregistration JSON; recheck schedules before P3j run',
                registration_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in registered},
                script_sha256=digest(Path(__file__)))
    assert not hits
    with (out/'historical-protection.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
