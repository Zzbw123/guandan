"""Pre-run scan of actual artifact seed fields, not protocol proposals."""
from pathlib import Path
import gzip,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.p3j_protocol import VALIDATION
from experiments.p3e_common import digest,write

def scan():
    wanted=set(VALIDATION);files={};hits=[];errors=[]
    def walk(v,path,key=''):
        if isinstance(v,dict):
            for k,x in v.items():walk(x,path,k)
        elif isinstance(v,list):
            for x in v:walk(x,path,key)
        elif 'seed' in key.lower() and type(v) is int and v in wanted:hits.append(dict(path=path,key=key,value=v))
    for p in sorted((ROOT/'artifacts/evaluations').rglob('*')):
        if not p.is_file() or 'p3j-' in p.as_posix():continue
        name=p.relative_to(ROOT).as_posix()
        if not name.endswith(('.json','.jsonl','.jsonl.gz')):continue
        try:
            with (gzip.open if name.endswith('.gz') else open)(p,'rt',encoding='utf-8-sig') as f:
                if name.endswith('.json'):walk(json.load(f),name)
                else:
                    for line in f:
                        if line.strip():walk(json.loads(line),name)
            files[name]=digest(p)
        except (ValueError,UnicodeError) as e:errors.append(dict(path=name,error=str(e)))
    return dict(status='PASS' if not hits and not errors else 'FAIL',validation_seeds=VALIDATION,
        scope='historical JSON/JSONL/gzip JSONL seed fields; prospective P3j artifacts excluded',
        files=files,hits=hits,errors=errors)
if __name__=='__main__':
    r=scan();write(ROOT/'artifacts/evaluations/p3j-seed-availability.json',r)
    print(json.dumps({k:v for k,v in r.items() if k!='files'}));print('files',len(r['files']))
    raise SystemExit(0 if r['status']=='PASS' else 1)
