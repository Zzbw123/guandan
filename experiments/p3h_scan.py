"""Seed-named-field audit of local history before P3h training."""
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3h_protocol import VALIDATION
from p3e_common import write,digest

def scan():
    wanted=set(VALIDATION);hits=[];files={};errors=[]
    def walk(value,path,key=''):
        if isinstance(value,dict):
            for k,v in value.items():walk(v,path,k)
        elif isinstance(value,list):
            for v in value:walk(v,path,key)
        elif 'seed' in key.lower() and type(value) is int and value in wanted:
            hits.append(dict(path=path,key=key,value=value))
    for path in sorted((ROOT/'artifacts/evaluations').rglob('*')):
        if not path.is_file() or path.name=='p3h-seed-availability.json':continue
        name=path.relative_to(ROOT).as_posix()
        if not (name.endswith('.json') or name.endswith('.jsonl') or name.endswith('.jsonl.gz')):continue
        try:
            opener=gzip.open if name.endswith('.gz') else open
            with opener(path,'rt',encoding='utf-8-sig') as f:
                if name.endswith('.json'):walk(json.load(f),name)
                else:
                    for line in f:
                        if line.strip():walk(json.loads(line),name)
            files[name]=digest(path)
        except (ValueError,UnicodeError) as e:errors.append(dict(path=name,error=str(e)))
    return dict(status='PASS' if not hits and not errors else 'FAIL',validation_seeds=VALIDATION,
        scope='local artifacts/evaluations JSON, JSONL and gzip JSONL seed-named fields',
        files=files,hits=hits,errors=errors)

if __name__=='__main__':
    result=scan();write(ROOT/'artifacts/evaluations/p3h-seed-availability.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}));print('files',len(result['files']))
    if result['status']!='PASS':raise SystemExit(1)
