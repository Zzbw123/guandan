"""Read historical plain JSON/JSONL seed fields before freezing P3e validation."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3e_protocol import VALIDATION
from p3e_common import ROOT,write,digest

def scan():
    wanted=set(VALIDATION); hits=[]; files={}; errors=[]
    def walk(value,path,key=''):
        if isinstance(value,dict):
            for k,v in value.items():walk(v,path,k)
        elif isinstance(value,list):
            for v in value:walk(v,path,key)
        elif 'seed' in key.lower() and type(value) is int and value in wanted:
            hits.append(dict(path=path,key=key,value=value))
    base=ROOT/'artifacts/evaluations'
    for path in sorted(base.rglob('*')):
        if path.suffix not in ('.json','.jsonl') or path.name.startswith('p3e'):continue
        name=path.relative_to(ROOT).as_posix()
        try:
            with path.open(encoding='utf-8-sig') as f:
                if path.suffix=='.json':walk(json.load(f),name)
                else:
                    for line in f:
                        if line.strip():walk(json.loads(line),name)
            files[name]=digest(path)
        except (ValueError,UnicodeError) as e:errors.append(dict(path=name,error=str(e)))
    return dict(status='PASS' if not hits and not errors else 'FAIL',validation_seeds=VALIDATION,
                scope='historical plain JSON/JSONL seed-named fields; compressed replays contain no authoritative new seed schedules',
                files=files,hits=hits,errors=errors)
if __name__=='__main__':
    result=scan();write(ROOT/'artifacts/evaluations/p3e-seed-availability.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}));print('files',len(result['files']))
    if result['status']!='PASS':raise SystemExit(1)
