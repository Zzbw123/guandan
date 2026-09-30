"""Read historical seed fields only; never open sealed test payloads for parsing."""
from pathlib import Path
from hashlib import sha256
import gzip,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.p3l_protocol import VALIDATION


def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def seed_files(root=ROOT):
    return sorted(p for p in (root/'artifacts').rglob('*') if p.is_file()
        and p.name.endswith(('.json','.jsonl','.json.gz','.jsonl.gz'))
        and not any(part.startswith('p3l-') for part in p.relative_to(root).parts))


def scan(root=ROOT):
    wanted=set(VALIDATION);files={};hits=[];errors=[];sealed=[]
    def walk(value,path,key=''):
        if isinstance(value,dict):
            # A sealed-test configuration may declare the reserved seed range;
            # only seed metadata is inspected, unrelated payloads are ignored.
            for k,x in value.items():
                if isinstance(x,(dict,list)) or 'seed' in k.lower():walk(x,path,k)
        elif isinstance(value,list):
            for x in value:walk(x,path,key)
        elif 'seed' in key.lower() and type(value) is int and value in wanted:
            hits.append(dict(path=path,key=key,value=value))
    for p in seed_files(root):
        name=p.relative_to(root).as_posix()
        try:
            files[name]=digest(p)
            if any(token in name.lower() for token in ('sealed','reserved-test','reserved_test','test-games','test_games')):
                sealed.append(name);continue
            with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig') as f:
                if name.endswith(('.json','.json.gz')):walk(json.load(f),name)
                else:
                    for line in f:
                        if line.strip():walk(json.loads(line),name)
        except (OSError,ValueError,UnicodeError,EOFError) as e:
            errors.append(dict(path=name,error=str(e)))
    return dict(status='PASS' if not hits and not errors else 'FAIL',validation_seeds=VALIDATION,
        scope='all historical artifact JSON/JSONL/gzip seed fields; prospective p3l paths excluded; sealed payloads hash only',
        files=files,hash_only_sealed=sealed,hits=hits,errors=errors)

if __name__=='__main__':
    result=scan()
    out=ROOT/'artifacts/evaluations/p3l-seed-availability.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','hits')}));print('files',len(result['files']),'hits',len(result['hits']))
    raise SystemExit(0 if result['status']=='PASS' else 1)
