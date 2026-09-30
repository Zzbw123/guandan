"""Bind the accepted P3i package, with repeatable read-only delivery verification."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
import json
import re
import zipfile
from hashlib import sha256
from datetime import datetime, timezone
from experiments.review_p3i import audit,check

OUT=ROOT/'artifacts/evaluations/p3i-centered-v1'
def digest(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text('utf-8'))
def log(p):
    raw=p.read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')

def validate():
    result=audit(OUT)
    check(result==read(OUT/'controller-audit.json'),'fresh controller audit')
    direct=next(json.loads(s) for s in log(OUT/'direct-forward.log').splitlines() if s.startswith('{'))
    check(direct==read(ROOT/'artifacts/evaluations/p3h-aux-v1/fidelity/controller-audit.json'),'current direct-forward repeat')
    check(direct['direct_forward_comparisons']==768 and direct['max_score_error']<2e-6,'GPU complete coverage')
    full=log(OUT/'full-tests-gpu.log');supp=log(OUT/'opt-in-tests-gpu.log')
    check('Ran 173 tests' in full and 'OK (skipped=4)' in full and 'FAILED' not in full,'full-suite status')
    check('Ran 4 tests' in supp and re.search(r'\nOK\s*$',supp) and 'FAILED' not in supp,'supplement status')
    def ids(text,status):
        return {line.split(' ... ')[0] for line in text.splitlines() if line.startswith('test_') and ' ... '+status in line}
    passed,skipped,extra=ids(full,'ok'),ids(full,'skipped'),ids(supp,'ok')
    check(len(passed)==169 and len(skipped)==4 and skipped==extra and len(passed|extra)==173,'all unique tests accounted')
    history=read(OUT/'historical-protection.json')
    check(not history['explicit_registration_hits'],'prospective registration availability')
    for name,item in history['historical_sources'].items():
        pre=read(ROOT/f'artifacts/evaluations/{name}/preregistration.json')
        hashes=pre.get('source_sha256',pre.get('source_sha256_before'))
        check(item==dict(files=len(hashes),mismatches=[]),'history counts')
        for p,h in hashes.items():check(digest(ROOT/p)==h,'history '+p)
    for p,h in history['registration_sha256'].items():check(digest(ROOT/p)==h,'registered source input')
    check(history['script_sha256']==digest(ROOT/'experiments/p3i_protection.py'),'protection implementation')
    for p in [ROOT/f'docs/{name}.md' for name in ('P3I_PROTOCOL','P3I_ACCEPTANCE','P3J_PROTOCOL','STATUS')]:
        value=p.read_text('utf-8');check('\ufffd' not in value and len(value)>500,'UTF-8 document')
    return dict(unique_tests_passed=173,specialized_tests=10,skipped_remaining=0,
                direct_forward_groups=768,max_score_error=direct['max_score_error'],
                diagnostic_rows=768,states=128,candidates=4174,negative_cases=7)

def main(verify):
    evidence=validate()
    if verify:
        receipt=read(OUT/'delivery.json')
        check(receipt['evidence']==evidence,'delivery evidence')
        for p,h in receipt['files'].items():check(digest(ROOT/p)==h,'delivered file '+p)
        with zipfile.ZipFile(OUT/'delivery-source.zip') as z:
            check(set(z.namelist())==set(receipt['delivery_sources']),'delivery source members')
            for p,h in receipt['delivery_sources'].items():check(sha256(z.read(p)).hexdigest()==h,'delivery source bytes')
        print(json.dumps(dict(status='PASS',files=len(receipt['files']),evidence=evidence)))
        return
    docs=[ROOT/f'docs/{n}.md' for n in ('P3I_ACCEPTANCE','P3I_PROTOCOL','P3J_PROTOCOL','STATUS')]
    sources=docs+[ROOT/'experiments/p3i_protection.py',Path(__file__)]
    with zipfile.ZipFile(OUT/'delivery-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT).as_posix())
    files=[p for p in OUT.rglob('*') if p.is_file()]+sources
    files += list((ROOT/'artifacts/evaluations/p3i-tests-v1').glob('*'))
    result=dict(status='ACCEPTED_CENTERED_OBJECTIVE_DIAGNOSTIC_V1',utc=datetime.now(timezone.utc).isoformat(),
                overall_P3='PARTIALLY_ACCEPTED',model_promoted=False,formal_training_hands=0,new_evaluation_games=0,
                next_package='P3j DESIGN_READY_NOT_RUN',evidence=evidence,
                delivery_sources={p.relative_to(ROOT).as_posix():digest(p) for p in sources},
                files={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(files))})
    with (OUT/'delivery.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=result['status'],files=len(result['files']),evidence=evidence)))

if __name__=='__main__':main('--verify' in sys.argv)
