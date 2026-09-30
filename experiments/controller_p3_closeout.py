"""Read-only P3 evidence-chain audit, with separately named engineering/strength gates.

Historical scopes reuse signed-off disk receipts and check registered source and
immutable artifact hashes. This does not pretend to replay all prior studies.
P3l receives a fresh full replay/statistical/batch/numerical controller audit.
"""
from pathlib import Path
from hashlib import sha256
import json,sys,re
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.p3e_common import check,read,write,digest

HISTORY=['p3a-cpu-v1','p3b-gpu-v1','p3c-gpu-wave-v1','p3d-validation-v2','p3e-scale-v2',
    'p3f-teacher-v1','p3g-fit-v1','p3h-aux-v1','p3i-centered-v1','p3j-centered-v1','p3k-gradients-v1']

def audit_history():
    base=ROOT/'artifacts/evaluations';records={};artifact_total=0
    for name in ['p2-validation-v1']+HISTORY:
        folder=base/name;pre=read(folder/'preregistration.json')
        sources=pre.get('source_sha256',pre.get('source_sha256_before'))
        check(bool(sources),'historical source registry')
        for path,h in sources.items():check(digest(ROOT/path)==h,'historical source '+path)
        record=dict(registered_sources=len(sources),preregistration_sha256=digest(folder/'preregistration.json'))
        if name in HISTORY:
            receipt=folder/'controller-audit.json';check(read(receipt)['status']=='PASS','historical acceptance '+name)
            record['controller_audit_sha256']=digest(receipt)
        pins={}
        for p in (folder/'delivery-receipt.json',folder/'delivery.json'):
            if not p.exists():continue
            delivery=read(p)
            for path,value in delivery.get('artifact_sha256',delivery.get('files',{})).items():
                # Old copies of mutable STATUS/README are historical snapshots,
                # not a requirement that current project status never advance.
                if not path.startswith('artifacts/'):continue
                expected=value.get('sha256') if isinstance(value,dict) else value
                check(digest(ROOT/path)==expected,'historical immutable artifact '+path)
                pins[path]=expected
            record['delivery_receipt_sha256']=digest(p)
        record['immutable_artifacts']=len(pins);artifact_total+=len(pins);records[name]=record
    return dict(scopes=records,immutable_artifacts_checked=artifact_total,
        method='accepted prior controller receipts, current registered-source hashes and immutable delivery-artifact hashes; no repeated full historical replay')

def main():
    root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
    check(not output.exists(),'fresh closure receipt')
    history=audit_history();audit=read(root/'controller-audit.json')
    for name in ('controller-audit.json','controller-negative.json','batch-order-audit.json','controller-actual-batches.json'):
        check(read(root/name)['status']=='PASS','P3l independent '+name)
    check(len(audit['training'])==6 and sum(v['hands'] for v in audit['training'].values())==3600,'full training budget')
    check(len(audit['evaluations'])==8 and sum(v['games'] for v in audit['evaluations'].values())==4368,'full evaluation budget')
    for s in (314380,314381,314382):check(audit['training'][f'constant-{s}']['old_control_bitwise_equal'],'600-hand control equivalence')
    check(audit['model_promoted'] is False and audit['reserved_test_executed'] is False,'scope remains separate')
    eng=ROOT/'artifacts/evaluations/p3l-engineering-acceptance.json'
    check(read(eng)['status']=='PASS','engineering prerequisite')
    full=ROOT/'artifacts/evaluations/p3l-full-tests-gpu.log';log=full.read_text('utf-8-sig')
    check('Ran 215 tests' in log and log.rstrip().endswith('OK') and 'skipped=' not in log,'full CUDA suite')
    # Re-verify all frozen bytes, not only a success status in the controller file.
    from experiments.review_p3l import verify_freeze
    verify_freeze(root)
    for path,h in read(ROOT/'artifacts/evaluations/p3l-resume-test.json')['artifacts'].items():
        check(digest(ROOT/path)==h,'persistent recovery evidence')
    evidence={p.relative_to(ROOT).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}
    for p in (eng,full,ROOT/'docs/P3_COMPLETION_PLAN.md'):
        evidence[p.relative_to(ROOT).as_posix()]=digest(p)
    result=dict(status='PASS',overall_P3='ACCEPTED_LEARNING_LOOP_V1',
        completion_basis='original project P3 train/save/load/evaluate engineering milestone; P5 owns further playing-strength improvements',
        validation_gate=audit['validation_gate'],model_promoted=False,baseline='greedy-v1',reserved_test_executed=False,
        history=history,p3l_training_hands=3600,p3l_evaluation_games=4368,full_tests=215,
        candidate_sha256=audit['candidate_sha256'],artifact_sha256=evidence,script_sha256=digest(Path(__file__)),
        remaining_outside_P3=['P5 strength/promotion','P4 desktop product','P6 full-match rules',
            'unbounded long-duration stability, cross-device recovery and clean-machine rebuild not established'])
    write(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_sha256','history')}))
if __name__=='__main__':main()
