"""Final scoped receipt after controller audit and visual inspection."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]

def digest(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    root=Path(sys.argv[1]).resolve();audit=read(root/'controller-audit.json');negative=read(root/'controller-negative.json');qa=read(root/'visual-qa.json')
    assert audit['status']==negative['status']==qa['status']=='PASS'
    assert audit['model_promoted'] is audit['reserved_test_executed'] is False
    assert (root/'report.md').read_bytes()==(ROOT/'docs/P3F_ACCEPTANCE.md').read_bytes()
    trace=read(root/'descriptives/trace.json');assert trace['group_rows']==546
    for name,value in trace['exports'].items():assert digest(root/name)==value
    for name,value in read(root/'receipt.json')['artifact_sha256'].items():assert digest(root/name)==value
    for name,value in read(root/'preregistration.json')['source_sha256'].items():assert digest(ROOT/name)==value
    assert audit['review_sha256']==negative['review_sha256']==digest(ROOT/'experiments/review_p3f.py')
    revision=read(root/'audit-revision.json')
    assert revision['new_review_sha256']==audit['review_sha256']
    assert revision['old_review_sha256']==digest(ROOT/'artifacts/evaluations/p3f-controller-work-v1/review_p3f-v1.py')
    resolved=dict(status='PASS',revision_sha256=digest(root/'audit-revision.json'),
        full_review_sha256=digest(root/'controller-audit.json'),negative_review_sha256=digest(root/'controller-negative.json'),
        full_training_hands=4800,full_evaluation_games=4368,experiment_rerun=False,
        note='All raw replays re-reviewed after correcting the guard teardown contract; old reviewer and failure retained.')
    with (root/'audit-revision-acceptance.json').open('x',encoding='utf-8') as f:json.dump(resolved,f,indent=2)
    paths=[p for p in root.rglob('*') if p.is_file()]
    for folder in ['p3f-score-diagnostic-v1','p3f-target-trace-v1','p3f-controller-work-v1']:
        paths.extend(p for p in (ROOT/'artifacts/evaluations'/folder).rglob('*') if p.is_file())
    if audit.get('controller_work_directory'):
        paths.extend(p for p in Path(audit['controller_work_directory']).rglob('*') if p.is_file())
    paths.extend(ROOT/name for name in read(root/'preregistration.json')['source_sha256'])
    paths.extend(ROOT/f'artifacts/evaluations/{name}' for name in [
        'p3f-diagnostic-controller.json','p3f-seed-availability.json','p3f-test-receipt.json','p3f-tests-gpu.log','p3f-tests-cpu.log',
        'p3f-controller-progress.log','p3f-controller-final.log'])
    paths.extend(ROOT/name for name in ['docs/P3F_ACCEPTANCE.md','docs/STATUS.md','README.md',
        'experiments/review_p3f.py','experiments/review_p3f_completed.py','experiments/review_p3f_diagnostics.py','experiments/report_p3f.py',
        'experiments/research_p3f_figures.py','experiments/research_figure_export.py','experiments/deliver_p3f.py',
        'experiments/update_p3f_status.py'])
    receipt=dict(status='SCOPED_ACCEPTANCE_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
        engineering=audit['engineering'],validation_gate=audit['validation_gate'],model_promoted=False,
        reserved_test_executed=False,overall_P3='PARTIALLY_ACCEPTED',
        artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))})
    with (root/'delivery-receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    for name,value in receipt['artifact_sha256'].items():assert digest(ROOT/name)==value
    print(json.dumps(dict(status=receipt['status'],files=len(receipt['artifact_sha256'])),ensure_ascii=False))

if __name__=='__main__':main()
