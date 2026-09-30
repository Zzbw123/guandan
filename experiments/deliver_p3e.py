"""Record non-experimental delivery provenance after the fixed run and audit."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]

def digest(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def main():
    root=Path(sys.argv[1]).resolve();audit=read(root/'controller-audit.json');negative=read(root/'controller-negative.json')
    assert audit['status']==negative['status']=='PASS'
    first=ROOT/'artifacts/evaluations/p3e-scale-v1'
    pre1=read(first/'preregistration.json');pre2=read(root/'preregistration.json')
    assert pre1['specification']==pre2['specification'] and pre1['source_sha256']==pre2['source_sha256']
    assert not (first/'training').exists()
    assert not list(first.rglob('results.jsonl'))
    write(root/'restart-receipt.json',dict(status='PASS',first_run='p3e-scale-v1',
        reason='py -3.14 launcher selected an interpreter without torch; failed before policy startup',
        failed_before_any_training_or_evaluation=True,same_specification=True,same_source_hashes=True,
        runtime_path_source='artifacts/evaluations/p3b-gpu-environment.json',
        runtime_interpreter=sys.executable,torch_installation_changed=False,
        failed_log_sha256=digest(first/'evaluate-old.log')))
    write(root/'data-usage.json',dict(development_training_unique_deals=1600,initializers=3,training_hands=4800,
        development_diagnostic_deals=list(range(108000,108026)),
        previously_viewed_validation=[list(range(200000,200130)),list(range(201000,201065))],
        newly_viewed_validation=list(range(202000,202065)),reserved_test_executed=False,
        model_promoted=False,main_baseline='greedy'))
    extras=[ROOT/'docs/P3E_ACCEPTANCE.md',ROOT/'docs/STATUS.md',ROOT/'README.md',ROOT/'scripts/run_gpu.ps1',
        ROOT/'experiments/review_p3e.py',ROOT/'experiments/report_p3e.py',ROOT/'experiments/research_p3e_figures.py',
        ROOT/'experiments/research_figure_export.py',Path(__file__),
        ROOT/'artifacts/evaluations/p3e-unittest-gpu.log',ROOT/'artifacts/evaluations/p3e-unittest-cpu.log']
    files=[p for p in root.rglob('*') if p.is_file()]+extras
    write(root/'delivery-receipt.json',dict(status='SCOPED_ACCEPTANCE_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
        engineering=audit['engineering'],validation_gate=audit['validation_gate'],model_promoted=False,
        overall_P3='PARTIALLY_ACCEPTED',artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files}))
    print('delivery receipt written')
if __name__=='__main__':main()
