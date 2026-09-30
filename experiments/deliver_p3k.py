"""Archive and verify scoped P3k delivery, preserving the previous STATUS bytes."""
from pathlib import Path
import sys,json,hashlib,zipfile,re
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.p3e_common import read,write,digest,check
from experiments.p3k_run import OUT,bindings

def history():
    prior=read(ROOT/'artifacts/evaluations/p3j-centered-v1/delivery-receipt.json')
    for name,value in prior['artifact_sha256'].items():
        p=OUT/'previous-status.md' if name=='docs/STATUS.md' else ROOT/name
        check(digest(p)==value,'previous delivery protected '+name)
    return dict(status='PASS',previous_delivery_files=len(prior['artifact_sha256']),
                previous_status_sha256=digest(OUT/'previous-status.md'))

def validate():
    bindings(OUT);old=history()
    for n in ('controller-audit.json','controller-negative.json','postprocess-audit.json'):
        check(read(OUT/n)['status']=='PASS','required acceptance '+n)
    log=(OUT/'full-tests-gpu.log').read_text('utf-8-sig')
    check('Ran 197 tests' in log and re.search(r'\nOK\s*$',log) and 'skipped ' not in log and 'FAILED' not in log,'197 tests without skips')
    check((ROOT/'docs/P3K_ACCEPTANCE.md').read_bytes()==(OUT/'report.md').read_bytes(),'report identity')
    check(read(OUT/'controller-audit.json')['model_promoted'] is False,'no promotion')
    return old

def main():
    old=validate()
    if '--verify' in sys.argv:
        receipt=read(OUT/'delivery-receipt.json')
        for name,value in receipt['artifact_sha256'].items():check(digest(ROOT/name)==value,'delivery SHA '+name)
        check(receipt['status']=='SCOPED_ACCEPTANCE_COMPLETE','delivery status')
        print(json.dumps(dict(status='PASS',files=len(receipt['artifact_sha256']),overall_P3='PARTIALLY_ACCEPTED')));return
    check(digest(ROOT/'docs/STATUS.md')==digest(OUT/'previous-status.md'),'STATUS unchanged since freeze')
    write(OUT/'historical-source-protection.json',old)
    status=(OUT/'previous-status.md').read_text('utf-8');marker='## 下一工作包：P3k 梯度冲突、损失尺度与动作间隔诊断'
    check(status.count(marker)==1,'unique next package marker');status=status.split(marker)[0]
    lines=status.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('当前交付：'):
            lines[i]='当前交付：**P0–P2及P3a–P3k在各自范围内验收通过**。P3k完成234观察×9固定权重的梯度/尺度诊断，未新增训练或评测。最近主候选P3j对greedy新验证26.73% [22.69%, 30.77%]、NOT_ESTABLISHED。P3整体仍为PARTIALLY_ACCEPTED，模型未晋级，贪心主基线保留；下一包见文末。'
        elif line.startswith('| P3 学习闭环 |'):
            lines[i]='| P3 学习闭环 | PARTIALLY_ACCEPTED | P3k诊断工程通过；最近P3j主候选验证NOT_ESTABLISHED，未晋级；下一包见文末。 |'
    (ROOT/'docs/STATUS.md').write_text('\n'.join(lines)+'\n\n'+(OUT/'status-addendum.md').read_text('utf-8'),encoding='utf-8',newline='\n')
    scripts=list((ROOT/'experiments').glob('*p3k*.py'))+list((ROOT/'tests').glob('test_p3k*.py'))
    docs=[ROOT/'docs'/n for n in ('P3K_PROTOCOL.md','P3K_ACCEPTANCE.md','P3L_PROTOCOL.md','STATUS.md')]
    with zipfile.ZipFile(OUT/'controller-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in scripts+docs:z.write(p,p.relative_to(ROOT).as_posix())
    files=[p for p in OUT.rglob('*') if p.is_file()]+scripts+docs
    files += [ROOT/'artifacts/evaluations'/n for n in ('p3k-preflight.log','p3k-run.log','p3k-audit.log')]
    write(OUT/'delivery-receipt.json',dict(status='SCOPED_ACCEPTANCE_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
        engineering='ACCEPTED_FIXED_WEIGHT_GRADIENT_DIAGNOSTIC_V1',overall_P3='PARTIALLY_ACCEPTED',
        model_promoted=False,reserved_test_executed=False,new_training_hands=0,new_evaluation_games=0,
        full_tests_passed=197,artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(files))}))
    print(json.dumps(dict(status='SCOPED_ACCEPTANCE_COMPLETE',files=len(set(files)))))

if __name__=='__main__':main()
