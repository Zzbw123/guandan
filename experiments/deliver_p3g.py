"""Finalize scoped diagnostic delivery after controller audit and visual QA."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]

def digest(path):return sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    out=Path(sys.argv[1]).resolve();audit=json.loads((out/'controller-audit.json').read_text('utf-8'));assert audit['status']=='PASS'
    previous=json.loads((ROOT/'artifacts/evaluations/p3f-teacher-v1/delivery-receipt.json').read_text('utf-8'))
    for name,label in [('docs/STATUS.md','previous-status.md'),('README.md','previous-readme.md')]:
        path=ROOT/name;assert digest(path)==previous['artifact_sha256'][name]
        with (out/label).open('xb') as f:f.write(path.read_bytes())
    # Remove the obsolete generated rerun recipe; disk audit below is the supported evidence replay.
    for path in (out/'report.md',ROOT/'docs/P3G_ACCEPTANCE.md'):
        text=path.read_text('utf-8').replace('# 固定开发诊断重新执行；必须新目录，重跑不是新的独立验证\n./scripts/run_gpu.ps1 scripts/p3g_run.py artifacts/evaluations/my-p3g-repeat\n','')
        path.write_text(text,encoding='utf-8',newline='\n')
    status=ROOT/'docs/STATUS.md';text=status.read_text('utf-8');lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('当前交付：'):
            lines[i]='当前交付：**P0核心规格、P1规则环境、P2固定基线评测、P3a–P3f训练/验证工程及P3g教师拟合诊断在各自范围内验收通过**。P3g新增624局开发评测，7508状态/788421完整候选的六模型评分与独立复算通过。DMC后教师排序一致率下降，但配对开发胜率变化有正有负、区间均含0，未证明棋力退化。P3整体仍为PARTIALLY_ACCEPTED，模型未晋级；下一包P3h检验教师偏好辅助损失，桌面端后置。'
        elif line.startswith('| P3 学习闭环 |'):
            lines[i]='| P3 学习闭环 | PARTIALLY_ACCEPTED | P3g教师拟合与DMC变化诊断工程通过；下一包为教师偏好辅助损失的单条件配对试验。P3f固定教师候选独立验证仍为31.35% [27.88%,34.81%]、NOT_ESTABLISHED；本轮无新验证或模型晋级。 |'
    text='\n'.join(lines)+'\n';marker='## 下一工作包：P3g 教师拟合质量与DMC阶段变化诊断';assert text.count(marker)==1
    text=text.split(marker)[0]
    text+='''## P3g 教师拟合质量与DMC变化诊断（2026-09-26）

**ACCEPTED_TEACHER_FIT_DIAGNOSTIC_V1**。未新训练、未用新验证或保留测试。完整协议和验收为`docs/P3G_PROTOCOL.md`、`docs/P3G_ACCEPTANCE.md`，证据目录`artifacts/evaluations/p3g-fit-v1/`。

新增三个教师阶段模型各208局开发评测，共624局；对应最终DMC模型复用P3f624局。同26个原始发牌、8变体/组，差值为最终DMC−教师阶段：

| 初始化 | 教师阶段胜率 | 最终DMC胜率 | 差值/百分点 | 95%描述区间/百分点 |
|---|---:|---:|---:|---:|
'''
    for seed,v in audit['paired'].items():
        text+=f"| {seed} | {100*v['phase1']:.2f}% | {100*v['final']:.2f}% | {100*v['delta']:+.2f} | [{100*v['ci95'][0]:+.2f},{100*v['ci95'][1]:+.2f}] |\n"
    text+='''
区间条件于各初始化；同一开发发牌重复使用，不能作新独立验证。教师阶段在自己访问的开发轨迹中出完漏选均为0；跨最终模型状态时仍有少量漏选，不能外推任意状态。

从200手教师演示、三组最终开发轨迹和三组教师阶段开发轨迹按事前规则选取7508状态、788421完整候选，6权重共4730526个评分值。原演示多候选子集上，教师top-1一致率57.43%–64.22%，标准化遗憾0.0068–0.0092；DMC后一致率45.17%–52.79%，遗憾0.0423–0.0900，平均分数跨度明显收窄。其余六来源的同状态检查也显示偏离教师排序，不能等同于棋力下降或实现错误；DMC与教师回归目标不同。

总控逐局独立重建新增624局与全部7508个选择状态，核验完整候选、教师目标、行为分母和分组统计；固定112状态×6模型的直接forward抽查通过。22项新增CUDA解释器测试与11类审计负例通过；旧源哈希未变。两张PNG/SVG图、78行发牌组CSV与84行评分汇总CSV已保存并视觉检查。

## 下一工作包：P3h 教师偏好辅助损失的单条件配对试验

保持既有教师阶段权重、规则、编码、网络和后续DMC手数一致，对照原DMC与加入一个固定权重教师偏好辅助损失的条件。先冻结完整候选损失定义、权重、分母、初始化/发牌、计算差异、唯一验证候选与新验证日程，再执行；不据已看验证反复扫参或选优。该方法能否改善棋力仍待检验，P3h尚未训练或验收。

贪心主基线保留，P4桌面端后置，完整比赛/进还贡/P6仍未完成；9000000..9009999保留测试封存。
'''
    status.write_text(text,encoding='utf-8',newline='\n')
    readme=ROOT/'README.md'
    with readme.open('a',encoding='utf-8',newline='\n') as f:f.write('''

2026-09-26 **P3g教师拟合与后续DMC变化诊断已验收**。新增624局开发评测；对7508状态、788421完整候选作六模型同状态评分，完整复算通过。教师阶段开发胜率32.69%/31.73%/34.62%，对应DMC后30.29%/34.13%/30.29%；三组差值区间均包含0。教师排序一致率下降及分数跨度压缩已有诊断证据，尚不能据此宣称棋力退化。

本轮无新训练、无新验证、无晋级。详见 [P3G_ACCEPTANCE.md](docs/P3G_ACCEPTANCE.md) 与 [P3G_PROTOCOL.md](docs/P3G_PROTOCOL.md)。证据、78行发牌组CSV、84行评分汇总CSV和两张PNG/SVG位于`artifacts/evaluations/p3g-fit-v1/`。新增22项测试、11类审计负例通过，旧源码哈希保持一致。

```powershell
# 从磁盘独立重建评测/诊断状态，复算指标并作固定CUDA直接forward抽查；不训练
./scripts/run_gpu.ps1 experiments/review_p3g.py artifacts/evaluations/p3g-fit-v1
```

P3整体仍为PARTIALLY_ACCEPTED。下一工作包P3h将先冻结教师偏好辅助损失的单条件配对试验与新验证日程；辅助损失能否提升棋力仍未知。最终界面仍采用桌面端，P4后置。
''')
    figures=list((out/'descriptives/outputs/figures').glob('*.png'))
    write(out/'descriptives/visual-qa.json',dict(status='PASS',method='controller viewed actual PNG exports',checks=['full labels readable','no cropped panels','fixed sequential proportion and regret scales','paired values/intervals match audit','overlapping initial point labels replaced by numeric legend'],figure_sha256={p.name:digest(p) for p in figures}))
    trace=json.loads((out/'descriptives/trace.json').read_text('utf-8'))
    for name,h in trace['artifact_sha256'].items():assert digest(out/name)==h
    pre=json.loads((out/'preregistration.json').read_text('utf-8'))
    for group in ('source_sha256','input_sha256'):
        for name,h in pre[group].items():assert digest(ROOT/name)==h,name
    assert (out/'report.md').read_bytes()==(ROOT/'docs/P3G_ACCEPTANCE.md').read_bytes()
    files=[p for p in out.rglob('*') if p.is_file()]
    files += [ROOT/'docs/STATUS.md',ROOT/'README.md',ROOT/'docs/P3G_ACCEPTANCE.md',ROOT/'docs/P3G_PROTOCOL.md']
    files += [ROOT/name for name in ('scripts/p3g_run.py','scripts/p3g_evaluate.py','experiments/p3g_guard.py','experiments/p3g_scores.py','experiments/review_p3g.py','experiments/report_p3g.py','experiments/figures_p3g.py','experiments/deliver_p3g.py','tests/test_p3g_guard.py','tests/test_p3g_scores.py')]
    receipt=dict(status='SCOPED_ACCEPTANCE_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),engineering='ACCEPTED_TEACHER_FIT_DIAGNOSTIC_V1',overall_P3='PARTIALLY_ACCEPTED',validation_gate='NOT_ESTABLISHED',model_promoted=False,training_executed=False,new_validation_executed=False,reserved_test_executed=False,artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(files))})
    write(out/'delivery-receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],files=len(receipt['artifact_sha256']))))

if __name__=='__main__':main()
