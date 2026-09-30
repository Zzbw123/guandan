"""Update live status only after the P3f independent audit has passed."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    root=Path(sys.argv[1]).resolve();a=json.loads((root/'controller-audit.json').read_text(encoding='utf-8'));assert a['status']=='PASS'
    m=a['evaluations']['validation-teacher']['summary']['dmc|dmc|greedy']['metrics']['win_rate']
    p=next(x for x in a['paired'] if x['validation'] and x['matchup']=='dmc|dmc|greedy')
    rate=100*m['estimate'];lo,hi=[100*v for v in m['ci95']];dl,dh=p['ci95_delta_pp']
    path=ROOT/'docs/STATUS.md';text=path.read_text(encoding='utf-8')
    old=next(line for line in text.splitlines() if line.startswith('当前交付：'))
    new=f'当前交付：**P0核心规格、P1规则环境、P2固定基线评测、P3a CPU学习闭环、P3b GPU运行时、P3c同步训练/恢复、P3d固定预算验证、P3e规模对照及P3f教师偏好回归配对实验在各自范围内验收通过**。P3f完成4800手训练、4368局评测；固定教师候选对贪心{rate:.2f}%，相对配对DMC对照提升{p["delta_pp"]:.2f}个百分点，但仍未达到晋级门槛。P3整体仍为部分完成，模型未晋级，桌面端后置。'
    text=text.replace(old,new)
    old=next(line for line in text.splitlines() if line.startswith('| P3 学习闭环 |'))
    text=text.replace(old,f'| P3 学习闭环 | PARTIALLY_ACCEPTED | P3f诊断、3初始化教师/DMC配对训练及固定候选验证工程通过。教师对greedy {rate:.2f}% [{lo:.2f}%,{hi:.2f}%]，NOT_ESTABLISHED；相对本轮control改善，尚未超过贪心，需进一步学习质量诊断。 |')
    marker='## 下一工作包：P3f 动作评分诊断与单一训练改进对照'
    assert marker in text;text=text[:text.index(marker)]
    lines=['## P3f 动作评分诊断与教师偏好回归配对实验（2026-09-26）','',
        '**ACCEPTED_PAIRED_TEACHER_STUDY_V1**；固定教师候选验证 **NOT_ESTABLISHED**。证据目录`artifacts/evaluations/p3f-teacher-v1/`，完整规程和验收为`docs/P3F_PROTOCOL.md`、`docs/P3F_ACCEPTANCE.md`。',
        '',
        '前置诊断重演1248局P3e开发评测牌谱，对1853状态的22317完整候选比较原评分、chunk37和直接forward，最大误差2.68e-7、动作排序全部一致；568条真实训练输入/终局标签及3批梯度匹配。只说明所查路径通过，未证明编码全局充分性，也没有改动规则。','',
        '3个配对初始化，每臂800手：前200手为DMC对照或贪心教师完整候选偏好回归，后600手共同DMC；双方在200手统一重置Adam和随机状态、仅传递模型参数。共4800手、500388条决策观察、2950次CUDA更新，800份唯一训练发牌。两种训练目标的标签数/更新数/耗时不同，不声称等计算量。','',
        '| 初始化 | control开发胜率 | teacher开发胜率 | 配对差值/百分点 | 95%区间/百分点 |','|---|---:|---:|---:|---:|']
    for r in a['paired']:
        if not r['validation']:
            x,y=r['ci95_delta_pp'];lines.append(f'| {r["seed"]} | {100*r["control"]:.2f}% | {100*r["teacher"]:.2f}% | {r["delta_pp"]:+.2f} | [{x:+.2f},{y:+.2f}] |')
    lines+=['','开发同26原始发牌、8变体/组。三个点估计均提高，第一组差值区间包含0；区间条件于各初始化，不宣称跨初始化总体稳定性。',
        '',f'独立验证事前固定初始化314380，两臂各对greedy/random/team评520局，共3120局；65个新原始发牌组、每级5组。教师对greedy {rate:.2f}% [{lo:.2f}%,{hi:.2f}%]，对照{100*p["control"]:.2f}%；配对差值+{p["delta_pp"]:.2f}pp [{dl:.2f},{dh:.2f}]。本轮相对弱对照改善，但教师绝对表现未达到>=55%且区间下界>50%的门槛；不替换贪心主基线。',
        '',
        '全部4368评测对局及4800手训练牌谱由总控独立核验，零失败/非法动作/超时。GPU全套133测试通过，CPU128通过、5项GPU测试跳过；11类审计负例均拒绝。历史注册源均保持原hash。两阶段权重传递、Adam/RNG重置与计数核验通过，两个PNG/SVG图包和546行场景发牌组CSV已保留。首轮审计误拒绝守护进程正常terminate的-15退出码，按冻结实现修正并完整重审；原正式实验未改动。',
        '',
        '最终标准检查点仅记录第二阶段600手/150wave；总800手由外部lineage证明。未来续训须读取第一阶段记录，避免重复使用100000..100199而误认为新数据。203000..203064现为已查看验证，不得重标为新验证；9000000..9009999保留测试继续封存。',
        '',
        '## 下一工作包：P3g 教师拟合质量与DMC阶段变化诊断','',
        '先在固定开发集评估现有第一阶段教师权重，复核完整候选的教师排序一致率、可出完机会和动作分数，再与对应最终DMC模型配对比较。目的是区分教师阶段尚未学好、分布变化或后续DMC保留不足；这些仍是待验证解释。诊断前不扩训、不改规则或网络，不根据本轮验证结果反复选择模型。依据开发证据再冻结一个训练改进条件和新验证日程。',
        '',
        'P3保留贪心主要固定基线，随机与队友启发式作为其他对照。P4桌面端后置，完整比赛/进还贡/P6仍未完成。','']
    path.write_text(text+'\n'.join(lines),encoding='utf-8',newline='\n')
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    old='下一包P3f为动作评分/训练目标诊断及单一训练改进条件对照，须另行冻结预算和新验证数据；P3整体仍为PARTIALLY_ACCEPTED。'
    assert old in text;text=text.replace(old,'后续教师偏好回归配对对照已在下文P3f完成；P3整体仍为PARTIALLY_ACCEPTED。')
    text+=f'''\n\n2026-09-26 **P3f教师偏好回归与DMC配对实验已验收，模型未晋级**。3个初始化、两臂各800手，共4800手训练和4368局评测。相同阶段重置条件下，以前200手贪心偏好回归替换DMC，随后统一训练600手DMC。三个开发点估计均提高；固定初始化314380在65组新验证上对greedy为{rate:.2f}%（95%区间{lo:.2f}%–{hi:.2f}%），比配对control高{p['delta_pp']:.2f}个百分点（差值区间{dl:.2f}–{dh:.2f}）。这尚未达到棋力晋级门槛。完整证据见 [P3F_ACCEPTANCE.md](docs/P3F_ACCEPTANCE.md) 与 [P3F_PROTOCOL.md](docs/P3F_PROTOCOL.md)。

```powershell
# 从磁盘重新核验全部训练/评测牌谱、完整候选、权重衔接与统计，不重复训练
py -3.13 experiments/review_p3f.py artifacts/evaluations/p3f-teacher-v1
# 按原固定预算复现；重用已看验证日程，不构成新的独立验证
./scripts/run_gpu.ps1 scripts/p3f_run.py --output artifacts/evaluations/my-p3f-repeat
py -3.13 experiments/review_p3f.py artifacts/evaluations/my-p3f-repeat --write
```

P3f完整证据、图表及546行场景发牌组CSV位于`artifacts/evaluations/p3f-teacher-v1/`。GPU133项测试通过；当前规则、旧训练和旧评测源码保持原hash。最终checkpoint只含第二阶段600手DMC计数，前200手通过phase1/和reset.json关联；未来续训必须同时检查第一阶段已用种子。新验证203000..203064现已查看，保留测试继续封存。下一包P3g先诊断现有教师阶段的排序拟合质量及后续DMC的变化，再决定单一训练改进；桌面端仍后置。
'''
    readme.write_text(text,encoding='utf-8',newline='\n');print('STATUS and README updated from audited P3f results')

if __name__=='__main__':main()
