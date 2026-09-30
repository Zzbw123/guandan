"""Write the scoped P3g report from controller-audited data."""
import json
from pathlib import Path
import statistics
import sys

def main():
    root=Path(sys.argv[1]).resolve();project=Path(__file__).resolve().parents[1]
    a=json.loads((root/'controller-audit.json').read_text('utf-8'));assert a['status']=='PASS'
    f=lambda x:f'{100*x:.2f}%'
    pp=lambda x:f'{100*x:+.2f}'
    text=['# P3g 教师拟合质量与后续DMC变化诊断验收','',
        '日期：2026-09-26。总控结论：**ACCEPTED_TEACHER_FIT_DIAGNOSTIC_V1**。通过的是既有权重的开发诊断与可复算证据；没有新训练、没有新独立验证，模型未晋级。P3整体仍为 **PARTIALLY_ACCEPTED**，P3f既有棋力验证仍为 **NOT_ESTABLISHED**。','',
        '## 结论','',
        '本轮将教师拟合偏差和后续DMC的评分变化分开量化。教师阶段在原演示的固定多候选子集上，最高分动作一致率尚不充分，但标准化教师遗憾较小。相同状态上，后续DMC普遍降低教师最高分一致率、增大偏好误差，并压缩动作分数跨度。这支持“偏离教师偏好”的观察；开发胜率变化有正有负，不能称为已证明棋力退化或灾难性遗忘。','',
        '贪心教师的细小排序差可能使top-1不同而遗憾很小；教师分数既不是胜率也不是最优价值。DMC学习终局回报，目标本就不同；相对教师MSE上升不能单独判定DMC训练错误。','',
        '## 同发牌开发评测','',
        '新增三个教师阶段模型各208局，共624局；复用P3f对应最终模型624局。均为开发seed108100..108125，26个原始发牌、13级牌各2组，每组8个轮转/换队变体。差值为最终DMC−教师阶段；按级牌分层、原始发牌成组bootstrap5000次，种子314395。区间条件于各初始化，重复使用开发数据，不作为新的确认性验证。','',
        '| 初始化 | 教师200手后胜率 | 再DMC600手后胜率 | 配对变化/百分点 | 95%描述区间/百分点 |',
        '|---|---:|---:|---:|---:|']
    for seed,v in a['paired'].items():text.append(f"| {seed} | {f(v['phase1'])} | {f(v['final'])} | {pp(v['delta'])} | [{pp(v['ci95'][0])}, {pp(v['ci95'][1])}] |")
    text+=['',f"三个初始化的描述均值：教师阶段{f(statistics.mean(v['phase1'] for v in a['paired'].values()))}，最终DMC{f(statistics.mean(v['final'] for v in a['paired'].values()))}。没有把3次初始化合为独立总体显著性检验。",'',
        '## 同状态拟合诊断','',
        f"固定7来源：教师演示200手只纳入一次，3个最终模型和3个教师阶段模型各208局开发轨迹。逐步重建共1448份轨迹记录（重复发牌和不同模型轨迹不增加独立样本数）。每局取全部一次出完机会和前4个其他多候选状态；训练轨迹四座全取，开发只取focal队。共{a['selected_states']}状态、{a['candidate_sets']}完整候选；六权重评分合计{a['model_score_values']}个值。",'',
        '以下一致率、遗憾、MSE和跨度均只使用多候选状态；教师并列最高分任一动作均算一致。MSE先在每观察完整候选上平均，再对观察等权平均。出完漏选单独使用全部出完机会作为分母，含单候选机会。此选择偏重每局较早选择和出完机会，是有目的的诊断子集，不是全轨迹无偏估计。','',
        '### 原教师演示子集上的拟合','',
        '| 初始化/权重阶段 | 教师top-1一致率 | 标准化教师遗憾 | 候选MSE | 平均分数跨度 |',
        '|---|---:|---:|---:|---:|']
    for seed in a['paired']:
        for phase in ('phase1','final'):
            m=a['aggregates']['training'][f'{phase}-{seed}']['multi_choice']
            text.append(f"| {seed}/{phase} | {f(m['mean_teacher_top1'])} | {m['mean_normalized_regret']:.5f} | {m['mean_candidate_mse']:.5f} | {m['mean_spread']:.5f} |")
    text+=['','### 对应最终模型开发轨迹的同状态比较','',
        '每行比较相同初始化的两权重；两者面对同一来源的全部候选。全6模型×7来源的交叉结果保存在score-metrics.csv，不据此挑选候选。','',
        '| 轨迹来源/权重阶段 | 教师top-1一致率 | 标准化教师遗憾 | 候选MSE | 平均分数跨度 | 出完漏选/机会 |',
        '|---|---:|---:|---:|---:|---:|']
    for seed in a['paired']:
        source=f'final-{seed}'
        for phase in ('phase1','final'):
            m=a['aggregates'][source][f'{phase}-{seed}']['multi_choice'];allm=a['aggregates'][source][f'{phase}-{seed}']['all']
            text.append(f"| {source}/{phase} | {f(m['mean_teacher_top1'])} | {m['mean_normalized_regret']:.5f} | {m['mean_candidate_mse']:.5f} | {m['mean_spread']:.5f} | {allm['finish_misses']}/{allm['finish_opportunities']} |")
    text+=['','### 出完机会与分布边界','',
        '三个教师阶段模型在各自实际开发轨迹中出完漏选均为0；但在最终模型访问的状态上，相应教师阶段权重有少量漏选。这说明“自身轨迹没漏选”不能外推为任意状态都正确，也说明同状态交叉检查是必要的。所有7来源×6模型的机会数、漏选数、饱和比例和分数数组均保留。','',
        '## 独立验收','',
        '- P3f交付回执222个文件在本轮开始时全部匹配；预注册冻结86个源码和34个输入，源ZIP及运行前后哈希一致。旧规则、编码、网络、训练和评测实现均未修改。',
        '- 新增624局从磁盘独立重建初始牌、全部合法候选、动作、终局和行为统计；零失败、非法动作和超时；三个spawn推理进程已回收。队伍决策只收到PlayerObservation和完整Action列表。',
        f"- 对全部{a['selected_states']}选中状态独立重建选择集合、教师目标和指标；7来源每来源前16个状态、每状态六权重，共{a['direct_forward_comparisons']}组完整候选直接forward抽查，最大分数差{a['max_forward_error']:.10g}，达到预注册容差。这是固定抽查，未声称全状态数值等价。",
        f"- {len(a['negative_cases'])}类负例均拒绝："+'、'.join(a['negative_cases'])+'。',
        '- 新增22项测试在原CUDA解释器全部通过，含临时raw载入/完整候选CUDA推理、spawn超时与故障回收、哈希/绑定篡改及评分分母测试。独立指标公式另与1000组合成输入比对一致。原133项历史测试未在本轮重跑，相关源码哈希保持不变。',
        '- 第一阶段raw按外部文件/清单/模型哈希核验，仅加载模型用于推理；不伪造标准检查点的已训练手数、Adam或恢复状态。本轮没有梯度更新。',
        '- 新增评测与score数组绑定预注册、模型及输入；复用的最终模型开发证据保持P3f原始哈希。9000000..9009999保留测试未使用，已看验证清单未扩展。','',
        '## 下一工作包：P3h 教师偏好辅助损失的单条件配对试验','',
        '本轮支持优先检验“在DMC中适度保留教师偏好能否改善开发表现”，并不预设其会提高棋力。下一包保持P3f教师阶段权重、规则、编码、网络和后续DMC手数一致，对比原DMC与加入一个固定权重教师偏好辅助损失的条件；完整候选不裁剪。先冻结损失定义、权重、批次分母、初始化/发牌、计算差异、唯一验证候选与未看验证日程，再执行。不得在已看验证上扫权重或反复选优。若教师偏好保留增强而棋力没有改善，须如实保留该负结果。',
        '', 'P3g本包到此结束；P3h尚未训练或验收。贪心主基线保留，P4桌面端继续后置，完整比赛/进还贡/P6未实现。','',
        '## 证据与复算','',
        '- 协议：docs/P3G_PROTOCOL.md；正式目录：artifacts/evaluations/p3g-fit-v1。',
        '- 入口：preregistration.json、source.zip、run-receipt.json、controller-audit.json、controller-forward.jsonl.gz。',
        '- 完整评分：scores/scores.jsonl.gz；分来源报告：scores/report.json。',
        '- 表：descriptives/paired-groups.csv（78组行）、descriptives/score-metrics.csv（84汇总行）；图：descriptives/outputs/figures/development-change.png、teacher-fidelity.png及SVG。',
        '', '```powershell',
        './scripts/run_gpu.ps1 experiments/review_p3g.py artifacts/evaluations/p3g-fit-v1',
        '```','']
    content='\n'.join(text)
    for path in (root/'report.md',project/'docs/P3G_ACCEPTANCE.md'):
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(content)
    print('P3g report written from controller audit')

if __name__=='__main__':main()
