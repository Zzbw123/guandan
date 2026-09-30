"""Render an evidence-linked P3e review after independent audit succeeds."""
import json
from pathlib import Path
import sys

def main():
    root=Path(sys.argv[1]).resolve();a=json.loads((root/'controller-audit.json').read_text(encoding='utf-8'))
    assert a['status']=='PASS'
    pct=lambda x:f'{x*100:.2f}%'
    def rate(job,name):
        r=a['evaluations'][job]['ratios'][name]
        return f"{r['numerator']}/{r['denominator']} ({pct(r['rate']) if r['rate'] is not None else 'NA'})"
    s=['# P3e 开发集诊断、训练规模对照与独立验证验收','',
       f"2026-09-26 总控结论：工程 **{a['engineering']}**；候选独立验证 **{a['validation_gate']}**。未自动晋级，不替换贪心主基线。P3整体仍为PARTIALLY_ACCEPTED。",'',
       '## 冻结设计与实际执行','',
       '本轮只改变累计训练手数，沿用gd-gpu-wave-v1、gd-features-v1和gd-hand-v1。3个初始化使用相同1600份训练发牌，各保存400/800/1200/1600手检查点；400与1600来自同一训练轨迹，实际共训练4800手（不是4800份独立发牌）。全部训练完成后才执行六个模型的开发对照。初始化314370的1600手模型为事先指定的唯一独立验证候选，没有按开发结果挑选最好模型。','',
       '| 初始化 | 训练手数 | 决策样本 | CUDA更新 | 训练秒数 |','|---|---:|---:|---:|---:|']
    for t in a['training']:s.append(f"| {t['seed']} | {t['hands']} | {t['samples']} | {t['updates']} | {t['elapsed_s']:.3f} |")
    s+=['','训练时间包含定期检查点；测试/审计任务可能并发，不用于GPU吞吐基准或跨轮加速比。CUDA完成推理、损失、梯度和Adam；CPU承担规则与特征。逐wave设备检查与完整终局回放均通过。训练磁盘日志保存终局摘要和验证标记，不宣称离线重演了未保存的完整训练牌谱。','',
        '## 开发集行为诊断','',
        '开发seed108000..108025共26原始发牌，每级2组，每组4轮换×2换队。旧P3d候选对greedy及greedy对greedy参考各208局。参考策略在其自身轨迹上计数，不能把两者机会次数差当同状态的因果效果。','',
        '| 指标 | 旧P3d候选 | greedy参考 |','|---|---:|---:|']
    for title,key in [('可选过牌','optional_pass_rate'),('可盖过队友时实际盖过','teammate_overtake_rate'),('一次出完机会中未出完','missed_finish_rate'),('<=5张残局可选过牌','endgame_optional_pass_rate')]:
        s.append(f"| {title} | {rate('old',key)} | {rate('baseline',key)} |")
    s+=['','强制过牌与可选过牌分开；盖过队友或未立即出完只是可复核的行为分类，不等价于已证明的错误。旧候选有13/180次一次出完机会未出完，例如trial dmc|dmc|greedy:108003:2:0:0的第92步（零基）手中只有一张牌且可以合法打出，却选择了pass。当时手牌/所选动作/可出完动作在behavior.jsonl中，隐藏手牌没有传入策略。','',
        '## 固定规模的配对开发对照','',
        '| 初始化 | 400手胜率 | 1600手胜率 | 差值（百分点） | 差值95%成组区间 |','|---|---:|---:|---:|---|']
    for r in a['paired']:
        lo,hi=r['ci95_delta_pp'];s.append(f"| {r['seed']} | {pct(r['win_rate_400'])} | {pct(r['win_rate_1600'])} | {r['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    improved=sum(r['delta_pp']>0 for r in a['paired'])
    crossing=sum(r['ci95_delta_pp'][0]<=0<=r['ci95_delta_pp'][1] for r in a['paired'])
    s+=['',f'3次初始化中{improved}次胜率点估计上升，{crossing}组差值区间包含0。改善幅度依赖初始化；区间包含0不代表证明没有改善。单纯将预算从400增至1600手，尚未在本轮三个初始化中形成一致的强改善证据，不能据最大增幅推断稳定泛化。']
    s+=['','每个模型208局，但统计单位为26个原始发牌；组内8变体先平均。同初始化的两个预算逐发牌配对，再按级牌分层bootstrap5000次、seed314375。区间仅描述固定初始化的发牌不确定性，不是跨初始化总体区间；六场景共用同26组，不能拼成156独立组，也没有多重显著性排名。','']
    for n,v in a['stability'].items():s.append(f"- {n}手：3次初始化胜率均值{pct(v['mean'])}，样本SD {v['sd']*100:.2f}个百分点，范围{pct(v['min'])}–{pct(v['max'])}。")
    s+=['','三个初始化和26个开发发牌组是预定资源预算，不宣称80%功效或已证明长期/多初始化稳定性。全部结果保留；没有因某次成绩好坏增加训练。在线损失分段在controller-audit.json中，样本来自变化的自对弈策略，不能用损失下降替代棋力验证。','',
        '各检查点一次出完机会遗漏如下，可与配对胜率一起检查：','',
        '| 初始化 | 400手遗漏 | 1600手遗漏 |','|---|---:|---:|']
    for r in a['paired']:s.append(f"| {r['seed']} | {rate(str(r['seed'])+'-400','missed_finish_rate')} | {rate(str(r['seed'])+'-1600','missed_finish_rate')} |")
    s+=['','## 唯一候选的独立验证','',f"候选SHA256：`{a['candidate_sha256']}`。绑定发生在本轮验证对局前。验证seed202000..202064，65个原始发牌（每级5组），对3个固定对手各520局，共1560局。",'',
        '| 对手 | 胜局/总局 | 胜率 | 95%分层成组区间 |','|---|---:|---:|---|']
    for opponent in ('greedy','random','team'):
        summary=a['evaluations']['validation']['summary'][f'dmc|dmc|{opponent}'];m=summary['metrics']['win_rate'];lo,hi=m['ci95']
        wins=round(sum(g['win_rate']*8 for g in summary['clusters']))
        s.append(f"| {opponent} | {wins}/520 | {pct(m['estimate'])} | [{pct(lo)}, {pct(hi)}] |")
    s+=['','唯一确认性主比较为对greedy，门槛点估计>=55%且非退化95%区间下界>50%，并要求全部工程检查通过。random/team是描述性次要比较。统计单位是65份原始发牌而非520独立局；按级牌分层、层内重抽原始组5000次。检查全部组完整、层组数、退化和分布；不需要正态/方差齐性前提。P3d和P3e的独立验证发牌不同，不能直接以其胜率差证明训练改善。另两个1600手模型未做独立验证，不将这个固定候选的验证结论外推到所有初始化。','',
        '## 工程与总控复核','',
        '- 新增13项诊断/日程测试；完整111项在GPU解释器通过，CPU Python3.13为109通过、2项CUDA测试跳过。',
        '- 3224局评测（416参考诊断、1248模型开发对照、1560独立验证）全部完成，零失败/非法动作/超时；总控按原始日程从磁盘重演全部评测牌谱，逐步核对候选枚举数量与GPU评分覆盖。',
        '- 独立实现分层bootstrap、逐组胜率和关键行为分子分母，与生产汇总一致。全部12份训练检查点验证模型/Adam/计数/seed历史，CUDA设备证据、样本及更新预算一致。',
        '- 缺局、重复局、错误胜负、错误区间、伪造晋级、行为分母篡改、候选选择篡改等7类负例被拒绝。',
        '- P2/P3a/P3b/P3c/P3d历史注册源码保持原hash；新实验源码快照与运行前后hash一致。',
        '- GPU每请求2秒硬时限、启动60秒，固定基线返回后2秒软检查，外层每作业1800秒；与P3d边界一致。未裁剪候选，未新增策略隐藏信息。','',
        'v1因py -3.14指向一套未安装torch的解释器，在首个模型加载时失败；未进行训练或任何诊断/验证对局。v1失败记录保留。v2改用既有Python314 GPU解释器，协议、源码、seed预算和候选选择完全相同。没有安装/删除任何运行时。新增run_gpu.ps1读取已记录解释器并预检CUDA，避免启动器漂移。','',
        '## 来源、复现与保留边界','',
        '使用experimental-design、statistical-analysis、reproducibility-skill、visualization-skill；沿用项目Python/Matplotlib和artifacts证据目录。总控决定协议、日程、算法边界、运行与独立验收；gpt-6-sol/high仅实现观察行为计数及限定测试。','',
        '```powershell',
        '# 只读完整复算已有P3e结果',
        './scripts/run_gpu.ps1 experiments/review_p3e.py artifacts/evaluations/p3e-scale-v2',
        '# 原预算重跑；这是原验证集复现，不是新的未查看验证',
        './scripts/run_gpu.ps1 scripts/p3e_run.py --output artifacts/evaluations/my-p3e-repeat',
        './scripts/run_gpu.ps1 experiments/review_p3e.py artifacts/evaluations/my-p3e-repeat --write',
        '```','',
        '正式目录artifacts/evaluations/p3e-scale-v2包含preregistration、source-snapshot、3组训练日志与12个检查点、9个评测作业的绑定/结果/测量/牌谱/行为计数、receipt、controller-audit、controller-negative。descriptives提供原始发牌组CSV、PNG、可编辑SVG及SHA追踪。','',
        '202000..202064现在属于已查看验证；201000..201064仍为旧已查看验证。保留测试9000000..9009999未使用。greedy继续作为主基线；本包没有P4桌面、P5高水平棋力或P6完整比赛结论。','']
    path=root/'report.md'
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(s))
    print(path)
if __name__=='__main__':main()
