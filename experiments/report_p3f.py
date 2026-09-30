"""Render the accepted-scope P3f record from independent disk audit evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from p3e_common import read,write,digest,check

def pct(v):return f'{100*v:.2f}%'
def interval(v):return f'[{100*v[0]:.2f}%, {100*v[1]:.2f}%]'

def main():
    root=Path(sys.argv[1]).resolve();a=read(root/'controller-audit.json');negative=read(root/'controller-negative.json')
    check(a['status']==negative['status']=='PASS','audits must pass')
    diag=read(ROOT/'artifacts/evaluations/p3f-diagnostic-controller.json')
    tests=read(ROOT/'artifacts/evaluations/p3f-test-receipt.json')
    final=a['evaluations']['validation-teacher'];win=final['summary']['dmc|dmc|greedy']['metrics']['win_rate']
    hands=sum(r['total_hands'] for r in a['training'].values());samples=sum(r['total_samples'] for r in a['training'].values());updates=sum(r['total_updates'] for r in a['training'].values())
    games=sum(e['games'] for e in a['evaluations'].values());rootrel=root.relative_to(ROOT).as_posix()
    lines=['# P3f 教师偏好回归与DMC配对实验验收','',f'日期：2026-09-26。总控结论：**{a["engineering"]}**；通过的是本机配对训练、验证和可复算证据工程。',
        f'固定教师候选对greedy：**{pct(win["estimate"])}，95%区间{interval(win["ci95"])}**；预注册验证门槛 **{a["validation_gate"]}**。模型未晋级，P3整体仍为 **PARTIALLY_ACCEPTED**。','',
        '## 前置诊断','',
        f'- 完整重演P3e开发评测1248局，扫描61137个focal决策；选择全部1085个出完机会与每job前128个其他多候选状态，共{diag["score_states"]}状态、{diag["scored_candidates"]}候选。',
        f'- 原score_many、chunk37和直接forward最大差异{diag["max_score_error"]:.10g}；实际动作和argmax一致，无排名变化或所查出完/非出完编码冲突。',
        '- 4手真实训练的568条观察/动作/所属队终局标签多重集匹配，3批MSE反向梯度与独立公式一致。该模型仅供诊断，没有进入评测。',
        f'- P3e交付回执的{diag["p3e_receipt_files"]}个文件在本轮开始时全部匹配。',
        '- 上述证据只排查本轮指定状态、计算路径和标签，不证明编码全局充分性；未把可出完时的其他选择归咎于未发现的实现bug。','',
        '## 冻结条件和预算','',
        '两个训练条件、3个配对初始化314380/314381/314382，每模型800手，seed100000..100799；前200手为control DMC或teacher四座贪心演示偏好回归，后600手统一DMC。两臂统一在阶段边界重置Adam和随机状态，仅传递模型参数。',
        'teacher每观察完整候选分数min-max映射至[-0.8,0.8]，相等则0，含singleton；每64个观察FIFO更新一次，尾批保留，一轮，无候选裁剪。每观察先求候选MSE，再对观察平均。该目标不是胜率或校准价值。',
        '完整过程匹配环境手数，没有匹配更新次数或计算时间。第一阶段改变行为数据分布及目标函数，结果只归于整个第一阶段程序差异；control不是无重置P3e的直接复刻。',
        f'本轮正式执行 **{hands}手训练、{samples}条决策观察记录、{updates}次CUDA更新、{games}局评测**。教师每条观察含完整候选的回归目标，DMC每条观察只含所选动作的终局回报，因此观察数不是等量监督标签。4800手训练仅对应800个唯一发牌seed，重复执行不增加独立样本量。','',
        '| 训练条件/初始化 | 前200手样本/更新 | 后600手样本/更新 | 全程秒数 |',
        '|---|---:|---:|---:|']
    for job,r in sorted(a['training'].items()):
        p1,p2=r['phase1'],r['phase2'];lines.append(f'| {job} | {p1["samples"]}/{p1["updates"]} | {p2["samples"]}/{p2["updates"]} | {r["seconds"]:.2f} |')
    lines+=['','## 开发对照','',
        '同一26个原始发牌，13级牌各2组，每组8个轮转/换队变体；共6×208=1248局。先对原始发牌内变体平均，再对配对差值按级牌分层bootstrap5000次。区间条件于各初始化，不把三个初始化合成跨种子总体显著性。','',
        '| 初始化 | control对greedy | teacher对greedy | 配对变化/百分点 | 95%区间/百分点 |',
        '|---|---:|---:|---:|---:|']
    for r in a['paired']:
        if not r['validation']:
            lo,hi=r['ci95_delta_pp'];lines.append(f'| {r["seed"]} | {pct(r["control"])} | {pct(r["teacher"])} | {r["delta_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    lines+=['','三初始化汇总仅为描述统计：','']
    for arm,s in a['stability'].items():lines.append(f'- {arm}：均值{pct(s["mean"])}，样本标准差{100*s["sd"]:.2f}个百分点，范围{pct(s["min"])}–{pct(s["max"])}。')
    lines+=['','## 独立验证','',
        '训练前固定两臂初始化314380最终模型，不依据开发胜率选模型；新验证seed203000..203064，65个原始发牌、13级牌各5组，每组8变体。每臂对greedy/random/team各520局，两臂共3120局。每个场景的独立样本单位为65个发牌组。','',
        '| 对手 | control胜率及95%区间 | teacher胜率及95%区间 | teacher−control/百分点及95%区间 |',
        '|---|---:|---:|---:|']
    for r in a['paired']:
        if r['validation']:
            cm=a['evaluations']['validation-control']['summary'][r['matchup']]['metrics']['win_rate']
            tm=final['summary'][r['matchup']]['metrics']['win_rate'];lo,hi=r['ci95_delta_pp']
            lines.append(f'| {r["matchup"].split("|")[-1]} | {pct(cm["estimate"])} {interval(cm["ci95"])} | {pct(tm["estimate"])} {interval(tm["ci95"])} | {r["delta_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    lines+=['','主指标只有固定teacher候选对greedy胜率；门槛为点估计≥55%、区间下界>50%且非退化、零错误。control及随机/队友基线、配对差值均为次要点态描述；不声明全矩阵显著性。即使相对很弱的control提高，也不能替代绝对棋力门槛。','',
        '## 工程验收与可复算性','',
        f'- GPU测试{tests["gpu"]["passed"]}/{tests["gpu"]["total"]}通过；CPU测试{tests["cpu"]["passed"]}通过，{tests["cpu"]["skipped"]}项GPU条件测试跳过。新增完整候选目标、稠密/分块损失梯度更新、阶段存取、原始/记录版DMC逐位一致测试均实际通过CUDA验证。',
        f'- 总控逐手审计4800手训练牌谱和4368局评测牌谱，独立重新枚举完整候选并核终局、样本/更新计数、CUDA设备证据。评测零失败、零非法动作、零超时，子进程全部关闭。',
        '- 两臂相同初始化参数hash一致，阶段载入前后参数hash一致，Adam为空、计数归零，配对采样/torch/CUDA随机状态一致；三份教师演示终局摘要完全一致。',
        '- 最终标准检查点真实记录600手DMC/150wave，phase1 raw payload记录前200手；总800手由外部lineage证明。raw文件不是全流程任意中断恢复检查点。',
        '- 后续继续训练必须同时读取外部lineage，并排除第一阶段100000..100199已用种子；最终检查点的used_deal_seeds只覆盖第二阶段，不能单靠它认定全流程发牌未重复。',
        f'- 冻结源码ZIP、运行前后源码、检查点、全套结果哈希一致；历史源保护数量：{json.dumps(a["protected_sources"],ensure_ascii=False)}。',
        f'- {len(negative["cases"])}种负例全部拒绝：缺失/重复/篡改结果、伪造区间/晋级、行为分母、候选替换、未回收进程、错误权重传递、Adam未重置、随机状态错误。',
        '- 首轮审计误将推理子进程退出码限定为0而拒绝评测。冻结GPUInferenceGuard.close实际主动terminate再join，Windows multiprocessing映射为-15；8个任务均为该预期回收码且全部对局成功。修正审计契约后重新从磁盘完整核验，未修改或重跑正式实验。旧审计源码及失败日志保留，见audit-revision.json。',
        '- 预注册文件、时间戳、完整source ZIP、所有进度和运行日志均保留；验证种子扫描覆盖历史JSON/JSONL及gzip JSONL的seed字段。',
        '- 本轮未调用保留测试9000000..9009999；已看验证203000..203064从此不得复用为未见验证。','',
        '## 证据入口','',
        '- 协议：`docs/P3F_DIAGNOSTIC_PROTOCOL.md`、`docs/P3F_PROTOCOL.md`。',
        '- 前置证据：`artifacts/evaluations/p3f-diagnostic-controller.json`，以及对应score/target目录。',
        f'- 正式目录：`{rootrel}`；`preregistration.json`、`receipt.json`、`controller-audit.json`、`controller-negative.json`。',
        '- 图：`descriptives/outputs/figures/paired-development.png`、`paired-validation.png`及可编辑SVG；`descriptives/groups.csv`与`trace.json`。',
        f'- 固定teacher候选SHA256：`{a["candidate_sha256"]}`。','',
        '复算命令（不重复训练；追加`--write`仅用于尚未生成审计文件的已完成运行目录）：','',
        '```powershell',f'py -3.13 experiments/review_p3f.py {rootrel}','```','']
    text='\n'.join(lines)
    for path in (root/'report.md',ROOT/'docs/P3F_ACCEPTANCE.md'):
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    write(root/'data-usage.json',dict(training_execution_hands=hands,unique_training_deals=800,
        development_deals=26,development_games=1248,validation_deals=65,validation_games=3120,
        new_viewed_validation=[203000,203064],previous_viewed_validation=[[200000,200129],[201000,201064],[202000,202064]],
        reserved_test_executed=False,model_promoted=False,candidate_sha256=a['candidate_sha256']))
    print(json.dumps(dict(report=str(root/'report.md'),hands=hands,samples=samples,updates=updates,games=games,gate=a['validation_gate']),ensure_ascii=False))

if __name__=='__main__':main()
