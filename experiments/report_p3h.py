"""Write the scoped P3h scientific report from completed independent audits."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'src'))
from p3e_common import read,write,check,digest

def pct(v):return f'{100*v:.2f}%'
def ci(v):return f'[{100*v[0]:.2f}%, {100*v[1]:.2f}%]'

def main():
    root=Path(sys.argv[1]).resolve();a=read(root/'controller-audit.json');n=read(root/'controller-negative.json')
    f=read(root/'fidelity/controller-audit.json');t=read(ROOT/'artifacts/evaluations/p3h-test-receipt.json')
    batch_audit=read(root/'batch-order-audit.json')
    check(a['status']==n['status']==f['status']==t['status']==batch_audit['status']=='PASS','all acceptance prerequisites')
    final=a['evaluations']['validation-aux'];metric=final['summary']['dmc|dmc|greedy']['metrics']['win_rate']
    hands=sum(v['hands'] for v in a['training'].values());samples=sum(v['samples'] for v in a['training'].values())
    updates=sum(v['updates'] for v in a['training'].values());games=sum(v['games'] for v in a['evaluations'].values())
    rel=root.relative_to(ROOT).as_posix()
    primary_pair=next(r for r in a['paired'] if r['validation'] and r['matchup']=='dmc|dmc|greedy')
    dlo,dhi=primary_pair['ci95_delta_pp']
    text=['# P3h 教师偏好辅助损失单条件配对试验验收','',
        f'日期：2026-09-26。总控工程结论 **{a["engineering"]}**；固定主候选验证 **{a["validation_gate"]}**。',
        f'固定aux-314380对greedy胜率 **{pct(metric["estimate"])}，95%成组区间{ci(metric["ci95"])}**。P3整体仍为 **PARTIALLY_ACCEPTED**，模型未晋级，贪心主基线保留。','',
        f'同发牌新验证中，相对control对greedy的差值为{primary_pair["delta_pp"]:+.2f}个百分点，95%配对区间[{dlo:+.2f}, {dhi:+.2f}]。这是条件于固定初始化的次要点态比较，区间下界接近0；不外推跨初始化稳定性，也不能代替绝对晋级门槛。三个开发初始化的变化有正有负，区间均含0；固定教师诊断的top-1、遗憾和MSE变化也不一致。', '',
        '## 已冻结的设计与实际执行','',
        '继承P3f三个teacher前200手权重314380/314381/314382。每个初始化分control和aux，两臂都仅传递模型参数，并重置Adam、Python策略RNG与torch CPU/CUDA RNG。每模型新增600手，seed100200..100799，4环境同步采样；没有重训原200手。',
        '`L = mean_selected_DMC_MSE + lambda * mean_observation(mean_full_candidate_teacher_MSE)`，control的lambda=0，aux的lambda=0.1。教师score沿用P3f完整候选min-max归一化至[-0.8,0.8]；相等/单候选目标为0。每个DMC批观察的全部候选均参与辅助项；每观察先平均，再观察等权，累加两个梯度后只更新一次Adam，尾批保留。',
        '该辅助项回归绝对评分，既不是概率也不是最优价值；可能与终局价值目标冲突。固定0.1只检验一个条件，没有扫参。匹配环境手数，未匹配轨迹、更新次数、标签数或计算时间。运行期间GPU有其他图形负载；shared-gpu-observation.json保留实测快照，各任务耗时不作算法速度比较。',
        f'实际新增 **{hands}手训练、{samples}条决策观察、{updates}次CUDA更新、{games}局评测**。600个训练发牌seed被三个初始化和两个条件重复使用，不是3600个独立发牌。每次策略调用仅获得PlayerObservation与完整合法Action列表。','',
        '| 条件/初始化 | 新增手数 | 决策观察 | CUDA更新 | 辅助评分候选 | 训练与保存秒数 |',
        '|---|---:|---:|---:|---:|---:|']
    for job,v in sorted(a['training'].items()):text.append(f'| {job} | {v["hands"]} | {v["samples"]} | {v["updates"]} | {v["auxiliary_candidates"]} | {v["seconds"]:.2f} |')
    text += ['', '## 开发配对比较','',
        '使用已看开发seed108100..108125，13级牌各2组、每组8个轮转/换队变体，六模型各208局。配对差值为aux−control。区间按级牌分层、原始发牌成组bootstrap5000次，固定seed314405；条件于各初始化，不进行跨初始化总体显著性推断。','',
        '| 初始化 | control对greedy | aux对greedy | 差值/百分点 | 95%描述区间/百分点 |',
        '|---|---:|---:|---:|---:|']
    for r in a['paired']:
        if not r['validation']:
            lo,hi=r['ci95_delta_pp'];text.append(f'| {r["seed"]} | {pct(r["control"])} | {pct(r["aux"])} | {r["delta_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    text += ['', '这是重复使用的开发日程；不能作为新独立验证。原始组差值、零差值组和极端值保存在controller-audit.json及groups.csv，零组缺失。未删异常发牌或按结果追加样本。','',
        '## 新验证与绝对门槛','',
        '训练前唯一指定aux-314380为主候选，control-314380为同发牌对照。新验证seed204000..204064，65个原始发牌组、13级牌各5组，每组8个变体；对greedy/random/team每条件1560局，两条件共3120局。独立样本单位为65个原始发牌组。','',
        '| 对手 | control胜率及95%区间 | aux胜率及95%区间 | aux−control/百分点及95%区间 |',
        '|---|---:|---:|---:|']
    for r in a['paired']:
        if r['validation']:
            cm=a['evaluations']['validation-control']['summary'][r['matchup']]['metrics']['win_rate']
            am=final['summary'][r['matchup']]['metrics']['win_rate'];lo,hi=r['ci95_delta_pp']
            text.append(f'| {r["matchup"].split("|")[-1]} | {pct(cm["estimate"])} {ci(cm["ci95"])} | {pct(am["estimate"])} {ci(am["ci95"])} | {r["delta_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    text += ['',
        '唯一主验证指标为aux-314380对greedy胜率，预设要求点估计>=55%、区间下界>50%、非退化且零失败/非法动作/超时。相对control的增幅不能代替绝对棋力门槛；其他对手与配对差值均为次要点态描述。没有以不同验证发牌上的P3f/P3h胜率差证明提升。','',
        '## 固定同状态教师偏好诊断','',
        '从原teacher-314380阶段一回放按轨迹顺序取前128个多候选观察，六个最终模型评分同一完整候选集合。选择规则先于正式训练冻结，未按评分结果选状态。它是有目的的有限诊断子集，不是全状态无偏估计，不出具总体置信区间。','',
        '| 初始化 | control教师top-1 | aux教师top-1 | control/aux标准化遗憾 | control/aux候选MSE |',
        '|---|---:|---:|---:|---:|']
    for seed in (314380,314381,314382):
        c,b=[f['summary'][f'{arm}-{seed}'] for arm in ('control','aux')]
        text.append(f'| {seed} | {pct(c["top1"])} | {pct(b["top1"])} | {c["regret"]:.5f} / {b["regret"]:.5f} | {c["mse"]:.5f} / {b["mse"]:.5f} |')
    text += ['',
        f'共128观察、{f["candidates"]}完整候选、六模型全部768组直接forward对照；最大数值差{f["max_score_error"]:.10g}。重新构造每个选择状态，独立核算教师分数、argmax、遗憾和分母。教师一致率、遗憾和MSE不等于棋力。','',
        '## 工程验收、追溯与限制','',
        f'- 新增{t["total"]}项测试在实际CUDA解释器全部通过，无跳过；覆盖零权重逐位对照、不等候选数/单候选的稠密联合损失及梯度、8769完整候选、检查点恢复、真实spawn推理和超时/崩溃/非法返回回收。',
        '- 三个control的全部600手训练结果与P3f对应后续DMC模型参数、Adam、Python/torch CPU/CUDA RNG、计数逐位一致；本轮的两臂从同一教师权重和同一随机状态出发。',
        '- 额外从六个任务的真实回放与初始化随机流独立还原逐次探索抽样及wave末样本打乱，验证实际探索动作、每个批次的观察/完整候选数量及最终策略RNG。预计批次身份顺序的SHA256保存在batch-order目录；这不是aux全部梯度的独立重算。',
        f'- 从磁盘独立重建全部{hands}手训练与{games}局评测牌谱、完整候选数、终局、行为分母；逐批核对辅助观察/候选覆盖及损失汇总，独立复算胜率和分层成组bootstrap。零失败、非法动作和超时；所有推理进程已关闭回收。',
        f'- {len(n["cases"])}类审计负例全部拒绝：'+', '.join(n['cases'])+'。',
        '- 专用检查点版本为gd-p3h-checkpoint-v1 / gd-p3h-wave-v1，保存辅助权重及真实600手/150wave历史、模型/Adam/RNG。此前200手依靠外部reset谱系；继续训练须额外排除100000..100199。不能用旧GPUTrainer恢复为原DMC，也不声称半局或跨设备恢复。',
        '- 原规则、编码、网络与历史训练评测源文件保持原hash；本轮源码ZIP、输入权重、运行前后源和结果回执全部校验。独立审计及报告脚本后续加入，并记录各自SHA256，交付时另存controller-source.zip；它们没有修改已冻结实验。',
        '- 测试和全部600手control复现是有限数值证据；本轮没有逐更新独立重算aux全部梯度。固定768组forward比较不外推为所有状态数值等价。',
        '- 每模型评测推理启动60秒、单次决策2秒硬截止，固定基线沿用返回后的软时限；每个任务外层3600秒截止。训练及验证总数通过，不构成长期吞吐或桌面延迟承诺。',
        '- 本轮新验证204000..204064现在属于已查看验证集，不能重标为新测试；9000000..9009999保留测试未使用。','',
        '## 证据入口','',f'- 正式目录：`{rel}`。协议：`docs/P3H_PROTOCOL.md`。',
        '- `preregistration.json`、`source-snapshot.zip`、`receipt.json`：冻结条件与运行回执。',
        '- `controller-audit.json`、`controller-negative.json`：全量磁盘重演、独立复算和负例。',
        '- `fidelity/scores.jsonl.gz`、`fidelity/controller-audit.json`：固定同状态全部评分及直接forward核验。',
        '- `descriptives/groups.csv`、`descriptives/outputs/figures/paired-development.png`、`paired-validation.png`及SVG：原始发牌组与可检查图。',
        f'- 固定主候选SHA256：`{a["candidate_sha256"]}`。','',
        '```powershell',f'py -3.13 experiments/review_p3h.py {rel}',
        f'./scripts/run_gpu.ps1 experiments/inspect_p3h.py {rel} --verify','```','']
    report='\n'.join(text)
    for path in (root/'report.md',ROOT/'docs/P3H_ACCEPTANCE.md'):
        if path.exists():
            check(path.read_bytes()==report.encode('utf-8'),'existing report must match regeneration exactly')
        else:
            with path.open('x',encoding='utf-8',newline='\n') as h:h.write(report)
    usage=dict(new_training_hands=hands,unique_continuation_deal_seeds=600,
        training_seeds_previously_used_in_p3f=True,inherited_teacher_hands_per_model=200,
        development_deals=26,development_games=1248,validation_deals=65,validation_games=3120,
        new_viewed_validation=[204000,204064],model_promoted=False,reserved_test_executed=False)
    if (root/'data-usage.json').exists():check(read(root/'data-usage.json')==usage,'existing data usage must match')
    else:write(root/'data-usage.json',usage)
    print(json.dumps(dict(report=str(root/'report.md'),hands=hands,samples=samples,updates=updates,games=games,gate=a['validation_gate'])))
if __name__=='__main__':main()
