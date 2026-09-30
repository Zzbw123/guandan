"""Evidence-gated Chinese report and reproducible P5b delivery receipt."""
from pathlib import Path
from hashlib import sha256
import json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.p3e_common import check,read,write,digest
from experiments.p5b_protocol import INITS,ARMS
from experiments.controller_p5b_audit import freeze

def metric(result):return result['metrics']['win_rate']
def pct(value):return f'{100*value:.2f}%'
def ci(value):return f"[{100*value['ci95'][0]:.2f}%, {100*value['ci95'][1]:.2f}%]"+('（退化）' if value['degenerate'] else '')
def delta(value):return f"{100*value['estimate']:+.2f} [{100*value['ci95'][0]:+.2f}, {100*value['ci95'][1]:+.2f}]"

def main():
    check(len(sys.argv)==2,'formal run path required');root=Path(sys.argv[1]).resolve()
    check(not (root/'delivery-receipt.json').exists(),'fresh delivery')
    freeze(root);audit=read(root/'controller-audit.json')
    check(audit['status']=='PASS','controller acceptance')
    for path,h in audit['controllers'].items():check(digest(ROOT/path)==h,'controller unchanged')
    log=ROOT/'artifacts/evaluations/p5b-full-tests-gpu-v1.log';text=log.read_text('utf-8-sig')
    check('Ran 237 tests' in text and text.rstrip().endswith('OK') and 'skipped=' not in text,'full GPU regression')
    training=audit['training'];evaluations=audit['evaluations'];numerics=audit['first_wave_numerics']
    check(sum(r['hands'] for r in training.values())==9600 and sum(r['games'] for r in evaluations.values())==4368,'complete budgets')
    counts={key:sum(r.get(key,0) for r in training.values()) for key in ('samples','updates','decisions','scored_candidates','frozen_candidates')}
    primary=metric(evaluations['validation-mixed']['summary']['dmc|dmc|greedy'])
    relative=audit['paired_validation']['dmc|dmc|greedy']['mixed_minus_selfplay']
    degeneracy=('主比较的成组bootstrap分布退化；退化区间只复述本次样本，不能证明真实胜率恒定或提供有用的总体精度保证。沿用原协议的非退化要求，不事后更换区间方法以争取通过。'
                if primary['degenerate'] else '主比较的成组bootstrap区间非退化。')
    train_table=['| 初始化 | selfplay样本 | mixed样本 | selfplay更新 | mixed更新 |','|---|---:|---:|---:|---:|']
    dev_table=['| 初始化 | selfplay开发胜率 | mixed开发胜率 | mixed−selfplay/百分点及95%区间 |','|---|---:|---:|---|']
    for seed in INITS:
        a=training[f'selfplay-{seed}'];b=training[f'mixed-{seed}']
        train_table.append(f"| {seed} | {a['samples']} | {b['samples']} | {a['updates']} | {b['updates']} |")
        a=metric(evaluations[f'selfplay-{seed}']['summary']['dmc|dmc|greedy'])
        b=metric(evaluations[f'mixed-{seed}']['summary']['dmc|dmc|greedy'])
        d=audit['paired_development'][str(seed)]['dmc|dmc|greedy']['mixed_minus_selfplay']
        dev_table.append(f"| {seed} | {pct(a['estimate'])} | {pct(b['estimate'])} | {delta(d)} |")
    val_table=['| 对手 | selfplay胜率及95%区间 | mixed胜局/520 | mixed胜率及95%区间 | 配对差/百分点及95%区间 |',
               '|---|---|---:|---|---|']
    for opponent in ('greedy','random','team'):
        key='dmc|dmc|'+opponent;a=metric(evaluations['validation-selfplay']['summary'][key]);b=metric(evaluations['validation-mixed']['summary'][key])
        d=audit['paired_validation'][key]['mixed_minus_selfplay']
        val_table.append(f"| {opponent} | {pct(a['estimate'])} {ci(a)} | {round(b['estimate']*520)} | {pct(b['estimate'])} {ci(b)} | {delta(d)} |")
    behavior_table=['| 行为口径 | selfplay计数及比例 | mixed计数及比例 |','|---|---|---|']
    for key,label in [('optional_pass_rate','有出牌可选时过牌'),('teammate_overtake_rate','可回应队友时盖过队友'),
                      ('missed_finish_rate','有一次出完机会但未出完'),('endgame_optional_pass_rate','五张及以下手牌时可选过牌')]:
        cells=[]
        for arm in ARMS:
            value=evaluations['validation-'+arm]['ratios'][key]
            rate='不适用' if value['rate'] is None else pct(value['rate'])
            cells.append(f"{value['numerator']}/{value['denominator']} = {rate}")
        behavior_table.append(f"| {label} | {cells[0]} | {cells[1]} |")
    absolute=('通过本轮验证门槛，但不自动晋级；保留测试需另行设计。' if audit['validation_gate']=='VALIDATION_GATE_PASSED'
              else '未达到预设绝对晋级门槛，不能以相对弱对照改善替代；不晋级、不替换greedy。')
    relative_text=('主配对差区间下界大于0，支持这个固定初始化、固定预算下mixed配方相对selfplay的优势；不是跨初始化总体证明。'
                   if relative['ci95'][0]>0 else '主配对差区间全部低于0，支持本次固定初始化与预算下mixed配方落后于selfplay；不外推所有混合策略。'
                   if relative['ci95'][1]<0 else '主配对差区间包含0，本轮未建立固定主候选相对selfplay的优势；这也不证明等效。')
    doc=f'''# P5b 等发牌预算固定策略池配对研究验收

日期：2026-09-27。总控工程结论：**ACCEPTED_PAIRED_POOL_STUDY_V1**。唯一主候选验证：**{audit['validation_gate']}**。model_promoted=false，greedy主基线保留，P5整体PARTIALLY_ACCEPTED。

## 事前设计与执行

按P5B_PROTOCOL.md执行三个配对初始化314510/314511/314512、两臂selfplay/mixed，每作业从随机初始化开始训练1600手/400wave，共9600手执行。六作业使用同一1600份开发发牌100000..101599，重复执行不增加独立发牌数。无教师预训练、无辅助损失；frozen是本次初始化固定副本。运行顺序按初始化分块随机，最终wave400是唯一候选；中间检查点仅供恢复/审计。

共{counts['decisions']}次训练决策，其中{counts['samples']}条current样本进入学习，{counts['updates']}次CUDA更新；current非探索完整评分{counts['scored_candidates']}候选，frozen评分{counts['frozen_candidates']}候选。每臂同1600手不意味着同样本、同更新或等计算量。

{chr(10).join(train_table)}

正式证据目录{root.relative_to(ROOT).as_posix()}，冻结协议、日程、源码与输入ZIP均可核验。新验证207000..207064运行前扫描788份历史JSON/JSONL/gzip文件，零命中、零解析错误。所有训练结束后绑定六个final SHA，再执行开发和验证，未按开发结果选择模型。

## 开发集配对结果

复用26原始发牌、每组8变体，每模型208局对greedy，总1248局。区间条件于固定初始化，是已查看开发集上的点态描述；三个初始化不是总体稳定性证明。

{chr(10).join(dev_table)}

## 固定主候选的新验证

主候选事前固定mixed-314510，selfplay-314510为同时期对照；两者各对三对手评1560局，共3120局。65原始发牌组、13级牌各5组、每组8变体；按级牌分层、原始发牌成组bootstrap5000次，seed314523。下表差值均为mixed−selfplay，配对差与random/team是预设次要点态描述。

{chr(10).join(val_table)}

表中“退化”表示bootstrap重采样分布没有变化；例如本批全零胜局得到的[0,0]只复述样本，不能解释成真实总体胜率确定为零，也不是有效的总体精度保证。

唯一主比较mixed对greedy为 **{pct(primary['estimate'])} {ci(primary)}**。{absolute}

{degeneracy}

{relative_text}本实验的估计对象是相同手数下整个训练配方的差异：队友/对手分布、学习座位数、轨迹、样本数和更新数同时改变，不能将结果归因为隔离后的纯策略池效应。固定池含greedy/team及未训练frozen，不能外推到强历史模型池或其他对手分布。

正式评测时己方两个座位均由同一候选模型控制；训练中包含异质队友不等于已证明异质队友评测中的泛化。不能用与P3l等旧研究不同验证发牌上的胜率差，证明本轮相对旧候选的棋力变化。

## 预设行为描述

以下合并各候选对三类对手的1560局验证。两策略访问的状态与机会分母不同；这些是可重演的行为分类，不能单独锁定棋力差异的原因，也不将所有合法过牌判定为错误。

{chr(10).join(behavior_table)}

唯一主候选：training/mixed-314510/final/checkpoint.pt，SHA-256：

`{audit['candidate_sha256']}`

## 总控独立验收

- 9600手训练和4368局评测全部从磁盘完整重演；评测零失败、非法动作或超时。训练逐手核对发牌、角色、动作索引、完整合法候选数、current样本归属、终局标签、探索序列和批次打乱；所有checkpoint边界RNG/计数匹配。
- 三组两臂初始模型、固定池、空Adam、全部RNG逐位一致；固定池全程不变；初始/wave1/wave200/final源、配置、运行时与发牌历史核对通过。
- 六个真实首wave共{sum(x['samples'] for x in numerics)}样本、{sum(x['updates'] for x in numerics)}次更新，独立完整神经策略评分{sum(x['direct_policy_requests'] for x in numerics)}请求/{sum(x['direct_policy_candidates'] for x in numerics)}候选；直接网络表达式、DMC梯度及Adam重算后，模型最大绝对误差{max(x['max_model_abs_error'] for x in numerics):.6g}，Adam最大绝对误差{max(x['max_adam_abs_error'] for x in numerics):.6g}。容差rtol3e-4、atol3e-6；未逐更新重算全部历史梯度或全部训练策略评分。
- 所有评测指标、行为计数与分母、成组胜率区间、配对差值及区间独立复算；模型守护的候选覆盖、CUDA设备、请求截止和关闭记录通过。
- {len(audit['negative_rejections'])}类训练/统计审计篡改均拒绝；新增7项工程测试及全套 **237项GPU测试通过、零跳过**，包含8769候选、新守护真实加载和挂起/崩溃/非法输出回收。
- 历史注册和不可变交付哈希保持一致；详情见controller-audit.json的historical。源码及输入ZIP复核通过；原开发方案和历史冻结实现未改写。

## 剩余范围

207000..207064现在是已查看验证，不能再次当新验证；保留测试9000000..9009999未使用。本包未自动晋级、未替换greedy。P5后续方向需根据本次有效样本量与固定池结构提出新假设并单独预注册，不能追加本轮训练或验证来追逐门槛。P4桌面端、P6完整比赛继续后置。

训练为可信本地同步wave，基线逐调用没有硬隔离；正式模型评测为spawn启动60秒/请求2秒硬截止。没有跨硬件逐位恢复、无限时长稳定性或干净机器重建结论。复现入口见正式目录README.md。
'''
    target=ROOT/'docs/P5B_ACCEPTANCE.md';check(not target.exists(),'fresh acceptance');target.write_text(doc,encoding='utf-8')
    readme=f'''# P5b 复核与复现

当前目录绑定等手数、三初始化、固定最终候选研究。结果与边界见docs/P5B_ACCEPTANCE.md。

在项目根目录使用PowerShell，复核已保存数据：

```powershell
./scripts/run_gpu.ps1 scripts/p5b_test.py
$env:RUN_P3F_CUDA='1'
$env:RUN_P3G_CUDA='1'
./scripts/run_gpu.ps1 scripts/validate.py
```

重新完整审计全部牌谱、统计、首wave数值与篡改负例，写入全新报告（不使用审计缓存）：

```powershell
./scripts/run_gpu.ps1 experiments/controller_p5b_recheck.py {root.relative_to(ROOT).as_posix()} artifacts/evaluations/p5b-recheck-NEW.json
```

输出文件必须不存在。原controller-audit.json和本目录数据保持不变。

正式重新训练需单独新目录和新的预注册设计；本轮207000..207064已经查看，不可当新验证。scripts/p5b_run.py记录的是这次冻结实验，重放仅能称复现，不能生成新的独立验证证据。训练和评测使用scripts/run_gpu.ps1绑定的GPU解释器。

源码快照：source-snapshot.zip；输入：inputs-snapshot.zip；独立审计代码：controller-snapshot.zip；候选：candidates.json；完整训练：training/*/waves.jsonl.gz；原始评测：evaluations/*/results.jsonl及replays.jsonl.gz；交付索引：delivery-receipt.json。
'''
    (root/'README.md').write_text(readme,encoding='utf-8')
    with zipfile.ZipFile(root/'controller-snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in list(audit['controllers'])+['experiments/controller_p5b_delivery.py','experiments/controller_p5b_recheck.py']:z.write(ROOT/path,path)
    status_path=ROOT/'docs/STATUS.md';status=status_path.read_text('utf-8');ls=status.splitlines()
    ls[4]=f"当前交付：**P5b等发牌预算固定策略池配对研究工程验收通过：ACCEPTED_PAIRED_POOL_STUDY_V1**。完成9600手训练/4368局评测、独立重演与统计复算，全套237项GPU测试无跳过。唯一主候选mixed-314510对greedy {pct(primary['estimate'])} {ci(primary)}，{audit['validation_gate']}，模型未晋级、greedy保留。P5整体PARTIALLY_ACCEPTED；P4桌面端/P6完整比赛继续后置。"
    status='\n'.join(ls)+'\n'
    status=re.sub(r'\| P5 棋力增强 \|[^\n]+','| P5 棋力增强 | PARTIALLY_ACCEPTED | P5a采样工程、P5b固定预算配对研究已验收；尚无已晋级学习模型，后续研究需新协议。 |',status,count=1)
    begin=status.index('## 2026-09-27 P5b执行中');end=status.index('## 2026-09-27 P5a',begin)
    update=f'''## 2026-09-27 P5b固定策略池配对研究验收

**ACCEPTED_PAIRED_POOL_STUDY_V1**；唯一主候选验证 **{audit['validation_gate']}**。协议P5B_PROTOCOL.md，验收P5B_ACCEPTANCE.md，正式目录{root.relative_to(ROOT).as_posix()}。

三个从头初始化、selfplay/mixed两臂各1600手，共9600次执行、1600份共用发牌，{counts['samples']}学习样本、{counts['updates']}次CUDA更新。同手数不代表同样本/更新或等计算量。无教师预训练，frozen是随机初始化固定副本。六个final模型训练后冻结，未按开发表现挑选。

开发1248局、新验证3120局全部完成。事先固定mixed-314510对greedy {pct(primary['estimate'])} {ci(primary)}，配对差{delta(relative)}个百分点。{absolute}207000..207064已查看；9000000..9009999继续封存。

全部训练/评测牌谱独立重演，完整候选、角色、样本归属、RNG/计数、统计/配对区间和行为分母通过；六首wave数值重算通过。{len(audit['negative_rejections'])}类审计篡改拒绝，237项GPU测试无跳过；历史源/不可变产物哈希一致。后续P5新假设单独设计，P4/P6继续后置。

'''
    status_path.write_text(status[:begin]+update+status[end:],encoding='utf-8')
    paths=[p for p in root.rglob('*') if p.is_file()]+[target,status_path,log,ROOT/'docs/P5B_PROTOCOL.md',Path(__file__)]
    receipt=dict(status='PASS',scope='ACCEPTED_PAIRED_POOL_STUDY_V1',validation_gate=audit['validation_gate'],model_promoted=False,
        reserved_test_executed=False,full_gpu_tests=237,artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths})
    write(root/'delivery-receipt.json',receipt)
    print(json.dumps(dict(status='PASS',files=len(receipt['artifact_sha256']),primary_win_rate=primary['estimate'],validation_gate=audit['validation_gate'])))

if __name__=='__main__':main()
