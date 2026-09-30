"""Publish a scoped acceptance only after frozen-source and independent audit gates."""
from pathlib import Path
import sys,json,csv,re,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.p3e_common import check,read,write,digest
from experiments.p5e_protocol import INITS,ARMS
from experiments.controller_p5e_audit import freeze,historical
from experiments.controller_p5e_training import rows


def metric(v):return v['metrics']['win_rate']
def pct(x):return f'{100*x:.2f}%'
def interval(v):return f"[{100*v['ci95'][0]:.2f}%, {100*v['ci95'][1]:.2f}%]"+('（退化）' if v['degenerate'] else '')
def delta(v):return f"{100*v['estimate']:+.2f} [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]"
def table(headers,records):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in records])


def main():
    check(len(sys.argv)==3,'run path and full test receipt')
    out=Path(sys.argv[1]).resolve();testpath=Path(sys.argv[2]).resolve()
    check(not (out/'delivery-receipt.json').exists(),'fresh delivery')
    pre=freeze(out);a=read(out/'controller-audit.json');t=read(testpath)
    check(a['status']==t['status']=='PASS' and t['skipped']==0,'audit and full tests')
    check(a['scope']=='ACCEPTED_PAIRED_LABEL_BALANCED_STUDY_V1' and a['ordinary_p5b_compatibility']=='BITWISE_EQUAL_ALL_THREE','accepted scope')
    check(a['preregistration_sha256']==digest(out/'preregistration.json'),'audit registration')
    for name,h in {**t['sources'],**a['controllers']}.items():check(digest(ROOT/name)==h,'tested/frozen source '+name)
    check(digest(testpath.with_suffix('.log'))==t['log_sha256'],'full test log hash')
    historical()
    tr=a['training'];ev=a['evaluations'];num=a['first_four_wave_numerics']
    check(sum(x['hands'] for x in tr.values())==9600 and sum(x['games'] for x in ev.values())==4368,'complete fixed budgets')
    counts={k:sum(x[k] for x in tr.values()) for k in ('samples','updates','decisions','scored_candidates','frozen_candidates','single_class_batches','dual_class_batches')}
    primary=metric(ev['validation-label_balanced']['summary']['dmc|dmc|greedy'])
    relative=a['paired_validation']['dmc|dmc|greedy']['label_balanced_minus_ordinary']
    training=[];development=[];validation=[]
    for seed in INITS:
        x=tr[f'ordinary-{seed}'];y=tr[f'label_balanced-{seed}']
        training.append([seed,x['samples'],y['samples'],x['updates'],y['updates'],
            f"{y['dual_class_batches']}/{y['updates']}",f"{y['positive']}/{y['samples']}"])
        x=metric(ev[f'ordinary-{seed}']['summary']['dmc|dmc|greedy']);y=metric(ev[f'label_balanced-{seed}']['summary']['dmc|dmc|greedy'])
        development.append([seed,pct(x['estimate']),pct(y['estimate']),delta(a['paired_development'][str(seed)]['dmc|dmc|greedy']['label_balanced_minus_ordinary'])])
    for opponent in ('greedy','random','team'):
        key='dmc|dmc|'+opponent;x=metric(ev['validation-ordinary']['summary'][key]);y=metric(ev['validation-label_balanced']['summary'][key])
        validation.append([opponent,f"{round(x['estimate']*520)}/520; {pct(x['estimate'])} {interval(x)}",
            f"{round(y['estimate']*520)}/520; {pct(y['estimate'])} {interval(y)}",delta(a['paired_validation'][key]['label_balanced_minus_ordinary'])])
    behavior=[]
    for key,label in [('optional_pass_rate','可出牌时选择过牌'),('missed_finish_rate','有一次出完机会时未出完'),
                      ('teammate_overtake_rate','可回应队友时盖过队友'),('endgame_optional_pass_rate','五张及以下手牌时可选过牌')]:
        cells=[]
        for arm in ARMS:
            v=ev['validation-'+arm]['ratios'][key]
            cells.append(f"{v['numerator']}/{v['denominator']} = "+('不适用' if v['rate'] is None else pct(v['rate'])))
        behavior.append([label,*cells])
    batch_rows=[]
    for job in pre['specification']['training_order']:
        for w in rows(out/'training'/job/'waves.jsonl.gz'):
            for b in w['batches']:batch_rows.append(dict(job=job,wave=w['wave'],**b))
    check(len(batch_rows)==counts['updates'],'batch export denominator')
    with (out/'training-batches.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,list(batch_rows[0]));writer.writeheader();writer.writerows(batch_rows)
    clusters=[]
    for job,v in ev.items():
        for matchup,s in v['summary'].items():
            for c in s['clusters']:clusters.append(dict(job=job,matchup=matchup,**c))
    fields=['job','matchup']+sorted(set().union(*(set(c) for c in clusters))-{'job','matchup'})
    with (out/'deal-clusters.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(clusters)
    absolute=('通过本轮预设验证门槛，但不自动晋级；保留测试需另行预注册。' if a['validation_gate']=='VALIDATION_GATE_PASSED'
              else '未通过绝对晋级门槛，不晋级、不替换greedy；相对弱对照的改善不能替代绝对门槛。')
    interpretation=('配对差区间下界大于0，支持这个固定初始化、预算和评测分布下均衡目标相对普通目标的优势；不证明跨初始化总体稳定性。'
        if relative['ci95'][0]>0 else '配对差区间上界小于0，支持这个固定初始化与预算下均衡目标相对普通目标下降；不外推所有均衡方案。'
        if relative['ci95'][1]<0 else '配对差区间包含0，本轮未建立该固定主候选相对普通目标的优势；不证明等效。')
    scan=read(ROOT/'artifacts/evaluations/p5e-seed-availability.json')
    nstates=sum(v['samples'] for v in num);nupdates=sum(v['updates'] for v in num)
    doc=f'''# P5e 标签均衡目标等发牌配对研究验收

2026-09-29。总控工程结论：**ACCEPTED_PAIRED_LABEL_BALANCED_STUDY_V1**。唯一主候选新验证：**{a['validation_gate']}**。P5整体PARTIALLY_ACCEPTED，model_promoted=false，greedy主基线保留。

## 固定设计与完整预算

遵循P5E_PROTOCOL.md，三个配对初始化314510/314511/314512，ordinary/label_balanced两臂都用相同mixed池，从头初始化各训练1600手/400wave。共9600次执行、仅1600份共用开发发牌100000..101599；没有教师预训练或动态池。除目标外配置相同，完整候选不截断。目标干预之后访问轨迹、样本数、更新数可变化，不能称等计算量。

总计{counts['samples']}学习样本、{counts['updates']}次CUDA更新、{counts['decisions']}次训练决策；current非探索评分{counts['scored_candidates']}完整候选、frozen评分{counts['frozen_candidates']}候选。六作业按事前分块随机顺序执行，所有final模型冻结后才开始评测，没有按开发结果改选模型或追加训练。

{table(['初始化','ordinary样本','均衡样本','ordinary更新','均衡更新','均衡双类批次/更新','均衡正标签/样本'],training)}

均衡目标仅在双类批次中重新加权；单类批次退回普通MSE。标签比例按决策计，不能当作独立牌局胜率。全部{len(batch_rows)}行批次计数/权重/损失导出到training-batches.csv，检查点保存initial、wave1/2/3/4、wave200、final。

## 开发集结果

26个已查看原始发牌组，每组8变体，每模型208局greedy，共1248局。差值为label_balanced−ordinary，单位百分点；区间条件于固定初始化，只作点态描述。

{table(['初始化','ordinary胜率','均衡胜率','配对差与95%区间/百分点'],development)}

## 固定主候选新验证

事前指定label_balanced-314510，ordinary-314510为同期配对对照。新验证208000..208064在运行前检查{len(scan['files'])}份历史JSON/JSONL/gzip文件，零命中、零解析错误。65原始发牌组、13级牌各5组、每组8变体；两臂各对三对手评1560局，共3120局。区间按级牌分层、原始发牌成组bootstrap5000次，seed314553。

{table(['对手','ordinary胜局/胜率/95%区间','均衡胜局/胜率/95%区间','配对差与95%区间/百分点'],validation)}

唯一主要绝对比较为均衡候选对greedy：**{pct(primary['estimate'])} {interval(primary)}**。{absolute}

{interpretation}配对差与random/team均为预设次要点态描述，不是多重比较校正后的总体声明。退化区间（若有）只复述本批样本，不能解释为真实总体胜率确定。不得将这次与P5b不同验证发牌上的胜率差直接当作训练进步。

## 行为描述与机制边界

下表分别合并两候选对三对手的1560局验证；不同策略的访问轨迹和机会分母不同。合法过牌不自动判错，行为差不能单独证明标签重加权修复了某一因果机制。

{table(['行为','ordinary分子/分母及比例','均衡分子/分母及比例'],behavior)}

## 总控独立验收

- 全部9600手训练与4368局评测完成并独立重演，评测零失败、非法动作和超时；全量训练核对角色、发牌、完整候选、动作、current样本归属、终局标签、探索与打乱RNG、批次顺序/标签数/权重/分母/损失聚合及计数。
- 三对初始化完全相同，固定池不变，所有预定checkpoint来源、运行时、目标、RNG、谱系和计数核对通过。三个ordinary作业的400wave原日志字段，以及共有initial/wave1/wave200/final模型、Adam、RNG和计数与P5b mixed逐位一致。
- 六作业前四wave共24wave、{nstates}学习样本、{nupdates}更新，独立网络表达式完整核对{sum(v['direct_policy_requests'] for v in num)}神经请求/{sum(v['direct_policy_candidates'] for v in num)}候选的排序；独立逐样本权重表达式重算损失/梯度/Adam。实际包含{sum(v['dual_class_batches'] for v in num)}双类与{sum(v['single_class_batches'] for v in num)}单类批次；最大模型误差{max(v['max_model_abs_error'] for v in num):.9g}，最大Adam误差{max(v['max_adam_abs_error'] for v in num):.9g}，容差rtol3e-4/atol3e-6。未声称逐更新重算全部2400wave历史梯度或全量训练策略评分。
- 完整评测牌谱、候选覆盖、CUDA测量、行为分子/分母、胜率区间和配对差区间独立复算通过；{len(a['negative_rejections'])}类训练/统计篡改拒绝。缓存绑定输入/控制器/预注册哈希。
- 全套 **{t['tests']}项GPU测试通过、零跳过**；新guard覆盖两种目标真实载入、8769完整候选与错误哈希/挂起/崩溃/非法响应关闭。历史冻结源码和不可变产物哈希保持一致。所有协议、执行/控制器源码和输入ZIP可复核。

唯一主候选SHA-256：`{a['candidate_sha256']}`。路径：training/label_balanced-314510/final/checkpoint.pt。完整逐发牌聚类导出{len(clusters)}行deal-clusters.csv。

## 剩余边界

208000..208064现已查看，不能因负结果回收成新验证；9000000..9009999继续封存。P5尚无已晋级学习模型。本轮不追加训练、扫权重或改选初始化。后续工作需依据本次结果另定协议，保持工程验收与棋力晋级分开；P4桌面端与P6完整比赛继续后置。

训练仍为可信本地同步wave；模型评测spawn启动60秒、请求2秒硬截止，固定基线沿用返回后2秒检查。没有无限时长稳定性、跨硬件逐位恢复或干净机器重建结论。
'''
    target=ROOT/'docs/P5E_ACCEPTANCE.md';check(not target.exists(),'fresh acceptance');target.write_text(doc,encoding='utf-8')
    (out/'README.md').write_text(f'''# P5e 复核入口

协议docs/P5E_PROTOCOL.md；结果docs/P5E_ACCEPTANCE.md。

在项目根目录使用记录的CUDA解释器：

```powershell
./scripts/run_gpu.ps1 scripts/p5e_test.py
./scripts/run_gpu.ps1 scripts/p5d_full_test.py artifacts/evaluations/p5e-full-tests-NEW.log
./scripts/run_gpu.ps1 experiments/controller_p5e_audit.py {out.relative_to(ROOT).as_posix()} --output artifacts/evaluations/p5e-audit-NEW.json
```

所有输出必须全新。审计缓存通过输入/源码/预注册哈希核对后可复用；上述审计仍会重做24wave数值与篡改检查。preregistration.json和源码/输入ZIP绑定运行；candidates.json在评测前锁定final；controller-audit.json记录独立验收，delivery-receipt.json绑定交付。

本轮验证208000..208064已查看。冻结入口scripts/p5e_run.py的再次运行只能称复现，不能当新验证；新的训练假设必须另立协议、独立目录和验证范围，不能覆盖当前数据。
''',encoding='utf-8')
    status_path=ROOT/'docs/STATUS.md';s=status_path.read_text('utf-8')
    headline=f'当前交付：**P5e标签均衡目标配对研究完成：ACCEPTED_PAIRED_LABEL_BALANCED_STUDY_V1**。9600手训练/4368局评测全部独立核验，全套{t["tests"]}项GPU测试零跳过。固定主候选对greedy {pct(primary["estimate"])} {interval(primary)}，{a["validation_gate"]}；模型未晋级，greedy保留。P5仍PARTIALLY_ACCEPTED，P4/P6后置。'
    s=re.sub(r'当前交付：[^\n]+',headline,s,count=1)
    s=re.sub(r'\| P5 棋力增强 \|[^\n]+','| P5 棋力增强 | PARTIALLY_ACCEPTED | P5e标签均衡目标配对研究已验收；棋力验证结论单列，尚无晋级模型，后续研究须另立协议。 |',s,count=1)
    section=f'''## 2026-09-29 P5e标签均衡目标配对研究验收

**ACCEPTED_PAIRED_LABEL_BALANCED_STUDY_V1**；固定主候选验证 **{a['validation_gate']}**。协议P5E_PROTOCOL.md，验收P5E_ACCEPTANCE.md，证据{out.relative_to(ROOT).as_posix()}。

三个配对初始化、同mixed池ordinary/label_balanced两臂各1600手，共9600次执行、1600份共用发牌，{counts['samples']}学习样本、{counts['updates']}CUDA更新。全部4368局评测完成；主候选label_balanced-314510对greedy {pct(primary['estimate'])} {interval(primary)}，同期配对差{delta(relative)}个百分点。{absolute}

全部训练/评测牌谱及批次标签/权重/分母、RNG与统计独立核验通过；24个真实wave数值重算、三个ordinary与P5b mixed全400wave及共有checkpoint核心状态逐位一致。{len(a['negative_rejections'])}类篡改拒绝，全套{t['tests']}GPU测试零跳过。208000..208064已查看，9000000..9009999继续封存。P5仍PARTIALLY_ACCEPTED；P4/P6后置，后续研究另立协议。

'''
    marker='## 2026-09-29 P5d标签均衡目标工程验收';check(marker in s,'status insertion anchor')
    s=s.replace(marker,section+marker,1);status_path.write_text(s,encoding='utf-8')
    paths=[p for p in out.rglob('*') if p.is_file()]+[target,status_path,ROOT/'docs/P5E_PROTOCOL.md',testpath,testpath.with_suffix('.log'),Path(__file__)]
    write(out/'delivery-receipt.json',dict(status='PASS',scope=a['scope'],validation_gate=a['validation_gate'],
        model_promoted=False,reserved_test_executed=False,full_gpu_tests=t['tests'],skipped=0,
        artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths}))
    print(json.dumps(dict(status='PASS',counts=counts,primary=primary,relative=relative,tests=t['tests'],clusters=len(clusters))))


if __name__=='__main__':main()
