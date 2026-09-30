"""Independently check every summary denominator and write the evidence-backed report."""
from pathlib import Path
import sys,json,csv,math,statistics,copy
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.p3e_common import read,write,digest,check
from experiments.p3k_run import OUT,paths

def values_for(rows,metric):
    if metric.startswith('loss_'):return [r['metrics']['losses'][metric[5:]] for r in rows]
    arm=next((a for a in ('absolute','centered') if metric.startswith(a+'_')),None)
    if not arm:return [r['metrics'][metric] for r in rows]
    key=metric[len(arm)+1:];g=[r['metrics']['geometry'][arm] for r in rows]
    if key in ('conflict','strong_conflict'):
        return [None if v['cosine'] is None else v['cosine']<(-.1 if key=='strong_conflict' else 0) for v in g]
    if key=='reverse_dmc':return [None if v['dmc_descent_factor'] is None else v['dmc_descent_factor']<0 for v in g]
    return [v[key] for v in g]

def check_summary(rows,summaries):
    seen=set()
    for s in summaries:
        key=(s['source'],s['model'],s['subset'],s['metric']);check(key not in seen,'unique summary row');seen.add(key)
        selected=[r for r in rows if r['identity']['source']==s['source'] and r['model']==s['model']
                  and (s['subset']=='all' or (s['subset']=='singleton')==r['metrics']['singleton'])]
        values=values_for(selected,s['metric']);finite=sorted(x for x in values if x is not None);n=len(finite)
        check(s['states']==len(selected) and s['valid']==n and s['nulls']==len(values)-n,'summary count and null denominator')
        calc=dict(mean=sum(finite)/n if n else None,median=(finite[(n-1)//2]+finite[n//2])/2 if n else None,
                  minimum=finite[0] if n else None,maximum=finite[-1] if n else None)
        for k,v in calc.items():
            check(s[k] is None if v is None else s[k] is not None and math.isclose(s[k],v,rel_tol=1e-12,abs_tol=1e-12),'independent summary '+k)
    check(len(seen)==9*9*3*22,'complete source/model/subset/metric table')

def main():
    audit=read(OUT/'controller-audit.json');check(audit['status']=='PASS','controller gate')
    with (OUT/'probes.jsonl').open(encoding='utf-8') as f:rows=[json.loads(s) for s in f]
    summary=read(OUT/'summary.json');selection=read(OUT/'selection.json');run=read(OUT/'run-receipt.json')
    check_summary(rows,summary)
    altered=copy.deepcopy(summary);altered[0]['mean']+=.1
    try:check_summary(rows,altered)
    except ValueError:negative=True
    else:raise ValueError('changed summary accepted')
    check((OUT/'outputs/metrics.csv').read_bytes()==(OUT/'outputs/tables/source-model-summary.csv').read_bytes(),'metrics CSV identity')
    with (OUT/'outputs/metrics.csv').open(encoding='utf-8',newline='') as f:csvrows=list(csv.DictReader(f))
    check(len(csvrows)==len(summary),'CSV row coverage')
    for row,entry in zip(csvrows,summary):
        for k,v in entry.items():check(row[k]==('' if v is None else str(v)),'CSV exact field')
    unique=len({(r['observation_sha256'],json.dumps(r['action'],sort_keys=True),r['reward']) for r in selection})
    names=list(paths()[0]);source='teacher-314380'
    lookup={(r['source'],r['model'],r['subset'],r['metric']):r for r in summary}
    def val(name,metric,field='median',src=source):return lookup[src,name,'multi',metric][field]
    check(read(OUT/'outputs/figures/visual-qa.json')['status']=='PASS','figure visual QA')
    report='''# P3k 固定权重梯度、尺度与动作间隔诊断验收

日期2026-09-26。**ACCEPTED_FIXED_WEIGHT_GRADIENT_DIAGNOSTIC_V1**。新增正式训练0手、新增评测0局；模型未晋级，P3仍PARTIALLY_ACCEPTED。

## 结论及其实际范围

在事前选定的小型开发子集上，0.1倍中心化辅助梯度通常比DMC梯度小，方向也并不一致。9来源×9模型的多候选分组内，范数比中位数范围0.63%–1.86%。这支持下一轮检验相对梯度尺度，不足以确定P3j失败原因，更不证明增大权重能改善棋力。

同一teacher-314380来源的13个多候选观察，六个最终模型的加权中心化范数比中位数为0.83%–1.24%；输出跨度的观察均值为0.143–0.450，而三个教师起点为1.005–1.126。该子集没有输出达到|q|>=.95。可描述为这些状态上的候选分数跨度缩小，不能推广为整个策略饱和或训练崩溃。

## 样本、完整候选与信息边界

冻结37个源码/协议文件和28个输入文件，随后从9轨迹来源各选13局、每局按既定三分位时点取2观察，共117个来源牌局、234个观察、1078个完整合法候选。每份权重都在相同观察上测试，共2106探针、6318份梯度对照。每个来源的26观察包括所有席位的教师演示或焦点队的最终开发行为，来源间分布和角色不同。

其中128观察为单候选，106为多候选（每来源8–18个）；单候选centered梯度严格为0，cosine为null，保留在all分母并单独列出。多候选子集中15观察的教师首选存在并列，未删除。所选状态最多48候选；本包诊断不是8769候选压力试验，完整回归中保留历史大候选测试。

三份教师来源选中的是相同演示轨迹；234观察去重后只有182个不同的公开观察/执行动作/终局标签组合。重复权重和来源不增加独立样本；不生成p值或总体置信区间。图仅显示multi子集，CSV同时完整保留all/multi/singleton。

模型仅接收公开PlayerObservation和合法Action编码。完整牌局只供实验管理重建；实际执行动作的±1回报从独立重演终局取得，不传入模型特征或教师目标。跨模型比较的DMC标签属于原行为策略，具有off-policy限制，不能解释为每份权重自己采取该策略的反事实回报。

## 同一教师来源的配对诊断

下表每模型均使用同样13个多候选观察。A/C分别指absolute/centered辅助目标；两者范数比都已乘实际权重0.1。cosine为观察中位数，输出跨度为观察均值。

| 固定模型 | A范数比 | C范数比 | A cosine | C cosine | 平均输出跨度 |
|---|---:|---:|---:|---:|---:|
'''
    for name in names:
        report+=f"| {name} | {100*val(name,'absolute_weighted_norm_ratio'):.2f}% | {100*val(name,'centered_weighted_norm_ratio'):.2f}% | {val(name,'absolute_cosine'):+.3f} | {val(name,'centered_cosine'):+.3f} | {val(name,'model_span','mean'):.3f} |\n"
    report+='''
完整81个来源/权重组合及三种子集见outputs/tables/source-model-summary.csv（5346行），没有仅保留表现较好的来源或初始化。各组同时记录cosine<0、cosine<-.1比例、有效/null分母及极值，弱负值不被解释为统计显著冲突。

## 极端值、饱和与动作间隔

不能只用中位数决定提高权重：多候选单探针中，加权absolute范数比最大22.85，centered最大1.655。共有15个“模型/来源观察/辅助目标”组合的DMC下降因子为负，其中包含重复教师来源；不是15个独立牌局。这表示该固定点的联合负梯度存在局部DMC上升方向，不等于实际Adam步或历史训练发生了同样事件。

达到|q|>=.95的只有5个模型/观察探针，全部来自absolute-314382来源seed108110的step50同一个观察。不能据此把输出饱和判为普遍根因。教师前两名差距、并列数、模型前两名差距、tanh导数与完整评分均已保存，可核查这个局部尾部现象。

## 总控独立验收

- 9项新增CUDA数值测试及197项全套回归通过，无跳过；实际运行Python3.14.3、PyTorch2.12.0+cu130、RTX5060 Laptop GPU。生产代码、测试实现与最终验收职责分开，子代理仅编写有边界的测试文件。
- 总控逐步独立重演117个来源牌局、检查每步状态digest，全部234观察的选择身份、动作、标签、完整候选与公开观察哈希一致。
- 全部2106探针的3种梯度逐元素与独立稠密表达式比较，最大元素绝对误差7.153e-7；随后控制器在独立进程重新计算全部2106稠密探针，核验完整评分、损失、Gram矩阵及参数块求和。最大评分误差1.192e-7。
- Gram点积绝对误差最大1.714e-4；这不是单个梯度元素误差。控制器按事前冻结的相对1e-3/绝对1e-5混合阈值核验，较大范数点积通过相对阈值；主逐元素梯度检查另用rtol3e-4、atol3e-6，未因结果调整。
- 九份权重计算前后模型哈希一致，没有优化器更新；源文件/输入绑定再次通过。
- 9类负例全部拒绝：缺探针、重复探针、错误终局回报、丢候选、篡改cosine、篡改Gram、错误状态、未登记模型、伪晋级。后处理另拒绝篡改汇总均值，并独立复算全部5346行汇总和CSV分母。
- 首次运行前修复了脚本导入路径和Trial字段名；正式冻结后的计算/审计没有失败或条件变更。首次图右侧色条文字越界，保留原版本并调整画布边距，最终PNG/SVG已视觉核验；没有修改数据。

## 后续单条件协议与未完成项

P3L_PROTOCOL.md已交付，状态DESIGN_READY_NOT_RUN。先实现并验收有上限的批次梯度范数配比：d=||gD||、c=||gC||；任一<=1e-12时alpha=0，否则alpha=min(1,0.1*d/c)。固定alpha后组合gD+alpha*gC，保持一次Adam步和完整候选分母。

该规则把辅助梯度范数控制在DMC的10%以内，权重上限1；在非零分支使原始联合梯度的DMC下降因子不低于0.9。这个一阶界限不保证Adam步、有限步长损失下降或棋力。需要先完成稠密数值、单步Adam、恢复/故障和constant路径逐位一致等工程验收，再启动协议中的固定预算配对试验。

本轮没有实现或试跑normcap，没有新增验证。拟用206000..206064尚须历史可用性扫描；不能称其已确认未使用。205000..205064维持已查看，9000000..9009999继续封存。最近棋力证据仍是P3j主候选对greedy26.73% [22.69%,30.77%]，NOT_ESTABLISHED；贪心主基线保留，桌面端/完整比赛继续后置。

## 可复现入口与证据

运行与复核方法见同目录README.md。preregistration.json和source-snapshot.zip固定计算；selection.json保存状态来源；probes.jsonl保存完整评分、目标与梯度内积；controller-audit.json、controller-negative.json、postprocess-audit.json、full-tests-gpu.log与最终delivery-receipt.json串联验收。outputs/metrics.csv与汇总表字节一致；outputs/figures/gradient-map.png及可编辑SVG由81行figure-data.csv生成。

适配器：reproducibility-skill用于输入/产物/哈希闭环；experimental-design 1.1用于后续配对块与重复单位；visualization-skill用于Python图表的颜色语义、完整色标、数据追溯与视觉QA。没有调用需要安装的新方法包；实际绘图版本见outputs/figures/trace.json。本项目尚未进入论文或最终提交，诊断验收不等于提交验收。
'''
    # Verify numerical report constants against the frozen outputs before writing prose.
    check(unique==182 and audit['candidates']==1078 and audit['singleton_states']==128,'report counts')
    check(run['max_gradient_abs_error']<7.154e-7 and audit['max_gram_abs_error']<1.714e-4,'reported numerical bounds')
    write(OUT/'postprocess-audit.json',dict(status='PASS',summary_rows=len(summary),csv_rows=len(csvrows),unique_labeled_states=unique,
        negative_changed_summary_rejected=negative,summary_sha256=digest(OUT/'summary.json'),script_sha256=digest(Path(__file__))))
    (OUT/'report.md').write_text(report,encoding='utf-8');(ROOT/'docs/P3K_ACCEPTANCE.md').write_text(report,encoding='utf-8')
    readme='''# P3k 复现与证据入口

这是开发诊断交付，不是新训练/新评测或模型晋级。工程ACCEPTED，P3整体PARTIALLY_ACCEPTED。

在项目根目录运行。CUDA解释器由项目scripts/run_gpu.ps1记录；正式执行用Python3.14.3。py -3.13仅用于不需要torch的报告/绘图。

```powershell
./scripts/run_gpu.ps1 experiments/p3k_run.py --freeze
./scripts/run_gpu.ps1 experiments/p3k_run.py --run
./scripts/run_gpu.ps1 experiments/p3k_audit.py
py -3.13 experiments/figures_p3k.py
py -3.13 experiments/report_p3k.py
py -3.13 experiments/deliver_p3k.py --verify
```

冻结/计算/审计脚本使用独占创建，拒绝覆盖既有正式产物。上面前三条是生成该交付的历史顺序，不能在已有目录直接重跑；完整复现需把同一源码及所有哈希绑定输入复制到新的隔离项目根目录，保持相对路径，并让artifacts/evaluations/p3k-gradients-v1为空。无需改原协议或覆盖现有结果。图表后需人工视觉检查并生成visual-qa.json，之后才能生成报告。最终delivery --verify为当前项目可直接执行的只读入口。

全套测试历史命令：RUN_P3F_CUDA=1、RUN_P3G_CUDA=1，PYTHONPATH包含项目根/src/experiments，使用CUDA解释器运行scripts/validate.py；197项通过。开关仅作用于该进程环境。

数据：selection.json与probes.jsonl；汇总：outputs/metrics.csv、outputs/tables/source-model-summary.csv；图：outputs/figures/gradient-map.png/svg；实验记录：outputs/tables/experiment-log.csv。所有原始输入只读、源快照固定，前一STATUS原字节保存于previous-status.md。

局限：234个来源观察只有182个不同带标签观察；三个教师来源重复；128单候选；跨模型终局标签off-policy；梯度为固定权重逐状态诊断，不是实际batch/Adam历史。本包没有论文或提交审计。
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    with (OUT/'outputs/tables/experiment-log.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['stage','status','evidence','new_training_hands','new_evaluation_games'])
        for stage,evidence in [('freeze','preregistration.json'),('probe','run-receipt.json'),('controller','controller-audit.json'),('negative','controller-negative.json'),('full-tests','full-tests-gpu.log'),('postprocess','postprocess-audit.json')]:w.writerow([stage,'PASS',evidence,0,0])
    (OUT/'status-addendum.md').write_text('''## P3k 固定权重梯度、尺度与动作间隔诊断（2026-09-26）

**ACCEPTED_FIXED_WEIGHT_GRADIENT_DIAGNOSTIC_V1**。9来源×13局×2观察=234观察（182不同带标签观察）、1078完整候选，交叉9权重共2106探针。128单候选保留，centered梯度严格为0；三份教师来源的选定轨迹重复，不能当独立重复。新增训练0手、评测0局。

同一教师来源13个多候选观察上，六个最终模型的0.1倍centered梯度范数中位数为DMC的0.83%–1.24%，cosine随模型/来源变化；固定子集输出跨度缩小。多候选单点的centered比值可达1.655，不能只按中位数无上限增大权重。达到|q|>=.95的5个探针都来自同一个观察，不支持普遍饱和根因。全部为固定权重、off-policy终局标签诊断，不是历史训练梯度或棋力证据。

6318份梯度逐元素独立表达式对照通过，最大误差7.153e-7；控制器独立重演117个来源牌局、全部234观察及2106稠密梯度/评分复核通过。197项全套测试无跳过，9类审计负例及汇总篡改负例全部拒绝，5346行汇总独立复算。9模型权重前后哈希一致。报告、数据、PNG/SVG和源快照见docs/P3K_ACCEPTANCE.md及artifacts/evaluations/p3k-gradients-v1/。

最近主候选棋力结论仍为P3j NOT_ESTABLISHED；模型未晋级，P3整体PARTIALLY_ACCEPTED。205000..205064已查看，9000000..9009999继续封存。

## 下一工作包：P3l 有上限的批次梯度范数配比工程前置

设计见docs/P3L_PROTOCOL.md，DESIGN_READY_NOT_RUN。先实现隔离的normcap目标/训练器与专用检查点并独立验收；d=||gD||、c=||gC||，零范数分支alpha=0，其余alpha=min(1,0.1*d/c)，保持一次Adam步和完整状态/候选分母。原始梯度的一阶界限不代表Adam或棋力保证。

完成稠密梯度、全参数范数、单步Adam、constant逐位一致、跨进程恢复与故障拒绝后，再冻结协议中的600手×两臂×三初始化试验。normcap-314380为预先固定主候选；不选开发最优初始化，不扫权重。拟用206000..206064尚须历史可用性扫描，本轮未使用。

P3l尚未实现或运行；P4桌面端与P6完整比赛继续后置。
''',encoding='utf-8')
    print(json.dumps(dict(status='PASS',summary_rows=len(summary),unique_states=unique)))

if __name__=='__main__':main()
