"""Gate and bind P5a delivery after the controller audit and full GPU suite."""
from pathlib import Path
from hashlib import sha256
import json
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/evaluations'


def digest(p): return sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text('utf-8'))
def check(ok,label):
    if not ok: raise ValueError(label)


def main():
    folder=BASE/'p5a-pool-v2'
    check(not (folder/'delivery-receipt.json').exists(),'fresh delivery')
    audit_path=BASE/'p5a-controller-audit-v2.json'; audit=read(audit_path)
    check(audit['status']=='PASS' and audit['training_hands']==16 and audit['fresh_process_resume']=='BITWISE_EQUAL','controller acceptance')
    check(audit['controller_sha256']==digest(ROOT/'experiments/controller_p5a_audit.py'),'controller source')
    for path,h in {**audit['source_sha256'],**audit['artifact_sha256']}.items():
        check(digest(ROOT/path)==h,'bound source or data '+path)
    log=BASE/'p5a-full-tests-gpu-v1.log'; text=log.read_text('utf-8-sig')
    check(re.search(r'Ran 230 tests in ',text) and text.rstrip().endswith('OK') and 'skipped=' not in text,'full GPU suite')
    with zipfile.ZipFile(folder/'source-snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in audit['source_sha256']: z.write(ROOT/path,path)
        for path in ('experiments/controller_p5a_audit.py','experiments/controller_p5a_delivery.py'):
            z.write(ROOT/path,path)
    with zipfile.ZipFile(folder/'source-snapshot.zip') as z:
        for path,h in audit['source_sha256'].items(): check(sha256(z.read(path)).hexdigest()==h,'source archive '+path)
    acceptance='''# P5a 固定策略池与异质角色采样工程验收

日期：2026-09-27。总控结论：**ACCEPTED_FIXED_POOL_SAMPLING_V1**。P5整体仍为 **PARTIALLY_ACCEPTED**；棋力 **NOT_ESTABLISHED**，model_promoted=false，主基线仍为greedy-v1。

## 实施范围

遵循P5A_PROTOCOL.md，保留规则、编码、网络和原DMC MSE。新增隔离训练器、专用检查点及运行入口。固定池含current、frozen、greedy、team；frozen是本次随机初始化的固定副本，不是已训练强对手。16手日程覆盖四种相对角色组合和四个focal座位；只有current实际选择的动作进入训练样本。所有策略输入仅为PlayerObservation与完整合法候选。

## 实际预算与独立审计

验收主目录为artifacts/evaluations/p5a-pool-v2；新进程恢复目录为p5a-pool-resume-v2。初始化314500、开发种子108800..108815，16手、4个wave。所有牌谱逐步重演，角色、发牌来源、策略决策、探索随机序列、完整候选数、团队终局标签、样本分母和打乱后的RNG均独立核对。

| 项目 | 本次证据 |
|---|---|
| 完成主工程运行 | 16/16手，1575次决策，526条current学习样本 |
| CUDA更新 | 4次，每wave恰好一个批次；样本数185/113/132/96 |
| current非探索评分 | 471次请求，15463个完整候选 |
| frozen评分 | 347次请求，3915个完整候选 |
| 非current决策 | 1049次，仅保留牌谱/审计日志，不进入学习样本 |
| 跨进程恢复 | wave2恢复到wave4；新增执行8手与主运行后8手重复，不增加独立发牌数 |
| 精确恢复比较 | wave3/4完整日志、模型、Adam、策略RNG、CPU/CUDA RNG、计数、发牌历史、冻结池全部逐位一致 |
| 冻结池完整性 | 初始化身份与权重来源独立核对；各wave前后不变，无训练梯度 |

总控审计没有调用生产train_wave、lineup或score_many；直接按固定日程从seed重建环境，并用独立全候选forward重算神经策略选择。基线选择复用已冻结的基线实现；规则与编码仍使用项目已验收实现，不宣称重新独立实现规则或编码。

四次更新均用独立网络表达式、平方误差求和/批次分母、反向梯度与Adam重算。每wave仅一次更新，可从前后检查点的Adam一阶动量反推出实际施加的梯度；共核对1642500个梯度元素，最大绝对误差1.192093e-7。四个wave末模型与Adam最大绝对误差均为0，平均损失一致。数值检查容差rtol=3e-4、atol=3e-6；这是本包数值审计容差，不是棋力门槛。

## 测试与拒绝路径

真实RTX5060 Laptop、PyTorch2.12.0+cu130上，新增15项GPU/合同测试通过，全项目 **230项测试通过、零跳过**。

- selfplay模式两wave与旧GPUTrainer模型/Adam/全部RNG/计数/损失/评分计数逐位一致。
- current和frozen分别在8769候选夹具上，以chunk37及1024完整评分，与稠密forward一致；不截断候选。
- 策略入参边界、learner样本归属、角色轮换、开发seed限制、非法基线返回、冻结权重变更、检查点覆盖/哈希/语义篡改拒绝通过。
- Adam已更新后注入异常，训练器拒绝继续和保存；重载之前检查点重做与正常路径逐位一致。磁盘恢复拒绝保留全局CPU/CUDA RNG。
- 独立审计器拒绝12类改动：角色、候选分母、样本归属、选择索引、探索标记、团队标签、发牌seed、样本数、评分计数、更新计数、策略池身份、最终策略RNG。

历史12份注册与654项不可变交付产物哈希一致；另核对P3l的196项注册源和P3全阶段收口绑定的120项证据。此项复用历史验收并复核哈希，不声称本轮重演全部历史实验。

## 保留的失败与修复

首个p5a-pool-v1已完成同16手，但p5a-pool-resume-v1新进程尚未设置确定性CUDA运行参数，严格runtime校验拒绝加载。修复仅为运行入口在恢复前设置已有确定性参数，并增加真实新进程回归测试；随后按原预算在v2新目录重跑。两次主运行共32手执行但只有16份发牌，不能计成32份独立数据。首轮数据、失败记录及开发测试v1/v2/v3均保留。

首个控制器审计已通过四wave与12类负例，末尾生成文件索引时相对路径未resolve导致报告写出失败；修复控制器路径处理后完整重审，成功报告为p5a-controller-audit-v2.json。未改变已完成训练、牌谱、预算或协议。

## 完成边界和下一步

本包通过固定策略池采样、保存/恢复和独立审计工程。训练仍为可信本地同步wave，基线调用没有新增逐调用硬隔离；不是无限时长稳定性、跨硬件逐位恢复、动态模型池、外部权重导入或完整比赛能力验收。

正式新验证0局，保留测试9000000..9009999未使用；没有产生可用于晋级的棋力结论。P5下一包为P5b固定预算selfplay与mixed配对试验：须先冻结初始化、预算、日程、唯一主候选、主比较与验证seed可用性，再开始。两模式样本数/更新数可能不同，必须明确是等手数还是等更新预算；不默认等计算量。>=55%且95%成组区间下界>50%的绝对门槛保持不变。P4桌面端、P6完整比赛继续后置。

复现命令见artifacts/evaluations/p5a-pool-v2/README.md。delivery-receipt.json绑定协议、测试日志、审计、源码ZIP、数据与本文；STATUS.md为可继续更新的当前状态入口。
'''
    acceptance_path=ROOT/'docs/P5A_ACCEPTANCE.md'
    check(not acceptance_path.exists(),'fresh acceptance document')
    acceptance_path.write_text(acceptance,encoding='utf-8')
    readme='''# P5a 工程验收复现入口

在项目根目录使用PowerShell；所有输出路径必须全新，已有工程数据不覆盖。

```powershell
./scripts/run_gpu.ps1 scripts/p5a_test.py
$env:RUN_P3F_CUDA='1'
$env:RUN_P3G_CUDA='1'
./scripts/run_gpu.ps1 scripts/validate.py
./scripts/run_gpu.ps1 scripts/p5a_run.py artifacts/evaluations/p5a-pool-rerun-NEW
./scripts/run_gpu.ps1 scripts/p5a_run.py artifacts/evaluations/p5a-pool-resume-NEW --resume artifacts/evaluations/p5a-pool-rerun-NEW/wave-2
./scripts/run_gpu.ps1 experiments/controller_p5a_audit.py artifacts/evaluations/p5a-pool-rerun-NEW artifacts/evaluations/p5a-pool-resume-NEW artifacts/evaluations/p5a-audit-NEW.json
```

只复核本次已保存结果时：

```powershell
./scripts/run_gpu.ps1 experiments/controller_p5a_audit.py artifacts/evaluations/p5a-pool-v2 artifacts/evaluations/p5a-pool-resume-v2 artifacts/evaluations/p5a-audit-recheck-NEW.json
```

成功验收：docs/P5A_ACCEPTANCE.md；控制器报告：artifacts/evaluations/p5a-controller-audit-v2.json；全套测试：artifacts/evaluations/p5a-full-tests-gpu-v1.log。source-snapshot.zip保存绑定的源文件与控制器，恢复要求相同源/运行时；不要在当前项目覆盖历史源来绕过检查。未证明干净机器重建或跨硬件恢复。

16手只验证工程；恢复执行、重跑和单测均不增加独立发牌数，不代表棋力改善。最新主候选结论仍为P3l NOT_ESTABLISHED，保留测试继续封存。
'''
    (folder/'README.md').write_text(readme,encoding='utf-8')
    status_path=ROOT/'docs/STATUS.md'; status=status_path.read_text('utf-8')
    lines=status.splitlines()
    lines[4]='当前交付：**P3学习闭环工程完成；新增P5a固定策略池/异质角色采样工程验收通过：ACCEPTED_FIXED_POOL_SAMPLING_V1**。P5仍PARTIALLY_ACCEPTED。本轮16手、526条学习样本、4次CUDA更新，跨进程恢复与独立审计通过，全套230项GPU测试无跳过。未运行新棋力验证；最近主候选仍为P3l对greedy 28.85% [25.38%,32.31%]，NOT_ESTABLISHED，未晋级。P4桌面端和P6完整比赛继续后置。'
    status='\n'.join(lines)+'\n'
    status=re.sub(r'\| P5 棋力增强 \|[^\n]+', '| P5 棋力增强 | PARTIALLY_ACCEPTED | P5a固定策略池采样/恢复/审计工程通过；下一包P5b需预注册正式配对试验。棋力未建立、模型未晋级。 |',status,count=1)
    start=status.index('## 2026-09-27 当前新增工作：P5a')
    end=status.index('## P1 本机验收',start)
    summary='''## 2026-09-27 P5a固定策略池工程验收

**ACCEPTED_FIXED_POOL_SAMPLING_V1**。协议与完整验收见P5A_PROTOCOL.md/P5A_ACCEPTANCE.md，主证据目录artifacts/evaluations/p5a-pool-v2，恢复证据p5a-pool-resume-v2。

固定current/frozen/greedy/team池与16手角色轮换，只有current决策进入学习。16手、1575决策、526样本、4次CUDA更新；current非探索15463候选、frozen3915候选全部评分。frozen为本次初始化固定副本，未声称强对手。总控完整重演、seed/角色/策略选择/样本归属/标签/探索与打乱RNG通过；四次更新独立梯度和Adam复算，1642500梯度元素最大绝对误差1.192093e-7，模型与Adam终态误差0。

wave2新进程恢复到wave4，完整日志与模型/Adam/RNG/计数逐位一致。新增15项及全套230项GPU测试零跳过，含8769候选、旧selfplay逐位一致、中途Adam失败后拒绝/重载、检查点负例；12类独立审计篡改均拒绝。历史12份注册/654项产物、P3l196项源和P3收口120项证据hash保持一致。

v1恢复入口缺少确定性runtime初始化而被拒绝，修复后以相同预算在v2重跑，原数据和失败证据保留；两次主运行仅16份发牌，不计为32份独立样本。首版审计报告路径处理失败也保留，修复后完整重审。

P5整体PARTIALLY_ACCEPTED。下一步单独设计并预注册P5b等预算selfplay/mixed配对研究；明确等手数与不同样本/更新数边界。新增正式验证0局、未晋级，保留测试9000000..9009999继续封存。

'''
    status_path.write_text(status[:start]+summary+status[end:],encoding='utf-8')
    paths=list(folder.rglob('*'))+list((BASE/'p5a-pool-resume-v2').rglob('*'))
    paths += [p for p in BASE.glob('p5a*') if p.is_file()]
    paths += [ROOT/'docs/P5A_PROTOCOL.md',acceptance_path,status_path,Path(__file__),ROOT/'experiments/controller_p5a_audit.py']
    receipt=dict(status='PASS',scope='ACCEPTED_FIXED_POOL_SAMPLING_V1',full_gpu_tests=230,skipped=0,
                 model_promoted=False,strength='NOT_ESTABLISHED',reserved_test_executed=False,
                 artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths if p.is_file()},
                 source_sha256=audit['source_sha256'])
    with (folder/'delivery-receipt.json').open('x',encoding='utf-8') as f: json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status='PASS',bound_files=len(receipt['artifact_sha256']),source_files=len(audit['source_sha256']))))


if __name__=='__main__': main()
