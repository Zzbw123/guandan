"""Bind independently accepted P5d evidence and update the mutable status index."""
from pathlib import Path
import sys,json,csv,re
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.p5d_package import read,write,digest,protected


def main():
    out=Path(sys.argv[1]).resolve();test_path=Path(sys.argv[2]).resolve()
    audit=read(out/'controller-audit.json');tests=read(test_path);pre=read(out/'preregistration.json')
    assert audit['status']==tests['status']=='PASS' and tests['skipped']==0
    assert audit['scope']=='ACCEPTED_LABEL_BALANCED_ENGINEERING_V1'
    assert audit['fresh_process_resume']==audit['ordinary_p5a_compatibility']=='BITWISE_EQUAL'
    assert digest(test_path.with_suffix('.log'))==tests['log_sha256']
    for mapping in (tests['sources'],audit['source_sha256'],audit['artifact_sha256'],pre['inputs']):
        for name,h in mapping.items():assert digest(ROOT/name)==h,name
    protected()
    assert not (out/'delivery-receipt.json').exists()
    assert not (ROOT/'docs/P5D_ACCEPTANCE.md').exists()
    rows=[];summary={}
    for objective in ('ordinary','label_balanced'):
        values=[read(out/objective/f'wave-{w}.json') for w in range(1,5)]
        for value in values:
            for batch in value['batches']:rows.append(dict(arm=objective,wave=value['wave'],**batch))
        summary[objective]=dict(hands=16,samples=sum(x['samples'] for x in values),updates=values[-1]['updates'],
            decisions=sum(h['steps'] for x in values for h in x['hands']),
            current_candidates=sum(x['scored_candidates'] for x in values),
            frozen_candidates=sum(x['frozen_candidates'] for x in values),
            positive=sum(b['positive'] for x in values for b in x['batches']),
            negative=sum(b['negative'] for x in values for b in x['batches']),
            single_class_batches=sum(b['single_class'] for x in values for b in x['batches']))
    with (out/'batch-summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'summary.json',summary)
    numerical=[w['numerical'] for values in audit['waves'].values() for w in values]
    gradient_elements=sum(x['gradient_elements_checked'] for x in numerical)
    gradient_error=max(x['max_gradient_abs_error'] for x in numerical)
    model_error=max(x['max_model_abs_error'] for x in numerical)
    adam_error=max(x['max_adam_abs_error'] for x in numerical)
    table='\n'.join('| '+name+' | '+' | '.join(str(v[k]) for k in ('hands','samples','updates','positive','negative','single_class_batches','current_candidates','frozen_candidates'))+' |' for name,v in summary.items())
    first_log=ROOT/'artifacts/evaluations/p5d-full-tests-gpu-v1.log'
    failed_note=''
    if test_path.stem!='p5d-full-tests-gpu-v1' and first_log.exists():
        failed_note='首轮完整测试日志保留为p5d-full-tests-gpu-v1.log：测试期间新增目标源码发生最终字面调整，来源一致性检查未通过，未据此签发验收；锁定源码后重新运行完整测试，以当前回执为准。'
    doc=f'''# P5d 标签均衡目标 GPU 工程验收

2026-09-29。总控结论：**ACCEPTED_LABEL_BALANCED_ENGINEERING_V1**。P5整体PARTIALLY_ACCEPTED；model_promoted=false，greedy主基线保留。最近正式棋力结论仍为P5b NOT_ESTABLISHED。

## 改动与已冻结范围

遵循P5D_PROTOCOL.md，在独立P5d训练器中新增ordinary/label_balanced目标及专用检查点。双类批次的正负标签各占损失一半，单类批次保留原MSE；终局奖励仍为±1，完整合法候选、角色、采样顺序和信息边界保持原合同。原P5a–P5c源码和证据保持原哈希。

损失重加权改变拟合目标，不是原期望回报的无偏修正，也不产生缺失的成功轨迹。本次不判断它能否修复过牌倾向或提升棋力。

## 固定预算与实测结果

初始化314500；两臂各16手、共32次主训练执行，但仅16份共用开发发牌108800..108815。label_balanced从wave2在新进程恢复到wave4，额外8手重复执行，不增加独立发牌。测试中的额外开发对局与正式工程预算分开计数。没有新开发对局评测、正式验证或保留测试。

| 臂 | 手数 | 学习样本 | CUDA更新 | 正标签 | 负标签 | 单类批次 | current完整候选评分 | frozen完整候选评分 |
|---|---|---|---|---|---|---|---|---|
{table}

按批次的真实标签数、分母、权重、损失见batch-summary.csv。均衡只在单个实际minibatch内定义，wave日志按样本数汇总不同更新时刻的损失，不能当作固定模型全wave损失。两臂训练路径可随参数分化，样本/候选数不同不构成工程错误，也不代表棋力差。

## 总控独立验收

全部32手从seed独立重建并逐步核对：角色、完整候选、动作选择、探索和打乱RNG、学习样本归属、团队终局标签、计数与冻结池。神经策略用独立线性层表达式及257分块重新评分；未调用生产train_wave、目标函数或score_many进行这些核验。规则/编码及固定基线仍复用已验收实现。

全部{sum(x['batches'] for x in numerical)}批次用独立每样本权重表达式重算损失、梯度及Adam。检查{gradient_elements}个可从单步Adam动量反推的梯度元素；最大梯度绝对误差{gradient_error:.9g}，最大模型误差{model_error:.9g}，最大Adam误差{adam_error:.9g}；容差rtol3e-4/atol3e-6。每批损失及wave聚合独立复算通过。

ordinary的initial及四wave模型、Adam、策略/CPU/CUDA RNG、计数、冻结池与既有P5a-v2逐位一致，旧wave日志字段也完全一致。label_balanced跨进程恢复后的完整checkpoint payload与wave3/4日志逐位一致。比较的是完整张量/元数据，不依赖torch序列化文件字节相同。

独立审计拒绝{len(audit['negative_rejections'])}类篡改，包括角色、候选分母、样本归属、动作、探索、奖励、seed、计数、池身份、RNG，以及目标、批次数、标签计数、权重、分母、单类标记、起点、损失与同时修改批次/总损失。完整拒绝证据见controller-audit.json。

GPU全套 **{tests['tests']}项测试通过、零跳过**，使用记录的RTX5060 Laptop / PyTorch2.12.0+cu130。新增14项测试覆盖目标梯度、单类/单例、符号翻转、输入拒绝、短尾批、旧路径兼容、Adam已更新后异常封锁与重载、checkpoint语义篡改及失败加载RNG保持，以及独立审计元数据拒绝。工程预算只覆盖同机同运行时，未声称长时训练或跨硬件恢复。

正式运行前冻结协议/源码/历史输入并保存ZIP，运行后、独立审计后和交付前哈希复核通过。既有P5c交付绑定中STATUS.md为允许更新的状态入口，其余冻结证据不改动。

## 修复记录与下一步

正式运行前审查修复ordinary分支的日志权重：实际MSE本来正确，但双类日志误写了均衡权重；现已与实际使用的1/1权重一致，并加入直接断言。非法输入测试已消除共享标签张量被原地修改而掩盖拒绝原因的问题。

{failed_note}

P5d是后续研究的工程前置，没有新的棋力效果估计。下一步P5e应另行预注册ordinary与label_balanced的等发牌配对研究，固定多初始化、预算、唯一主候选、新验证seed与完整审计；不追加P5b预算、不按这16手选模型。绝对晋级门槛仍为对greedy胜率>=55%且95%成组区间下界>50%，不能用相对弱对照改善替代。已查看验证范围保持原标签，9000000..9009999继续封存。P4桌面端、P6完整比赛后置。
'''
    (ROOT/'docs/P5D_ACCEPTANCE.md').write_text(doc,encoding='utf-8')
    readme='''# P5d 工程复现

项目根目录PowerShell，使用记录的CUDA解释器；所有输出目录/日志路径必须全新：

```powershell
./scripts/run_gpu.ps1 scripts/p5d_test.py
./scripts/run_gpu.ps1 scripts/p5d_full_test.py artifacts/evaluations/p5d-full-tests-NEW.log
./scripts/run_gpu.ps1 experiments/p5d_package.py artifacts/evaluations/p5d-balanced-NEW
./scripts/run_gpu.ps1 experiments/controller_p5d_audit.py artifacts/evaluations/p5d-balanced-NEW
```

只复查正式证据时，调用controller_p5d_audit.py并用--output指定全新JSON路径即可，不重训。preregistration.json绑定协议/源码/输入，source-snapshot.zip保存源码，run-receipt.json绑定运行产物；controller-audit.json记录独立重算，delivery-receipt.json绑定最终交付。恢复要求相同源码和运行时，不能修改历史文件绕过来源校验。

两臂共32手仅16份发牌，恢复重复8手，均为工程检查；没有棋力验证或模型晋级。长期研究另立协议。
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    status=ROOT/'docs/STATUS.md';text=status.read_text('utf-8')
    text=text.replace('更新时间：2026-09-27。','更新时间：2026-09-29。',1)
    headline=f'当前交付：**P5d标签均衡损失GPU工程通过：ACCEPTED_LABEL_BALANCED_ENGINEERING_V1**。两臂共32手、16份共用发牌，完整重演/逐批数值审计、ordinary历史逐位兼容、新进程恢复通过；全套{tests["tests"]}项GPU测试零跳过。P5整体PARTIALLY_ACCEPTED，最近正式棋力结论仍为P5b NOT_ESTABLISHED，greedy保留；P4/P6继续后置。'
    text=re.sub(r'当前交付：[^\n]+',headline,text,count=1)
    text=re.sub(r'\| P5 棋力增强 \|[^\n]+','| P5 棋力增强 | PARTIALLY_ACCEPTED | P5a/P5b/P5c已验收，P5d标签均衡损失工程已验收；下一步P5e须预注册配对研究，尚无已晋级模型。 |',text,count=1)
    section=f'''## 2026-09-29 P5d标签均衡目标工程验收

**ACCEPTED_LABEL_BALANCED_ENGINEERING_V1**。协议P5D_PROTOCOL.md，验收P5D_ACCEPTANCE.md，证据artifacts/evaluations/{out.name}。ordinary/label_balanced各16手，共32次执行、16份共用开发发牌；均衡臂wave2新进程恢复重复8手，不增加独立样本。双类批次各占一半损失，单类保留MSE，原奖励/规则/完整候选不变。

全部32手由总控独立重建、逐步复算动作与RNG；全部批次独立梯度/Adam数值审计通过。ordinary与P5a-v2四wave模型/Adam/RNG/旧日志逐位一致，均衡臂跨进程完整恢复逐位一致；{len(audit['negative_rejections'])}类审计篡改拒绝，全套{tests['tests']}GPU测试零跳过，历史冻结源/不可变产物hash保持一致。

本包未作棋力评测，不证明标签不平衡是过牌或负结果原因。P5仍PARTIALLY_ACCEPTED，model_promoted=false，最近正式结果仍P5b NOT_ESTABLISHED。下一步P5e另行冻结等发牌ordinary/label_balanced配对研究，不追加旧实验预算；保留测试继续封存，P4/P6后置。

'''
    text=text.replace('## 2026-09-27 P5c固定池负结果诊断验收',section+'## 2026-09-27 P5c固定池负结果诊断验收',1)
    status.write_text(text,encoding='utf-8')
    paths=[p for p in out.rglob('*') if p.is_file()]
    paths+=[ROOT/'docs/P5D_PROTOCOL.md',ROOT/'docs/P5D_ACCEPTANCE.md',status,
            test_path,test_path.with_suffix('.log'),Path(__file__)]
    if failed_note:paths.append(first_log)
    write(out/'delivery-receipt.json',dict(status='PASS',scope=audit['scope'],full_gpu_tests=tests['tests'],
        skipped=0,model_promoted=False,strength='NOT_ESTABLISHED',validation_games=0,
        artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths}))
    print(json.dumps(dict(status='PASS',summary=summary,tests=tests['tests'],negative_rejections=len(audit['negative_rejections']))))


if __name__=='__main__':main()
