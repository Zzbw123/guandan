"""Build readable diagnostics and exact acceptance receipt after controller audit."""
from pathlib import Path
import sys,csv,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.p3e_common import read,write,check,digest
from experiments.p5c_diagnostic import JOBS,protected

def pct(a,b):return '不适用' if b==0 else f'{100*a/b:.2f}%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    out=Path(sys.argv[1]).resolve();a=read(out/'controller-audit.json');s=read(out/'summary.json')
    t=read(ROOT/'artifacts/evaluations/p5c-full-tests-gpu-v1.json')
    check(a['status']==t['status']=='PASS' and t['skipped']==0,'acceptance gates')
    for path,h in t['sources'].items():check(digest(ROOT/path)==h,'tested source unchanged')
    check(digest(ROOT/'artifacts/evaluations/p5c-full-tests-gpu-v1.log')==t['log_sha256'],'tested log')
    pre=read(out/'preregistration.json');receipt=read(out/'receipt.json')
    for key in ('inputs','sources'):
        for path,h in pre[key].items():check(digest(ROOT/path)==h,'frozen source/input')
    for name,h in receipt['artifacts'].items():check(digest(out/name)==h,'computed evidence')
    protected()
    train=[];cross=[];blocks=[]
    for job in JOBS:
        r=s['training'][job]
        train.append([job,r['samples'],pct(r.get('positive',0),r['samples']),
            pct(r['current_seat_wins'],r['current_seat_hands']),pct(r['optional_pass'],r['optional']),
            pct(r['finish_miss'],r['finish_opportunity'])])
    for seed in (314510,314511,314512):
        for block in range(4):
            r=s['training'][f'mixed-{seed}/block-{block}']
            blocks.append([seed,block,r['samples'],pct(r.get('positive',0),r['samples']),
                pct(r['current_seat_wins'],r['current_seat_hands']),pct(r['optional_pass'],r['optional'])])
    # Every model has exactly the same selected states and therefore the same opportunity denominator.
    combined={}
    for model in JOBS:
        cells=[r for key,r in s['scoring'].items() if key.endswith('|'+model)]
        values={key:sum(c.get(key,0) for c in cells) for key in set().union(*(set(c) for c in cells))}
        combined[model]=values
        cross.append([model,values['optional'],pct(values['optional_pass'],values['optional']),
            f"{values['pass_margin_sum']/values['pass_margin_count']:+.6f}" if values.get('pass_margin_count') else '不适用',
            values['finish_opportunity'],pct(values['finish_miss'],values['finish_opportunity'])])
    differences=[]
    for seed in (314510,314511,314512):
        x=combined[f'selfplay-{seed}'];y=combined[f'mixed-{seed}']
        differences.append([seed,f"{100*(y['optional_pass']-x['optional_pass'])/x['optional']:+.2f}",
            f"{100*(y['finish_miss']-x['finish_miss'])/x['finish_opportunity']:+.2f}"])
    for filename,group in [('training-summary.csv',s['training']),('score-summary.csv',s['scoring'])]:
        fields=['group']+sorted(set().union(*(set(v) for v in group.values())))
        with (out/filename).open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fields);writer.writeheader()
            for key,v in group.items():writer.writerow(dict(group=key,**v))
    document=f'''# P5c 固定池负结果诊断验收

2026-09-27。**ACCEPTED_POOL_COLLAPSE_DIAGNOSTIC_V1**。P5整体PARTIALLY_ACCEPTED，模型未晋级；greedy主基线保留。

本包在P5b负结果后制定分析口径，只分析既有训练轨迹和六个final权重。新增训练0手、评测0局；没有运行新验证或保留测试。协议P5C_PROTOCOL.md，证据目录artifacts/evaluations/p5c-diagnostic-v1。

## 全量训练分布

六作业共9600手、1600份共用发牌。下表是训练轨迹的描述，不是正式对局评测。

{table(['作业','学习决策数','正标签/学习决策','胜/学习座位-牌局','可选过牌率','出完漏选率'],train)}

正标签率按决策加权；每一局中的多个决策共享团队终局标签。座位-牌局胜率对current座位计数，selfplay同一团队的两个座位相关，mixed也有重复座位，均不是独立牌局样本数。两种分母必须分开解释。动作与终局标签的联合计数见training-summary.csv，不是对候选动作的反事实价值估计。

## mixed角色分组

block0=current/greedy/current/greedy；block1=current/team/greedy/team；block2=current/frozen/team/frozen；block3=current/greedy/frozen/team（从focal相对座位起）。frozen为随机初始化固定副本。

{table(['初始化','block','样本','正标签率','座位-牌局胜率','可选过牌率'],blocks)}

各block每作业400手；角色、访问状态、学习座位数同时变化。不同block的标签/行为差异只能定位后续干预方向，不能隔离纯对手强度或纯样本量的因果效应。完整分期与block计数保存在CSV/JSON，没有按有利结果隐藏分组。

## 相同状态交叉评分

按冻结规则从384次来源牌局（64份共用发牌）得到{a['states']}个来源状态；同一来源状态交叉六个final权重，共{a['independently_scored_rows']}组评分、{a['independently_scored_candidates']}个候选评分值。来源重复不代表统计独立。每手最多取首个current决策、首个可选过牌机会和首个一次出完机会，重合去重；没有机会不补挑。

{table(['模型','可选过牌机会','选择过牌率','平均过牌评分间隔','出完机会','出完漏选率'],cross)}

过牌间隔=max(pass评分)−max(play评分)，正值表示过牌候选评分更高。分数不是校准胜率；这批有目的选取的状态不能代表所有真实对局状态。

{table(['配对初始化','mixed−selfplay过牌率/百分点','mixed−selfplay出完漏选率/百分点'],differences)}

这些差异使用完全相同的状态和机会分母，避免只比较不同访问轨迹的比例。它们描述固定权重的动作排序，不证明过牌导致此前胜率下降，也不能把所有合法过牌定义为错误。按六个来源作业分开的完整结果见score-summary.csv。

## 独立验收与边界

- 诊断入口重演9600手全部决策，逐步核验完整候选、角色、样本归属、状态摘要和终局奖励。
- 控制器从原日志独立重算全部9600手的学习样本、pass/play及正负标签计数；另外独立重演384手、重建全部选择状态与行为分母。没有把这384手独立复查描述成全量行为计数的第二次独立重演。
- 六模型全部选择状态用独立线性层表达式、257分块重算，与生产1024分块比较；最大绝对误差{a['max_abs_error']:.9g}，容差rtol3e-4/atol3e-6，argmax全部相同。所有汇总计数和评分统计独立复算。
- 新增11项测试通过，覆盖强制过牌、单候选、无机会、完整分母、平局、空/非有限输入与12类真实轨迹/评分篡改；全套**{t['tests']}项GPU解释器测试通过、零跳过**。历史冻结源码与交付产物hash未变。
- 运行前冻结源码/协议/输入并保存ZIP，运行结束与审计后复核；输入权重只读。完整分数位于scores.jsonl.gz，逐手计数位于hands.json，状态定位与完整候选索引位于states.json。
- 首次预检在创建正式目录前发现P5b交付回执没有source_sha256字段，修正新入口为读取实际artifact_sha256；原注册源码仍通过P5b preregistration核验。该次未执行统计、训练或评测，失败记录另存preflight-note.json。

本包没有恢复训练、调整奖励、修改规则或测试新配方。最近正式棋力结论仍为P5b NOT_ESTABLISHED；207000..207064已查看，9000000..9009999继续封存。P4桌面端、P6完整比赛后置。
'''
    target=ROOT/'docs/P5C_ACCEPTANCE.md';check(not target.exists(),'fresh acceptance')
    target.write_text(document,encoding='utf-8')
    write(out/'preflight-note.json',dict(stage='before formal directory and any statistics',error="KeyError: source_sha256",
        resolution='P5b delivery has artifact_sha256 only; source verification remains via original preregistration',historical_files_modified=False))
    (out/'README.md').write_text('''# P5c复现

在项目根目录使用已记录CUDA解释器，输出目录必须不存在：

```powershell
./scripts/run_gpu.ps1 experiments/p5c_diagnostic.py artifacts/evaluations/p5c-diagnostic-reproduction-v1
./scripts/run_gpu.ps1 experiments/controller_p5c_audit.py artifacts/evaluations/p5c-diagnostic-reproduction-v1
./scripts/run_gpu.ps1 scripts/p5c_test.py
```

正式证据只读。preregistration.json绑定来源日志/模型/协议/实现；receipt.json绑定计算产物，controller-audit.json记录独立审计，delivery-receipt.json绑定完整交付。重新运行不会产生新发牌或新训练；全部输入仍为P5b训练数据。全项目测试日志及源码回执位于上一级p5c-full-tests-gpu-v1.log/.json。
''',encoding='utf-8')
    status=ROOT/'docs/STATUS.md';old=status.read_text('utf-8')
    start=old.index('当前交付：');end=old.index('\n\n',start)
    headline=f'当前交付：**P5c固定池奖励分布与同状态评分诊断通过：ACCEPTED_POOL_COLLAPSE_DIAGNOSTIC_V1**。重演既有9600手训练、独立重建384手/{a["states"]}来源状态并交叉六权重；新增训练/评测均为0，全套{t["tests"]}项GPU测试零跳过。P5整体PARTIALLY_ACCEPTED，最近正式棋力结论仍为P5b NOT_ESTABLISHED，greedy保留；P4/P6继续后置。'
    old=old[:start]+headline+old[end:]
    old=old.replace('P5a采样工程、P5b固定预算配对研究已验收；尚无已晋级学习模型，后续研究需新协议。','P5a/P5b已验收，P5c奖励/评分诊断已验收；尚无已晋级学习模型，下一训练干预需新协议。')
    marker='## 2026-09-27 P5b固定策略池配对研究验收'
    section=f'''## 2026-09-27 P5c固定池负结果诊断验收

**ACCEPTED_POOL_COLLAPSE_DIAGNOSTIC_V1**。协议P5C_PROTOCOL.md，验收P5C_ACCEPTANCE.md，证据artifacts/evaluations/p5c-diagnostic-v1。全量9600手重演与奖励/角色/动作计数；控制器独立重算所有标签计数、重演384手，重建{a['states']}状态、{a['independently_scored_rows']}组同状态交叉评分及{a['independently_scored_candidates']}完整候选评分，直接forward最大误差{a['max_abs_error']:.9g}、argmax全部一致。新11项及全套{t['tests']}GPU测试零跳过，历史hash未变。

本次为P5b负结果触发的描述性机制诊断，不是新棋力评测或因果证明；新增训练0手、评测0局。按决策的正标签比例与按current座位-牌局胜率分别报告，六权重在相同机会分母上比较过牌/出完排序。完整分组结果和边界见验收文档。下一步依据诊断另行冻结受控训练干预，不追加P5b预算；模型未晋级，保留测试继续封存。

'''
    status.write_text(old.replace(marker,section+marker,1),encoding='utf-8')
    artifacts={p.relative_to(ROOT).as_posix():digest(p) for p in out.iterdir() if p.is_file()}
    for path in ['docs/P5C_ACCEPTANCE.md','docs/P5C_PROTOCOL.md','docs/STATUS.md',
        'artifacts/evaluations/p5c-full-tests-gpu-v1.log','artifacts/evaluations/p5c-full-tests-gpu-v1.json','experiments/deliver_p5c.py']:
        artifacts[path]=digest(ROOT/path)
    write(out/'delivery-receipt.json',dict(status='PASS',scope=a['scope'],model_promoted=False,full_gpu_tests=t['tests'],artifact_sha256=artifacts))
    print(document)

if __name__=='__main__':main()
