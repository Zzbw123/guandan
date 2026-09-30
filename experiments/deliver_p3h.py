"""Finalize the independently reviewed P3h package after actual figure inspection."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import zipfile
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1]

def digest(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text('utf-8'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    out=Path(sys.argv[1]).resolve();a=read(out/'controller-audit.json');f=read(out/'fidelity/controller-audit.json')
    for p in (out/'controller-audit.json',out/'controller-negative.json',out/'fidelity/controller-audit.json',
              out/'historical-source-protection.json',out/'batch-order-audit.json',out/'descriptives/visual-qa.json'):
        assert read(p)['status']=='PASS',p
    assert (out/'report.md').read_bytes()==(ROOT/'docs/P3H_ACCEPTANCE.md').read_bytes()
    pre=read(out/'preregistration.json')
    tests=read(ROOT/'artifacts/evaluations/p3h-test-receipt.json')
    assert tests['status']=='PASS' and tests['skipped']==0
    assert digest(ROOT/'artifacts/evaluations/p3h-tests.log')==tests['log_sha256']
    for key in ('source_sha256','input_sha256'):
        for name,h in pre[key].items():assert digest(ROOT/name)==h,name
    previous=read(ROOT/'artifacts/evaluations/p3g-fit-v1/delivery-receipt.json')
    assert digest(out/'previous-status.md')==previous['artifact_sha256']['docs/STATUS.md']
    readme=ROOT/'README.md';assert digest(readme)==previous['artifact_sha256']['README.md']
    with (out/'previous-readme.md').open('xb') as h:h.write(readme.read_bytes())
    m=a['evaluations']['validation-aux']['summary']['dmc|dmc|greedy']['metrics']['win_rate']
    win=f'{100*m["estimate"]:.2f}% [{100*m["ci95"][0]:.2f}%, {100*m["ci95"][1]:.2f}%]'
    hands=sum(v['hands'] for v in a['training'].values());samples=sum(v['samples'] for v in a['training'].values())
    updates=sum(v['updates'] for v in a['training'].values());games=sum(v['games'] for v in a['evaluations'].values())
    next_text=(out/'next-package.md').read_text('utf-8')
    status=ROOT/'docs/STATUS.md';text=status.read_text('utf-8');lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('当前交付：'):
            lines[i]=f'当前交付：**P0–P2及P3a–P3h在各自范围内验收通过**。P3h完成固定教师偏好辅助损失配对试验，新增{hands}手训练和{games}局评测，完整磁盘复算通过。固定aux候选对greedy新验证为{win}、{a["validation_gate"]}。P3整体仍为PARTIALLY_ACCEPTED，模型未晋级；桌面端后置。'
        elif line.startswith('| P3 学习闭环 |'):
            lines[i]=f'| P3 学习闭环 | PARTIALLY_ACCEPTED | P3h辅助损失配对试验工程通过；主候选新验证{win}、{a["validation_gate"]}；下一包见文末。 |'
    text='\n'.join(lines)+'\n';marker='## 下一工作包：P3h 教师偏好辅助损失的单条件配对试验'
    assert text.count(marker)==1;text=text.split(marker)[0]
    text+=f'''## P3h 教师偏好辅助损失单条件配对试验（2026-09-26）

**{a['engineering']}**；唯一主候选新验证 **{a['validation_gate']}**。正式证据目录`artifacts/evaluations/p3h-aux-v1/`；协议与完整验收见`docs/P3H_PROTOCOL.md`、`docs/P3H_ACCEPTANCE.md`。

从P3f三个teacher前200手权重开始，两臂分别使用原DMC与DMC+0.1倍完整候选教师偏好MSE，统一重置Adam与全部RNG；每臂每初始化续训600手，共{hands}手、{samples}观察、{updates}次CUDA更新。对应600个既有开发训练发牌seed，重复使用不增加独立发牌数。完整候选保留；每观察候选平均后观察等权，每批只更新一次Adam。

| 初始化 | control开发胜率 | aux开发胜率 | 配对差值/百分点 | 95%描述区间/百分点 |
|---|---:|---:|---:|---:|
'''
    for r in a['paired']:
        if not r['validation']:
            lo,hi=r['ci95_delta_pp'];text+=f'| {r["seed"]} | {100*r["control"]:.2f}% | {100*r["aux"]:.2f}% | {r["delta_pp"]:+.2f} | [{lo:+.2f},{hi:+.2f}] |\n'
    text+=f'''
开发同26个原始发牌、8变体/组；条件于初始化的点态区间，非新独立验证。训练前唯一指定aux-314380主候选，在新验证seed204000..204064共65组对greedy为 **{win}**；同发牌control及random/team对照详见验收。共{games}局评测，零失败/非法动作/超时，全部完整重演与独立统计通过。未达绝对门槛不能用相对弱control的提升代替。

三份control训练600手后的模型/Adam/RNG/计数与P3f原DMC逐位一致。新增8项CUDA/边界检查、9类审计负例通过；固定128状态/{f['candidates']}候选的六模型教师偏好诊断及全部768组direct-forward独立检查通过。训练数值检查不等于逐更新重算aux所有梯度。旧源保持原hash；专用gd-p3h检查点明确辅助权重和600手历史，继承200手需外部谱系。共享GPU负载使墙钟耗时不可用于算法速度比较。

204000..204064现在属于已查看验证；9000000..9009999继续封存。模型未晋级，贪心主基线保留，P4桌面端与P6完整比赛仍未完成。

'''+next_text+'\n'
    status.write_text(text,encoding='utf-8',newline='\n')
    with readme.open('a',encoding='utf-8',newline='\n') as h:h.write(f'''

2026-09-26 **P3h固定教师偏好辅助损失配对试验已在工程范围验收**。新增{hands}手训练、{games}局评测；主候选对greedy新验证{win}，结论{a['validation_gate']}，模型未晋级。完整结果、辅助机制诊断和限制见 [P3H_ACCEPTANCE.md](docs/P3H_ACCEPTANCE.md)，固定规程见 [P3H_PROTOCOL.md](docs/P3H_PROTOCOL.md)。训练/评测磁盘重演、独立成组区间、三个control逐位复现和新检查点/推理验证通过。

```powershell
# 独立磁盘重演与统计复算；不会重复训练或使用新验证
py -3.13 experiments/review_p3h.py artifacts/evaluations/p3h-aux-v1
# 固定128个既有开发观察的完整CUDA直接forward复核
./scripts/run_gpu.ps1 experiments/inspect_p3h.py artifacts/evaluations/p3h-aux-v1 --verify
```

P3整体仍为PARTIALLY_ACCEPTED；下一研究包见 [STATUS.md](docs/STATUS.md)。最终界面为桌面端，P4后置。
''')
    controller_sources=sorted(set(list((ROOT/'experiments').glob('*p3h*.py'))+
        list((ROOT/'scripts').glob('p3h*.py'))+list((ROOT/'tests').glob('test_p3h*.py'))))
    with zipfile.ZipFile(out/'controller-source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in controller_sources:archive.write(p,p.relative_to(ROOT).as_posix())
    files=[p for p in out.rglob('*') if p.is_file()]
    files += [ROOT/'docs/STATUS.md',ROOT/'README.md',ROOT/'docs/P3H_PROTOCOL.md',ROOT/'docs/P3H_ACCEPTANCE.md']
    files += list((ROOT/'experiments').glob('*p3h*.py'))+list((ROOT/'scripts').glob('p3h*.py'))+list((ROOT/'tests').glob('test_p3h*.py'))
    files += [ROOT/'artifacts/evaluations/p3h-tests.log',ROOT/'artifacts/evaluations/p3h-test-receipt.json',ROOT/'artifacts/evaluations/p3h-seed-availability.json']
    write(out/'delivery-receipt.json',dict(status='SCOPED_ACCEPTANCE_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
        engineering=a['engineering'],overall_P3='PARTIALLY_ACCEPTED',validation_gate=a['validation_gate'],model_promoted=False,
        reserved_test_executed=False,artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(files))}))
    print(json.dumps(dict(status='SCOPED_ACCEPTANCE_COMPLETE',files=len(set(files)),validation=win)))

if __name__=='__main__':main()
