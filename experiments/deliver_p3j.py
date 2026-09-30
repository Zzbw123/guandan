"""Finalize or verify P3j only after controller, figures, and full-suite evidence."""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import json,sys,zipfile,re
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text('utf-8'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def check(v,m):
    if not v:raise ValueError(m)

def protection():
    dirs=['p2-validation-v1','p3a-cpu-v1','p3b-gpu-v1','p3c-gpu-wave-v1','p3d-validation-v2',
          'p3e-scale-v2','p3f-teacher-v1','p3g-fit-v1','p3h-aux-v1','p3i-centered-v1']
    values={}
    for name in dirs:
        pre=read(ROOT/f'artifacts/evaluations/{name}/preregistration.json')
        hashes=pre.get('source_sha256',pre.get('source_sha256_before'))
        for p,h in hashes.items():check(digest(ROOT/p)==h,'historical source '+p)
        values[name]=len(hashes)
    return dict(status='PASS',source_counts=values)

def validate(out):
    a=read(out/'controller-audit.json')
    for file in ('controller-audit.json','controller-negative.json','batch-order-audit.json',
                 'fidelity/controller-audit.json','fidelity/decomposition-audit.json','descriptives/visual-qa.json'):
        check(read(out/file)['status']=='PASS','prerequisite '+file)
    for seed in (314380,314381,314382):check(a['training'][f'absolute-{seed}']['old_control_bitwise_equal'] is True,'full absolute reproducibility')
    pre=read(out/'preregistration.json');receipt=read(out/'receipt.json')
    for field in ('source_sha256','input_sha256'):
        for p,h in pre[field].items():check(digest(ROOT/p)==h,'frozen file '+p)
    for p,h in receipt['artifact_sha256'].items():check(digest(out/p)==h,'completed run file '+p)
    tests=read(ROOT/'artifacts/evaluations/p3j-test-receipt.json')
    check(tests['status']=='PASS' and tests['total']==15 and tests['skipped']==0,'specialized tests')
    for p,h in tests['source_sha256'].items():check(digest(ROOT/p)==h,'tested source')
    check(tests['log_sha256']==digest(ROOT/'artifacts/evaluations/p3j-tests.log'),'specialized test log')
    full=(out/'full-tests-gpu.log').read_text('utf-8-sig')
    check('Ran 188 tests' in full and re.search(r'\nOK\s*$',full) and 'skipped ' not in full and 'FAILED' not in full,'full suite')
    check((out/'report.md').read_bytes()==(ROOT/'docs/P3J_ACCEPTANCE.md').read_bytes(),'acceptance report identity')
    return a,protection()

def main():
    out=Path(sys.argv[1]).resolve();a,old=validate(out)
    if '--verify' in sys.argv:
        r=read(out/'delivery-receipt.json')
        for p,h in r['artifact_sha256'].items():check(digest(ROOT/p)==h,'delivery '+p)
        check(old==read(out/'historical-source-protection.json'),'historical protection unchanged')
        with zipfile.ZipFile(out/'controller-source.zip') as z:
            for p in z.namelist():check(sha256(z.read(p)).hexdigest()==digest(ROOT/p),'controller archive '+p)
        print(json.dumps(dict(status='PASS',files=len(r['artifact_sha256']),validation_gate=a['validation_gate'])))
        return
    write(out/'historical-source-protection.json',old)
    m=a['evaluations']['validation-centered']['summary']['dmc|dmc|greedy']['metrics']['win_rate']
    win=f'{100*m["estimate"]:.2f}% [{100*m["ci95"][0]:.2f}%, {100*m["ci95"][1]:.2f}%]'
    hands=sum(r['hands'] for r in a['training'].values());samples=sum(r['samples'] for r in a['training'].values())
    updates=sum(r['updates'] for r in a['training'].values());games=sum(r['games'] for r in a['evaluations'].values())
    status=ROOT/'docs/STATUS.md';prior=read(ROOT/'artifacts/evaluations/p3i-centered-v1/delivery.json')
    check(digest(status)==prior['files']['docs/STATUS.md'],'previous status archive binding')
    with (out/'previous-status.md').open('xb') as f:f.write(status.read_bytes())
    lines=status.read_text('utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('当前交付：'):
            lines[i]=f'当前交付：**P0–P2及P3a–P3j在各自范围内验收通过**。P3j完成中心化辅助损失配对试验：{hands}手GPU训练、{games}局评测；固定centered-314380对greedy新验证{win}、{a["validation_gate"]}。P3整体仍为PARTIALLY_ACCEPTED，模型未晋级，贪心主基线保留；下一包见文末。'
        elif line.startswith('| P3 学习闭环 |'):
            lines[i]=f'| P3 学习闭环 | PARTIALLY_ACCEPTED | P3j工程通过；主候选新验证{win}、{a["validation_gate"]}；下一包见文末。 |'
    text='\n'.join(lines)+'\n';marker='## 下一工作包：P3j 中心化辅助损失单条件配对试验'
    check(text.count(marker)==1,'unique next-package marker');text=text.split(marker)[0]
    text+=f'''## P3j 中心化辅助损失单条件配对试验（2026-09-26）

**{a['engineering']}**；固定主候选新验证 **{a['validation_gate']}**。证据目录`artifacts/evaluations/p3j-centered-v1/`；原设计和完整验收见`docs/P3J_PROTOCOL.md`、`docs/P3J_ACCEPTANCE.md`。

继承三个固定teacher-200手模型，两臂同为0.1辅助权重，absolute回归完整候选绝对目标，centered按状态去除平均偏移。每臂每初始化续训600手，共{hands}手、{samples}观察、{updates}次CUDA更新。600个开发发牌重复使用，不增加独立发牌数。

| 初始化 | absolute开发胜率 | centered开发胜率 | 差值/百分点 | 95%描述区间/百分点 |
|---|---:|---:|---:|---:|
'''
    for r in a['paired']:
        if not r['validation']:
            lo,hi=r['ci95_delta_pp'];text+=f'| {r["seed"]} | {100*r["absolute"]:.2f}% | {100*r["centered"]:.2f}% | {r["delta_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |\n'
    text+=f'''
开发复用26原始发牌，每组8变体。唯一主候选centered-314380在新验证205000..205064共65组对greedy为 **{win}**；其他对手和配对差值见完整验收。总{games}局评测，零失败/非法/超时，完整牌谱与统计独立复算通过。

15项新增检查及188项全套测试在CUDA解释器全部通过。三个absolute完整600手模型/Adam/策略RNG/torch CPU和CUDA RNG/计数与P3h对应aux逐位一致；中心化跨进程完整wave恢复与故障重载通过。128状态/4174完整候选的768组forward核验及误差分解、10类审计负例与逐批次顺序/分母审计通过。

专用gd-p3j-checkpoint-v1记录objective和新增600手历史；继承200手仍需外部谱系。误差或相对弱对照改善不能替代绝对>=55%且区间下界>50%的棋力门槛。205000..205064现已查看，保留测试9000000..9009999继续封存。旧源hash保持一致；P4桌面端和P6完整比赛继续后置。

'''+(out/'next-package.md').read_text('utf-8')+'\n'
    status.write_text(text,encoding='utf-8',newline='\n')
    sources=sorted(set(list((ROOT/'experiments').glob('*p3j*.py'))+list((ROOT/'scripts').glob('p3j*.py'))+
        list((ROOT/'tests').glob('test_p3j*.py'))+[ROOT/'docs/P3J_PROTOCOL.md',ROOT/'docs/P3J_ACCEPTANCE.md',status]))
    with zipfile.ZipFile(out/'controller-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT).as_posix())
    files=[p for p in out.rglob('*') if p.is_file()]+sources
    files += [ROOT/f'artifacts/evaluations/{n}' for n in ('p3j-tests.log','p3j-test-receipt.json','p3j-seed-availability.json','p3j-preflight-v1.log')]
    write(out/'delivery-receipt.json',dict(status='SCOPED_ACCEPTANCE_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
        engineering=a['engineering'],overall_P3='PARTIALLY_ACCEPTED',validation_gate=a['validation_gate'],model_promoted=False,
        reserved_test_executed=False,full_tests_passed=188,artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(files))}))
    print(json.dumps(dict(status='SCOPED_ACCEPTANCE_COMPLETE',validation=win,files=len(set(files)))))
if __name__=='__main__':main()
