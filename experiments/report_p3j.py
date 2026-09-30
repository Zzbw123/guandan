"""Scoped P3j report from completed independent evidence; no outcome selection."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text('utf-8'))
def pct(v):return f'{100*v:.2f}%'
def interval(v):return f'[{100*v[0]:.2f}%, {100*v[1]:.2f}%]'

def main():
    out=Path(sys.argv[1]).resolve();a=read(out/'controller-audit.json')
    n=read(out/'controller-negative.json');f=read(out/'fidelity/decomposition-audit.json')
    b=read(out/'batch-order-audit.json');t=read(ROOT/'artifacts/evaluations/p3j-test-receipt.json')
    for r in (a,n,f,b,t):assert r['status']=='PASS'
    hands=sum(v['hands'] for v in a['training'].values());samples=sum(v['samples'] for v in a['training'].values())
    updates=sum(v['updates'] for v in a['training'].values());games=sum(v['games'] for v in a['evaluations'].values())
    primary=a['evaluations']['validation-centered']['summary']['dmc|dmc|greedy']['metrics']['win_rate']
    rel=out.relative_to(ROOT).as_posix()
    lines=['# P3j 中心化教师偏好辅助损失配对试验验收','',
        f'日期：2026-09-26。工程结论 **{a["engineering"]}**；唯一主候选验证 **{a["validation_gate"]}**。',
        f'固定centered-314380对greedy胜率 **{pct(primary["estimate"])}，95%成组区间{interval(primary["ci95"])}**。',
        'P3整体仍为PARTIALLY_ACCEPTED；本包没有替换贪心主基线或交付桌面端。工程通过与模型晋级分开记录。','',
        '## 固定设计与实际执行','',
        '沿用P3i交付的P3J_PROTOCOL.md，比较absolute与centered两臂，辅助权重同为0.1；仅将完整候选绝对MSE替换为按状态中心化MSE。继承P3f三个teacher-200手权重314380/314381/314382，均只传模型参数，重置Adam、Python策略RNG及torch CPU/CUDA RNG。',
        '每模型新增600手，seed100200..100799，4环境同步wave；lr=.001、batch256、chunk1024、epsilon=.1，团队终局回报±1。既有规则/编码/网络保持原样。',
        f'实际执行 **{hands}手训练、{samples}条观察、{updates}次CUDA更新、{games}局评测**。600个训练发牌被6模型重复使用，不能算成3600个独立发牌；继承的200手不重复计入新训练。',
        '运行顺序按初始化分块、块内两臂随机排序，具体列表和随机种子事前写入preregistration。固定最后600手权重，不选中间模型、不扫参数、不追加样本。两臂轨迹和更新数可能不同，环境手数相同不等于计算预算相同。','',
        '| 模型 | 新增手数 | 决策观察 | CUDA更新 | 完整辅助候选 | 训练与保存秒数 |',
        '|---|---:|---:|---:|---:|---:|']
    for job,r in sorted(a['training'].items()):
        lines.append(f'| {job} | {r["hands"]} | {r["samples"]} | {r["updates"]} | {r["auxiliary_candidates"]} | {r["seconds"]:.2f} |')
    lines += ['', '## 开发配对结果','',
        '同26个已查看原始发牌，13级牌各2组，每组8个轮换/换队变体。每模型208局。差值为centered−absolute，区间条件于当前初始化，为点态描述；不能当作新的独立验证或跨初始化总体区间。','',
        '| 初始化 | absolute胜率 | centered胜率 | 差值/百分点 | 95%配对区间/百分点 |',
        '|---|---:|---:|---:|---:|']
    for r in a['paired']:
        if not r['validation']:
            lo,hi=r['ci95_delta_pp'];lines.append(f'| {r["seed"]} | {pct(r["absolute"])} | {pct(r["centered"])} | {r["delta_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    lines += ['', '## 新验证与绝对门槛','',
        '训练前唯一指定centered-314380最终600手模型为主候选。absolute-314380作同发牌对照；205000..205064共65个新原始发牌、每级5组，每组8变体，各对greedy/random/team评520局，两臂共3120局。',
        '先平均组内变体，再按级牌分层、原始发牌成组bootstrap5000次（seed314415）；两臂差值保留配对关系。65组沿用资源约束预算，未承诺统计效能。','',
        '| 对手 | absolute胜率及95%区间 | centered胜率及95%区间 | 差值/百分点及95%区间 |',
        '|---|---:|---:|---:|']
    for r in a['paired']:
        if r['validation']:
            metrics=[a['evaluations']['validation-'+arm]['summary'][r['matchup']]['metrics']['win_rate'] for arm in ('absolute','centered')]
            x,y=metrics;lo,hi=r['ci95_delta_pp']
            lines.append(f'| {r["matchup"].split("|")[-1]} | {pct(x["estimate"])} {interval(x["ci95"])} | {pct(y["estimate"])} {interval(y["ci95"])} | {r["delta_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    lines += ['',
        '唯一主比较仍要求centered候选对greedy点估计>=55%、95%区间下界>50%、非退化、零失败/非法/超时。相对absolute的改善不能替代绝对门槛。配对差值、其他对手与开发结果均为次要点态描述，不作多重比较校正后的发现，也不以区间含0宣称无效或等效。',
        '本轮没有建立中心化条件的稳定改善：开发314381有正向点态区间，但另外两组区间含0；固定主候选新验证对greedy和team的差值区间均含0，对random的次要点态区间位于0以下。不能据最好的开发初始化替换主候选，亦不将区间含0解释为等效。',
        '原始发牌组的均值/标准差/中位数/极值、零/全胜组和缺失计数在descriptives.json；组数据在plot_data_groups.csv。全部预设结果保留，未删极端组。这里不采用正态独立局假设或事后更换显著性检验。','',
        '## 固定同状态机制诊断','',
        '复用原teacher-314380阶段前128个多候选观察，4174完整候选；六模型逐组直接forward检查全部通过。下表按状态等权，中心化MSE与偏移平方之和为原MSE，独立总体方差表达式复算通过。','',
        '| 模型 | 教师top-1 | 标准化遗憾 | 原MSE | 均值偏移平方 | 中心化MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for job,r in f['summary'].items():
        lines.append(f'| {job} | {pct(r["top1"])} | {r["regret"]:.5f} | {r["mse"]:.5f} | {r["offset_mse"]:.5f} | {r["centered_mse"]:.5f} |')
    lines += ['',
        '三个centered模型在此固定教师状态子集上的中心化MSE均高于配对absolute模型；top-1和遗憾变化仍不一致。这不等于各自实际训练分布上的辅助损失更高，也不证明实现错误；两臂的后续轨迹和DMC梯度均会变化。',
        '该子集是事先选定的有限开发诊断，不是无偏状态样本；不生成总体置信区间，不把误差下降、教师一致率或遗憾直接视为棋力。','',
        '## 工程与独立验收','',
        '- 15项新增GPU/边界检查全部通过：稠密与批量分块中心化梯度和一次Adam更新、全局均值、单候选零项但保留状态分母、8769候选全量覆盖、非有限输入拒绝。',
        '- absolute短程对照与P3h aux逐位一致；正式三个absolute模型完整600手后的模型/Adam/Python策略RNG/torch CPU及CUDA RNG/计数与P3h对应aux逐位一致。',
        '- 中心化检查点由独立Python进程恢复后，下一完整wave的参数、Adam、RNG、计数和轨迹逐位相同；注入学习失败后禁止继续训练或提交，重载边界重做一致。正式spawn推理覆盖8769候选并验证超时/崩溃/非法返回后回收。',
        '- 专用gd-p3j-checkpoint-v1 / gd-p3j-wave-v1保存objective和0.1权重；旧P3h加载器拒绝该版本。检查点只计新增600手，继承200手由reset及输入manifest关联。不能宣称半局、跨设备、跨驱动恢复。',
        f'- 总控独立重建全部{hands}手训练及{games}局评测牌谱、完整候选、终局和行为计数，并独立复算分层配对区间；零失败/非法动作/超时。',
        '- 从真实训练轨迹独立重建所有探索抽样与wave内shuffle，逐批核验辅助观察/候选分母及最终策略RNG。该检查不等于所有辅助梯度逐更新重算。',
        f'- {len(n["cases"])}类审计负例全部拒绝：'+', '.join(n['cases'])+'。',
        '- 首次前置测试因新检查点文件一处字符串引号错误停止；修复后15项重跑通过，失败日志保留。报告阶段误差分解脚本重复传递mse字段，修复后重跑并保留原失败日志；只修改后处理脚本，未改冻结训练、权重或原评分。正式实验未因结果改条件或重选日程。',
        '- 原登记源码哈希、输入权重、冻结源码ZIP、运行结果回执全部核对；后续报告/图表代码另存控制器源码归档。每模型推理2秒硬截止、启动60秒，每任务3600秒外层截止；固定基线保持返回后的软时限。',
        '- 墙钟耗时包含本机其他工作负载，不用于算法速度比较。新验证205000..205064现已查看，不能在调参后重标为未看验证；9000000..9009999未使用。','',
        '## 证据入口','',f'正式目录：`{rel}`。原设计：`docs/P3J_PROTOCOL.md`。',
        f'唯一主候选SHA256：`{a["candidate_sha256"]}`。',
        '`preregistration.json`与`source-snapshot.zip`冻结实验；`controller-audit.json`、`controller-negative.json`和`batch-order-audit.json`记录独立审计；fidelity/保存完整评分及误差分解。',
        '开发/验证PNG和SVG、原始发牌组CSV及描述统计见descriptives/。最终全套测试与文件保护结果见delivery-receipt.json。','',
        '```powershell',f'py -3.13 experiments/review_p3j.py {rel} --cache artifacts/evaluations/p3j-controller-cache',
        f'./scripts/run_gpu.ps1 experiments/inspect_p3j.py {rel} --verify','```','']
    text='\n'.join(lines)
    for p in (out/'report.md',ROOT/'docs/P3J_ACCEPTANCE.md'):
        with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    usage=dict(new_training_hands=hands,training_samples=samples,CUDA_updates=updates,unique_training_deals=600,
        new_evaluation_games=games,development_groups=26,validation_groups=65,new_viewed_validation=[205000,205064],
        model_promoted=False,reserved_test_executed=False)
    with (out/'data-usage.json').open('x',encoding='utf-8') as f:json.dump(usage,f,indent=2)
    print(json.dumps(usage))
if __name__=='__main__':main()
