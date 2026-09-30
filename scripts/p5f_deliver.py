"""Publish verified P5f engineering/study receipts and scoped project status."""
from _bootstrap import ROOT
import sys
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from pathlib import Path
import re
from experiments.p5f_run_common import read,write,digest,check,verify_frozen


def main(engineering,study,tests):
    engine=read(engineering/'receipt.json');verify_frozen(engineering)
    log=tests.read_text(encoding='utf-8-sig')
    match=re.search(r'Ran (\d+) tests?',log)
    check(match and int(match.group(1))>=274 and log.rstrip().endswith('OK') and 'skipped=' not in log,
          'full GPU regression with no skips')
    count=int(match.group(1))
    text=['# P5f 动作排序预训练验收','',
          '2026-09-29。训练与研究优先；本包无新独立棋力验证、无模型晋级，P5仍PARTIALLY_ACCEPTED。','',
          '## 工程与固定小样本检查','',
          f"工程结论：**{engine['scope']}**；独立512步小样本检查：**{engine['fit_gate']}**。",
          f"多候选观察数{engine['probe_summary']['multi']['observations']}，教师最优集合命中率{engine['probe_summary']['multi']['accuracy']:.2%}。",
          '两目标各四批独立稠密损失/梯度/Adam复算；新进程batch2恢复至batch4的完整payload和日志逐位一致。',
          '首轮默认Adam数值敏感性失败日志保留；正式冻结前固定两臂eps=1e-4，未按拟合/棋力结果选取参数。',
          f'全套{count}项GPU测试通过，零跳过。','']
    if study is not None:
        verify_frozen(study);audit=read(study/'controller-audit.json')
        check(audit['status']=='PASS','controller audit')
        text+=['## 固定配对研究','',f"**{audit['scope']}**；开发门控 **{audit['development_gate']}**。",
            '1600份共用训练发牌，六作业各一次完整教师预训练；对照和排序臂数据、批次、更新预算相同。所有final冻结后才评分和对局。',
            '26份已查看开发发牌，greedy/random每模型各208局，总2496局；没有新验证或封存测试。','',
            '| 初始化/目标 | 学习观察 | 更新 | 训练探针命中 | 开发多候选命中 | 对greedy胜率及95%区间 | 对random胜率及95%区间 |',
            '|---|---:|---:|---:|---:|---|---|']
        for job,item in audit['training'].items():
            f=audit['teacher_fit'][job]['summary'];ev=audit['evaluations'][job]['summary']
            def metric(opponent):
                m=ev['dmc|dmc|'+opponent]['metrics']['win_rate']
                return f"{m['estimate']:.2%} [{m['ci95'][0]:.2%}, {m['ci95'][1]:.2%}]"
            text.append(f"| {job} | {item['samples']} | {item['updates']} | {f['train_probe']['multi']['accuracy']:.2%} | {f['development']['multi']['accuracy']:.2%} | {metric('greedy')} | {metric('random')} |")
        text+=['','配对差为ranking−regression，单位百分点；区间条件于固定初始化，开发描述性用途。','',
               '| 初始化/对手 | 配对差 | 95%区间 |','|---|---:|---|']
        for key,item in audit['paired'].items():
            text.append(f"| {key} | {100*item['difference']:+.2f} | [{100*item['ci95'][0]:+.2f}, {100*item['ci95'][1]:+.2f}] |")
        text+=['','## 总控独立核验','',
            '全部训练/拟合开发教师牌谱从seed重建、独立教师评分与动作/事件/状态/终局一致；所有完整候选与编码等价分数一致。',
            '六作业全部批次观察顺序/候选分母/更新覆盖复算，六作业各首两批真实梯度/Adam重算（12批，不称所有历史梯度已重算）。',
            '全部拟合记录以独立网络表达式复核；全部2496局开发牌谱、行为分母、胜率及成组成对区间独立复算。',
            '三对初始参数逐位一致，checkpoint/source/corpus/候选hash保持一致；工程单测包括篡改及故障拒绝。','']
        next_step=('已满足开发门控；后续另立selfplay续训与排序监督保持的配对协议。' if
                   audit['development_gate']=='READY_FOR_RL_STUDY_DESIGN' else
                   '开发门控未通过，按计划停止向RL扩量；根据训练探针与开发拟合差异另立表示/拟合或数据聚合实验，不能继承此结果宣称棋力提高。')
        brief=f"P5f完整候选动作排序预训练配对研究完成：{audit['scope']}；开发门控{audit['development_gate']}。2496局开发评测已独立核验；棋力仍NOT_ESTABLISHED，模型未晋级。"
        target=study
    else:
        next_step='固定512步可拟合检查未过99%门槛，正式六作业未启动。下一步先定位目标/表示可拟合性，禁止因失败直接追加探针预算。'
        brief=f"P5f动作排序预训练工程完成；固定小样本检查{engine['fit_gate']}，正式研究未启动。模型未晋级。"
        target=engineering
    text+=['## 下一步与边界','',next_step,
           '工程与开发拟合不等于绝对棋力晋级；原>=55%且95%区间下界>50%正式门槛不变。9000000..9009999继续封存，P4桌面端/P6完整比赛后置。',
           '',f"工程证据：{engineering.relative_to(ROOT).as_posix()}。",f"最终证据：{target.relative_to(ROOT).as_posix()}。"]
    acceptance=ROOT/'docs/P5F_ACCEPTANCE.md'
    with acceptance.open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    status=ROOT/'docs/STATUS.md';old=status.read_text(encoding='utf-8')
    lines=old.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('当前交付：'):
            lines[i]='当前交付：**'+brief+'** P5仍PARTIALLY_ACCEPTED，P4/P6后置。';break
    updated='\n'.join(lines)+'\n'
    insertion='## 2026-09-29 P5f动作排序预训练\n\n'+brief+' 全套'+str(count)+'项GPU测试零跳过。详见P5F_PROTOCOL.md/P5F_ACCEPTANCE.md。'+next_step+'\n\n'
    position=updated.index('## ')
    status.write_text(updated[:position]+insertion+updated[position:],encoding='utf-8')
    artifacts={p.relative_to(ROOT).as_posix():digest(p) for p in target.rglob('*') if p.is_file()}
    artifacts[acceptance.relative_to(ROOT).as_posix()]=digest(acceptance)
    artifacts[tests.relative_to(ROOT).as_posix()]=digest(tests)
    write(target/'delivery-receipt.json',dict(status='PASS',scope=brief,full_gpu_tests=count,skipped=0,
         artifact_sha256=artifacts,model_promoted=False,reserved_test_executed=False))
    print(brief,flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),None if sys.argv[2]=='none' else Path(sys.argv[2]).resolve(),Path(sys.argv[3]).resolve())
