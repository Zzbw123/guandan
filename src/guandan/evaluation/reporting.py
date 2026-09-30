"""Human-readable report plus exact quantiles of recorded timing samples."""
import math


def sample_summary(values):
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Invalid timing/count sample")
    ordered = sorted(values)
    return {"count": len(ordered), "p50": ordered[math.ceil(len(ordered) * .5) - 1],
            "p95": ordered[math.ceil(len(ordered) * .95) - 1], "max": ordered[-1]}


def format_rate(metric):
    estimate = metric["estimate"] * 100
    ci = metric["ci95"]
    return f"{estimate:.2f}% [{ci[0]*100:.2f}, {ci[1]*100:.2f}]" if ci else f"{estimate:.2f}% [区间不可用]"


def markdown_report(report):
    lines = ["# P2 固定基线评测报告", "", f"执行状态：**{report['status']}**。数据集合：`{report['config']['split']}`。",
             "", "本报告针对 gd-hand-v1 单副牌团队表现；没有学习模型，没有模型晋级。所有次要区间均为点态区间，不构成全矩阵显著性排名。",
             "", f"原始发牌：{report['config']['deal_count']} 组；计划 {report['expected_games']} 局；实际记录 {report['recorded_games']} 局；失败 {report['failed_games']} 局。",
             f"非法动作 {report['illegal_actions']}；观测软超时 {report['timeouts']}；源文件未变：{report['source_unchanged']}。",
             "", "## 预先指定的主要比较", "", "队友启发式 + 队友启发式，对贪心 + 贪心。平衡点 50%。",
             "", f"结论标记：`{report['primary']['conclusion']}`。"]
    primary = report["statistics"].get(report["config"]["primary_matchup"])
    if primary:
        lines += [f"团队胜率及 95% 区间：**{format_rate(primary['metrics']['win_rate'])}**；相对 50% 的效应为 {primary['effect_vs_50pp']:+.2f} 个百分点。",
                  f"独立统计单位 {primary['independent_deals']} 组，运行 {primary['games']} 局。"]
    lines += ["", "## 同策略组队完整矩阵", "", "行策略两人组队对列策略两人组队；格内为胜率及 95% 分层组 bootstrap 区间。", "",
              "| 本队 \\ 对手 | random | greedy | team |", "|---|---|---|---|"]
    for own in ("random", "greedy", "team"):
        cells = []
        for other in ("random", "greedy", "team"):
            data = report["statistics"].get(f"{own}|{own}|{other}")
            cells.append(format_rate(data["metrics"]["win_rate"]) if data else "无有效结果")
        lines.append(f"| {own} | " + " | ".join(cells) + " |")
    lines += ["", "## 所有队友组合", "", "random=随机，greedy=贪心，team=队友启发式。队友是队内另一位；对手列表示对面两人都使用该策略。",
              "", "| focal | 队友 | 对手 | 胜率 [95% CI] | 团队回报 | 本队升级数 | 对手升级数 | 独立组 / 局数 | 决策 p95 ms |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for key in sorted(report["statistics"]):
        s = report["statistics"][key]
        m = s["metrics"]
        focal, mate, opponent = key.split("|")
        latency = report["performance"][key]["decision_ms"]["p95"]
        lines.append(f"| {focal} | {mate} | {opponent} | {format_rate(m['win_rate'])} | {m['team_reward']['estimate']:.4f} | {m['focal_level_gain']['estimate']:.4f} | {m['opponent_level_gain']['estimate']:.4f} | {s['independent_deals']} / {s['games']} | {latency:.3f} |")
    lines += ["", "## 数据质量与假设检查", "", "在统计之前核对完整调度、唯一试验ID、元数据、终局结果与组内变体数；失败局不丢弃。按原始发牌合并相关变体，以级牌分层重采样，未假设组均值正态。结果仅覆盖已固定基线和所抽取牌局。",
              "", "下表为原始发牌组胜率的离散分布。均值、标准差、中位数和极值以及全部逐组记录保存在 report.json。退化区间不表示总体没有不确定性。",
              "", "| 场景 | 每组胜率 → 组数 |", "|---|---|"]
    for key, data in sorted(report["statistics"].items()):
        distribution = data["quality"]["cluster_win_rate"]["distribution"]
        lines.append(f"| {key.replace('|', ' / ')} | {distribution} |")
    if report["statistics_errors"] or report["worker_errors"]:
        lines += ["", "## 阻止验收的问题", "", f"统计错误：`{report['statistics_errors']}`", f"工作进程错误：`{report['worker_errors']}`"]
    lines += ["", "## 复现与边界", "", "固定预算，不按结果追加抽样或更换主要比较；bootstrap 区间按级牌分层，以原始发牌为单位。完整场景之间也共享发牌，不可将它们再合并成独立样本。",
              "", "validation 可用于后续工程决策，但不是保留测试集；reserved_test 没有执行。软超时只在策略调用返回后检测；任意挂死代码的强制终止不在本阶段承诺内。",
              "", "无神经网络权重，model_hash/feature_version 为 null；基线源码哈希及版本见 preregistration.json。",
              "", "复算原始记录与哈希：", "", "```powershell", f"python scripts/evaluate.py --audit \"{report['output_directory']}\"", "```", ""]
    return "\n".join(lines)
