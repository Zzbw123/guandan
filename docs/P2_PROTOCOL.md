# P2 预先固定的评测规程 gd-evaluation-v1

冻结日期：2026-09-24，首次查看本阶段结果之前。总控制定本规程；子代理仅按协议实现。保持 P1 规则、观察协议和三类基线源码不变。只评价单副牌团队表现。

## 问题与固定预算

评价 random-v1、greedy-v1、team-heuristic-v1 的同策略组队及异质队友表现。对每个 focal、teammate、opponent 三元组都运行，三者均取 random/greedy/team，共 27 个场景。对手两位均使用 opponent。同策略队伍 9 场景，异质队友 18 场景，全部报告，不挑选有利结果。

唯一预先指定的主要比较是 `team|team|greedy`，指标为团队胜率，平衡点 50%。其余为描述性比较；各自 95% 区间不是全矩阵同时置信区间，不据此宣称多重比较后的显著排名。主比较下界不超过 50% 就报告“未建立优于该基线的证据”，不更换主要指标或加样本追显著。

本轮 validation 固定 130 个原始发牌，每个级牌 10 副。每场景运行所有预定对称变体。总计 46,800 个完成对局的预算（9×8×130 + 18×16×130）。复现重复跑不增加独立样本量。

种子区间：development 100000..109999 已用于 P1；validation 200000..200129 本轮首次用于固定基线评测；reserved_test 9000000..9009999 封存，不由本评测入口执行。每个种子只属一个集合，重复/越界/集合重叠必须拒绝。本轮报告不是后续学习模型的最终保留测试结果。

## 发牌、换座与换队

每原始发牌按 HandEnv.reset(seed) 得到 4 手牌；级牌=2+(deal_index mod 13)。不交换任意两个人的顺序，只做保持对家关系的循环旋转。

对 rotation r=0..3，将原座位 j 的整手牌放到 (j+r) mod 4，首攻物理座位固定为 0。因此每份原手牌都能首攻，原有顺时针相对关系与搭档关系保留。

每个 rotation 配合 swap b=0/1：focal 位置为 (r+b+2f) mod 4、teammate 为对面、两名对手为其余座位。若 focal=teammate，则 f 只取 0（8 变体）；否则 f=0/1（16 变体），确保 focal 与不同队友互换两手牌。每个策略组合均覆盖原始两队和所有物理座位。相同对阵的重复旋转、同一原始牌局的不同场景都不能视为独立新样本。

策略随机数由独立 policy_seed、deal_index、rotation、原手牌座位经 SHA-256 派生，不使用 deal_seed、不使用隐藏牌。共同随机数用于配对，策略只收到初始化随机数种子、不可变 PlayerObservation、合法动作。策略不得获得 Trial、GameState 或发牌种子。固定政策 seed、环境 seed 和实现版本即可复现对局结果；耗时不要求逐位一致。

## 指标与统计

每场景按原始发牌成组，先对同组全部 8 或 16 次对局求平均，再对 130 组求平均。报告独立组数、总对局数、逐组分布、标准差、最小/最大与中位数。

主指标：focal 所在团队的胜率；效应大小为胜率相对 50% 的百分点差。辅助指标：±1 团队回报；focal 团队当局获得的升级数（赢为1/2/3，输为0）；对手获得的升级数。辅助收益独立报告，不将人为差值当作规则或训练回报。

区间：按级牌分层、以原始发牌组为重采样单位，层内有放回抽取相同组数，进行 5,000 次 bootstrap，使用 2.5%/97.5% 分位数（线性插值）形成点态 95% 区间。保留换座结果的相关性以及固定级牌配比。至少 2 个组/层才报告区间；完全退化分布明确标记，不解释成总体没有不确定性。不做正态分布假设或额外 t 检验，不提供未经规划的 p 值。

缺失处理：必须验证所有预期 trial_id 恰好各出现一次，元数据与调度一致，分组完整；不删除失败对局再算漂亮胜率。任一失败、非法出牌、超时、代码变化都阻止完整评测通过。该场景区间不出具，失败原样记录。

性能：统计每次合法动作枚举和策略选牌的墙钟耗时，汇总真实样本的 p50/p95/max；报告候选数分布。单步 1,000 ms 是固定观测超时阈值，当前基线调用返回后才可判定，是软时限，不冒充可强制中断任意挂死模型的 watchdog。每局另有 1,000 步硬护栏。模型沙箱/强制超时留到 P3/P4。

P2 的 `decision_ms` 只计策略 `act()` 选牌耗时，`enumeration_ms` 独立计枚举；这与 P1 包含枚举的 `decision_ms` 口径不同，不能直接混比。所有原始耗时和候选数量写入 `measurements.jsonl.gz`，审计时重新计算分位数。

## 记录与晋级边界

输出 config 快照、事前清单、source/config 哈希、seed 列表、逐局 JSONL、逐组统计、全矩阵 JSON/Markdown 和进度日志。输出目录必须新建，不覆盖；源哈希运行前后核对。实验完整记录包含种子/内部摘要，只供研究管理，绝不流入策略。

基线没有神经网络权重或特征编码：model_hash=null，feature_version=null；用基线源码 SHA-256、策略版本和 observation/action/rules 版本标识。模型晋级一律 NOT_APPLICABLE：本阶段没有学习候选，也未动用 reserved_test。P3 的候选须另外冻结主要对手、检查点与晋级预算后评测。

## 总控分派的实现接口

- `schedule.build_trials(matchup, seeds) -> list[Trial]`；`schedule.deal_hands(trial)` 返回旋转后初始手牌；`schedule.policy_rng_seed(policy_seed, trial, seat)` 派生策略种子。
- `runner.run_trial(trial, policy_seed, max_steps=1000, decision_timeout_ms=1000.0) -> (row, measurements)`。row 包含 trial_id、matchup_id、deal_index、deal_seed、level、rotation、swap、flip、focal_seat、focal_team、policies、starting_player，以及 status(ok/error)、win、team_reward、focal_level_gain、opponent_level_gain、finish_order、terminal_digest、steps、illegal_actions、timeouts、error。measurements 含 enumeration_ms、decision_ms、candidate_counts 三个原始数组。
- `statistics.summarize_matchup(rows, expected_trials, bootstrap_replicates, bootstrap_seed) -> dict`：必须检查完整调度、类型、合法结果关系，失败/缺失/重复抛 ValueError；输出场景指标、CI、逐组统计与质量诊断。
- `statistics.cluster_bootstrap(values, strata, repetitions, seed, confidence=0.95) -> dict`：输出 estimate、ci95（或 None）、method、degenerate、reason，遵循上述分层组重采样。

总控负责配置校验、并行编排、产物写入、全矩阵报告、代码独立复核和验收。实施期间发现规则问题须单独立项，不能静默修改 P1 验收版本。
