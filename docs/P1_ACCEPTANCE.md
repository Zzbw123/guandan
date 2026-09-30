# P0/P1 总控验收记录

验收范围：`gd-hand-v1` 本地研究规则、单副牌环境及基线冒烟基础。最终状态和机器记录见 `STATUS.md` 与 `artifacts/evaluations/`。该范围不含官方规则认证、学习模型棋力或桌面产品验收。

总控结论：**ACCEPTED_FOR_RESEARCH_V1**。Python 3.13.5 与 3.14.3 各 32 项测试通过；8 进程 10,000 副牌全部完成且逐步回放一致，失败 0，总耗时 152.430 秒，960,191 次决策，源文件运行前后未改变。单进程 200 副复测通过。详细指标见 STATUS.md 和机器 JSON，不把本结论扩大为 P2–P6 通过。

## 责任与审查

总控定义数据协议、规则、状态机语义、枚举等价准则及验收方法，并拥有基线和实验入口。三名 `gpt-6-sol / high` 子代理分别实现动作引擎、环境状态机、独立黑盒测试；之后复用动作代理只读监控 10,000 局日志。

总控独立阅读并复核核心实现，补充跨级牌比较和基线信息隔离检查，要求修复并复查：过牌动作的布尔/整数混淆、回放元数据的布尔/整数等值混淆，以及 108 张不均分研究样例的标记。子代理报告没有直接当作阶段验收结论。

## 条款与验证映射

| 规则条款 | 对应测试或检查 |
|---|---|
| R01 实体牌/队伍 | `test_deck_faces_and_roundtrip`、`test_full_108_card_deal_and_observation_are_immutable`、双上与团队结算测试 |
| R02 发牌/首攻/级牌 | 重复种子测试；压力跑局四座轮流首攻、13 级牌覆盖 |
| R03 级牌比较 | `test_all_level_single_order_and_equal_values`、`test_sequence_comparison_does_not_promote_level` |
| R04 配牌 | 双配声明、不可替王、同花顺花色声明单测；39 个随机六张手牌独立穷举 |
| R05 基本牌型/王对三带二 | 独立子集分类 oracle、`test_joker_pair_full_house_and_bomb_order` |
| R06 连续牌型/A 边界 | A 低/高连续牌型、禁止绕接、钢板/连对 oracle |
| R07 炸弹层级 | 王炸/同花顺比较单测；4..10 张炸弹定向枚举 |
| R08 声明/严格压制 | 同实体普通顺子/同花顺双声明、同值不可压、不同普通牌型不可压 |
| R09 领出/过牌/再接 | `test_cover_resets_pass_set_and_earlier_passer_can_return`、非法领出过牌 |
| R10 活跃玩家/接风 | `test_finished_incumbent_pass_cycle_gives_partner_lead`、`test_active_incumbent_retains_next_free_lead` |
| R11/R12 终局/收益 | 双上不伪排名、第三位结束保留第四手牌、升级收益 3/2/1、团队 ±1 |
| R13 拒绝原子性 | 错误座位/过期/重复/手牌外/非法声明/错误类型拒绝且摘要不变 |
| R14 信息边界 | 隐藏牌交换前后观测/合法动作/三类基线策略输出一致；不可变观测 |
| R15 等价去重/完整性 | 同 face 副本等价与花色消耗单测；10 类牌型独立 oracle，非规范副本可验证 |
| R16 研究回放 | 显式 `research_full`、逐步事件/摘要核对、类型及内容篡改拒绝、批量全局重演 |

独立 oracle 不使用引擎的分类或校验函数：逐一枚举小手牌的实体子集及最多 13² 配牌点数组合，用独立分类器比较 `(kind, main_rank, consumed_faces)` 集合，并独立核查实际动作的配牌声明。它覆盖有限样例，不声称是对全部 27 张手牌的形式化完备性证明。

## 可复现入口

```powershell
python scripts/validate.py
py -3.13 scripts/validate.py
python scripts/play_hand.py --seed 42 --level 2 --agent team
python scripts/benchmark.py --hands 10000 --workers 8 --batch-size 50 --start-seed 100000 --replay-every 1 --output artifacts/evaluations/rerun-p1.json
```

压力测试中的不变量检查使用 Python `assert`，验收命令不使用 `-O`/`-OO`。环境的输入拒绝使用显式异常，不依赖断言。跑局报告绑定实际源文件 SHA-256，并核对运行前后源文件未变。

`enumeration_ms` 是 `legal_actions()` 墙钟时间（含跟牌过滤）；`decision_ms` 是合法枚举加基线选牌，均不含环境 step。吞吐率包含 step 不变量检查和每副牌的序列化重演、读取重演。并行竞争下的延迟与单进程延迟分别报告；没有神经网络推理指标。

## 仍未覆盖的边界

- P2 的预注册对阵、换座/换队、以原始发牌分组的置信区间和模型晋级尚未完成；本轮不报告基线强弱结论。
- P3 的 DMC、GPU 训练/推理、样本队列、检查点恢复与学习实验尚未开始。
- P4 桌面端未实现，也未做桌面 UI/打包验证。
- P6 完整比赛、升级过 A 和进还贡尚未实现。
- 没有从官方完整原文获得逐条认证，也未做第三方规则引擎差分；所选研究变体详见 rules.md。
- 没有采用外部源码或权重；候选预审并非全部源码/权重许可证尽调。
