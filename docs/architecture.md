# 实施决策与接口

2026-09-24 用户明确：训练与研究优先；P4 做桌面端，覆盖原方案“本地浏览器首发”假设。桌面技术栈在 P3 后按本机打包验证选择，规则、训练不能依赖 UI。

P0/P1 已交付零第三方运行依赖的 Python 核心、实体牌/牌型/枚举、HandEnv、严格玩家观测、可校验回放、基线冒烟与基准入口。Python 3.13 与 3.14 均可使用，验收记录具体解释器。P2 增加配对评测，P3 再锁定 PyTorch/CUDA 组合并做训练恢复。

总控拥有 types.py、cards.py、规格与验收；子代理按总控协议实现动作引擎、状态机和独立测试，统一 gpt-6-sol/high。总控复核并运行验收，子代理报告不能替代验收。

HandEnv 契约：无参构造后 `reset(seed, rules_config=None, initial_level=2, starting_player=0)`；`observe(player_id)`；`legal_actions(player_id)`（只有当前玩家可调用）；`step(player_id, action, state_version=None)`；`serialize_replay()` 返回显式全信息研究回放；`HandEnv.replay(replay_dict)` 重演并逐步校验摘要。

内部状态允许测试与研究审计访问，但策略入口只接收不可变 PlayerObservation 与 Action 列表。 `state_digest()` 是全信息审计摘要，不进入观测。动作状态版本在 core 可选（便于研究），服务层将强制版本校验。

P2 的 contracts/config/schedule/runner/statistics/reporting 分别负责实验协议、数据集边界、对称调度、隔离执行、成组统计及报告。`scripts/evaluate.py` 拥有事前清单、并行编排、不可覆盖产物和磁盘原始记录复算。全部仍使用标准库，具体验收状态以 STATUS.md 为准。

实验源快照把事前记录的全部源文件打包，每项 SHA-256 与事前清单一致；方便后续代码改变时仍从相应快照复现。它不含模型权重或隐藏手牌记录；完整研究对局记录另存，并不进入策略接口。

P3a已加入`learning/encoding/model/training/checkpoint/evaluation`：玩家可见特征、共享座位DMC、完整对局采样与终局回报更新、严格版本/哈希检查点和固定开发集集成评测。`scripts/train.py`可独立续训，`scripts/p3_acceptance.py`验证CPU闭环和精确恢复，`scripts/audit_learning.py`独立从磁盘复算。规格见P3_PROTOCOL.md，已验收边界见P3A_ACCEPTANCE.md。

P3a的48局仅证明加载CPU模型可运行。2026-09-26已推进至P3c GPU同步批量训练，P3整体仍PARTIALLY_ACCEPTED。学习模型必须另经独立预注册评测才准许棋力结论。

新增`src/guandan_gpu/training.py`与`checkpoint.py`，保留原CPU学习模块：前者将多个玩家可见请求合并完整候选评分，并在固定4环境整批终局/回放后更新；后者在成功wave边界保存模型/Adam/独立策略RNG/CPU及CUDA RNG，并以临时目录到正式目录的重命名提交。单进程同步调度，不包含异步队列和陈旧actor。每批行为版本w-1，更新后模型版本w；下一批使用新参数。具体算法变更版本`gd-gpu-wave-v1`，不与P3a每局更新轨迹混用。

独立进程恢复验证限定本机同版本CUDA运行时。入口`train_gpu.py`限制开发seed与有界wave预算；`p3c_acceptance.py`执行固定工程实验，`audit_gpu_training.py`独立复算证据。P3c交付时留下的外部推理watchdog和独立验证已在下述P3d推进；长期训练与桌面P4继续后置。

P3d已完成首轮固定预算与独立验证（2026-09-26）：新增控制层放`experiments/p3d/`与`scripts/p3d_*.py`，不改`src`，避免旧GPU检查点源码集合漂移。`guard.py`的持久spawn进程只接收玩家观测和合法候选，输出索引/覆盖数/耗时，GPU推理2秒硬截止；失败后永久关闭并有界回收。该机制是故障隔离，不是恶意代码安全沙箱。环境与基线保持在评测父进程，基线返回后软检查；外层进程总截止覆盖整体挂起。

`protocol.py`冻结400手训练、唯一最终候选、65组×8变体×3对手日程及主比较门槛。`p3d_run.py`先注册源码/清单，再独立训练，绑定候选SHA后才验证。`p3d_review.py`作为总控只读入口，规范化JSON对象键后调用冻结审计函数，独立复算bootstrap并重演全部研究牌谱。此轮工程通过而候选对greedy仅22.12%，不晋级；P3e继续开发集诊断和训练研究，P4仍后置。
