# P5d 按批次正负标签均衡目标的 GPU 工程前置

2026-09-29，总控事前设计；版本 gd-p5d-label-balanced-engineering-v1。依据 P5c 描述性诊断提出待检验干预，不把标签不平衡认定为棋力下降的原因。本协议在本包正式训练前冻结，结果另写 P5D_ACCEPTANCE.md。

## 目标与边界

完成一个可审计、可精确恢复的损失干预实现。原 mixed 固定池、角色日程、规则、特征、网络、终局团队奖励 ±1、探索与完整合法候选不变。只有 current 的实际决策进入学习；完整状态、seed、终局奖励和研究回放不传入策略。frozen 仍为本次随机初始化的固定副本。

本包只作工程验收，不估计干预的棋力效果；不运行开发对局评测、新验证或保留测试。P5b 的预算和结论保持原样。

## 唯一干预与精确定义

配置 objective 为 ordinary 或 label_balanced。设一个实际 minibatch 有 N 个样本，预测 q_i，原终局标签 y_i∈{-1,+1}，正负计数 N+、N-。

- ordinary：原 PyTorch MSE，L=Σ(q_i-y_i)²/N，调用原函数以保持逐位兼容。
- label_balanced：两类都存在时，L=Σ正类(q_i-1)²/(2N+)+Σ负类(q_i+1)²/(2N-)；等价每样本权重 w+=N/(2N+)、w-=N/(2N-) 后取全批平均。
- 单类批次退回 ordinary MSE；存在类权重 1，缺失类权重 0。缺失的成功轨迹不能靠重新加权产生。
- 权重只依赖当前实际 minibatch 的标签计数，不使用验证结果、对手身份或候选动作类别。标签保持 ±1，不截断/复制/丢弃样本，不更改打乱顺序。尾批以实际长度计数，每批一次 Adam 更新。
- 每批记录 objective、start、n、positive、negative、positive_weight、negative_weight、single_class、loss。wave mean_loss=Σ(batch_loss×batch_n)/wave_n，是不同更新时刻的训练日志，不能解释成固定权重全 wave 的均衡损失。

这是对拟合目标的改变，不是对原期望回报的无偏修正；输出不得解释为已校准胜率。不直接惩罚合法过牌，也不把一次出完排序当作棋力代理。批次中同手决策共享终局标签，样本数不等于独立重复数。

## 固定工程预算

初始化314500，两臂 ordinary / label_balanced，各16手、4个wave，num_envs=4、epsilon=.1、lr=.001、batch_size=256、chunk_size=1024。总32次主训练执行，共用16份既有开发发牌108800..108815；第n手级牌2+n%13、起手n%4，沿用P5a四种角色组合和四个focal座位。ordinary 后 label_balanced 固定执行顺序，不以耗时作速度比较。

每臂保存initial和每wave检查点。label_balanced 的wave2在新进程恢复到wave4，额外8手与主运行重复；不增加独立发牌数。ordinary的全部四wave与既有P5a-v2记录比较核心状态/日志逐位一致，不把新增checkpoint元数据视为算法差异。单元测试使用开发seed与临时目录，额外执行不计入上述研究预算。

每个正式子进程最长600秒，失败保留目录并停止；重试使用新目录，不按模型行为换seed或追加手数。GPU入口使用scripts/run_gpu.ps1记录的CUDA解释器；float32、确定性算法、TF32关闭。检查点独立版本gd-p5d-checkpoint-v1/gd-p5d-wave-v1，完整绑定objective、模型、Adam、RNG、计数、冻结池、来源与运行时，旧格式不得误载。训练失败后必须重载成功wave边界，不保存部分更新状态。

## 冻结与独立验收

正式训练前通过配置JSON往返与新增目标/合同测试，冻结协议、执行源码、审计源码、测试、历史输入哈希并保存ZIP。输出目录不可覆盖。历史P0–P5c已注册源和不可变交付产物前后复核；STATUS.md为可变状态入口，历史快照哈希不要求恒定。

总控从seed重建全部32手，按独立角色日程核验全部决策、完整候选、策略选择、探索与打乱RNG、样本归属、终局标签和计数。神经策略选择使用独立线性层表达式及257分块复算。逐批用独立样本权重表达式重算损失、梯度与Adam，核验模型/优化器终态；若wave仅一次更新，从Adam前后一阶矩反推实际梯度。数值容差rtol3e-4/atol3e-6；恢复与ordinary历史兼容要求逐位相同。

测试至少覆盖不均衡两类、单类/单例、标签翻转、空/非有限/错误shape/标签/设备/目标拒绝、实际尾批、ordinary兼容、更新中途失败后拒绝/重载，以及检查点objective/版本篡改拒绝。独立审计拒绝角色、候选、标签、样本数、目标、批次计数/权重/分母/损失、RNG等篡改。完整GPU回归零跳过后出具验收。

通过结论仅为ACCEPTED_LABEL_BALANCED_ENGINEERING_V1，P5仍PARTIALLY_ACCEPTED，model_promoted=false。最近正式棋力结论仍为P5b NOT_ESTABLISHED；greedy保留，已查看验证至207000..207064不回收为新数据，9000000..9009999继续封存。后续P5e须另行冻结等发牌ordinary/mixed标签均衡配对研究的初始化、预算、候选和新验证范围；不由本包短运行挑选最优权重。P4桌面端、P6完整比赛继续后置。
