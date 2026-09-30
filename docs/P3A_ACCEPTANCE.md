# P3a CPU 学习闭环验收

日期：2026-09-24。总控结论：**ACCEPTED_CPU_LEARNING_LOOP_V1**。P3 整体仍为 **PARTIALLY_ACCEPTED**。通过的是小规模 CPU 训练、存取、恢复与评测工程；没有通过棋力、模型晋级、GPU 或大规模训练验收。

## 冻结设计与实现

总控制定 `P3_PROTOCOL.md`，负责检查点、实验入口、独立测试、磁盘复算及最终验收；两名 gpt-6-sol/high 子代理分别实现编码/网络和采样/训练，未承担设计或验收决策。

- `gd-features-v1`：2993维玩家可见状态、149维动作；相对座位、双副实体副本等价、公开累计出牌、最后16条公开历史；更久历史仅保留累计特征，是有限记忆初始模型。
- `gd-dmc-network-v1`：共享四座位的动作价值网络，终局团队±1回报，整局Monte Carlo MSE/Adam更新。完整候选逐块评分，不裁剪合法动作。
- CPU单线程确定性；检查点只在完整成功对局的更新边界保存。保存参数、优化器、RNG、计数、使用过的训练seed、配置、编码/规则版本、源码哈希和运行时版本；受限加载与固定候选SHA-256核验。
- 失败训练器锁定，不能把半局或部分更新状态标为可恢复。没有实现局中断点、异常进程资源回收或强制推理看门狗。

## 总控实测证据

| 项目 | 结果 |
|---|---|
| CPU运行时 | Python 3.13.5、PyTorch 2.9.1+cpu、float32、1线程 |
| GPU状态 | 已有CPU版torch不支持CUDA；未安装CUDA版、未运行GPU训练 |
| Python 3.13全套测试 | 78/78通过，包括P1/P2旧60项 |
| Python 3.14兼容测试 | 66通过，12项PyTorch测试明确跳过；不作为3.14学习支持证据 |
| 编码边界 | 隐藏手牌互换、相对座位旋转、实体副本交换不变性；配牌声明可区分 |
| 高候选输入 | P1构造手牌8769候选全部经网络评分，每块至多256；不是全局最坏输入证明 |
| 正式闭环训练 | seed100500..100503，4局，662决策样本，12次Adam更新；参数确实变化 |
| 保存恢复 | 第2局后存盘；后2局连续/重载路径的参数、Adam、Python/torch RNG、计数逐位一致，逐局输出完全一致 |
| 训练与恢复耗时 | 3.696秒，包含初始化、存取、4局连续训练及2局恢复复跑；不作为纯采样吞吐 |
| 加载模型评测 | 固定开发seed101000、101001，3对手×2发牌×8对称变体，共48/48合法终局与回放通过 |
| 评测耗时 | 8.289秒，单进程CPU |
| 模型决策延迟 | 2696次；p50 0.5586ms、p95 1.2363ms、max 18.3823ms；只计act，不含枚举 |
| 评测最大候选数 | 1172；8769候选另由构造单测覆盖 |
| 源码保护 | P2事前注册40文件全部哈希未变；本轮45项注册源码/测试/协议运行前后相同，ZIP逐项核对一致 |
| 磁盘独立复算 | 候选哈希、原始48行调度、奖励方向、全部胜率、训练记录、源ZIP和P2保护均通过 |

固定候选 `candidate/checkpoint.pt` SHA-256：

`8d27fbba3fc77297b4dc2453d0ef42db6cd471bdf2e96d66b50cda660b2cf2a5`

| 对手（双方同策略组队） | 模型胜局/局数 | 开发集描述胜率 |
|---|---:|---:|
| greedy-v1（预先指定主对手） | 0/16 | 0% |
| random-v1 | 3/16 | 18.75% |
| team-heuristic-v1 | 0/16 | 0% |

三个场景都只有相同的2份原始发牌，不是48个独立样本；不出具置信区间、显著性或晋级判定。仅训练4局的模型明显尚不具备替代固定基线的证据。`promotion=NOT_ELIGIBLE`。没有使用validation或reserved_test，也没有根据结果改模型或追加评测以追求胜率。

## 证据与复现

目录 `artifacts/evaluations/p3a-cpu-v1/`：

- `preregistration.json`、`source-snapshot.zip`：运行前配置、seed、源码及可恢复快照。
- `training.jsonl`、`episode-2/`、`resume-audit.json`：训练与恢复证据。
- `candidate/manifest.json`、`candidate/checkpoint.pt`：固定候选。
- `evaluation-preregistration.json`、`evaluation.jsonl`、`report.json`：评测前绑定哈希和日程、原始48局、结果。
- `controller-audit.json`：独立磁盘复算；审计器自身SHA-256登记其中。审计器在正式实验完成后新增，未改变被执行的策略/训练源码。

全套测试日志：`artifacts/evaluations/p3a-unittest-python313.log`、`p3a-unittest-python314.log`。

额外用户入口验证：`scripts/train.py` 从固定4局候选恢复，使用开发seed100504续训1局，产物另存 `artifacts/evaluations/p3a-cli-resume-smoke/`。这是CLI可执行性检查；该5局模型不替换固定候选，不加入上述48局统计或晋级。

```powershell
py -3.13 scripts/validate.py
py -3.13 scripts/audit_learning.py artifacts/evaluations/p3a-cpu-v1
# 使用一个尚不存在的输出目录，完整重跑本工作包
py -3.13 scripts/p3_acceptance.py --output artifacts/evaluations/my-p3a-repeat
# 从检查点边界继续训练；指定未被该检查点训练过的开发seed
py -3.13 scripts/train.py --resume artifacts/evaluations/p3a-cpu-v1/candidate --episodes 4 --seed-start 100504 --output artifacts/evaluations/my-dmc-resume
```

依赖记录为 `requirements-learning-cpu.txt`，本轮直接使用已有运行时，未执行安装或升级。此清单仅锁定直接torch依赖；运行时精确版本还写入每个检查点，不宣称已验证一个全新环境的完整传递依赖重建。

## 下一门槛 P3b

先在隔离环境实测RTX5060的PyTorch/CUDA组合与模型前后向、全候选推理和CPU采样性能，比较端到端耗时后决定设备；不预先假定小网络GPU一定更快。然后实现并验证多进程采样/参数同步/恢复，冻结明确训练预算及对手池。正式学习候选须预注册独立验证预算、候选哈希、主要greedy对手与晋级门槛；已查看P2验证seed200000..200129不能重标为最终测试。保留测试集继续封存。桌面端仍后置。
