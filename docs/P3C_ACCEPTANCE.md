# P3c GPU 同步批量训练与恢复验收

日期：2026-09-26。总控结论：**ACCEPTED_GPU_WAVE_TRAINING_V1**。这是P3b之后的GPU持续训练工程包；P3整体仍为PARTIALLY_ACCEPTED。无棋力或模型晋级结论。

## 实现与事前范围

协议`P3C_GPU_TRAINING_PROTOCOL.md`在实验前冻结并进入源码ZIP。GPU算法位于新增`src/guandan_gpu`，不改旧规则/CPU学习源。4环境单进程同步推进，非探索请求共同评分，完整保留候选；整批所有终局及回放通过后按固定环境顺序合并样本，一次shuffle、一遍minibatch MSE/Adam更新。每批参数冻结、批间模型版本递增。不是多进程/异步actor系统。

网络/特征沿用既有版本，团队终局±1。初始化314260、epsilon0.1、lr0.001、batch256、chunk1024。开发seed103000..103015，共4批；级牌2+i%13、首攻i%4。验证集/保留测试集均未使用。

## 实测与总控复核

| 项目 | 结果 |
|---|---|
| GPU运行时 | Python3.14.3 / PyTorch2.12.0+cu130 / CUDA13.0 / RTX5060 Laptop GPU |
| 数值设置 | float32、TF32关闭、确定性算法、CPU线程1、CUBLAS配置:4096:8 |
| 训练 | 16/16手完成及研究回放通过，2648样本，12次Adam更新 |
| 分批样本 | 560、721、707、660；每批3次更新 |
| CUDA证据 | 12次训练前向、84次参数梯度、252次Adam状态张量设备检查；Adam包含CUDA step状态 |
| 训练循环墙钟 | 5.006859秒，包括真实采样/编码/推理/学习/完整回放，不含检查点I/O |
| 完整验收脚本 | 19.276234秒，包括高候选、保存、独立进程恢复与故障检查；不含后续独立审计/回归测试 |
| 精确恢复 | 第2批后新Python进程恢复后2批，逐局结果、损失、模型、Adam、Python/CPU/CUDA RNG、计数逐位一致 |
| 故障边界 | 14类检查全部符合预期；采样失败后拒绝保存/继续，重载后重做第3批与连续运行一致 |
| 高候选/多请求 | 8769及333候选同时评分、全部覆盖；与既有CUDA逐请求实现最大绝对差1.490116e-8 |
| 独立磁盘审计 | 日程/样本/更新/设备计数/时延分位数/完整检查点/源码ZIP重新核对通过 |
| 审计负例 | 样本总数、候选截断、时延分位数、负例覆盖、恢复记录5种篡改被拒绝 |
| CLI边界 | 既有输出目录、保留seed、26批超预算、重复seed均以退出码2拒绝；无非法输出目录 |
| 全套测试 | Python3.14：84/84通过；Python3.13：82通过，2项CUDA测试明确跳过 |
| 历史保护 | P2/P3a/P3b各40/45/28项注册源文件原hash一致；P3a/P3b独立审计重跑通过 |

高候选同次交替5次测量：旧逐块传输实现p50 **92.723ms**、p95 **93.726ms**；新批量上传实现p50 **84.917ms**、p95 **85.462ms**。计时包括Python编码、上传、完整CUDA评分与CPU返回。只报告本机本次少量重复，不将其视为长期性能承诺。不得用历史P3b的不同运行状态时延计算提速倍数；每局更新与每4局更新的训练循环也不能直接比较算法效率。

## 检查点与故障证据

最终文件`artifacts/evaluations/p3c-gpu-wave-v1/wave-4/checkpoint.pt`：4,948,937字节，SHA256：

`fc69a095c165818864a9daad742053eb3fd266135ae494a914c6e29dcde9fbef`

检查点绑定版本、源码、运行时、训练配置、计数、seed历史、模型、完整Adam、独立Python策略RNG、torch CPU RNG和所有CUDA RNG。仅受限weights_only加载；先校验后恢复，拒绝坏检查点时全局RNG保持原状态。写入临时目录并flush/fsync后重命名提交；Windows实测提交前异常不产生正式目录，未完成目录被拒绝加载，既有检查点拒绝覆盖。这不是断电耐久性或所有文件系统上的原子提交证明。

14类检查：版本、计数、越界seed、重复seed、Adam step、CUDA RNG结构、非有限参数、截断文件、覆盖旧目录、提交前中断、读取未提交目录、采样故障、失败后保存、失败后继续。负例载荷和中断临时目录作为测试证据保留在`negative-cases/`，它们不是可用候选。

## 复现与证据清单

```powershell
py -3.14 scripts/p3c_acceptance.py --output artifacts/evaluations/my-p3c-repeat
py -3.14 scripts/audit_gpu_training.py artifacts/evaluations/my-p3c-repeat
py -3.14 scripts/validate.py
```

固定交付目录`artifacts/evaluations/p3c-gpu-wave-v1/`内包括：事前preregistration与33项源码ZIP、逐批training、high-branch原始分数/计时、wave-1..4完整检查点、fresh-process-resume及子进程日志、resume-audit、negative-audit、controller-audit、controller-negative、report。测试日志位于其父目录`p3c-unittest-python314.log`和`p3c-unittest-python313.log`。独立控制器负例入口为`experiments/p3c/controller_checks.py`，其hash记在controller-negative；该附加审计文件不是训练算法的一部分。

采用reproducibility-skill进行种子、版本、入口和来源追踪；沿用项目现有artifacts目录契约，本包不产生论文图表。总控负责协议、检查点、入口、实验与独立审计；gpt-6-sol/high仅实现限定训练模块和4个测试，总控重读并完成实际GPU验收。

## 尚未满足的门槛

- 未执行长期训练或正式独立棋力评测，16手仅为工程验证；恢复复跑不增加独立样本数。
- 不支持半局/半批续接，失败批从上一成功检查点重做；CLI日志与检查点并非跨文件事务，故障时以已提交检查点计数为准。
- 精确恢复只在本机本次运行时验证；不承诺跨驱动、跨设备、跨版本或代码变化后的逐位结果。
- 外部推理watchdog尚未交付，当前1000步护栏不能杀死卡在单次网络调用内的进程。
- P3d需冻结训练预算、候选hash/冻结时点、未查看验证seed、成组评测和晋级门槛；贪心仍为主基线，随机/队友启发式作为并列报告对照；保留测试集继续封存。
- 桌面P4、增强P5、完整比赛P6未启动。
