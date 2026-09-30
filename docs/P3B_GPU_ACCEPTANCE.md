# P3b GPU 运行时验收

2026-09-25 总控结论：**ACCEPTED_GPU_RUNTIME_V1**。用户已指定直接使用GPU，原CPU/GPU对比路线停止；P3整体仍为 **PARTIALLY_ACCEPTED**。

## 已执行结果

| 项目 | 本机实测 |
|---|---|
| 运行时 | Python3.14.3、PyTorch2.12.0+cu130、CUDA13.0 |
| 硬件 | RTX5060 Laptop GPU、8151MiB、capability12.0、驱动610.47 |
| 网络计算 | 原DMCNetwork，float32，CUDA推理/损失/反向/Adam；TF32关闭，确定性模式 |
| 固定工作量 | 开发seed102200..102215，共16手；2484条决策样本；47次参数更新 |
| 实际设备证据 | 2300次策略action层hook在CUDA；47次训练forward及329次参数梯度hook在CUDA |
| 合法终局/研究回放 | 16/16通过 |
| 高候选覆盖 | 8769个合法候选，34块256+1块65；20次测量，无截断 |
| 高候选评分墙钟 | p50 1058.6733ms，p95 1152.4812ms；含编码、传输、评分及返回CPU |
| 16手学习循环 | 96.972秒，含真实环境、特征、网络、设备验证和回放 |
| 完整探测 | 124.598秒，含高候选重复测试、初始化与权重存储 |
| 独立磁盘审计 | 候选重建、原始分位数、日程、损失、更新数、CUDA证明、权重、28项源码ZIP均通过 |
| 历史保护 | P2的40项、P3a的45项注册文件保持一致；原P3a候选及其48局历史证据重新复算通过 |
| 审计负例 | 缺候选、错误分位数、重复对局、CPU梯度标志、CPU损失设备、错误权重hash共6类全部拒绝 |
| 入口保护 | 已有输出目录拒绝覆盖，报告SHA不变 |

16手耗时分项：枚举2.321秒、完整候选评分77.127秒、更新10.571秒、回放1.530秒、局内其他5.386秒；余下是循环日志、最终参数哈希等开销。当前GPU执行包含同步、逐块传输和逐步完整性检查，尚未进行批量推理优化。以上是小规模有界运行的实际墙钟，不能据此承诺长期吞吐或桌面响应时间。此次没有继续CPU学习或进行CPU/GPU速度对比。

## 实验权重与边界

`artifacts/evaluations/p3b-gpu-v1/runtime-weights.pt`，1,645,521字节，SHA-256：

`bf9c7c9a82ab6f8a9f713bb308df3a0914d06529ec9cdbe7b1cd4667ff3a63ef`

独立审计按`weights_only=True`读取，验证有限float32参数、参数内容哈希及初末参数确有变化。它是16手GPU运行时实验的最终权重，没有保存完整Adam/RNG状态，不是可恢复检查点，不能直接通过原CPU版train.py恢复。不替换P3a固定候选；没有对手评测、棋力结论或晋级；验证集与保留测试集未使用。

GPU网络计算通过不等于整个训练系统通过。下一包应实现GPU批量评分/持续采样、明确模型同步边界以及GPU训练检查点与恢复，完成后再冻结正式训练预算和独立棋力评测协议。所有合法候选仍须完整评分。

## 证据与复现

固定证据目录：`artifacts/evaluations/p3b-gpu-v1/`。包括preregistration.json、source-snapshot.zip、report.json、progress.jsonl、runtime-weights.pt、controller-audit.json、controller-negative.json、p3a-protection-audit.json、cli-boundary.json。

```powershell
py -3.14 scripts/probe_gpu_learning.py --output artifacts/evaluations/my-gpu-repeat
py -3.14 scripts/audit_gpu_probe.py artifacts/evaluations/my-gpu-repeat --output artifacts/evaluations/my-gpu-repeat/controller-audit.json
py -3.14 experiments/p3b/gpu_controller_negative.py artifacts/evaluations/my-gpu-repeat --output artifacts/evaluations/my-gpu-repeat/controller-negative.json
```

直接依赖`requirements-learning-gpu.txt`，实际依赖闭包版本`requirements-learning-gpu.lock`；解释器/驱动/来源见`artifacts/evaluations/p3b-gpu-environment.json`。本轮复用了已有共享GPU运行时，没有新装或卸载全局包，没有验证从空环境重建全部依赖。采用reproducibility-skill进行种子、入口、版本和证据追踪；总控负责设计、修正计时口径、运行与独立审计；gpt-6-sol/high只实现边界明确的探测脚本。

## 旧环境清理状态

**未完成删除。** 自动审批拒绝了删除命令，返回`blocked by policy`，该命令未运行。

- 项目`.venv`是此前拟安装2.9.1 CUDA运行时的空环境，约535194字节，没有安装torch；不是旧CPU训练实际所用的全局Python。
- `D:/Guandan-Runtime/wheels/torch-2.9.1+cu128-cp313-cp313-win_amd64.whl`在切换时已完成下载，2,862,036,321字节，官方SHA已通过；这是已停止路线的CUDA包，尚未删除。
- 全局Python3.13的CPU torch2.9.1+cpu被torchvision、ultralytics、ultralytics-thop依赖，未证明闲置，保留共享Python与这些依赖。当前GPU入口不依赖它。
- 下载和安装均没有继续运行。路径复查与阻断记录见`artifacts/evaluations/p3b-gpu-switch-cleanup.json`。保留旧CPU实验记录与固定候选作为历史证据，未将它们当作环境缓存清理。
