# P3k 复现与证据入口

这是开发诊断交付，不是新训练/新评测或模型晋级。工程ACCEPTED，P3整体PARTIALLY_ACCEPTED。

在项目根目录运行。CUDA解释器由项目scripts/run_gpu.ps1记录；正式执行用Python3.14.3。py -3.13仅用于不需要torch的报告/绘图。

```powershell
./scripts/run_gpu.ps1 experiments/p3k_run.py --freeze
./scripts/run_gpu.ps1 experiments/p3k_run.py --run
./scripts/run_gpu.ps1 experiments/p3k_audit.py
py -3.13 experiments/figures_p3k.py
py -3.13 experiments/report_p3k.py
py -3.13 experiments/deliver_p3k.py --verify
```

冻结/计算/审计脚本使用独占创建，拒绝覆盖既有正式产物。上面前三条是生成该交付的历史顺序，不能在已有目录直接重跑；完整复现需把同一源码及所有哈希绑定输入复制到新的隔离项目根目录，保持相对路径，并让artifacts/evaluations/p3k-gradients-v1为空。无需改原协议或覆盖现有结果。图表后需人工视觉检查并生成visual-qa.json，之后才能生成报告。最终delivery --verify为当前项目可直接执行的只读入口。

全套测试历史命令：RUN_P3F_CUDA=1、RUN_P3G_CUDA=1，PYTHONPATH包含项目根/src/experiments，使用CUDA解释器运行scripts/validate.py；197项通过。开关仅作用于该进程环境。

数据：selection.json与probes.jsonl；汇总：outputs/metrics.csv、outputs/tables/source-model-summary.csv；图：outputs/figures/gradient-map.png/svg；实验记录：outputs/tables/experiment-log.csv。所有原始输入只读、源快照固定，前一STATUS原字节保存于previous-status.md。

局限：234个来源观察只有182个不同带标签观察；三个教师来源重复；128单候选；跨模型终局标签off-policy；梯度为固定权重逐状态诊断，不是实际batch/Adam历史。本包没有论文或提交审计。
