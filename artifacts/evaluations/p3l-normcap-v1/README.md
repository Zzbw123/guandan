# P3l 研究交付与P3工程收口

2026-09-27。P3l包验收ACCEPTED_NORMCAP_AUXILIARY_STUDY_V1；P3全阶段工程ACCEPTED_LEARNING_LOOP_V1；棋力NOT_ESTABLISHED，model_promoted=false。唯一主候选对greedy 28.85% [25.38%,32.31%]，不替换greedy。

从项目根目录运行。scripts/run_gpu.ps1使用已验证GPU解释器，先预检CUDA。只使用本机已有运行时，不需安装依赖。

```powershell
# 从冻结记录完整重演3600手训练和4368评测、复算统计及15类负例之外的主审计
./scripts/run_gpu.ps1 experiments/review_p3l.py artifacts/evaluations/p3l-normcap-v1

# 核对已绑定的逐batch探索/打乱/候选分母证据；已存在文件只在hash相同后复用
./scripts/run_gpu.ps1 experiments/verify_p3l_batch_order.py artifacts/evaluations/p3l-normcap-v1

# 全阶段证据链复核；输出必须是新文件（重复时换文件名）
./scripts/run_gpu.ps1 experiments/controller_p3_closeout.py artifacts/evaluations/p3l-normcap-v1 artifacts/evaluations/p3l-normcap-v1/p3-completion-recheck.json

# 按同一固定预算和日程重跑，目标目录必须不存在
./scripts/run_gpu.ps1 scripts/p3l_run.py artifacts/evaluations/p3l-repeat-v1
```

重跑同一206000..206064只是复现，不是新的独立验证；不得将调参后重用这些发牌标为未查看。run会拒绝工程receipt、源码、历史seed清单或输入漂移；若项目继续演进，使用已保存source-snapshot.zip和绑定输入恢复同版本，或另行制定新协议，不能改旧清单绕过拒绝。保留测试9000000..9009999未使用。

## 证据链

- preregistration.json、source-snapshot.zip、inputs-snapshot.zip：196份冻结源码/协议、620输入绑定、工程门槛和固定日程。
- training/：六作业原始波次/研究牌谱、teacher-200输入谱系、最终600手专用检查点；继承200手不在episodes字段中，不能忽略reset.json。
- evaluations/：八个完整评测作业，最终候选SHA、逐局结果/耗时/全信息研究牌谱/行为计数。研究牌谱仅供实验管理，不传给策略。
- receipt.json：原始执行完成；此状态本身不等于验收。
- controller-audit.json、controller-negative.json：完整磁盘独立审计、配对统计、15类篡改负例。包审计时P3尚待收口，阶段最终结论以后续p3-completion-audit.json为准。
- batch-order-audit.json、batch-order/：所有1800批独立重建探索/打乱和逐状态候选分母。
- controller-actual-batches.json：六个真实首批次的稠密目标/梯度/Adam数值抽查，共1536状态/19340候选；不声称逐更新重算全部梯度。
- descriptives/：1800行batch-gradients.csv、273行paired-deal-groups.csv、summary.json和trace.json；没有新的模型选择或假设检验。
- p3-completion-audit.json：P3全阶段工程PASS、12份历史源码注册、654项历史不可变产物复核。
- delivery-receipt.json：最终交付文件/验收文档/控制器源码hash索引。

P3l专用版本：gd-p3l-checkpoint-v1 / gd-p3l-wave-v1。只接受同源码、同本机确定性CUDA运行时的成功wave边界恢复；不承诺局中恢复、跨设备逐位一致、干净机器重建或无限时长稳定。

完整结论见项目docs/P3L_ACCEPTANCE.md和docs/P3_COMPLETION_ACCEPTANCE.md。215项GPU全套测试日志在父目录p3l-full-tests-gpu.log；工程独立数值/恢复证据在父目录p3l-controller-*、p3l-resume-test.json、p3l-engineering-acceptance.json。早期开发测试与源码漂移拒绝记录完整保留。
