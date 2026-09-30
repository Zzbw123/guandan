# P5a 工程验收复现入口

在项目根目录使用PowerShell；所有输出路径必须全新，已有工程数据不覆盖。

```powershell
./scripts/run_gpu.ps1 scripts/p5a_test.py
$env:RUN_P3F_CUDA='1'
$env:RUN_P3G_CUDA='1'
./scripts/run_gpu.ps1 scripts/validate.py
./scripts/run_gpu.ps1 scripts/p5a_run.py artifacts/evaluations/p5a-pool-rerun-NEW
./scripts/run_gpu.ps1 scripts/p5a_run.py artifacts/evaluations/p5a-pool-resume-NEW --resume artifacts/evaluations/p5a-pool-rerun-NEW/wave-2
./scripts/run_gpu.ps1 experiments/controller_p5a_audit.py artifacts/evaluations/p5a-pool-rerun-NEW artifacts/evaluations/p5a-pool-resume-NEW artifacts/evaluations/p5a-audit-NEW.json
```

只复核本次已保存结果时：

```powershell
./scripts/run_gpu.ps1 experiments/controller_p5a_audit.py artifacts/evaluations/p5a-pool-v2 artifacts/evaluations/p5a-pool-resume-v2 artifacts/evaluations/p5a-audit-recheck-NEW.json
```

成功验收：docs/P5A_ACCEPTANCE.md；控制器报告：artifacts/evaluations/p5a-controller-audit-v2.json；全套测试：artifacts/evaluations/p5a-full-tests-gpu-v1.log。source-snapshot.zip保存绑定的源文件与控制器，恢复要求相同源/运行时；不要在当前项目覆盖历史源来绕过检查。未证明干净机器重建或跨硬件恢复。

16手只验证工程；恢复执行、重跑和单测均不增加独立发牌数，不代表棋力改善。最新主候选结论仍为P3l NOT_ESTABLISHED，保留测试继续封存。
