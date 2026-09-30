# P5b 复核与复现

当前目录绑定等手数、三初始化、固定最终候选研究。结果与边界见docs/P5B_ACCEPTANCE.md。

在项目根目录使用PowerShell，复核已保存数据：

```powershell
./scripts/run_gpu.ps1 scripts/p5b_test.py
$env:RUN_P3F_CUDA='1'
$env:RUN_P3G_CUDA='1'
./scripts/run_gpu.ps1 scripts/validate.py
```

重新完整审计全部牌谱、统计、首wave数值与篡改负例，写入全新报告（不使用审计缓存）：

```powershell
./scripts/run_gpu.ps1 experiments/controller_p5b_recheck.py artifacts/evaluations/p5b-pool-v1 artifacts/evaluations/p5b-recheck-NEW.json
```

输出文件必须不存在。原controller-audit.json和本目录数据保持不变。

正式重新训练需单独新目录和新的预注册设计；本轮207000..207064已经查看，不可当新验证。scripts/p5b_run.py记录的是这次冻结实验，重放仅能称复现，不能生成新的独立验证证据。训练和评测使用scripts/run_gpu.ps1绑定的GPU解释器。

源码快照：source-snapshot.zip；输入：inputs-snapshot.zip；独立审计代码：controller-snapshot.zip；候选：candidates.json；完整训练：training/*/waves.jsonl.gz；原始评测：evaluations/*/results.jsonl及replays.jsonl.gz；交付索引：delivery-receipt.json。
