# P5e 复核入口

协议docs/P5E_PROTOCOL.md；结果docs/P5E_ACCEPTANCE.md。

在项目根目录使用记录的CUDA解释器：

```powershell
./scripts/run_gpu.ps1 scripts/p5e_test.py
./scripts/run_gpu.ps1 scripts/p5d_full_test.py artifacts/evaluations/p5e-full-tests-NEW.log
./scripts/run_gpu.ps1 experiments/controller_p5e_audit.py artifacts/evaluations/p5e-balanced-v1 --output artifacts/evaluations/p5e-audit-NEW.json
```

所有输出必须全新。审计缓存通过输入/源码/预注册哈希核对后可复用；上述审计仍会重做24wave数值与篡改检查。preregistration.json和源码/输入ZIP绑定运行；candidates.json在评测前锁定final；controller-audit.json记录独立验收，delivery-receipt.json绑定交付。

本轮验证208000..208064已查看。冻结入口scripts/p5e_run.py的再次运行只能称复现，不能当新验证；新的训练假设必须另立协议、独立目录和验证范围，不能覆盖当前数据。
