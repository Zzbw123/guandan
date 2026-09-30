# P5c复现

在项目根目录使用已记录CUDA解释器，输出目录必须不存在：

```powershell
./scripts/run_gpu.ps1 experiments/p5c_diagnostic.py artifacts/evaluations/p5c-diagnostic-reproduction-v1
./scripts/run_gpu.ps1 experiments/controller_p5c_audit.py artifacts/evaluations/p5c-diagnostic-reproduction-v1
./scripts/run_gpu.ps1 scripts/p5c_test.py
```

正式证据只读。preregistration.json绑定来源日志/模型/协议/实现；receipt.json绑定计算产物，controller-audit.json记录独立审计，delivery-receipt.json绑定完整交付。重新运行不会产生新发牌或新训练；全部输入仍为P5b训练数据。全项目测试日志及源码回执位于上一级p5c-full-tests-gpu-v1.log/.json。
