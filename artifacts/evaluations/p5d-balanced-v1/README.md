# P5d 工程复现

项目根目录PowerShell，使用记录的CUDA解释器；所有输出目录/日志路径必须全新：

```powershell
./scripts/run_gpu.ps1 scripts/p5d_test.py
./scripts/run_gpu.ps1 scripts/p5d_full_test.py artifacts/evaluations/p5d-full-tests-NEW.log
./scripts/run_gpu.ps1 experiments/p5d_package.py artifacts/evaluations/p5d-balanced-NEW
./scripts/run_gpu.ps1 experiments/controller_p5d_audit.py artifacts/evaluations/p5d-balanced-NEW
```

只复查正式证据时，调用controller_p5d_audit.py并用--output指定全新JSON路径即可，不重训。preregistration.json绑定协议/源码/输入，source-snapshot.zip保存源码，run-receipt.json绑定运行产物；controller-audit.json记录独立重算，delivery-receipt.json绑定最终交付。恢复要求相同源码和运行时，不能修改历史文件绕过来源校验。

两臂共32手仅16份发牌，恢复重复8手，均为工程检查；没有棋力验证或模型晋级。长期研究另立协议。
