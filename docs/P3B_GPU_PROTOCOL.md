# P3b GPU 直接执行协议

2026-09-25 用户指定“别搞CPU了，直接上GPU”，覆盖此前CPU/GPU性能对比与新装2.9.1环境路线。使用本机已存在且实际CUDA前后向通过的 Python3.14.3 / PyTorch2.12.0+cu130 / RTX5060 Laptop GPU。此次不继续CPU学习基准，不修改共享运行时。规则枚举、环境管理、特征构建和磁盘操作仍在CPU上执行，所有网络推理、损失、反向传播及Adam更新在CUDA上执行。

版本 `gd-gpu-runtime-v1`，事前冻结16手开发集seed102200..102215，级牌2+(i%13)、首攻i%4。初始化seed271828、float32、torch线程1、TF32关闭、确定性算法及CUBLAS_WORKSPACE_CONFIG=:4096:8；epsilon0.1、lr0.001、batch64、完整候选chunk256。使用原DMCNetwork和玩家可见特征，不改变规则，不把发牌seed或完整状态传策略。

开始先验证8769候选构造手牌，全部合法候选进行CUDA评分，暖机后20次记录原始墙钟，保存分块覆盖和全部分数。学习按每副完整自我对局的团队终局±1更新，一次随机打乱轨迹遍历；每局完整研究回放校验。记录网络/梯度实际device、有限损失与参数、逐局样本数/更新次数/摘要/耗时，失败写failure.json。真实总墙钟覆盖Python和设备传输，不仅计GPU内核。CUDA计时边界同步。

输出目录必须不存在；开始前保存preregistration与源码哈希ZIP，结束比对源码。完成16手后权重另存runtime-weights.pt及SHA-256，仅是运行时实验权重，不是已有CPU检查点的GPU恢复，也没有保存完整优化器/RNG恢复协议。不覆盖P3a候选、不进行晋级或胜率评测、不访问验证与保留测试集。

验收范围为 `ACCEPTED_GPU_RUNTIME_V1`：本机CUDA网络评分、学习更新、合法终局与回放、完整候选覆盖、磁盘独立复算及历史源码保护。P3整体仍PARTIALLY_ACCEPTED；后续GPU训练检查点/恢复、持续采样与模型同步、正式训练预算和独立棋力评测仍须独立验收。

旧CPU及CPU/GPU对比记录保留为历史证据。项目空`.venv`及替代路线下载包属于可清理项；全局Python3.13的CPU torch被torchvision/ultralytics依赖，未证明闲置，不自动删除共享Python或其依赖。实际清理结果以单独收据和路径复查为准。
