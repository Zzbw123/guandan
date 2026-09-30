# 掼蛋强化学习研究工具

按 [开发方案](掼蛋强化学习工具开发方案.md) 开工。当前优先交付可复现的规则与研究环境；用户已指定最终产品为桌面端。阶段结果与尚未达到的门槛以 [docs/STATUS.md](docs/STATUS.md) 为准。

GPU命令统一使用 `./scripts/run_gpu.ps1`。2026-09-26复核发现 `py -3.14` 已指向另一套没有torch的Python；原GPU环境仍可用。启动脚本从已有GPU环境记录读取解释器，先检查CUDA再执行项目脚本。必要时可用环境变量 `GUANDAN_GPU_PYTHON` 明确指定已验证的解释器；严格检查点加载仍会核对版本、源码和运行时。

要求 Python 3.13 或更新版本。P0/P1 使用标准库，无需下载依赖、权重或安装 CUDA。以下 PowerShell 命令在项目目录执行：

```powershell
python scripts/validate.py
python scripts/play_hand.py --seed 42 --level 2 --agent team --verbose
python scripts/benchmark.py --hands 100 --workers 1 --output artifacts/evaluations/my-smoke.json
```

`play_hand.py` 自动重演验证并写入全信息研究牌谱；每一步的实体牌和配牌声明可供定位规则问题。研究牌谱包含所有初始手牌，仅用于实验与调试，不是普通玩家视角数据。`benchmark.py` 输出源文件哈希、种子、规则版本、候选数、枚举耗时、不变量/重演检查与逐批进度；已存在的结果文件不会被覆盖。

```powershell
python scripts/benchmark.py --hands 10000 --workers 8 --replay-every 1 --output artifacts/evaluations/my-stress-10000.json
powershell -NoProfile -File scripts/hardware.ps1
```

四个座位轮换首攻，级牌覆盖 2..A，随机/贪心/队友启发式混合跑局。这是环境压力验收，不能据其团队获胜次数推断棋力；严格配对评测与置信区间属于 P2。

P2 独立评测入口：

```powershell
# 完整 27 场景，130 份原始发牌，46,800 副变体对局
python scripts/evaluate.py --config configs/evaluation/p2-validation.json --output artifacts/evaluations/my-p2-validation

# 从原始记录复算统计、性能分位数和哈希
python scripts/evaluate.py --audit artifacts/evaluations/my-p2-validation
```

结果目录包含事前配置与哈希 `preregistration.json`、逐局 `results.jsonl`、压缩原始耗时 `measurements.jsonl.gz`、全矩阵 `report.md`/`report.json` 和审计 `audit.json`。置信区间以原始发牌为组、按级牌分层重采样；失败或缺失局不能被删除后继续出具有效结果。保留测试集不由该入口执行。规程见 [P2_PROTOCOL.md](docs/P2_PROTOCOL.md)。

P3a 已完成小规模 CPU 学习闭环。历史CPU入口使用本机已验证的 **Python 3.13.5 + PyTorch 2.9.1+cpu**，依赖记录见 `requirements-learning-cpu.txt`。基础规则和P2仍可仅使用标准库；当前Python3.14已具备下文GPU运行时。

```powershell
# 独立复算已交付的CPU实验
py -3.13 scripts/audit_learning.py artifacts/evaluations/p3a-cpu-v1
# 重跑4局训练、精确恢复核验与48局集成评测；输出目录必须尚不存在
py -3.13 scripts/p3_acceptance.py --output artifacts/evaluations/my-p3a-repeat
# 新的有界训练；单次入口允许1..100局，只使用development seed
py -3.13 scripts/train.py --episodes 4 --seed-start 100600 --output artifacts/evaluations/my-dmc-training
# 从固定候选的完整对局边界恢复
py -3.13 scripts/train.py --resume artifacts/evaluations/p3a-cpu-v1/candidate --episodes 4 --seed-start 100504 --output artifacts/evaluations/my-dmc-resume
```

检查点包含网络、Adam、随机数、使用过的发牌seed和版本/源码哈希；不支持半局恢复，输出不覆盖既有目录，恢复时拒绝不兼容运行时和改动后的算法源码。发牌seed只用于环境管理，策略只接收玩家观测及完整合法动作。验收细节见 [P3A_ACCEPTANCE.md](docs/P3A_ACCEPTANCE.md)。

目录：`src/guandan/rules` 为权威牌型逻辑，`env` 为单副牌状态机，`agents` 仅接收合法玩家观测；`learning`实现特征、DMC、训练、检查点与开发集集成评测。P4 桌面端复用同一引擎，P6 扩展完整升级赛制。

规则以 [gd-hand-v1](docs/rules.md) 为准：双配牌声明、所有玩家余牌张数公开等选择明确记录，不称为官方赛事认证实现。P3a只训练4局的历史工程模型对greedy和team各0/16，对random为3/16，仅2个原始发牌组；P3d已执行GPU模型独立验证，未达到主比较门槛。模型没有晋级，不提供校准胜率提示，不代表高水平掼蛋棋力。P3整体仍为部分完成，当前下一包见文末及STATUS.md。

2026-09-25 用户改为直接GPU执行。当前入口使用本机已有 **Python3.14.3 + PyTorch2.12.0+cu130**，协议见 [P3B_GPU_PROTOCOL.md](docs/P3B_GPU_PROTOCOL.md)。网络推理、反向与Adam更新全部使用CUDA；CPU承担规则、特征与实验管理。直接依赖为 `requirements-learning-gpu.txt`，实际已安装的传递版本记录为 `requirements-learning-gpu.lock`；本轮复用了既有GPU环境，没有声称完成全新安装重建。

```powershell
# 固定16手开发集GPU学习运行时验证；输出目录必须不存在
./scripts/run_gpu.ps1 scripts/probe_gpu_learning.py --output artifacts/evaluations/my-p3b-gpu
# 从磁盘独立复算、核对CUDA执行证据、源码和实验权重
./scripts/run_gpu.ps1 scripts/audit_gpu_probe.py artifacts/evaluations/my-p3b-gpu --output artifacts/evaluations/my-p3b-gpu/controller-audit.json
```

输出含事前配置、源码ZIP、8769候选的完整CUDA评分与20次原始耗时、逐手损失/更新/回放/设备hook证据和 `runtime-weights.pt`。该权重仅供运行时实验，没有完整Adam/RNG恢复状态，不是正式可恢复训练检查点，不替换P3a固定候选；训练棋力和模型晋级仍未验证。

本机P3b固定16手已完成，2484条样本、47次CUDA更新，独立审计通过；历史实测耗时和权重哈希见 [P3B_GPU_ACCEPTANCE.md](docs/P3B_GPU_ACCEPTANCE.md)。后续批处理与可恢复训练已在P3c实现，见下文。

先前CPU/GPU对比协议与脚本保留作历史研究材料，未完成正式CPU/CUDA对比验收；CPU smoke只用于排查入口。当前不再按旧方案建立`.venv`或下载2.9.1运行时。环境清理状态见 `artifacts/evaluations/p3b-gpu-switch-cleanup.json`，不得把清理请求视为已经删除。

2026-09-26 **P3c GPU同步批量训练与跨进程恢复已验收**。当前GPU训练入口为`train_gpu.py`：4环境同步采样、合并完整合法候选评分、整批终局后更新一次样本集；模型、Adam及Python/CPU/CUDA随机数状态在成功批边界保存。新包位于`src/guandan_gpu/`，旧CPU模块与规则源保持原样。

```powershell
# 独立复算已交付GPU训练证据
./scripts/run_gpu.ps1 scripts/audit_gpu_training.py artifacts/evaluations/p3c-gpu-wave-v1
# 重跑固定16手、跨进程恢复与14类故障验证，使用新的输出目录
./scripts/run_gpu.ps1 scripts/p3c_acceptance.py --output artifacts/evaluations/my-p3c-repeat
# 新的4批、16手开发集GPU训练，每批自动提交检查点
./scripts/run_gpu.ps1 scripts/train_gpu.py --waves 4 --seed-start 103100 --output artifacts/evaluations/my-gpu-training
# 从已交付完整检查点续训2批（8手），禁止重复使用该模型训练过的seed
./scripts/run_gpu.ps1 scripts/train_gpu.py --resume artifacts/evaluations/p3c-gpu-wave-v1/wave-4 --waves 2 --seed-start 103016 --output artifacts/evaluations/my-gpu-resume
```

CLI限制每次1..25批，只允许开发seed100000..109999；既有输出目录不覆盖。恢复要求版本/源码/运行时匹配，半批故障从上个成功检查点重做；`.incomplete-*`目录不能作为已提交检查点。16手2648样本/12更新、逐位恢复、测试与时延口径见 [P3C_ACCEPTANCE.md](docs/P3C_ACCEPTANCE.md)。后续固定预算训练与独立验证已在P3d执行，见下文。

2026-09-26 **P3d固定预算训练与独立验证工程通过，模型未晋级**。一次从头400手GPU训练产生48862样本、231次更新；固定最终模型在65个原始发牌组、1560个配对变体对局中，对greedy胜率22.12%（95%成组区间19.04%–25.19%），对random为81.92%，对team为18.85%。本候选未达到预设主比较门槛，保留贪心主基线。详见 [P3D_ACCEPTANCE.md](docs/P3D_ACCEPTANCE.md)。

```powershell
# 只读独立复算：原始分组统计、完整1560局磁盘牌谱、候选及源码hash
./scripts/run_gpu.ps1 experiments/p3d_review.py artifacts/evaluations/p3d-validation-v2
# 固定预算复现；约400手训练+1560局验证，输出目录必须不存在
./scripts/run_gpu.ps1 scripts/p3d_run.py --output artifacts/evaluations/my-p3d-repeat
# 为新复现目录生成独立审计和负例回执（不覆盖已有回执）
./scripts/run_gpu.ps1 experiments/p3d_review.py artifacts/evaluations/my-p3d-repeat --write
```

P3d入口固定100批训练预算，并独立于通用`train_gpu.py`的25批单次限制；训练/评测各有1800秒进程总截止。GPU策略置于spawn进程，单请求2秒硬截止，传输内容只有玩家观测与完整合法动作。固定基线在环境进程中仍用软时限。测试故障注入模式不向正式CLI暴露。

复现仍使用原验证日程，不是新的独立测试。201000..201064已查看；保留测试集继续封存。候选仅在gd-hand-v1研究规则和本轮三个固定对手下验证，未验证多初始化泛化/完整比赛/桌面产品。后续开发诊断与训练规模研究见下文P3e。

2026-09-26 **P3e开发诊断与三初始化训练规模对照已验收，固定候选未晋级**。3个初始化各训练1600手，共4800手、620778样本、3029次CUDA更新；完整评测3224局。400→1600手的开发胜率差分别为+2.88、+33.17、+5.77个百分点，其中两组差值区间含0，改善幅度依赖初始化。事先指定的314370/1600手候选在65组新验证发牌上对greedy胜率20.19%（95%成组区间17.50%–22.88%），未达到门槛。详见 [P3E_ACCEPTANCE.md](docs/P3E_ACCEPTANCE.md) 和 [P3E_PROTOCOL.md](docs/P3E_PROTOCOL.md)。

```powershell
# 独立复算3224局牌谱、完整行为计数、训练检查点与统计
./scripts/run_gpu.ps1 experiments/review_p3e.py artifacts/evaluations/p3e-scale-v2
# 固定原预算复现；约4800手训练+3224局评测，不覆盖既有目录
./scripts/run_gpu.ps1 scripts/p3e_run.py --output artifacts/evaluations/my-p3e-repeat
./scripts/run_gpu.ps1 experiments/review_p3e.py artifacts/evaluations/my-p3e-repeat --write
```

新验证seed202000..202064已查看，重跑不构成新的独立验证。完整证据与发牌组CSV/PNG/SVG位于`artifacts/evaluations/p3e-scale-v2/`；v1因Python启动器指向变化在首个模型加载时失败，保留原记录。后续教师偏好回归配对对照已在下文P3f完成；P3整体仍为PARTIALLY_ACCEPTED。


2026-09-26 **P3f教师偏好回归与DMC配对实验已验收，模型未晋级**。3个初始化、两臂各800手，共4800手训练和4368局评测。相同阶段重置条件下，以前200手贪心偏好回归替换DMC，随后统一训练600手DMC。三个开发点估计均提高；固定初始化314380在65组新验证上对greedy为31.35%（95%区间27.88%–34.81%），比配对control高10.19个百分点（差值区间5.77–14.62）。这尚未达到棋力晋级门槛。完整证据见 [P3F_ACCEPTANCE.md](docs/P3F_ACCEPTANCE.md) 与 [P3F_PROTOCOL.md](docs/P3F_PROTOCOL.md)。

```powershell
# 从磁盘重新核验全部训练/评测牌谱、完整候选、权重衔接与统计，不重复训练
py -3.13 experiments/review_p3f.py artifacts/evaluations/p3f-teacher-v1
# 按原固定预算复现；重用已看验证日程，不构成新的独立验证
./scripts/run_gpu.ps1 scripts/p3f_run.py --output artifacts/evaluations/my-p3f-repeat
py -3.13 experiments/review_p3f.py artifacts/evaluations/my-p3f-repeat --write
```

P3f完整证据、图表及546行场景发牌组CSV位于`artifacts/evaluations/p3f-teacher-v1/`。GPU133项测试通过；当前规则、旧训练和旧评测源码保持原hash。最终checkpoint只含第二阶段600手DMC计数，前200手通过phase1/和reset.json关联；未来续训必须同时检查第一阶段已用种子。新验证203000..203064现已查看，保留测试继续封存。下一包P3g先诊断现有教师阶段的排序拟合质量及后续DMC的变化，再决定单一训练改进；桌面端仍后置。
