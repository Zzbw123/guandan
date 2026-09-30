# P0 复用预审（2026-09-24）

本轮选择：规则、状态机、基线自主实现；未引入任何外部项目代码、训练数据或模型权重。P0 核心运行无第三方依赖。不把候选仓库 README 当验收结论。

| 候选 | 查验范围与发现 | 本轮决策 |
|---|---|---|
| [rlcard-guandan](https://github.com/Choysang/rlcard-guandan) | 根目录 LICENSE 为 MIT；查看了 rules.md，其自述包括规则修复、双上补名次、圈边界辅助字段简化与 others_hands 全信息特征。未审计全部源码和附带权重授权。 | 仅做差异清单参考；不依赖、不复制。避免把全信息特征引入本项目策略。 |
| [DanZero+](https://github.com/submit-paper/Danzero_plus) | README 的依赖面向旧 Python/TensorFlow/PyTorch 组合；本轮根目录页面未见明确许可证文件。未审查传递授权。 | 仅保留算法研究入口，暂不移植代码/权重。 |

候选具体页面：[rlcard rules](https://raw.githubusercontent.com/Choysang/rlcard-guandan/main/docs/rules.md)、[MIT 文本](https://raw.githubusercontent.com/Choysang/rlcard-guandan/main/LICENSE)。这是公开文档预审，不是第三方引擎差分验收；后续正式复用必须固定 commit、文件范围、许可证及哈希。

规则权威边界：[体育总局关于竞技掼蛋规则的报道](https://www.sport.gov.cn/n20001280/n20067662/n20067613/c25277668/content.html)可确认规则体系；未取得官方完整技术条文逐条核验。本项目依 `rules.md` 明确选定的研究变体开发，版本不暗示官方认证。

其他方案参考（DouZero、OpenGuanDan、DanLM）本轮没有复制或技术验收；不宣称已完成其开源尽调。P3 若需要复用，再单独审查。
