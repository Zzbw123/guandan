# P3a CPU 学习闭环协议

2026-09-24 总控事前冻结，版本 `gd-dmc-cpu-v1`。本工作包只验收学习工程闭环，不宣称棋力、不晋级、不启用保留测试集。P1/P2 已注册源文件保持不变。

## 玩家可见编码 gd-features-v1

编码入口严格只接受 `PlayerObservation` 和合法 `Action`；拒绝终局、非当前玩家和不匹配的规则/观测版本。不得输入 GameState、发牌 seed、摘要、研究回放或别人的手牌。

所有座位按 `(seat-player_id)%4` 变为本人、下家、队友、上家。54 个牌面用 `card%54` 排序，去除双副牌实体副本编号；计数除以 2。动作种类固定顺序 pass/single/pair/triple/full_house/straight/pair_chain/triple_chain/bomb/straight_flush/joker_bomb。

动作 149 维：实体消耗牌面计数54、种类one-hot11、main_rank one-hot18（0..17）、配牌声明65（rank 2..14，suit -1..3，rank优先，计数/2）、出牌数/27一维。配牌来源是同一红桃级牌牌面，保留声明秩与花色，不编码可交换副本ID。

当前公共状态449维，顺序为：本人手牌54；所有历史实际出牌的按相对座位累计牌面216；四人余牌数/27共4；四人的出完名次/4共4（未出完0）；本轮pass mask4；级牌one-hot13；上一出牌者one-hot5（无上一人索引4）；上一动作149（无则全0）。不编码state_version、terminal、settlement。

历史按时间保留最后16个公开事件，左侧补零。每事件159维：相对出牌者4、动作149、trick_closed1、本事件finished座位mask4、present1。按时间展平并接在449维后，得到状态向量2993维。更久历史的牌面消耗仍由公共累计特征覆盖；这是有限记忆基线，不声称完全信息集编码。合法动作集合始终完整，禁止截断候选。

## 网络与学习

共享四座位网络：state Linear(2993,128) + action Linear(149,128,bias=False)，ReLU，Linear(128,64)，ReLU，Linear(64,1)，Tanh。动作价值预测范围[-1,1]。DMC采用完整自我对局的团队终局±1回报，gamma=1，不引入升级数塑形。每局结束后按所有实际决策的(state,action,own_team_reward)执行一遍随机打乱的minibatch MSE/Adam更新；不重用跨局缓存，不在局内改参数。

CPU确定性模式、torch线程数1，默认学习率0.001、batch64、epsilon0.1、评分chunk256。利用state投影共享降低成本；评估所有候选，等分时选原列表最早候选。epsilon均匀探索同样使用完整列表。策略随机数独立于发牌seed。

模块接口：`learning.encoding.encode_observation(obs)->tuple[float,...]`；`encode_action(action)->tuple`；常量 STATE_DIM=2993/ACTION_DIM=149/FEATURE_VERSION。`learning.model.DMCNetwork.forward(states,actions)->Tensor[B]`；`score_actions(model,obs,actions,chunk_size=256)->Tensor[N]` 返回CPU结果；`DMCAgent(model,chunk_size=256).act(obs,actions)` 不接收环境。

`learning.training.Trainer(config)` 的config为JSON兼容dict，含 seed/epsilon/lr/batch_size/chunk_size；拥有model/optimizer/rng（random.Random）/episodes/updates/config。`train_episode(deal_seed,level,starting_player)->dict` 仅允许development 100000..109999；环境管理层生成轨迹，策略决策只调用观测/合法动作编码；1000步护栏，全部特征/损失/梯度须有限；返回episode、seed、level、steps、samples、mean_loss、team_rewards、terminal_digest、updates、replay_verified，重演完整研究牌谱但不把它传策略。

## 检查点与恢复

只在完整对局更新后保存，边界为episode_boundary；不承诺中途崩溃续接。保存model/Adam状态、Python策略RNG、torch CPU RNG、已完成episodes/updates、训练配置、规则/观测/动作/特征/网络版本、运行时版本和已使用训练seed。受限weights_only加载；核对文件SHA-256、版本、shape、配置及预算；禁止覆盖旧检查点。恢复后下一局必须与未中断训练产生逐位一致的CPU模型/优化器/RNG状态及相同轨迹摘要。CPU实测结论不推广到其他平台/版本/GPU。

## 本轮事前预算与验收

训练开发seed 100500..100503，共4局（恢复复跑不增加独立样本）。初始化seed=314159；前2局存检查点，随后2局分别连续/重载执行，比较逐位状态。固定最终候选后只作开发集集成评测：seed 101000、101001，每seed按P2同队对称方案8变体，对greedy/random/team共48局；模型+模型对同策略基线队。主要对手greedy，所有结果均报告，2原始发牌组不出具显著性或晋级结论。评测清单提前绑定checkpoint哈希和固定日程。模型评测epsilon=0。

独立验收包含：不泄露隐藏牌/seed、相对座位及实体副本交换不变性、配牌声明可区分、全候选分块评分、团队奖励方向、更新参数确实改变、保存加载动作一致、篡改/版本不符拒绝、完整恢复一致、三基线合法终局、P1/P2旧测试回归及受保护源码哈希。本轮不改规则。

后续P3b才锁定GPU实测、并行采样、训练预算和未查看验证集的晋级协议。本轮不安装或升级现有全局依赖，不启动长时训练。

实现参考仅用官方接口文档：[PyTorch checkpoint教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models)，采用state_dict及weights_only加载。依赖以本机成功执行的版本记入验收，不以网页最新版推断本机兼容性。
