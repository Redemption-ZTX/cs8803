# PLAN-002: IL / Baseline Exploitation / MAPPO 双主线

- **日期**: 2026-04-13
- **状态**: 当前主线
- **负责人**:
- **取代关系**:
  - 本计划取代 [PLAN-001](plan-001-il-baseline-exploitation.md) 作为当前阶段总路线
  - [PLAN-001](plan-001-il-baseline-exploitation.md) 保留，作为上一个阶段的转向记录

## 1. 为什么需要 PLAN-002

到 2026-04-13 为止，我们已经有了三个比较稳定的判断：

1. 仅靠继续细调 shaping，已经进入收益递减区。
2. baseline exploitation 方向是对的：
   - baseline 是可无限查询的 teacher
   - baseline 本身也有可分析、可利用的弱点
3. 旧 centralized-critic 结论被说重了：
   - 旧 [shared_cc_warm225](../../ray_results/PPO_shared_cc_warm225_20260409_123447) 不是公平对照
   - 它不能用来否定 MAPPO / centralized critic

因此，现在的阶段性问题不再是：

- “继续把 shaping 调到更细”

而是：

- “如何同时利用 teacher 初始化、对手弱点、以及更强的多智能体训练结构”

所以我们把当前阶段总路线升级为双主线：

- **主线 A**：IL / BC bootstrap
- **主线 B**：MAPPO / centralized-critic 公平对照

同时保留：

- **并行分析线**：baseline weakness analysis
- **后期增强线**：ensemble wrapper
- **并行鲁棒性线**：opponent-pool fine-tune（见 [SNAPSHOT-018](../experiments/snapshot-018-mappo-v2-opponent-pool-finetune.md)）
- **并行适配探索线**：先测 `MAPPO + v4 shaping` 是否成立（见 [SNAPSHOT-020](../experiments/snapshot-020-mappo-v4-fair-ablation.md)），再决定是否值得为 team-level PPO v4 补 `FrozenTeamCheckpointPolicy`
- **并行根因验证线**：先用 [SNAPSHOT-022](../experiments/snapshot-022-role-differentiated-shaping.md) 测 “reward asymmetry 本身” 是否有用，再用 [SNAPSHOT-024](../experiments/snapshot-024-striker-defender-role-binding.md) 测真正的 striker/defender field-role specialization
- **冠军点增益线**：在 [SNAPSHOT-024](../experiments/snapshot-024-striker-defender-role-binding.md) 证明 field-role binding 可成立之后，再用 [SNAPSHOT-025](../experiments/snapshot-025-bc-champion-field-role-binding.md) 测它能否在当前冠军 [SNAPSHOT-017](../experiments/snapshot-017-bc-to-mappo-bootstrap.md) 底座上继续带来净增益；若出现频繁 `kl/total_loss=inf` 但行为未灾难性崩坏，再并行启动 [SNAPSHOT-025b](../experiments/snapshot-025b-bc-champion-field-role-binding-stability-tune.md) 仅做优化稳定性修复对照

## 2. 总目标

### 2.1 主目标

训练出一个稳定超过当前 shaping 主线，并有机会追平或超过旧强点：

- 当前 shaping 可信 best 参考：`0.746`
- 旧强点参考：
  - [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225) `0.764`
  - [role checkpoint-30](../../ray_results/PPO_role_cpu32_20260408_193132/RoleSpecializedPPOTrainer_Soccer_20344_00000_0_2026-04-08_19-31-54/checkpoint_000030/checkpoint-30) `0.786`

### 2.2 次目标

在作业 rubric 下，同时铺好：

- 性能 agent
- reward / observation modification agent
- imitation learning / novel concept bonus agent

## 3. 主线结构

### 3.1 Mainline-A: IL / BC Bootstrap

对应 snapshot：

- [SNAPSHOT-012](../experiments/snapshot-012-imitation-learning-bc-bootstrap.md)

核心问题：

- baseline teacher 能不能给我们一个比 scratch PPO 更强、更对齐的初始化？

阶段目标：

1. 稳定采 baseline self-play teacher 数据
2. 明确 team-level BC 数据格式
3. 训练 BC 起点
4. 在 BC 起点上接 PPO / MAPPO fine-tune

当前状态：

- baseline self-play teacher collection 已完成
- dataset manifest / shard 结构已确认稳定
- 最小版 BC trainer / deployment wrapper 已落地
- BC smoke train 与 checkpoint reload 已通过
- 下一阶段已正式拆出为 [SNAPSHOT-017](../experiments/snapshot-017-bc-to-mappo-bootstrap.md)：player-level BC bridge 与最小 `BC -> MAPPO` warm-start smoke 已通过，下一步进入正式 short-run / fair-run 对照

### 3.2 Mainline-B: MAPPO / Centralized-Critic 公平对照

对应 snapshot：

- [SNAPSHOT-014](../experiments/snapshot-014-mappo-fair-ablation.md)

核心问题：

- 在与当前主线尽量公平的预算和评估规则下，centralized critic / MAPPO 能不能真正带来增益？

阶段目标：

1. 不再沿用旧 `shared_cc` 的 orthogonal 结论
2. 用当前主线可比的预算和口径重开 centralized-critic
3. 判断它是否能减少当前 shaping 线最难解决的：
   - long-game instability
   - late defensive collapse

推荐首轮拆分：

1. `MAPPO + no shaping`
2. `MAPPO + shaping-v1`
3. `MAPPO + shaping-v2`

说明：

- 这三条应当作为同一组算法对照
- 不要和 `IMPALA / Ape-X / SAC` 混在同一轮首测中

## 4. 并行分析线

### 4.1 Baseline Weakness Analysis

对应 snapshot：

- [SNAPSHOT-013](../experiments/snapshot-013-baseline-weakness-analysis.md)

当前已经确认的弱点：

- baseline 不是开局最脆
- baseline 更像在长局里会漂
- 对弱对手的少数输局也集中在长局窗口

这条线的作用是：

- 给 IL fine-tune 提供目标
- 给 MAPPO / PPO 后续对手建模提供 exploitable hypothesis
- 给未来 wrapper / ensemble 设计提供依据

## 5. 后期增强线

### 5.1 Ensemble Wrapper

前置条件：

- 至少 `3` 个以上可信候选 checkpoint / agent wrapper

目标：

- 将多个候选做成一个真正可提交的 ensemble agent
- 不依赖“多提交几个 agent 就自动相当于 ensemble”

这条线在当前阶段不抢主资源，但应作为最终乘法器保留。

## 6. 当前明确暂缓的

以下方向仍暂缓：

- 进一步复杂化 shaping
- IMPALA 之外的更大算法切换（Ape-X / Rainbow / SAC 等）
- 辅助损失 / 表示学习
- MCTS-lite 推理
- 观测镜像增强
- PBT / self-play league 的大规模铺开

说明：

- 这些方向不是永久放弃
- 而是必须等 `IL` 和 `MAPPO` 至少有一条跑出明确结果后再决定

## 7. 评估规则

沿用 [engineering-standards.md](../architecture/engineering-standards.md#checkpoint-选模规则)：

1. 训练内只用 `baseline 50` 做候选筛选
2. 正式复核按 `top 5% + ties`
3. 主判据是 `baseline 500`
4. `random 500` 只用于最终 `1-2` 个候选补充确认
5. failure capture 跟随 `baseline 500` 的最终候选

## 8. 近期执行顺序

### 8.1 已完成

1. 建立 [PLAN-001](plan-001-il-baseline-exploitation.md) 并转向 IL / baseline exploitation
2. 建立 [SNAPSHOT-012](../experiments/snapshot-012-imitation-learning-bc-bootstrap.md)
3. 建立 [SNAPSHOT-013](../experiments/snapshot-013-baseline-weakness-analysis.md)
4. 修正 [SNAPSHOT-005](../experiments/snapshot-005-observation-memory-and-centralized-critic-ablation.md) 对 centralized critic 的旧负结论
5. 建立 [SNAPSHOT-014](../experiments/snapshot-014-mappo-fair-ablation.md)

### 8.2 现在开始做

1. 正式启动 team-level BC 训练
2. 开始 baseline weakness analysis 的第三条线：
   - baseline vs current best agent
3. 启动公平 MAPPO 首轮对照：
   - `no shaping`
   - `shaping-v1`
   - `shaping-v2`

### 8.3 然后再做

1. 比较 `BC -> PPO` 与 `scratch PPO`
2. 比较 `MAPPO` 与当前 shaping PPO
3. 只有在两条主线都拿到可信 best 之后，再做 ensemble

## 9. 当前判断

当前阶段最重要的变化不是“又多了一个算法候选”，而是：

- 我们不再把性能提升押在 shaping 微调上
- 而是同时押在：
  - **更强初始化**（IL / BC）
  - **更强多智能体训练结构**（MAPPO / centralized critic）
  - **更明确的对手利用**（baseline weakness analysis）

一句话总结：

> 当前主线已经从 “继续细调 PPO shaping” 升级为 “IL + MAPPO 双主线，baseline exploitation 为支撑，ensemble 为后期增强”。
