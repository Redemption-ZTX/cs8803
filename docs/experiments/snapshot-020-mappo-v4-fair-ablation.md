# SNAPSHOT-020: MAPPO + v4 Shaping Fair Ablation

- **日期**: 2026-04-15
- **负责人**:
- **状态**: 已完成首轮结果

## 1. 背景

[SNAPSHOT-016](snapshot-016-shaping-v4-survival-anti-rush-ablation.md) 已经证明：

- `PPO + v4 shaping` 的官方 `baseline 500` 最好约 **0.768**
- 相比 `PPO + v2`，`v4` 的主要收益是：
  - 输局更长
  - `late_defensive_collapse / unclear_loss` 更少
- 但它的代价也很明确：
  - 更多失败被推向 `low_possession / poor_conversion`

[SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) 则证明：

- `MAPPO + shaping-v2` 能到 **0.786**
- centralized critic / shared-policy 结构本身是真正有增益的

因此，一个自然的问题是：

**如果把 `v4` 的 reward shaping 语义放进 MAPPO，而不是 PPO，会发生什么？**

这条线的重要性不只在“会不会比 `MAPPO+v2` 更强”，还在于它决定了下一步的工程方向：

- 如果 `MAPPO + v4 shaping` 自己就不成立，那么没有必要为了 `PPO v4` 去做 `FrozenTeamCheckpointPolicy`
- 如果 `MAPPO + v4 shaping` 成立，那么它就可以成为：
  - 一个新的 fair MAPPO 对照分支
  - 以及未来 [SNAPSHOT-023](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) 的更低风险替代路径

换句话说，`020` 的作用是：

> **先验证“v4 风格的 anti-rush / survival 目标”是否值得进入 MAPPO 主干。**

只有这件事成立，才值得考虑把 team-level PPO checkpoint 真的塞进 opponent pool。

## 2. 核心问题

### Q1

`v4` 的 shaping 收益，是否在 MAPPO 上依然成立？

### Q2

如果成立，它是：

- 真的提高了官方 `baseline 500`
- 还是只是把比赛拖长、却没有改善最终胜率

### Q3

如果它不成立，那么 `PPO v4` 更可能只是 PPO-specific 的补丁，而不是值得进入 pool 的新风格来源。

## 3. 实验设计

### 3.1 对照关系

本 snapshot 只测一件事：

- 在 [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) 的 fair MAPPO 设定下
- 保持 `multiagent_player + shared-policy + centralized critic`
- 只把 shaping 从 `v2` 换成 `v4`

也就是：

- **保留 MAPPO**
- **保留公平对照语义**
- **只改 reward shaping**

### 3.2 预期训练配置

- 入口：复用 [train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py)
- `variation = multiagent_player`
- `multiagent = True`
- `FCNET_HIDDENS = 512,512`
- `custom_model = shared_cc_model`
- `gamma = 0.99`
- `lambda = 0.95`
- `warm-start = scratch`

### 3.3 shaping 配置

在 `v2` 基础上叠加 `v4` 的两个新项：

- `defensive_survival_bonus`
- `fast_loss_penalty`

也就是：

- `time_penalty = 0.001`
- `ball_progress_scale = 0.01`
- `opponent_progress_penalty_scale = 0.01`
- `possession_bonus = 0.002`
- `possession_dist = 1.25`
- `progress_requires_possession = 0`
- `deep_zone_outer_threshold = -8`
- `deep_zone_outer_penalty = 0.003`
- `deep_zone_inner_threshold = -12`
- `deep_zone_inner_penalty = 0.003`
- `defensive_survival_threshold = -2`
- `defensive_survival_bonus = 0.001`
- `fast_loss_threshold_steps = 40`
- `fast_loss_penalty_per_step = 0.01`

### 3.4 训练预算

与 fair MAPPO 首轮保持同量级：

- `500 iter`
- 训练内仍用 `baseline 50 / random 50`
- 选模规则继续沿用 `top 5% + ties -> baseline 500`

## 4. 判据

### 4.1 主判据

正式 `baseline 500`：

- **若 ≥ 0.79**：说明 `MAPPO + v4 shaping` 至少有资格当一个成立的平行变体
- **若 > 0.786**：说明它超过当前 `MAPPO + v2` strongest scratch line

### 4.2 机制判据

至少满足一项：

1. 相比 `MAPPO + v2`，`late_defensive_collapse` 明显下降
2. 输局步数显著拉长，但赢局步数不过度拖慢
3. `low_possession / poor_conversion` 没有像 `PPO v4` 那样明显恶化

### 4.3 失败判据

任一触发即视为 `v4-style objective` 不值得进入 MAPPO 主干：

1. `baseline 500 < 0.77`
2. official 分数不升，且 failure capture 只显示“拖长败局”而无结构性收益
3. `low_possession / poor_conversion` 明显恶化，重复 PPO-v4 的问题

## 5. 对 023 的关系

本 snapshot 明确是 [SNAPSHOT-023](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) 的前置判断。

逻辑是：

- 若 `020` 不成立：
  - 说明 `v4` 风格本身不值得往 MAPPO 主干里引
  - 那么 `023` 的优先级就明显下降
- 若 `020` 成立：
  - 说明 `v4` 风格在 MAPPO 里有真实价值
  - 此时再讨论 `FrozenTeamCheckpointPolicy` 是否值得做，才更合理

也就是说：

**020 决定”v4 风格值不值得进主干”，023 才决定”要不要为了 team-level PPO v4 再补一层 opponent adapter”。**

**编号变更说明**：本文原使用 `021` 指代 FrozenTeamCheckpointPolicy adapter 工作。2026-04-15 编号 021/022 被重用于 `low_possession` 根因 A/B 假设检验（obs expansion / role-differentiated shaping），本 adapter 工作移至 **023**。

## 6. 不做的事

本 snapshot 不做以下事：

- 不改 opponent pool
- 不做 rolling self
- 不引入 `PPO v4` checkpoint 作为 frozen opponent
- 不同时做 `BC -> MAPPO`
- 不改 MAPPO 架构

避免把“v4 shaping 是否值得进入 MAPPO”这个问题和其他变量搅在一起。

## 7. 相关

- [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) — fair MAPPO 主干
- [SNAPSHOT-016](snapshot-016-shaping-v4-survival-anti-rush-ablation.md) — PPO 上的 v4 shaping 结果
- [SNAPSHOT-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) — 当前最强 baseline 主线
- [SNAPSHOT-019](snapshot-019-mappo-v2-opponent-pool-anchor-heavy-rebalance.md) — opponent-pool 配比消融
- [SNAPSHOT-023](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) — 预留：team-level PPO opponent adapter

## 8. 实施计划

预计新增：

- [soccerstwos_h100_cpu32_mappo_vs_baseline_shaping_v4_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_vs_baseline_shaping_v4_512x512.batch)
- 若需要，单独的 `SNAPSHOT-020` 运行目录前缀

verdict 结果将 append 到本文件 §9 之后，不回改 §1-§8 的预注册部分。

## 9. 首轮结果（原始 run + 续跑 run 合并口径）

### 9.1 运行目录

- 原始 run：
  - [PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251)
- 续跑 run：
  - [PPO_mappo_vs_baseline_shaping_v4_512x512_resume_20260415_155350](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_resume_20260415_155350)

### 9.2 原始 run 结果

- 原始 run 在 `iteration 470` 中断收尾：
  - `done = False`
  - `timesteps_total = 18.8M`
  - `episode_reward_mean ≈ -0.818`
- 训练内 `baseline 50` 最好点出现在原始 run 的中后段：
  - [checkpoint-360](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/MAPPOVsBaselineTrainer_Soccer_28129_00000_0_2026-04-15_03-53-14/checkpoint_000360/checkpoint-360) = `0.86`
  - [checkpoint-430](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/MAPPOVsBaselineTrainer_Soccer_28129_00000_0_2026-04-15_03-53-14/checkpoint_000430/checkpoint-430) = `0.86`
  - [checkpoint-420](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/MAPPOVsBaselineTrainer_Soccer_28129_00000_0_2026-04-15_03-53-14/checkpoint_000420/checkpoint-420) = `0.80`

### 9.3 续跑 run 结果

- 续跑 run 正常结束：
  - `iteration 500`
  - `done = True`
  - `timesteps_total = 20.0M`
  - `best_reward_mean = -0.7816 @ 500`
- 但续跑尾段的训练内 eval 没有超过原始 run 高点：
  - [checkpoint-480](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_resume_20260415_155350/MAPPOVsBaselineTrainer_Soccer_e1d3b_00000_0_2026-04-15_15-54-15/checkpoint_000480/checkpoint-480) = `0.68`
  - [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_resume_20260415_155350/MAPPOVsBaselineTrainer_Soccer_e1d3b_00000_0_2026-04-15_15-54-15/checkpoint_000490/checkpoint-490) = `0.76`

### 9.4 合并解读

- `MAPPO + v4 shaping` 不是坏线：
  - 它能稳定学起来
  - reward 从原始 run 尾段到续跑收尾继续改善（`-0.818 -> -0.782`）
- 但当前最强窗口仍然停留在原始 run 的中后段，而不是续跑尾段。
- 这说明：
  - `v4` 风格在 MAPPO 主干里是可行的
  - 但继续把训练拖到 `500 iter`，并没有自动把 `baseline 50` 再往上推

### 9.5 训练内结论

当前最值得正式 `baseline 500` 复核的，仍然是原始 run 的三个候选：

- [checkpoint-360](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/MAPPOVsBaselineTrainer_Soccer_28129_00000_0_2026-04-15_03-53-14/checkpoint_000360/checkpoint-360)
- [checkpoint-420](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/MAPPOVsBaselineTrainer_Soccer_28129_00000_0_2026-04-15_03-53-14/checkpoint_000420/checkpoint-420)
- [checkpoint-430](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/MAPPOVsBaselineTrainer_Soccer_28129_00000_0_2026-04-15_03-53-14/checkpoint_000430/checkpoint-430)

续跑尾点 [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_resume_20260415_155350/MAPPOVsBaselineTrainer_Soccer_e1d3b_00000_0_2026-04-15_15-54-15/checkpoint_000490/checkpoint-490) 仅作为补充参考，不作为当前主候选。

### 9.6 官方 `baseline 500` 复核

正式复核结果如下：

- [checkpoint-360](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/MAPPOVsBaselineTrainer_Soccer_28129_00000_0_2026-04-15_03-53-14/checkpoint_000360/checkpoint-360) = `0.718`
- [checkpoint-420](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/MAPPOVsBaselineTrainer_Soccer_28129_00000_0_2026-04-15_03-53-14/checkpoint_000420/checkpoint-420) = `0.740`
- [checkpoint-430](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/MAPPOVsBaselineTrainer_Soccer_28129_00000_0_2026-04-15_03-53-14/checkpoint_000430/checkpoint-430) = `0.756`
- [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_resume_20260415_155350/MAPPOVsBaselineTrainer_Soccer_e1d3b_00000_0_2026-04-15_15-54-15/checkpoint_000490/checkpoint-490) = `0.764`

这说明：

- `020` 的正式最好点出现在续跑 run 尾段，而不是原始 run 的中后段高窗
- 但最好正式分数也只有 `0.764`，仍明显低于 [SNAPSHOT-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) 的 `0.812` 与 [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md) 的 `0.842`
- 这条线没有证明 `v4 shaping` 值得进入当前 MAPPO 主干

### 9.7 Failure Capture

对最佳正式点 [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_resume_20260415_155350/MAPPOVsBaselineTrainer_Soccer_e1d3b_00000_0_2026-04-15_15-54-15/checkpoint_000490/checkpoint-490) 的 baseline failure capture 结果：

- capture `win_rate = 0.786`
- 失败主标签：
  - `late_defensive_collapse = 44`
  - `low_possession = 31`
  - `unclear_loss = 19`
  - `poor_conversion = 7`

机制解读：

- `v4` 在 MAPPO 上并没有形成比 `v2` 或 opponent-pool 主线更高的层级
- 它的问题也不是单一桶爆炸，而是整体不够强
- `low_possession` 仍然偏高，说明 `v4` 没有修掉我们当前最在意的控球问题

### 9.8 当前结论

`020` 不是坏线，但它没有证明 `MAPPO + v4 shaping` 值得进入当前主干：

- 正式最好点仅 [checkpoint-490](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_resume_20260415_155350/MAPPOVsBaselineTrainer_Soccer_e1d3b_00000_0_2026-04-15_15-54-15/checkpoint_000490/checkpoint-490) = `0.764`
- failure capture 也显示其主要问题仍是 `late_defensive_collapse + low_possession`
- 因此，`v4 shaping` 目前更像 PPO-specific 的局部补丁，而不是值得优先并入 MAPPO 主线的目标
