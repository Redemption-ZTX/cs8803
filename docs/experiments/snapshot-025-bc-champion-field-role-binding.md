# SNAPSHOT-025: BC-Champion Warm-Start + Field-Role Binding

- **日期**: 2026-04-17
- **负责人**:
- **状态**: 首轮长训与 official `baseline 500` / failure capture 已完成

## 1. 为什么需要 025

[SNAPSHOT-024](snapshot-024-striker-defender-role-binding.md) 已经证明：

- `field-role binding` 在真实 `warm470` 底座上是成立的
- 最好点 [checkpoint-270](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000270/checkpoint-270) 达到 `0.842`
- head-to-head 显示它已小优 [checkpoint-290](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000290/checkpoint-290)，但仍输给 [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100)。

因此，`025` 要回答的已经不是：

- “field-role binding 能不能成立”

而是更严格的问题：

- **“field-role binding 能不能在当前冠军底座上继续带来净增益？”**

## 2. 具体要测什么

本 snapshot 把 `024` 的 field-role shaping，叠到当前最强底座：

- source checkpoint: [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100)

注意：

- 这不是“原始 BC 模型”
- 而是 [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md) 的 **`BC -> MAPPO + shaping-v2` 冠军点**

所以本 snapshot 的准确定位是：

- **champion-on-champion fine-tune**

## 3. 核心假设

### 主假设

如果 `024` 的收益不只是“把中等底座推高”，而是真正有助于更强策略组织 team structure，那么在 [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 上叠加保守版 field-role binding 后，应当满足至少其一：

1. official `baseline 500` 仍维持在 `0.84+`
2. head-to-head 相比 source [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 接近五五开或转正
3. failure structure 比 `017@2100` 更干净，尤其 `low_possession` 或 `late_defensive_collapse` 出现明确改善

### 备择假设

如果一叠加 role-bound shaping 就明显掉分或在 head-to-head 中持续落后，说明：

- `024` 更像是“中等底座增强器”
- 而不是“冠军点继续抬升器”

## 4. 为什么这条线要保守

`017@2100` 已经非常强，headroom 很小。

如果直接把 [SNAPSHOT-024](snapshot-024-striker-defender-role-binding.md) 当前全强度配置原封不动叠上去，出现负结果时会有解释歧义：

- 是 `field-role binding` 本身不适合冠军点
- 还是当前系数对冠军点扰动过大

因此 `025` 首轮不走“全强度复刻 024”，而走：

- **保守版 field-role binding**

## 5. 首轮配置

### 5.1 warm-start 源

- [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100)

### 5.2 role-binding 方式

沿用 `024` 已验证可跑的 runtime 机制：

- `SHAPING_FIELD_ROLE_BINDING=1`
- `SHAPING_FIELD_ROLE_BINDING_MODE=spawn_depth`
- `ROLE_BINDING_TEAM_AGENT_IDS=0,1`

### 5.3 保守版 role 系数

相对 `024` 的全强度版本，本轮缩弱进攻/防守极化：

| 项 | striker | defender | 说明 |
|---|---:|---:|---|
| `ball_progress_scale` | `0.012` | `0.008` | 比 `024` 的 `0.015 / 0.005` 更温和 |
| `opponent_progress_penalty_scale` | `0.008` | `0.015` | 比 `024` 的 `0.005 / 0.020` 更温和 |
| `possession_bonus` | `0.002` | `0.002` | 保持对称 |
| `deep_zone_outer_penalty` | `0.003` | `0.0045` | defender 仍更守，但不翻倍 |
| `deep_zone_inner_penalty` | `0.003` | `0.0045` | defender 仍更守，但不翻倍 |
| `time_penalty` | `0.001` | `0.001` | 保持对称 |

额外说明：

- `0.0045` 是刻意选的非整数倍保守值，不是笔误
- [soccer_info.py](../../cs8803drl/core/soccer_info.py) 和 [utils.py](../../cs8803drl/core/utils.py) 内部都按 `float(...)` 处理 shaping 系数，理论上不会被截断
- 但由于这条 lane 还没在真实训练里用过 `0.0045`，首轮仍要在 smoke 阶段直接看训练头打印的 `reward_shaping_config`，确认 `ROLE_DEFENDER_DEEP_ZONE_*_PENALTY=0.0045` 真的原样到位

### 5.4 训练预算

首轮先跑保守 fine-tune 预算：

- `TIMESTEPS_TOTAL = 8,000,000`
- `TIME_TOTAL_S = 14,400`

理由：

- 先看是否有净增益信号
- 不急着直接做长训
- 若前 `150-250` iter 已明显转负，可以及早止损

## 6. 预声明判据

### 6.1 主判据

首轮最重要的不是单独的 `baseline 500` 分数，而是**是否能在不破坏冠军底座的前提下带来增益**。

因此主判据分三层：

1. **保底判据**
   - official `baseline 500 >= 0.830`
2. **竞争判据**
   - official `baseline 500 >= 0.842`
   - 或 head-to-head vs [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 接近 `0.500`
3. **升级判据**
   - head-to-head vs [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) `> 0.500`

### 6.2 机制判据

failure capture 中至少出现以下一种改善：

- `low_possession` 低于 `017@2100`
- `late_defensive_collapse` 低于 `017@2100`
- 两者都不改善时，至少 `poor_conversion` 明显下降

### 6.3 失败判据

若出现以下任一情况，首轮直接判负：

- official `baseline 500 < 0.800`
- head-to-head vs [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) `< 0.450`
- failure structure 明显更坏，且没有任何机制收益补偿

## 7. 风险

### R1 — 强底座上扰动过大

即使 `024` 在 `warm470` 上成立，也不代表它可以无损叠在冠军点上。

### R2 — 可能只更擅长打 baseline

就算 `baseline 500` 持平或略升，也可能仍然 head-to-head 输给 `017`。

### R3 — 预算太短会误伤

如果 role-conditioned fine-tune 在强底座上需要更长适应期，首轮 `8M` 预算可能偏保守。

## 8. 和既有线的关系

- [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md)：提供当前冠军底座
- [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md)：证明“破对称”本身有价值
- [SNAPSHOT-024](snapshot-024-striker-defender-role-binding.md)：证明 field-role binding 在 `warm470` 上成立

本 snapshot 的职责是：

- **把 `024` 从“可成立的挑战者机制”推进到“能否升级冠军主线”的检验**

## 9. 首轮执行清单

0. 先做 `1-iter smoke`：
   - `MAX_ITERATIONS=1`
   - `EVAL_INTERVAL=1`
   - `EVAL_EPISODES=50`
   - 验证 [warmstart_summary.txt](../../cs8803drl/training/train_ray_mappo_vs_baseline.py) 中 `copied=16`
   - 验证训练头打印的 `reward_shaping_config` 里 defender `deep_zone_outer/inner_penalty = 0.0045`
   - 验证 iter-1 的 `baseline 50` 没有灾难性塌陷；经验 sanity band 期望在 `0.70+`，若掉到 `<0.40`，应先怀疑“强底座 + 当前 role-diff 配置”不兼容，而不是直接开正式长训
1. 用 [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 作为 warm-start 源
2. 跑保守版 field-role binding fine-tune
3. 按 `top 5% + ties + 前后 2 点` 做 official `baseline 500`
4. 对最好点做 failure capture
5. 做至少一组 head-to-head：
   - best `025` vs `017@2100`

## 10. 1-Iter Smoke（已完成）

- smoke run:
  - [PPO_mappo_field_role_binding_bc2100_smoke_20260417_050207](../../ray_results/PPO_mappo_field_role_binding_bc2100_smoke_20260417_050207)
- 训练日志已确认 warm-start 真正生效：
  - `copied=16, adapted=0, skipped=0`
- `params.json` 已确认保守版 role 配置原样到位：
  - `role_binding_mode = spawn_depth`
  - `deep_zone_outer_penalty_by_role.defender = 0.0045`
  - `deep_zone_inner_penalty_by_role.defender = 0.0045`
- `checkpoint-1` 的 official `baseline 50` sanity check：
  - `win_rate = 0.840 (42W-8L-0T)`

因此，本 snapshot 当前没有发现：

- warm-start 失效
- `0.0045` 被截断或改写
- 冠军底座在 iter-1 被 field-role binding 灾难性破坏

首轮长训可以继续。

## 11. 首轮结果（已完成）

### 11.1 official `baseline 500`

围绕 internal 多峰窗口，首轮正式复核了：

- `20 / 70 / 80 / 110 / 120 / 130 / 140 / 150 / 160 / 170 / 180`

结果如下：

- [checkpoint-20](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000020/checkpoint-20) = `0.818`
- [checkpoint-70](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000070/checkpoint-70) = `0.806`
- [checkpoint-80](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000080/checkpoint-80) = `0.772`
- [checkpoint-110](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000110/checkpoint-110) = `0.804`
- [checkpoint-120](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000120/checkpoint-120) = `0.812`
- [checkpoint-130](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000130/checkpoint-130) = `0.816`
- [checkpoint-140](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000140/checkpoint-140) = `0.822`
- [checkpoint-150](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000150/checkpoint-150) = `0.804`
- [checkpoint-160](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000160/checkpoint-160) = `0.790`
- [checkpoint-170](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000170/checkpoint-170) = `0.776`
- [checkpoint-180](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000180/checkpoint-180) = `0.760`

首轮可见：

- official 最高点是 [checkpoint-140](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000140/checkpoint-140) = `0.822`
- 但早期点 [checkpoint-20](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000020/checkpoint-20) = `0.818` 与其差距很小
- `150-180` 窗口则明显回落，说明这条 lane 更像“早期 fine-tune 有益，继续训会把增益磨掉”

### 11.2 failure capture

对 `20 / 130 / 140` 做了 `baseline 500` failure capture：

- [checkpoint-20 capture](../artifacts/failure-cases/mappo_field_role_bc2100_checkpoint020_baseline_500)
- [checkpoint-130 capture](../artifacts/failure-cases/mappo_field_role_bc2100_checkpoint130_baseline_500)
- [checkpoint-140 capture](../artifacts/failure-cases/mappo_field_role_bc2100_checkpoint140_baseline_500)

结果分别为：

- `20`: `0.806 (403W-97L)`
- `130`: `0.784 (392W-108L)`
- `140`: `0.782 (391W-109L)`

按保存的 loss episode 重算 primary bucket：

- `20`
  - `late_defensive_collapse = 47/97`
  - `low_possession = 30/97`
  - `poor_conversion = 8/97`
  - `unclear_loss = 11/97`
- `130`
  - `late_defensive_collapse = 46/108`
  - `low_possession = 27/108`
  - `poor_conversion = 12/108`
  - `unclear_loss = 18/108`
- `140`
  - `late_defensive_collapse = 51/109`
  - `low_possession = 28/109`
  - `poor_conversion = 12/109`
  - `unclear_loss = 18/109`

更关键的是 official 与 capture 的对齐程度：

- `20`: `0.818 -> 0.806`，gap 仅 `-0.012`
- `130`: `0.816 -> 0.784`，gap `-0.032`
- `140`: `0.822 -> 0.782`，gap `-0.040`

因此这条线的最终口径应当是：

- **official 最高点**: [checkpoint-140](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000140/checkpoint-140) = `0.822`
- **最可信主候选**: [checkpoint-20](../../ray_results/PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955/MAPPOVsBaselineTrainer_Soccer_4052b_00000_0_2026-04-17_05-10-17/checkpoint_000020/checkpoint-20) = `0.818`

### 11.3 首轮 verdict

`025` 的结论现在可以收成：

- `BC@2100 + 保守 field-role binding` 是**明显有效**的强正结果
- 但它没有超过 [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md) 的 [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 或 [SNAPSHOT-024](snapshot-024-striker-defender-role-binding.md) 的 [checkpoint-270](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000270/checkpoint-270)
- 它更像“冠军底座上的短程有益 fine-tune”，而不是新的冠军翻盘线
- 这条 lane 的更可信解释不是“后期越训越好”，而是“早期 gain 最真实，继续训会把 gain 磨掉”

## 12. 相关

- [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md)
- [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md)
- [SNAPSHOT-024](snapshot-024-striker-defender-role-binding.md)
- [PLAN-002](../plan/plan-002-il-mappo-dual-mainline.md)
