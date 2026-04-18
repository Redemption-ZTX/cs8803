# SNAPSHOT-025b: BC-Champion Field-Role Binding Stability Tune

- **日期**: 2026-04-17
- **负责人**:
- **状态**: 已完成首轮结果

## 1. 为什么需要 025b

[SNAPSHOT-025](snapshot-025-bc-champion-field-role-binding.md) 已证明：

- `BC@2100 -> field-role binding` 的 `1-iter smoke` 没有灾难性破坏冠军底座
- `checkpoint-1` 的 official `baseline 50` 仍有 `0.840 (42W-8L)`

但 `025` 正式长训当前暴露出一个新的工程问题：

- `progress.csv` 中 `kl` / `total_loss` 高频出现 `inf`
- TensorBoard 持续打印 `NaN or Inf found in input tensor`
- 同时训练内 `baseline 50` 仍保持可用区间，没有立即崩盘

因此，`025b` 的问题已经不是：

- “field-role binding 在冠军底座上能不能成立”

而是更具体的：

- **“在不改变 025 机制定义的前提下，能否通过更保守的 PPO 更新把数值抖动压下来？”**

## 2. 设计原则

`025b` 是一个**稳定性修复版**，不是新机制实验。

为保证与 `025` 的对比干净，本 snapshot 明确遵循：

1. **不改 warm-start 源**
2. **不改 field-role binding 机制**
3. **不改 role shaping 系数**
4. **只改 PPO 优化强度**

这样等 `025 / 025b` 同步跑完后，我们就能把：

- “机制收益”
- “优化不稳定”

拆开判断，而不是把两者混在一起。

## 3. 与 025 完全保持一致的部分

### 3.1 warm-start 源

- [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100)

### 3.2 role-binding 机制

- `SHAPING_FIELD_ROLE_BINDING=1`
- `SHAPING_FIELD_ROLE_BINDING_MODE=spawn_depth`
- `ROLE_BINDING_TEAM_AGENT_IDS=0,1`

### 3.3 role shaping 系数

完全沿用 [SNAPSHOT-025](snapshot-025-bc-champion-field-role-binding.md) 的保守版配置：

| 项 | striker | defender |
|---|---:|---:|
| `ball_progress_scale` | `0.012` | `0.008` |
| `opponent_progress_penalty_scale` | `0.008` | `0.015` |
| `possession_bonus` | `0.002` | `0.002` |
| `deep_zone_outer_penalty` | `0.003` | `0.0045` |
| `deep_zone_inner_penalty` | `0.003` | `0.0045` |
| `time_penalty` | `0.001` | `0.001` |

## 4. 仅在 025b 中修改的项

### 4.1 优化强度收紧

相对 `025`：

| 项 | 025 | 025b | 目的 |
|---|---:|---:|---|
| `LR` | `3e-4` | `1e-4` | 降低 warm-start 后的更新冲击 |
| `NUM_SGD_ITER` | `10` | `4` | 减少每批数据上的过度优化 |
| `SGD_MINIBATCH_SIZE` | `1024` | `2048` | 降低小 batch 噪声 |
| `CLIP_PARAM` | `0.20` | `0.15` | 收紧 PPO policy step |

### 4.2 不改预算

为保持可比性，`025b` 仍使用：

- `TIMESTEPS_TOTAL = 8,000,000`
- `TIME_TOTAL_S = 14,400`

## 5. 核心假设

### 主假设

如果 `025` 当前的主要问题是**优化过猛**，而不是机制本身不兼容，那么 `025b` 应出现以下至少两项改善：

1. `progress.csv` 中 `kl / total_loss = inf` 的频率显著下降
2. 训练内 `baseline 50` 不再出现明显中段塌陷
3. 后段 best checkpoint 的 official `baseline 500` 不低于 `025`
4. head-to-head 相比 source [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 不弱于 `025`

### 反假设

如果 `025b` 明显更稳，但 baseline / head-to-head 同时下降，那么说明：

- `025` 的抖动里包含一部分“高更新强度带来的探索收益”
- `025b` 更稳，但可能过保守

如果 `025b` 仍然高频 `inf`，那就说明：

- 当前问题不只是优化过猛
- 下一步需要重新考虑 shaping 强度本身

## 6. 预声明判据

### 6.1 稳定性判据

`025b` 的首要成功标准不是立刻超过 `025`，而是：

- `kl/total_loss = inf` 的迭代比例显著低于 `025`
- TensorBoard warning 频率显著下降

### 6.2 性能判据

在稳定性改善的前提下，至少满足其一：

1. official `baseline 500 >= 0.830`
2. official `baseline 500` 不低于 `025` 最终最好点
3. head-to-head vs [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 优于 `025`

## 7. 风险

### R1 — 过度保守

`025b` 可能只是把不稳定压下去，但同时把潜在收益一起压掉。

### R2 — 根因并非优化

如果 `025b` 仍频繁 `inf`，那么问题更可能来自：

- 冠军底座与当前 field-role shaping 的目标冲突
- 而不是单纯学习率/epoch 太大

## 8. 首轮结果

### 8.1 稳定性修复是否成功

成功，而且幅度很明显。

对照：

- [SNAPSHOT-025](snapshot-025-bc-champion-field-role-binding.md)：`bad iters = 84 / 200`
- `025b`：[progress.csv](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/progress.csv) 中 `bad iters = 5 / 200`

这说明：

- `025` 里高频的 `kl / total_loss = inf`
- 主要是 PPO 更新过猛
- 而不是 `BC@2100 + field-role binding` 机制本身不兼容

换句话说，`025b` 已经把这个并行修复臂的核心问题回答清楚了：

- **优化稳定性修复方向是对的**

### 8.2 official `baseline 500`

对 [checkpoint_eval.csv](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/checkpoint_eval.csv) 的多峰保护窗口做 official `baseline 500` 复核后，结果如下：

| checkpoint | official `baseline 500` |
|---|---:|
| `40` | `0.806` |
| `50` | `0.808` |
| `60` | `0.830` |
| `70` | `0.836` |
| `80` | `0.842` |
| `130` | `0.816` |
| `140` | `0.820` |
| `150` | `0.838` |
| `160` | `0.794` |
| `170` | `0.808` |
| `180` | `0.810` |
| `190` | `0.826` |

首轮 official 峰值为：

- [checkpoint-80](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80) = `0.842`

强备选窗口为：

- [checkpoint-70](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000070/checkpoint-70) = `0.836`
- [checkpoint-150](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000150/checkpoint-150) = `0.838`

### 8.3 failure capture

对 `70 / 80 / 150` 三点分别做 `baseline 500` failure capture 后，得到：

| checkpoint | official | capture | saved losses |
|---|---:|---:|---:|
| `70` | `0.836` | `0.840` | `80` |
| `80` | `0.842` | `0.836` | `82` |
| `150` | `0.838` | `0.828` | `86` |

主要 failure bucket 如下：

#### `checkpoint-70`

- `late_defensive_collapse = 44/80`
- `low_possession = 25/80`
- `poor_conversion = 3/80`
- `unclear_loss = 7/80`

#### `checkpoint-80`

- `late_defensive_collapse = 41/82`
- `low_possession = 28/82`
- `poor_conversion = 4/82`
- `unclear_loss = 8/82`

#### `checkpoint-150`

- `late_defensive_collapse = 34/86`
- `low_possession = 26/86`
- `poor_conversion = 10/86`
- `unclear_loss = 13/86`

### 8.4 failure bucket 占比与对齐 `BC@2100`

把 §8.3 的原始计数换算成占比后，可以直接和当前冠军基线（[checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100)）对齐：

| checkpoint | total | late_def_collapse | low_poss | poor_conv | unclear_loss |
|---|---:|---:|---:|---:|---:|
| `70` (0.836) | 80 | **55.0%** | 31.3% | 3.8% | 8.8% |
| `80` (0.842) | 82 | **50.0%** | 34.1% | 4.9% | 9.8% |
| `150` (0.838) | 89 | 38.2% | 29.2% | 11.2% | 14.6% |
| **`017@2100` (0.842)** | — | **46.5%** | — | — | 11.6% |
| **`017@2100` low_poss + poor_conv 合计** | — | — | — | 39.5% | — |

对照 [SNAPSHOT-025](snapshot-025-bc-champion-field-role-binding.md) §6.2 的机制判据（"`low_possession` 低于 `017@2100` / `late_defensive_collapse` 低于 `017@2100`"）：

- `025b@80`: `late_def_collapse = 50.0%` 比 `017@2100` 的 `46.5%` 更差；`low_poss + poor_conv = 39.0%` 与 `017@2100` 的 `39.5%` 持平
- `025b@70`: `late_def_collapse = 55.0%` 明显更差；`low_poss + poor_conv = 35.1%` 略低

因此在**失败结构**层面，`025b` 并不是 `017@2100` 的严格升级：

- 它没有显著削减 `low_possession`
- 它的 `late_defensive_collapse` 占比反而偏高

但在**整体对抗**层面（§9 head-to-head），`025b` 明确压过 `017@2100`（0.556~0.562）。

把两个层面合起来，首轮更准确的定性是：

- `025b` 赢在**整体策略执行的稳定性与对抗健壮度**
- 而不是赢在**特定失败模式的消除**

这个细节对后续方向有直接影响——如果想突破 0.84 天花板并同时改善失败结构，单纯在 `BC@2100 + field-role binding` 上反复调整不够；需要架构级（[SNAPSHOT-027](snapshot-027-team-level-ppo-coordination.md)）或 reward 级（[SNAPSHOT-026](snapshot-026-reward-liberation-ablation.md)）的变化。

### 8.5 首轮 verdict

`025b` 的结果比 [SNAPSHOT-025](snapshot-025-bc-champion-field-role-binding.md) 更干净，也更强：

1. **稳定性修复成功**
   - 从 `84/200` bad iter 降到 `5/200`
2. **mechanism 被保住了**
   - official `baseline 500` 从 `025` 的可信主候选 `0.818` 抬到 `0.836~0.842`
3. **`70/80` 已形成真正的双候选窗口**
   - `80` 是 official 最高点
   - `70` 的 `official -> capture` 对齐更好（`0.836 -> 0.840`）
   - `70` 的 `poor_conversion` 与 `unclear_loss` 也最低

因此，在只看 baseline 与 failure capture 时，`025b` 的更准确收口是：

- **official 峰值：** [checkpoint-80](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80) = `0.842`
- **更平衡、也更可信的结构型候选：** [checkpoint-70](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000070/checkpoint-70) = `0.836 official / 0.840 capture`

这说明 `025` 的主要瓶颈确实是优化稳定性，而不是 field-role binding 机制本身。

## 9. head-to-head

三组 head-to-head 已完成：

| matchup | result |
|---|---:|
| `025b@70 vs 017@2100` | `278W-222L = 0.556` |
| `025b@70 vs 024@270` | `293W-207L = 0.586` |
| `025b@80 vs 017@2100` | `281W-219L = 0.562` |

这说明：

1. `025b` 不只是 baseline specialists，而是**真正进入冠军位对话**
2. `025b@70` 已经同时压过 [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md) 与 [SNAPSHOT-024](snapshot-024-striker-defender-role-binding.md)
3. 在共有对手 `017@2100` 上，[checkpoint-80](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80) 又略强于 [checkpoint-70](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000070/checkpoint-70)

因此，把 head-to-head 也纳入后，`025b` 的首轮总收口更新为：

- **当前总主候选：** [checkpoint-80](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80)
- **并列强备选 / 结构更干净的点：** [checkpoint-70](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000070/checkpoint-70)

当前排序也应更新为：

- **`025b > 017 > 024 > 018 > 022`**

## 10. 下一步

首轮最关键的问题已经回答完了。后续若继续推进，价值最高的是：

1. 针对 `80/70` 再补一个更强对手或交叉 head-to-head，确认二者谁是最终提交点
2. 若 `027` 或 `026` 某条线继续上升，再用 `025b@80/70` 作为新的冠军参照物

## 11. 相关

- [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md)
- [SNAPSHOT-024](snapshot-024-striker-defender-role-binding.md)
- [SNAPSHOT-025](snapshot-025-bc-champion-field-role-binding.md)
- [PLAN-002](../plan/plan-002-il-mappo-dual-mainline.md)
