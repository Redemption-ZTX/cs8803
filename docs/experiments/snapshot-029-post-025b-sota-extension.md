# SNAPSHOT-029: Post-025b SOTA Extension (Per-Agent Trio)

- **日期**: 2026-04-17
- **负责人**:
- **状态**: 进行中 / `029-A/B/C` 已因 quota 中断并完成续跑恢复

## 0. 续跑说明

`029-A/B/C` 首轮正式 run 都在 `checkpoint-200` 左右因 home quota exhausted 中断，随后分别从原 run root 继续 restore。当前应统一以各自 run root 下的 **merged summary** 作为正式总摘要。

### 029-A

- run root: [PPO_mappo_029A_pbrs_on_025b80_512x512_formal](../../ray_results/PPO_mappo_029A_pbrs_on_025b80_512x512_formal)
- restore checkpoint: [checkpoint-200](../../ray_results/PPO_mappo_029A_pbrs_on_025b80_512x512_formal/MAPPOVsBaselineTrainer_Soccer_bf184_00000_0_2026-04-17_12-16-10/checkpoint_000200/checkpoint-200)
- resumed trial: [MAPPOVsBaselineTrainer_Soccer_98f8c_00000_0_2026-04-17_19-17-26](../../ray_results/PPO_mappo_029A_pbrs_on_025b80_512x512_formal/MAPPOVsBaselineTrainer_Soccer_98f8c_00000_0_2026-04-17_19-17-26)
- merged summary: [merged_training_summary.txt](../../ray_results/PPO_mappo_029A_pbrs_on_025b80_512x512_formal/merged_training_summary.txt)

### 029-B

- run root: [PPO_mappo_029B_bwarm170_to_v2_512x512_formal](../../ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal)
- restore checkpoint: [checkpoint-200](../../ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000200/checkpoint-200)
- resumed trial: [MAPPOVsBaselineTrainer_Soccer_b3bb8_00000_0_2026-04-17_19-18-11](../../ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_b3bb8_00000_0_2026-04-17_19-18-11)
- merged summary: [merged_training_summary.txt](../../ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/merged_training_summary.txt)

### 029-C

- run root: [PPO_mappo_029C_025b80_oppool_peers_512x512_formal](../../ray_results/PPO_mappo_029C_025b80_oppool_peers_512x512_formal)
- restore checkpoint: [checkpoint-200](../../ray_results/PPO_mappo_029C_025b80_oppool_peers_512x512_formal/MAPPOVsOpponentPoolTrainer_Soccer_50be3_00000_0_2026-04-17_12-13-05/checkpoint_000200/checkpoint-200)
- resumed trial: [MAPPOVsOpponentPoolTrainer_Soccer_bdbfc_00000_0_2026-04-17_19-18-28](../../ray_results/PPO_mappo_029C_025b80_oppool_peers_512x512_formal/MAPPOVsOpponentPoolTrainer_Soccer_bdbfc_00000_0_2026-04-17_19-18-28)
- merged summary: [merged_training_summary.txt](../../ray_results/PPO_mappo_029C_025b80_oppool_peers_512x512_formal/merged_training_summary.txt)

所有 `029` lane 的 canonical summary 都由 [print_merged_training_summary.py](../../scripts/tools/print_merged_training_summary.py) 生成，且 canonical 输出明确包含 **`merged_training_summary.txt` + `training_loss_curve_merged.png`** 两部分。后续任何 best/final 结论都应以 merged summary 与根目录 `checkpoint_eval.csv` 为准，而不是只看 resumed trial 自己的终端打印。

## 1. 为什么需要 029

截至当前：

- **真 SOTA**：[025b@80](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80) = official `0.842` + H2H 压过 `017` / `024`
- **baseline specialist**：[B-warm @250](../../ray_results/PPO_mappo_liberation_B_warm_bc2100_512x512_20260417_062526/MAPPOVsBaselineTrainer_Soccer_d3698_00000_0_2026-04-17_06-25-59/checkpoint_000250/checkpoint-250) = official `0.864` 但 H2H 输给 `017`/`025b`
- **mechanism 突破点**：[B-warm @170](../../ray_results/PPO_mappo_liberation_B_warm_bc2100_512x512_20260417_062526/MAPPOVsBaselineTrainer_Soccer_d3698_00000_0_2026-04-17_06-25-59/checkpoint_000170/checkpoint-170) = `low_possession 20.2%`，项目首次打破 22-28% 不变量

[SNAPSHOT-026](snapshot-026-reward-liberation-ablation.md) 的 H2H 结果留下一个核心问题：**PBRS 的机制收益为什么没有转化成真正的技能提升？**

有三个互相正交的假设：

| 假设 | 核心诊断 | 对应 lane |
|---|---|---|
| **H1: 源底座不够强** | PBRS 只能"包装" BC@2100 去对付 baseline，如果底座是更强的 025b@80，PBRS 的机制价值可能真正转化 | **029-A** |
| **H2: PBRS 学到的机制可以被 v2 接手** | 从 B-warm @170 出发换成 v2 shaping，让 "低 low_poss 的持球能力" 被带进 025b 那套能赢 H2H 的策略空间 | **029-B（修正）** |
| **H3: 训练时没见过同级对手** | 所有 lane 都只见 baseline，所以 policy 的防 exploit 能力没有机会训练 | **029-C** |

029 的职责是**三条并行 lane 各自测一个假设**，每条都基于已知 good config 做最小干扰式改动。

并行跑是为了排除"时间因素"——如果串行跑，后面的 lane 可能被前面的结果影响设计（或被推迟）。三条并行能在一轮 GPU 内给出三个独立答案。

## 2. 与既有 lane 的关系

| snapshot | 和 029 的关系 |
|---|---|
| [017 BC@2100](snapshot-017-bc-to-mappo-bootstrap.md) | per-agent BC→MAPPO 冠军点；是 029 三条的背景基线 |
| [018 opponent pool](snapshot-018-mappo-v2-opponent-pool-finetune.md) | 提供 029-C 的 opponent pool 基础设施 |
| [025b](snapshot-025b-bc-champion-field-role-binding-stability-tune.md) | 提供 029-A/C 的 warm-start 源与 PPO tight params |
| [026 B-warm](snapshot-026-reward-liberation-ablation.md) | 提供 029-A 的 PBRS shaping 设计；**提供 029-B 的 warm-start 源（@170）** |

三条 lane 的每条 **只改一个因素**（factorial 设计）：

| lane | warm-start 源 | shaping | PPO | opponent | 改动点 |
|---|---|---|---|---|---|
| 025b@80（参照）| BC@2100 | v2 | tight (025b) | baseline | — |
| **029-A** | **025b@80** | **PBRS** | tight (025b) | baseline | base + shaping |
| **029-B** | **B-warm @170** | v2 | tight (025b) | baseline | **base 升级到"PBRS-trained 机制点"** |
| **029-C** | **025b@80** | v2 | tight (025b) | **pool** | base + opponent |

这样每条 lane 都可以和 025b@80 做干净的归因。

## 3. 路径 A — `025b + PBRS`

### 3.1 设计

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | 025b@80 | 当前 SOTA |
| shaping | PBRS goal-proximity | 026B 配置 |
| network | 512x512 | 025b / 026B 同款 |
| `lr` | 1e-4 | 025b |
| `num_sgd_iter` | 4 | 025b |
| `clip_param` | 0.15 | 025b |
| `goal_proximity_scale` | 0.01 | 026B |
| `gamma_pbrs` | 0.99 | 026B |
| opponent | baseline 100% | 纯对照 |

### 3.2 假设

**H1 成立** 意味着：

- 029-A 官方 500 ≥ 0.85（突破 025b 的 0.842）
- H2H vs 025b@80 ≥ 0.52（不输、最好赢）
- failure capture 中 `low_possession` 显著低于 025b 的 34.1%（PBRS 机制收益保留）

**H1 不成立** 意味着：

- 029-A 接近 0.85 但 H2H vs 025b 约 0.5（底座升级没打破 baseline-specific exploit）
- 或 029-A 官方 500 < 0.842（PBRS + 025b 底座兼容性差，两种收益互斥）

## 4. 路径 B — `B-warm + 025b stability`

### 4.1 原设计已失效

最初的 029-B 是"B-warm + 025b tight PPO params"。实现时发现：**现有 026 B-warm 的 batch 已经用了 `lr=1e-4 / sgd_iter=4 / clip=0.15`**（和 025b 同款），不是 025 原版的猛参数。

所以"PPO update 太猛导致 @170→@250 相位冲过头"这个假设**已经被现有 B-warm 数据否证**——用 tight params 仍然出现相同轨迹。沿着 H2 继续走没有新信息。

### 4.2 修正后的设计：`B-warm @170 → v2 shaping handoff`

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | **B-warm @170** | **PBRS-trained mechanism 点** |
| shaping | **v2** | 025b 的赢 H2H 配置 |
| network | 512x512 | 匹配 warm-start 源 |
| `lr` | 1e-4 | 025b |
| `num_sgd_iter` | 4 | 025b |
| `clip_param` | 0.15 | 025b |
| opponent | baseline 100% | 纯对照 |

### 4.3 假设（修正后）

**H2（修正）：PBRS 学到的"持球机制"能被 v2 shaping 接手。**

B-warm @170 的状态：
- `low_possession 20.2%`（项目首次打破 22-28% 不变量）
- official WR 仅 `0.842`，和 BC@2100 持平
- H2H 表现未知（有空跑出来）

v2 shaping 的状态：
- 在 BC@2100 底座上稳定 fine-tune 到 025b@80 = `0.842 official + H2H 赢 BC`
- 带来"策略稳健性 / 对同级不吃亏"

**H2 成立** 意味着两者叠加后：
- 029-B 官方 500 ≥ 0.85
- **`low_possession` 仍 ≤ 25%**（PBRS 机制收益被保住）
- H2H vs 025b@80 ≥ 0.52（赢过源 shaping 同款但 base 不同的 025b）

**H2 不成立** 意味着：
- 029-B ≈ 025b@80（0.842 / low_poss 34%）——v2 shaping 直接把 PBRS 学到的"非推进持球"习惯冲掉，policy 退化为 BC→v2 的老轨迹
- 或 029-B < 025b（两种 shaping 交接时出现 value function mismatch，训练不稳）

### 4.4 备选 C 的候选地位

选项 C（`B-warm + entropy_coeff=0.005`）在本 snapshot 保留为 **fallback**：

- 如果 029-B 出负结果（v2 shaping 把 PBRS 机制冲掉），下一轮可以起 029-D 测 "entropy 能否在 PBRS 学到的相位上保住机制"
- 如果 029-B 出正结果，entropy 方向不需要单独跑

## 5. 路径 C — `025b + opponent pool`

### 5.1 设计

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | 025b@80 | 当前 SOTA |
| shaping | v2 shaping | 025b 同款 |
| network | 512x512 | 025b 同款 |
| `lr` | 1e-4 | 025b |
| `num_sgd_iter` | 4 | 025b |
| `clip_param` | 0.15 | 025b |
| opponent pool | baseline 40% / 017@2100 20% / 024@270 20% / 025b@80-self 20% | **新组合** |

### 5.2 为什么是这个 pool 组成

- **baseline 40%**：评分目标权重最大，仍是主对手
- **017 / 024 / 025b self**：同级强敌，强制 policy 学习 "防同级 exploit"
- **不加 random / 022**：random 不提供对抗信号；022 的 H2H 弱于主线，不值得浪费 slot

### 5.3 假设

**H3 成立** 意味着：

- 029-C 官方 `baseline 500` ≥ 0.842（不因 pool 训练丧失 baseline 对抗能力）
- H2H vs 017/024/025b-original 分别 ≥ 0.55（明显压过它们）
- failure capture 中 `late_defensive_collapse` 比 025b 更低（防守更稳）

**H3 不成立** 意味着：

- 029-C 官方 500 < 0.82（pool 训练让 baseline 胜率掉太多）
- 或 H2H 和 025b-original 接近（peer training 没有带来额外技能）

## 6. 执行矩阵

| lane | warm-start | 改动 | iter | 预算 |
|---|---|---|---|---|
| **029-A** | 025b@80 | + PBRS | 300 | ~4h |
| **029-B** | B-warm @170 | v2 shaping handoff | 300 | ~4h |
| **029-C** | 025b@80 | + opponent pool | 300 | ~6-8h |
| 补：`B-warm @170 完整 H2H` | — | 纯 eval | — | ~2h CPU |

**并行跑**：A/B/C 同时提交，3-4 槽 GPU 全用上。`@170 H2H` 用 CPU 时间或 GPU 空隙完成。

## 7. 预声明判据

### 7.1 主判据

| lane | 阈值 | 逻辑 |
|---|---|---|
| **029-A** | official ≥ 0.85 且 H2H vs 025b ≥ 0.52 | 突破当前 SOTA + 压过源底座 |
| **029-B** | official ≥ 0.85 且 `low_possession` ≤ 25% | PBRS 机制收益被 v2 shaping 接手 |
| **029-C** | H2H vs 017/024 ≥ 0.55 且 official ≥ 0.84 | peer 训练带来防同级 exploit 能力且不损失 baseline 对抗 |

### 7.2 机制判据

- **029-A**: `low_possession` 严格低于 025b 的 34.1%（PBRS 机制收益在更强底座上保留）
- **029-B**: `low_possession` ≤ 25%（PBRS 学到的持球机制被 v2 接手，不会回退到 BC→v2 老轨迹的 34%）
- **029-C**: `late_defensive_collapse` 低于 025b 的 50.0%（peer 训练改善防守）

### 7.3 失败情形

| 条件 | 解读 |
|---|---|
| 三条 lane 全部 official < 0.82 | 025b@80 已是 per-agent 架构上限，后续提升需要 027/028 team-level |
| 029-A 好但 029-B/C 差 | PBRS 的价值真实存在，但 stability 和 pool 路径不 work |
| 029-B 好且 `low_poss` 保持低 | PBRS→v2 handoff 是一个干净的新 shaping 管线，值得推广 |
| 029-C 的 H2H 强但 baseline WR 降到 0.80 | pool 训练有泛化收益但牺牲 baseline specificity，提示 "pool 组成需要重调" |

## 8. 与 027/028 的关系

029 是 **per-agent 架构上的最后一批扩展**：

- 如果 029 任一条出 SOTA（H2H + official 都强），说明 per-agent 架构还有升级空间，027/028 的 team-level 必须拿出更强结果才有意义
- 如果 029 三条都失败（官方 < 0.82 或 H2H 全输），说明 per-agent 架构确实到顶，027/028 的 team-level 是唯一出路
- 如果 027/028 成功且 029 某条也成功，可以把成功的 per-agent trick（PBRS / pool / stability）迁移到 team-level

**029 不和 027/028 竞争，而是和 027/028 做互补的上限检验**。

## 9. 工程需求

### 9.1 复用资产

- 025b@80 checkpoint：作为 029-A / 029-C 的 warm-start 源
- 026B 的 PBRS shaping 实现（`compute_goal_proximity_shaping()`）：029-A / 029-B 复用
- 018 的 opponent pool 基础设施：029-C 复用

### 9.2 需要新增

- **029-A batch**：warm-start 从 025b + PBRS + 025b tight params
- **029-B batch**：warm-start 从 **B-warm @170** + v2 shaping + 025b tight params
- **029-C batch**：warm-start 从 025b + v2 shaping + 新 opponent pool 组成
- opponent pool 组成需要确认 018 的 pool config 接口能否接受 017/024/025b 的 checkpoint（都是 shared_cc_policy 格式，应该兼容）

### 9.3 1-iter smoke 要点

每条 lane 都要先跑 1-iter smoke 确认：
- warm-start 生效（`copied=16, skipped=0`）
- PPO config 参数到位（特别是 025b 的 `lr=1e-4 / sgd_iter=4 / clip=0.15`）
- shaping config 正确（029-A/B 的 PBRS scale / 029-C 的 v2 系数）
- 029-C 还要确认 opponent pool 的四个成员都能 instantiate

## 10. 执行清单

1. 起 3 条 batch 脚本，每条跑 smoke 确认配置
2. 并行提交 029-A / B / C 三个 job
3. 同时起 `B-warm @170 vs {017, 024, 025b}` 的 H2H 补测（CPU 时间）
4. 训练完成后，对每条 lane 做：
   - `top 5% + ties + 前后 2 点` 的 official `baseline 500` 扫描
   - best ckpt 的 failure capture
   - best ckpt vs 025b@80 / 017@2100 / 024@270 的 H2H
5. 按 §7 判据做 verdict
6. 如果某条 lane 出新 SOTA，更新 `MEMORY.md`、`experiments/README.md`、`CLAUDE.md`（如相关）

## 11. 首轮 verdict（official 500 + failure capture + H2H）

### 11.1 三条 lane 的 official `baseline 500`

| lane | 最佳 ckpt | official 500 | 多峰窗口 |
|---|---|---:|---|
| **029-A** (025b + PBRS) | 70 | 0.842 | 70 (0.842), 80 (0.830), 90 (0.818) |
| **029-B** (B-warm@170 → v2) | **190** | **0.868** | 180 (0.834), **190 (0.868)**, 200 (0.844), 280 (0.844) |
| **029-C** (025b + opp pool) | 270 | 0.848 | 180 (0.822), 230 (0.822), 270 (0.848) |

参照点：`017@2100 = 025b@80 = 0.842`、`026B@250 = 0.864`。

### 11.2 H2H 关键对战

| 对战 | 029X 胜率 | 解读 |
|---|---:|---|
| **029B@190 vs 025b@80** | **0.492** (246W-254L) | 统计平局（±0.022 噪声带内）|
| **029B@190 vs 017@2100** | 0.512 (256W-244L) | 029B 微赢 |
| **029B@190 vs 026B@250** | **0.562** (281W-219L) | 029B 明显赢 PBRS specialist |
| 029A@70 vs 025b@80 | 0.458 (229W-271L) | 029A 输 0.042 |

### 11.3 029B@190 失败桶（80 局 loss）

| 桶 | 029B@190 | B-warm @170（PBRS 学到的）| 025b@80 | 017@2100 |
|---|---:|---:|---:|---:|
| late_def_collapse | **51.2%** | 51.7% | 50.0% | 46.5% |
| **low_possession** | **26.2%** | **20.2%** | 34.1% | — |
| poor_conversion | 12.5% | 7.9% | 4.9% | — |
| unclear_loss | 8.8% | 14.6% | 9.8% | 11.6% |
| **low_poss + poor_conv** | **38.8%** | **28.1%** | 39.0% | 39.5% |

### 11.4 三条假设的最终判决

| 假设 | 子判据 | 结果 |
|---|---|---|
| **H1**（029-A：强底座释放 PBRS）| official ≥ 0.85 + H2H vs 025b ≥ 0.52 | ❌ **否决**：official 仅持平 (0.842)、H2H 输 (0.458) |
| **H2**（029-B：PBRS → v2 handoff）| official ≥ 0.85 + low_poss ≤ 25% + H2H ≥ 0.52 | ⚠️ **部分成立**：WR 突破 ✅ / low_poss 26.2% 边界失败 ❌ / H2H 平 ❌（统计上没赢）|
| **H3**（029-C：peer pool）| official ≥ 0.84 + H2H vs 017/024 ≥ 0.55 | ⚠️ **弱正**：official 0.848 marginal、H2H 未测、failure 结构和 025b 几乎相同 |

### 11.5 H2 部分成立的更准确解读

`029-B@190` 在 vs baseline 上完成了 +0.026 突破，但失败结构显示：

- B-warm @170 学到的 `low_poss = 20.2%` 持球机制，被 v2 shaping 接手 20 iter 后**冲掉到 26.2%**
- 即 PBRS 学到的"非推进持球"习惯**没耐 v2 shaping 接管**
- 029B@190 更准确的定性是 **"v2 shaping 的更好局部最优"**——init 不同（B-warm @170 vs BC@2100）让 PPO 找到了一个比 025b 更佳的 v2 局部点
- 而**不是** "PBRS 机制 + v2 收益的纯加法"

也就是说：**029B@190 是工程上的清晰 SOTA，但不是机制上的 PBRS-v2 完美融合**。

### 11.6 SOTA 切换

| 旧 | 新 |
|---|---|
| **025b@80** = `0.842 official + H2H 赢 017` | **029B@190** = `0.868 official + H2H 平 025b + H2H 赢 017/026B` |

完整 SOTA 链：

```
017 BC@2100 (0.842, ref)
  → 025b@80 (0.842, H2H 赢 017)
  → 029B@190 (0.868, H2H 平 025b, H2H 赢 017/026B) ← 当前 SOTA
```

新 SOTA 同时满足：
- vs baseline 项目最高 (0.868)
- 在对等对抗（H2H vs 025b）中至少不输
- 明确压过所有此前的 baseline-specialist（026B@250）和 BC champion (017@2100)

### 11.7 对项目级假设的修订（snapshot-026 §16.5）

snapshot-026 §16.5 当时的猜测：

> `low_possession` 不是纯架构问题，**通过修改 reward 设计可以将它压到 20% 附近**

029B 的失败桶证明：**这个机制突破不耐 v2 shaping 接手**。20.2% 是 PBRS active 时的状态；一旦换回 v2，机制就被冲掉到 26.2%。

更准确的修订版结论：

> `low_possession 22-28%` 不变量是 **v2 shaping 自己的 reward landscape 局部最优带**——只要训练 reward 是 v2 系列，`low_poss` 就会停在这个区间。PBRS 训练能临时压到 20%，但这个状态**不稳定**，换回 v2 就退回。

对 027/028 的修正预言：**team-level 架构如果不改 reward shaping（用 v2），`low_poss` 大概率仍在 22-28% 区间**。要真的打破，可能需要 team-level + PBRS 同时训。

### 11.8 后续优先级

短期：
- **029B@190 / 029C@270 的 H2H 完整矩阵**（vs 024@270 等）补完，确认排序
- **更新 MEMORY.md + README**：029B@190 = 当前 SOTA

中期（依赖 027/028 结果）：
- 027/028 出结果后，把 029 的发现作为对照
- 如果 027/028 也突破 0.86，开始考虑两个方向叠加（team-level + PBRS-init handoff）

长期（已被本 snapshot 排除的方向）：
- ~~PBRS 直接加在 025b~~ —— H1 否决
- ~~peer-only opponent pool~~ —— H3 弱正，不再加大投入
- ~~把 v2 的 `ball_progress` 替换为 `goal_proximity_pbrs`~~ —— 026 + 029 联合证据表明 PBRS 的"机制收益"不可迁移，不应作为默认 shaping

## 12. 相关

- [SNAPSHOT-017: BC@2100](snapshot-017-bc-to-mappo-bootstrap.md)（029-B 的 warm-start）
- [SNAPSHOT-018: opponent pool](snapshot-018-mappo-v2-opponent-pool-finetune.md)（029-C 的基础设施）
- [SNAPSHOT-025b: current champion](snapshot-025b-bc-champion-field-role-binding-stability-tune.md)（029-A/C 的 warm-start + 029-B 的 stability tune 来源）
- [SNAPSHOT-026: B-warm](snapshot-026-reward-liberation-ablation.md)（029-A/B 的 PBRS 设计 + 三相位训练轨迹观察）
- [SNAPSHOT-027: team-level scratch](snapshot-027-team-level-ppo-coordination.md)（并行架构实验）
- [SNAPSHOT-028: team-level BC→PPO](snapshot-028-team-level-bc-to-ppo-bootstrap.md)（并行架构实验）
