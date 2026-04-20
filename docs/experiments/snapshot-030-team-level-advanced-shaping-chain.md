# SNAPSHOT-030: Team-Level Advanced Shaping Chain

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 已完成首轮结果（`030-A / 030-D`）

## 0. 方法论前置（必读）

### 0.1 现状

per-agent 路径已经形成完整的 **base → advanced shaping** 链：

```
017 (BC + v2)                                  = 0.842 baseline (ref)
  → 025b (BC + v2 + field-role binding)        = 0.842 + H2H 赢 017
  → 029B (BC + B-warm@170 init + v2 handoff)   = 0.868 + H2H 平 025b ← per-agent SOTA
```

team-level 路径目前只跑到 **base**：

```
027 (scratch + v2)                             = 0.804
028A (BC + v2)                                 = 0.844 baseline
```

最近 H2H 数据显示 028A@1220 vs 017 = `0.466`（输 0.068），看上去 team-level 落后 per-agent。但**这是不公平的对比**。

### 0.2 不公平在哪里

当前所有方法栈——v2 shaping 系数、BC pipeline 设计、网络架构（512×512）、超参（lr=1e-4 / sgd_iter=4 / clip=0.15）、training trick——都是为 **single / per-agent multi-agent setup** 设计、调优、验证的。把它们 port 到 team-level 后，得到的"team-level 表现"实际上是：

> **"为 per-agent 优化过的方法栈，在 team-level 架构上的表现"**

这不等于"team-level 架构本身的上限"。

### 0.3 信息论直觉

team-level 架构 actor 看到 672 维联合 obs，per-agent 架构 actor 只看 336 维 own obs。**信息量更大的输入理论上应该有更高上限**——因为 per-agent 架构能学到的策略集是 team-level 能学到的策略集的真子集（team-level policy 可以 ignore 队友 obs 退化为 per-agent policy）。

如果 team-level 在所有指标上都被 per-agent 打败，**只能说明我们的方法栈没用上多出来的那一半信息**，不能说明"team-level 架构没用"。

### 0.4 本 snapshot 的范围与限制

本 snapshot **只做 port 实验**：把 per-agent 的 advanced shaping trick 直接搬到 team-level，作为：

1. **最低门槛检查**：team-level 架构能否从已知有效的 per-agent trick 中吃到收益
2. **基线对比**：为后续 team-level-native 方法实验（snapshot-031+）提供对照
3. **架构判断的下界**：如果 port 实验都能让 team-level 显著进步，说明架构有 headroom；如果 port 全失败，架构的 headroom 必须靠 native 方法挖

**不**做 team-level-native 方法（独立 snapshot 处理）：
- 双 encoder 网络（两人 obs 分别编码后融合）
- 跨 agent attention
- 团队协作 auxiliary loss
- 联合动作的 hierarchical decomposition
- 团队级 reward shaping（基于 formation / 区域分工的奖励）

### 0.5 口径修订记录（保留旧口径）

本 snapshot 在 2026-04-18 晚根据 `028A` 的 official / failure-capture / H2H 结果做了一次口径修订。这里显式保留原口径，避免后续读文档时看不出为什么设计改了。

| 项目 | 原口径 | 修订后口径 | 修订原因 |
|---|---|---|---|
| team-level warm-start base | `028A@1220` | **`028A@1060`** | `1220` 是 official 峰值，但 `500ep` capture 从 `0.844` 回落到 `0.796`；`1060 = 0.810 official / 0.806 capture` 更稳，更适合作为 port 实验基座 |
| 首轮并行 lane | `030-A + 030-B` | **`030-A + 030-D`** | `030-B` 的 team-level opponent-pool infra 目前并不存在；`030-A / 030-D` 已能直接落地，且信息密度更高 |
| `030-B` 工程判断 | “基础设施已存在，可首轮并行” | **“基础设施未完成，延后”** | 现有 opponent-pool trainer 只支持 per-agent shared-cc learned opponent，不支持 team-level learned policy pool |
| `030-C` 触发条件 | 不变 | 不变 | 仍需 `030-D` 先跑出 team-level PBRS 机制点 |

## 1. 三条 lane 的设计

按 per-agent advanced 的 029 模板，但适配 team-level base。

| 030 lane | 对应 per-agent | warm-start | shaping | opponent |
|---|---|---|---|---|
| **030-A** | 025b (field-role) | **028A@1060** | v2 + field-role binding | baseline |
| **030-B** | 029-C (opp pool) | **028A@1060** | v2 | pool |
| **030-C** | 029-B (PBRS handoff) | （需 030-D 先跑出 PBRS mech 点）| — | — |
| **030-D** | 029-A (PBRS on base) | **028A@1060** | PBRS only | baseline |

030-D 同时是 030-C 的前置：必须先有 team-level PBRS 训练，才知道有没有"team-level B-warm @170 等价物"，才能做 handoff。

按风险/收益排序，**首轮先跑 030-A 和 030-D 并行**：
- 030-A 风险最低（25b 在 per-agent 上明确成功，H2H 赢 017）
- 030-D 是最低工程成本的开放探索（PBRS 在 per-agent 029-A 上失败，但 team-level 上可能不同）
- 030-B 工程最复杂，而且当前 learned opponent-pool 基础设施只覆盖 per-agent，不覆盖 team-level
- 030-C 等 030-D 出结果再决定

`030-B` 现在保留在本 snapshot 中，但正式降级为“第二轮 / infra 补完后再起”的 lane。

## 2. 路径 A — `028A + team-level field-role binding`

### 2.1 设计

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | **028A@1060** | team-level base（更稳的主候选） |
| shaping 类型 | v2 + spawn-depth field-role binding | 025b 配置 |
| network | 512x512 | 匹配 028A |
| `lr` | 1e-4 | 025b/029 同款 |
| `num_sgd_iter` | 4 | 同上 |
| `clip_param` | 0.15 | 同上 |
| opponent | baseline 100% | 纯对照 |
| `SHAPING_FIELD_ROLE_BINDING` | 1 | 启用 |
| `SHAPING_FIELD_ROLE_BINDING_MODE` | spawn_depth | 025b 同款 |

### 2.2 工程检查点

- [ ] 确认 `RewardShapingWrapper` 在 `team_vs_policy` 模式下正确路由 per-agent role-bound reward
- [ ] 1-iter smoke：训练 log 应该看到 `role_by_agent = {0: 'striker', 1: 'defender'}`（或反之）

### 2.3 假设

**H_A 成立**：team-level 架构能从 per-agent 验证过的 field-role binding 中吃到同等收益。

预期：
- 028A→030-A 的 official 500 提升 +0.00 到 +0.02（类比 017→025b 在 per-agent 上的收益）
- H2H vs `028A@1060` ≥ 0.52
- H2H vs 017@2100 改善（从输 0.068 拉到 ≥ 0.50）

**H_A 不成立**：
- field-role binding 在 team-level 上无效或负效应
- 暗示 team-level 架构对"per-agent 风格的 reward asymmetry"不响应
- 进一步暗示需要 team-level-native asymmetry 设计（如 sub-policy / hierarchical action）

## 3. 路径 B — `028A + opponent pool`

### 3.1 设计

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | **028A@1060** | team-level base |
| shaping | v2（不变）| 028A 同款 |
| network | 512x512 | 同上 |
| PPO | tight params 同 029-C | |
| opponent pool | baseline 40% / 017@2100 20% / 025b@80 20% / 028A@1060-self 20% | 复用 029-C 思路但 self 替换 |

注意：029-C 的 pool 是 `baseline / 017 / 024 / 025b-self`。030-B 的 pool 是 `baseline / 017 / 025b / 028A-self`——把 024 换成 025b（因为 025b 比 024 更强，且 028A 还没和 025b 在 H2H 中赢过）。

### 3.2 工程关键点

team-level policy 用的是 `trained_team_ray_agent` deployment wrapper，per-agent SOTA (017/025b) 用的是 `trained_shared_cc_agent`。当前 learned opponent-pool 基础设施 [train_ray_mappo_vs_opponent_pool.py](../../cs8803drl/training/train_ray_mappo_vs_opponent_pool.py) 明显是 **per-agent shared-cc 专用**，并不支持 team-level learned policy pool。

所以 030-B 现阶段的工程状态应视为：
- baseline/random mixed opponent：已存在
- single frozen team-level teammate checkpoint：已存在
- **learned opponent pool（含 team-level self slot）**：**未完成**

因此 030-B 不再作为首轮 lane，而是保留为下一轮 infra 任务。

### 3.3 假设

**H_B 成立**：team-level policy 在 per-agent peer pool 中训练后，对同级 per-agent 的 H2H 表现改善（从输 0.07 拉到 ≥ 0.50）。

如果 H_B 成立但 H_A 失败，说明 team-level 的核心弱点在 **out-of-distribution against per-agent style**，而 opponent pool 训练能直接修复这个弱点。

## 4. 路径 D — `028A + PBRS only`（开放探索）

### 4.1 设计

| 参数 | 值 | 来源 |
|---|---|---|
| warm-start | **028A@1060** | team-level base |
| shaping | PBRS only（同 029-A 配置） | |
| network | 512x512 | |

### 4.2 假设

**H_D 成立**：team-level 架构上 PBRS 比 v2 更适合，能压低 `low_poss`。

注意 029-A（per-agent + PBRS）失败了。但 team-level 的 reward landscape 不一样——PBRS 的"球离门近就好"信号在 team-level 联合视角下可能引发不同行为。

如果 H_D 成立且产生明显 `low_poss` 改善（≤ 25%），030-D 的 best ckpt 就是 030-C 的 PBRS handoff 起点。

如果 H_D 失败，030-C 也不必跑——直接说明 PBRS 在 team-level 上也不是钥匙。

## 5. 执行矩阵

| lane | 复杂度 | 预算 | 优先级 |
|---|---|---|---|
| **030-A** field-role binding | 低 | ~6h | **首轮 #1** |
| **030-B** opp pool | 高（需补 team-level pool infra） | ~8h+ | 第二轮 |
| **030-D** PBRS only | 低 | ~6h | **首轮 #2** |
| **030-C** PBRS handoff | 中（依赖 030-D 结果） | ~6h | 第二轮 |

GPU 同时跑 2 条 lane（A/D），单轮 `~6h` 级别即可完成首轮。

## 6. 预声明判据

### 6.1 主判据

| 030 lane | official 500 阈值 | H2H 阈值 |
|---|---|---|
| 030-A | ≥ 0.86（追平 026B@250 / 接近 029B@190）| H2H vs 017 ≥ 0.50 + H2H vs `028A@1060` 略赢 |
| 030-B | ≥ 0.84（不损失 baseline WR）| H2H vs 017/025b/029B 都 ≥ 0.48 |
| 030-D | ≥ 0.84 | failure capture 中 `low_poss` ≤ 25% |

**任一 lane 触发 H2H ≥ 0.50 vs 029B**，team-level 架构就证明了它的潜力，下一步明确转向 team-level + 更多 trick。

### 6.2 失败情形

| 条件 | 解读 |
|---|---|
| 三条 lane 全部 official < 0.83 | per-agent advanced trick 在 team-level 上不通用 → 必须做 team-level-native 方法（snapshot-031+）|
| 030-A 显著 +0.02 但 030-D 失败 | 架构感知 reward asymmetry 但不感知 PBRS 差异 → 优先继续 field-role 方向 |
| 030-A/B/D 都接近 028A 水平（≈ 0.84） | per-agent trick 完全没在 team-level 上 transfer → 转 native 方法 |

## 7. 与现有 lane 的关系

| snapshot | 关系 |
|---|---|
| [025b](snapshot-025b-bc-champion-field-role-binding-stability-tune.md) | 030-A 的 per-agent 模板 |
| [029-A](snapshot-029-post-025b-sota-extension.md) | 030-D 的 per-agent 模板 |
| [029-B](snapshot-029-post-025b-sota-extension.md) | 030-C 的 per-agent 模板 |
| [029-C](snapshot-029-post-025b-sota-extension.md) | 030-B 的 per-agent 模板 |
| [028A](snapshot-028-team-level-bc-to-ppo-bootstrap.md) | 030 所有 lane 的 warm-start 源；当前首选基座为 `028A@1060` |

## 8. 不做的事

- **不做 team-level-native 方法**（双 encoder / cross-agent attention / 团队级 reward）—— 这是 snapshot-031+ 的范围
- **不做架构改动**（FCNet 层数 / actor-critic 拆分）—— 030 只测 reward & opponent 改动
- **不并行做 030-C**（必须等 030-D 出结果决定有无必要）

## 9. 工程依赖

### 9.1 已存在

- 028A@1060 checkpoint（首选 team-level warm-start base）
- 028A@1220 checkpoint（保留作 official 峰值参考点）
- 027/028 team-level 训练入口
- per-agent field-role binding env vars（`SHAPING_FIELD_ROLE_BINDING=*`）
- per-agent PBRS env vars（`SHAPING_GOAL_PROXIMITY_*`）
- per-agent opp pool 基础设施（`POOL_*_CHECKPOINT`）
- team-level H2H wrapper [trained_team_ray_opponent_agent.py](../../cs8803drl/deployment/trained_team_ray_opponent_agent.py)

### 9.2 需要确认

- field-role binding 的 spawn-depth role 计算在 `team_vs_policy` env 下是否正确
- `RewardShapingWrapper` 的 per-agent reward 在 `TeamVsPolicyWrapper.step()` sum 之前是否仍然 role-aware
- `030-D` 的 PBRS 在 team-level base 上是否能稳定压低 `low_poss`

### 9.3 可能需要新增

- `030-B` 需要独立的 team-level learned opponent-pool trainer，或给现有 opponent-pool infra 补 team-level 分支

## 10. 执行清单

1. 确认工程依赖 §9.2 的三项
2. 起 batch：`030-A / 030-D`
3. 1-iter smoke 验证：
   - 030-A：log 中能看到 `role_by_agent = {...}`
   - 030-D：log 中能看到 `shaping_goal_prox: 0.01`
4. 提交首轮 2 条并行（A/D）
5. 按 §6 判据做 verdict
6. 如果 `030-D` 触发 `030-C` 条件，下一轮起 `030-C`
7. 如果决定继续做 `030-B`，先补 team-level learned opponent-pool infra
8. 决定是否进入 snapshot-031（team-level-native 方法）

## 11. 相关

- [SNAPSHOT-025b: per-agent field-role binding](snapshot-025b-bc-champion-field-role-binding-stability-tune.md)
- [SNAPSHOT-027: team-level scratch](snapshot-027-team-level-ppo-coordination.md)

## 12. 首轮训练结果

### 12.1 `030-A`：`028A@1060 + field-role binding`

- run dir:
  - [030A_team_field_role_on_028A1060_512x512_20260418_051107](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107)
- internal best:
  - [checkpoint-20](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_9578c_00000_0_2026-04-18_05-11-29/checkpoint_000020/checkpoint-20) = `0.920 @ baseline-50`
- official `baseline 500` 高点:
  - [checkpoint-360](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_9578c_00000_0_2026-04-18_05-11-29/checkpoint_000360/checkpoint-360) = `0.832`
  - [checkpoint-290](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_9578c_00000_0_2026-04-18_05-11-29/checkpoint_000290/checkpoint-290) = `0.830`
  - [checkpoint-300](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_9578c_00000_0_2026-04-18_05-11-29/checkpoint_000300/checkpoint-300) = `0.830`
- failure capture:
  - [checkpoint-360](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_9578c_00000_0_2026-04-18_05-11-29/checkpoint_000360/checkpoint-360) = `0.842`
  - [checkpoint-300](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_9578c_00000_0_2026-04-18_05-11-29/checkpoint_000300/checkpoint-300) = `0.796`

`030-A` 的主要读法:
- internal `0.920 @ 20` 没有在 official 上站住，因此不应把首轮结论收在早期 warm-start 峰值。
- 更可信的主候选应收口到 [checkpoint-360](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_9578c_00000_0_2026-04-18_05-11-29/checkpoint_000360/checkpoint-360)。
- `360` 的 official / capture 对齐良好（`0.832 -> 0.842`），说明这条线更像稳定的中后段强窗口，而不是短程假峰。

### 12.2 `030-D`：`028A@1060 + team-level PBRS`

- run dir:
  - [030D_team_pbrs_on_028A1060_512x512_20260418_051114](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030D_team_pbrs_on_028A1060_512x512_20260418_051114)
- internal best:
  - [checkpoint-30](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030D_team_pbrs_on_028A1060_512x512_20260418_051114/TeamVsBaselineShapingPPOTrainer_Soccer_99393_00000_0_2026-04-18_05-11-35/checkpoint_000030/checkpoint-30) = `0.900 @ baseline-50`
- official `baseline 500` 高点:
  - [checkpoint-320](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030D_team_pbrs_on_028A1060_512x512_20260418_051114/TeamVsBaselineShapingPPOTrainer_Soccer_99393_00000_0_2026-04-18_05-11-35/checkpoint_000320/checkpoint-320) = `0.862`
  - [checkpoint-360](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030D_team_pbrs_on_028A1060_512x512_20260418_051114/TeamVsBaselineShapingPPOTrainer_Soccer_99393_00000_0_2026-04-18_05-11-35/checkpoint_000360/checkpoint-360) = `0.856`
  - [checkpoint-330](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030D_team_pbrs_on_028A1060_512x512_20260418_051114/TeamVsBaselineShapingPPOTrainer_Soccer_99393_00000_0_2026-04-18_05-11-35/checkpoint_000330/checkpoint-330) = `0.840`
- failure capture:
  - [checkpoint-320](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030D_team_pbrs_on_028A1060_512x512_20260418_051114/TeamVsBaselineShapingPPOTrainer_Soccer_99393_00000_0_2026-04-18_05-11-35/checkpoint_000320/checkpoint-320) = `0.820`
  - [checkpoint-360](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030D_team_pbrs_on_028A1060_512x512_20260418_051114/TeamVsBaselineShapingPPOTrainer_Soccer_99393_00000_0_2026-04-18_05-11-35/checkpoint_000360/checkpoint-360) = `0.808`
- key H2H:
  - [030D_320_vs_028A_1060.log](/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/030D_320_vs_028A_1060.log): `268W-232L = 0.536`
  - [030D_320_vs_025b_080.log](/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/030D_320_vs_025b_080.log): `234W-266L = 0.468`
  - [030D_320_vs_025b_080_rerun.log](/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/030D_320_vs_025b_080_rerun.log): `225W-275L = 0.450`
  - [030D_320_vs_029B_190.log](/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/030D_320_vs_029B_190.log): `219W-281L = 0.438`
  - [030D_320_vs_029B_190_rerun.log](/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/030D_320_vs_029B_190_rerun.log): `222W-278L = 0.444`
  - [030D_320_vs_030A_360.log](/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/official-evals/headtohead/030D_320_vs_030A_360.log): `252W-248L = 0.504`

`030-D` 的主要读法:
- 这条线的 official ceiling 明显高于 `030-A`，说明 team-level PBRS 确实能在 `028A` 底座上制造更强的高点。
- 但 `320/360` 的 `500ep` capture 都比 official 低 `0.04~0.05`，因此这条线当前更像“高 ceiling 候选”，而不是已经完全坐实的稳定主候选。
- H2H 给出的信息比较清楚：`030D@320` 已经真实超过 [028A@1060](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060)，但仍低于 [025b@80](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80) 与 [029B@190](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190)。
- `030D@320 vs 025b@80` 的 rerun 从 `0.468` 收紧到 `0.450`，因此这条线目前更适合写成“真实低于 `025b`”，而不是“可能只是边缘噪声”。
- 追加 H2H 进一步说明 `030D@320` 和 [030A@360](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_9578c_00000_0_2026-04-18_05-11-29/checkpoint_000360/checkpoint-360) 的差距目前很小：`030D@320 vs 030A@360 = 0.504`，因此现在更适合写成“略强”，不适合写成“已经明显超过”。

## 13. 首轮 verdict

首轮结果支持下面这个更克制、也更稳定的收口：

- `030-A` 是真实正结果线，且 [checkpoint-360](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030A_team_field_role_on_028A1060_512x512_20260418_051107/TeamVsBaselineShapingPPOTrainer_Soccer_9578c_00000_0_2026-04-18_05-11-29/checkpoint_000360/checkpoint-360) 是一个 official/capture 对齐良好的可信主候选。
- `030-D` 是 `030` 里更强的一条线；[checkpoint-320](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/030D_team_pbrs_on_028A1060_512x512_20260418_051114/TeamVsBaselineShapingPPOTrainer_Soccer_99393_00000_0_2026-04-18_05-11-35/checkpoint_000320/checkpoint-320) 既有最高 official ceiling，又能在 direct H2H 中真实超过 `028A@1060`。
- 但 `030-D` 还不足以改写顶线：它目前仍然输给 `025b@80` 和 `029B@190`。
- 其中 `vs 025b@80` 的 rerun 已把这点进一步坐实。
- 追加 H2H 也显示两条 `030` lane 对 `029B@190` 都还明显落后：`030A@360 = 0.450`，`030D@320 = 0.444`。

因此，当前更合适的定位不是“team-level 已经追平 per-agent 顶线”，而是：

- `030-A`: 更稳的 team-level advanced 正结果
- `030-D`: 更强的 team-level advanced 候选，但当前仍属于“强增量线”，不是总冠军线
- 当前最克制的排序应写成：`030D` 略强，`030A` 更稳，两者都仍低于 `029B`

## 14. 评测口径更新

随着 `028/029/030/033` 这些 frontier 点越来越接近，单次 official `500` 已经开始不够稳定地支撑最终排序。后续在本仓库中默认采用下面的分层口径：

- `official 500`:
  - 继续作为窗口扫描与候选筛选的默认配置
  - 适合从 `top 5% + ties + window` 中快速找出最值点
- `failure capture 500`:
  - 继续作为稳定性与失败结构判断的默认配套
  - 用来识别 official 高点是否被高估
- `H2H 500`:
  - 继续作为“是否真的超过 base / 旧冠军 / 同代对照”的默认规模
- `official 1000`:
  - 不再默认全量跑
  - 只在下列情况升级：
    - official 和 capture 差距超过约 `0.03`
    - H2H 与 official 给出不同排序
    - 候选间 official 差距落在 `0.015~0.020` 以内
    - 该点将被写入 snapshot 作为正式主候选或冠军位判断

按这个新口径，`030D@320` 这类“official 高、capture 回落、且已经进入前沿比较”的 checkpoint，就是后续最适合升级到 `official 1000` 的典型对象。

## 15. Follow-up：`030D-control`

在 `030D` 首轮结果出来之后，又出现了一个额外但很关键的问题：

- `030D` 的正信号到底来自 **team-level PBRS**
- 还是来自 **以 `028A@1060` 为起点继续训练 500 iter** 这件事本身

为把这两个因素拆开，补一个最小对照：

| lane | warm-start | shaping | 目的 |
|---|---|---|---|
| `030D` | `028A@1060` | goal-proximity PBRS + 其余 `030D` continuation 条件 | 主实验 |
| `030D-control` | `028A@1060` | **关闭 goal-proximity PBRS**，其余 continuation 条件保持与 `030D` 一致 | 隔离 continuation 效应 |

这里特意不把 `030D-control` 改成 `028A` 的原始 `v2` 全量配置，而是使用“与 `030D` 完全同 skeleton、只关闭 PBRS”的最小对照。这样后续如果 `030D` 明显强于 `030D-control`，解释会更直接。
- [SNAPSHOT-028: team-level BC bootstrap](snapshot-028-team-level-bc-to-ppo-bootstrap.md)
- [SNAPSHOT-029: per-agent advanced shaping trio](snapshot-029-post-025b-sota-extension.md)
