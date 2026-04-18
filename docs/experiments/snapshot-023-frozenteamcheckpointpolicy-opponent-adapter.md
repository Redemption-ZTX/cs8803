# SNAPSHOT-023: FrozenTeamCheckpointPolicy Opponent Adapter

- **日期**: 2026-04-15
- **负责人**:
- **状态**: 预注册 / **On-hold**（等待触发条件，见 §3）

## 1. 背景与编号说明

本 snapshot 原计划编号为 **021**，作为 [SNAPSHOT-020](snapshot-020-mappo-v4-fair-ablation.md) 的 conditional follow-up：**只有当 020 证明 v4 风格在 MAPPO 主干上有价值时，才值得做 team-level PPO v4 → multiagent_player pool 的兼容适配工程**。

2026-04-15 编号 021/022 被重用于 [`low_possession` 跨 lane 不变量](snapshot-013-baseline-weakness-analysis.md#12-bc%E2%86%92mappo-数据对-11-的进一步强化2026-04-15-后补)的根因 A/B 假设检验（[SNAPSHOT-021 obs expansion](snapshot-021-actor-teammate-obs-expansion.md) / [SNAPSHOT-022 role-differentiated shaping](snapshot-022-role-differentiated-shaping.md)），本 adapter 工作遂移至 **023**。

## 2. 工程目标

在 [`train_ray_mappo_vs_opponent_pool.py`](../../cs8803drl/training/train_ray_mappo_vs_opponent_pool.py) 的 frozen-opponent 基础设施上增加一个新的 wrapper class，让**以 `single_player=True + team_vs_policy` 语义训练的 team-level policy**（例如 [v4 PPO @ ckpt400](../../ray_results/PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133/TeamVsBaselineShapingPPOTrainer_Soccer_9cd69_00000_0_2026-04-13_05-31-56/checkpoint_000400/checkpoint-400)）**可以作为** pool 中的 frozen opponent 被调度。

目前的兼容缺口（见 [先前讨论](../../docs/experiments/snapshot-019-mappo-v2-opponent-pool-anchor-heavy-rebalance.md)）：

| 维度 | pool 训练期望 | v4 PPO 原始接口 |
|---|---|---|
| env variation | `multiagent_player` | `team_vs_policy` |
| agent 查询模式 | 每 agent 独立 query | team-level policy 一次决策两个 agent |
| obs space | per-agent own_obs | single_player 的 concat-style obs |
| model | `shared_cc_model` | vanilla PPO fcnet (no CC) |

## 3. 触发条件（上游依赖）

本 snapshot 的**启动依赖于以下任一信号出现**：

### 条件 A — [SNAPSHOT-020](snapshot-020-mappo-v4-fair-ablation.md) 的正向结果

snapshot-020 证明 MAPPO + v4 shaping 比 MAPPO + v2 shaping 强（500-ep ≥ 0.79 + late_collapse 或 low_poss 明显下降）。

**2026-04-15 当前状态**：[snapshot-020 首轮结果](../README.md) 显示 `MAPPO + v4 shaping` 最好点 `0.764`，**不成立**。

**这意味着条件 A 当前未触发**，v4 风格在 MAPPO 主干上**未被验证有价值**。在 020 结果不翻转之前，做 023 工程**期望 ROI 很低**——因为如果 MAPPO+v4-shaping 自己都打不过 MAPPO+v2，加真 v4 PPO 作为 frozen opponent 预期更没用。

### 条件 B — 直接需求 PPO 作为 pool diversity source

如果后续出现对 "pool 里必须有 non-MAPPO 架构" 的需求（例如 ensemble 多样性分析、跨算法 transfer 研究），**即使 020 失败**本 snapshot 也可能被启动。

### 条件 C — 其他 team-level checkpoint 成为强候选

未来若出现新的 team-level policy（例如其他 env 设置下训出的强 checkpoint），需要以 frozen opponent 身份进入 multiagent_player pool，本 adapter 同样适用。

**当前状态**：三个条件均未满足。本 snapshot 作为**工程预案**存档，不主动启动。

## 4. 实现设计

### 4.1 新增类：`FrozenTeamCheckpointPolicy`

位置：[cs8803drl/core/frozen_team_policy.py](../../cs8803drl/core/frozen_team_policy.py)（新建）

继承 `ray.rllib.policy.policy.Policy`，内部行为：

```
initialize:
  - 加载 team-level checkpoint (PPO fcnet 权重)
  - 记录 team checkpoint 的 obs / action space
  - 初始化 per-episode state cache

compute_actions_from_input_dict(input_dict):
  input_dict 包含本次 query 的 agent_id（来自 multiagent_player dispatch）
  - 从 input_dict 提取 agent_id → 决定是 team0_slot_0 还是 team0_slot_1
  - 把 own_obs 存入 per-episode cache
  - 若两个 slot 的 obs 都已收到:
      - 组合成 team-level obs (按 team checkpoint 的 concat 约定)
      - 调用 team policy 一次，获得 team-level action (含两 agent 的决策)
      - 缓存两个 agent 的 action，清空 obs cache
  - 返回当前 agent_id 对应的 action
```

### 4.2 两个关键难点

**(1) Ray multiagent dispatch 的 per-agent 调用可能 batched**

Ray 可能在同一 call 里 batch 多个 env 实例的多个 agent。adapter 必须按 env_id × agent_id **精确配对** own_obs，不能错配。

缓解：用 `input_dict["eps_id"]` 或 `episode_id` 作为 cache key，保证同 episode 内两 slot obs 对齐。

**(2) 同 episode 两 agent 的调用顺序不保证**

Ray 可能先 call agent 0 再 agent 1，也可能反过来。adapter 必须对两种顺序都稳定。

缓解：pool two obs before computing action；只有两个都到了才 query team policy，否则 return "placeholder" action 并在下一次调用时补齐——但这会破坏 Ray 的 per-step 一致性。更稳的方案：**强制 ray config 中 envs_per_worker = 1 或 2**，让 call 顺序可预测。

### 4.3 Smoke test 计划

1. 构造最小单 env + 2 agent 的 MAPPO training，用 `FrozenTeamCheckpointPolicy` 包装 v4 PPO @ ckpt400 作为 team1
2. 验证：
   - checkpoint 能加载无异常
   - 1 iter 训练能走完
   - eval log 显示 v4 正确返回 action（不是随机 / 不是恒定）
3. 验证 v4 作为 team1 时的胜负统计大致接近 v4 自己对 baseline 的 0.24 输率（即 pool@290 应能约 50% 赢 v4）

## 5. 预声明判据（仅在本 snapshot 启动后适用）

### 5.1 工程主判据

1. 集成 v4 PPO 进 pool 后，完整 300 iter fine-tune 能跑完不崩
2. 训练期间 per-agent reward 流统计显示 v4 正常参与（team1 平均胜率不是 0% 或 100% 的极端值）

### 5.2 性能主判据

本 snapshot 完成工程后，**用 v4 PPO 代替 [SNAPSHOT-019](snapshot-019-mappo-v2-opponent-pool-anchor-heavy-rebalance.md) 中的 MAPPO no-shape 角色**，跑 30/20/20/20/10 配比（baseline/anchor/v1/no-shape/**v4**）：

- **500-ep WR ≥ 0.81**（对比 snapshot-019 anchor-heavy 配比的结果 0.788）
- 机制判据：失败桶分布**和 019 相比产生可解释的结构变化**（例如 late_collapse 进一步下降，或 unclear_loss 占比朝 BvB 的 47% 靠拢）

### 5.3 失败情形

1. Adapter 实现崩溃（race condition / batch 错位）→ 工程不 ready，推迟或放弃
2. 500-ep WR < 0.78 → 加 v4 PPO 反而损害 pool 平衡，不值得继续扩展
3. 数据显示 pool@final 的失败分布和 019 几乎一致 → v4 加了也等于没加，放弃

## 6. 不做的事

- **不在 020 证明 v4 风格有价值之前启动本 snapshot**
- **不改 v4 PPO 原始训练流程**（只做 inference-time adapter）
- **不把 adapter 机制泛化到其他 single_player checkpoint**（除非有第三个用例出现）
- **不改 multiagent_player env 本身的 dispatch 逻辑**

## 7. 风险

### R1 — 条件 A 大概率不会满足

snapshot-020 首轮已给出 `MAPPO + v4 shaping = 0.764` 的负结果。即使后续再跑也**不太可能翻到 > 0.786**。这意味着本 snapshot **很可能永远不会启动**。

处理方式：
- 接受这个现实
- 本文件作为"如果未来改变主意，工程设计已在此"的存档
- 主动不去启动它，把 GPU 资源留给 021/022 根因检验 或 report/submission 工作

### R2 — 工程复杂度 vs 收益的不对称

FrozenTeamCheckpointPolicy 的实现预算是 **1-2 天工程**（Ray multiagent dispatch 的边界条件调 debug 尤其耗时）。而预期性能收益基于 snapshot-020 的负结果**很可能 ≤ +0.01 WR**。

投入产出比极低。除非有触发条件 B 或 C 出现，**不推荐主动启动**。

### R3 — 已有 "MAPPO+v4-shaping" 替代品

snapshot-020 的 `MAPPO + v4 shaping` 输出已经存在（见 [run dir](../../ray_results/PPO_mappo_vs_baseline_shaping_v4_512x512_20260414_235000)——路径占位，按实际路径填），这是"v4 风格行为" 的 **MAPPO-架构版本**。如果 pool diversity 需要"v4 风格行为"，**直接用 MAPPO+v4-shaping checkpoint 作为 frozen opponent**，不需要 adapter。

这是本 snapshot 被**工程上绕过**的最自然方式——**大概率是 023 永远不会被启动的原因**。

## 8. 与其他 snapshot 的关系

| snapshot | 关系 |
|---|---|
| [SNAPSHOT-019](snapshot-019-mappo-v2-opponent-pool-anchor-heavy-rebalance.md) | opponent pool 配比研究 / 提及 023 作为"加跨架构成员"条件分支 |
| [SNAPSHOT-020](snapshot-020-mappo-v4-fair-ablation.md) | **023 的上游 gate**——020 结果决定 023 是否启动 |
| [SNAPSHOT-016](snapshot-016-shaping-v4-survival-anti-rush-ablation.md) | v4 PPO 本身的训练和结果（checkpoint 来源）|

## 9. 下一步

**目前无下一步**。本 snapshot 静态存档，等待：

- snapshot-020 翻转证据（需要 v4 风格在 MAPPO 上证明有价值），或
- 明确指令启动，或
- 新的 team-level policy 候选出现需要 frozen adapter

若触发，按 §4 设计 → §5 判据 → verdict 落到本文件 §10+ (append-only)。
