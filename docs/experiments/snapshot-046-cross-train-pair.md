# SNAPSHOT-046: Cross-Train Pair — Train vs Frozen 031A, Eval vs Baseline

- **日期**: 2026-04-19
- **负责人**:
- **状态**: 预注册 / 待 FrozenTeamCheckpointPolicy adapter (snapshot-023) 实施
- **依赖**:
  - [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md): `031A@1040` 作为 frozen opponent
  - [SNAPSHOT-023](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) FrozenTeamCheckpointPolicy adapter (待实施)

## 0. Known gap 诊断（动机来源）

当前 045 / 040 / 041 / 044 全部 lane 训练对手都是 **baseline (CEIA rule-based agent)**。结果集中卡在 **vs baseline 1000ep WR ~0.86 ceiling**。这个 ceiling 的归因有两种竞争假设：

**假设 A (environmental ceiling)**: baseline 有 ~14% inevitable noise（baseline vs random = ~0.86，存在 14% lucky scoring/clearance），任何 policy 都打不过 14%。

**假设 B (PPO ceiling)**: PPO 学习力不够，单次 50ep eval 偶尔触 0.94/0.96/0.98 说明 episode-level 上没硬 ceiling。1000ep 收敛到 0.86 是 PPO 找的局部最优。

如果是 A，stronger policy 也卡 0.86 (无解)。
如果是 B，**stronger policy 应该能突破 0.86**。

### 假设 B 的直接测试

> **如果训练一个新 policy P，唯一对手是 031A@1040 (frozen, 当前 SOTA)，让 P 达到 H2H vs 031A ≥ 0.55，然后 eval P vs baseline。如果 P vs baseline > 0.86，证明 PPO ceiling 假设 B (假设 A 否决)；并验证 skill transitivity (vs strong → vs weak 也强)。**

## 1. 核心假设

> **PPO 训练 vs baseline 收敛到 0.86 是 PPO 在该 reward landscape 的局部最优。如果换训练对手为强对手 (031A@1040)，PPO 找的局部最优会显著移动 → 学到更 general policy → vs baseline WR 也提升。**

子假设：

- **H1 (skill transitivity 成立)**: P 训完 vs 031A H2H ≥ 0.55，eval P vs baseline ≥ 0.88 (+2pp 突破 ceiling)
- **H2 (skill transitivity 强成立)**: P vs baseline ≥ 0.90 (**首次达到 9/10 grading 阈值**)
- **H3 (assumption A 验证)**: P vs baseline ≈ 0.86 (no improvement) → 证明 environmental ceiling，0.86 真是 baseline noise floor + PPO 能力共同上限
- **H4 (P 训不出来)**: P 训不到 H2H vs 031A ≥ 0.5，证明 031A 在自身策略空间内已经是 attractor，cross-train 也 stuck (rare 但可能)

如果 H2 突破，**项目首次直接达成 grading 目标 (9/10 vs baseline = 0.90)**。

## 2. 与 SNAPSHOT-043 的关系

| 维度 | SNAPSHOT-043 (frontier league) | **SNAPSHOT-046 (本)** |
|---|---|---|
| 对手分布 | baseline 40% + 4 frontier × 15% (mix) | **frozen 031A@1040 100%** (single, 最强对手) |
| 训练目标 | 学多样性 → 间接提升 baseline | 直接 push policy 超越 031A → 间接提升 baseline |
| Trainable | 031A@1040 warmstart | **scratch from random init (cleanest test)** OR 031A@1040 warmstart |
| Eval | vs baseline (grading) + vs frontier (general) | **vs baseline (grading) only** + 直接 H2H vs 031A |
| 测试假设 | "diverse opponents 提升 robustness" | "skill transitivity: vs stronger → vs weaker 也强" |
| 实施成本 | ~200 行 adapter + 12h GPU | 共享 adapter + ~14h GPU (scratch) 或 ~3h (warmstart) |

**两个并行实验**: 不同 hypothesis，可对照"diversity vs intensity"哪个更有效。

## 3. 三条候选 lane

| Lane | Trainable 起点 | 对手 | Reward | 测试什么 |
|---|---|---|---|---|
| **046A** (主线 scratch) | scratch (random init Siamese) | frozen 031A@1040 100% | v2 (031A 同款) | scratch + 训 strong opponent 能否 organically 学到 baseline-strong policy |
| **046B** (warmstart, 快验证) | 031A@1040 warmstart | frozen 031A@1040 100% | v2 | 已经 strong 的 policy 训 vs 自己能否变更强 (类 self-play 单步) |
| **046C** (条件: scratch + sparse) | scratch | frozen 031A@1040 100% | sparse only (`USE_REWARD_SHAPING=0`) | 极端 minimal-prior 测试，看纯 outcome signal 能否驱动 |

**首轮跑 046B (3h 快验证)**:
- 如果 H1 成立 (P vs baseline ≥ 0.88) → 强证据，启动 046A 14h 完整 scratch 试 H2 突破 0.90
- 如果 P vs baseline = 0.86 (saturation) → assumption A 倾向，跳过 046A/C，看 self-play 后续路径
- 如果 P 训退化 (loses to baseline 0.5)，可能 mode collapse exploit 031A 特定弱点，需要诊断

## 4. 训练超参（046B 主线）

```bash
WARMSTART_CHECKPOINT=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/.../checkpoint_001040/checkpoint-1040

# Architecture (031A 同款)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128

# Opponent: 100% frozen 031A
OPPONENT_POOL_BASELINE_PROB=0.0       # NO baseline in training
OPPONENT_POOL=frozen_031A:1.0
OPPONENT_FROZEN_CKPT_frozen_031A=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/.../checkpoint_001040/checkpoint-1040
OPPONENT_FROZEN_LANE_frozen_031A=team_level

# Reward (v2 同 031A)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001
... (same v2 as 031A)

# PPO
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000
SGD_MINIBATCH_SIZE=2048

# Budget
MAX_ITERATIONS=200       # 046B warmstart 200 iter ~3h
EVAL_INTERVAL=10
EVAL_EPISODES=50
EVAL_OPPONENTS=baseline,random  # eval 时仍测 vs baseline (grading axis)
CHECKPOINT_FREQ=10
```

**关键差异 vs 040 系列**: training 时 `OPPONENT_POOL` 设为 100% frozen 031A，eval 时仍 vs baseline 评分。这正是测 "train on stronger, eval on weaker" 的核心设计。

## 5. 预注册判据

### 5.1 主判据（成立门槛）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| training reward 收敛 | > -0.5 by iter 100 | P 真在学（不是被 031A 打成蜡烛） |
| H2H vs 031A@1040 (frozen, training opponent) | ≥ 0.50 by end | P 至少打平训练对手 |
| **vs baseline 1000ep peak** | **≥ 0.86** | 至少不退化 (031A baseline) |

### 5.2 突破判据（强成立）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| **vs baseline 1000ep peak** | **≥ 0.88** (+2pp) | 直接证明 skill transitivity |
| **vs baseline 1000ep peak** | **≥ 0.90** | 🎯 **达成 grading 9/10 阈值** |
| H2H vs 031A frozen | ≥ 0.55 | P 真比训练对手强 |
| H2H vs 029B@190 | ≥ 0.55 | 多个 frontier 也强 (验证 general，不是只 over-fit 031A) |

### 5.3 失败判据

| 条件 | 解读 |
|---|---|
| training reward < -1 by iter 100 | 训练 collapse, vs frozen-strong 太难 |
| H2H vs 031A < 0.45 by end | P 学不动训练对手, mode collapse |
| vs baseline ≈ 0.86 | assumption A (environmental ceiling) 倾向，0.86 真是上限 |
| vs baseline < 0.80 | over-fit to 031A's 特定弱点, 失去 baseline general |

### 5.4 lane 优先级

1. **046B (warmstart, 3h)** — 首轮主线，快速 verdict
2. **046A (scratch, 14h)** — 仅当 046B 显示 +2pp 增益时启动
3. **046C (scratch + sparse, 14h)** — 仅当 046A 突破 + 想进一步移除 v2 prior 时启动

## 6. 与 045 的关系

045 测 "**reward 加 learned signal** 能否突破 0.86"
046 测 "**对手换 strong** 能否突破 0.86"

两者**正交**：
- 如果 045 突破 → reward signal 是关键 → reward path 优先
- 如果 046 突破 → opponent distribution 是关键 → self-play / league 路径优先
- 如果都突破 → **两者可叠加** (046 + learned reward)
- 如果都不突破 → 0.86 是 PPO ceiling，必须换 algo (forbidden) 或 deploy-time tricks (snapshot-034)

## 7. 风险

### R1 — Mode collapse to 031A-specific exploits

P 训完只见 031A，可能学到"专破 031A 的特定弱点"。这类 policy 在 H2H vs 031A 强但 vs baseline 弱（baseline 跟 031A 行为 distribution 完全不同）。

**缓解**: 监控 50ep eval 时 vs baseline + vs random 的 WR 变化。如果 vs baseline 退化 ≥ 5pp 提前 abort。

### R2 — Training instability vs strong opponent

031A 是 strong policy，scratch P 早期会**全输**。reward signal 几乎全 -1，PPO advantage 信号微弱，可能不收敛或收敛到 "do nothing"。

**缓解**:
- 046B 用 warmstart (P starts at 031A 水平，不是 -1 起点)
- 046A scratch 才有此风险，需要长 budget 和 entropy regularization
- 加 dense v2 shaping (046A/B 都保留) 提供细粒度 signal

### R3 — Adapter 实施 bug

snapshot-023 设计的 FrozenTeamCheckpointPolicy adapter **从未实施**。需要先写 adapter，包括：
- per-episode opponent sampling (固定 100% frozen_031A)
- frozen policy load + inference (复用 trained_team_ray_agent 加载逻辑)
- 多 worker 序列化兼容 (Ray 1.4 worker process)

预估 ~一天工程。如果 adapter 写不通，046 整 lane 阻塞。

### R4 — vs frontier H2H 不同步退化

P 可能 H2H vs 031A 强但 H2H vs 029B/025b 不一致 (不同 frontier 不一样)。这暗示 P 学到的不是 general skill 而是 031A-specific exploit。

**缓解**: 训练完 H2H vs 多个 frontier 验证 general，不只测 vs 031A。

## 8. 不做的事

- 不在 046 内做 reward 实验（保持 reward 是 v2 唯一变量，与 045 区分）
- 不做 dynamic opponent (训练中加新 frozen ckpt) — 复杂度爆炸
- 不混 specialist (044 失败已 abandon)
- 不一次起 046A/B/C，按 phase gate
- **不交 046 produced policy 作为 grading 提交** (它训练时没见 baseline，可能 mode collapse 风险高)

## 9. 执行清单

### Phase 0: FrozenTeamCheckpointPolicy adapter 实施 (~一天)
1. 写 `cs8803drl/branches/frozen_team_checkpoint.py`，实现 adapter
2. wire 到 `train_ray_team_vs_baseline_shaping.py` 通过 `OPPONENT_POOL_*` env vars 解析
3. smoke test: 1 iter run, 确认 frozen opponent 真的在 inference

### Phase 1: 046B warmstart 快验证 (~3h)
4. 写 batch `soccerstwos_h100_cpu32_team_level_046B_cross_on_031A1040_512x512.batch`
5. 启动 200 iter
6. invoke [`/post-train-eval`](../../.claude/skills/post-train-eval/SKILL.md) (lane 046B, 包括 H2H vs 031A + 029B + 025b)
7. 按 §5 判据决定下一步

### Phase 2: 046A scratch (条件，14h)
8. 仅在 046B 显示 ≥ 0.88 vs baseline 时启动
9. scratch budget 50M steps

### Phase 3: 046C sparse-only (条件，14h)
10. 仅在 046A 突破 + 想进一步移除 v2 prior 时启动

## 6. 实施进度（append-only，2026-04-19）

### Phase 0: FrozenTeamPolicy adapter 实施

**新增 / 修改文件**:

- `cs8803drl/core/frozen_team_policy.py` (新建, ~210 行) — `FrozenTeamPolicy` 适配器：
  per-env 状态机，跟踪 `TeamVsPolicyWrapper.step()` 内每步两次 `opponent_policy(per_agent_obs)` 调用的 parity；用 `concat(obs_2, last_obs_3_from_prev_step)` 拼成 672-dim joint obs 查询 team-level Siamese policy，缓存 `action_3` 给同一步第二次调用；提供 `reset_episode()` 清状态，并通过 `install_reset_hook()` idempotently monkey-patch `TeamVsPolicyWrapper.reset` 串接清理。
  - 关键修复：复用 `cs8803drl.core.utils._get_checkpoint_policy` 会强转 `MultiDiscrete -> Discrete(18)` 导致 policy 返回 single int；改写 `_build_team_policy_from_checkpoint` 直接照 `cs8803drl/deployment/trained_team_ray_agent.py` 的方式，用真正的 `MultiDiscrete([3,3,3,3,3,3])` 注册 dummy env，再 `load_policy_weights` 灌权，确保 `compute_single_action` 返回 6-dim joint action。
- `cs8803drl/core/utils.py` (CAREFUL, +30 行)：
  - 新增 `_find_wrapper(env, cls)` helper 走 `.env` 链找子 wrapper。
  - `create_rllib_env` 解析 `env_config["team_opponent_checkpoint"]`，与 `opponent_mix` 互斥（同时设抛 ValueError）。命中时定位底层 `TeamVsPolicyWrapper` 并 `set_opponent_policy(FrozenTeamPolicy(...))`，per-agent obs/action space 来自 `tvp_wrapper.env`（即未被 join 的 336-dim / 3-dim）。
- `cs8803drl/training/train_ray_team_vs_baseline_shaping.py` (CAREFUL, +12 行)：
  读 `TEAM_OPPONENT_CHECKPOINT` env var；若设置则把 `team_opponent_checkpoint` 注入 `env_config` 并跳过 `opponent_mix={"baseline_prob": ...}`（mutually exclusive），并打印一行 `[snapshot-046] TEAM_OPPONENT_CHECKPOINT set — overriding baseline_prob` 提示。
- `scripts/smoke/smoke_frozen_team_policy.py` (新建, ~230 行) — Phase A plumbing 验证脚本（不跑训练）。
- `scripts/batch/experiments/soccerstwos_h100_cpu32_046B_warm_vs_frozen_031A_512x512.batch` (新建, ~115 行) — 200-iter 训练 batch（**未提交**，留给 user）。

### Phase A — 适配器 plumbing smoke (2026-04-19, GPU node atl1-1-03-011-18-0)

| 项 | 结果 |
|---|---|
| env build with `team_opponent_checkpoint` | PASS (0.6s) |
| `tvp.opponent_policy is FrozenTeamPolicy` | True |
| reset clears `_call_parity` 到 0 | True |
| 50 random-action steps，无 exception | PASS (14.2s, 284 ms/step single env) |
| opponent action global variance | 0.600 (>0，非 constant) |
| per-dim hist | dim0 {0:25,1:59,2:16}, dim1 {0:47,1:23,2:30}, dim2 {0:17,1:38,2:45} |
| MultiDiscrete([3,3,3]) range check | PASS (min=0, max=2 across 100 calls) |
| 自然终止 episode 数 / 平均长度 | 2 / 19.5 (16-23) |
| Mid-test explicit reset 清状态 | PASS (cached_obs_3 全 0, parity=0) |

**发现的 bug**: `_get_checkpoint_policy` 强转 MultiDiscrete -> Discrete(sum(nvec))，team policy 返回 1-dim 而非 6-dim；patch：FrozenTeamPolicy 改用自己的 trainer 构建路径（同 `trained_team_ray_agent.py`）保留 MultiDiscrete。

### Phase B — 5-iter 训练 smoke (2026-04-19, GPU node, H100 venv)

| 项 | 结果 |
|---|---|
| 5 iter 全部完成无 crash | PASS (~3 min wall) |
| iter_rate | 0.10 it/s (~10s/iter on 8K batch with 4 workers x 2 envs) |
| Adapter 时间不灾难 | sample_ms ≈ 9.9-10.1s/iter，learn_ms 213-448ms — 远低于 5 min/iter target |
| `episode_reward_mean` finite | True (+0.537 → +0.906，单调向上) |
| 727 episodes total | OK |
| Warmstart load 成功 | `[warmstart] loaded default_policy from checkpoint: ...checkpoint-1040` |
| best_reward_mean | +0.9058 @ iter 5 |
| GPU 峰值 mem | 任务结束后 0 MiB / 81559 MiB；学习侧 model 极小（Siamese 256x256 + 256x128），无内存压力 |

**注**: Phase B 仅 5 iter 不触发独立 `EVAL_INTERVAL=5` checkpoint eval（first checkpoint 写入与 train 完成同步，subprocess 评估可能未来得及完成）。WR vs baseline ≈ 0.5 的初始预期未独立验证；用 200-iter 主跑的 EVAL_INTERVAL=10 才会持续产生 `checkpoint_eval.csv` 行。

### 启动 200-iter 准备状态

**READY for 200-iter launch**（plumbing 完整、warmstart 正常、reward 流非崩溃）。

- batch 文件: `scripts/batch/experiments/soccerstwos_h100_cpu32_046B_warm_vs_frozen_031A_512x512.batch`
- 推荐 `BASE_PORT` 区间: 56000-58000（与 040/045 系列错开 50 槽位，避免端口冲突）
- 文件已就绪但**未 sbatch 提交**——按 user 决定。

### 仍需注意

1. Phase B 没真正测出 eval-pipeline 端到端（需 ≥ 10 iter），200-iter 主跑前 10 iter 内观察首个 `checkpoint_eval.csv` 行，确认 vs baseline WR 落在合理区间（warmstart 起点 ≈ 0.86，第一行不应远离）。
2. `FrozenTeamPolicy` 1-frame stale obs_3 的近似在 ms 级 soccer dynamics 下可接受，但若 200-iter 训完 H2H vs 031A << 0.5 且行为异常，需 bisect 怀疑这条捷径。
3. 当前 utils.py 旧的 `opponent_mix` 路径有疑似潜在 bug：`tmp_env` 早于 `env_config["opponent_policy"]=...` 创建，赋值后未重建 env，opponent_policy 实际未替换。本次未修，留待独立 ADR。046 路径走 `set_opponent_policy` 已绕开此问题。

## 12. 046D 提前终止 + 046E 启动 (2026-04-19, append-only)

### 12.1 046D 提前终止 (iter 30)

[user 反馈](../management/WORKLOG.md): "046D 是 warmstart, 不是 scratch；真正的 skill transitivity 测试是 scratch 版本"。

- 046D iter 30 数据 (50ep internal eval):
  - iter 10: 0.94 (warmstart-inherited from 031B@1220)
  - iter 20: 0.86
  - iter 30: 0.84
- 已经显示 saturation pattern (跟 046B Siamese saturate 一致)
- ckpt 30 后 kill, 节省 GPU 给 046E

### 12.2 046E 设计

| 项 | 046B | 046D (paused) | **046E** |
|---|---|---|---|
| 架构 | Siamese | cross-attention | **cross-attention** |
| Warmstart | 031A@1040 | 031B@1220 | **scratch (random init)** |
| 训练对手 | frozen 031A | frozen 031B | **frozen 031B@1220** |
| Max iter | 200 | 200 (killed @30) | **800** (scratch 需更多) |
| Timesteps | 8M | 8M | **32M** |

**关键差异**: 046E 是真正的 "scratch + 强对手" skill transitivity 测试。
- 046B/D 都是 warmstart + 强对手, 大概率 saturate at warmstart 起点 (确实如此)
- 046E 从随机权重出发, 唯一对手是 031B@1220 (0.882) — 初始 WR ≈ 0%, 信号极稀疏
- 如果 046E 能学起来 (1000ep peak ≥ 0.85): **强证据 cross-attention 架构 + cross-train 可以 generalize 到 vs baseline**
- 如果 046E fail (1000ep peak < 0.70): 强证据 reward sparsity 主导, cross-train 路径死

### 12.3 046E 预注册判据

| 1000ep peak | verdict |
|---|---|
| ≥ 0.880 | **breakthrough** — scratch + cross-train 可以从 0 学到 SOTA, iteration self-play 启动 |
| ∈ [0.85, 0.88) | strong, 跟 031B 持平 (但 cost 14h GPU) |
| ∈ [0.70, 0.85) | partial 学到，但 below 031B base, 不 valuable |
| < 0.70 | 信号稀疏致命, cross-train 路径**关闭** |

### 12.4 执行

- **batch**: [scripts/batch/experiments/soccerstwos_h100_cpu32_046E_scratch_vs_frozen_031B_cross_attention_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_046E_scratch_vs_frozen_031B_cross_attention_512x512.batch)
- **节点**: atl1-1-03-010-30-0
- **tmux**: `train_046E`
- **PORT_SEED=46** → BASE_PORT 57905
- **预计**: ~6h (800 iter, 复杂度高于 200-iter 046D 因 scratch 收敛慢)

## 11. 046D — 加 cross-attention 架构变种 (2026-04-19，append-only)

### 11.1 motivation

[046B](#3-三条候选-lane) 用 Siamese 架构 (031A) + warmstart 031A + frozen 031A 测 skill transitivity，**200 iter trajectory 0.86 mean，没突破 ceiling**（peak 0.94 @ 10 是 warmstart inheritance, mean 0.85-0.88）。

[snapshot-031 §13](snapshot-031-team-level-native-dual-encoder-attention.md#13-031b-首轮结果-2026-04-19append-only) 已证 cross-attention (031B) 在 baseline-axis 上显著比 Siamese 强 (+2.2pp)。**问题**：cross-attention 是否在 cross-train (vs frozen self) setting 上**也比 Siamese 表现好**？

046D 是这个问题的答案。

### 11.2 设计

| 项 | 046B | **046D** |
|---|---|---|
| 架构 | Siamese (031A 同款) | **cross-attention (031B 同款)** |
| Warmstart | 031A@1040 (1000ep 0.860) | **031B@1220 (1000ep 0.882)** |
| 训练对手 | frozen 031A@1040 | **frozen 031B@1220** |
| Eval | vs baseline | **vs baseline** (同) |
| Iter / budget | 200 / ~3-4h | 200 / ~3-4h |
| Adapter | FrozenTeamPolicy (✓ 046B 验证) | 同 (复用) |

### 11.3 假设

**H_046D**: 在 cross-attention 架构上做 cross-train（vs frozen self），**仍然 saturate 在架构本身的 ceiling**（031B 0.882 ± 0.01）。即「skill transitivity」在 cross-attention 上**也不成立**，跟 Siamese 一样。

**反假设 (low prior)**: cross-attention 在 cross-train 上有 unique 的 lift，1000ep peak 显著突破 0.882 (e.g., ≥ 0.90)。这会推翻 046B 的"saturate"结论，并启动 iteration self-play (031B → 031B' → 031B''...)。

### 11.4 预注册判据

| 阈值 | verdict |
|---|---|
| 1000ep peak ≥ 0.90 | **breakthrough** — cross-attention 有 cross-train 优势，启动 iteration self-play |
| 1000ep peak ∈ [0.882, 0.90) | 弱方向性 +Δ，跟 031B 持平 ± noise，**evidence weak** for cross-train |
| 1000ep peak ∈ [0.86, 0.882) | saturation，跟 046B 一致，**reinforces** "cross-train 在 baseline-axis 没用" |
| 1000ep peak < 0.86 | regression，cross-train **伤害** 031B 起点 |

### 11.5 执行

- **Batch**: [scripts/batch/experiments/soccerstwos_h100_cpu32_046D_warm_vs_frozen_031B_cross_attention_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_046D_warm_vs_frozen_031B_cross_attention_512x512.batch)
- **关键 env**:
  - `WARMSTART_CHECKPOINT = TEAM_OPPONENT_CHECKPOINT = 031B@1220`
  - `TEAM_CROSS_ATTENTION=1, TEAM_CROSS_ATTENTION_TOKENS=4, TEAM_CROSS_ATTENTION_DIM=64` (031B 同款)
  - `TEAM_SIAMESE_ENCODER=1, ENCODER_HIDDENS=256,256, MERGE_HIDDENS=256,128` (同 031B)
- **节点**: atl1-1-03-010-30-0 (load 0.22)
- **tmux**: `train_046D`
- **PORT_SEED=46** → BASE_PORT 57905
- **预计**: ~90 min (per 046B timing on similar config) - 4h depending on 节点 contention

## 10. 相关

- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031A 来源 (frozen opponent)
- [SNAPSHOT-023](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) — adapter 设计
- [SNAPSHOT-043](snapshot-043-frontier-selfplay-pool.md) — frontier league (并行实验，对照 "diversity vs intensity")
- [SNAPSHOT-045](snapshot-045-architecture-x-learned-reward-combo.md) — 同期实验，正交假设 (reward path)
- [rank.md](rank.md) — 数值依据

## 13. 046E verdict — sample-inefficiency confirmed, hypothesis 部分支持 (2026-04-19, append-only)

### 13.1 训练完成

`046E_scratch_vs_frozen_031B_cross_attention_512x512` (10:30 → 16:00, ~5.5h on atl1-1-03-010-30-0):
- 完成 760 iter (out of 800 budget, time limit 触发)
- best_reward_mean: +0.628 @ iter 751 (训练后期 reward 仍在升, 没饱和)
- best_eval_baseline (50ep internal): 0.78 @ ckpt 450
- best_eval_random (50ep internal): 0.90 @ ckpt 450
- run_dir: `ray_results/046E_scratch_vs_frozen_031B_cross_attention_512x512_20260419_095358/`

### 13.2 Baseline 100ep × 5 top ckpts (post-train)

| ckpt | 50ep (training) | 100ep (post) | reading |
|---|---|---|---|
| 450 | 0.78 | 0.650 | 50ep noise overshoot |
| 570 | 0.78 | 0.690 | |
| 600 | 0.76 | 0.740 | |
| **620** | 0.76 | **🏆 0.810** | 真 peak (50ep 没 capture) |
| 680 | 0.76 | 0.790 | |

**100ep peak = 0.810 @ ckpt 620** (n=100 SE ±0.04). 比 031B@1220 baseline (1000ep 0.882) **-7.2pp**, 比 031A@1040 (0.860) **-5pp**.

### 13.3 H2H vs 031B@1220 across training windows (KEY)

| iter | vs 031B WR (n=500) | trajectory reading |
|---|---:|---|
| 50 | **0.148** | 训练初期, 学生输 85% 给训练对手 — sparse signal 极稀少 |
| 150 | 0.294 | 学习中 (+15pp/100iter) |
| 300 | 0.372 | 持续接近 (+8pp/150iter) |
| 450 | 0.382 | 第一平台 |
| 600 | 0.420 | 持续接近 (+4pp/150iter) |
| **750** | **0.476** | 末段, 接近平手但**没超越** opponent |

**单调上升轨迹** (0.148 → 0.476 = +33pp over 750 iter), **没有 over-specialization 收敛**。学生确实在学习, 只是慢。

### 13.4 verdict 修正 — 不是 over-specialization, 是 sample-inefficient

**之前判断错误**: 我看 best_eval_baseline drop 后 (450 peak → 后期下降) 推测「over-specialization to training opp」。100ep 复测显示 ckpt 620 是真 peak (50ep 在 ckpt 450 的 0.78 是 noise spike), 没有真 drop。学生 baseline 实际峰值在 ckpt 620 = 0.810。

**真实问题**: vs 强对手训练, sparse reward signal 极稀少, 学习速度比 vs baseline 训练慢 4-5×:
- 031B (vs baseline 训): ~200 iter 到 0.86 baseline
- 046E (vs 031B 训): 750 iter 到 0.81 baseline + 0.476 vs 031B
- 单纯算 GPU 投入产出比, 046E **明显 dominated** by 031B 同期训练方法

### 13.5 Hypothesis 评估

| 预注册 hypothesis | 验证结果 |
|---|---|
| H_046E: scratch + vs SOTA opp 加快/改善 generalization | **❌ 部分否定** — 没加快 (反而慢 4-5×), generalization 也没显著好 (vs baseline -7pp) |
| H_046E sub: 强 opp 提供更密集 learning signal | **❌ 反向** — 强 opp 反而稀疏 (学生大部分时间输, sparse reward 落空) |

**根本问题**: 当 student 弱于 frozen opp 时, sparse env reward 大部分集中在 "team1 (opp) wins" — student 看不到正 reward, 学不到 winning patterns。**vs 强对手 + sparse env reward = sample-inefficient by construction**.

### 13.6 战略 implication — 046 路径修正方向

**046 系列单一固定强对手 lane 关闭** (046E final), 但路径不死, 需要修:

| 修正方向 | 描述 | 状态 |
|---|---|---|
| **Curriculum**: gradual scale opp strength (baseline → mid → strong) | 经典 fix sample-eff, 但工程复杂 | backlog |
| **Pool**: AlphaStar-lite mixed pool 覆盖 0.5-0.88 spectrum | 用户多次提到 self-play league | backlog (5+ 天 eng) |
| **Dense reward + strong opp** (replace sparse env with V(s) from learned predictor) | **Direction 1.b Option A 配合 vs 强对手** | 候选, 见 §13.7 |

### 13.7 跟 Direction 1.b 的 synergy

snapshot-051/045 path 验证了 reward signal 在 episode-level metric 空间已 saturate。但 Direction 1.b prototype (`docs/experiments/artifacts/v3_dataset/direction_1b/`) 78.8% val acc 验证 **per-step state 真能预测 outcome** — 这个 dense V(s) 信号正好 attack 046E 的 sparse-reward 瓶颈:

- 046E sparse env reward 让 vs 强对手训练慢 4-5×
- Direction 1.b option A: r(s) = logit P(W|s) 给每步 dense feedback
- **组合 (vs 强对手 + dense reward) 可能 ≠ 单独之和**

→ 046 + 1.b option A 是**互补不冗余**的。等 031B-noshape verdict (~10h) 决定 reward shaping 是否仍然必要后, 再决定是否启动这个组合 lane。

### 13.8 Raw recap

```
046E baseline 100ep (top 5 ckpts):
  ckpt-620: WR=0.810  ← peak
  ckpt-680: WR=0.790
  ckpt-600: WR=0.740
  ckpt-570: WR=0.690
  ckpt-450: WR=0.650

046E H2H vs 031B@1220 (n=500):
  iter 50:  WR=0.148
  iter 150: WR=0.294
  iter 300: WR=0.372
  iter 450: WR=0.382
  iter 600: WR=0.420
  iter 750: WR=0.476
```

完整 logs: [docs/experiments/artifacts/official-evals/046E_baseline100.log](../../docs/experiments/artifacts/official-evals/046E_baseline100.log) + [headtohead/046E_iter*_vs_031B_1220.log](../../docs/experiments/artifacts/official-evals/headtohead/)
