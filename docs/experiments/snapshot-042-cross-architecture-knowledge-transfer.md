# SNAPSHOT-042: Cross-Architecture Knowledge Transfer (per-agent → team-level Siamese)

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 预注册 / A3 已激活实现 / A1-A2 仍待实施
- **依赖**:
  - per-agent SOTA: [`029B@190`](../../ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190) (0.868 official 500 / 0.846 1000ep)
  - team-level SOTA: [`031A@1040`](../../ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040) (0.860 1000ep avg, [snapshot-031 §11](snapshot-031-team-level-native-dual-encoder-attention.md#11))

## 0. 动机与已知 gap

031A 和 029B 都到 0.86 平台但**完全不同方法栈**：
- **029B** = per-agent flat MLP `[512, 512]`，单 agent 处理 336-dim obs，行为通过 BC + PPO + PBRS handoff 学到
- **031A** = team-level Siamese `[256,256] + [256,128]`，双 agent 共享 encoder 处理 672-dim obs，scratch 训 1250 iter

两个达到的策略**完全独立**：
- 031A 的 Siamese 架构不知道 029B 学到的"per-agent skill"
- 029B 的 per-agent 网络不知道 031A 架构能利用的"team-level coordination"

[snapshot-031 §11.6](snapshot-031-team-level-native-dual-encoder-attention.md#116) 已经显示：031A H2H vs 029B = 0.552 (`***`)，team-level 架构在 peer-axis 上更强。但 baseline 1000ep 两者打平 (0.860 / 0.846)。

> **本 snapshot 的核心问题**：把 029B 已学到的 per-agent skill 移植到 031A 的 Siamese 架构上，能否得到 **架构优势 + 已学习行为** 双 buff，把 baseline 1000ep 推过 0.88？

## 1. 核心假设

> **031A 的 Siamese 架构是 1000ep 0.86 的 ceiling，不是因为架构本身的容量极限，而是因为 1250 iter scratch 还没探索完它的策略空间。029B 的 v2-shaping policy 已经在 per-agent obs 子空间上充分优化，把这种"已优化的 per-agent decision pattern" 注入 Siamese encoder 作为初始化，可以让 031A 跳过早期探索期，直接收敛到更强的 policy。**

子假设：

- **H1 (encoder transfer 提升 sample efficiency)**: Siamese encoder 用 029B FCNet 初始化 → 早期 reward 收敛比 scratch 快 50%+
- **H2 (encoder transfer 提升 ceiling)**: 用 029B encoder 初始化 + fine-tune 后，1000ep ≥ 0.87 (>scratch 031A 0.860)
- **H3 (transfer 失败)**: 029B 学到的 per-agent feature 不适合 team-level merge MLP 的下游消费 → policy 退化 → 1000ep < 0.85

如果 H2 成立，**架构组合 = 项目首个 0.87+ baseline**，且证明 "已知 per-agent 知识可以注入 team-level 架构"。

## 2. 三条候选实施路径

| 路径 | 核心机制 | 实施复杂度 | 预期成功率 |
|---|---|---|---|
| **A1: Encoder weight transfer** | 029B FCNet 第一两层 weights → 031A 的 `shared_encoder` | 中 | 中 |
| **A2: Behavior cloning distillation** | 用 029B 在 2v2 env 跑 trajectories，BC 训 SiameseTeamModel | 中 | 中-高 |
| **A3: KL distillation co-train** | 训练 031A 时加 KL(student π‖teacher 029B) loss | 高 | 中 |

### 2.1 路径 A1 (encoder weight transfer)

**最简洁的"知识注入"** —— 直接 weight load：

```
029B FCNet (per-agent):
  hidden0:  Linear(336 → 512) + ReLU       ← 处理单 agent 336-dim obs
  hidden1:  Linear(512 → 512) + ReLU       ← 中间表示
  logits:   Linear(512 → 9)                ← 单 agent 9 dim action

031A SiameseTeamTorchModel:
  shared_encoder:
    Linear(336 → 256) + ReLU                ← ⚠️ 维度不匹配 029B 的 512!
    Linear(256 → 256) + ReLU
  merge_mlp:
    Linear(512 → 256) + ReLU
    Linear(256 → 128) + ReLU
  logits_layer: Linear(128 → 18)            ← 双 agent joint 18 dim
  value_layer:  Linear(128 → 1)
```

**两种 reconcile 方式**：

#### A1a: 重做 031A 用 [512, 512] encoder
- 改 batch 用 `TEAM_SIAMESE_ENCODER_HIDDENS=512,512` `TEAM_SIAMESE_MERGE_HIDDENS=512,256`
- 这让 Siamese encoder 维度匹配 029B
- 然后 weight surgery: 029B `hidden0/1` → 031A `shared_encoder.0/2`
- 还需 scratch 训 1250 iter（参数量 ~700K vs 320K，更慢）
- **总成本**: 14h+3h fine-tune

#### A1b: PCA 投影 029B encoder 到 031A 维度
- 029B hidden 是 512-dim feature，031A encoder 是 256-dim
- 用 PCA 把 029B 的 `hidden1` 输出投影到 256 维（保留主 variance）
- 把投影后的 weights 复制到 031A `shared_encoder` 第二层
- **风险**：信息损失，可能不如 scratch
- **成本**: 1-2h surgery + 200 iter fine-tune (~3h)

**推荐 A1a**: 维度匹配最干净，但成本高（一次性）。

### 2.2 路径 A2 (BC distillation)

不去碰 weights，直接用 029B 的**输出**作为监督：

```
1. 用 029B 在 multiagent_player env 跑 2000 episodes：
   - 记录 (joint_obs, joint_action) 对
   - joint_obs = concat(obs_agent_0, obs_agent_1)  (672-dim)
   - joint_action = concat(action_029B_for_agent_0, action_029B_for_agent_1)  (6 discrete dims)

2. BC 训练 SiameseTeamModel:
   - input: joint_obs (672-dim)
   - output: 18-dim logits (双 agent)
   - loss: cross-entropy vs joint_action labels
   - epochs: 50
   - LR: 1e-3 (BC 通常更高)

3. PPO fine-tune:
   - WARMSTART_CHECKPOINT = BC checkpoint
   - v2 shaping 同 031A
   - 200 iter
```

**优点**：
- 不需要 weight 兼容性
- 029B 的"每步决策"被精确蒸馏到 team-level 架构
- 已有 [imitation_bc.py](../../cs8803drl/branches/imitation_bc.py) infra

**缺点**：
- 029B 处理 per-agent obs 时只看自己的 336 维，不知道 teammate；蒸馏到 team-level 时，teammate obs 被网络 implicit 用上，可能让 policy 学到"看 teammate 但不利用"的疏忽 pattern
- BC 数据采集 ~2h GPU

**成本**: 2h 采集 + 1h BC + 3h PPO fine-tune = ~6h

### 2.3 路径 A3 (KL distillation co-train)

最复杂但最 elegant：

```
PPO loss = standard_ppo_loss + α * KL(π_031A(·|joint_obs) ‖ π_029B_factored(·|joint_obs))
```

其中 `π_029B_factored(joint_action | joint_obs) = π_029B(action_0 | obs_0) * π_029B(action_1 | obs_1)`（独立 factored，因为 029B 不知道协作）。

`α` 是 distillation weight，慢慢衰减（前期模仿 029B，后期自由探索）。

**优点**：
- 同时训 PPO RL signal + 029B teacher signal
- 不限制于固定起点，可以慢慢偏离 029B

**缺点**：
- 训练时需要 query 029B inference（每个 batch 几千次 forward）→ 慢 30-50%
- α 调参敏感
- 实现复杂

**成本**: ~250 行新代码 + 14h training

## 3. 推荐执行顺序

按 ROI 排序：

1. **首选: A1a (encoder transfer + retrained 031A)**
   - 实施成本中等（~150 行 weight surgery code + 14h+3h training）
   - 维度匹配最干净
   - 直接测试 "encoder 知识能否提升 team-level 架构"

2. **备选: A2 (BC distillation)**
   - 如果 A1a 失败或想用更弱依赖的方式
   - 已有 BC infra 可复用

3. **当前激活: A3 (KL co-train)**
   - 虽然原始预注册里把 A3 放在后面，但当前实现成本已经压到可接受
   - 先以 `029B@190 -> 031A@1040` 的 factor-wise KL stage-2 版本做一次可运行验证
   - 若结果不理想，再回到 A1a/A2

## 4. 训练超参（路径 A1a 主线）

### Stage 0: 重做 031A 用 [512, 512] encoder (一次性)

```bash
# Same as 031A but encoder/merge dims doubled
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=512,512        # ← changed from 256,256
TEAM_SIAMESE_MERGE_HIDDENS=512,256          # ← changed from 256,128
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048
TIMESTEPS_TOTAL=50000000 MAX_ITERATIONS=1250
USE_REWARD_SHAPING=1  # v2 同 031A
```

跑 14h，得到 `031A_v512`。

### Stage 1: weight surgery

```python
# scripts/tools/transfer_029B_encoder_to_031A_v512.py
import torch
src = torch.load("ray_results/PPO_mappo_029B_.../checkpoint_000190/checkpoint-190", ...)
dst = torch.load("ray_results/031A_v512_.../checkpoint_001250/checkpoint-1250", ...)

# 029B FCNet first two Linear layers
src_w0 = src["state_dict"]["_logits_branch.0.weight"]   # [512, 336]
src_b0 = src["state_dict"]["_logits_branch.0.bias"]
src_w1 = src["state_dict"]["_logits_branch.2.weight"]   # [512, 512]
src_b1 = src["state_dict"]["_logits_branch.2.bias"]

# 031A_v512 shared_encoder (Sequential of Linear+ReLU+Linear+ReLU)
dst["state_dict"]["shared_encoder.0.weight"] = src_w0
dst["state_dict"]["shared_encoder.0.bias"]   = src_b0
dst["state_dict"]["shared_encoder.2.weight"] = src_w1
dst["state_dict"]["shared_encoder.2.bias"]   = src_b1

# merge_mlp / logits / value 保持 031A_v512 自己学到的
torch.save(dst, "ray_results/042_warmstart/checkpoint-from-transfer/checkpoint-0")
```

### Stage 2: PPO fine-tune

```bash
WARMSTART_CHECKPOINT=ray_results/042_warmstart/checkpoint-from-transfer/checkpoint-0
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=512,512
TEAM_SIAMESE_MERGE_HIDDENS=512,256
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
MAX_ITERATIONS=200    # 同 040 budget
USE_REWARD_SHAPING=1  # v2 same as 031A
EVAL_INTERVAL=10
```

200 iter ~3h。

## 5. 预注册判据

### 5.1 主判据（成立门槛）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| Stage 0 (031A_v512 scratch) 1000ep peak | ≥ 0.85 | 大网络 scratch 至少不退化于 [256, 256] 版的 0.86 |
| Stage 2 (encoder transferred) 早期收敛 (iter 50) | reward_mean ≥ +1.5 | encoder transfer 提供有用初始化 (vs scratch 早期 -2.0) |
| Stage 2 1000ep peak | ≥ 0.86 | 至少不退化于 031A 平台 |

### 5.2 突破判据（强成立）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| Stage 2 1000ep peak | ≥ 0.87 | encoder transfer 真带来 +1pp 增益 |
| H2H vs 031A@1040 (orig) | ≥ 0.52 | 比 scratch 31A 真有提升 |
| 收敛速度: reach reward_mean +2.0 by iter | ≤ 50 | encoder transfer 的 sample efficiency 验证 (031A scratch 用了 ~200 iter) |

### 5.3 失败判据

| 条件 | 解读 |
|---|---|
| Stage 0 1000ep < 0.85 | [512, 512] encoder scratch 反而比 [256, 256] 差 → 大网络 overfit |
| Stage 2 1000ep < 0.84 | encoder transfer 让 policy 在 fine-tune 早期 collapse |
| H2H vs 031A < 0.50 | 转移的知识对 team-level 架构没用 |

## 6. 风险

### R1 — 架构维度 mismatch 引入大网络 overfit

把 031A 从 [256, 256] 升级到 [512, 512]，参数量从 ~320K 翻到 ~1M。team-level scratch 在大网络上是否能稳定收敛**未知**——可能 [256, 256] 的紧凑性恰好是 031A 强的原因。

**缓解**: Stage 0 必须先 standalone validate 0.85+ 才进入 Stage 2。如果 Stage 0 失败，回 A1b (PCA projection) 或 A2 (BC)。

### R2 — Encoder feature 不兼容下游 merge

029B 的 encoder 学到的 feature 是为"独立预测单 agent action"优化的。031A 的 merge MLP 期待"两 agent feature 一起做 joint reasoning"。这两种期望可能根本不兼容：029B feature 可能没有"协作信号"。

**缓解**: 如果 Stage 2 fine-tune 不收敛，考虑 A2 (BC) 或 A3 (KL distill)，这两个方式让网络在 joint context 下学习。

### R3 — Per-agent 行为在 team-level env 下的 distribution shift

029B 在 per-agent env (`multiagent_player`) 训练，每个 agent 看 336-dim obs，独立决策。031A 在 team-level env (`team_vs_policy`) 训练，policy 看 672-dim joint obs，输出 joint action。

虽然 obs 维度可以适配（agent 0 的 336-dim 进 encoder），但 reward 信号的 attribution 完全不同：
- 029B: 每个 agent 自己的 reward
- 031A: team-level reward (双 agent 共享)

这意味着 029B encoder 学到的 feature 可能无法在 team-level reward 下产生有效梯度。

**缓解**: 这是真实的 risk，没有缓解方法，靠 Stage 2 fine-tune 强行调整。如果 fine-tune 不够，加大 budget (500-1000 iter)。

## 7. 不做的事

- **不做 cross-attention 转移**: 031B (cross-attention) 的 attention 模块是新的，没有可转移源
- **不混 Stage 2 shaping handoff (snapshot-040)**: 042 是 *architectural transfer*，040 是 *reward Stage 2*。两者正交，不在同一 snapshot 内组合
- **不一次起 A1+A2+A3**: 先 A1a 单独验证

## 8. 执行清单

### Phase 1 (大投资): Stage 0 重做 031A_v512
1. 写 batch `soccerstwos_h100_cpu32_team_level_031A_v512_scratch_v2_512x512.batch`
2. 跑 14h scratch
3. eval pipeline 出 1000ep
4. 判据: 1000ep ≥ 0.85 才进 Phase 2

### Phase 2 (核心实验): A1a encoder transfer
5. 写 `scripts/tools/transfer_029B_encoder_to_031A_v512.py`
6. 验证 weight surgery (1-iter forward smoke + load-back roundtrip)
7. 写 batch `soccerstwos_h100_cpu32_team_level_042A1a_transfer_on_031Av512_512x512.batch`
8. 200 iter fine-tune
9. eval pipeline + H2H vs 031A_v512 / 029B / 025b

### Phase 3 (备选): A2 BC distillation if A1a fails
10. 用 029B 跑 trajectories: `scripts/data/collect_029B_team_trajectories.py`
11. BC 训 SiameseTeamModel: 复用 [imitation_bc.py](../../cs8803drl/branches/imitation_bc.py)
12. PPO fine-tune from BC ckpt

## 11. 042A3 首轮结果 (2026-04-19，append-only)

### 11.1 训练完成

run dir: `ray_results/042A3_team_kl_distill_from_029B_on_031A1040_merged/`

- 设计: 031A@1040 warmstart + KL distill anchor 拉到 029B@190
- final_iteration: 200 (8M steps)
- best_reward_mean: +2.2291 @ iter 85
- final_reward_mean: +2.2100
- best_eval_baseline (50ep internal): 0.920 (46W-4L) @ iter 90
- best_eval_random (50ep internal): 1.000

合并自 174-iter formal trial + 30-iter resume170。

### 11.2 数值健康度

`progress.csv` 204 iter 检查（formal + resume）：

| 检查 | Trial 1 (170 iter) | Trial 2 (30 iter) |
|---|---|---|
| total_loss inf 行 | 0 | 0 |
| policy_loss inf 行 | 0 | 0 |
| 标准 kl inf 行 | 0 | 0 |
| **distill_kl inf 行** | **0** | — |
| max\|total_loss\| | 2.56 | 2.36 |
| max\|kl\| | **2.32** | 2.32 |
| max\|distill_kl\| | 2.32 | — |

**clean，无 inf**。值得注意：标准 kl ≤ 2.32 显著小于 031B 的 10.71——KL distill anchor 把 policy 拽在 029B 周围，policy 不大幅 drift。

### 11.3 1000ep 官方 eval (top 5% + ties + ±1, 5 ckpts, 2026-04-19 ~07:11)

按 [pick_top_ckpts](../../scripts/eval/pick_top_ckpts.py) 选 5 ckpt 跑 1000ep on `atl1-1-03-015-30-0` (port shelf 52005, parallel -j 5, total elapsed 244s)。

| ckpt | 1000ep WR | NW-ML | 内部 50ep | Δ (1000-50) |
|---:|---:|:---:|---:|---:|
| **80** | **🏆 0.863** | **863-137** | 0.86 | +0.003 |
| 90 | 0.846 | 846-154 | 0.92 | -0.074 |
| 100 | 0.862 | 862-138 | 0.76 | +0.102 |
| 180 | 0.857 | 857-143 | 0.84 | +0.017 |
| 190 | 0.855 | 855-145 | 0.92 | -0.065 |

mean = 0.857，极窄 ±0.008。

### 11.4 关键发现

1. **042A3 stable around 0.857, peak 0.863** — KL distill 路径把 029B 的 ~0.86 baseline 稳定继承到 031A，**没破天花板**
2. trajectory **极稳**: 5 ckpt span 范围 [0.846, 0.863] = ±0.008，远比 031B [0.819, 0.882] = ±0.06 紧
3. **vs 031A base 0.860**: max +0.003pp、mean -0.003pp = **统计上持平**。KL distill 没贡献 baseline-axis 增益
4. 内部 50ep vs 1000ep 互相 noise（90 -0.074 / 100 +0.102），再次说明 50ep 不可信
5. 但 **stability 是 ROI**: 042A3 的窄方差降低了 "选错 ckpt" 的风险（031A 的 1000ep 单 run 能差 0.014，031B 24 ckpt 跨度 0.06）

### 11.5 Verdict

**042A3 = 「KL distill 稳住 031A，但未突破 0.86」**：
- 主判据 §5.1 (1000ep ≥ 0.87): **未达** (max 0.863)
- 次判据 §5.2 (跨架构知识迁移成立): 部分支持 — anchor 工作但增量 ≈ 0
- 失败模式 §5.3 (轨迹稳定性): **强成立** — 5 ckpt 范围 ±0.008

策略含义：KL distill 适合用作 **stability regularizer**，不是 SOTA breakthrough lane。如果未来某个 lane volatile 但有 spike 潜力，可以叠加 KL distill 收紧分布。

### 11.6 Raw recap (verification)

```
=== Official Suite Recap (parallel) ===
.../checkpoint_000080/checkpoint-80   vs baseline: win_rate=0.863 (863W-137L-0T)
.../checkpoint_000100/checkpoint-100  vs baseline: win_rate=0.862 (862W-138L-0T)
.../checkpoint_000180/checkpoint-180  vs baseline: win_rate=0.857 (857W-143L-0T)
.../checkpoint_000190/checkpoint-190  vs baseline: win_rate=0.855 (855W-145L-0T)
.../checkpoint_000090/checkpoint-90   vs baseline: win_rate=0.846 (846W-154L-0T)
[suite-parallel] total_elapsed=243.8s tasks=5 parallel=5
```

完整 log: [docs/experiments/artifacts/official-evals/042A3_baseline1000.log](../../docs/experiments/artifacts/official-evals/042A3_baseline1000.log)

### 11.7 与 031B 的对照（关键）

031B (cross-attention scratch) 和 042A3 (KL distill on 031A) 同期完成、同 base 系列，对比：

| | 031B | 042A3 |
|---|---|---|
| 路径 | 架构改进 (cross-attention scratch) | 知识迁移 (KL distill 029B → 031A) |
| 训练时长 | 1250 iter (50M steps) | 200 iter (8M steps) |
| max 1000ep | **0.882** | 0.863 |
| mean 1000ep (top ckpt 区间) | ~0.851 | 0.857 |
| 1000ep range | [0.819, 0.882] = ±0.06 | [0.846, 0.863] = ±0.008 |
| 数值健康 | clean (max\|kl\|=10.71) | clean (max\|kl\|=2.32) |
| breakthrough? | ✅ +2.2pp on max | ❌ 持平 |
| stable? | ❌ volatile spike-revert | ✅ tight band |

**结论**: 架构改进 (031B) 是 SOTA 突破路径；知识迁移 (042A3) 是 stability 路径。两者**正交不冲突**。后续可考虑 **042A3 的 KL distill 配方叠加到 031B 上** 看能否同时拿稳定 + breakthrough（潜在 snapshot-051 候选）。

## 9. 相关

- [SNAPSHOT-029](snapshot-029-post-025b-sota-extension.md) — 029B 来源
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031A 架构
- [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md) — team-level BC infra (A2 可复用)
- [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md) — per-agent BC→PPO bootstrap (类似精神)
- [rank.md](rank.md) — 数值依据
