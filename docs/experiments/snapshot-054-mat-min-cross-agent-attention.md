## SNAPSHOT-054: MAT-min — Cross-AGENT Attention 残差块（架构 step 3 重试）

- **日期**: 2026-04-19
- **负责人**: Self
- **状态**: 训练中 (lane open) — launch 18:50:19 EDT, atl1-1-03-011-18-0 (jobid 5017695), 1250 iter scratch, ETA ~12h

## 0. 背景与定位

[SNAPSHOT-052](snapshot-052-031C-transformer-block-architecture.md) 在 2026-04-19 给出**决定性 REGRESSION**：

| step | 架构 | 1000ep peak | Δ vs prev |
|---|---|---|---|
| 0 | 028A flat MLP | 0.783 | base |
| 1 | 031A Siamese encoder | 0.860 | **+7.7pp** |
| 2 | 031B + within-agent token cross-attention | **0.882** | +2.2pp |
| 3a | 052A: + FFN/LN/residual refinement | 0.800 | **-8.2pp** ⚠️ |
| 3b | 052: + true MHA + merge 收缩 | 0.774 | -2.6pp from 052A |

`031B` 0.882 是当前架构 ceiling。052 的失败教训（[snapshot-052 §7.6](snapshot-052-031C-transformer-block-architecture.md#76-关键-lesson)）：
- LayerNorm 改 PPO gradient flow → R2 实证
- transformer block 在 ~1M params 上 over-parameterized → R1 实证
- "标准 transformer" 在 RL 上不是 free lunch

但**架构 axis 还未走到 ceiling**——只是 "transformer-style refinement" 这一支被否决。本 snapshot 试**完全不同的架构 family**：**cross-AGENT attention**（agent-to-agent，而非 031B 的 within-agent token-to-token）。

## 1. 核心假设

### H_054

> 显式建模 **agent-to-agent** 信息交换（每个 agent 看队友的特征）能突破 `031B` 0.882 的架构 ceiling。**1000ep peak ≥ 0.890** (+0.8pp)；若 ≥ 0.900 直接达 grading 门槛。

### 子假设

- **H_054-a**: 031B 的 within-agent attention 只在每个 agent 的 own observation 上 attend，**没有显式机制让 agent_0 的 policy 看 agent_1 的状态**（merge 在 policy/value head 之前 concat，但 head 之前没有 cross-pollination）。MAT (Multi-Agent Transformer, Wen 2022) 在 SMAC 上证明 explicit pairwise agent interaction 比 implicit 更有效。
- **H_054-b**: cross-AGENT attention 仅 ~98K extra params (+21% vs 031B 0.46M)，**不会触发 R1 over-parameterization**。
- **H_054-c**: **不引入 LayerNorm 或 FFN**（052 失败原因），仅 attention + residual。最小架构改动。
- **H_054-d**: V proj **zero-init**，保证训练初期 cross-agent 输出 = 0 → 模型**初始化等价于 031B**，graceful degrade，避免 cold-start 干扰。

## 2. 设计

### 2.1 架构 spec

继承 031B 全部，**只在 cross-attention 输出后插入一个 cross-agent attention 残差块**。

```
Input: 672-dim joint obs (= concat(agent_0_obs_336, agent_1_obs_336))

Step 1 — Siamese encoder (031B 同款):
    obs_336 → Linear(336→256) → ReLU → Linear(256→256) → ReLU → feat_256 (per agent)

Step 2 — Within-agent token cross-attention (031B 同款):
    feat_256_per_agent → Tokenize (4 tokens × 64 dim each)
    Single-head token attention (hand-written)
    → attended_256 per agent

Step 3 NEW — Cross-AGENT attention residual block:
    agent_stack = stack([attended_0, attended_1])      # (B, 2, 256)
    Q = Linear(256→64)(agent_stack)                    # (B, 2, 64)
    K = Linear(256→64)(agent_stack)                    # (B, 2, 64)
    V = Linear(256→256)(agent_stack)  ← ZERO-INIT      # (B, 2, 256)
    attn_weights = softmax(Q @ K^T / sqrt(64))         # (B, 2, 2)
    cross_agent_out = attn_weights @ V                 # (B, 2, 256)
    final_0 = attended_0 + cross_agent_out[:, 0]       # residual
    final_1 = attended_1 + cross_agent_out[:, 1]       # residual

Step 4 — Merge to joint (031B 同款):
    concat(feat_0, final_0, feat_1, final_1) → 1024 dim
    Linear(1024→256) → ReLU → Linear(256→128) → ReLU → policy/value heads
```

### 2.2 关键设计决策（**严格规避 052 pitfall**）

| 052 做的 | 054 做的 | 理由 |
|---|---|---|
| 加 LayerNorm | **不加** | 052 R2 实证: LayerNorm × PPO 梯度冲突 |
| 加 FFN (256→512→256) | **不加** | 052 R1 实证: 参数膨胀 50% → over-fit |
| 改 merge 拓扑 | **不改** | 维持 1024 dim merge，跟 031B 完全一致 |
| 改 attention 头数 (1→4) | **不改** | head_dim 16 太小，052 验证 |
| FFN 后做激活 | **无 FFN** | — |
| 默认 init | V proj **zero-init** | 训练初期 cross-agent_out=0，模型 = 031B；避免冷启 |

**新增参数预算**: ~98K (Q+K+V projections, dim 256 → 64/64/256)，**+21%** vs 031B 0.462M → 0.560M。这跟 052A 的 +50% 完全不同量级。

### 2.3 训练超参（**100% 继承 031B**）

```bash
# Architecture (新)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64
TEAM_CROSS_AGENT_ATTN=1                # 新 env var
TEAM_CROSS_AGENT_ATTN_DIM=64           # 新 env var

# PPO (跟 031B 一字不变)
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048
ROLLOUT_FRAGMENT_LENGTH=1000
GAMMA=0.99 GAE_LAMBDA=0.95 ENTROPY_COEFF=0.0

# v2 shaping (跟 031B 一字不变)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001
SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01
SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8
SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12
SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Budget (跟 031B 一字不变)
MAX_ITERATIONS=1250
TIMESTEPS_TOTAL=50000000
TIME_TOTAL_S=43200
EVAL_INTERVAL=10
EVAL_EPISODES=50
CHECKPOINT_FREQ=10
```

唯一的不同就是架构。这样 verdict 明确归因到 cross-agent attention 本身。

### 2.4 工程实现（已落地）

- [cs8803drl/branches/team_siamese.py](../../cs8803drl/branches/team_siamese.py): 新增
  - `SiameseCrossAgentAttnTeamTorchModel(SiameseCrossAttentionTeamTorchModel)` — 继承 031B 模型，重写 `forward` 加 cross-agent attention 残差块
  - `TEAM_SIAMESE_CROSS_AGENT_ATTN_MODEL_NAME = "team_siamese_cross_agent_attn_model"`
  - `register_team_siamese_cross_agent_attn_model()`
- [cs8803drl/training/train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py): 添加 env var 解析 (`TEAM_CROSS_AGENT_ATTN`, `TEAM_CROSS_AGENT_ATTN_DIM`) 与 model wiring 分支（**位于 `team_cross_attention` 分支之前**，确保 cross-agent 优先）
- [scripts/eval/_launch_054_mat_min_scratch.sh](../../scripts/eval/_launch_054_mat_min_scratch.sh): 新增 launch script，`PORT_SEED=31` (BASE 57155, 远离 053A=23/051D=51/031B-noshape=13)

### 2.5 Smoke 验证（已通过）

- 模型 import OK
- 总参数 0.560M（031B baseline 0.462M, **+98K = +21%**）
- forward + backward shape 正确
- 初始化时 `cross_agent_residual_norm = 0.0` 确认（V zero-init 生效，等价 031B）

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 主: 1000ep peak ≥ 0.890 | +0.8pp vs 031B | **MAT-min 成立**, 突破架构 ceiling |
| §3.2 突破: 1000ep peak ≥ 0.900 | +1.8pp vs 031B | **直接达到 grading 门槛**, 项目 declare success |
| §3.3 持平: 1000ep peak ∈ [0.875, 0.890) | -0.7pp~+0.8pp | cross-agent attention **没用**, 架构 ceiling 确认在 031B |
| §3.4 退化: 1000ep peak < 0.870 | < 031A base | cross-agent **伤害** (尽管 zero-init 应防止), 架构 axis 关闭 |

**和 052 的关键差异**: 052 是「往 031B 上加东西，结果加错了」；054 是「换一种加法，看是不是机制本身的问题」。如果 054 也退化，说明**架构 step 3 整个方向都死**（不是 transformer 特定问题），031B 0.882 就是真 ceiling。

## 4. 风险

### R1 — Cross-agent attention 的 capacity 太小，没增益

只 2 个 token (2 个 agents)，attention weights 是 2×2 矩阵，softmax 后差异有限。**缓解**: 即使如此，V projection 是 256-dim 的非线性变换，capacity 应当够；只有当 cross-agent 信息**根本不重要**时才会无增益。

### R2 — Zero-init 可能让 cross-agent 一直保持 ≈0

V proj 从 zero 开始，gradient 来自 PPO loss 反向传播。如果 PPO 没有显著的 advantage 信号 push V，cross-agent 残差可能保持小，等价 031B。**缓解**: 这其实是**安全特性**——最坏情况退化到 031B（0.882），不会比 031B 差太多。如果训练后 `cross_agent_residual_norm` 仍 ≈ 0，说明 cross-agent 信息确实不被需要，这本身是有价值的科学结论。

### R3 — Diminishing returns 在 step 3 已经反转

052 step 3 是 -8 ~ -11pp。即使 054 是 +0.5pp，它也不会突破 0.90。**缓解**: 接受这个 risk；本 snapshot 的核心价值不是「破 0.90」，而是「确认架构 ceiling 在哪」。如果 054 ≈ 0.882，就可以合理 conclude **架构 axis ≤ 0.882**，转 reward / curriculum / self-play 等其他突破路径。

### R4 — Cross-agent attention 训练动力学不稳

之前 031B `max|kl|=10.71` 已偏高，加 cross-agent 后可能更高。**缓解**: 跟 031B 同 PPO 配置 (CLIP_PARAM=0.15)，看 KL trajectory；若 KL 漂移到 ≥30 提前 stop。

## 5. 不做的事

- **不加 LayerNorm** — 052 实证 R2
- **不加 FFN** — 052 实证 R1
- **不改 merge 拓扑** — 维持 031B 一致性
- **不改 attention head 数** — head_dim 16 太小已证
- **不混入 reward shaping 改动** — 跟 031B 一字不变
- **不在没 smoke pass 之前 launch 12h 训练** — 已 smoke pass
- **不与 053A / 051D / 031B-noshape 共享节点** — PORT_SEED=31 隔离

## 6. 执行清单

- [x] 1. 实现 `SiameseCrossAgentAttnTeamTorchModel` class（~2h）
- [x] 2. 注册 model + 添加 env var 路径（~1h）
- [x] 3. Smoke test（forward/backward/zero-init）（~30 min）
- [x] 4. 写 launch script（PORT_SEED 隔离）（~30 min）
- [x] 5. Pre-launch health check（4 free node 候选）
- [x] 6. **Launch 1250 iter scratch on 5017695 / atl1-1-03-011-18-0** (18:50:19 EDT)
- [ ] 7. 实时 monitor: 第一 iter 完成 / KL trajectory 正常
- [ ] 8. 训完 invoke `/post-train-eval` lane name `054`（top 5%+ties+±1）
- [ ] 9. Stage 2 capture peak ckpt（500ep, v2 桶分析）
- [ ] 10. Stage 3 H2H：only if Stage 1 ≥ 0.880（vs 031B@1220）
- [ ] 11. Verdict append 到本 snapshot §7

## 7. Verdict（待 1000ep eval 后填入, append-only）

_Pending — ETA 训练完成 ~2026-04-20 06:50 EDT, eval ~2026-04-20 07:30 EDT_

## 8. 后续发展线（054 outcome → 下一步路径图）

054 的 verdict 决定下一步动作。三种 outcome 三种动作：

### 8.1 Outcome A — 突破 (054 1000ep peak ≥ 0.890)

cross-agent attention 机制有效。直接做 HPO 收敛而不是再加结构：

- **054-HPO**: 在 054 同架构上扫 `TEAM_CROSS_AGENT_ATTN_DIM ∈ {32, 64, 128}` × `LR ∈ {5e-5, 1e-4, 2e-4}`，找最佳点
- 不再加深结构，让架构 cap 跑满 → 转 reward / curriculum / self-play 突破 0.90 grading 门槛

### 8.2 Outcome B — 一般 (054 1000ep peak ∈ [0.875, 0.890), **本节点重心**)

cross-agent **机制方向对**，但 single-block 表达力不够。**深化路径** Tier 1 → Tier 3，按 ROI 排序：

#### Tier 1 — 简单深化（同 family，工程 ~2-4h，1 个新 model class）

- **054B — Stacked Cross-Agent (2-round)**: 把 054 的 cross-agent block 堆 2 层（仍**无 LN/FFN**，只 attention + residual）
  - 理论支撑: **TarMAC (Das et al. 2019)** 在 SMAC 上证明 multi-round agent message passing > single-round；2 轮 = "每个 agent 看队友对自己的回应"，**meta-cognitive interaction**
  - 参数预算: 2 × 98K = 196K extra (+42% vs 031B)，仍远低于 052A 的 +50% FFN 膨胀
  - 设计: block_2 输入是 block_1 的输出，V proj 同样 zero-init for graceful degrade
  - 风险: 2 层都 zero-init → cold-start 信号更弱，可能学不动 → fallback 让 block_2 V proj 用 small-init (`std=1e-2`)

- **054C — Multi-Head Cross-Agent (2 heads × 32 dim)**: 054 single-head 改 2 heads, head_dim 32
  - 跟 052 的 4-head 失败差异: head_dim 32 vs 16 (32 是 transformer 最小常用 head_dim)
  - 参数预算: 跟 054 同 (Q/K/V proj 总维度不变, 只 split head)
  - 设计: pytorch `nn.MultiheadAttention` with `embed_dim=64, num_heads=2`，V proj 仍 zero-init

#### Tier 2 — 中度深化（引入新 feature, 工程 ~6-8h）

- **054D — Cross-Agent + Edge Features (relative-position-aware)**: K/V projections 不仅来自 agent feature, 还 concat 一个**relational token** = `[teammate_pos - self_pos, teammate_vel - self_vel, ball_pos - midpoint]`
  - 理论支撑: **GAT (Veličković et al. 2018)** 的 edge-aware attention; **MAGAT (Li et al. 2020)** 在 multi-agent navigation 上验证
  - 直觉: 单纯 feature attention 可能让 agent_0 "知道队友的特征向量"但**不知道空间关系**; explicit edge feature 把 relational geometry 显式喂给 attention
  - 参数预算: K/V proj 输入 dim 256+~12 (edge feat) → ~3K extra params

- **054E — Parallel within-agent + cross-agent fork-merge**: 把 054 的 sequential (within-agent → cross-agent) 改成 parallel:
  - branch_a = within_agent_attn(feat)
  - branch_b = cross_agent_attn(feat)  ← directly on encoder output, not on attn output
  - final = feat + branch_a + branch_b
  - 直觉: sequential 让 cross-agent 只看 "已经被 within-agent 过滤的信号"; parallel 让两条 attention 看原始 encoder feature, 信息更原始
  - 参数预算: 跟 054 同 (cross-agent 仍单 block)

#### Tier 3 — 跨 family（新机制，工程 ~8-12h）

- **054F — GAT-style with leaky_relu (替代 softmax)**: GATv2 (Brody et al. 2022) 用 `LeakyReLU(W·[Q;K])` 而非 dot-product 算 attention coefficient
  - 理论支撑: GATv2 证明 dynamic vs static attention 的差异; 在 graph tasks 上稳定优于 GAT v1
  - 直觉: 2 个 agent 时, softmax(Q·K) 容易 saturate 到 [0.9, 0.1] 极端分布; LeakyReLU 给更平滑的 weight
  - 参数预算: 跟 054 同

- **054G — Recurrent Cross-Agent (small GRU at pair level)**: 把 cross-agent 残差换成一个小 GRU cell, hidden_dim 64
  - 直觉: GRU 维护 "agent-pair history" hidden state, 跨 step 累积 cooperation pattern
  - 风险: PPO + recurrent layer 训练不稳, RLLib v1.4 LSTM/GRU pipeline 历史上有 bug
  - 参数预算: GRU cell ~50K params

### 8.3 Outcome C — 退化 (054 1000ep peak < 0.870)

V zero-init 应该已防止退化，所以这种结果**反过来很 informative**:
- 说明 cross-agent 信号在小 budget (1250 iter) 下确实学到**有害的方向**
- **架构 axis 整个方向死** (transformer-style refinement 死, cross-agent 死)
- 不再加深，转其他 axis (reward / curriculum / self-play / cross-train)
- 本 snapshot 即终点，**不再开 054B/C/D**

### 8.4 深化 stop 条件（防止陷入架构 axis）

- 任何 054X (X ∈ {B,C,D,E,F,G}) ≥ 0.890 → **去 9.1 HPO** 路径，不再加新 X
- 054B + 054C + 054D 三连 ≤ 0.882 (031B baseline 平) → **架构 axis ceiling 确认在 0.882**, 关闭整个 054 lane
- 任何变体在中段（500 iter）KL ≥ 30 或 entropy collapse → 提前停, 不浪费 GPU
- 总 budget: 最多再开 3 个 054X 变体 (== 36h GPU); 如果 3 个都失败, 强制收敛

### 8.5 优先级排序（如 054 = mediocre, 决策树）

按 "ROI = 预期 Δ vs 工程成本" 排:

| 优先级 | 变体 | 预期 Δ | 工程 | ROI | 启动条件 |
|---|---|---|---|---|---|
| 1 | 054B (Stacked) | +1~2pp | 2h | high | 054 ∈ [0.880, 0.890) (有正向苗头) |
| 2 | 054C (Multi-Head) | +0.5~1pp | 3h | mid | 054 ∈ [0.875, 0.890) |
| 3 | 054D (Edge Features) | +1~3pp | 6h | mid | 054B/C 没用; 直觉是 spatial relation 是 missing piece |
| 4 | 054E (Parallel) | ±0.5pp | 4h | low-mid | 054B/C/D 都没用; 长尾 |
| 5 | 054F (GATv2) | ±1pp | 6h | low | 想换 attention family 试试 |
| 6 | 054G (Recurrent) | ±2pp | 10h | low | 最后选项, RLLib v1.4 GRU 不稳 |

**默认动作**: 054 verdict 出来后, 如果 mediocre, 立即开 **054B (Stacked)** ——它是最自然的"加深"方向, 工程最便宜, 跟 054 同 family 比较干净。

## 9. 相关

- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) §12 — 031B cross-attention 设计来源
- [SNAPSHOT-052 §7](snapshot-052-031C-transformer-block-architecture.md#7-verdict) — 架构 step 3 失败教训（refinement block 路径）
- [SNAPSHOT-053](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md) — 并行 reward 突破路径
- [team_siamese.py](../../cs8803drl/branches/team_siamese.py) — 已修改
- [train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py) — 已修改
- [_launch_054_mat_min_scratch.sh](../../scripts/eval/_launch_054_mat_min_scratch.sh) — launch script

### 理论支撑

- **MAT (Multi-Agent Transformer, Wen et al. 2022)** — 在 SMAC 上证明 explicit cross-agent attention（agent-to-agent）显著优于 implicit (concat) merge
- **ATOC (Jiang & Lu 2018), AC-Atten, TarMAC** — communicative agent attention 系列；都强调 agent-pairwise interaction 是关键
- **本实现是 MAT-min**: 只取 cross-agent attention 一个核心，**剥掉 transformer block 其他组件**（基于 052 失败教训）
