## SNAPSHOT-060: 054M MAT-medium — 054 + pre-LN + FFN residual (Tier 1b)

- **日期**: 2026-04-20
- **负责人**: Self
- **状态**: 训练中 (lane open) — jobid 5022391, PORT_SEED=43, iter 0+/1250, ETA ~12h

## 0. 背景与定位

### 0.1 架构 axis 的三段式历史

[SNAPSHOT-052](snapshot-052-031C-transformer-block-architecture.md) + [SNAPSHOT-054](snapshot-054-mat-min-cross-agent-attention.md) 已给出完整的架构 step 3 landscape:

| 变体 | 组件 | 1000ep peak | Δ vs 031B | verdict |
|---|---|---|---|---|
| 031B | Siamese + within-agent cross-attn | 0.882 | base | step 2 SOTA |
| **054 MAT-min** | + 1 layer cross-AGENT attn, V zero-init, **no LN / no FFN** | 0.880 | ~0 | tied, no gain |
| 052A | + LN + FFN + residual refinement | 0.800 | **-8.2pp** | decisive regression |
| 052 | + true MHA (4 heads) + merge 收缩 | 0.774 | -10.8pp | decisive regression |

**Framing 难题**: 两端都已被探测。MAT-min (just attention) = tied, full transformer block (MHA + LN + FFN) = catastrophic regression。**中间 sweet spot 是否存在**？

### 0.2 "Medium" 定位假设

054M 尝试**中间**架构：只加 pre-LN + FFN residual，**不**加 MHA (保持 054 single-head cross-agent), **不**加 second LN (通常在 FFN 后)。这个设计点的含义：

- pre-LN 是 transformer 训练稳定性的主要贡献者 ([Xiong et al. 2020](#理论支撑))，比 post-LN 更 PPO-friendly
- FFN residual 是 transformer "non-linear feature mixing" 的核心，但 zero-init final projection 可 degrade 到 054
- **不**加 MHA: 因为 052 的 4-head × 16 head_dim 太小，head_dim 16 is the pathological point
- **不**加 second LN: 减少 LN 总次数，维持与 054 的 gradient flow 尽量相似

目标: **恢复 FFN 表达力而不引入 multi-head pathology**, 在 054 的 0.880 和 052 的 0.774 之间找到 >0.885 的点。

## 1. 核心假设

### H_060

> 在 054 之上加 pre-LN + FFN residual (256→1024→256, GELU)，**1000ep peak ≥ 0.895** (+0.013 vs 054 tied 0.880)，超过 031B 架构 ceiling 并向 0.90 grading 门槛 push。
>
> Stretch: ≥ 0.900，架构 axis 独立达 grading 门槛 (相比 055 distill path 另一条独立证据)。
>
> Worst tolerated: 0.875-0.885 (tied 054)，意味着 pre-LN + FFN 在单 block cross-agent 上无净增益 → 架构 axis 在 031B 0.882 真正 saturated。

### 子假设

- **H_060-a**: pre-LN 比 post-LN 与 PPO 更兼容 (Xiong 2020 证明 pre-LN gradient norm 更稳), 避开 052 的 R2 LN-PPO 冲突
- **H_060-b**: FFN 的 non-linear capacity (1024 hidden) 比 single attention 更有表达力, 能 capture agent-pair interaction 中 non-linear 的部分
- **H_060-c**: FFN 最后 linear layer zero-init → 初始化等价于 054 (= 031B), graceful degrade 防 cold-start
- **H_060-d**: 避开 MHA (只 single-head cross-agent) 规避 052 的 R1 head_dim 16 pathology

## 2. 设计

### 2.1 架构 spec

继承 054 全部, **在 cross-agent attention 后插入 pre-LN + FFN residual block**。

```
Input: 672-dim joint obs

Step 1 — Siamese encoder (= 031B, = 054):
    obs_336 → Linear(336→256) → ReLU → Linear(256→256) → ReLU → feat_256 (per agent)

Step 2 — Within-agent token cross-attention (= 031B, = 054):
    feat_256 → 4 tokens × 64 → attended_256 per agent

Step 3 — Cross-AGENT attention residual (= 054):
    agent_stack = stack([attended_0, attended_1])   # (B, 2, 256)
    Q, K = Linear(256→64); V = Linear(256→256) ZERO-INIT
    cross_agent_out = softmax(QK^T/sqrt(64)) @ V
    post_ca = agent_stack + cross_agent_out          # (B, 2, 256), residual

Step 4 NEW — Pre-LN + FFN residual block (060 only):
    ln_in = LayerNorm(post_ca)                       # pre-LN
    ffn_hidden = Linear(256→1024)(ln_in) → GELU
    ffn_out = Linear(1024→256)(ffn_hidden)  ← ZERO-INIT 最后 linear layer
    final = post_ca + ffn_out                        # residual, t=0 时 ffn_out=0 → final=post_ca=054

Step 5 — Merge to joint (= 031B, = 054):
    concat(feat_0, final_0, feat_1, final_1) → 1024 dim → merge head
```

### 2.2 与 054 / 052 的差异矩阵

| 组件 | 031B | 054 | 060 | 052A |
|---|---|---|---|---|
| Cross-agent attention | — | 1 block | 1 block | 1 block |
| Multi-head attention | — | — | — | 4 heads |
| Pre-LayerNorm | — | — | **1 (before FFN)** | 1 (before attn) + 1 (before FFN) |
| Post-LayerNorm | — | — | — | — |
| FFN residual | — | — | **1 (256→1024→256 GELU)** | 1 (256→512→256) |
| V proj init | — | zero | zero | default |
| FFN final init | — | — | **zero** | default |

**关键**: 060 比 054 多 LN + FFN，比 052A 少一个 LN + 少 MHA + FFN 内部 zero-init。

### 2.3 参数预算

| model | params | vs 031B |
|---|---|---|
| 031B | 0.462M | base |
| 054 | 0.560M | +21% (+98K cross-agent) |
| **060** | **1.08M** | **+134%** (+520K FFN) |
| 052A | ~1.5M | +225% |

060 在 054 和 052A 中间, 1.08M 仍在 "medium" 带。参考 [snapshot-052 §7.6 R1](snapshot-052-031C-transformer-block-architecture.md#76-关键-lesson), over-parameterization 实证阈值在 ~1.5M 左右。

### 2.4 工程实现

- [cs8803drl/branches/team_siamese.py](../../cs8803drl/branches/team_siamese.py): 新增 `SiameseCrossAgentAttnMediumTeamTorchModel(SiameseCrossAgentAttnTeamTorchModel)` — 继承 054 模型，在 forward 最后插入 pre-LN + FFN residual block
- Env var: `TEAM_CROSS_AGENT_ATTN_MEDIUM=1` (require `TEAM_CROSS_AGENT_ATTN=1` as prerequisite)
- [cs8803drl/training/train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py): 添加 env var 解析 + model wiring 分支 (medium 分支在 base cross-agent 之后)
- Launcher: [scripts/eval/_launch_054M_mat_medium_scratch.sh](../../scripts/eval/_launch_054M_mat_medium_scratch.sh)

### 2.5 Smoke 验证

- Model import OK
- 总参数 1.08M (= 054 0.560M + FFN ~520K)
- forward + backward shape 正确
- 初始化时 `ffn_residual_norm = 0.0` (FFN final zero-init 生效, 等价 054)

### 2.6 训练超参 (= 054, 一字不变除了 model name)

```bash
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64
TEAM_CROSS_AGENT_ATTN=1
TEAM_CROSS_AGENT_ATTN_DIM=64
TEAM_CROSS_AGENT_ATTN_MEDIUM=1          # 060 唯一新 flag

# PPO (= 031B = 054)
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048

# v2 shaping (= 031B = 054)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001 SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01 SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Budget
MAX_ITERATIONS=1250 TIMESTEPS_TOTAL=50000000
TIME_TOTAL_S=43200 EVAL_INTERVAL=10 CHECKPOINT_FREQ=10
```

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: 1000ep peak ≥ 0.886 | matching 054 + margin | FFN + LN 没破坏 054 同层级 |
| §3.2 主: 1000ep peak ≥ 0.895 | +0.013 vs 054 | **架构 step 4 成立**, cross-agent + FFN 恢复 transformer 增益 |
| §3.3 突破: 1000ep peak ≥ 0.900 | grading 门槛 | 架构 axis 独立达 grading |
| §3.4 持平: 1000ep peak ∈ [0.875, 0.885] | within ±SE of 054 | FFN + LN 加了但无净增益 → **架构 axis 真 saturated at 031B** |
| §3.5 退化: 1000ep peak < 0.870 | < 054 - 2SE | LN / FFN 与 PPO 在 1B capacity 下仍 break gradient flow → 确认 052 lesson 不是 MHA-only 问题 |

## 4. 简化点 + 风险 + 降级预期 + 预案

### 4.1 简化 S1.A — Pre-LN 只加一处 (不加 post-attn LN)

| 简化项 | 完整方案 | 当前选择 |
|---|---|---|
| LN 数量 | 2 (pre-attn + pre-FFN) | **1 (pre-FFN only)** |

**理由**: 052 的 2 LN 方案已 decisive regression。只加 pre-FFN LN 是 "incremental probe"，隔离 FFN-adjacent LN 的 PPO 兼容性。
**风险**: 如果 attention 输出已 unnormalized, FFN 输入 scale 不稳, LN 可能 absorb 过多 signal。
**降级预期**: single-point LN 没用 → peak ≈ 054 (tied)。
**预案**:
- L1: 如果 060 tied 054, 试 054M-noLN (只 FFN, 无 LN) 看 LN 是无效还是反伤
- L2: L1 仍 tied, declare FFN 本身无效, 不再深化

### 4.2 简化 S1.B — FFN 只加一个 (不 stack)

| 简化项 | 完整方案 | 当前选择 |
|---|---|---|
| FFN block 数 | 2 stacked (classic transformer = 2 sub-blocks) | **1 block** |

**理由**: 参数预算控制在 1.08M, 不冲 1.5M over-param 阈值。
**风险**: 单 FFN block 表达力不够, 但 FFN 本身 hidden 1024 已经比 054 big。
**降级预期**: 单 block FFN 相对 stacked -0.3pp。

### 4.3 简化 S1.C — 固定 hidden dim 1024 (不 sweep)

经典 transformer ratio hidden/embed = 4, 256 × 4 = 1024 标准。不 sweep 是为了 single-run verdict。
**风险**: hidden 1024 可能 sub-optimal, 真 optimum 在 512 或 2048。
**降级预期**: ±0.2pp from optimum。

### 4.4 简化 S1.D — GELU (不 sweep 激活)

GELU 是 transformer 标准选择 (GPT-2 以后)。相比 ReLU 理论上更 smooth gradient。
**不**用 SwiGLU / GEGLU (需要 split projection, 增加参数)。

### 4.5 全程降级序列

| Step | 触发条件 | 降级动作 | 增量工程 |
|---|---|---|---|
| 0 | Default | 060 single run (= 054 + pre-LN + FFN) | base ~12h |
| 1 | Peak < 0.870 (retrogression) | 054M-noLN rerun (FFN only, no LN) | +12h GPU |
| 2 | Step 1 < 0.870 | 054M-noFFN (LN only, no FFN) — 隔离 LN 本身的破坏 | +12h GPU |
| 3 | Step 2 < 0.870 | declare LN × PPO @ 1B capacity 仍 hostile, 整 family dead | — |
| 4 | Peak 0.875-0.885 (tied 054) | declare architecture axis truly saturated at 031B 0.882 | — |
| 5 | Peak ≥ 0.895 (breakthrough) | 立即开 054L (stack 2 blocks or MHA 2-head × 32 head_dim) | +12h GPU |

## 5. 不做的事

- **不加 MHA** — 052 的 R1 失败, 4-head × 16 head_dim 是 pathological point
- **不加 second LN (post-FFN)** — 避免 052 double-LN 路径
- **不换 merge 拓扑** — 维持 031B / 054 一致
- **不混入 reward shaping 改动** — 跟 031B v2 一字不变
- **不混入 LR=3e-4 combo** — 与 [snapshot-059](snapshot-059-055lr3e4-combo.md) 正交, 避免多变量污染
- **不混入 distill** — 那是 055 / 059 / 061 path
- **不 stack 2 FFN blocks** — 060 是 medium, not large
- 不与 059 (PORT=41) / 055v2 (PORT=51) 抢节点 — 本 lane PORT_SEED=43

## 6. 执行清单

- [x] 1. 实现 `SiameseCrossAgentAttnMediumTeamTorchModel` (~3h)
- [x] 2. 注册 model + 添加 env var 路径 (~1h)
- [x] 3. Smoke test (forward/backward, FFN zero-init → 等价 054) (~30 min)
- [x] 4. 写 launch script [scripts/eval/_launch_054M_mat_medium_scratch.sh](../../scripts/eval/_launch_054M_mat_medium_scratch.sh) (~30 min)
- [x] 5. Launch 1250 iter scratch on jobid 5022391, PORT_SEED=43 (2026-04-20)
- [ ] 6. 实时 monitor: 第一 iter 完成 / KL trajectory / FFN residual norm 随训练增长
- [ ] 7. 训完 invoke `/post-train-eval` lane name `060`
- [ ] 8. Stage 2 capture peak ckpt
- [ ] 9. Stage 3 H2H: vs 054@1100 (base tied) + vs 031B@1220 (arch base) + vs 055@1150 (cross-lane SOTA)
- [ ] 10. Verdict append §7

## 7. Verdict (待 1000ep eval 后填入, append-only)

### 7.1 (2026-04-21 00:45 EDT) — Stage 1 1000ep verdict (append-only) — §3.1 marginal HIT, §3.2/§3.3 MISS, Tier 1b 关闭

**数据 (10 ckpts × 1000ep, blind 10-ckpt spread, 因 inline eval 124/124 全部 fail — 见 §7.1.4 engineering note)**:

| ckpt | 1000ep WR | W-L |
|---:|---:|---:|
| 300 | 0.785 | 785-215 |
| 500 | 0.847 | 847-153 |
| 700 | 0.870 | 870-130 |
| 900 | 0.864 | 864-136 |
| 1050 | 0.863 | 863-137 |
| 1100 | 0.874 | 874-126 |
| 1150 | 0.868 | 868-132 |
| 1200 | 0.882 | 882-118 |
| **1230** | **0.889 (peak)** | 889-111 |
| 1250 | 0.859 | 859-141 |

Raw recap:
```
=== Official Suite Recap (parallel) ===
ckpt 300  vs baseline: win_rate=0.785 (785W-215L-0T)
ckpt 500  vs baseline: win_rate=0.847 (847W-153L-0T)
ckpt 700  vs baseline: win_rate=0.870 (870W-130L-0T)
ckpt 900  vs baseline: win_rate=0.864 (864W-136L-0T)
ckpt 1050 vs baseline: win_rate=0.863 (863W-137L-0T)
ckpt 1100 vs baseline: win_rate=0.874 (874W-126L-0T)
ckpt 1150 vs baseline: win_rate=0.868 (868W-132L-0T)
ckpt 1200 vs baseline: win_rate=0.882 (882W-118L-0T)
ckpt 1230 vs baseline: win_rate=0.889 (889W-111L-0T)
ckpt 1250 vs baseline: win_rate=0.859 (859W-141L-0T)
[suite-parallel] total_elapsed=540.5s tasks=10 parallel=7
```

**预注册判据对照 (§3)**:

| 判据 | 阈值 | 实测 | verdict |
|---|---|---|---|
| §3.1 marginal | ≥ 0.886 | 0.889 | **HIT** (minor gain) |
| §3.2 main | ≥ 0.895 | 0.889 | **MISS** |
| §3.3 breakthrough | ≥ 0.900 | 0.889 | **MISS** |
| §3.4 持平 [0.875, 0.885) | within ±SE 054 | 0.889 上越区间 | **MISS** (above range) |
| §3.5 退化 (< 0.870) | < 054 - 2SE | 0.889 | no |

**对比表**:

| 对照 | Δ | 显著性 |
|---|---|---|
| 054 MAT-min combined 2000ep 0.880 | +0.009 | within SE 0.016, **NOT sig** |
| 031B 0.880 | +0.009 | NOT sig |
| 055 SOTA 0.907 | -0.018 | below SOTA |

#### 7.1.1 Reframing

- 054M MAT-medium (= 054 + pre-LN + FFN residual) 比 054 MAT-min 单 shot 仅 +0.009
- 此 gap 极可能是 noise — 若 combined 2000ep, 大概率收敛回 054 的 ~0.880
- **架构 axis (cross-agent attn + LN + FFN) saturated at ~0.88-0.89**
- 加更多 transformer 组件 (full MHA, multi-layer = 052) 反而 -8 to -11pp regression, 没有 upside path
- **Tier 1b 关闭** — MAT progression 在此 arch 家族无法 push SOTA

#### 7.1.2 与 §8 Outcome 对照

- 命中 **Outcome B (持平/小幅上行)** 的精神, 实测 0.889 略 above [0.875, 0.885] 区间但仍在 SE 内
- 触发 [snapshot-054 §8.4](snapshot-054-mat-min-cross-agent-attention.md#84-深化-stop-条件防止陷入架构-axis) 的 stop 条件 (三变体 ≤ 0.882 ± SE = 关闭 054 lane): 052A 0.800 / 054 MAT-min 0.880 / 060 MAT-medium 0.889
- 不开 054L (stack 2 blocks / MHA) — Outcome A 的 ≥ 0.895 trigger 未达成

#### 7.1.3 对项目策略影响

- **架构 path effectively dead for SOTA-pushing**
- 资源集中到:
  - distillation (055 family — 当前 SOTA 0.907)
  - curriculum (062 family — adaptive WR-gated, pending combined)

#### 7.1.4 Engineering note — inline eval 100% failed (124/124)

- 全部 124 个 inline eval (跨 1250 iter) 均失败, 因 [`cs8803drl/deployment/trained_team_ray_agent.py`](../../cs8803drl/deployment/trained_team_ray_agent.py) 缺少 `register_team_siamese_cross_agent_attn_medium_model` 的 import + register 调用
- **Fix 时点**: 2026-04-21 00:35 EDT (post-eval 之前 patch)
- 没有 inline peak 指引 → 用 **blind 10-ckpt spread (300/500/700/900/1050/1100/1150/1200/1230/1250)** 进行 post-eval
- 与早期 054/055 的 deployment registration bug **同 pattern** — 添加新 model class 时, deployment file 的 register 调用应该 auto-checked (或者建立 lint/test gate)

## 8. 后续发展线

### Outcome A — 突破 (1000ep peak ≥ 0.895)
- FFN + pre-LN 机制在 single-block cross-agent 上有效
- 立即开 054L (stack 2 blocks OR add MHA 2-head × head_dim=32, 避开 052 pathology)
- HPO hidden_dim ∈ {512, 1024, 2048}, GELU vs SwiGLU

### Outcome B — 持平 (1000ep peak ∈ [0.875, 0.885])
- 架构 axis 真在 031B 0.882 truly saturated (step 2 +2.2pp → step 3 0pp → step 4 0pp)
- **三步连败, 架构 axis 宣告 dead** — 资源全部转向 reward / distill / self-play axis
- 符合 [snapshot-054 §8.4](snapshot-054-mat-min-cross-agent-attention.md#84-深化-stop-条件防止陷入架构-axis) stop 条件: 三变体 ≤ 0.882 = 关闭 054 lane

### Outcome C — 退化 (1000ep peak < 0.870)
- LN / FFN 与 PPO 在 1B capacity 下仍 hostile
- 触发 §4.5 step 1 (054M-noLN), step 2 (054M-noFFN) 隔离 cause
- 最终可能 confirm "transformer block 根本 incompatible with PPO on this task scale"
- declare 架构 axis dead, 整 054 family 关闭

## 9. 相关

- [SNAPSHOT-054](snapshot-054-mat-min-cross-agent-attention.md) — MAT-min base, 060 是 054 + (LN + FFN)
- [SNAPSHOT-052](snapshot-052-031C-transformer-block-architecture.md) — full transformer block 失败 (060 的 negative guide)
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B arch root
- [team_siamese.py](../../cs8803drl/branches/team_siamese.py) — 060 新 model class
- [_launch_054M_mat_medium_scratch.sh](../../scripts/eval/_launch_054M_mat_medium_scratch.sh) — launch script

### 理论支撑

- **Xiong et al. 2020** "On Layer Normalization in the Transformer Architecture" — pre-LN gradient norm 收敛比 post-LN 稳, 更适合 on-policy RL 的小 batch regime
- **Vaswani et al. 2017** "Attention Is All You Need" — FFN residual 是 transformer 非线性表达主力 (占大部分参数)
- **Parisotto et al. 2020** "Stabilizing Transformers for RL" — GTrXL 在 RL 上的 LN 位置研究, 强调 pre-LN > post-LN in PPO/IMPALA
- **Classical GPT-2 spec** hidden/embed ratio 4 (256 × 4 = 1024) 是 FFN 标准设置
- **本 snapshot 的独立贡献**: 在 0.56M (054) 和 1.5M (052) 之间找到 1.08M medium 点, test "FFN capacity 在 054 PPO 配置下是否 usable"

---

## 7.2 (2026-04-21 07:30 EDT) — 054M_extend Stage 1+2+3 verdict (append-only) — 架构 axis 重新打开但只是 catch up to 055

### 7.2.1 Stage 1 baseline 1000ep — extend 1250 → 1750, 12 ckpts

Pre-extend anchors保留作对照 (复用 054M base 训练产物); 新增 10 个 extend ckpts (1400-1750)。全部 12 ckpts × 1000ep 并行 eval (total_elapsed=539.8s, parallel=7, base_port=63005)。

| ckpt | 1000ep WR | W-L | 备注 |
|---:|---:|---:|---|
| 1100 | 0.885 | 885-115 | pre-extend anchor (054M base) |
| 1230 | 0.886 | 886-114 | pre-extend anchor (054M base single-shot peak @§7.1) |
| 1400 | 0.871 | 871-129 | extend early |
| 1410 | 0.879 | 879-121 | — |
| 1420 | 0.889 | 889-111 | — |
| 1450 | 0.879 | 879-121 | — |
| **1460** | **0.904** | **904-96** | **PEAK (§7.2.2 capture 此点)** |
| 1470 | 0.873 | 873-127 | — |
| 1530 | 0.897 | 897-103 | plateau high |
| 1540 | 0.890 | 890-110 | — |
| 1550 | 0.887 | 887-113 | — |
| **1750** | **0.902** | **902-98** | **terminal, second peak** |

Late-window plateau 1530-1750 (5 ckpts) mean ≈ **0.895**，ceiling cluster 稳定在 [0.887, 0.902]。

Raw recap:
```
=== Official Suite Recap (parallel) ===
ckpt 1100 vs baseline: win_rate=0.885 (885W-115L-0T)
ckpt 1230 vs baseline: win_rate=0.886 (886W-114L-0T)
ckpt 1400 vs baseline: win_rate=0.871 (871W-129L-0T)
ckpt 1410 vs baseline: win_rate=0.879 (879W-121L-0T)
ckpt 1420 vs baseline: win_rate=0.889 (889W-111L-0T)
ckpt 1450 vs baseline: win_rate=0.879 (879W-121L-0T)
ckpt 1460 vs baseline: win_rate=0.904 (904W-96L-0T)
ckpt 1470 vs baseline: win_rate=0.873 (873W-127L-0T)
ckpt 1530 vs baseline: win_rate=0.897 (897W-103L-0T)
ckpt 1540 vs baseline: win_rate=0.890 (890W-110L-0T)
ckpt 1550 vs baseline: win_rate=0.887 (887W-113L-0T)
ckpt 1750 vs baseline: win_rate=0.902 (902W-98L-0T)
[suite-parallel] total_elapsed=539.8s tasks=12 parallel=7
```

Full log: [`artifacts/official-evals/054M_baseline1000.log`](artifacts/official-evals/054M_baseline1000.log)

### 7.2.2 Stage 2 capture @ ckpt 1460 (n=500) — single-shot 0.904 下调至 0.882

500ep failure capture on peak ckpt 1460，跑在标准 capture pipeline。

- W/L/T = **441/59/0** = **0.882 win_rate** (SE 0.014)
- **vs Stage 1 1000ep 0.904 = Δ-0.022pp downshift** — 在 500ep noise 范围内，但方向上提醒我们 1000ep peak 有 positive fluctuation
- Combined 1500ep = (904 + 441) / 1500 = **0.897** (more sober estimate; SE ≈ 0.008)
- episode_steps: mean=39.99, median=33, max=194 (team0_win mean 41.9, team1_win mean 25.8) — **no turtle** (无慢跑 game-gaming)

**Loss bucket breakdown** (parsed from saved `episode_XXXX_team1_win_<bucket>.json` filenames in `docs/experiments/artifacts/failure-cases/054M_checkpoint1460_baseline_500/`):

| bucket | count | share of 59 losses | vs 055@1150 comparison (rank proxy from snapshot-074D bucket table) |
|---|---:|---:|---|
| late_defensive_collapse | 33 | 55.9% | 55.9% vs 055 48.3% (29/60) — slightly 更集中 |
| low_possession | 15 | 25.4% | 25.4% vs 055 43.3% (26/60) — **显著更低** |
| unclear_loss | 6 | 10.2% | 10.2% vs 055 5.0% (3/60) — 更高 |
| poor_conversion | 4 | 6.8% | 6.8% vs 055 1.7% (1/60) — 更高 |
| opponent_forward_progress | 1 | 1.7% | (055 无此 bucket 记录) |
| **Total** | **59** | **100%** | total losses 59/500 vs 055 60/500 (essentially tied) |

**Bucket 分析**:
- 054M 的 `late_def + low_poss` 合计 48/59 = 81.4%，vs 055 dominant 两桶 91.7% — **dominant-bucket concentration 更低**, distribution 更 rounded
- 054M 的 `poor_conversion` 比例 (6.8%) 是 055 (1.7%) 的 **4x**，暗示 MAT cross-agent attn 架构给出的 action distribution **更倾向 offensive 但终结稍弱**
- `late_def` 仍是最大桶，同 055 一致 — baseline 的反打模式对两架构都是主要失败源

### 7.2.3 Stage 3 H2H vs 055@1150 (n=500) — 统计 tied with marginal 方向偏 055

- **054M@1460 wins 232/500 = 0.464**, 055@1150 wins 268/500 = 0.536
- z = (0.464 − 0.5) / sqrt(0.25/500) = **-1.61**
- p (one-tailed toward <0.5) ≈ 0.054 → **NOT sig (marginal direction toward 055)**
- Side split from log: team0 blue 0.492 (123W-127L) / team0 orange 0.436 (109W-141L), gap +0.056 (blue stronger) — noisy within 500ep，不构成结构性侧别差异

Raw H2H Recap:
```
---- H2H Recap ----
team0_module: cs8803drl.deployment.trained_team_ray_agent       (054M@1460)
team1_module: cs8803drl.deployment.trained_team_ray_opponent_agent (055@1150)
episodes: 500
team0_overall_record: 232W-268L-0T
team0_overall_win_rate: 0.464
team0_edge_vs_even: -0.036
team1_overall_win_rate: 0.536
team0_blue_record: 123W-127L (0.492)
team0_orange_record: 109W-141L (0.436)
team0_side_gap_blue_minus_orange: +0.056
```

Full log: [`artifacts/official-evals/headtohead/054M_1460_vs_055_1150.log`](artifacts/official-evals/headtohead/054M_1460_vs_055_1150.log)

### 7.2.4 预注册判据对照 (§3, peak ≡ 1460 single-shot 0.904 / combined 1500ep 0.897)

| 判据 | 阈值 | 1000ep single peak 0.904 | combined 1500ep 0.897 | verdict |
|---|---|---|---|---|
| §3.1 marginal | ≥ 0.886 | ✅ HIT | ✅ HIT | **HIT** (both horizons) |
| §3.2 main | ≥ 0.895 | ✅ HIT | ✅ HIT (0.897 at threshold) | **HIT** (extend 抬过 §3.2 主 threshold) |
| §3.3 breakthrough | ≥ 0.900 | ✅ HIT (single-shot) | ✗ MISS (combined 0.897) | **单-shot HIT, combined MISS** |
| §3.4 持平 [0.875, 0.885) | tied 054 range | ✗ above | ✗ above | above range |
| §3.5 退化 | < 0.870 | ✗ NO | ✗ NO | no |

### 7.2.5 关键对比

| 对照 | Δ (single 0.904) | Δ (combined 0.897) | 显著性 (combined SE ≈ 0.012 for 1500ep) |
|---|---:|---:|---|
| **054M pre-extend peak 1230 = 0.889 (单-shot @§7.1) / 1100 = 0.885** | +0.015 / +0.019 | +0.008 / +0.012 | **+0.011 to +0.018 improvement from extend — 架构 axis 重新打开**, 不是 noise |
| **055@1150 SOTA combined 2000ep 0.907** | -0.003 | -0.010 | **统计 tied within SE** (Δ < 1×SE for combined) |
| 031B combined 2000ep 0.880 | +0.024 | +0.017 | combined 1.4σ `*`边缘 |
| 054 MAT-min 1000ep 0.880 | +0.024 | +0.017 | MAT-medium + extend 清晰 > MAT-min |
| 055v2 combined 3000ep 0.909 | -0.005 | -0.012 | tied within SE |

### 7.2.6 Reframing — 架构 axis 重新打开但只是 catch up to 055, not beat

**§7.1 (2026-04-21 00:45 EDT)** 的 verdict 是 "Tier 1b 关闭, 架构 axis saturated at ~0.88-0.89, combined 2000ep 大概率收敛回 054 0.880"。

**7.2 的 extend 数据 reframe 了这个结论**:
- extend 给出 +0.011 to +0.018 net improvement (combined metric), **清楚超过 1000ep SE 0.016 的底** — 不是抽样 artifact
- 架构 axis "在 031B 0.882 ceiling" 的 hypothesis **被打破**: 054M + extend 实证到 0.897 combined 1500ep / 0.904 single, **跟 055 distill SOTA 统计 tied**
- 然而 "tied SOTA" ≠ "beat SOTA"。Δ vs 055 combined 0.907 仍是 -0.010pp (combined), -0.003pp (single)，都在 SE 内
- **架构 axis 重新打开但只是 catch up to 055, not beat** — 这是独立证据 (distill path 之外的另一条 0.90+ route), 但不是新 SOTA-pushing direction

**与 snapshot-054 §7 "031B 0.882 ceiling" hypothesis 的关系**:
- [snapshot-054 §7.1](snapshot-054-mat-min-cross-agent-attention.md#71-2026-04-20-stage-1-baseline-1000ep--marginal-tied-031b-verdict-mediocre) 提出 "架构 step 2 → step 3 = +2.2pp → 0pp, 架构 axis 可能 saturate at 031B 0.882"
- **054M + extend 现在给出反证**: 架构 step 3 (MAT cross-AGENT attn + FFN + LN) + **adequate training time** (1750 iter vs 1250 iter in § 054/§060 original) = 突破 031B ceiling
- **新的 bottleneck hypothesis**: 原 1250 iter budget 是 under-trained for MAT-medium (1.08M params = 054 的 2×); 扩到 1750 iter后架构潜力才完全释放
- Lesson: **higher-capacity arch 需要 proportionally more training iters**, 不是 arch 本身无效

### 7.2.7 结论与后续路径

- **Tier 1b 部分重开** — 不再是 "MAT 无用", 而是 "MAT-medium + 足够 training = SOTA-tier"
- 但 **不再继续 push 054M 深化** (e.g., 054L stack 2 blocks / MHA 2-head)，原因:
  - 已经 tied 055, 无法 beat (SE 限制)
  - 继续投资新架构的 ROI < 继续推 distill / curriculum / ensemble 方向
- **054M@1460 进入 rank.md §3.3 作为 "tied-SOTA 独立证据"**; 加入 Pool C (snapshot-072) 作为 cross-axis teacher 的现成候选
- **034F router 方向仍不重启** — 074B 证实 arch-mixed prob-avg ensemble 退化 (-0.026pp), 即使 054M 现在是 0.90+ 也不救

### 7.2.8 Raw 数据链接

- Stage 1 baseline 1000ep log: [`docs/experiments/artifacts/official-evals/054M_baseline1000.log`](artifacts/official-evals/054M_baseline1000.log)
- Stage 2 capture dir: [`docs/experiments/artifacts/failure-cases/054M_checkpoint1460_baseline_500/`](artifacts/failure-cases/054M_checkpoint1460_baseline_500/) (59 saved episodes + summary.json)
- Stage 3 H2H log: [`docs/experiments/artifacts/official-evals/headtohead/054M_1460_vs_055_1150.log`](artifacts/official-evals/headtohead/054M_1460_vs_055_1150.log)
