# SNAPSHOT-045: Architecture × Learned Reward Combo (031A + 036D learned reward)

- **日期**: 2026-04-19
- **负责人**:
- **状态**: 实现就绪 / 待启动
- **依赖**:
  - [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md): `031A@1040` (Siamese architecture, scratch v2, 1000ep avg 0.860)
  - [SNAPSHOT-036D](snapshot-036d-learned-reward-stability-fix.md): learned reward model `ray_results/reward_models/036_stage2/reward_model.pt` (W/L 多头 v2 labels, 184k samples)

## 0. Known gap 诊断（动机来源）

当前 reward × architecture 矩阵存在**未填的格子**：

| Architecture × Reward | v2 only | v2 + learned reward | learned only |
|---|---|---|---|
| **flat MLP per-agent** (MAPPO) | 029B 0.846 (1000ep) | **036D 0.860** (+1.4pp) | ? (从未测) |
| **Siamese team-level** | **031A 0.860** | **? (本 snapshot 045A)** | **? (本 snapshot 045B)** |

learned reward 在 per-agent 上贡献 +1.4pp (029B → 036D)。如果同样 +1.4pp 加到 031A 的 Siamese 上 → **0.874**，**首次明确突破 0.86 ceiling**。

040 系列已测 "031A + various shaping (PBRS / event / depenalized / entropy)" → 全部 saturation 在 0.86。但 **learned reward 是唯一没测过的 X**。

## 1. 核心假设

> **031A@1040 的 0.860 是 v2-shaping × Siamese architecture 联合 ceiling。把 036D 的 learned reward (data-driven W/L signal) 叠加到 031A 上，能否复现 029B → 036D 的 +1.4pp，把 1000ep 推到 0.87+，首次明确突破 0.86 ceiling？**

子假设：

- **H1 (combo +1.4pp)**: 031A + learned reward (045A) → 1000ep 0.874 (036D-style 增益叠加到 031A 高 base)
- **H2 (combo additive)**: 0.874 是 v2-architecture ceiling 之外的，证明 learned reward 提供**架构无关**的 +1.4pp signal source
- **H3 (combo also saturation)**: 045A 也卡 0.86 → learned reward 的 +1.4pp 不能跨架构 transfer，可能是 per-agent 特定的 artifact
- **H4 (learned-only baseline)**: 045B (learned only, no v2) → 大概率 0.78-0.82 (弱于 v2-only)，作为 045A 增益的"learned-share"测算对照

如果 H1 成立，045A 是项目首个 ≥ 0.87 的 lane，距离 0.90 grading 阈值仅 -3pp。

## 2. 三条候选 lane

| Lane | Base | Reward | shaping env vars |
|---|---|---|---|
| **045A** (主线) | 031A@1040 (Siamese, warmstart) | v2 + learned (036D 配方) | `USE_REWARD_SHAPING=1` `LEARNED_REWARD_MODEL_PATH=...` `LEARNED_REWARD_SHAPING_WEIGHT=0.003` `LEARNED_REWARD_WARMUP_STEPS=10000` |
| **045B** (对照: learned-only) | 031A@1040 | learned only (no v2) | `USE_REWARD_SHAPING=0` `LEARNED_REWARD_MODEL_PATH=...` `LEARNED_REWARD_SHAPING_WEIGHT=0.003` |
| **045C** (条件: scratch combo) | scratch | v2 + learned | 同 045A 但无 warmstart, 14h budget |

**首轮先跑 045A**: 它是直接对比 036D → 031A+036D 的 combo。如果 045A 突破 → 045B 测 learned reward 在 Siamese 架构上**单独**多少贡献。如果 045A saturation → 045B 也大概率没意义，跳过。

045C 视 045A 结果决定是否启动（scratch 比 warmstart 慢但更干净测 architecture × reward 联合训练）。

## 3. 训练超参（045A 主线）

```bash
WARMSTART_CHECKPOINT=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040

# Architecture (031A 同款)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128

# v2 shaping (031A 同款)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001
SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01
SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Learned reward (036D 配方)
LEARNED_REWARD_MODEL_PATH=ray_results/reward_models/036_stage2/reward_model.pt
LEARNED_REWARD_SHAPING_WEIGHT=0.003
LEARNED_REWARD_WARMUP_STEPS=10000
LEARNED_REWARD_APPLY_TO_TEAM1=0

# PPO (031A 同款)
LR=1e-4
CLIP_PARAM=0.15
NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000
SGD_MINIBATCH_SIZE=2048

# Stage 2 budget (200 iter, 同 040 系列)
MAX_ITERATIONS=200
EVAL_INTERVAL=10
EVAL_EPISODES=50
CHECKPOINT_FREQ=10
```

每 lane budget: ~3-4h on H100。

## 4. 预注册判据

### 4.1 主判据（成立门槛）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| official 1000ep | ≥ 0.86 | 至少不退化于 031A base |
| **official 1000ep peak** | **≥ 0.875** | learned reward 真带来 +1.5pp 增益 (= 036D's +1.4pp + 一点) |

### 4.2 突破判据（强成立）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| **official 1000ep peak** | **≥ 0.880** | **首次明确突破 0.86 ceiling** (+2pp on max) |
| H2H vs 031A@1040 | ≥ 0.52 | 直接证明 combo 真带来 H2H 优势 |
| H2H vs 036D@150 | ≥ 0.55 | 比 036D peer-axis 强 (验证 architecture buff) |

### 4.3 失败判据

| 条件 | 解读 |
|---|---|
| 1000ep < 0.85 | learned reward 在 Siamese 上反向（可能 reward shock + Siamese encoder 灵敏度差异） |
| 1000ep ≈ 0.86 | combo saturation: learned reward 的 +1.4pp 是 per-agent 特定 artifact，不 transfer |
| 1000ep peak ∈ [0.866, 0.874] | 部分 transfer (~+0.6-1.4pp)，方向性正但未达突破 |

### 4.4 lane 优先级

1. **045A (combo)** — 首轮主线，3h
2. **045B (learned only)** — 仅在 045A 突破时跑，作为 v2 share/learned share 量化对照
3. **045C (scratch combo)** — 仅在 045A 突破 + 045B 也有信号时跑，验证 scratch 训练能否复现 combo 增益

## 5. 风险

### R1 — Reward distribution shift

learned reward model 是在 029B/025b/017/028A 等 **per-agent v2-shaped 轨迹**上训的。031A 是 team-level Siamese，其 trajectory distribution 跟 reward model 训练分布**不同**：
- per-agent: agent 0/1 各自最大化自己的 reward
- team-level (031A): 联合 policy 优化总和

Reward model 对 031A 的 trajectory 可能 mis-fire，给出错误的 dense signal。

**缓解**: λ=0.003 (小) + warmup_steps=10000 让 PPO 主要靠 v2 + sparse 训前 10 iter，learned reward 慢慢介入。如果 inf 率 > 30% 提前 abort。

### R2 — Combo saturation (H3 真相)

如果 040 全 4 lane (PBRS/event/depenalized/entropy) 全部 saturate 在 0.86，learned reward 大概率也 saturate。这暗示 0.86 是 **031A architecture × v2 shaping** 的硬 ceiling，跟具体 X 无关。

**缓解**: 这本身就是 045 的 verdict。如果 045A 也 saturation，证明 v2-architecture ceiling 是真的，下一步必须换更激进路径（self-play, scratch 不同架构, 或 0.86 接受为 PPO 上限）。

### R3 — 045B mis-fire

learned reward model 是 v2-shaped trained，pure-learned (无 v2) 训练时 trajectory distribution 完全 different，model 给出的 dense signal 可能完全 wrong direction。

**缓解**: 045B 仅作为对照，预期较低 (0.78-0.82)。失败的话不深究，主结论由 045A 给出。

## 6. 不做的事

- 不做 reward model 重训（用 036/039 现成 reward_model.pt）
- 不并行 launch 045A/B/C，按 phase gate
- 不混 PBRS / event 进 045A（保持 reward 端 X 单一变量）

## 7. 执行清单

1. 已写 batch [soccerstwos_h100_cpu32_team_level_045A_combo_on_031A1040_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_045A_combo_on_031A1040_512x512.batch)，并补齐 team-level trainer 的 `LEARNED_REWARD_*` 接线（沿用 `036D` reward model 与 `031A` 同款 PPO / Siamese 配置，不额外混入 `040D` 的 entropy 变量）
2. 在 GPU 节点 launch (~3h)
3. 训练完用 [post-train-eval skill](../../.claude/skills/post-train-eval/SKILL.md) 跑 1000ep + capture + H2H (vs 031A@1040 + 036D@150)
4. 按 §4 判据决定是否启动 045B / 045C
5. verdict 写 §11

## 11. 045A 首轮结果 (2026-04-19, append-only)

### 11.1 训练完成

run dir: `ray_results/045A_team_combo_on_031A1040_formal_rerun1/` (200 iter, 8M steps)

- best_reward_mean: +2.2297 @ iter 5（早期高 reward, 后期 -0.5 ~ -0.8 区间，说明 learned reward 在 late training 拉低 raw reward）
- best_eval_baseline (50ep internal): **0.940 @ iter 140** — 但 50ep SE ±0.028, 单点 spike artifact
- best_eval_random: 1.000

### 11.2 数值健康度

`progress.csv` 200 iter 检查：

| 检查 | 结果 |
|---|---|
| total_loss inf | 0 |
| policy_loss inf | 0 |
| kl inf | 0 |
| max\|total_loss\| | 2.56 |
| max\|kl\| | 2.32 |

**clean，无 inf**。对比 036D (per-agent + learned) 31.7% inf — 045A 在 031A 强 base 上 + warmup 10000 + λ=0.003 让 learned reward 慢慢介入，避免了数值爆炸。

### 11.3 1000ep 官方 eval (top 10% fallback per skill, 8 ckpts on atl1-1-03-013-19-0, port 62005, total elapsed 469s)

[pick_top_ckpts](../../scripts/eval/pick_top_ckpts.py) top 5% 只给 3 个 ckpt → 触发 [skill fallback rule](../../.claude/skills/post-train-eval/SKILL.md) 重跑 `--top-pct 10` 得 8 ckpts。

| ckpt | 1000ep WR | NW-ML | 内部 50ep | Δ (1000-50) |
|---:|---:|:---:|---:|---:|
| 10 | 0.839 | 839-161 | 0.90 | -0.061 |
| 20 | 0.851 | 851-149 | 0.88 | -0.029 |
| 130 | 0.844 | 844-156 | 0.86 | -0.016 |
| 140 | 0.863 | 863-137 | **0.94** | **-0.077** |
| 150 | 0.862 | 862-138 | 0.88 | -0.018 |
| 170 | 0.853 | 853-147 | 0.86 | -0.007 |
| **180** | **🏆 0.867** | **867-133** | 0.90 | -0.033 |
| 190 | 0.865 | 865-135 | 0.82 | +0.045 |

mean = **0.856**, peak = **0.867 @ 180**, range [0.839, 0.867]。

### 11.4 严格按 [§4 判据](#4-预注册判据)

| 阈值 | 045A | verdict |
|---|---|---|
| §4.1 主: 1000ep ≥ 0.86 | ✅ peak 0.867 / mean 0.856 | 持平 031A (0.860)，不退化 |
| §4.1 主: peak ≥ 0.875 | ❌ 0.867 (差 -0.008pp) | learned reward **没拿到 +1.5pp** |
| §4.2 突破: peak ≥ 0.880 | ❌ | 没突破 0.86 ceiling |
| **§4.3 失败: ≈ 0.86 → combo saturation** | ✅ **CONFIRMED** | **H3 确认** |

### 11.5 关键发现 — learned reward 给 ~+0.7pp，**两个架构都拿到 sub-noise gain**

| Architecture × Reward | v2 only (1000ep) | v2 + learned (1000ep) | Δ | 来源 |
|---|---|---|---|---|
| **flat MLP per-agent** (MAPPO) | **029B@190 = 0.846** | **036D@150 = 0.860** | **+1.4pp** | rank.md §3.3 |
| **Siamese team-level** | **031A@1040 = 0.860** | **045A@180 = 0.867** | **+0.7pp** | §11.3 上方 |

注意 029B 的 500ep 是 0.868（rank.md §3.1）但 1000ep 是 0.846（§3.3） —— 1000ep 才是稳定 SOTA 数字（snapshot-029 §11.4 column 误标 "1000ep" 实际是 500ep，doc bug，已识别 needs fix）。

**学习 reward 在两个架构都**有方向性 gain，但都**未达统计显著**:
- per-agent: +1.4pp vs SE ±0.012 (1000ep) → 边缘显著（z ≈ 1.2, 上界 [0.838, 0.882] 与 029B 上界 [0.824, 0.868] 几乎不重叠）
- team-level: +0.7pp vs SE ±0.016 (1000ep) → 远未显著（落在 031A 95% CI 内）

→ **learned reward 是 "weak signal source"**：在 0.86+ student 上 marginal value 0.7-1.4pp，但**不能跨过架构的硬上限 0.86**。

team-level 拿到的 gain 是 per-agent 的一半 (+0.7 vs +1.4)，可能因 031A Siamese 共享 encoder 已经"内化"了一些 learned reward 想 inject 的协调 signal，learned reward 的边际作用被部分抵消。

### 11.6 跟 040(4 lanes) 合并起来的更大 picture

**031A architecture 上 6 个 reward-axis lanes 全 saturate**:

| lane | reward variant | 1000ep peak | Δ vs 031A 0.860 |
|---|---|---:|---:|
| 031A base (v2 only) | v2 | 0.860 | — |
| 040A | v2 + PBRS | 0.865 | +0.005 |
| 040B | v2 + event | 0.864 | +0.004 |
| 040C | v2 + depenalized | 0.863 | +0.003 |
| 040D | v2 + entropy-only | 0.864 | +0.004 |
| **045A** | **v2 + learned reward** | **0.867** | **+0.007** |
| 042A3 | v2 + KL distill | 0.863 | +0.003 |

range: [0.860, 0.867] = ±0.007pp around 031A base。每条 lane 都有方向性 +Δ 但都在 1000ep SE ±0.016 内 → **统计上无法区分 from base**。

**注意**: 在 per-agent 上，learned reward 给了 +1.4pp（029B 0.846 → 036D 0.860），同样的 PBRS 在 per-agent 上 026B 也给了 +1.8pp（026B 1000ep 0.864 vs base 0.846）。**reward 对 per-agent 的边际值 ≈ 1-2pp**，对 team-level (031A) **降到 0.5-1pp**。

可能解释:
- per-agent 架构 (336 dim ego obs) 信息少，reward signal 边际值高
- team-level 架构 (672 dim concat obs) 已"内化"更多协调信息，reward 边际作用被部分抵消
- 与 architecture step 1 (flat → Siamese) 一样，**好架构 absorb reward signal 减少其重要性**

### 11.7 strategic implication — 架构是唯一被验证的 +ΔWR 路径

- 6 个 reward-axis lanes (040 ×4 + 045A + 042A3) 在 031A 上**全 sub-noise**
- 1 个 architecture-axis lane (031B cross-attention) 突破到 0.882 (+2.2pp，**显著**)
- ratio: 6:1 evidence ratio for "architecture > reward" 路径

→ 031A 架构 ceiling 0.86 是**真正的 architecture-imposed 上限**，reward 边际值在这个 ceiling 上被 saturate。**033/036 在 per-agent 上看到的 +1-2pp reward gain 是 per-agent 架构信息瓶颈的 artifact，不能假设到 team-level**。

### 11.8 045B / 045C 是否启动

按预注册 §2.4「lane 优先级」:

> 045A 突破 → 045B (learned only) 量化 v2/learned share；
> 045A saturation → 045B 也大概率没意义，跳过。
> 045C 视 045A 决定。

045A confirmed saturation → 原计划 045B / 045C 都 skip。

**实际操作 (2026-04-19)**: 为了正式 falsify 「v2 淹没了 learned signal」的假设（H_replace），仍然 launch 了 045B (learned only on 031A@1040, 200 iter)。结果见 §12。045C (learned-only on per-agent base) 仍 skip — per-agent 上 036D 已经做过类似变体，且 045B 结果若 negative 就不需要再加一条 lane 来确认。

### 11.9 Raw recap (verification)

```
=== Official Suite Recap (parallel) ===
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.867 (867W-133L-0T)
.../checkpoint_000190/checkpoint-190 vs baseline: win_rate=0.865 (865W-135L-0T)
.../checkpoint_000140/checkpoint-140 vs baseline: win_rate=0.863 (863W-137L-0T)
.../checkpoint_000150/checkpoint-150 vs baseline: win_rate=0.862 (862W-138L-0T)
.../checkpoint_000170/checkpoint-170 vs baseline: win_rate=0.853 (853W-147L-0T)
.../checkpoint_000020/checkpoint-20  vs baseline: win_rate=0.851 (851W-149L-0T)
.../checkpoint_000130/checkpoint-130 vs baseline: win_rate=0.844 (844W-156L-0T)
.../checkpoint_000010/checkpoint-10  vs baseline: win_rate=0.839 (839W-161L-0T)
[suite-parallel] total_elapsed=468.8s tasks=8 parallel=7
```

完整 log: [docs/experiments/artifacts/official-evals/045A_baseline1000.log](../../docs/experiments/artifacts/official-evals/045A_baseline1000.log)

## 12. 045B 首轮结果 — learned-only on 031A1040 (2026-04-19, append-only)

### 12.1 设计回顾 — 为什么仍跑 045B

§11.8 原计划 skip 045B，但留了一个未关闭的反事实：「会不会是 v2 reward 信号太强，把 learned reward 完全淹没了？」如果是，那 045A 的 saturation 不能用来证明 learned reward weak — 只能说"v2 + learned"组合没用。

要 falsify 这个，必须做 **learned-only** （USE_REWARD_SHAPING=0, 所有 SHAPING_*=0），让 learned reward 单独驱动 student。如果 045B 还是 saturate ≈ 0.86, 则可以排除 H_replace 「v2 淹没」假设；如果 045B 突破 0.88, 则 045A 的 saturation 是 v2 主导的副作用。

### 12.2 训练完成

run dir: `ray_results/045B_learned_only_on_031A1040_512x512_20260419_095729/` (200 iter, 8M steps)

- best_eval_baseline (50ep internal): **0.96 @ iter 110** — 50ep SE ±0.028, 单点 spike，需 1000ep 验证
- best_eval_random: 1.000

数值健康度（与 045A 相同 base 031A@1040 + warmup 10000 + λ=0.003）：clean，未观察到 inf。

### 12.3 1000ep 官方 eval (top 10% fallback per skill, 6 ckpts on atl1-1-03-014-23-0, port 56305, total elapsed 253s)

[pick_top_ckpts](../../scripts/eval/pick_top_ckpts.py) top 5% 只给 3 个 ckpt → 触发 [skill fallback rule](../../.claude/skills/post-train-eval/SKILL.md) 重跑 `--top-pct 10` 得 6 ckpts (100/110/120/140/150/160)。

| ckpt | 1000ep WR | NW-ML | 内部 50ep | Δ (1000-50) |
|---:|---:|:---:|---:|---:|
| 100 | 0.859 | 859-141 | 0.86 | -0.001 |
| 110 | 0.836 | 836-164 | **0.96** | **-0.124** |
| 120 | 0.843 | 843-157 | 0.92 | -0.077 |
| 140 | 0.854 | 854-146 | 0.92 | -0.066 |
| **150** | **🏆 0.870** | **870-130** | 0.94 | -0.070 |
| 160 | 0.857 | 857-143 | 0.94 | -0.083 |

mean = **0.853**, peak = **0.870 @ 150**, range [0.836, 0.870]。

注意 50ep 内部 eval 普遍偏高 6-10pp（除 ckpt 100），说明 045B 的 50ep eval noise 比 045A 更大。1000ep 才是真信号。

### 12.4 严格按 [§4 判据](#4-预注册判据) — 045B 与 045A 平行对比

| 阈值 | 045A combo | 045B learned-only | verdict |
|---|---|---|---|
| §4.1 主: 1000ep mean ≥ 0.86 | 0.856 (-0.004) | 0.853 (-0.007) | 都未达 |
| §4.1 主: peak ≥ 0.875 | 0.867 (-0.008) | 0.870 (-0.005) | 都未达，045B 更接近 |
| §4.2 突破: peak ≥ 0.880 | ❌ 0.867 | ❌ 0.870 | 都没突破 |
| §4.3 saturation ≈ 0.86 | ✅ peak 0.867 | ✅ peak 0.870 | 都 saturate |

差距 045B - 045A：peak +0.003pp, mean -0.003pp。**两者 statistically tied**（1000ep SE ±0.016, |Δ| < 0.5σ on both）。

### 12.5 关键发现 — H_replace 「v2 淹没 learned reward」**被 falsify**

| Scenario | reward 配置 | 1000ep peak | 1000ep mean |
|---|---|---:|---:|
| **031A base** (baseline) | v2 only | 0.860 | 0.856 |
| **045A combo** | v2 + 0.003·learned | 0.867 | 0.856 |
| **045B learned-only** | 0.003·learned only | 0.870 | 0.853 |

三组**统计上无法区分**（互相落在 SE ±0.016 的 95% CI 内）。

**关键含义**：
1. **不是 v2 淹没** — 把 v2 完全去掉，learned reward 单独驱动 student，结果跟 v2-only 几乎一样。如果 v2 真的"挡住"了 learned signal, 045B 应该明显 ≠ 031A base，但 0.870 vs 0.860 在噪声里。
2. **learned reward 在 0.86 ceiling 上 marginal value 接近零** — 给 +0.7pp（045A）或 +1.0pp（045B），都 sub-noise。这跟 §11.5 的结论 "team-level architecture absorb reward signal" 一致 — 不是 v2 干扰，是架构本身已 saturated。
3. **λ=0.003 的 reward 强度可能太弱** — 045B 把 v2 完全去掉只比 045A 多了 0.003 pp peak, 说明 learned reward 在该 λ 下对策略的影响接近边界值。但提高 λ 在 036D 上验证过会引发 numerical instability（snapshot-036c → 036d fix）, 不一定有 headroom。

### 12.6 Updated 6:1 → 6:1 evidence ratio for "architecture > reward" 不变

| lane | reward variant | 1000ep peak | Δ vs 031A 0.860 |
|---|---|---:|---:|
| 031A base (v2 only) | v2 | 0.860 | — |
| 040A | v2 + PBRS | 0.865 | +0.005 |
| 040B | v2 + event | 0.864 | +0.004 |
| 040C | v2 + depenalized | 0.863 | +0.003 |
| 040D | v2 + entropy-only | 0.864 | +0.004 |
| 045A | v2 + learned reward | 0.867 | +0.007 |
| **045B** | **learned reward only** | **0.870** | **+0.010** |
| 042A3 | v2 + KL distill | 0.863 | +0.003 |

range: [0.860, 0.870] = ±0.010pp around 031A base。**7 条 reward-axis lanes 全 sub-noise**。架构 axis（031B cross-attention 0.882）仍是唯一显著突破。

### 12.7 对 in-flight 051A/051B 的影响

snapshot-051 (Stage 2) 的设计是同样的 combo / learned-only 二选一：
- **051A** = combo (v2 + 0.003·learned_051) on **031B@1220** warmstart (cross-attention)
- **051B** = learned-only (0.003·learned_051) on **031B@1220**

051 vs 045 唯一的变量是 **reward model 的训练数据**：045 用的是 029B vs baseline failures (per-agent, 1500ep), 051 用的是 strong-vs-strong H2H failures (5 pairs × 400ep, mixed team-level/per-agent)。

**根据 045B 的结论可以预测 051**：
- 如果 「reward model 数据源」 是 leverage 点 → 051 应该明显比 045 强（因为 strong-vs-strong 的 failure pattern 跟 baseline 不同，可能更接近 architecture ceiling 处的"真正难点"）
- 如果 「architecture ceiling 0.882 (031B)」 是真硬上限 → 051 任何 reward 变种都 saturate ≈ 0.882

045B 已经 falsify 了「v2 淹没」假设, 所以 051 的 falsifiable 假设也只剩下「reward model 数据源 matters」。如果 051A/051B 1000ep 都 ≈ 0.88 → reward 路径在 031B 上也 saturate，应该把后续算力转向 architecture (031C 052A)。

### 12.8 Raw recap (verification)

```
=== Official Suite Recap (parallel) ===
.../checkpoint_000150/checkpoint-150 vs baseline: win_rate=0.870 (870W-130L-0T)
.../checkpoint_000100/checkpoint-100 vs baseline: win_rate=0.859 (859W-141L-0T)
.../checkpoint_000160/checkpoint-160 vs baseline: win_rate=0.857 (857W-143L-0T)
.../checkpoint_000140/checkpoint-140 vs baseline: win_rate=0.854 (854W-146L-0T)
.../checkpoint_000120/checkpoint-120 vs baseline: win_rate=0.843 (843W-157L-0T)
.../checkpoint_000110/checkpoint-110 vs baseline: win_rate=0.836 (836W-164L-0T)
[suite-parallel] total_elapsed=253.2s tasks=6 parallel=6
```

完整 log: [docs/experiments/artifacts/official-evals/045B_baseline1000.log](../../docs/experiments/artifacts/official-evals/045B_baseline1000.log)

## 8. 相关

- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031A base (架构 baseline) + 031B (cross-attention 突破)
- [SNAPSHOT-036D](snapshot-036d-learned-reward-stability-fix.md) — learned reward model 来源
- [SNAPSHOT-040](snapshot-040-team-level-stage2-on-031A.md) — 同 base 不同 X 的 saturation 对照（4 lanes）
- [SNAPSHOT-041](snapshot-041-per-agent-stage2-pbrs-on-036D.md) — 同 reward 不同 architecture 的 saturation 对照
- [SNAPSHOT-042](snapshot-042-cross-architecture-knowledge-transfer.md) — KL distill 持平 saturation 对照
- [rank.md](rank.md) — 数值依据

## 13. 045A H2H Stage-3 verdict — 045A 是 031A noisy clone, 不是新 model (2026-04-19, append-only)

### 13.1 触发原因

snapshot-045 §11.8 / §12 跳过了 H2H + capture (saturation gate)。基于 v2 桶 capture (snapshot-051 §8.6 落地的 045A wasted_possession 55% fingerprint), 重新评估 045A 是否在 peer-axis 上有 marginal value。Stage 3 H2H ×4 在 013-19-0 / 015-2-0 上跑 1000ep。

### 13.2 H2H 结果 (n=1000 each, port 55405-55465)

| 对手 | 045A W-L | WR | blue / orange | z | p | verdict |
|---|---|---|---|---|---|---|
| **vs 031B@1220** | 491-509 | **0.491** | 0.494 / 0.488 | -0.57 | 0.28 | NS, 平 031B (没显著负) |
| **vs 036D@150** | 570-430 | **0.570** | 0.568 / 0.572 | 4.43 | <0.0001 | **★★★ 显著击败** |
| **vs 029B@190** | 575-425 | **0.575** | 0.578 / 0.572 | 4.74 | <0.0001 | **★★★ 显著击败** |
| **vs 031A@1040** | 492-508 | **0.492** | 0.508 / 0.476 | -0.51 | 0.31 | NS, **平 base 031A** |

### 13.3 关键发现 — 045A combo lane 没在 peer-axis 上获得任何 base 之上的优势

**045A vs 031A H2H = 0.492 (NS)** 是决定性证据:
- 045A 是 031A@1040 + v2 + 0.003·learned_045 reward 的 200-iter combo lane
- 如果 learned reward 真的产生不同 policy, H2H 应该有 ≠ 0.5 的方向性 (无论 + 或 -)
- 实际 H2H 几乎完美 50% → **learned reward 没产生新 policy direction, 只是 0.86 ceiling 的 noisy 复刻**

**vs 036D / 029B 的 +7pp 显著优势是架构差距 (team-level >> per-agent), 不是 045A 本身的价值**:
- 031A vs 029B (snapshot-031 §11.5 H2H) = 0.552 (z=3.29) — 同样 +5pp 显著
- 031A vs 025b (snapshot-031 §11.6) = 0.532 (z=2.02) — 类似
- 045A 跟 031A 是同 base, peer-axis 表现自然继承 031A 对 per-agent 的架构优势
- → 这个 +7pp 不是「045A 比 031A 更强」, 是「045A ≈ 031A, 而 031A 比 per-agent 强」

### 13.4 v2 桶 fingerprint 不等于 policy diversity

snapshot-051 §8.6 我用 v2 桶 (045A wasted_possession 55%, 比 031A +13pp) 论证 045A 是 ensemble orthogonal candidate。**13.2 H2H 反驳**:

| 信号源 | 045A vs 031A 差异 | 解读 |
|---|---|---|
| baseline 1000ep WR | +0.7pp marginal sub-noise | 几乎相同 |
| v2 桶 wasted_possession | **+13.0pp** | 看似 orthogonal |
| **H2H peer-axis** | **0.492 (NS)** | **决策空间几乎重合** |

→ **v2 桶 fingerprint 是 episode-level statistic, 不是 step-level decision boundary**。两个 policy 可以有同样的"输球时控球高"统计但实际 step decision 完全等价 (都打不过 baseline 的同一种 trap)。**v2 桶不能作为 ensemble selection 判据**, 这是 snapshot-051 §8.6 的方向错误, 在 snapshot-034 §11 也得到 ensemble eval 的实证 (034ea anti-lift)。

### 13.5 045A 最终定位

| 维度 | 045A 评估 |
|---|---|
| **single submission candidate** | ❌ 比 031A 平, 比 031B / 051A 弱, 没必要替换 |
| **ensemble member** | ❌ 跟 031A peer 平 → 加进 ensemble 是冗余 (snapshot-034 §11 实证 anti-lift) |
| **research lane** | ✅ 关 — 045 lane (snapshot-045 §11.8 + §12) verdict 完整, 045A H2H 补完, 045B 0.870 / learned-only sub-noise 都已落地 |
| **价值** | 作为「v2 + learned 在 031A base 上 saturation 的对照点」, 给 snapshot-051 提供 base × reward 4 组对比的两组数据 |

### 13.6 Raw recap

```
=== 045A_180 vs 031B_1220 (n=1000) ===
team0_overall: 491-509 = 0.491 (blue 0.494 / orange 0.488)

=== 045A_180 vs 036D_150 (n=1000) ===
team0_overall: 570-430 = 0.570 (blue 0.568 / orange 0.572)

=== 045A_180 vs 029B_190 (n=1000) ===
team0_overall: 575-425 = 0.575 (blue 0.578 / orange 0.572)

=== 045A_180 vs 031A_1040 (n=1000) ===
team0_overall: 492-508 = 0.492 (blue 0.508 / orange 0.476)
```

完整 logs: [headtohead/](../../docs/experiments/artifacts/official-evals/headtohead/) (4 个 045A_180_vs_*.log)
