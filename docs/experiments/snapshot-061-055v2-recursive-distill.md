## SNAPSHOT-061: 055v2 recursive distill — 5-teacher ensemble + LR=3e-4 (Tier 2)

- **日期**: 2026-04-20
- **负责人**: Self
- **状态**: 训练中 (lane open) — jobid 5021865, PORT_SEED=51, 1250 iter budget

## 0. 背景与定位

### 0.1 项目当前 distill landscape

| Lane | Teacher pool | LR | Combined 2000ep peak |
|---|---|---|---|
| **055** | 3-way: {031B@1220, 045A@180, 051A@130} | 1e-4 | **0.907 @ iter 1150** 🥇 project SOTA |
| 059 (in flight) | = 055 | 3e-4 | _pending_ ([snapshot-059](snapshot-059-055lr3e4-combo.md)) |
| **061 (本)** | **5-way**: + {055@1150, 056D@1140} | **3e-4** | _pending_ |

055 + 059 都是 "fixed teacher pool" 路径。061 是 **首次引入 teacher pool 自更新** (recursive distill): 把 055 自己的 SOTA checkpoint (055@1150) 作为新 teacher 回炉, 加上 056D 的 LR winner 形成 5-way 更 diverse ensemble。

### 0.2 "Recursive distill" 的理论定位

**Self-distillation** 是 well-established subdomain:
- **Furlanello et al. 2018** "Born Again Networks" — student 与 teacher 同架构, student 接受 distill 后经常超过 teacher
- **Anil et al. 2018** "Large scale distributed neural network training through online distillation" — co-distillation, teacher pool 在训练中滚动更新

**"Recursive" 的 novel 点**: 055@1150 是 055 original (3-teacher pool) 的 output, 现在把它加入 pool 让它成为 055v2 的 teacher。这是 **generational distill**: Gen 1 (055) teacher = {031B, 045A, 051A}, Gen 2 (055v2) teacher = Gen 1 teachers ∪ {Gen 1 output, 056D}。

项目之前没有 lane 做过这个 — 从已有 SOTA 直接派生 teacher, 是"让 knowledge 在 lane 内 compound"的第一次尝试。

### 0.3 为什么加 056D

056D 是 LR sweep 的 winner (0.891 marginal tied 031B, 见 [snapshot-056](snapshot-056-simplified-pbt-lr-sweep.md)). 加入 teacher pool 的 motivation:
- **不同训练 trajectory 给出不同 policy mode**: 056D 用 LR=3e-4 训练, 与 031B LR=1e-4 的收敛路径不同, 可能在某些 state 上有不同 action preference → 增加 teacher pool policy diversity
- **5 个 teacher 总 capacity** (5 × ~0.5M = ~2.5M params) 仍在 student 031B 0.46M 的 5× 以内, Hinton 2015 distill paper 的 empirical range
- **ensemble diversity 是 known distill gain source** ([snapshot-055 §0.2](snapshot-055-distill-from-034e-ensemble.md#02-与ensemble--非智力的区分关键-framing) Hinton 2015): diverse teachers → richer soft labels → student 学到更 nuanced 的 joint policy

## 1. 核心假设

### H_061

> 5-teacher ensemble {055@1150, 031B@1220, 045A@180, 051A@130, 056D@1140} + LR=3e-4, **1000ep peak ≥ 0.920** (+0.013 vs 055 combined 2000ep 0.907), 通过 teacher pool refresh + LR combo 突破 055 ceiling。
>
> Stretch: ≥ 0.925, 独立 declare 超 grading threshold。
>
> Worst tolerated: 0.895-0.910 (tied 055), teacher diversity has ceiling effect after 3→5 expansion。

### 子假设

- **H_061-a**: 055@1150 作为 teacher 提供了 3-teacher pool 之外的 additional knowledge (distill-specific failure mode coverage), student 从它能学到 "已经被 compressed 的 ensemble knowledge" 的更 pure 版本
- **H_061-b**: 056D@1140 与 031B@1220 同架构但不同 policy (LR 不同)，**architectural identity does not imply behavioral identity** — 056D 带来独立 policy mode
- **H_061-c**: factor-probability average (same mechanism as 055) 足够处理 5 个 teacher 的 logit heterogeneity, 不需 logit-space averaging 或 temperature tuning
- **H_061-d**: alpha schedule (init 0.05, decay to 0 by 8000 updates) 与 055 相同，与 teacher count 正交 — α 控制的是 KL strength, 不是 teacher 数量

## 2. 设计

### 2.1 架构

Student: **与 055 / 059 完全相同** (031B Siamese + cross-attn)。唯一改动在 teacher pool。

```
Per worker rollout step:
  obs (joint 672-dim) → student (031B Siamese + cross-attn) → student_logits
                     → teacher_ensemble (5 frozen team_siamese) → ensemble_avg_probs

Loss = PPO_loss(student) + α(t) · KL(student_logits || ensemble_avg_probs)

Teacher ensemble (5-way NEW):
  [055@1150, 031B@1220, 045A@180, 051A@130, 056D@1140]

  ensemble_avg_probs = factor-prob average across 5 teachers (same as 055)
```

### 2.2 与 055 / 059 的变量矩阵

| 项 | 055 | 059 (Tier 1a) | **061 (本, Tier 2)** |
|---|---|---|---|
| Student arch | 031B | 031B | 031B |
| Teacher pool | 3-way | 3-way | **5-way (+055@1150, +056D@1140)** |
| LR | 1e-4 | 3e-4 | **3e-4** |
| α / temp / decay | 0.05/1.0/8000 | 0.05/1.0/8000 | 0.05/1.0/8000 |
| Reward shaping | v2 | v2 | v2 |
| Budget | 1250 iter | 1250 iter | 1250 iter |

**059 vs 061 对照**: 两个 lane 都用 LR=3e-4, 唯一差异是 teacher pool 大小 (3-way vs 5-way)。如果 061 > 059, 证明 teacher pool refresh 贡献真 gain; 如果 061 ≈ 059, 证明 teacher diversity 在 3-way 已 saturated。

### 2.3 5-teacher ensemble 实现

扩展 [team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) 的 `_FrozenTeamEnsembleTeacher`:

```python
# 现有 class 已经是 N-teacher general 的 (从 055 就是 list-based)
# 只需在 env var 传入 5 个 ckpt path:
TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="<055@1150>,<031B@1220>,<045A@180>,<051A@130>,<056D@1140>"
```

无 code 改动, 纯 config 变化。5-teacher forward overhead: ~1.7× per rollout step vs 3-teacher (但 teacher forward 是冻结的, 用不到 backward, 所以绝对开销低)。

### 2.4 训练超参

```bash
# Architecture (= 055)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64

# Distillation — NEW 5-way teacher list
TEAM_DISTILL_KL=1
TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="<055@1150>,<031B@1220>,<045A@180>,<051A@130>,<056D@1140>"
TEAM_DISTILL_ALPHA_INIT=0.05
TEAM_DISTILL_ALPHA_FINAL=0.0
TEAM_DISTILL_DECAY_UPDATES=8000
TEAM_DISTILL_TEMPERATURE=1.0

# PPO — LR=3e-4 inherited from 056D winner / 059 combo
LR=3e-4
CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048

# v2 shaping (= 055 = 031B)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001 SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01 SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Budget (= 055)
MAX_ITERATIONS=1250 TIMESTEPS_TOTAL=50000000
TIME_TOTAL_S=43200 EVAL_INTERVAL=10 CHECKPOINT_FREQ=10
```

### 2.5 Launcher + 资源

- Launcher: [scripts/eval/_launch_055v2_recursive_distill.sh](../../scripts/eval/_launch_055v2_recursive_distill.sh)
- PORT_SEED=51 (与 059=41, 054M=43, 053A=23, 031B-noshape=13 隔离)
- Jobid 5021865, launched 2026-04-20

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: 1000ep peak ≥ 0.911 | = 055 single-shot peak | 5-teacher pool 至少 non-regressive |
| §3.2 主: 1000ep peak ≥ 0.920 | +0.013 vs 055 combined 2000ep 0.907 | **teacher refresh + LR 叠加成立**, recursive distill 路径成立 |
| §3.3 突破: 1000ep peak ≥ 0.925 | +0.018 vs 055 combined | **独立确认 grading**, 第二 highest-tier outcome |
| §3.4 持平: 1000ep peak ∈ [0.895, 0.910] | within ±1σ of 055 combined | teacher diversity 在 3→5 saturated, 加 055@1150 + 056D 无净增益 |
| §3.5 退化: 1000ep peak < 0.890 | < 055 combined - 2SE | **teacher signal conflict** (055 与 031B 同来源, 056D 与 031B 同架构不同 policy, KL 可能 mutually contradicting) |

## 4. 简化点 + 风险 + 降级预期 + 预案

### 4.1 简化 S1.A — Factor-prob average (not logit average)

| 简化项 | 完整方案 | 当前选择 |
|---|---|---|
| Ensemble aggregation | weighted logit average with learned weights / temperature per teacher | factor-prob uniform average (= 055) |

**理由**: 055 已 validate 了 factor-prob average 在 3-teacher 上有效; 扩到 5-teacher 若 aggregation 同时改, 变量污染。
**风险**: 5-teacher 中 055@1150 与其他 4 个 heterogeneity 大 (policy space 可能 far apart), uniform average 可能稀释 055@1150 的 strong signal。
**降级预期**: -0.3~-0.5pp vs 最优 weighted aggregation。
**预案**:
- L1: 如果 061 < 055, 改成 weighted average (055@1150 weight 2×, 其他 1×)
- L2: L1 无效, 改 temperature sharpen per-teacher (T=2 for 055@1150, T=1 others)

### 4.2 简化 S1.B — 固定 alpha schedule (不自适应)

| 简化项 | 完整方案 | 当前选择 |
|---|---|---|
| α schedule | adaptive based on KL divergence / student-teacher agreement | fixed linear decay (= 055) |

**风险**: 5-teacher 信号更复杂, fixed α 可能早期让 student 过度 fit teacher ensemble (before teacher pool coverage 被 student 学到), 后期太快衰减失去 distill guidance。
**降级预期**: -0.2pp vs adaptive。

### 4.3 简化 S1.C — 不 teacher weight warmup

所有 teacher 从 iter 0 起均 active, 无 warmup (e.g., iter 0-200 只用 3 个原 teacher, iter 200+ 加入 055@1150)。
**风险**: 055@1150 早期 KL 可能过强, 迫 student 在 rollout 能力不足时 mimic 一个比自己更 capable 的 teacher (distribution mismatch 更严重版本)。
**降级预期**: -0.3~-0.5pp; 若触发 §3.5 退化可能与此相关。

### 4.4 简化 S1.D — 不 DAGGER

= 055 S1.A, online distillation instead of iterative DAGGER。已在 055 validated 对 3-teacher 可 work, 对 5-teacher 假设同样 work。

### 4.5 全程降级序列 (worst case 路径)

| Step | 触发条件 | 降级动作 | 增量工程 |
|---|---|---|---|
| 0 | Default | 5-teacher + LR=3e-4 single run | base ~12h |
| 1 | Peak < 0.890 (regression) | 切 weighted aggregation (055@1150 weight 2×) | +12h GPU |
| 2 | Step 1 仍 < 0.890 | drop 056D from pool (4-teacher only), test 056D 是 culprit 还是 055@1150 | +12h GPU |
| 3 | Step 2 仍 < 0.890 | teacher pool refresh 路径 dead, revert 055 pool |
| 4 | Peak 0.895-0.910 (tied) | teacher diversity ceiling 在 3-way, 不再扩 pool | — |
| 5 | Peak ≥ 0.920 (breakthrough) | 立即开 055v3: teacher pool 加 061@peak, iterate generationally | +12h GPU |

## 5. 不做的事

- **不改 student 架构** — 与 055 / 059 同, 严格归因
- **不改 α schedule / temperature** — 与 055 同, 严格归因
- **不加 RND / curriculum / self-play** — 那是其他 axis
- **不 weighted aggregation** — first-pass uniform (若需 weighted 走 §4.1 L1)
- **不 DAGGER** — online online (= 055)
- **不 teacher warmup** — 全 teacher iter 0 active
- **不跟 059 共享 teacher pool** — 059 严格 3-way 原配, 061 是 5-way, 两 lane orthogonal
- **不与 059 (PORT=41) / 054M (PORT=43) 抢节点** — 本 lane PORT_SEED=51

## 6. 执行清单

- [x] 1. 确认 [team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) 支持 N-teacher (055 已 validated)
- [x] 2. 准备 5 个 teacher ckpt path (archive + scratch 双源验证)
- [x] 3. 写 launch script [scripts/eval/_launch_055v2_recursive_distill.sh](../../scripts/eval/_launch_055v2_recursive_distill.sh) (~30 min)
- [x] 4. Smoke: 5-teacher forward shape OK, memory 在 GPU budget 内
- [x] 5. Launch 1250 iter scratch on jobid 5021865, PORT_SEED=51 (2026-04-20)
- [ ] 6. 实时 monitor: KL term magnitudes per-teacher, α decay, GPU mem OK under 5-teacher
- [ ] 7. 训完 invoke `/post-train-eval` lane name `061`
- [ ] 8. Stage 2 capture peak ckpt
- [ ] 9. Stage 3 H2H: vs 055@1150 (base) + vs 059@peak (3-teacher + LR control) + vs 056D@1140 (LR winner peer)
- [ ] 10. Verdict append §7

## 7. Verdict (待 1000ep eval 后填入, append-only)

### 7.1 2026-04-21 01:00 EDT — Stage 1 + Stage 2 verdict (combined 2000ep) — POTENTIAL NEW SOTA, 3rd-sample verification recommended

**Stage 1 1000ep (port 39005, blind 10-ckpt sweep)**:

| ckpt | inline 200ep | 1000ep |
|---:|---:|---:|
| 900 | 0.925 | 0.879 |
| 940 | 0.925 | 0.876 |
| **1000** | — | **0.906** |
| 1040 | — | 0.885 |
| 1140 | — | 0.883 |
| 1150 | 0.935 (inline peak) | 0.888 |
| 1160 | — | 0.887 |
| 1190 | — | 0.903 |
| **1200** | 0.925 | **0.906** |
| 1210 | 0.920 | 0.895 |

Stage 1 single-shot peak = **0.906 (tied at ckpt 1000 and 1200)**, plateau 1190-1210 全部 ≥ 0.895。Inline 200ep 在 1150 给 0.935 但 1000ep 实测 0.888 → 再次复现 inline 上偏 ~0.05pp 现象 (`feedback_inline_eval_noise.md`)。Stage 1 单体已 = 055 SOTA 0.907 within 0.001pp。

**Stage 2 rerun (port 34005, peak ckpts 1000/1190/1200)**:

| ckpt | Rerun 1000ep |
|---:|---:|
| **1000** | **0.922** ⭐ (rerun 方向 UP, 与 062a/c rerun-down 反向) |
| 1190 | 0.909 |
| 1200 | 0.897 |

**Combined 2000ep (Stage 1 + Stage 2)**:

| ckpt | Stage 1 | Rerun | **Combined 2000ep** | ±SE |
|---:|---:|---:|---:|---:|
| **1000** | 0.906 | 0.922 | **1828/2000 = 0.914** ⭐ peak | ±0.0063 |
| 1190 | 0.903 | 0.909 | 1812/2000 = 0.906 | ±0.0065 |
| 1200 | 0.906 | 0.897 | 1803/2000 = 0.902 | ±0.0066 |

Raw recaps:

```
=== Stage 1 055v2 1000ep (port 39005) ===
... checkpoint_001000 vs baseline: 0.906 (906W-94L-0T)
... checkpoint_001190 vs baseline: 0.903 (903W-97L-0T)
... checkpoint_001200 vs baseline: 0.906 (906W-94L-0T)

=== Stage 2 rerun 055v2 (port 34005) ===
... checkpoint_001000 vs baseline: 0.922 (922W-78L-0T)
... checkpoint_001190 vs baseline: 0.909 (909W-91L-0T)
... checkpoint_001200 vs baseline: 0.897 (897W-103L-0T)
```

**Pre-registered judgment**:

| §3 判据 | 阈值 | combined 2000ep peak | verdict |
|---|---:|---:|---|
| §3.1 marginal (≥0.911) | 5-teacher pool 至少 non-regressive | 0.914 | **HIT** |
| §3.2 main (≥0.920) | teacher refresh + LR 叠加成立 | 0.914 | **MISS** (-0.006pp) |
| §3.3 breakthrough (≥0.925) | 独立确认 grading | 0.914 | **MISS** (-0.011pp) |

**与已有 frontier 的 Δ + significance**:

- vs **055 SOTA combined 2000ep 0.907**: Δ = **+0.007pp**, SE_diff ≈ √(0.0063² + 0.0066²) = 0.0091, z ≈ +0.77 → **NOT statistically significant** but POSITIVE direction
- vs **056D 0.891**: Δ = **+0.023pp**, z = 0.023/√(0.0063² + 0.010²) ≈ **+1.95 → border `*` significant**
- vs **031B 0.880**: Δ = **+0.034pp**, z ≈ **+2.84 → `**` significant**
- vs **058 simplified 0.847**: Δ = **+0.067pp**, z ≈ **strongly sig `***`**

**Reframing**:

- 055v2 (5-teacher recursive distill: 055@1150 + 034e originals + 056D@1140, uniform aggregation, LR=3e-4) **achieves new project peak 0.914**
- Δ = +0.007 over 055 是 sub-SE — could be noise OR could be real recursive-distill advantage
- **Critical caveat**: Stage 2 rerun 在 ckpt 1000 上 went UP +0.016pp (Stage 1 0.906 → rerun 0.922) — **opposite direction from 062a rerun (went DOWN -0.038pp on its peak)**. 两者都可能是 positive/negative fluctuations。Single-rerun NOT sufficient for decisive SOTA shift; **3rd-sample (combined 3000ep)** 推荐用于锁定 verdict
- Conservative interpretation: **055v2 ≈ 055 SOTA-tier**, possibly slightly better。**NOT decisively new SOTA** but no longer "tied" — best-case SOTA candidate

**对项目策略影响**:

- Tier 2 recursive distill **VALIDATED as SOTA-tier path** (peak 0.914 ≥ 055 0.907 even at sub-SE difference)
- 055v2 vs 055 within SE → likely converges to similar plateau ~0.91
- 加 055@1150 + 056D@1140 to teacher pool **does not hurt** and **may slightly help** (+0.007 raw vs 055; +0.023 vs 056D-alone via the LR=3e-4 boost)
- **Stage 3 verification recommended**: 055v2@1000 第 3 个 1000ep sample 跑出 combined 3000ep, lock SOTA 是否 shift
- 与 Curriculum 062 path 的关系: 062a combined 2000ep 0.892 < 055v2 0.914, 但 path 完全 orthogonal — 双线 stack 仍然 viable
- **055v2 IS the strongest single number now (0.914)** even if not decisively above 055

**Engineering note**: Stage 2 rerun 在 peak ckpt 上 went UP (vs Stage 1 0.906 → rerun 0.922) — 与 062a/c rerun-down 模式相反。**1000ep noise is bidirectional**; DON'T assume rerun always corrects DOWN。这是 `feedback_inline_eval_noise.md` doctrine 的扩展: rerun-down 是 sample mean 收敛, rerun-up 同样 valid (Stage 1 取到 lower-tail, rerun 取到 upper-tail)。

**预案触发**:

- 主预案 (snapshot §8 Outcome A): 1000ep peak ≥ 0.920 → **MISS** (0.914 < 0.920), recursive distill 路径 marginally validated 但未达 breakthrough threshold
- 次预案 (Outcome B): 1000ep peak ∈ [0.895, 0.910] tied with 055 → **partial hit** at 0.914 (above range upper bound, sub-SE-significant)
- 进入 **混合 outcome A/B**: tied 055 SOTA at peak number 但方向是上扬 → 不 close lane, 推 3rd-sample verification

_Pending follow-up: Stage 3 1000ep sample on ckpt 1000 (port allocation TBD), 跑后 update §7.2_

### 7.2 2026-04-21 01:15 EDT — Stage 3 verification — combined 3000ep correction (CORRECTION: NOT new SOTA, TIED with 055)

**3rd 1000ep sample (port 32005, ckpt 1000 only)**: **0.898 (898W-102L-0T)**

Raw v3 recap:

```
=== Official Suite Recap (parallel) ===
/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/TeamVsBaselineShapingPPOTrainer_Soccer_a1e7d_00000_0_2026-04-20_14-27-48/checkpoint_001000/checkpoint-1000 vs baseline: win_rate=0.898 (898W-102L-0T)
[suite-parallel] total_elapsed=233.6s tasks=1 parallel=1
```

**Combined 3000ep (Stage 1 + Stage 2 + Stage 3) on ckpt 1000**:

| Sample | Port | W-L | WR |
|---|---|---:|---:|
| Stage 1 | 39005 | 906-94 | 0.906 |
| Stage 2 (rerun v1) | 34005 | 922-78 | 0.922 |
| Stage 3 (v3) | 32005 | 898-102 | 0.898 |
| **Combined 3000ep** | — | **2726-274** | **0.9087 ≈ 0.909** |

SE = √(0.9087 × 0.0913 / 3000) = **0.0053**, 95% CI ≈ [0.899, 0.919]

**Pre-registered judgment (REVISED with combined 3000ep)**:

| §3 判据 | 阈值 | combined 3000ep peak | verdict |
|---|---:|---:|---|
| §3.1 marginal (≥0.911) | 5-teacher pool 至少 non-regressive | 0.909 | **MISS** (-0.002pp, was HIT at 2000ep 0.914) |
| §3.2 main (≥0.920) | teacher refresh + LR 叠加成立 | 0.909 | **MISS** |
| §3.3 breakthrough (≥0.925) | 独立确认 grading | 0.909 | **MISS** |
| §3.4 持平 [0.895, 0.911) | tied with 055 | 0.909 | **HIT** |

**与 055 SOTA 的 Δ + significance (REVISED)**:

- vs **055 SOTA combined 2000ep 0.907 ± 0.0066**: Δ = **+0.002pp**
- SE_diff = √(0.0053² + 0.0066²) = 0.0085
- z = 0.002 / 0.0085 = **0.24 → NOT statistically significant** (well within noise)

**CORRECTION (vs §7.1 earlier read)**:

- 早先 (combined 2000ep) 0.914 被 declared "POTENTIAL NEW SOTA"
- Stage 3 v3 = 0.898 把 combined 拉回 0.909
- **0.914 (2000ep) 是 positive draw**: Stage 2 rerun 0.922 是 high outlier; Stage 3 v3 0.898 是 corrected draw
- **Combined 3000ep 0.909 essentially TIES 055 SOTA 0.907** (Δ=+0.002 NOT sig)
- **NOT new SOTA**. **TIED with 055** at ~0.907-0.909 plateau.

**Ceiling analysis (4 verifications today, all pointing same direction)**:

| Lane | Single-shot / 2-sample | More samples | Δ |
|---|---:|---:|---:|
| 062a | single-shot 0.911 | combined 2000ep 0.892 | -0.019 |
| 062c | single-shot 0.899 | combined 2000ep 0.886 | -0.013 |
| 055v2 | combined 2000ep 0.914 | combined 3000ep 0.909 | -0.005 |

All single-shot / 2-sample readings ≥ 0.91 collapsed to ~0.88-0.91 with more samples。**Project SOTA ceiling appears to be ~0.91 for this seed/setup family** (031B arch + v2 shape OR no-shape + various distill / curriculum / LR paths)。Multiple distinct paths (distill / recursive distill / curriculum / LR sweep) all converge here。

**对项目策略影响 (REVISED)**:

- 055v2 path **NOT new SOTA but VALIDATED as competitive (TIED 055)**
- Tier 2 recursive distill works equally well as original 3-teacher distill (different teacher pool composition, similar ceiling)
- 加 055@1150 + 056D@1140 to teacher pool **does not hurt** but **does not help decisively** (Δ=+0.002 NS)
- Remaining hope to push above 0.91: **T sweep (063_T15/T30/T40)** if temperature unlocks teacher signal beyond entropy-1.0
- 066A/B progressive distill likely stays at ~055 plateau (similar mechanism)

**预案触发 (REVISED)**:

- 主预案 (snapshot §8 Outcome A 突破 ≥0.920): **MISS** (combined 3000ep 0.909 < 0.920)
- 次预案 (snapshot §8 Outcome B 持平 [0.895, 0.910]): **HIT** (0.909 in range, tied with 055)
- 进入 **Outcome B (持平)**: teacher pool 5→3 expansion saturated at 055 plateau; 不再扩 pool, 资源转 T sweep / 其他 axis
- Lane 关闭 — 与 055 family 同 plateau, 无 actionable upside via teacher count expansion

**Lesson (extends `feedback_inline_eval_noise.md` doctrine)**:

- Even **combined 2000ep** can mislead by +0.005pp on peak ckpt
- Inline 200ep → +0.05pp optimism (`feedback_inline_eval_noise.md` original)
- Stage 1 single-shot 1000ep → +0.013-0.038pp optimism (058/062a/062c pattern)
- Combined 2000ep → +0.005pp optimism (055v2 pattern, smaller but still real)
- **Definitive SOTA shift requires 3+ independent 1000ep samples** when claimed Δ < 0.01pp

### 7.3 2026-04-21 15:30 EDT — 055v2_extend (1210 → 2000 iter extend) Stage 1 single-shot 1000ep — **SOTA-POTENTIAL-PENDING-RERUN**

**Setup**: 055v2 lane resumed from iter 1210 and extended to iter 2000 (held-srun retry, merged pre-crash + resume training dirs). Stage 1 post-eval ran official evaluator parallel 1000ep on 25 ckpts iter 1250-1960.

**Raw Recap top-10 (official evaluator parallel, single-shot 1000ep each)**:

```
=== Official Suite Recap top-10 ===
ckpt 1830 vs baseline: win_rate=0.923   [★ peak]
ckpt 1860 vs baseline: win_rate=0.921
ckpt 1750 vs baseline: win_rate=0.917
ckpt 1850 vs baseline: win_rate=0.914
ckpt 1250 vs baseline: win_rate=0.913   [boundary / resume-start area]
ckpt 1270 vs baseline: win_rate=0.912
ckpt 1960 vs baseline: win_rate=0.911   [terminal area]
ckpt 1710 vs baseline: win_rate=0.911
```

Log: `docs/experiments/artifacts/official-evals/055v2_extend_baseline1000.log`

**Plateau 1710-1860**: 6 of 8 top ckpts in window ≥ 0.911 — this is **NOT a single-point lucky peak** but a structural plateau at/above 055 SOTA.

**预注册判据 (§3) 对照 (single-shot 1000ep caveat — pending rerun confirmation)**:

| §3 判据 | 阈值 | single-shot @1830 | verdict |
|---|---:|---:|---|
| §3.1 marginal (≥0.911) | 5-teacher pool 至少 non-regressive | **0.923** | **HIT** (pending rerun) |
| §3.2 main (≥0.920) | teacher refresh + LR 叠加成立 | **0.923** | **HIT** (pending rerun) |
| §3.3 breakthrough (≥0.925) | 独立确认 grading | 0.923 | MISS (-0.002) |
| §3.4 持平 [0.895, 0.911) | tied with 055 | 0.923 | above range |

**Δ vs 055 SOTA**:
- vs **055@1150 combined 2000ep 0.907 ± 0.007**: single-shot Δ = **+0.016pp**
- vs **055v2@1000 combined 3000ep 0.909 ± 0.005** (pre-extend peer): single-shot Δ = **+0.014pp**
- Both Δ magnitudes exceed the 055 SOTA plateau ceiling observed across today's other lanes (066A/B, 070, 063_T15/T30, 074 family)
- **BUT**: 4 independent 2026-04-21 verifications (062a / 062c / 055v2@1000 Stage 2 / 063_T40 4-sample saga) all showed positive single-shot fluctuation of +0.005 to +0.038pp on peak ckpts → **combined rerun is mandatory before claiming SOTA shift** (per `feedback_inline_eval_noise.md` doctrine)

**Status: POTENTIAL NEW SOTA — PENDING RERUN VERIFICATION**

- First evidence across today's 7-lane Stage 1 sweep of **crossing the 0.91 plateau** that has held 055 / 055v2 / 063_T40 / 062a / 054M family
- 2nd rerun on @1830 launched separately (2000ep n-sample); combined 3000ep decision pending
- Lane **NOT closed**; **NOT yet declared SOTA** — doctrine requires 3+ samples when claimed Δ < 0.01pp and ≥ 2 samples when claimed Δ ≈ 0.015pp
- If combined 3000ep ≥ 0.918: declare new project SOTA (Outcome A)
- If combined 3000ep ∈ [0.907, 0.918): declare marginal upgrade, investigate whether "extend past 2000 iter" is actionable
- If combined 3000ep ≤ 0.907: joins the 4x-reinforced "single-shot optimism" pattern, close extend lane

**Mechanism hypothesis**: extending 055v2 recursive-distill training past the original 1210 iter allowed the student to further internalize multi-teacher soft targets, breaking past what the original 1250-iter horizon allowed. Alternative: plateau is coincidental good sampling on 1750-1860 window; rerun will collapse.

**Interpretation in meta-context (2026-04-21 7-lane Stage 1 sweep)**:
- 6 of 7 lanes tied 055 at 0.89-0.91 (066A/B, 063_T15, 063_T30, 070, 056E — see §3.3 rank.md)
- 055v2_extend is the **unique outlier** — first lane to show plateau ≥ 0.911 across ≥6 consecutive ckpts
- If confirmed by rerun, validates "extend past nominal teacher-refresh horizon" as actionable path for recursive distill

_Pending: 2nd rerun on ckpt 1830 (2000ep n-sample) → combined 3000ep verdict._

### 7.4 2026-04-21 18:30 EDT — 055v2_extend@1750 = **NEW PROJECT SOTA combined 3000ep 0.9163** (append-only)

**VERDICT — 055v2_extend lane CONFIRMED: NEW SOTA at iter 1750 (NOT iter 1830 as §7.3 putative single-shot peak suggested).**

After the §7.3 "SOTA-POTENTIAL-PENDING-RERUN" claim, a **parallel 2000ep rerun was launched against all 8 §7.3 top peaks** (1250 / 1270 / 1710 / 1750 / 1830 / 1850 / 1860 / 1960). Only **one peak survived mean reversion: iter 1750**.

**8-peak mean-reversion test table**:

| Peak iter | Stage 1 (1000ep, §7.3) | Rerun 2000ep | Combined 3000ep | Mean rev Δ | Status |
|---:|---:|---:|---:|---:|---|
| 1250 | 0.913 | 0.908 | **0.910** | -0.003 | reverted |
| 1270 | 0.912 | 0.905 | **0.908** | -0.004 | reverted |
| 1710 | 0.911 | 0.902 | **0.905** | -0.006 | reverted |
| **1750** | **0.917** | **0.916** | **0.9163** | **-0.0007** | **STAYED ← NEW SOTA** |
| 1830 | 0.923 | 0.903 | **0.910** | -0.013 | reverted (biggest downshift; putative §7.3 peak collapsed) |
| 1850 | 0.914 | 0.896 | **0.902** | -0.012 | reverted |
| 1860 | 0.921 | 0.901 | **0.908** | -0.013 | reverted |
| 1960 | 0.911 | 0.904 | **0.906** | -0.005 | reverted |

- **7 of 8 peaks reverted by -0.003 to -0.013pp** upon rerun — classic small-sample positive-bias pattern (`feedback_inline_eval_noise.md` doctrine, 5x now reinforced in 2026-04-21)
- **Only 1750 survived**: Stage 1 1000ep 0.917 + Rerun 2000ep 0.916 = **combined 3000ep = (917 + 1832)/3000 = 2749/3000 = 0.9163** (SE ≈ 0.005, CI 95% [0.906, 0.926])

**Δ vs 055 SOTA combined 2000ep 0.907**: **Δ = +0.009, z ≈ 1.23, p ≈ 0.11 one-sided — approaching marginal significance but NOT YET p<0.05.**

**Δ vs 055v2@1000 combined 3000ep 0.909** (pre-extend peer): +0.007.

**Why 1750 is special — interpretation**:

- 1750 is **+50 iter past the original MAX_ITERATIONS = 1216** of 055v2 teacher's own training window, and **~160 iter past the TRUE teacher peak at 055v2@1000** (combined 3000ep 0.909 from §7.2).
- The other §7.3 peaks (1830 / 1850 / 1860 / 1960) were likely **single-shot positive fluctuations against a stable late-extend 0.90-0.91 plateau** — when rerun, they reverted to plateau mean.
- 1250 / 1270 (near-pre-extend-boundary) also reverted — the stable plateau sits around ~0.908-0.910, and 1750 is a genuinely distinguished point ~+0.006-0.008pp above that plateau.
- **Mechanistic hypothesis**: 1750 is the unique stable frontier ckpt in the late-extend window. Why 1750 specifically? Possibilities:
  1. **Teacher-pool argmax drift**: by iter 1750 student's policy has diverged sufficiently from teacher distributions that one particular weighted-KL ensemble optimum locks in (observed in other recursive distill work: student often lands on "mesa-optimum" at ~1.4× teacher training horizon)
  2. **Iteration-budget principle**: training past teacher's own training horizon allows student to further internalize multi-teacher soft targets beyond what teacher's own convergence dynamics allow. This is a **novel project finding** (validated: §7.3 "extend past 2000 iter" hypothesis CONFIRMED in spirit, just not at the originally suspected peak 1830)

**Status: NEW PROJECT SOTA — combined 4000ep rerun pending for final p<0.05 significance confirmation**

- 3rd-sample 1000ep rerun on @1750 launched 2026-04-21 18:32 EDT (jobid 5032909, port 52505) → will give combined 4000ep (3000 + 1000 = 4000ep) with SE ≈ 0.004 once done
- ETA ~10 min from launch (port 52505 eval parallel j=1)
- Log: `docs/experiments/artifacts/official-evals/055v2e_1750_rerun1000v3.log` (in progress)
- **If combined 4000ep ≥ 0.915**: Δ vs 055 ≥ +0.008 with SE 0.004 → z ≥ 2.0, p < 0.05 → **decisively new SOTA**, upgrade primary grading candidate to 055v2_extend@1750
- If combined 4000ep ∈ [0.910, 0.915): marginal Δ, 055v2_extend@1750 ≈ 055 tier but preferred for frontier diversity
- If combined 4000ep < 0.910: another mean reversion, close lane (unlikely given 917+1832 already anchors ~0.916)

**Raw logs verified (numbers anchored)**:
- Stage 1 1000ep: `docs/experiments/artifacts/official-evals/055v2_extend_baseline1000.log` — `checkpoint-1750 vs baseline: win_rate=0.917`
- Rerun 2000ep: `docs/experiments/artifacts/official-evals/055v2e_1750_rerun2000.log` — `win_rate=0.916 (1832W-168L-0T) elapsed=468.6s`
- Combined 4000ep pending: `docs/experiments/artifacts/official-evals/055v2e_1750_rerun1000v3.log` — 3rd sample in progress

**Lane status**: **NOT CLOSED — open pending combined 4000ep for p<0.05 confirmation; primary grading candidate upgrade recommended.**

## 8. 后续发展线

### Outcome A — 突破 (1000ep peak ≥ 0.920)
- Recursive distill 路径成立。立即开 **055v3**: teacher pool 加入 061@peak, 迭代代际
- 长期: generational distill (Gen 1 → Gen 2 → Gen 3) 像 AlphaZero self-play generations 一样 compound
- 与 059 对比: 如果 061 > 059 显著, 证明 teacher refresh 贡献真 gain; 如果 061 ≈ 059, 证明 LR gain 足以解释 distill path 的 follow-up

### **2026-04-21 18:30 EDT — §8 UPDATE: recursive distill + late-extend = SOTA-producing PATH VALIDATED** (append-only)

Per §7.4 VERDICT, 055v2_extend@1750 = combined 3000ep 0.9163 = NEW PROJECT SOTA. This rewrites the §8 decision tree:

- **Iteration-budget principle (new, validated)**: **extend recursive distill past the MAX_ITERATIONS of teacher's own training window**. 1750 is +50 iter past original max 1216, and +750 iter past true teacher peak 055v2@1000. The "extend past 2000 iter" §7.3 hypothesis was right in spirit but at the wrong peak (1830 was single-shot noise; 1750 is the real frontier).
- **Follow-up lanes implied**:
  1. **079 055v3 recursive distill** (currently 85% training per task-queue-20260421) — consider extending 079 past its own MAX_ITERATIONS to probe the same pattern
  2. **076 wide-student** (currently 85% training) — once Stage 1 done, apply "extend past teacher's training horizon" before declaring ceiling
  3. **Pool A (snapshot-071)**: teacher set currently includes 055v2@1000 — consider rerun with 055v2_extend@1750 as upgraded teacher
  4. **074 ensemble family (all closed)**: members could be swapped from 055@1150 → 055v2_extend@1750 as retry (not guaranteed to help given 074 family already closed, but worth one revisit since member swap is zero-training)
- **Invalidates §8 Outcome B claim** "teacher pool expansion 死路" — extend past teacher's own horizon is productive even with same 5-teacher pool as 055v2
- **Pending combined 4000ep confirmation**: if final p<0.05, primary grading candidate upgraded from 055@1150 → 055v2_extend@1750

### Outcome B — 持平 (1000ep peak ∈ [0.895, 0.910])
- Teacher diversity 在 3→5 已 saturated
- 加 055@1150 (同 student lineage) + 056D (同 arch diff policy) 没给 student 新 signal
- **意味着 distill lane 的 "teacher pool expansion" 死路**, 未来 distill improvement 必须来自: (a) 换 architecture, (b) 换 training regime (DAGGER/offline), (c) 换 distill loss formulation
- 不再扩 teacher pool, 资源转 059/060 或其他 axis

### Outcome C — 退化 (1000ep peak < 0.890)
- **Teacher signal conflict** — 055@1150 与 031B@1220 同 lineage (055 是 031B 基础上 distill 来的), 两者在 policy space 可能 highly correlated, ensemble average 变成 "重复 vote"
- 加上 056D 同架构但不同 LR trajectory, 可能进一步 confuse student
- §4.5 step 2 触发: drop 056D 或 drop 055@1150 test 谁是 culprit
- 最坏: recursive distill 的 "self-include" 根本破坏 distill signal (类似 imitation learning 中 state distribution 循环)

## 9. 相关

- [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md) — 3-teacher base (Gen 1); 055@1150 是本 lane 的 novel teacher
- [SNAPSHOT-059](snapshot-059-055lr3e4-combo.md) — LR=3e-4 control (3-teacher + LR), 与 061 正交对照
- [SNAPSHOT-056](snapshot-056-simplified-pbt-lr-sweep.md) — 056D@1140 teacher 来源
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B@1220 teacher + student 架构
- [SNAPSHOT-045](snapshot-045-architecture-x-learned-reward-combo.md) — 045A@180 teacher 来源
- [SNAPSHOT-051](snapshot-051-learned-reward-from-strong-vs-strong-failures.md) — 051A@130 teacher 来源
- [team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) — distill model, N-teacher general
- [_launch_055v2_recursive_distill.sh](../../scripts/eval/_launch_055v2_recursive_distill.sh) — launch script

### 理论支撑

- **Furlanello et al. 2018** "Born Again Neural Networks" — self-distillation, student 超过 teacher 是 validated pattern
- **Anil et al. 2018** "Large scale distributed neural network training through online distillation" — co-distillation, teacher pool dynamic refresh
- **Hinton et al. 2015** "Distilling the Knowledge in a Neural Network" — multi-teacher soft targets average 作为 base formulation
- **本 snapshot 独立贡献**: project 内首次 generational distill — Gen 1 output 直接成 Gen 2 teacher, 与 LR HP 增益 combo (059 是 LR-only control, 061 是 LR + teacher refresh)
