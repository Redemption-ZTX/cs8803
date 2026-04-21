## SNAPSHOT-059: 055 + lr=3e-4 combo (Tier 1a)

- **日期**: 2026-04-20
- **负责人**: Self
- **状态**: 训练中 (lane open) — jobid 5022392, PORT_SEED=41, iter 9+/1250, ETA ~10h36m

## 0. 背景与定位

### 0.1 项目当前 ceiling 拓扑 (2026-04-20)

| 类别 | SOTA | 评注 |
|---|---|---|
| Single-model scratch ensemble-distill | **055@1150 = 0.907 combined 2000ep** | 项目当前 single-model SOTA, 见 [snapshot-055 §7.13](snapshot-055-distill-from-034e-ensemble.md#713-2026-04-20-1325-edt-baseline-rerun-v2--0551150-combined-2000ep--0907-supersedes-single-shot-0911) |
| Single-model LR HPO (lr=3e-4) | 056D@1140 = 0.891 marginal | tied 031B, +0.009 vs 055 baseline lr=1e-4; 见 [snapshot-056](snapshot-056-simplified-pbt-lr-sweep.md) |
| Architecture axis | 031B@1220 = 0.882 (0.880 combined) | 054 MAT-min 0.880 tied, axis saturated at this param scale |
| Ensemble (probability avg) | 034E-frontier = 0.890 | "非智力提升", 被 055 distill 超越 |

**Gap**: 055@1150 combined 2000ep 0.907 是当前唯一越过 0.900 grading threshold 的 verified 单模型读数。但训练 hyperparameters 继承自 031B (LR=1e-4)，**没有吃到 056D lr=3e-4 的增益**。

### 0.2 Combo 假设的理论定位

[SNAPSHOT-056](snapshot-056-simplified-pbt-lr-sweep.md) 在 2026-04-20 decisive 地证明 LR=3e-4 在 031B 架构 + v2 shaping 下给出 +0.009 marginal gain (056D 0.891 vs 031B baseline 0.882)。**单独增益不算大**，但 056D 的收敛 trajectory (peak 出现在 iter 1140, 与 031B peak 1220 相似 window) 说明 LR=3e-4 **没破坏训练稳定性**，只是略加速收敛。

**Combo 假设**: 既然 distill loss + PPO env reward 是两条独立的 gradient source，LR 对两条 source 都起 scaling 作用。如果 055 的 distill path 是 under-trained (kill event cut at iter ~1020 第一轮训练, round-2 restart 接 1240 iter)，**更高 LR 可能让 distill signal 更快 saturate**，在 1250 iter budget 内 peak 更高。

## 1. 核心假设

### H_059

> 055 distill 同架构 + 同 teacher ensemble + LR 从 1e-4 → 3e-4 (单变量改动)，**1000ep peak ≥ 0.920** (+0.013 vs 055 combined 2000ep 0.907)，突破 055 的 2000ep ceiling。
>
> Stretch: ≥ 0.925，grading 门槛有 absolute margin。
>
> Worst tolerated: 0.895-0.910 (tied 055)，意味着 LR 在 distill path 下 saturation 已在 lr=1e-4 附近，post-distill LR ceiling 无空间。

### 子假设

- **H_059-a**: 056D 的 +0.009 LR gain 不是 031B-specific 偶然, 而是 general-purpose PPO optimizer 收敛加速 — distill + LR 的两个 gain 在第一阶正交 (additive)
- **H_059-b**: 055 的 kill-cut + round-2 restart 导致 distill 轨迹不光滑, 更高 LR 能让 round-2 跑到更纯粹的 distill convergence
- **H_059-c**: distill KL term 与 PPO clip 的相互作用对 LR 不 catastrophically sensitive — α=0.05 (init) 线性衰减 → 8000 updates 是与 LR 正交的时间 axis

## 2. 设计

### 2.1 总架构 (= 055, 一字不变)

```
Per worker rollout step:
  obs (joint 672-dim) → student (031B Siamese + cross-attn) → student_logits
                     → teacher_ensemble (3 frozen team_siamese) → ensemble_avg_probs

Loss = PPO_loss(student) + α(t) · KL(student_logits || ensemble_avg_probs)

α(t) schedule: 0.05 → 0.0 linear decay over first 8K SGD updates

Teacher ensemble (identical to 055):
  [031B@1220, 045A@180, 051A@130]  ← 3-way 034E ensemble members
```

### 2.2 与 055 的唯一差异

| 项 | 055 | 059 | 来源 |
|---|---|---|---|
| LR | 1e-4 | **3e-4** | 056D 胜出 |
| 其他所有 hyperparams | — | **identical to 055** | 严格单变量 |

**关键**: 不加 055@1150 入 teacher pool (那是 Tier 2, 放在 [snapshot-061](snapshot-061-055v2-recursive-distill.md) 做 5-teacher recursive distill)。本 snapshot 保持 teacher = {031B@1220, 045A@180, 051A@130} 3-way 原配。

### 2.3 训练超参

```bash
# Architecture (= 055)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64

# Distillation (= 055)
TEAM_DISTILL_KL=1
TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="<031B@1220>,<045A@180>,<051A@130>"
TEAM_DISTILL_ALPHA_INIT=0.05
TEAM_DISTILL_ALPHA_FINAL=0.0
TEAM_DISTILL_DECAY_UPDATES=8000
TEAM_DISTILL_TEMPERATURE=1.0

# PPO — ONLY CHANGE
LR=3e-4                                 # was 1e-4 in 055
CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048

# v2 shaping (= 055, = 031B)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001 SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01 SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Budget (= 055)
MAX_ITERATIONS=1250 TIMESTEPS_TOTAL=50000000
TIME_TOTAL_S=43200 EVAL_INTERVAL=10 CHECKPOINT_FREQ=10
```

### 2.4 Launcher + 资源

- Launcher: [scripts/eval/_launch_055lr3e4_distill_scratch.sh](../../scripts/eval/_launch_055lr3e4_distill_scratch.sh)
- PORT_SEED=41 (远离 054M=43, 055v2=51, 053A=23, 031B-noshape=13)
- Jobid 5022392, launched 2026-04-20 (iter 9+/1250 at snapshot write time)

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: 1000ep peak ≥ 0.911 | = 055 single-shot peak | distill + LR combo 至少 non-regressive |
| §3.2 主: 1000ep peak ≥ 0.920 | +0.013 vs 055 combined 2000ep 0.907 | **LR gain stacks on distill**, combo 成立 |
| §3.3 突破: 1000ep peak ≥ 0.925 | +0.018 vs 055 combined | **独立确认 grading 门槛**, combo path 强 |
| §3.4 持平: 1000ep peak ∈ [0.895, 0.910] | within ±1σ of 055 combined | combo 没坏也没好, LR ceiling 在 distill 后 saturated |
| §3.5 退化: 1000ep peak < 0.895 | < 055 combined - SE | **LR=3e-4 破坏 distill signal**, 降级到 lr=1e-4 |

## 4. 简化点 + 风险 + 降级预期 + 预案

### 4.1 简化 S1.A — 单变量改 LR (不 sweep)

| 简化项 | 完整方案 | 当前选择 | 节省 |
|---|---|---|---|
| LR sweep | {1e-4, 2e-4, 3e-4, 5e-4, 1e-3} × 5 run | 单点 3e-4 | 4 个 GPU slot |

**风险**: LR=3e-4 可能不是 post-distill optimum；真 optimum 可能在 2e-4 (distill signal 比纯 PPO 更需要稳定 LR)。
**降级预期**: 单点 LR 相对理论 optimum -0.3~-0.5pp。
**预案**:
- L1: 如果 059 peak < 055@1150 0.907, 立即开 LR=2e-4 对照 run
- L2: L1 仍 < 0.907, 回 LR=1e-4 declare "distill path LR 已在 1e-4 saturated"

### 4.2 简化 S1.B — 不 sweep α / temperature

与 055 同问题, 但本 snapshot 严格单变量, α=0.05 / temp=1.0 完全继承 055。任何 α/temp 变动推到 [059X] follow-up。

### 4.3 简化 S1.C — 不混入其他 combo

本 snapshot **仅 distill + LR**, 不加 RND / curriculum / self-play。

**理由**: 保持单变量归因 — 若失败, 明确指向 "LR 在 distill path 下破坏训练"; 若成功, 直接 claim "LR gain stacks on distill, Tier 1 combo 成立"。

### 4.4 简化 S1.D — Teacher pool 与 055 相同

不加 055@1150 入 teacher pool。那是 [snapshot-061 055v2](snapshot-061-055v2-recursive-distill.md) 的职责。

### 4.5 全程降级序列 (worst case 路径)

| Step | 触发条件 | 降级动作 | 增量工程 |
|---|---|---|---|
| 0 | Default | 055 + LR=3e-4, single run | base ~10h |
| 1 | Peak < 0.895 (regression) | LR=2e-4 rerun (S1.A L1) | +10h GPU |
| 2 | Step 1 仍 < 0.895 | declare LR=1e-4 is distill-optimal, revert | — |
| 3 | Peak 0.895-0.910 (tied) | declare LR ceiling saturated post-distill, move to 055v2 Tier 2 | — |
| 4 | Peak ≥ 0.920 (breakthrough) | 立即 stack 进 055v3 (distill + LR + 更多 teacher) | +10h GPU |

## 5. 不做的事

- 不改 student 架构 (= 055 / = 031B student)
- 不改 teacher pool (= 055 3-way, 不加 055@1150 — 那是 061)
- 不 sweep α / temperature (= 055 固定点)
- 不混入 reward shaping 改动 (= 055, = 031B v2)
- 不与 054M (PORT_SEED=43) / 055v2 (PORT_SEED=51) 抢节点 (本 lane PORT_SEED=41)

## 6. 执行清单

- [x] 1. 确认 055 code path 支持 LR env var (= train_ray_team_vs_baseline_shaping.py 已有 LR env var)
- [x] 2. 写 launch script [scripts/eval/_launch_055lr3e4_distill_scratch.sh](../../scripts/eval/_launch_055lr3e4_distill_scratch.sh) (~30 min)
- [x] 3. Launch 1250 iter scratch on jobid 5022392, PORT_SEED=41 (2026-04-20)
- [ ] 4. 实时 monitor: KL decay normal, α 生效, first iter PPO stable under LR=3e-4
- [ ] 5. 训完 invoke `/post-train-eval` lane name `059`
- [ ] 6. Stage 2 capture peak ckpt (500ep)
- [ ] 7. Stage 3 H2H: vs 055@1150 (base combo) + vs 056D@1140 (LR source) + vs 031B@1220 (arch base)
- [ ] 8. Verdict append §7

## 7. Verdict (待 1000ep eval 后填入, append-only)

### 7.1 (2026-04-21 00:30 EDT) — Stage 1 1000ep verdict — **FAIL §3.1/§3.2/§3.3, HIT §3.4 持平区, Tier 1a 关闭**

**Data (10 ckpts × 1000ep, official suite parallel, total_elapsed=511.6s, parallel=7)**:

| ckpt | inline 200ep | 1000ep verified | Δ optimism |
|---:|---:|---:|---:|
| 810 | 0.915 | 0.874 | -0.041 |
| 1020 | — | 0.875 | — |
| **1030** | **0.945** (inline peak) | 0.893 | **-0.052** |
| 1040 | — | 0.892 | — |
| 1100 | — | 0.878 | — |
| 1130 | — | 0.876 | — |
| 1140 | 0.940 | 0.884 | -0.056 |
| 1150 | — | 0.895 | — |
| 1210 | 0.925 | 0.884 | -0.041 |
| **1220** | — | **0.898** (peak) | — |

**Verdict 对照预注册判据 §3**:
- §3.1 marginal (≥0.911): **FAIL** (0.898 < 0.911)
- §3.2 主 (≥0.920): **FAIL**
- §3.3 突破 (≥0.925): **FAIL**
- §3.4 持平 (∈ [0.895, 0.910]): **HIT** (0.898 落在 [0.895, 0.910] 区间) — Tier 1a 落入「持平 055 但 NOT 显著 better」区
- §3.5 退化 (< 0.895): NOT triggered (0.898 ≥ 0.895)

**Cross-lane 对照**:
- vs 055 SOTA combined 2000ep 0.907: Δ=**-0.009pp,within SE, NOT sig below SOTA**
- vs 056D LR=3e-4 standalone 0.891: Δ=**+0.007pp, NOT sig**
- vs 031B 0.882: Δ=**+0.016pp, NOT sig**

**关键 reframing — combo 1+1≠>2**:
- LR=3e-4 + distill 没产生 1+1>2 effect。combo peak 0.898 ≈ Max(distill alone 0.907, LR alone 0.891)。
- distill axis 拥有 ~8pp gain (vs 031B 0.880 → 055 0.907)
- LR axis isolated 只 ~1pp gain (031B 0.880 → 056D 0.891)
- LR axis combined with distill 几乎归零 (~+0.007pp = noise)
- **Distill 是真正 carry path,LR 是 saturated**

**对后续策略影响**:
- Tier 1a 此 lane 关闭 (LR axis 在 distill 上无叠加)
- Tier 3a (055-temp T 变体) 仍是最 promising path — temperature 是 distill 内部参数,可能 "+1+1>2"
- Tier 2 055v2 (5-teacher ensemble + LR=3e-4) — pending 完成,看是否 ensemble 多样性 + LR 共贡献
- 056E/F 上限可能也只达 0.89 (LR axis ceiling at ~0.89 for any LR)

**Inline eval noise 复现**: Inline 200ep 再次 -0.04~-0.06pp 高估 (符合 [`feedback_inline_eval_noise`](../../.claude/projects/-home-hice1-wsun377-Desktop-cs8803drl/memory/feedback_inline_eval_noise.md) memory)。1030 inline 0.945 vs 1000ep 0.893 = -0.052; 1140 inline 0.940 vs 1000ep 0.884 = -0.056。再次确认 inline 200ep 不是 SOTA verdict。

**Raw recap**:
```
=== Official Suite Recap (parallel) ===
.../checkpoint_000810/checkpoint-810 vs baseline: win_rate=0.874 (874W-126L-0T)
.../checkpoint_001020/checkpoint-1020 vs baseline: win_rate=0.875 (875W-125L-0T)
.../checkpoint_001030/checkpoint-1030 vs baseline: win_rate=0.893 (893W-107L-0T)
.../checkpoint_001040/checkpoint-1040 vs baseline: win_rate=0.892 (892W-108L-0T)
.../checkpoint_001100/checkpoint-1100 vs baseline: win_rate=0.878 (878W-122L-0T)
.../checkpoint_001130/checkpoint-1130 vs baseline: win_rate=0.876 (876W-124L-0T)
.../checkpoint_001140/checkpoint-1140 vs baseline: win_rate=0.884 (884W-116L-0T)
.../checkpoint_001150/checkpoint-1150 vs baseline: win_rate=0.895 (895W-105L-0T)
.../checkpoint_001210/checkpoint-1210 vs baseline: win_rate=0.884 (884W-116L-0T)
.../checkpoint_001220/checkpoint-1220 vs baseline: win_rate=0.898 (898W-102L-0T)
[suite-parallel] total_elapsed=511.6s tasks=10 parallel=7
```

## 8. 后续发展线

### Outcome A — 突破 (1000ep peak ≥ 0.920)
- distill + LR 正交叠加证成, 立即 stack 进 future distill variants (055v3 = 059 + 061 pool)
- 试 LR=5e-4 看 LR ceiling 在哪
- 长期: 与 055v2 recursive distill 汇合成 single strongest path

### Outcome B — 持平 (1000ep peak ∈ [0.895, 0.910])
- LR ceiling 在 distill path 下 saturated at 1e-4 neighborhood
- Combo hypothesis 被证伪, follow-up 往 teacher diversity (Tier 2 061) 走
- 不再 sweep LR — 死 axis

### Outcome C — 退化 (1000ep peak < 0.895)
- LR=3e-4 破坏 distill signal (KL term unstable under large PPO steps)
- revert LR=1e-4 作为 distill path 的 locked hyperparam
- 本 combo 路径关闭, resource 集中到 055v2 / 其他 axis

## 9. 相关

- [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md) — distill base, combo 的 structural parent
- [SNAPSHOT-056](snapshot-056-simplified-pbt-lr-sweep.md) — LR=3e-4 winner 来源
- [SNAPSHOT-061](snapshot-061-055v2-recursive-distill.md) — Tier 2 5-teacher recursive distill (含 055@1150)
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B student 架构
- [team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) — distill model
- [_launch_055lr3e4_distill_scratch.sh](../../scripts/eval/_launch_055lr3e4_distill_scratch.sh) — launch script

### 理论支撑

- **Hinton et al. 2015** "Distilling the Knowledge in a Neural Network" — KL distill base
- **Smith & Topin 2017** "Super-Convergence" — high LR 在 PPO 上的 convergence acceleration
- **Andrychowicz et al. 2021** "What Matters in On-Policy RL" — LR 对 PPO 是 first-order hyperparameter
- **056D 本地实证**: LR=3e-4 +0.009 marginal on 031B architecture + v2 shaping
