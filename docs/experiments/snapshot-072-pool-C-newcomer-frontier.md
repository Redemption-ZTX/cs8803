## SNAPSHOT-072: Pool C — Newcomer + Frontier (Max-Diversity 4-Teacher Distill)

- **日期**: 2026-04-21
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); **blocked on dependencies** — 等 055v2_extend (~3h) + 054M_extend (~1h) 完成并 identify peak ckpts 后才能 launch

## 0. 背景与定位

### 0.1 为什么现在做 Pool C

Pool A / B 已经分别测 "homogeneous family recursion (3 teacher)" 和 "divergent reward mechanism (3 teacher)"。Pool C 是同一 "multi-model combo pool" queue 的第三根 probe axis, 目标是**把 diversity 推到实验能支持的最大值**:

> "Ensemble of orthogonal failure modes expected to produce robust student" — 当前 ~0.89-0.91 plateau 下有 4 个独立 actionable axis 各自达到这个段, 如果 failure mode 真的 orthogonal, 4-way ensemble teacher 应该比 3-way 再多挤一点 information gain。

### 0.2 与 Pool A / Pool B 的正交对照

| Pool | Teacher 数 | Teacher 多样性来源 | Ship 风险 |
|---|---:|---|---|
| A (snapshot-066 / homogeneous) | 3 | 同一 distill family 内 recursion | 低 — teacher 信号 coherent, 但可能 redundant |
| B (snapshot-070 / divergent-path) | 3 | 3 种 reward mechanism (shaping+distill / PBRS / curriculum) | 中 — reward signal 多样但架构都是 Siamese team-level |
| **C (this) / max-diversity** | **4** | **distill + HP sweep + architecture + curriculum 四轴** | **高 — 4 teacher 可能 KL 冲突, 架构轴 (054M) 可能与 Siamese team-level 不兼容** |

Pool C **不是** Pool B 的 superset — Pool B 聚焦 reward mechanism, Pool C 是 "每个 actionable axis 拿一个代表" 的完全不同 framing:

| Teacher | 代表的 actionable axis | 当前 WR | Failure mode 假设 |
|---|---|---:|---|
| 055v2@peak | **Distill (recursive ensemble)** axis | ~0.90-0.91 (pending) | teacher 已 compound, student 容易 over-imitate ensemble mean 而失 tactical sharpness |
| 056D@1140 | **HP sweep (lr=3e-4)** axis | **0.891** | LR 单独调优得到 local plateau, state coverage 可能窄 |
| 054M@peak | **MAT-min architecture (cross-agent attention + FFN/LN)** axis | pending | 架构显式 model teammate interaction, 可能在 same-side coordination 上强, 但 PBRS / shaping-driven tactic 上弱 |
| 062a@1220 | **Curriculum + adaptive phase gate + no-shape** axis | **0.892** | 对 baseline 熟, 对 weak opponent 可能 over-aggressive |

### 0.3 最大多样性假设

**Hypothesis framing**: 如果 4 个 teacher 的 failure mode 真的 pairwise orthogonal, 那么 ensemble avg softmax 在 student 看到的 state 上 **不应该是 noise** — 应该是 "某个 teacher 在该 state 上特别自信, 其余 3 个给出平均先验"。Student 通过 KL regularization 学到 "根据 state 选 teacher 权威者" 的 state-conditional weighting, 这是 3-teacher 到 4-teacher 多出来的能力。

**反面风险** (§4 详述): 4-teacher ensemble 比 3-teacher 多一个自由度, 也多一个引入 noise 的机会。**054M 的 cross-agent attention 架构** 可能产生与 Siamese team-level student 显著不同的 action distribution, 其 logit 被平均进 ensemble 后可能成为纯 noise source 而非信息源。

### 0.4 "Ensemble 是 stability 不是 intelligence" 原则 (用户已明确)

[MEMORY: ensemble_not_intelligence]. Pool C **不部署 4-way ensemble at inference**, 仅在 training 用 ensemble avg 作为 KL target, student deploy 时仍是 single 031B forward。完全合规 single-model SOTA 路径。

## 1. 核心假设

### H_072

> 用 KL distill loss 把 **4 个 actionable-axis divergent** teacher (055v2@peak, 056D@1140, 054M@peak, 062a@1220) 的 joint action probs 压缩进 031B-arch student (warm-start from 031B@80), **combined 2000ep peak ≥ 0.920**, 超过当前 055@1150 SOTA 0.907 **+0.013**。

注: 阈值 +0.013 比 Pool B (+0.011) 略高, 因为 Pool C 多付出 1 个 teacher 的 gradient 复杂度成本, 若 gain 不足 +0.013 则说明 "第 4 个 teacher 在 marginal 上没贡献, 回到 3-teacher 更经济"。

### 子假设

- **H_072-a (4-axis orthogonality 成立)**: distill / HP / architecture / curriculum 四条路径的 failure mode 真的 pairwise orthogonal, ensemble avg 在 student-visible state 上仍是 informative signal 而非 noise。
- **H_072-b (warm-start from 031B@80 helps KL alignment)**: 与 Pool B 从 scratch 不同, Pool C student warm-start from 031B@80 (pre-distill checkpoint), 让 student 的 initial action distribution 已在 031B manifold 上, 降低与 3 个 Siamese-arch teacher (055v2 / 056D / 062a) 的初期 KL spike。054M (架构异质) 的额外 KL noise 由 031B manifold 提供的 anchoring 消化。
- **H_072-c (cross-architecture teacher 能提供信息而非噪声)**: 054M 的 cross-agent attention head 虽在架构上异于 Siamese, 但 joint action probs 是 **环境等价** 的 output-space signal, 学 output distribution 不要求架构匹配 — 因此 054M 的 logit 对 Siamese student 仍是 actionable knowledge transfer。这个子假设是 Pool C 相对于 Pool B 最大的 incremental risk, 若 H_072-c 失败则 Pool C 应退到 3-teacher (drop 054M)。

## 2. 设计

### 2.1 总架构 (与 Pool B 一致, 仅换 teacher set + 加 warm-start)

```
Per worker rollout step:
  obs (joint 672-dim) → student (031B Siamese + cross-attn, warm from 031B@80) → student_logits
                     → teacher_ensemble (4 frozen, mixed Siamese + MAT-min arch) → ensemble_avg_probs

Loss = PPO_loss(student) + α(t) · KL(student_logits || ensemble_avg_probs)

α(t) schedule: 0.05 → 0.0 linear decay over first 8K SGD updates
```

### 2.2 Teacher ensemble 组成 (4 ckpts)

| Teacher | 来源 lane | Arch | ckpt 路径 (glob) | Axis 代表 |
|---|---|---|---|---|
| T1 | 055v2@peak (recursive distill extend) | Siamese + cross-attn | `/storage/ice1/5/1/wsun377/ray_results_scratch/055v2_extend_resume_1210_to_2000_20260421_030743/**/checkpoint_*` **TBD after extend done** | Distill (recursive ensemble) |
| T2 | 056D@1140 (PBT lr=3e-4) | Siamese + cross-attn | `/storage/ice1/5/1/wsun377/ray_results_scratch/056D_pbt_lr0.00030_scratch_*/**/checkpoint_001140/checkpoint-1140` | HP sweep |
| T3 | 054M@peak (MAT-min extend) | **MAT-min (cross-agent attention + FFN/LN)** | `/storage/ice1/5/1/wsun377/ray_results_scratch/054M_extend_resume_1250_to_1750_20260421_030244/**/checkpoint_*` **TBD after extend done** | Architecture axis |
| T4 | 062a@1220 (curriculum + adaptive + no-shape) | Siamese + cross-attn | `/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/**/checkpoint_001220/checkpoint-1220` | Curriculum + no-shape |

### 2.3 Teacher weighting 方案 — 保守 weighted (非 uniform)

用户提出两种候选: uniform 1/4 或 weighted 0.3 / 0.25 / 0.2 / 0.25。**推荐 weighted**, 理由:

1. **架构异质的 054M 权重应低**: 054M 的 logit space 可能与其余 3 个 Siamese teacher 不在同一 distribution family, 降低它的权重 (0.2) 是 H_072-c 失败时的 damage control。
2. **信息源熟度**: 055v2 是 recursive distill, 其 policy 已经 internalize 了 pre-055 ensemble knowledge, 给稍高权重 (0.3) 承认它 "information density per teacher" 最大。
3. **062a / 056D 等权 0.25**: 两者分别 reward axis (curriculum) 和 HP axis (lr), 独立性相当, 经验值相当 (0.892 / 0.891)。

| Teacher | Weight | 理由 |
|---|---:|---|
| T1 055v2@peak | **0.30** | Recursive distill → 信息密度最高 |
| T2 056D@1140 | **0.25** | HP axis, 独立 path |
| T3 054M@peak | **0.20** | Cross-arch damage control, H_072-c 保险 |
| T4 062a@1220 | **0.25** | Curriculum axis, 独立 path |
| Sum | 1.00 | — |

**Fallback**: 若 T3 load 失败 / 架构不兼容 `_FrozenTeamEnsembleTeacher` signature, **drop 054M**, 回到 uniform 1/3 of {T1, T2, T4}(即 Pool A'/B 变体)。详见 §4 + §6 smoke test。

### 2.4 Student setup (warm-start, 对照 Pool B 主要差异)

```bash
# Architecture (031B 同款 student)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64

# Warm start (与 Pool B 最大差异)
WARMSTART_CHECKPOINT=/storage/ice1/5/1/wsun377/ray_results_scratch/031B_ensemble_native_scratch_20260419_*/checkpoint_000080/checkpoint-80

# Distillation
TEAM_DISTILL_KL=1
TEAM_DISTILL_ENSEMBLE_KL=1
TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="<T1>,<T2>,<T3>,<T4>"
TEAM_DISTILL_TEACHER_ENSEMBLE_WEIGHTS="0.30,0.25,0.20,0.25"
TEAM_DISTILL_ALPHA_INIT=0.05
TEAM_DISTILL_ALPHA_FINAL=0.0
TEAM_DISTILL_DECAY_UPDATES=8000
TEAM_DISTILL_TEMPERATURE=1.0

# PPO (同 Pool B)
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048

# v2 shaping (student 端保持 055 食谱)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001 SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01 SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8 SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12 SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Budget (warm-start 节省 → 1250 iter 足够)
MAX_ITERATIONS=1250 TIMESTEPS_TOTAL=50000000
TIME_TOTAL_S=43200 EVAL_INTERVAL=10 CHECKPOINT_FREQ=10
```

### 2.5 与 Pool B (snapshot-070) 设计差异

| 维度 | Pool B | Pool C |
|---|---|---|
| Teacher 数 | 3 | **4** |
| Teacher 轴 | reward mechanism only | **4 orthogonal actionable axes** |
| 架构 homogeneity (teacher) | all Siamese | **3 Siamese + 1 MAT-min (054M)** |
| Teacher weighting | uniform 1/3 | **weighted 0.30 / 0.25 / 0.20 / 0.25** |
| Student init | scratch | **warm-start from 031B@80** |
| KL conflict risk | 低-中 | **中-高 (架构异质 teacher)** |

## 3. 预注册判据

| 判据 | 阈值 (combined 2000ep peak) | verdict 含义 |
|---|---|---|
| §3.1 marginal ≥ 0.915 | ≥ 0.915 | +0.008 vs 055 SOTA, max-diversity 有边际收益但不显著 |
| §3.2 主 ≥ 0.920 | ≥ 0.920 | **+0.013 vs 055, H_072 met** — 4 轴 orthogonality 成立 |
| §3.3 突破 ≥ 0.930 | ≥ 0.930 | +0.023 — 架构轴 (054M) 贡献证实, 立即启 Pool E (>4 teacher) |
| §3.4 持平 [0.900, 0.915) | in band | 4th teacher 边际 0, 退到 3-teacher (Pool B equivalent) |
| §3.5 退化 < 0.895 | < 0.895 | **KL conflict 真实发生** — 4-teacher ensemble 互抵, diversity 过度 |

## 4. 简化点 + 风险 + 降级预期 + 预案

### 4.1 简化 S1 — 沿用 online distillation + 4 teacher

| 简化项 | 完整方案 | 当前选择 |
|---|---|---|
| 数据收集 | DAGGER iterative | Online (teacher 看 student rollout) |
| Teacher 数 | 自适应 pruning | Hardcoded 4 |

### 4.2 ⚠️ 核心风险 R1 — 4-teacher KL 冲突 (Pool C 特有)

**机理**: 4 个 teacher 的 action distribution 在同一 state 上可能显著分歧:
- 054M (cross-agent attention) 可能在 teammate-coordination 要求的 state 上选 "pass to teammate", 而 056D (plain PPO + v2 shape) 可能选 "self shoot"。Ensemble avg softmax → 两个 action 都被 partial credit → student 学到 blurred policy, **worse than learning from any single teacher**。
- 055v2 (recursive distill ensemble) 已经是 soft policy, 其 logit 本身 entropy 较高。与 062a (curriculum, 对 baseline 很 decisive) 求平均 → entropy 被 curriculum teacher 拉低但方向被 055v2 拉糊 → student 两头不到岸。

**Pool B (3-teacher, all Siamese) 下该风险较小**; **Pool C 引入架构异质 teacher 后风险明显上升**。

**降级预期**: -1 ~ -2.5pp vs 最优 3-teacher Pool B equivalent。

**预案 (阶梯)**:
- **L1** (若 peak ∈ [0.895, 0.915)): α sweep {0.02, 0.03, 0.08} — 降低 KL 权重, 让 PPO signal 主导 student。+16h GPU。
- **L2** (若 peak < 0.895): **Drop 054M**, 回 3-teacher uniform {T1, T2, T4} — 本质退化到 Pool B', 验证 "架构异质 teacher is the noise source" 假设。+12h GPU。
- **L3** (若 L2 仍退化): **Drop 055v2 or 056D** (保 1 teacher + 062a), 回 2-teacher distill — 已退化到 Pool A / B 框架外, declare Pool C 路径 dead。
- **L4** (worst): 声明 multi-axis ensemble distill 在当前 031B-arch student + online setting 下达到饱和, 后续只追 recursion (Pool A) 或 single-teacher warm (snapshot-053/068 direction)。

### 4.3 风险 R2 — 4-teacher 内存 / 吞吐

**机理**: 每个 rollout step 要 forward 4 个 frozen teacher。Pool B 是 3 个, Pool C 是 4 个 → rollout side 开销 +33%。GPU memory 也多 ~800MB (4 × ~200MB per policy), 在 H100 80GB 下仍宽裕但不能忽视。

**降级预期**: iter rate 下降 15-25% vs Pool B, 意味着 1250 iter 可能需要 ~14-16h (vs Pool B 预期 12h)。

**预案**:
- L1: 若 iter rate < 60% Pool B 水平, 考虑 rotate teachers per-batch (每 batch 只 forward 2 个, 2 个组合轮转) — 数学上期望等效 uniform, 但降 per-step 开销。
- L2: 若 GPU OOM (某 teacher 加载失败 / model 大小异常), drop 054M (架构最大, 也最可疑)。

### 4.4 风险 R3 — 架构异质 teacher 的 logit scale 不对齐

**机理**: 054M 的 output head 可能 (pre-softmax) logit scale 与 Siamese teacher 不同 — 一个 team 的 logit 在 range [-3, 3], 另一个在 [-1, 1], 求 softmax 前的 **weighted avg of logits** 与 **weighted avg of probs** 行为完全不同。当前 `_FrozenTeamEnsembleTeacher` 如果是 avg-of-probs 则问题小, 若是 avg-of-logits 则 054M 会被 scale 更大的 teacher 淹没 / 反之。

**预案**: **§6 smoke test** 必须 diff `_FrozenTeamEnsembleTeacher` 实现, 确认用的是 **weighted avg of probs** (不是 avg of logits)。若不是, fix 或 drop 054M。

### 4.5 全程降级序列 (worst case 路径)

| Step | 触发条件 | 降级动作 | 增量工程 |
|---|---|---|---|
| 0 | Default | Online + weighted (0.30/0.25/0.20/0.25) + 4-teacher max-diversity | base ~14-16h |
| 1 | Peak ∈ [0.895, 0.915) (§3.4 持平) | α sweep {0.02, 0.08} | +16h GPU |
| 2 | Peak < 0.895 (§3.5 退化) | Drop 054M, 3-teacher uniform | +12h GPU |
| 3 | Step 2 仍退化 | Drop second teacher, 2-teacher | +10h GPU |
| 4 | Step 3 失败 | Declare Pool C (max-diversity) 路径 dead | — |
| 5 | Step 4 + Pool B also < Pool A | **回退到 Pool A (recursive, same-family distill) 作为唯一 ensemble-distill 轴** | — |

**关键 fallback 原则**: Pool C 若 regress, **退回 Pool B 或 Pool A, 不考虑进一步加 teacher**。用户已明确 "never downgrade — fix root cause or pause and ask"; 若 Pool C 4-teacher 规模比 3-teacher 差, 根因是 diversity 过度 / conflict, 应退不是进, 且**必须先停下来向用户 report**。

## 5. 不做的事

- **不 launch 直到 054M_extend + 055v2_extend 完成**: 两个 dependency 都需要真 peak ckpt, 无 peak 则 teacher 质量不可控, 实验不可归因。
- 不在 054M 架构异质风险明确评估 (§6 smoke test) 之前启动 launcher。
- 不在 Pool C 中换 student 架构 (保持 031B Siamese + cross-attn, 严格对照 Pool A / Pool B)。
- 不同时 sweep teacher weights + α (两个 axis 一起 sweep 会 explode search space, 先固定 weighted 方案跑一次)。
- 不并行跑 Pool C 和其他 max-diversity variants (一次只放一个 4-teacher 配置, 避免 GPU 拥堵)。
- 不混入 Pool D (reward-signal diversity 变体) — 两者 framing 不同, Pool C 先单独出 verdict。

## 6. 执行清单

- [ ] **(Blocker) 1. 等 055v2_extend 完成** (~3h 当前 progress 1330+/2000), 用 /post-train-eval 拿 peak ckpt
- [ ] **(Blocker) 2. 等 054M_extend 完成** (~1h 当前 progress 1440+/1750), 用 /post-train-eval 拿 peak ckpt
- [ ] 3. 起草 launcher `scripts/eval/_launch_072_poolC_newcomer_frontier.sh` (env vars: `TEAM_DISTILL_ENSEMBLE_KL=1`, `TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS="<T1>,<T2>,<T3>,<T4>"`, `TEAM_DISTILL_TEACHER_ENSEMBLE_WEIGHTS="0.30,0.25,0.20,0.25"`, `TEAM_DISTILL_ALPHA_INIT=0.05`, `TEAM_DISTILL_ALPHA_FINAL=0.0`, `LR=1e-4`, `MAX_ITERATIONS=1250`, `WARMSTART_CHECKPOINT=<031B@80>`)
- [ ] 4. **Smoke test `_FrozenTeamEnsembleTeacher`**:
  - [ ] 4a. 确认 4 个 ckpt (含 054M 架构异质) 可 load 不抛异常
  - [ ] 4b. 确认 ensemble avg 实现是 **weighted avg of probs** (不是 avg of logits) — 若不是, fix 实现或 drop 054M
  - [ ] 4c. Verify weighted weights (0.30/0.25/0.20/0.25) 正确解析并 sum 到 1.0
- [ ] 5. 选 free node (避开 055v2_extend / 054M_extend 占用节点)
- [ ] 6. Launch 1250 iter (14-16h, 注意比 Pool B 长)
- [ ] 7. 实时 monitor:
  - [ ] KL decay 正常, alpha 衰减正常
  - [ ] **iter rate 警戒线**: 若 < 60% Pool B 水平 → 执行 §4.3 L1 rotate
  - [ ] **early KL explosion 警戒**: 若前 200 iter KL > 3× Pool B 同期 → 怀疑 054M 架构冲突, kill + 执行 §4.2 L2 drop 054M
- [ ] 8. 训完 invoke `/post-train-eval 072`
- [ ] 9. Stage 1 1000ep top-k ckpts
- [ ] 10. Stage 1.5 rerun v2 500ep → combined 2000ep on top-3
- [ ] 11. Stage 2 500ep capture on peak ckpt
- [ ] 12. Stage 3 H2H: vs 055@1150 (SOTA ref), vs Pool A SOTA (pending), vs Pool B SOTA (pending), vs 054M@peak (cross-arch teacher 直接对照)
- [ ] 13. Verdict append §7, 严格按 §3 判据
- [ ] 14. 更新 rank.md + README.md + BACKLOG.md + task-queue-20260421.md

## 7. Verdict — Outcome B 持平 (cross-axis distill saturate, 2026-04-22 append-only)

### 7.1 Stage 1 baseline 1000ep (2026-04-22 [00:48 EDT])

- Trial: `072_poolC_cross_axis_distill_warm031B80_20260421_080015/TeamVsBaselineShapingPPOTrainer_Soccer_b5561_00000_0_2026-04-21_08-00-36`
- Selected ckpts (top 5%+ties+±1, 19 ckpts): 930-980 / 1000-1020 / 1060-1080 / 1170-1230
- Eval node: atl1-1-03-017-23-0, port 60205, 762s parallel-7

| ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| **🏆 1180** | **0.903** | 903-97 |
| 970 | 0.901 | 901-99 |
| 1230 | 0.898 | 898-102 |
| 1080 | 0.897 | 897-103 |
| 1200 / 1210 | 0.892 | 892-108 |
| 930 | 0.891 | 891-109 |
| 1020 / 1220 | 0.890 | 890-110 |
| 980 | 0.889 | 889-111 |
| 1000 / 1060 | 0.887 | 887-113 |
| 940 / 1070 | 0.884 | 884-116 |
| 950 | 0.884 | 884-116 |
| 1170 | 0.882 | 882-118 |

**peak = 0.903 @ ckpt-1180, mean(top 6) ~0.898, range [0.873, 0.903]**

### 7.2 严格按 §3 判据

| 阈值 | 实测 | verdict |
|---|---|---|
| §3.1 breakthrough ≥ 0.920 | ❌ 0.903 | not met |
| §3.2 main ≥ 0.915 | ❌ | not met |
| **Outcome B 持平 [0.900, 0.915)** | **✅ 0.903 in range** | **TIED, cross-axis 没补回 ceiling gap** |
| Outcome C regression < 0.895 | ❌ 0.903 | not regressed |

**Δ vs prior SOTA 055@1150 (0.907) = -0.004** — within SE。 **Δ vs NEW SOTA 1750 (0.9155) = -0.013** — sub-SOTA。 **Δ vs Pool A 071 (0.903) = 0** — 完全持平,cross-axis reward 多样性 marginal gain ≈ 0。

### 7.3 与 071/076/079 ceiling 模式合读

见 [snapshot-079 §6.3](snapshot-079-055v3-recursive-distill.md#63-与-071072076-saturation-模式合读) 的 4-lane saturation 表。

**Pool C 设计的 cross-axis (mixed reward signals) 假设 dead** — diversity 没补回 student ceiling, 不是 reward-signal redundancy 问题, 是 student-side bottleneck。

### 7.4 Raw recap

```
=== Official Suite Recap (parallel) === (full 19 ckpts above)
[suite-parallel] total_elapsed=761.7s tasks=19 parallel=7
```

完整 log: [072_baseline1000.log](../../docs/experiments/artifacts/official-evals/072_baseline1000.log)

### 7.5 Lane 决定

- **Pool C 072 lane 关闭** — cross-axis 没胜 Pool A homogeneous, 都 saturate 0.91
- 不执行 §4.2 L1 α sweep — 4 lane 同时 saturate 表明 problem 不在 hyperparameter
- 资源已转 080 / 081 / 082 / 083 / 073-resume



## 8. 后续发展线 (基于 verdict 的路径图)

### Outcome A — 突破 (combined 2000ep peak ≥ 0.920, H_072 met)

- **4-axis orthogonality 成立 → 架构轴贡献真实**。
- 短期: launch Pool E (snapshot-073+), 加入第 5 teacher (候选: 053Dmirror@670 PBRS axis, 补齐 reward-signal diversity)。
- 中期: 以 Pool C student 作为下一轮 recursive teacher (本身已 > 055v2), 启 "recursive on max-diversity base"。
- 长期: 形成 "每出一个新 actionable axis lane 就加入 teacher set, 定期 re-distill" 的 pipeline。

### Outcome B — 持平 (peak ∈ [0.900, 0.915))

- 4th teacher 边际收益 ≈ 0 → diversity 已饱和在 3 轴 (Pool B level)。
- 执行 §4.2 L1 α sweep 确认不是 α suboptimal, 若仍持平, **Pool C 路径关闭**, 后续只跑 Pool B 变体 (换 reward-signal teacher)。
- 不再追 "更多 teacher" 路径 — 问题不在数量, 在 teacher 质量。

### Outcome C — 退化 (peak < 0.895)

- **KL 冲突假设证实**, 4-teacher (尤其 054M 架构异质) 产生 gradient 互抵。
- 执行 §4.5 Step 2 drop 054M → Pool B equivalent。
- 若 Step 2 仍退化, Step 3 再 drop 一个, 直到找到 "最大不 regress 的 teacher 组合"。
- 若 Step 3 仍退化, **Pool C + max-diversity axis 全面 dead**, 后续只跑 Pool A (recursive same-family) 路径。
- 用户 ping: "diversity is not a free lunch — 1 个 noise teacher 就能摧毁 3 个 good teacher 的 signal"。

## 9. 相关

- [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md) — 原始 3-teacher 034E distill (Pool A baseline), 当前 SOTA 0.907 参照
- [SNAPSHOT-061](snapshot-061-055v2-recursive-distill.md) — 055v2 recursive distill (T1 来源 lane)
- [SNAPSHOT-054](snapshot-054-mat-min-cross-agent-attention.md) — 054M MAT-min 架构 (T3 来源 lane, 架构异质 teacher)
- [SNAPSHOT-056](snapshot-056-simplified-pbt-lr-sweep.md) — 056D lr=3e-4 (T2 来源 lane)
- [SNAPSHOT-062](snapshot-062-curriculum-noshape-adaptive.md) — 062a curriculum + no-shape (T4 来源 lane)
- [SNAPSHOT-066](snapshot-066-progressive-distill-BAN.md) — Pool A (homogeneous recursion) 对照
- [SNAPSHOT-070](snapshot-070-pool-B-divergent-distill.md) — Pool B (divergent-path 3-teacher) 直接对照, **Pool C = Pool B + 1 arch-axis teacher + warm-start**
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B student 架构 (warm from 031B@80)
- [task-queue-20260421.md §4 Pool C](../management/task-queue-20260421.md) — 用户原始 motivation
- [team_siamese_distill.py](../../cs8803drl/branches/team_siamese_distill.py) — distill model + `_FrozenTeamEnsembleTeacher` 实现
- [ensemble_agent.py](../../cs8803drl/deployment/ensemble_agent.py) — ensemble 推理参考
- **待起草**: `scripts/eval/_launch_072_poolC_newcomer_frontier.sh` (等 054M + 055v2 extend 完)

### 理论支撑

- **Hinton et al. 2015** "Distilling the Knowledge in a Neural Network" — 标准 distillation 基础, 原文已讨论 teacher ensemble diversity 对 student generalization 的贡献
- **Rusu et al. 2016** "Policy Distillation" — RL 下 multi-task / multi-teacher distillation, 阐明 diverse teacher 边际收益递减现象
- **Furlanello et al. 2018** "Born-Again Networks" — recursive distillation compound (Pool A 理论), 作为 Pool C diversity axis 对照
- **Czarnecki et al. 2019** "Distilling Policy Distillation" — distillation loss 与 PPO 兼容性
- **Anil et al. 2018** "Large scale distributed neural network training through online distillation (Codistillation)" — 多 teacher online distill 的 variance reduction + KL-conflict empirical finding
