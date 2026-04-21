# SNAPSHOT-062: Curriculum + No-shape + Adaptive Phase Gating (Tier A4.1/A4.2)
- **日期**: 2026-04-20
- **负责人**: Self
- **状态**: 预注册; pending launch (approved 2026-04-20)

## 0. 背景
- 058 (simplified curriculum + v2 shape + fixed iter boundaries) peak 1000ep 0.847 — sub-031B, §3.5 退化区。但:
  - 50ep/1000ep gap = 0.073pp 异常大 → 真实 peak 未被 1000ep draw 抓到
  - 全程最简化 (A4.A + A4.B + A4.C 三项叠加), 无法代表 curriculum 真实潜力
  - User 2026-04-20 decision: 不关闭路径; 先升级到 L1 (adaptive) + L2 (boundary sweep)

- 031B-noshape (rank.md §3.3) 已 verified: combined 2000ep 0.875 ± 0.007, Δ vs 031B-with-v2 0.880 = -0.5pp, **NOT statistically significant** → shaping 主要加速 convergence, 不提升 peak
- 对 curriculum, 去 shape = reward 直接映射 WR (`reward_mean ≈ 2 × (WR - 0.5)`) → adaptive gate 无需猜 shaping offset, 设计大幅简化

## 1. 假设
- **H_062** (主): curriculum + no-shape + adaptive gate (WR-threshold) → 1000ep peak ≥ 0.886 (+0.4pp vs 031B 0.882 = marginal significant)
- **子假设**:
  - H_062-a: fixed schedule 的 phase 切点 (058 用 200/500/1000) 可能太早或太晚 → boundary sweep 验证
  - H_062-b: 058 reward >= gate 之前强制切 phase 导致 catastrophic forgetting → adaptive gate 防此
  - H_062-c: 去 shape 后 reward 信号更稳定 → 50ep/1000ep gap 收窄 (< 0.04)
  - H_062-d: sparse-only + curriculum 组合比 sparse-only alone (031B-noshape 0.875) 有 +0.01~+0.03 提升

## 2. 设计

### 2.1 3-variant boundary sweep

| Variant | Boundaries | 节奏 | 假设 |
|---|---|---|---|
| **062a** | 0/200/500/1000 | 058 原 schedule | 验证 gate 单独的贡献 |
| **062b** | 0/100/300/800 | 加速转换 | 058 可能 random phase 太长浪费 iter |
| **062c** | 0/300/700/1100 | 延长 random | 058 可能早切到 baseline 导致 unstable |

### 2.2 Adaptive Gate 设计

Phase 切换需要 **同时** 满足:
1. `train_iter >= next_phase.start_iter` (时间条件, 同 058)
2. `episode_reward_mean >= gate_reward[phase]` (新增: reward 门控)
3. `train_iter - last_advance >= min_phase_iters=50` (防震荡)

Gate reward thresholds (per phase):
- Phase 1 (random) → Phase 2 (mixed low): `reward >= -0.5` (WR ~ 0.25, 学会别总输)
- Phase 2 → Phase 3 (mixed high): `reward >= 0.0` (WR ~ 0.50, 持平)
- Phase 3 → Phase 4 (baseline): `reward >= 0.5` (WR ~ 0.75, 稳定赢)

Fallback: 若 iter 超过 next_phase.start_iter + 200 而 reward 仍不达标, 强制进 phase (避免永远卡死)

### 2.3 工程实现

**Code change**:
1. `cs8803drl/branches/curriculum.py` 加 `AdaptiveCurriculumPhaseScheduler(CurriculumPhaseScheduler)`:
   - `__init__(phases, gate_rewards, min_phase_iters=50, max_phase_wait=200)`
   - `try_advance(train_iter, recent_reward)` → returns current baseline_prob
   - tracks `_current_phase_idx`, `_last_advance_iter`
2. `CurriculumUpdateCallback.on_train_result` 加 adaptive path:
   - 读 `result["episode_reward_mean"]`
   - 若 `CURRICULUM_ADAPTIVE=1` 调用 `scheduler.try_advance(iter, reward)`
   - 否则 走原 `scheduler.baseline_prob_for_iter(iter)` path (backward compat)
3. Env vars:
   - `CURRICULUM_ADAPTIVE=1` (enable adaptive)
   - `CURRICULUM_GATE_REWARDS=-0.5,0.0,0.5` (gate per transition, len = phases-1)
   - `CURRICULUM_MIN_PHASE_ITERS=50`, `CURRICULUM_MAX_PHASE_WAIT=200`

### 2.4 训练超参
```bash
# 同 058 except:
USE_REWARD_SHAPING=0           # 去 v2
CURRICULUM_PHASES=...          # variant-specific
CURRICULUM_ADAPTIVE=1
CURRICULUM_GATE_REWARDS=-0.5,0.0,0.5
CURRICULUM_MIN_PHASE_ITERS=50
CURRICULUM_MAX_PHASE_WAIT=200
# 其他: 031B arch (TEAM_SIAMESE_ENCODER + TEAM_CROSS_ATTENTION), LR=1e-4, 1250 iter
```

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.886 | +0.4pp vs 031B | adaptive gate 有增益 |
| §3.2 主: peak ≥ 0.895 | +1.3pp | curriculum path 真验证 |
| §3.3 突破: peak ≥ 0.905 | 接近 055 distill | curriculum 进入 SOTA tier |
| §3.4 持平 031B-noshape: [0.870, 0.886) | sub-marginal | adaptive 不影响 peak, 仅换 convergence pattern |
| §3.5 退化: peak < 0.860 | curriculum 反伤 | adaptive 门控本身破坏 training |

## 4. 简化点 + 风险 + 降级 + 预案

### 4.1 简化 A4.1A — reward 作为 WR proxy 仍不完美
- reward_mean 是 rollout-time 信号, 不等于 independent eval WR
- 对 variance 敏感 (特别是 random opponent phase)
- **预案**: 若 gate 过度延迟 phase 切换 (iter 500 仍在 phase 1), 降 gate_reward -0.2 每阶段

### 4.2 简化 A4.1B — 只测 3 个 boundary
- 未覆盖 boundaries 全空间
- **预案**: 若 062a/b/c 都 sub-SOTA, sweep 6-8 个 boundary

### 4.3 简化 A4.2 — 未加 frontier-pool opponent (L3)
- 只 random + baseline, 无弱 self-play
- **预案**: 若 062 某 variant 接近 0.886+, 做 L3 follow-up (加 028A@200 早期 frontier 作为 weak self-play opponent)

### 4.4 全程降级序列
| Step | 触发 | 动作 |
|---|---|---|
| 0 | Default | 3 variants 并行 |
| 1 | 所有 variants < 0.870 | 降 gate_rewards |
| 2 | step 1 失败 | 重 sweep boundaries (L2 full) |
| 3 | step 2 失败 | 加 frontier-pool (L3) |
| 4 | step 3 失败 | 058 / curriculum 路径真关闭 |

## 5. 不做的事
- 不在 code 完成 + smoke pass 前 launch
- 不混入 shaping (保持 no-shape 纯实验)
- 不跟 Tier 1 / Tier 2 同时开 (已在 flight, 避免 node 抢)
- 不立刻做 L3 (frontier pool) / L4 (PAIRED RL teacher)

## 6. 执行清单
- [ ] 1. Code AdaptiveCurriculumPhaseScheduler (~30-45 min)
- [ ] 2. Smoke test: import + phase advance logic + callback integration
- [ ] 3. 3 launchers: `_launch_062a_noshape_adaptive.sh` etc.
- [ ] 4. 找 3 free nodes, launch
- [ ] 5. Verdict per variant, 决定 L3 是否启动

## 7. Verdict

### 7.1 062a Stage 1 1000ep verdict (2026-04-21 00:30 EDT, append-only)

**062a (curriculum + adaptive WR-gated phase + no-shape, baseline boundaries 0/200/500/1000)** Stage 1 post-eval done — 10 ckpts × 1000ep verified.

| ckpt | inline 200ep | 1000ep verified | Δ optimism |
|---:|---:|---:|---:|
| 680 | 0.910 | 0.864 | -0.046 |
| 890 | 0.920 | 0.868 | -0.052 |
| 1080 | 0.910 | 0.885 | -0.025 |
| 1090 | 0.925 | 0.891 | -0.034 |
| 1100 | 0.920 | 0.877 | -0.043 |
| 1110 | 0.925 | 0.878 | -0.047 |
| 1130 | 0.910 | 0.870 | -0.040 |
| 1180 | 0.930 (inline peak) | 0.888 | -0.042 |
| 1210 | 0.920 | 0.886 | -0.034 |
| **1220** | — | **0.911** ⭐ (peak) | — |

Raw recap:
```
=== Official Suite Recap (parallel) ===
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_000680/checkpoint-680 vs baseline: win_rate=0.864 (864W-136L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_000890/checkpoint-890 vs baseline: win_rate=0.868 (868W-132L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_001080/checkpoint-1080 vs baseline: win_rate=0.885 (885W-115L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_001090/checkpoint-1090 vs baseline: win_rate=0.891 (891W-109L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_001100/checkpoint-1100 vs baseline: win_rate=0.877 (877W-123L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_001110/checkpoint-1110 vs baseline: win_rate=0.878 (878W-122L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_001130/checkpoint-1130 vs baseline: win_rate=0.870 (870W-130L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_001180/checkpoint-1180 vs baseline: win_rate=0.888 (888W-112L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_001210/checkpoint-1210 vs baseline: win_rate=0.886 (886W-114L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/.../checkpoint_001220/checkpoint-1220 vs baseline: win_rate=0.911 (911W-89L-0T)
[suite-parallel] total_elapsed=485.3s tasks=10 parallel=7
```

**核心结果**:
- **Peak single-shot 1000ep = 0.911 @ iter 1220** (SE 0.016, CI [0.890, 0.927])
- vs **055 combined 2000ep 0.907**: Δ=**+0.004pp 在 SE 范围内** (single-shot 可能 match SOTA, 但需要 combined verify)
- vs **058 simplified curriculum 0.847 (combined 2000ep)**: Δ=**+0.064pp huge gain** — curriculum upgrade 显著有效
- vs **031B 0.880**: Δ=+0.031pp, z ≈ 1.97 (marginal sig at *)
- vs **031B-noshape 0.875**: Δ=+0.036pp, z ≈ 2.27 (significant *)

**预注册 §3 判据**:
- §3.1 marginal (≥0.886): **HIT** (+0.025)
- §3.2 主 (≥0.895): **HIT** (+0.016)
- §3.3 突破 (≥0.905): **HIT** ⭐
- §3.4 持平 [0.870, 0.886): missed below
- §3.5 退化 (<0.860): no

**重要 reframing**:
- **062a NEW SOTA-tier candidate** (single-shot 1000ep 0.911 ties 055 single-shot 0.911 before combined-corrected to 0.907)
- Curriculum + adaptive gate + no-shape 是与 distillation **completely independent** 的 path, 殊途同归到 ~0.91
- 单点测量 SE 0.016 → 062a vs 055 combined 0.907 在 SE 范围内, **需要 062a Stage 2 rerun 拿 combined 2000ep 才能定论**
- Stage 2 rerun 已 launched (b2hmqfz09 on 5021865, 1000ep × 3 ckpts {1180/1210/1220} port 37005)

**对项目策略影响**:
- Curriculum 路径从 "sub-SOTA dead end" (058 0.847) **复活**为 SOTA-tier path
- 跟 distill 形成两条独立 SOTA path, 可能联合 (062 + distill teacher)
- 062b/c (faster/slower boundaries) 等出可能进一步定位 boundary optimum
- 可能新方向: 062a + 5-teacher distill 混合 → 双线 stack

Inline 200ep 再次 -0.03~-0.05pp 高估 (consistent with 200ep noise memo)。

**062b/c verdict pending** (separate launches)。

### 7.2 Stage 2 rerun correction (2026-04-21 00:38 EDT) — peak revised down from 0.911 to 0.892 combined 2000ep

**APPEND-ONLY CORRECTION** to §7.1。Stage 2 rerun (port 37005, fresh n=1000 sample) on plateau ckpts {1180, 1210, 1220} just completed and SHARPLY corrected the §7.1 single-shot verdict。

| ckpt | Stage 1 1000ep | Stage 2 rerun 1000ep | Combined 2000ep | ±SE |
|---:|---:|---:|---:|---:|
| 1180 | 0.888 (888W-112L) | 0.884 (884W-116L) | 1772W-228L = 0.886 | ±0.007 |
| 1210 | 0.886 (886W-114L) | 0.892 (892W-108L) | 1778W-222L = 0.889 | ±0.007 |
| **1220** | 0.911 (911W-89L) | 0.873 (873W-127L) | 1784W-216L = **0.892** | ±0.007 |

Raw rerun recap:
```
=== Official Suite Recap (parallel) ===
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/TeamVsBaselineShapingPPOTrainer_Soccer_dd9fa_00000_0_2026-04-20_14-29-28/checkpoint_001180/checkpoint-1180 vs baseline: win_rate=0.884 (884W-116L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/TeamVsBaselineShapingPPOTrainer_Soccer_dd9fa_00000_0_2026-04-20_14-29-28/checkpoint_001210/checkpoint-1210 vs baseline: win_rate=0.892 (892W-108L-0T)
/storage/ice1/5/1/wsun377/ray_results_scratch/062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/TeamVsBaselineShapingPPOTrainer_Soccer_dd9fa_00000_0_2026-04-20_14-29-28/checkpoint_001220/checkpoint-1220 vs baseline: win_rate=0.873 (873W-127L-0T)
[suite-parallel] total_elapsed=246.9s tasks=3 parallel=3
```

**Verdict (CORRECTION)**:
- **TRUE combined 2000ep peak = 0.892 @ iter 1220** (NOT 0.911 single-shot Stage 1)
- Best by combined: 1220 = 0.892, 1210 = 0.889, 1180 = 0.886 (plateau-stable, all within 0.006 of each other)
- vs **055 combined 2000ep 0.907**: Δ=**-0.015pp**, z = -0.015 / sqrt(0.007² + 0.0066²) ≈ **-1.56 NOT sig but clearly behind**
- vs **056D combined 2000ep 0.891**: Δ=+0.001pp, **essentially tied**
- vs **031B combined 0.880**: Δ=+0.012pp, NOT sig
- vs **058 simplified combined 0.842**: Δ=**+0.050pp huge gain** (curriculum L1 + L2 + no-shape upgrade VALIDATED, but only to ~056D level not 055)

**预注册 §3 判据 (REVISED)**:
- §3.1 marginal (≥0.886): **HIT** (0.892)
- §3.2 主 (≥0.895): **MISS** (0.892 < 0.895)
- §3.3 突破 (≥0.905): **MISS** ⚠️ (downgraded from earlier "HIT" claim in §7.1)

**Reframing (CORRECTION)**:
- 062a is **NOT** new SOTA-tier (earlier §7.1 claim was based on Stage 1 single-shot which had +0.019pp positive fluctuation at iter 1220)
- 062a IS curriculum-path validated (vs 058 simplified +0.050) AND is competitive (~056D level)
- Curriculum + adaptive + no-shape ≈ Distill + LR=3e-4 (059) ≈ LR=3e-4 alone (056D) ≈ ~0.89 plateau
- **0.91 ceiling currently held only by 055 combined 0.907**

**Lesson reinforced (3rd time today)**:
- Inline 200ep noise: -0.04 to -0.12pp common (056extend / 056C / 062a 1180 / 062a 1210 etc)
- **Stage 1 single-shot 1000ep also unreliable**: 062a 1220 went 0.911 → 0.873 (Δ-0.038) on rerun
- Combined 2000ep is the minimum for SOTA-tier verdict
- See `feedback_inline_eval_noise.md` memory; this case extends the pattern to "even 1000ep single-shot can mislead by 0.04pp"

### 7.3 [2026-04-21 00:50 EDT] 062c slower boundaries verdict (Stage 1 + Stage 2 combined 2000ep)

**APPEND-ONLY**. Tier A4.1/4.2 slower-boundaries variant: curriculum + adaptive WR-gated phase + no-shape, boundaries `0/300/700/1100` (vs 062a baseline `0/200/500/1000`).

#### Stage 1 1000ep (port 38005, top 10 ckpts)

| ckpt | inline 200ep | Stage 1 1000ep |
|---:|---:|---:|
| 690 | 0.905 | 0.837 |
| 760 | 0.900 | 0.855 |
| 900 | 0.900 | 0.844 |
| 950 | — | 0.885 |
| 960 | 0.925 (inline peak) | 0.852 |
| 970 | — | 0.848 |
| 990 | 0.900 | 0.853 |
| 1010 | — | 0.883 |
| **1090** | 0.900 | **0.899** |
| 1100 | 0.900 | 0.887 |

#### Stage 2 rerun (port 35005) — top 3 single-shot peaks {1010, 1090, 1100}

| ckpt | Rerun 1000ep |
|---:|---:|
| 1010 | 0.864 |
| 1090 | 0.873 |
| 1100 | 0.875 |

#### Combined 2000ep

| ckpt | Stage 1 | Rerun | **Combined 2000ep** | ±SE |
|---:|---:|---:|---:|---:|
| 1010 | 0.883 | 0.864 | 1747W-253L = 0.874 | ±0.007 |
| **1090** | 0.899 | 0.873 | 1772W-228L = **0.886** (peak) | ±0.007 |
| 1100 | 0.887 | 0.875 | 1762W-238L = 0.881 | ±0.007 |

Raw recaps:
```
=== Stage 1 062c 1000ep (port 38005) ===
... checkpoint_001010 vs baseline: 0.883 (883W-117L)
... checkpoint_001090 vs baseline: 0.899 (899W-101L) [single-shot peak]
... checkpoint_001100 vs baseline: 0.887 (887W-113L)

=== Stage 2 rerun 062c (port 35005) ===
... checkpoint_001010 vs baseline: 0.864 (864W-136L)
... checkpoint_001090 vs baseline: 0.873 (873W-127L)
... checkpoint_001100 vs baseline: 0.875 (875W-125L)
```

#### Verdict

- **TRUE combined 2000ep peak = 0.886 @ iter 1090** (Stage 1 single-shot 0.899 had +0.013pp positive fluctuation, pulled down by rerun 0.873)
- Plateau iter 1010-1100 combined: [0.874, 0.886] mean 0.880
- vs **062a combined 0.892**: **Δ=-0.006pp** — slower boundaries LIGHTLY worse than baseline schedule
- vs **055 SOTA 0.907**: Δ=-0.021pp, below
- vs **056D 0.891**: Δ=-0.005pp, essentially tied
- vs **058 simplified 0.847**: Δ=**+0.039pp gain** (curriculum L1+L2+no-shape upgrade still validated)

**预注册 §3 判据**:
- §3.1 marginal (≥0.886): **HIT** (just at threshold)
- §3.2 main (≥0.895): **MISS**
- §3.3 breakthrough (≥0.905): **MISS**

#### Pattern emerging across 062 boundary sweep

- **062a** (baseline `0/200/500/1000`) combined = **0.892** (+0.045 vs 058 simplified)
- **062c** (slower `0/300/700/1100`) combined = **0.886** (+0.039 vs 058 simplified)
- **062b** (faster `0/100/300/800`) — pending post-eval

**Boundary verdict**: slower 比 baseline 略差 (-0.006)。Faster (062b) 待测,但 pattern 暗示 boundaries 在 baseline (062a) 附近就是 sweet spot, slower 不帮助。Adaptive WR-gate 让 schedule 之间差异变小 (boundaries 是 "earliest possible",adaptive 推迟了 actual 切换),所以 062a/c 的 boundary 差距变小。

**Lesson reinforced (4x today)**: Stage 1 single-shot 1000ep again misled by +0.013pp on peak ckpt (062c@1090: 0.899 → combined 0.886). Combined 2000ep is the only path to definitive verdict. See `feedback_inline_eval_noise.md`; cumulative evidence 4 separate cases (058 / 062a / 062c / earlier inline-eval failures).

## 8. 后续路径

### Outcome A — 突破 (任一 variant ≥ 0.895)
- 该 boundary 升级为 curriculum 主线
- 做 H2H vs 055/056D (迁移性测试)
- 考虑 L3 (frontier-pool opponent) 看是否还能推高

### Outcome B — 持平 031B-noshape
- 说明 adaptive gate 本身不是 bottleneck
- 下一层是 opponent diversity (L3) 或 teacher 机制 (L4)

### Outcome C — 退化
- Adaptive gate 反伤 (e.g., 卡死导致学习不足)
- 降级 step 1-2 执行

## 9. 相关

### 理论支撑
- **Soviany et al. 2022** "Curriculum Learning: A Survey" — adaptive curriculum 综述
- **Florensa et al. 2017** "Automatic Goal Generation for Reinforcement Learning Agents" — adaptive difficulty with reverse curriculum
- **Matiisen et al. 2017** "Teacher-Student Curriculum Learning" — student progress-based task sampling
- **Bengio et al. 2009** (baseline curriculum learning)

### 代码
- [curriculum.py](../../cs8803drl/branches/curriculum.py) — 待扩展 `AdaptiveCurriculumPhaseScheduler`
- [train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py) — `CurriculumUpdateCallback` 需加 adaptive path
- `scripts/eval/_launch_062a_noshape_adaptive.sh` (待创建)
- `scripts/eval/_launch_062b_noshape_adaptive_faster.sh` (待创建)
- `scripts/eval/_launch_062c_noshape_adaptive_slower.sh` (待创建)

### 相关 snapshots
- [snapshot-058 (simplified curriculum, sub-SOTA 0.847 verdict)](snapshot-058-real-curriculum-learning.md)
- [snapshot-055 (ensemble distill SOTA 0.907 combined)](snapshot-055-distill-from-034e-ensemble.md)
- [snapshot-056 (simplified PBT LR sweep, lr=3e-4 win)](snapshot-056-simplified-pbt-lr-sweep.md)
