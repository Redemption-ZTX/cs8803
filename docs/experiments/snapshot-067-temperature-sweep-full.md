## SNAPSHOT-067: Temperature Sweep — Full T ∈ {1.5, 2.0, 3.0, 4.0} Map (Tier 3a follow-up)

**Date**: 2026-04-20
**Status**: 预注册 / launching

### §0 背景
- snapshot-063 (Tier 3a) 启动 T=2.0 单点变量隔离测试 (4-04-20 14:30, 当前 iter 200/1250 ETA ~9h)
- 单点测试若 marginal 难判 temperature 路径整体价值;朋友 + user 决定升级到**完整 sweep map**
- 5-point sweep (T=1.0 已有 anchor 055=0.907,加 T=1.5/2.0(在跑)/3.0/4.0 共 4 new lanes,3 待 launch)
- 完整 sweep 给定论性 verdict: 看到 monotonic / peaked / flat 三种 pattern 之一,直接判 temperature 路径**采纳/不采纳**

### §1 假设
**H_067** (primary): 5-point T sweep 在 [1.0, 4.0] 上能给出 clear pattern,确定最优 T (或证明 T 路径整体不工作)。

子假设:
- H_067-a: monotonic increasing in [1.0, T*] then decreasing → 标准 KD pattern
- H_067-b: T_optimal 落在 [2.0, 3.0] (Hinton paper 经验)
- H_067-c: T 太高 (≥ 4.0) 会 flatten teacher signal,regression
- H_067-d: T=1.0 (no softening) 已是最优 → temperature 不重要,关 T sweep

### §2 设计

#### 2.1 5-point sweep coordinates

| Point | T | Status | Rationale |
|---|---|---|---|
| **anchor** | T=1.0 | ✅ already have data (055 = 0.907 combined 2000ep) | KD 原始最 aggressive 设定 |
| 1 | T=1.5 | 待 launch | mild softening, 介于 1.0 和 Hinton 范围 |
| 2 | T=2.0 | ✅ in flight (063, iter 200/1250 @ 17:50) | Hinton recommended low end |
| 3 | T=3.0 | 待 launch | Hinton recommended sweet spot |
| 4 | T=4.0 | 待 launch | edge of useful range, test regression |

#### 2.2 Isolation principle
所有 lane 完全相同 except T:
- 3-teacher 034e ensemble (031B@1220 + 045A@180 + 051A@130)
- LR=1e-4 (same as 055/063)
- α schedule 0.05 → 0 by 8000 updates
- 031B student arch (Siamese + cross-attn)
- v2 shaping
- 1250 iter scratch
- 同 BASELINE_PROB=1.0

#### 2.3 GPU + node 预算
- 3 new lanes × 12h = 36 GPU-hours (vs PBT-full 60h cancelled)
- 节点: 5024111 (T=3.0 优先), 5024112 PD (T=1.5), 第 4 节点待 sbatch / 等其他 lane 完成 (T=4.0)
- 每 trial 1 GPU full

### §3 预注册判据 (per lane,vs 055 0.907 combined 2000ep anchor)

| 判据 | 阈值 per lane | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.911 | +0.004 vs 055 | T 有 measurable effect |
| §3.2 main: peak ≥ 0.915 | +0.008 | T 检测到 2σ SE detectable |
| §3.3 breakthrough: peak ≥ 0.925 | +0.018 | T unlock major gain |
| §3.4 持平 [0.895, 0.911) | sub-marginal | T=1 was fine |
| §3.5 regression: peak < 0.890 | T 太软 | flatten teacher 反伤 |

**Sweep 整体 verdict criteria** (cross-lane):
- Pattern A (peaked): 找到 T_optimal,采纳 → 加 dense sweep around peak
- Pattern B (monotonic decreasing from T=1): T 路径不工作,关闭
- Pattern C (flat): T 不影响 peak,关 T 路径,投资 progressive distill (Tier 3b 066)

### §4 简化 + 风险 + 降级
- 4.1 5-point 不是 fine-grain sweep (没 T=1.25, 2.5, 3.5)。降级:发现 peak 后 dense sweep around T*
- 4.2 不 sweep α schedule × T 交叉 (固定 α 同 055)。降级:若 peak ∈ [0.911, 0.920], 试 α schedule sweep at best T
- 4.3 不混 LR sweep (避免 confound). 降级:若 best T 找到, 后续可在 best T 上 + LR=3e-4 (combo Tier 1a 之前)

### §5 不做的事
- 不在 snapshot draft 完成 + launchers ready 前 launch
- 不打断 063 (T=2.0) lane (需保留作为 sweep 中点)
- 不与 progressive distill (066A/B) 同时改 LR/arch/teacher
- 不做 T < 1.0 (degenerate, equivalent to argmax distill)

### §6 执行清单
- [ ] Snapshot 起草 (this file) ✅
- [ ] Create 3 launchers: `_launch_063_temp_T15.sh` / `_launch_063_temp_T30.sh` / `_launch_063_temp_T40.sh` (copy 063 with T env var change)
- [ ] Launch T=3.0 on 5024111 (16h fresh) immediately
- [ ] Watcher: 5024112 PD → R, launch T=1.5
- [ ] T=4.0: wait for additional sbatch or freed node
- [ ] Verdict cross-lane after all 4 lanes Stage 1 1000ep done

### §7 Verdict

#### 7.1 T=4.0 (063_T40) — 4-sample combined 3000ep tied 055 (2026-04-21 14:55 EDT, append-only)

**Raw samples on `063_T40@1060` vs baseline**:
| Sample | n | WR | (W-L) |
|---:|---:|---:|---:|
| Stage 1 | 1000 | 0.923 | 923-77 |
| rerun 2 | 500 | 0.930 | 465-35 |
| rerun 3 | 500 | 0.904 | 452-48 |
| rerun 4 | 1000 | 0.897 | 897-103 |
| **combined** | **3000** | **0.9123** | **2737-263** (SE 0.0051, CI [0.902, 0.923]) |

**vs 055 SOTA 0.907**: Δ = +0.005 **within SE, tied**。Stage 3 H2H `T40@1060 vs 055@1150` n=500 = **0.508 (254W-246L, z=0.36, p=0.72 NOT sig)**.

**Lesson**: first 3 samples (mean 0.919) were positive outliers; 4th sample 0.897 corrected to ~0.912 true mean. Small-sample upward bias reinforces [MEMORY feedback_inline_eval_noise.md].

#### 7.2 Sweep-level pattern so far

| T | WR | source | note |
|---:|---:|---|---|
| 1.0 | 0.907 | 055@1150 combined 2000ep | SOTA anchor |
| 1.5 | 0.912 | 063_T15 @1230 single-shot 1000ep | **tied 055**, 8/8 ≥ 0.90 plateau |
| 2.0 | 0.904 | 055temp @1030 single-shot 1000ep | [snapshot-063 §7](snapshot-063-055-temp-sharpening.md#7-verdict-2026-04-21-0555-edt) tied |
| 3.0 | 0.908 | 063_T30 @1210 single-shot 1000ep | **tied 055**, single peak |
| 4.0 | **0.912** | 063_T40 @1060 combined 3000ep | **tied within SE** |

**5 of 5 T points have verdicts; all 5 statistically tied with T=1.0** within SE 0.005-0.016。Pattern C (flat across [1.0, 4.0]) **CONFIRMED** — temperature is not an informative axis for this distillation setup.

#### 7.3 T=1.5 (063_T15) — Stage 1 single-shot 1000ep tied 055 (2026-04-21 15:30 EDT, append-only)

**Raw samples on `063_T15` (official evaluator parallel, single-shot 1000ep each)**:

```
=== Official Suite Recap top-10 ===
ckpt 1230 vs baseline: win_rate=0.912   [★ peak]
ckpt 1090 vs baseline: win_rate=0.910
ckpt 1210 vs baseline: win_rate=0.905
ckpt 1070 vs baseline: win_rate=0.905
ckpt 1180 vs baseline: win_rate=0.904
ckpt 1220 vs baseline: win_rate=0.902
ckpt 1080 vs baseline: win_rate=0.901
ckpt 1060 vs baseline: win_rate=0.900
```

Log: `docs/experiments/artifacts/official-evals/063_T15_baseline1000.log`

**Plateau 1060-1230 — 8 of 8 top ckpts ≥ 0.900**, most stable T variant observed across the sweep. Window spans 170 iter.

**vs 055 SOTA 0.907**: Δ = +0.005 (single-shot) — **within SE 0.010, statistically tied**。
**vs 063_T40 combined 3000ep 0.912**: effectively identical.
**§3 判据 (single-shot)**: §3.1 marginal (≥0.911) **HIT**, §3.2 main (≥0.915) **MISS**, §3.3 breakthrough (≥0.920) **MISS**, §3.4 持平 boundary.

**Verdict**: T=1.5 tied 055 within SE; **structural plateau stability** is the only notable property (8/8 vs typical 6/8). Not actionable as SOTA shift but suggests T=1.5 reduces per-ckpt training variance mildly. Does not change T-sweep closure.

#### 7.4 T=3.0 (063_T30) — Stage 1 single-shot 1000ep tied 055 (2026-04-21 15:30 EDT, append-only)

**Raw samples on `063_T30` (official evaluator parallel, single-shot 1000ep each)**:

```
=== Official Suite Recap top-10 ===
ckpt 1210 vs baseline: win_rate=0.908   [★ peak]
ckpt 800  vs baseline: win_rate=0.897
ckpt 1170 vs baseline: win_rate=0.896
ckpt 790  vs baseline: win_rate=0.891
ckpt 1040 vs baseline: win_rate=0.891
ckpt 860  vs baseline: win_rate=0.890
ckpt 850  vs baseline: win_rate=0.890
ckpt 1190 vs baseline: win_rate=0.889
```

Log: `docs/experiments/artifacts/official-evals/063_T30_baseline1000.log`

Mid-window plateau 0.89-0.91 with single peak @ 1210; no multi-ckpt ≥ 0.90 plateau (unlike T=1.5).

**vs 055 SOTA 0.907**: Δ = +0.001 — **tied within noise**。
**§3 判据 (single-shot)**: §3.1 marginal (≥0.911) **MISS** (-0.003), §3.4 持平 **HIT** (0.908 in range).

**Verdict**: T=3.0 tied 055 with single-peak pattern, fits Pattern C (flat) as predicted. No actionable improvement.

#### 7.5 Final sweep decision — T path CLOSED

With all 5 T points (T ∈ {1.0, 1.5, 2.0, 3.0, 4.0}) tied at 055 SOTA within SE:

- **Pattern C confirmed across 2× span around T=1** — temperature softening does not unlock teacher signal on Multi-Discrete 6×3 action RL distillation
- Hinton 2015 "T=2-4 optimal" intuition from image-classification does NOT translate to RL policy distillation with this action factorization
- **T path CLOSED**; no further T variants (no T=0.5, no T=5.0); no dense sweep around any T*
- Resources fully committed to progressive distill (066A/B), pool distill (070-073), wide-student (076), per-agent student (077), 055v3 recursive (079), and **055v2_extend** (the unique outlier showing plateau ≥ 0.911, pending rerun)

See: [snapshot-063 §7.7](snapshot-063-055-temp-sharpening.md#77-t40-063_t40-4-sample-combined-3000ep-correction--tied-055-t-sweep-remains-closed-2026-04-21-1455-edt-append-only) (T=4.0 correction saga)

### §8 后续路径
- Pattern A + T_optimal found:
  - L1: dense sweep around T_optimal (5 pts ±0.5)
  - L2: combine T_optimal + LR=3e-4 (snapshot-068 combo)
  - L3: combine T_optimal + progressive distill (066A/B winner)
- Pattern B (regression): T path closed, no further T variants
- Pattern C (flat): T not informative, double down on progressive (066) + LR (065 056E/F)

### §9 相关
- [snapshot-063](snapshot-063-055-temp-sharpening.md) — Tier 3a single-point T=2.0 (now extended to sweep)
- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — 055 = T=1.0 anchor of sweep
- [snapshot-066](snapshot-066-progressive-distill-BAN.md) — Tier 3b parallel direction
- [BACKLOG.md](BACKLOG.md) — "Temperature sharpening follow-up: T ∈ {1.5, 3.0, 4.0}" entry now being executed
- **Theoretical**:
  - Hinton et al. 2015 "Distilling the Knowledge in a Neural Network" — KD T paper, recommends T=2-4
  - Cho & Hariharan 2019 "On the Efficacy of Knowledge Distillation" — empirical T sweep on CIFAR
- **Code targets**:
  - `scripts/eval/_launch_063_temp_T15.sh`
  - `scripts/eval/_launch_063_temp_T30.sh`
  - `scripts/eval/_launch_063_temp_T40.sh`
