# SNAPSHOT-079: 055v3 — Single-Teacher Recursive Distill from 055v2@1000

- **日期**: 2026-04-21 (08:17 EDT launched)
- **状态**: 训练中 on 5028757 (~1.5h 剩)
- **前置**: [snapshot-061](snapshot-061-055v2-recursive-distill.md) (055v2 recursive distill 0.909) / [snapshot-075 §4](snapshot-075-strategic-synthesis-toward-0.93.md) (student capacity bound hypothesis)
- **关联**: [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) (distillation paradigm origin)

---

## 0. 背景

055 paradigm (Hinton-style distillation):
- Gen 1: `034E` 3-teacher ensemble (0.890) → `055` student **0.907** (+1.7pp student > teacher ensemble)
- Gen 2: `055v2` 5-teacher (034E + 055 + 056D) → student 0.909 (marginal +0.002 vs 055)

问题: Gen 2 的 +0.002 已接近 saturation → snapshot-075 §4 假设 **student capacity bound** (student 031B 0.46M vs teacher ensemble ~1.4M = 3× gap)。

## 1. Hypothesis H_079

### 1.1 主

> **Single-teacher distill from 055v2@1000 (0.909)** → student 是否能 **+1.7pp 再现**类似 055 的超越?
> - 预期 combined 2000ep peak ≥ 0.920 (+0.011 vs teacher)
> - Hinton 理论: student capacity ≥ single teacher 时,distill 能抽取 teacher 学不完全的 patterns

### 1.2 Anti

> 若 H_079 miss,student ≈ 0.905-0.910 tied teacher → 验证 snapshot-075 §4 "student capacity bound at single-teacher level too" 假设,而不是"多 teacher 噪声消除"是关键。

## 2. 设计

- **Teacher (1)**: **055v2@1000** (combined 3000ep 0.909 TIED 055 SOTA)
- **Student**: 031B Siamese + cross-attn (0.46M params, 同 055)
- **Warm-start**: 031B@80 (同 055v2 recipe)
- **Distill config**: `TEAM_DISTILL_ENSEMBLE_KL=1` with 1 teacher (functionally equivalent to `TEAM_DISTILL_KL=1`), α_init=0.05, α_final=0.0, T=1.0, decay 8000 updates
- **Reward**: v2 shaping (同 055)
- **Budget**: 1250 iter, 50M steps, 12h
- **LR**: 1e-4 (不用 3e-4,和 055 对齐 isolate T/teacher 变量)

## 3. 预注册 verdict

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 breakthrough: combined ≥ 0.920 | +0.011 vs teacher | Hinton 再现 re-confirmed, student > teacher 路径可 stack |
| §3.2 main: ≥ 0.915 | +0.006 | marginal student > teacher |
| **§3.3 tied: [0.905, 0.915)** | 接近 teacher | student ≈ teacher, recursion saturate 再确认 |
| §3.4 regression: < 0.900 | 退化 | single-teacher distill 路径对 saturated teachers 无用 |

## 4. 对比 snapshot-061 (055v2)

| Item | 055v2 (Gen 2) | **079 (Gen 3)** |
|---|---|---|
| Teacher count | 5 (034E 3 + 055 + 056D) | **1 (055v2)** |
| Teacher tier | mixed 0.87-0.91 | single SOTA 0.909 |
| Total teacher params | ~2.3M | 0.46M |
| Teacher-student ratio | 5× | **1× same capacity** |

If H_079 HIT: **single-teacher 同容量 distill 也能 +1pp** → student capacity 不是唯一瓶颈,teacher 的 "隐藏 pattern" 本身有 extractable gain。

## 5. 执行

- [x] Launcher `scripts/eval/_launch_079_055v3_recursive.sh`
- [x] Launched 2026-04-21 08:17 EDT PORT_SEED=47 on 5028757
- [ ] Complete + Stage 1 post-eval
- [ ] Verdict per §3

## 6. Verdict — §3.3 TIED (单老师 recursive 在 saturated teacher 上 saturate, 2026-04-22 append-only)

### 6.1 Stage 1 baseline 1000ep (2026-04-22 [00:35 EDT])

- Trial: `079_055v3_recursive_distill_warm031B80_20260421_081705/TeamVsBaselineShapingPPOTrainer_Soccer_0eb3f_00000_0_2026-04-21_08-17-25`
- Selected ckpts (top 5%+ties+±1, 19 ckpts): 700-720 / 770-790 / 980-1000 / 1050-1070 / 1120-1140 / 1180-1210
- Eval node: atl1-1-03-017-2-0, port 60405, 740s parallel-7

| ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| **🏆 1180** | **0.914** | 914-86 |
| 1130 | 0.909 | 909-91 |
| 1210 | 0.904 | 904-96 |
| 1120 | 0.895 | 895-105 |
| 1000 | 0.893 | 893-107 |
| 770 | 0.893 | 893-107 |
| 1200 | 0.892 | 892-108 |
| 1140 | 0.891 | 891-109 |
| 1060 | 0.887 | 887-113 |
| 720 | 0.885 | 885-115 |
| 460-1190 | 0.870-0.886 | tail |

**peak = 0.914 @ ckpt-1180, mean(top 6) ~0.901, range [0.870, 0.914]**

### 6.2 严格按 §3 判据

| 阈值 | 实测 | verdict |
|---|---|---|
| §3.1 breakthrough ≥ 0.920 | ❌ 0.914 | not met |
| §3.2 main ≥ 0.915 | ❌ 0.914 (just below) | not met |
| **§3.3 tied [0.905, 0.915)** | **✅ 0.914 in range** | **TIED teacher (saturation)** |
| §3.4 regression < 0.900 | ❌ | well above |

**Δ vs teacher 055v2@1000 (0.909) = +0.005pp** — within 1000ep SE (±0.012). 单 sample 1000ep 的真值 CI [0.902, 0.926]; 没 decisive 突破 teacher。

### 6.3 与 071/072/076 saturation 模式合读

| Lane | 设计 | Peak | vs 1750 SOTA |
|---|---|---|---|
| 071 Pool A 3-teacher homogeneous | 多 teacher 同家族 | 0.903 | -0.013 |
| 072 Pool C cross-axis reward | reward 多样性 | 0.903 | -0.013 |
| 076 wide-student 1.4× capacity | student 容量 | 0.905 | -0.011 |
| **079 single-teacher recursive** | **1 SOTA teacher** | **0.914** | -0.002 |

4 lane 同时 saturate ~0.90-0.91 → **distill paradigm 已饱和**, 不是 teacher 多样性 / 数量 / student 容量瓶颈。

### 6.4 Raw recap

```
=== Official Suite Recap (parallel) === (full 19 ckpts above)
[suite-parallel] total_elapsed=740.6s tasks=19 parallel=7
```

完整 log: [079_baseline1000.log](../../docs/experiments/artifacts/official-evals/079_baseline1000.log)

## 7. 后续

- **lane 关闭** — single-teacher recursive 在 saturated SOTA tier 上 marginal gain ≤ noise, 不值得 Gen 4 递归
- 资源已转 080 (Pool A v2 with 1750 teacher), 081 (orthogonal aggressive reward), 082 (two-stream arch), 083 (per-ray attn)
- snapshot-075 §4 student capacity 假设确认 — 但 076 的反证表明也不是 capacity 单变量, 是 **distill paradigm itself 的极限**
