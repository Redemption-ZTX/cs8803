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

## 6. Verdict

_Pending — training ~85% complete, ETA ~20:30 EDT_

## 7. 后续

若 §3.1 HIT (≥ 0.920):
- Gen 4: 用 079@peak 作 teacher → 079v4 单老师 recursive continue
- 验证 Hinton 再现是否无限递归或很快 saturate

若 §3.3 tied:
- 确认 student capacity 或其他 bottleneck 对 saturated teacher 无效
- 资源转 snapshot-076 wide-student DIR-A (已在跑)
