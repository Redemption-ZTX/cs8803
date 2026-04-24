# SNAPSHOT-080: Pool A v2 — Homogeneous Distill with 055v2_extend@1750 NEW SOTA Teacher

- **日期**: 2026-04-21 (18:47 EDT launched)
- **状态**: 训练中 on 5032911, first ckpt 18:47, ETA ~10h
- **前置**: [snapshot-071](snapshot-071-pool-A-homogeneous-distill.md) (original Pool A) / [snapshot-061 §7.4](snapshot-061-055v2-recursive-distill.md#74-055v2_extend-verdict-new-sota-at-iter-1750) (1750 SOTA) / [snapshot-074 §7.6](snapshot-074-034-next-deploy-time-ensemble.md) (ensemble deploy closed)

---

## 0. 背景

**Pool A (071) 用 055v2@1000 (combined 3000ep 0.909) 作 teacher** — 但 2026-04-21 18:30 EDT 发现 **055v2_extend@1750 (combined 4000ep 0.9155, H2H vs 055 sig)** 是 NEW PROJECT SOTA。

Pool A Stage 1 peak = 0.909 @ 1160 (tied 055, 没突破)。原因之一: **teacher 只有 055v2@1000 tier (0.909),student 天花板也就 ~0.91**。

> **Pool A v2 hypothesis**: 换成 **1750 (0.9155 SOTA)** 作 primary teacher + 055@1150 + 056D@1140 辅 → student 可能 +1-2pp 超 1750,**突破 0.93**。

## 1. Hypothesis H_080

### 1.1 主

> 3-teacher ensemble distill with **1750** (weighted dominant by virtue of being strongest) into 031B student warm-from-031B@80:
> - **Combined 2000ep peak ≥ 0.925** (+0.010 vs teacher 1750 0.9155, follows 055's +1.7pp over 034E pattern)
> - n=10 grading 通过概率 95%+

### 1.2 Sub-hypotheses

- **H_080-a**: Hinton student > teacher paradigm **stacks** — 055 was gen 1 (+1.7pp), 079 gen 2 (TBD), 080 gen 2 (with ensemble including gen 2 teacher 1750) — 期望 gen 进步渐减但仍 >0
- **H_080-b**: **将 1750 加入 teacher pool 比 Pool A 071 的 055v2@1000 更 effective** — 因 1750 combined 4000ep 验证 tier 比 055v2@1000 的 combined 3000ep tier 高 +0.006pp,差不大但 direction 正

### 1.3 Anti-hypothesis

> 若 peak < 0.915 (跟 Pool A 071 类似),**confirm teacher-student distill saturated at current SOTA tier**,换哪个 0.90+ teacher 都一样 → 唯一出路是新 axis (wide-student / per-agent / DAGGER / aggressive-reward)

## 2. 设计

- **Teachers (3)**:
  - **055v2_extend@1750** (NEW SOTA, combined 4000ep 0.9155) ← key swap
  - 055@1150 (combined 2000ep 0.907, prior SOTA anchor)
  - 056D@1140 (single-shot 0.891, LR-axis complement)
- **Weighting**: uniform 1/3 (same as Pool A 071)
- **Student**: 031B Siamese + cross-attn, warm-start 031B@80
- **Distill config**: α_init=0.05 α_final=0.0 T=1.0 decay=8000
- **Reward**: v2 shaping (same as 071)
- **Budget**: 1250 iter, 50M steps, 12h
- **LR**: 1e-4
- **PORT_SEED**: 80

## 3. 预注册 verdict

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 breakthrough: peak ≥ 0.930 | +0.014 vs 1750, decisive 10-match safety | grading primary candidate |
| §3.2 main: ≥ 0.920 | +0.005 vs 1750, ~2× SE (1500ep) | marginal SOTA shift confirmed |
| §3.3 marginal: ≥ 0.915 | +0 vs 1750 tied | Hinton plateau confirmed (teacher ≈ student) |
| §3.4 persist tied 055: [0.905, 0.915) | same plateau | 1750 teacher no extra value over 055v2@1000 |
| §3.5 regression: < 0.905 | distill interaction | teacher mix wrong |

## 4. 与 Pool A 071 对比

| 维度 | 071 (original) | **080 (v2)** |
|---|---|---|
| Teacher 1 (primary) | 055@1150 (0.907) | 055@1150 (0.907) same |
| Teacher 2 | **055v2@1000 (0.909)** | **055v2_extend@1750 (0.9155)** ★ |
| Teacher 3 | 056D@1140 (0.891) | 056D@1140 (0.891) same |
| **Best teacher** | 0.909 | **0.9155** (+0.006) |
| **Teacher avg** | 0.902 | **0.905** (+0.003) |

**Teacher avg 提升仅 +0.003** — 主要 gain 来自 **replacing 055v2@1000 with 055v2_extend@1750** 这单点。要期望 student 大幅超过 teacher 仍需依赖 Hinton-style extraction,不能仅靠 teacher 增强 +0.003 就线性 expect student +0.006。

## 5. 不做的事

- 不加 4th teacher (055v2@1000 保留冗余 — 071 已验证 Pool A 不 regress,不需 over-engineer)
- 不改 LR / α / T (isolate teacher 变量)
- 不加 MAT 架构 teacher (还没支持 — see backlog)

## 6. 执行

- [x] Launcher `scripts/eval/_launch_080_poolAv2_1750_teacher.sh` (copy 071 + swap ckpt)
- [x] Launched 2026-04-21 18:47 EDT PORT_SEED=80 on 5032911 first ckpt 18:47
- [ ] Stage 1 post-eval
- [ ] Stage 2 rerun if marginal
- [ ] Stage 3 H2H vs 1750 (decisive if student > teacher)
- [ ] Verdict per §3

## 7. Verdict — §3.4 PERSIST TIED (6/6 distill saturation confirmed, 2026-04-22 append-only)

### 7.1 Audit (per `feedback_audit_training_data_first.md`)

- Trial: `080_poolAv2_with_1750_teacher_warm031B80_20260421_184131/TeamVsBaselineShapingPPOTrainer_Soccer_4b60d_00000_0_2026-04-21_18-41-53`
- 125 ckpt dirs (iter 10-1250, CHECKPOINT_FREQ=10, full budget reached)
- CSV baseline rows: complete through iter 1240 (only 1 ckpt lag — acceptable)
- stop_reason: TERMINATED (not ERROR) ✅
- No failed rows in inline eval ✅
- Training reward trajectory stable, reward +1.7 late ✅
- **AUDIT PASSED** — proceed to Stage 1

### 7.2 Stage 1 baseline 1000ep (2026-04-22 [04:47-05:01 EDT])

- Selected ckpts (top 5%+ties+±1, 18 ckpts): 600-630 / 750-770 / 820-840 / 870-890 / 1120-1160
- Eval node: 5032911 (atl1-1-03-010-30-0), port 60905, 829s parallel-7

| ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| **🏆 1130** | **0.906** | 906-94 |
| 1120 | 0.899 | 899-101 |
| 750 | 0.896 | 896-104 |
| 630 | 0.894 | 894-106 |
| 1150 | 0.893 | 893-107 |
| 600 | 0.891 | 891-109 |
| 840 | 0.891 | 891-109 |
| 880 | 0.891 | 891-109 |
| 870 | 0.889 | 889-111 |
| 620 | 0.888 | 888-112 |
| 1160 | 0.888 | 888-112 |
| 610 | 0.887 | 887-113 |
| 830 | 0.887 | 887-113 |
| 1140 | 0.884 | 884-116 |
| 760 | 0.883 | 883-117 |
| 820 | 0.882 | 882-118 |
| 770 | 0.879 | 879-121 |
| 890 | 0.870 | 870-130 |

**peak = 0.906 @ ckpt-1130, mean(top 6) ~0.897, range [0.870, 0.906]**

### 7.3 严格按 §3 判据

| 阈值 | 实测 | verdict |
|---|---|---|
| §3.1 breakthrough ≥ 0.930 | ❌ 0.906 | not met |
| §3.2 main ≥ 0.920 | ❌ | not met |
| §3.3 marginal ≥ 0.915 | ❌ | not met |
| **§3.4 persist tied [0.905, 0.915)** | **✅ 0.906** | **TIED with 055 SOTA, NO lift from 1750 teacher swap** |
| §3.5 regression < 0.905 | ❌ (just above) | not regressed |

**Δ vs 1750 NEW SOTA combined 4000ep 0.9155 = -0.010** (within SE)。 **Δ vs 071 Pool A (0.903) = +0.003** — swapping 055v2@1000 teacher for 1750 teacher gave **less than noise** gain。 **Δ vs 055 SOTA 0.907 = -0.001** — essentially tied。

### 7.4 与 071/072/073/076/079 + 080 saturation 模式合读 (6/6 distill)

| Lane | 设计 | Peak | Δ vs 1750 SOTA |
|---|---|---|---|
| 071 Pool A 3-teacher homogeneous | 055+055v2@1000+056D | 0.903 | -0.013 |
| 072 Pool C cross-axis | reward 多样性 | 0.903 | -0.013 |
| 076 wide-student | 1.4× capacity | 0.905 | -0.011 |
| 079 single-teacher | 1 SOTA teacher recursive | 0.914 | -0.002 |
| 073 Pool D cross-reward | 3-teacher 3-reward-paths | 0.909 | -0.007 |
| **080 Pool A v2** | **Pool A 071 with 1750 SOTA teacher swap** | **0.906** | **-0.010** |

**6 lane 同时 saturate 0.90-0.91**, 设计变量正交 (teacher count / family / reward axis / student capacity / reward-path diversity / teacher tier),都 cap 同一处 → **distill paradigm's same-family teacher ceiling confirmed at 6/6 evidence**。

**但 NOTE (per user 2026-04-22)**: 6/6 saturation 只否定 "distill from same-family teachers" 子问题。**Distill from heterogeneous specialist library** (PIPELINE Phase 7: 081 + 101A + 103A/B/C + 1750 + 055 = 真 orthogonal teachers) 是**开放问题**,可能 break 0.91 ceiling。6/6 saturation 不是 distill paradigm 整体的上限判决。

### 7.5 Raw recap

```
=== Official Suite Recap (parallel) === (full 18 ckpts above)
[suite-parallel] total_elapsed=828.7s tasks=18 parallel=7
```

完整 log: [080_baseline1000.log](../../docs/experiments/artifacts/official-evals/080_baseline1000.log)

### 7.6 Lane 决定 (autonomous loop triage)

- **080 lane 关闭** — +0.003 over 071 in noise band, 1750 teacher swap 没 unlock
- **Skip Stage 2 failure capture** — 5/5 saturation 已 diagnosed 过,080 新信息 marginal,capture GPU ROI 低
- **Skip Stage 3 H2H** — "tied 055/1750" 已知,H2H 只会复验,不新增
- 资源转 Wave 2 DIR-A/E/G (等 081/103-series 就绪) + PIPELINE V1 integration
- PIPELINE Phase 7 re-open distill axis 的 pre-condition 仍待 PIPELINE V1 ≥ 0.92

### 7.7 后续触发

- 如果 081 + 103A/B/C 后 PIPELINE V1 ≥ 0.92 → trigger **Phase 7 re-do** (distill from heterogeneous library, 080 setup 作 baseline 对照)
- 否则 080 stays closed, 不 re-run


## 8. 后续

- **§3.1 / §3.2 HIT**: Pool A v2@peak 成为新 SOTA; 启动 Gen 4 recursive distill from Pool A v2
- **§3.3 tied**: 证实 Hinton stack saturated; focus shifts to snapshot-076 wide-student / 077 per-agent / 081 aggressive-reward (orthogonal directions)
- **§3.4/§3.5 regression**: rare; investigate teacher KL conflict (1750 vs 055 action distribution differences)

## 9. 相关

- [snapshot-071 Pool A original](snapshot-071-pool-A-homogeneous-distill.md) (predecessor, 0.909 tied)
- [snapshot-061 §7.4 055v2_extend SOTA story](snapshot-061-055v2-recursive-distill.md) (1750 discovery)
- [snapshot-075 strategic synthesis §8 DIR-A/B/C](snapshot-075-strategic-synthesis-toward-0.93.md) (alternative directions if 080 saturates)
- [snapshot-081](snapshot-081-aggressive-offense-reward.md) (concurrent orthogonal-reward lane)
