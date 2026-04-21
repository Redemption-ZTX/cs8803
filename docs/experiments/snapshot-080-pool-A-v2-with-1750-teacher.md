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

## 7. Verdict

_Pending — training first ckpt reached 18:47, ETA ~04:47 EDT next day_

## 8. 后续

- **§3.1 / §3.2 HIT**: Pool A v2@peak 成为新 SOTA; 启动 Gen 4 recursive distill from Pool A v2
- **§3.3 tied**: 证实 Hinton stack saturated; focus shifts to snapshot-076 wide-student / 077 per-agent / 081 aggressive-reward (orthogonal directions)
- **§3.4/§3.5 regression**: rare; investigate teacher KL conflict (1750 vs 055 action distribution differences)

## 9. 相关

- [snapshot-071 Pool A original](snapshot-071-pool-A-homogeneous-distill.md) (predecessor, 0.909 tied)
- [snapshot-061 §7.4 055v2_extend SOTA story](snapshot-061-055v2-recursive-distill.md) (1750 discovery)
- [snapshot-075 strategic synthesis §8 DIR-A/B/C](snapshot-075-strategic-synthesis-toward-0.93.md) (alternative directions if 080 saturates)
- [snapshot-081](snapshot-081-aggressive-offense-reward.md) (concurrent orthogonal-reward lane)
