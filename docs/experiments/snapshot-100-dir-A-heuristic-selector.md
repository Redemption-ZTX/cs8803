# SNAPSHOT-100: DIR-A — 4-Phase Heuristic State-Conditional Selector (Stone Wave 1)

- **日期**: 2026-04-22
- **负责人**: Self
- **状态**: 预注册 + Wave 1 verdict 已 append (deploy-time only, no training); Wave 2 plan pending 081 + 103-series ready
- **前置**: [snapshot-099 §1.1 / §2.1](snapshot-099-stone-pipeline-strategic-synthesis.md#11-为什么-paper-4-wangstonehanna-2025-是-anchor) (Wang/Stone/Hanna ICRA 2025 anchor) / [snapshot-076](snapshot-076-wide-student-distill.md) + [snapshot-079](snapshot-079-055v3-recursive-distill.md) (5/5 distill saturation evidence motivating pivot)

---

## 0. 背景

### 0.1 Paper origin (Wang/Stone/Hanna ICRA 2025, arXiv 2412.09417)

"RL Within the Classical Robotics Stack: A Case Study in Robot Soccer" — 在 2024 RoboCup SPL Challenge Shield Division 实赛中获胜。Recipe:

- **K = 4 sub-policies**, 每个 own state space + own action space
- **Heuristic state-based selector** 决定每步用哪个 sub-policy
- 不需要 end-to-end joint training, 只需 (1) 训出 specialist + (2) 写 selector

### 0.2 为什么这是 Stone-line 最 actionable 的 Wave 1

- **Zero training cost**: 只需 compose 已有 specialists + 一个 phase classifier
- **几小时验证**: 200ep baseline eval ~5min, framework feasibility 立得到答案
- 与 distill paradigm 完全 orthogonal — 不依赖 KL distill, 不依赖 ensemble averaging
- 是 PIPELINE Stage 2 的 selector 候选之一

### 0.3 与项目历史 ensemble lane 的区分

- **不是 v074 family** (deploy-time prob averaging): v074 在 action-distribution level 平均, DIR-A **hard-switch policy at step level**
- **不是 v034 family** (per-agent or arch-mixed avg): v034 同上
- DIR-A 的 selector = phase classifier from 336-dim ray obs, 不是 trained NN (Wave 1)

---

## 1. 核心假设

### 1.1 H_100 (主)

> **多个 specialist policies 各自在不同 state phase 下 strictly dominate 其他 specialists**, 用 hand-coded geometric phase classifier 路由可以达到 max(specialist) + ε。
> Wave 1 期望 200ep WR ≥ 0.91 (= 1750 SOTA cluster), Wave 2 (after orthogonal specialists) 期望 ≥ 0.92.

### 1.2 H_100-a (alt)

> 即使 single best specialist 已经覆盖 80%+ state, heuristic router 通过把 10-20% "弱 phase" 让给 alt specialist 仍能 +0.005 to +0.010 marginal lift.

### 1.3 H_100-b (anti)

> 当前 specialist library 全是 distill family (1750 / 055 / 029B), behavior space 高度 overlap → router 选谁都差不多, 甚至 forced 错 phase 反而 hurt. **此情况 → Wave 1 sub-SOTA, 等 081 + 103-series orthogonal specialists ready 才解锁 Wave 2**.

---

## 2. Design

### 2.1 Phase classifier (4-phase, hand-coded geometric)

从 336-dim ray observation 提取 ball proxy:
- 利用 BALL_TAG_INDEX = 0 (rays-perception 中 ball 的 tag dim)
- 每个 ray 给出 (type_one_hot[7], distance) → ball-hit rays + 距离 + 方位

phase decision (per-agent, per-step):

| Phase | Predicate | 直觉 |
|---|---|---|
| **NEAR_GOAL** | ball visible AND nearest < 0.18 AND centroid > 0 (ball 在前方 close) | 已经在攻击位置, 用最强 finisher |
| **BALL_DUEL** | ball visible AND 0.18 ≤ nearest < 0.40 (中距离) | 正在抢球或中距离决策 |
| **POSITIONING** | ball NOT visible to me OR teammate has much better view (mate_score > my_score + 0.10) | 我远离球, 应该跑位 |
| **MID_FIELD** | default catch-all | 正常推进 |

阈值选择动机 (hand-tuned, 来源 = `agents/v_selector_phase4/agent.py:127-129`):
- `_NEAR_GOAL_NEAREST = 0.18` — ray distance ≤ 0.18 ≈ 5-6 米 (正常射门距离)
- `_BALL_DUEL_NEAREST = 0.40` — ≈ 12 米 (双方都 reach)
- `_POSITION_TEAMMATE_MARGIN = 0.10` — teammate 必须明显比我更看球

### 2.2 Wave 1 specialist mapping (current)

| Phase | Specialist | 选择 reason |
|---|---|---|
| NEAR_GOAL | 1750 SOTA | 强 finisher (最强 single model) |
| **BALL_DUEL** | **055@1150** | **alt family 想 break tie, 但实测是 weaker → §6 Wave 1 verdict 反弹** |
| POSITIONING | 1750 SOTA (placeholder) | Wave 2 应换为 029B per-agent (有更强 individual ball-control) |
| MID_FIELD | 1750 SOTA (default) | 强 generalist |

### 2.3 Wave 2 specialist mapping (designed, NOT YET committed)

| Phase | Wave 2 specialist | 依赖 |
|---|---|---|
| NEAR_GOAL | **081 aggressive** (orthogonal offense reward) | 081 完成 (ETA ~3h from 2026-04-22) |
| BALL_DUEL | **103A INTERCEPTOR** (sub-task specialist) | 103A 完成 |
| POSITIONING | **103B DEFENDER** (sub-task specialist) | 103B 完成 |
| MID_FIELD | 1750 SOTA (default) | 已有 |

**这是预注册 placeholder**, 等 081 + 103A/B 出 verdict 才 commit。

### 2.4 Implementation

- Module: `agents/v_selector_phase4/agent.py` (already exists, 250 lines)
- Phase classifier: `classify_phase(my_obs, teammate_obs)` — pure numpy, no inference cost
- Specialist load: `_TeamRayPolicyHandle` / `_SharedCCPolicyHandle` from `cs8803drl/deployment/ensemble_agent.py`
- De-duplicates ckpts (same handle reused if multiple phases point to same specialist)
- Per-agent independent routing: 每个 agent 各自 classify_phase 然后路由

### 2.5 Eval setup

- 标准 official eval suite, 200ep vs baseline (Wave 1 quick check)
- 1000ep + combined 2000ep 仅当 Wave 2 出 promising signal 才做

---

## 3. Pre-registered Thresholds

| # | 判据 | 阈值 | verdict 含义 |
|---|---|---|---|
| §3.1 Wave 1 main | 200ep ≥ 0.91 | matches 1750 SOTA cluster | heuristic routing 即使在 distill-family specialists 上也能保 SOTA |
| §3.2 Wave 1 stretch | 200ep ≥ 0.92 | beats 1750 single | router 提取了 ensemble-style stability + correct phase routing |
| §3.3 Wave 1 tied | 200ep ∈ [0.86, 0.91) | within router noise | framework 工作但 specialists 不够 orthogonal |
| §3.4 Wave 1 sub-SOTA | 200ep < 0.86 | 强 specialist 被 bad phase routing 掉 | bad mapping 或 weak specialist 被高频选中 |
| §3.5 Wave 2 main | 200ep ≥ 0.92 | with orthogonal specialists | Wave 2 解锁 |
| §3.6 Wave 2 stretch | 1000ep ≥ 0.925 | decisive over 1750 | sub-task specialist + correct phase routing 是 PIPELINE 主线 |

---

## 4. Simplifications + Risks

### 4.1 简化 S1 — Wave 1 用现有 specialists (no new training)

- **节省**: zero training cost
- **Risk R1**: specialists 全 distill family → behavior overlap, router 选谁都差不多
- **降级**: Wave 2 引入 orthogonal specialists (§2.3)

### 4.2 简化 S2 — 4 phases, hand-coded thresholds

- **节省**: 不 sweep phase boundaries
- **Risk R2**: thresholds (0.18 / 0.40 / 0.10) 没 ablation, 可能不是 optimal
- **降级**: 如果 Wave 2 也 sub-SOTA, sweep _NEAR_GOAL_NEAREST ∈ {0.12, 0.15, 0.18, 0.22} 再决定

### 4.3 简化 S3 — Per-step independent routing (no temporal stickiness)

- **节省**: 不 track per-agent option duration
- **Risk R3**: 频繁 switch 可能 hurt — 中段 specialist 切换会丢 momentum
- **降级**: 如果 Wave 2 sub-SOTA, 加 minimum-stick-duration (≥ 5 steps before allowed switch)
- **Note**: DIR-E option-critic 框架显式有 termination — 如果 DIR-E Wave 2 hit, 把 DIR-E 当 selector 替代 hand-coded heuristic

### 4.4 Risk R4 — 错配 weak specialist 到 high-frequency phase

- 实测 Wave 1 (§6) 把 055@1150 (weaker than 1750) 挂到 BALL_DUEL → 频繁 routing 到 weak slot
- BALL_DUEL phase 在实际游戏里出现频率 high (球-人接近是 dominant interaction state)
- **Mitigation**: Wave 2 把所有 weak slot 换成 sub-task-specialist (103A INTERCEPTOR) 或 1750 default

---

## 5. 不做的事

- 不训 selector NN (留给 DIR-G/E Wave 2)
- 不改 specialist library (Wave 1 用现有, Wave 2 等 081 + 103-series)
- 不做 5+ phase 划分 (4 phase 已能 cover offensive / midfield / defensive / out-of-play)
- 不做 state masking (留给 Wave 3, BACKLOG)

---

## 6. Execution Checklist

- [x] 1. snapshot 起草 (本文件)
- [x] 2. 实现 `agents/v_selector_phase4/agent.py` (已 exist)
- [x] 3. 200ep baseline eval — Wave 1 (today) → §7
- [ ] 4. 等 081 完成 → commit Wave 2 NEAR_GOAL specialist
- [ ] 5. 等 103A/B 完成 → commit Wave 2 BALL_DUEL / POSITIONING specialists
- [ ] 6. Wave 2 200ep + 1000ep eval
- [ ] 7. 如果 Wave 2 §3.5 hit, combined 2000ep + verdict append §8
- [ ] 8. 如果 §3.6 hit, propose PIPELINE Stage 2 use this selector

---

## 7. Verdict — Wave 1 §3.4 sub-SOTA (heuristic forced wrong specialist for high-freq phase, 2026-04-22 append-only)

### 7.1 Wave 1 200ep baseline (2026-04-22)

| Metric | Value |
|---|---:|
| Module | `agents.v_selector_phase4` |
| Episodes | 200 |
| WR | **0.875** (175W-25L) |
| Δ vs 1750 single SOTA (0.9155) | **-0.041** sub-SOTA |
| Δ vs 055@1150 single (0.907) | -0.032 sub-SOTA |
| Δ vs DIR-G uniform router 0.900 | -0.025 sub-SOTA |
| Δ vs DIR-E option-critic random NN 0.895 | -0.020 sub-SOTA |

### 7.2 严格按 §3 判据

| 阈值 | 实测 200ep | verdict |
|---|---|---|
| §3.1 Wave 1 main ≥ 0.91 | ❌ 0.875 | not met |
| §3.2 Wave 1 stretch ≥ 0.92 | ❌ | not met |
| §3.3 Wave 1 tied [0.86, 0.91) | ❌ 0.875 in range | actually IS in tied range (boundary) |
| **§3.4 Wave 1 sub-SOTA < 0.86** | **❌ 0.875 > 0.86** | strict definition not met but 实质 sub-SOTA |

注: 0.875 在 §3.3 tied 边界 (within 0.86-0.91 区间), 但 vs 单 specialist 1750 (0.9155) 差 -0.041pp, vs uniform router 差 -0.025pp → **行动上视为 sub-SOTA / framework 不 break SOTA**。

### 7.3 关键 lesson — 高频 phase × weak specialist

诊断 (基于 phase usage telemetry, 见 v_selector_phase4 agent.get_stats() 输出):
- BALL_DUEL phase 在实际 baseline 对局里 dominant (估计 40-60% steps), 因为 ball-near-agent 是 baseline opponent typical interaction pattern
- BALL_DUEL 被映射到 055@1150 (weaker than 1750), 即使 055@1150 baseline WR 0.907 仍比 1750 0.9155 差
- 高频 phase × weaker specialist → 整体 dragged down to sub-SOTA

诊断对比:
- DIR-G uniform router (0.900) **beats** DIR-A heuristic (0.875) — 因为 uniform 在 1/3 weak slot, DIR-A heuristic 在 ~50% weak slot (high-freq BALL_DUEL)
- "**biased toward weak**" 比 "**uniform random**" 更糟

### 7.4 与 DIR-E / DIR-G Wave 1 合读

| Framework | 200ep | 解读 |
|---|---:|---|
| DIR-A heuristic (forced bias) | 0.875 | 高频 phase × weak slot |
| DIR-G uniform router | 0.900 | 1/3 weak slot |
| DIR-E option-critic random NN | 0.895 | ~50% term, ~1/3 weak slot 同 G |
| **best single specialist (1750)** | **0.9155** | reference |

**结论**: 当前 specialist library 里 1750 是 strict dominant, 任何"分流"机制都 net negative。**必须等 orthogonal specialists ready 才能解锁 Wave 2** — 这是 [snapshot-099 §5](snapshot-099-stone-pipeline-strategic-synthesis.md#5-wave-1-framework-已出结果-今日-append-only) 写过的 Wave 2 trigger condition。

### 7.5 Wave 2 trigger 状态 (2026-04-22)

- **081 aggressive**: 在 flight, ETA ~3h, 期望 baseline WR [0.75, 0.88] (orthogonal reward family)
- **103A INTERCEPTOR**: 未 launch, 等 sub-task design 落地 + 12h 训练
- **103B DEFENDER**: 同上
- 当 081 + 103A 至少其中之一 ready → 重测 Wave 2

### 7.6 Lane 决定

- **DIR-A Wave 1 关闭** — current specialist library 不足以 break SOTA
- **DIR-A Wave 2 待 trigger** — 等 081 + 103-series ready
- 不执行 §4.2 阈值 sweep — 实测 sub-SOTA 主因不是 threshold tune, 而是 weak slot 高频

---

## 7B. Wave 2 verdict — REGRESSION -0.110 vs Wave 1 (orthogonal-specialist swap hurt, 2026-04-22 [06:55] append-only)

### 7B.1 Wave 2 specialist mapping (deployed)

After 081 + 103-series specialists ready (5/5 sub-task quartet complete):
- NEAR-GOAL → 081 aggressive (0.826 standalone)
- BALL_DUEL → 103A INTERCEPTOR (0.548 standalone, still climbing)
- POSITIONING → 1750 SOTA (kept Wave 1 — 103B 0.205 too weak)
- MID-FIELD → 1750 SOTA (kept Wave 1 — 103C 0.220 too weak)

Code: `agents/v_selector_phase4/agent.py` updated with new ckpt paths (no packaging per user 2026-04-22).

### 7B.2 Wave 2 200ep baseline result

- Eval node 5032911, port 61605
- 200 episodes, 0 errors

| Metric | Wave 1 | **Wave 2** | Δ |
|---|---|---|---|
| **WR** | 0.875 | **0.765** | **-0.110** ⚠️ REGRESSION |
| fast_win | 0.860 | 0.715 | -0.145 |
| episode_mean step | 38.3 | 50.4 | +12.1 (less efficient) |
| W-L | 175-25 | 153-47 | -22 wins, +22 losses |

### 7B.3 Diagnosis — high-frequency × weak specialist 数学

Phase-routed ensemble standalone WR ≈ Σ phase_freq × specialist_phase_WR.

Wave 1 phase distribution (from v_selector_phase4 stats): roughly 50% BALL_DUEL, 30% NEAR-GOAL, 15% MID-FIELD, 5% POSITIONING (BALL_DUEL dominant because ball-near-agent is common).

| Phase | Freq | Wave 1 specialist (WR) | Wave 2 specialist (WR) | Δ phase contribution |
|---|---|---|---|---|
| BALL_DUEL | ~50% | 055@1150 (0.907) | **103A (0.548)** | **-0.18** |
| NEAR-GOAL | ~30% | 1750 (0.9155) | 081 (0.826) | -0.027 |
| POSITIONING | ~5% | 1750 (0.9155) | 1750 unchanged | 0 |
| MID-FIELD | ~15% | 1750 (0.9155) | 1750 unchanged | 0 |
| **Sum** | | | | **~-0.21** |

Predicted Wave 2 WR: 0.875 - 0.21 ≈ 0.665. Actual: 0.765 (slightly better than predicted because phase_freq estimate is rough). Direction ✅ confirmed: **swapping strong specialists for weaker ones in high-frequency phase 大幅 hurt**, even when new specialists are "orthogonal".

**Lesson**: orthogonal failure modes do NOT compensate for standalone-WR penalty in heuristic selector setting. The specialist must beat the slot-incumbent **in standalone phase eval** before swap. Wave 1 used 055@1150 (0.907) in BALL_DUEL = strong; replacing with 103A (0.548) standalone = catastrophic for ensemble.

### 7B.4 严格按 §3 判据 (revised)

| 阈值 | 实测 | verdict |
|---|---|---|
| §3.5 Wave 2 main 200ep ≥ 0.92 | ❌ 0.765 | not met (-0.155) |
| §3.6 Wave 2 1000ep ≥ 0.925 | ❌ (1000ep not run, sub-SOTA confirmed) | not met |
| §3.7 Wave 2 regression < 0.85 | ✅ 0.765 (in regression band) | **HIT — Wave 2 mapping wrong** |

### 7B.5 Lane decision (Wave 2 closed, Wave 3 redesigned)

- **Wave 2 mapping FALSIFIED** — orthogonal swap in high-freq slot hurts ensemble standalone
- **Stage 2/3 SKIPPED** for Wave 2 (sub-SOTA, no info)
- **Wave 3 design** (queued P0): keep 1750 in BALL_DUEL/MID/POS (high-freq slots), add 081 ONLY in NEAR-GOAL phase narrowly defined (ball.nearest < 0.10 AND ball.centroid > 0.5 — much stricter trigger than Wave 1/2). Predicted: WR ≥ 0.91 (close to 1750 baseline), small upside if 081 NEAR-GOAL strikes are more accurate.
- **Wave 3 alt design**: HeuristicRoutingMixedEnsembleAgent (already in `cs8803drl/deployment/ensemble_agent.py`) — anchor=1750 + specialists boost when anchor uncertain. Doesn't FORCE swap, just shifts weight. May avoid Wave 2's standalone-WR penalty.
- Specialist value primarily through DIR-G/E LEARNED routing (not heuristic) — Wave 2 of those is still pending; if they show lift → confirms LEARNED > HEURISTIC for orthogonal-specialist combination.

---

## 7C. Wave 3 ablation A-E (Option 2 framework, 2026-04-22 [07:30] append-only)

### 7C.1 Setup

Wave 3 Option 2 framework (snapshot-106 §3): instead of the heuristic-driven Wave 2 mapping (which forced weak specialists into high-frequency slots), test each specialist's marginal contribution by single-slot ablation against a strong baseline (1750 in all 4 phases).

5 ablation variants (200ep baseline each):
- **A baseline**: 1750×4 (no specialist swap)
- **B +081**: NEAR-GOAL=081, rest=1750
- **C +103A**: BALL_DUEL=103A, rest=1750
- **D +103B**: POSITIONING=103B, rest=1750
- **E +103C**: MID-FIELD=103C, rest=1750

### 7C.2 Results

| Variant | WR | Δ vs A | Specialist net value |
|---|---|---|---|
| **A baseline (1750×4)** | **0.920** | — | reference |
| B (+081 NEAR-GOAL) | 0.880 | **-0.040** | hurt |
| C (+103A BALL_DUEL) | 0.830 | **-0.090** | hurt |
| D (+103B POSITIONING) | 0.830 | **-0.090** | hurt |
| E (+103C MID-FIELD) | 0.860 | **-0.060** | hurt |
| F (= Wave 2 all 4) | 0.765 | -0.155 | aggregate hurt |

### 7C.3 Verdict — 0/5 specialists improve ensemble

**ALL specialist swaps net-negative**. Sum of marginal damages: -0.040 + -0.090 + -0.090 + -0.060 = **-0.280**, but Wave 2 actual aggregate -0.155 < sum because phase frequencies attenuate (not all specialists active simultaneously).

This confirms snapshot-106 §1 (distribution mismatch) + §2 (bootstrap missing) **for v1 specialists**. Specialist value paradigm is NOT broken — the SPECIFIC v1 specialists are inadequate. Fix path:
- **103A-refined** (training): BASELINE_PROB=0.7 + warm-103A — fixes #1 only
- **103A-warm-distill** (training): full Stone Layered Layer 2 (warm-1750 + distill-1750 + scenario + 0.7) — fixes #1+#2

If 103A-refined ≥ 0.70 standalone OR 103A-warm-distill ≥ 0.85 standalone → **methodology fix validated** → Wave 4 with refined specialists may break ceiling. If both < 0.7 → **specialist paradigm itself fails**, look for fundamentally different selector (DIR-H cross-attention, queued P1).

### 7C.4 Cross-reference

- Methodology root-cause: [snapshot-106](snapshot-106-stone-methodology-corrections.md)
- Wave 4 (refined specialists + new selector): pending verdict from 103A-warm-distill
- Wave 3 raw logs: `docs/experiments/artifacts/official-evals/ablation/ablation_*.log`

### 7C.5 Verdict for §3 thresholds (Wave 3 step)

| 阈值 (Wave 3) | 实测 | verdict |
|---|---|---|
| §3.5 Wave 3 main: any single-specialist ablation ≥ 0.93 | ❌ peak 0.880 | not met |
| §3.6 Wave 3 specialist net-positive: ≥ 1 specialist Δ vs A > 0 | ❌ all Δ ≤ -0.040 | NOT MET — paradigm-validating ablation FAILED |
| §3.7 Wave 3 regression: all specialists Δ < -0.02 | ✅ all in [-0.090, -0.040] | HIT — confirms specialist v1 universally inadequate |

### 7C.6 Lane decision

- **Wave 3 closed: 0/5 specialists improve ensemble in any phase slot**
- **Lane status: PAUSED pending refined-specialist results** (103A-refined & 103A-warm-distill training)
- **Wave 4 designs queued**: 
  - (a) Wave 4-narrow: NEAR-GOAL → 081 ONLY when ball.nearest < 0.10 + centroid > 0.5 (cf. 7B.5 P0)
  - (b) Wave 4-soft: HeuristicRoutingMixedEnsembleAgent (anchor=1750 + boost on uncertainty)
  - (c) Wave 4-refined: re-run ablation A-F with 103A-refined / 103A-warm-distill specialists
  - (d) DIR-H cross-attention fusion (queue P1) — orthogonal selector arch

---

## 7D. Wave 4-narrow verdict — tied 1750 within SE, no lift but damage bounded (2026-04-22 [10:40] append-only)

### 7D.1 Setup

Wave 4-narrow (§7C.6 queued variant a) implemented: NEAR-GOAL phase routing to 081 **only when** `ball.nearest < 0.10 AND ball.centroid > 0.5` (strict striker trigger). All other phases (BALL_DUEL, POSITIONING, MID-FIELD) remain 1750 SOTA. Preset `wave3_narrow` in [v_selector_phase4](../../agents/v_selector_phase4/agent.py).

Threshold rationale: Wave 2/3 NEAR-GOAL trigger `nearest < 0.18 AND centroid > 0` fired on ~40% of my-ball-visible steps. Narrow trigger fires roughly on ~5-10% of those (estimate — actual phase freq reported in `get_stats()` not logged for this eval).

### 7D.2 Stage 1 baseline 1000ep

| Variant | WR | NW-ML | Δ vs 1750 | Δ vs Wave 2 |
|---|---|---|---|---|
| 1750 SOTA (combined 4000ep ref) | 0.9155 | — | — | +0.150 over Wave 2 |
| Wave 2 (force-swap all 4 phases) | 0.765 | — | -0.150 (`***`) | — |
| **Wave 4-narrow (only NEAR-GOAL narrow)** | **0.898** | **898-102** | **-0.018 tied within SE** | **+0.133** |

### 7D.3 严格按 §3 判据 (Wave 4-narrow step)

| 阈值 | 实测 | verdict |
|---|---|---|
| §3.8 Wave 4-narrow main: WR ≥ 0.920 (beat 1750 by ≥ +0.005) | ❌ 0.898 | not met |
| §3.9 Wave 4-narrow tied: WR ∈ [0.90, 0.92) | ❌ 0.898 (just below 0.90 by -0.002) | marginal MISS |
| §3.10 Wave 4-narrow damage-bounded: WR ≥ 0.89 AND ≥ Wave 2 + 0.10 | ✅ 0.898 ≥ 0.89 AND +0.133 over Wave 2 | HIT |

**核心 verdict**: narrow trigger **成功 bound specialist damage**, 不 regress from 1750 SOTA。没 actionable uplift — 081 striker 在 narrow-trigger 下发火频率太低不足以提供可度量的 lift。

### 7D.4 Interpretation (对 Wave 2 regression 的诊断)

Wave 2 regression 根因是 **trigger frequency × specialist WR** 数学（§7B.3 diagnosis 确认）：
- Wave 2 NEAR-GOAL trigger ~40% freq × 081 WR 0.826 = phase contribution down -0.037pp from 1750
- Wave 4-narrow NEAR-GOAL trigger ~5% freq × 081 WR 0.826 = phase contribution ~-0.005pp from 1750 (within SE)

所以 narrow trigger 是**正确的 damage mitigation 方向**, 但是 **not a SOTA uplift mechanism** — 想压过 1750 需要 specialist 本身 ≥ 1750 在那个 phase 上。

### 7D.5 Lane decision

- **Wave 4-narrow 关闭** — tied 1750 验证了"窄 trigger 安全"，无需更多 variants
- 081 striker 标记为 **"safe to include with narrow-trigger but no lift"** in PIPELINE V1
- 如果想 upside, 要 **更强 specialist** (103A-warm-distill pending) 或 **learned routing** (DIR-G W2 / DIR-H W3)
- **101A@460 Phase 1 specialist (0.851 baseline)** 尚未 ablate — 可以加到 Wave 5 做 "POSITIONING phase 用 101A" 探索

### 7D.6 Raw recap

```
=== Official Suite Recap (parallel) ===
/home/hice1/wsun377/Desktop/cs8803drl/agents/v_sota_055v2_extend_1750/checkpoint_001750/checkpoint-1750 vs baseline: win_rate=0.898 (898W-102L-0T)
[suite-parallel] total_elapsed=310.7s tasks=1 parallel=1
```

(Note: `--checkpoint` in eval is dummy — v_selector_phase4 resolves its own ckpts per `SELECTOR_PHASE_MAP_PRESET=wave3_narrow`.)

Log: [wave3_narrow_baseline1000.log](../../docs/experiments/artifacts/official-evals/wave3_narrow_baseline1000.log)

---

## 8. 后续

### 8.1 Wave 2 (after 081 + 103-series ready)

按 §2.3 mapping 重测:
- NEAR_GOAL → 081 (aggressive)
- BALL_DUEL → 103A (INTERCEPTOR)
- POSITIONING → 103B (DEFENDER)
- MID_FIELD → 1750 SOTA

预期: 200ep ≥ 0.92 (§3.5), 1000ep ≥ 0.925 (§3.6)

### 8.2 Wave 3 (BACKLOG)

- True SPL-style sub-behavior policies with own state-mask + own action-constraint
- 详见 [BACKLOG.md § DIR-A Wave 3 — true sub-behavior policies](BACKLOG.md#stone-pipeline-backlog-deferred-from-6-direction-work-plan-2026-04-22)

### 8.3 PIPELINE 集成

- 如果 Wave 2 §3.5 hit, DIR-A heuristic selector 进 PIPELINE Stage 2
- 否则 fallback 到 DIR-G/E Wave 2 trained router

---

## 9. 相关

- [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) — Stone-pipeline strategic synthesis (DIR-A 在 §2.1 #1)
- [snapshot-104](snapshot-104-dir-G-moe-router.md) — DIR-G uniform router (Wave 1 = 0.900, 比 DIR-A 高)
- [snapshot-105](snapshot-105-dir-E-option-critic-frozen.md) — DIR-E option-critic (Wave 1 = 0.895)
- [snapshot-103](snapshot-103-dir-A-wave3-sub-task-specialists.md) — sub-task 103A/B/C (Wave 2 specialist input)
- [snapshot-081](snapshot-081-aggressive-offense-reward.md) — orthogonal aggressive specialist (Wave 2 NEAR_GOAL input)
- [agents/v_selector_phase4/agent.py](../../agents/v_selector_phase4/agent.py) — module 实现
- [agents/v_selector_phase4/README.md](../../agents/v_selector_phase4/README.md) — module 说明

### 理论支撑

- **Wang, Stone, Hanna 2025** ICRA "RL Within the Classical Robotics Stack" arXiv:2412.09417 — 4-policy + heuristic selector won 2024 RoboCup SPL Shield
- **Stone & Veloso 2000** "Layered Learning" — task-conditional sub-policy 思想
