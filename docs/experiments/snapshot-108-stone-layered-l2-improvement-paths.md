# SNAPSHOT-108: 103A-wd v2 mechanism analysis + improvement paths

- 日期: 2026-04-23 (写) / 2026-04-23 01:36 (walk-back update)
- 状态: 路径分析 + 路径 EV 重排 — **v2 5000ep verdict TIED 1750, walk-back from "marginal SOTA" earlier**

## 0. CRITICAL UPDATE 2026-04-23 01:34 (added post-5000ep verdict, before path execution)

**v2 ckpt 400 combined 5000ep eval result**: mean 0.9042 over 5 samples (0.920/0.900/0.875/0.906/0.920) = TIED with 1750 真值 0.9066 within SE.

**Walk-back from earlier "marginal NEW SOTA" claim**:
- v1 combined 3000ep 0.9157 was sample-variance outlier (3 samples narrow lucky range)
- v2 with 5 samples gives wider true distribution → real value ~0.91
- Stone Layered L2 = TIED at 0.91 ceiling, NOT above 1750

**EV repricing for paths in §3 below**:
- P1 longer training: EV LOWER (won't break 0.91 ceiling, just plateau more stably)
- P2 stronger anchor: EV LOWER (already at 1750 plateau, more anchoring just stays at 1750)
- P3 multi-scenario rotation: EV MEDIUM (broader scenario ≠ different ceiling, but might prevent overfit)
- P4 ensemble teacher: EV LOW (079/080 saturation at 0.91 already proves single-teacher distill ceiling)
- P5 recursive Layer 3: EV LOW (Layer 2 itself didn't break ceiling)

**NEW top P0 (per snapshot-109 audit)**: state/action bottleneck specialists per Wang/Stone/Hanna 2025 paper 4 — implement specialist that has REDUCED obs space (e.g., MID-FIELD only sees ball+goalposts) and REDUCED action space (e.g., 1-D ΔΘ kick angle only). This is the only path that fundamentally CHANGES the architecture-imposed 0.91 ceiling. Estimated: ~600 LOC + 8h GPU per specialist + ensemble compose.

**Implication**: §4 推荐 below revised — top P1 (longer training) demoted; new top P0 (state/action bottleneck) added. See snapshot-109 §3 for implementation list.

---
- 前置:
  - [snapshot-103 §7F](snapshot-103-dir-A-wave3-sub-task-specialists.md) — 103A-warm-distill verdict 历史 (v1 §7F.1-§7F.5b, v2 §7F.5c)
  - [snapshot-107](snapshot-107-stone-layered-p2-passdecision.md) — Phase 2 pass-decision design (104A in progress)
  - Memory `feedback_1750_sota_overstated.md` — 1750 真值 0.9066 (5×1000ep fresh)
  - Memory `feedback_opponent_pool_no_resample.md` — BUG-1 fix doctrine
  - 代码 ref: [team_siamese_distill.py:140-220](../../cs8803drl/branches/team_siamese_distill.py) (`_distill_updates` counter / `_current_alpha`)
  - Launcher ref: [_launch_103A_warm_distill_v2_bugfix.sh](../../scripts/eval/_launch_103A_warm_distill_v2_bugfix.sh)

## 1. v2 mechanism analysis

v2 vs v1 single-shot delta @ckpt 400 = +0.007 (913 → 920); terminal @ckpt 467 vs v1 489 = +0.022 (895 → 917); mid-late @340/350 = +0.021. The two bugs were independent in mechanism.

**BUG-2 (anchor schedule) — dominant contribution.** v1 had `DECAY_UPDATES=4000`; with ~78 PPO updates / iter, α reached 0 by iter ~53/500 (10% of training). For 90% of training the policy ran with **zero KL distill anchor**, so it drifted away from the 1750 prior. The v1 trajectory shows exactly this: peak 0.913 @400 then collapse to 0.895 @489 (terminal -0.018). v2 with `DECAY_UPDATES=39000` + `ALPHA_FINAL=0.005` (residual anchor through iter 500) keeps α in the [0.005, 0.05] band the entire run. The +0.022 terminal lift is the cleanest fingerprint of "anchor present in late window prevents drift". Estimated contribution to ckpt-400 +0.007: ≈ +0.005 to +0.006.

**BUG-1 (per-episode opp resample) — secondary.** Pre-fix: each of 40 workers lazy-locked a single opponent for the entire run. With BASELINE_PROB=0.7 the aggregate ratio was preserved (~28 always-baseline workers + ~12 always-random), but per-episode opponent variation = 0. Post-fix: every reset re-samples opponent. Mechanism is regularization — workers see both distributions instead of specializing. Expected effect: small (~+0.001-0.002 on ckpt 400) because aggregate distribution unchanged; visible mostly as variance reduction across ckpt cluster (v2 has tighter [0.891, 0.920] vs v1 [0.895, 0.923]).

**Better-aligned vs different-policy?** Late-window plateau at 0.917 (terminal) and 0.920 (peak) sits **above** the v1 peak of 0.923 collapsing to 0.895 — i.e., v2 holds the plateau the anchor was pulling toward, not a divergent exploit. The +0.013 single-shot above 1750 真值 (0.9066) is consistent with "1750-aligned + INTERCEPTOR specialty bonus", not "different policy hitting orthogonal weakness". This matches the §7F.11 Mechanism reading: 1750 generalist + BALL_DUEL specialty refinement = additive on baseline distribution.

## 2. Remaining limits

Five binding constraints, ordered by my estimate of which would lift the ceiling most:

1. **Encoder capacity (031B Siamese ~0.46M params)** — snapshot-082/083 closed the architecture axis at ~0.91; 1750 itself sits at 0.9066, our v2 sits at ~0.92 single-shot. We are < 0.015 above the architecture-saturated ceiling. Pushing to 0.93+ likely requires a different encoder (wider student, attention-stacked), not more training of 031B.

2. **Single sub-task scope (INTERCEPTOR only).** Scenario init focuses on BALL_DUEL (~25-30% of game states by 100A-class phase classifier estimate). The remaining 70-75% (NEAR_GOAL / POSITIONING / MID_FIELD) get only the warm + KL anchor signal — the specialty signal doesn't reach them. The +0.009 above 1750 真值 likely all comes from BALL_DUEL state subset.

3. **Anchor strength α_final=0.005 vs exploration tradeoff.** Pinning closer to 1750 would tie performance to 1750 (=0.9066 ceiling); too loose lets late drift. Current 0.005 is unstudied — could be over-tight (limiting BALL_DUEL specialization gain) or under-tight (still allows drift). Sensitivity analysis missing.

4. **Aux reward magnitude (tackle 0.05, clearance 0.03).** Same numbers as 103A v1; never re-swept. With episodes averaging 5-10 tackle/clearance events, total aux signal ~0.2-0.5/episode — same order as v2 base shaping. Could be in either direction.

5. **Training budget 500 iter (wall cut).** v1 iter 489 collapsed; v2 iter 467 still climbing (peak @400, terminal @467 = 0.917 ≈ ckpt 400 0.920). Untested whether more iter saturates or pushes higher. Cheapest test.

Note the warm source itself (1750 真值 0.9066) is **not** a hard ceiling — v2 already exceeds it by ~+0.013 single-shot — so "warm-source-bound" is no longer the binding limit it would have been pre-correction.

## 3. Path proposals

| # | Path | Hypothesis | Expected lift | GPU cost | Risk |
|---|---|---|---|---|---|
| **P1** | **Longer training (700-1000 iter) on v2 recipe** | Terminal @467 still at 0.917 = climbing; 250-500 more iter under maintained α schedule may extend plateau or peak | +0.002-0.005 single-shot (= -0.005 to +0.010 combined) | ~3h (extend, no new infra) | Low — worst case is plateau; v2 has no late-drift mechanism since anchor stays on |
| **P2** | **Stronger residual anchor sweep (α_final ∈ {0.002, 0.01, 0.02})** | Find the right anchor strength; current 0.005 is unstudied | +0.000-0.005 (one cell may be optimal) | 3 × 4h = 12h GPU | Medium — α_final=0.02 may pin too close to 1750 ceiling (=tied); α_final=0.002 may re-introduce drift |
| **P3** | **Multi-scenario rotation (INTERCEPTOR + DEFENDER + DRIBBLE per-episode random)** | Specialty signal reaches more state subsets, not just BALL_DUEL | +0.005-0.010 single-shot | 4-5h GPU + ~150 LOC env extension (DEFENDER and DRIBBLE scenario init already designed in snapshot-103) | Medium — 3 scenarios with conflicting reward signals may dilute each other; mitigation = identical aux reward across all |
| **P4** | **Ensemble teacher distill (1750 + 103A-wd v2 @400 + 055@1150)** | 3-teacher KL pool reduces single-teacher bias; v2 itself becomes part of the prior | +0.003-0.008 single-shot (Pool A v2 in snapshot-080 family showed marginal patterns) | ~5h (teacher loading 3× memory; existing infra in `team_siamese_distill.py`) | Medium — ensemble teacher direction has shown TIED-not-above pattern (079, 080); needs novel marginal source |
| **P5** | **Recursive Layer (use v2@400 as warm + own-anchor for Layer 3)** | Same recipe as 103A-wd but with stronger warm (0.92 vs 0.9066); analogous to 055v2 → 055v3 progression | +0.000-0.005 (very speculative; saturated SOTA tier patterns historically tied) | ~5h | High — 079 single-teacher recursive at saturated SOTA-tier showed TIED (peak 0.914 vs teacher 0.909 = +0.005 noise); same risk here |

EV/cost ranking: **P1 > P3 > P2 > P4 > P5**. P1 is cheapest signal-extraction (just continue an existing run-shape); P3 has the highest expected lift but needs scenario engineering already mostly designed; P2 is the missing sensitivity sweep; P4/P5 follow saturated-SOTA-tier patterns that have historically tied.

## 4. 推荐 next launch

**Fire P1 first** (longest-running training, single-variable change from v2). Concrete recipe: identical to `_launch_103A_warm_distill_v2_bugfix.sh` except `MAX_ITERATIONS=900`, `TIME_TOTAL_S=25200` (7h), and recompute `TEAM_DISTILL_DECAY_UPDATES = 900 × 78 = 70200` so anchor still spans full training. Cost ~7h on a free H100. If P1 lifts to ≥0.925 combined, Layer 2 paradigm has more room and P3 becomes worth firing on top of P1's terminal ckpt as warm.

**Fire P3 second** (parallel on a different node when one frees). Engineering cost is the binding factor — DEFENDER scenario init exists in 103B design but never landed; DRIBBLE in 103C same. Implement both as direct extensions of the existing `expert_coordination.py` pattern (~150 LOC), then launcher adds `SCENARIO_RESET=multi_subtask` + uniform-random per-episode pick. Identical aux reward (tackle 0.05 / clearance 0.03 / opp_progress 0.02) across all three to avoid signal conflict.

**Defer P4/P5** until P1+P3 verdict — they sit in the 079/080 saturation regime and their expected lift is below SE noise floor.

**Do NOT fire P2 standalone** — fold the α_final sweep into P3's design as a 2-cell sub-sweep ({0.005, 0.01}) on top of multi-scenario, costing only +1 lane.

## 5. Cross-references

- [snapshot-103 §7F.5c](snapshot-103-dir-A-wave3-sub-task-specialists.md) — 103A-wd v2 Stage 1 result (peak 0.920 @400 / terminal 0.917 @467)
- [snapshot-103 §7F.11](snapshot-103-dir-A-wave3-sub-task-specialists.md) — Mechanism reading (5 components necessary; encoder saturation context)
- [snapshot-107](snapshot-107-stone-layered-p2-passdecision.md) — orthogonal Phase 2 path (Layer 1 warm, NEW pass scenario); 104A in progress with same DECAY_UPDATES=39000 + ALPHA_FINAL=0.005 recipe (good)
- [snapshot-079](snapshot-079-055v3-recursive-distill.md) — recursive single-teacher distill at SOTA-tier = TIED precedent for P5
- [snapshot-080](snapshot-080-pool-A-v2-with-1750-teacher.md) — multi-teacher pool experience for P4
- [snapshot-082](snapshot-082-hierarchical-two-stream-siamese.md) / [snapshot-083](snapshot-083-per-ray-attention.md) — encoder-axis saturation evidence (ceiling at ~0.91 for 031B family)
- Memory `feedback_1750_sota_overstated.md` — 1750 真值 0.9066, our v2 at 0.92 single-shot = +0.013 ~2.3σ
- Memory `feedback_inline_eval_noise.md` — single-shot 1000ep noise; verify any P1/P3 result with combined ≥3000ep before SOTA claim
