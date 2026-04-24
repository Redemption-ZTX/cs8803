# Experiments Backlog

2026-04-20 backlog captured; prioritize after current Tier 1/2 + 062 verdicts.

## 054 series (MAT architecture progression)

- **054L (MAT-large)**: 054M + multi-head attention (4 heads). Intermediate between 054M (0.880 verdict pending 1000ep) and 052 full-transformer (REGRESSION -8~-11pp). Hypothesis: MHA adds capacity without the LayerNorm dual-block that broke 052. Code change ~1h, training ~12h.
- **054T (MAT-transformer-full)**: 054L + 2 transformer layers = close to 052 full version. Last step of the MAT progression. Code change ~1-2h, training ~12h.

## 056 series (LR sweep + PBT)

- **056E lr=5e-4**: Extend upward LR sweep. 3e-4 > 2e-4 > 1e-4 is monotonic at peak — maybe peak not reached. Launcher copy + LR change, training ~12h.
- **056F lr=7e-4**: Continue upward LR sweep. Beyond 056E in case monotonic trend continues. ~12h.
- **056-PBT-full**: Real PBT mechanism — population coordinator + worker weight exchange every N iter + HP mutation (LR ±20%, clip ±20%). Engineering ~1-2 days + training ~12h. High research value but highest cost.

## 058 series (curriculum progression)

After 062 verdicts:
- **058-L3 (frontier-pool opponent)**: add 028A@200 (early weak frontier) as an opponent alongside random+baseline. Tests if `weak self-play` fills the difficulty gap between random and baseline.
- **058-L4 (adaptive RL teacher, PAIRED)**: teacher network learns to propose opponents that maximize student learning progress. Most complex. ~1-2 days engineering.

## 055 series (distillation progression)

After 055v2 verdict:
- **055v3 recursive**: Use 055v2 (expected peak) + 055@1150 + 056D@1140 as teachers. Continued iterative distillation.
- **Temperature sharpening follow-up**: after 063 verdict, extend to `T ∈ {1.5, 3.0, 4.0}` if `T=2.0` is positive or marginal.
- **Logit-level distill**: distill on raw logits instead of factor-probs (more information-rich teacher signal).
- **Progressive distill**: multi-stage schedule (e.g., distill to intermediate checkpoint, then redistill).

## Stone-pipeline backlog (deferred from 6-direction work plan, 2026-04-22)

See [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) for the strategic synthesis covering this backlog and its parent work plan.

### Active deferrals (originally in plan, cut to Wave 2/3)

- **DIR-D-QMIX (deferred from 7-direction → 6-direction plan cut)**: Originally proposed as TPOT-RL aux head (Stone & Veloso 1998), then upgraded to QMIX (Rashid et al. 2018) — monotonic value-function factorization for cooperative MARL. Cut from initial Stone work plan because (a) overlapped with DIR-F VDN, (b) TPOT-RL paper's empirical lift was modest (+12.7pp goal share vs 1998 hand-coded baseline; modern PPO+GAE already covers similar ground). **Recovery condition**: revisit only if [snapshot-102 DIR-F VDN](snapshot-102-dir-F-vdn.md) shows positive lift (combined 2000ep ≥ 0.918) AND we want to test QMIX's monotonic NN mixer as an architecturally-stricter alternative to VDN's sum. If VDN is tied/regression, DIR-D-QMIX recovery is also closed (same monotonic-factor family, same binding constraint). Engineering: ~600 lines (mixer network + training loop change), ~6h GPU.
- **DIR-A Wave 3 — true sub-behavior policies**: After [snapshot-100 DIR-A Wave 2](snapshot-100-dir-A-heuristic-selector.md) (heuristic selector with orthogonal specialists from 081 aggressive + 103-series) is verified, train **true SPL-style specialists** with their own state-space mask (e.g., NEAR-GOAL only sees forward 7 rays) + own action-space constraint (e.g., MID-FIELD only outputs ΔΘ). 4 specialist lanes × ~3h GPU each = 12h. Engineering: env wrapper for state masking + action constraint, ~300 lines. Wave 3 only if Wave 2 confirms heuristic selector framework can break SOTA with orthogonal members. Cross-ref: [snapshot-103 §1.2 H_103-a](snapshot-103-dir-A-wave3-sub-task-specialists.md#12-子假设) — Wave 1 tests scenario init + reward shape only; Wave 3 = + state masking + action constraint.

### Stone-pipeline Wave 2/3 backlog (added 2026-04-22 after Wave 1 verdicts)

- **DIR-G + DIR-E Wave 2 router NN training (REINFORCE)**: After Wave 1 framework infra verified ([snapshot-104 §7](snapshot-104-dir-G-moe-router.md#7-verdict-wave-1-31-hit-uniform-router-framework-工作-beats-dir-a-heuristic-2026-04-22-append-only) DIR-G uniform 0.900, [snapshot-105 §7](snapshot-105-dir-E-option-critic-frozen.md#7-verdict-wave-1-31-hit-random-nn-framework-infra-工作-dir-g-within-se-2026-04-22-append-only) DIR-E random 0.895), train router NN via REINFORCE on episode returns. ~3h GPU each (small NN ~30k params). Wave 2 must wait for orthogonal specialists (081 + 103-series) to be ready — training with current 3 distill-family specialists will likely saturate at the same Wave 1 ceiling because the underlying issue is specialist redundancy, not router quality. Engineering: ~300 LOC training script + 2 new agent modules (`v_moe_router_trained`, `v_option_critic_trained`). DIR-E variant must include entropy regularization on β termination (Bacon 2017 §4 collapse warning).
- **103-series sub-task Wave 2 (state masking + action constraint, deferred from Wave 1)**: After [snapshot-103 §3.1](snapshot-103-dir-A-wave3-sub-task-specialists.md#3-pre-registered-thresholds) sanity gates pass for 103A/B/C, retrain each lane with **(a) state masking** (e.g., 103A INTERCEPTOR only sees forward 60° rays subset, 103B DEFENDER only sees opp-direction rays) **(b) action constraint** (e.g., 103C MID-FIELD-DRIBBLE only outputs ΔΘ turn actions, no sprint). Pushes specialists into sharper behavioral non-overlap with the generalist 1750. Engineering: ~400 LOC env wrapper for state masking + action filter, 3 new launchers. ~12h GPU parallel (3 nodes × 4h). Same architecture (031B) for PIPELINE substitutability.
- **PIPELINE Phase 6 — DAGGER between sub-behaviors**: After 4-5 sub-behaviors trained (081 + 101A + 103A/B/C), run DAGGER iteration where each sub-behavior is corrected by inter-policy labeling (e.g., MID-FIELD policy queried for NEAR-GOAL state when NEAR-GOAL policy fails). This is Stone Layered Learning's bootstrapping idea applied at sub-behavior boundaries. Engineering: ~500 lines DAGGER buffer + label generation, ~4h GPU. Pre-condition: PIPELINE Stage 1-4 ([snapshot-099 §6](snapshot-099-stone-pipeline-strategic-synthesis.md#6-pipeline-5-stage-integration-sketch)) must produce composable specialist library with at least 2 directions positive-lift verified.
- **PIPELINE Phase 7 — Distill from heterogeneous specialist library (re-open distill axis)** (added 2026-04-22): Once the PIPELINE produces a true **specialist library** (101A ball-control + 081 striker + 103A interceptor + 103B defender + 103C dribble + 1750 SOTA + 055@1150 ≈ 7+ genuinely orthogonal teachers, NOT same-family-redundant), distill from this library into a single 031B student. Critical distinction from the 071-079 5/5 saturation cluster: those distill lanes used **same-family teachers** (all 031B-distill descendants), so the diversity bottleneck was specialist redundancy, not student capacity or recipe. With a heterogeneous library where each member is genuinely behaviorally distinct (different scenarios trained, different reward shapes, different optimal phase), the Hinton 2015 student-beats-teacher paradigm has fresh signal. Pre-registered hypothesis: combined 4000ep peak ≥ 0.92 = decisive break of the 0.91 distill ceiling that has held against 5 same-family probes. Engineering: ~200 lines launcher (reuse `cs8803drl/branches/team_siamese_distill.py` ensemble distill code), ~12h GPU. **Sequencing (user directive 2026-04-22)**: this is a Phase 7 extension, NOT a PIPELINE V1 prerequisite. PIPELINE V1 (Phase 1-5: specialists + selector + composition) must be eval'd FIRST to confirm the assembled library actually beats the 0.91 ceiling. If PIPELINE V1 ≥ 0.92, only THEN consider Phase 7 distill on top of the confirmed-good library. Do not blindly distill from a mediocre PIPELINE — wasted compute, and the resulting student is bounded by the library's already-mediocre quality. **Pre-condition for Phase 7 launch**: (a) PIPELINE V1 combined 2000ep ≥ 0.92, AND (b) at least 4 sub-task / specialty lanes show meaningful behavioral difference (e.g., loss-bucket distribution disjoint, action distribution KL > 0.5 between specialist pairs).
- **DIR-E + DIR-G Hybrid selector**: If both DIR-E and DIR-G Wave 2 hit independently ([snapshot-105 §8.5](snapshot-105-dir-E-option-critic-frozen.md#85-hybrid-dir-e--dir-g-backlog)), test combination: DIR-E β termination decides "when to switch", DIR-G router decides "which to switch to". Theoretical best-of-both-worlds, but engineering ~600 LOC + retrain.

## General infrastructure

- **Document `run_with_flags.sh`** in `docs/architecture/engineering-standards.md` as the mandatory wrapper for all training/eval/capture/H2H.
- **Checkpoint sanitization** (to support proper recursive distill): strip `teacher_model.*` keys from checkpoints trained as distill students before using them as teachers. Apply in `cs8803drl/core/checkpoint_utils.py` and teacher-load paths. _This is Option B for 055v2 fix and is being implemented 2026-04-20._

## Notes

- **Never downgrade (rejected Option A's)**: do not drop items from pool / reduce sample / skip validations to make workload easier. See `feedback_no_downgrade.md` memory.
- **Always write snapshot-NNN.md first** before launching any new experiment.
- **Run_with_flags.sh wrapper mandatory** for all lanes.
