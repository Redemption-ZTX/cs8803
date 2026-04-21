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

## General infrastructure

- **Document `run_with_flags.sh`** in `docs/architecture/engineering-standards.md` as the mandatory wrapper for all training/eval/capture/H2H.
- **Checkpoint sanitization** (to support proper recursive distill): strip `teacher_model.*` keys from checkpoints trained as distill students before using them as teachers. Apply in `cs8803drl/core/checkpoint_utils.py` and teacher-load paths. _This is Option B for 055v2 fix and is being implemented 2026-04-20._

## Notes

- **Never downgrade (rejected Option A's)**: do not drop items from pool / reduce sample / skip validations to make workload easier. See `feedback_no_downgrade.md` memory.
- **Always write snapshot-NNN.md first** before launching any new experiment.
- **Run_with_flags.sh wrapper mandatory** for all lanes.
