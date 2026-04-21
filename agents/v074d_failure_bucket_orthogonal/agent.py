"""074D — failure-bucket-orthogonal ensemble.

Selects 3 members with the most orthogonal failure bucket distributions
(see snapshot-074D-failure-bucket-orthogonal.md §2). Bucket counts were
measured from ``docs/experiments/artifacts/failure-cases/<ckpt>_baseline_500``:

| member        | late_def | low_poss | unclear | poor_conv | total |
|---------------|---------:|---------:|--------:|----------:|------:|
| 055@1150      |       29 |       26 |       3 |         1 |    60 |
| 062a@1220     |       29 |       30 |       8 |         2 |    72 |
| 031B@1220     |       29 |       16 |      12 |         4 |    63 |
| 055v2@1000    |       19 |       21 |       1 |         6 |    50 |
| 056D@1140     |       23 |       21 |       7 |         2 |    56 |
| 053Acont@430  |       16 |       14 |       9 |         3 |    47 |
| 051A@130      |       32 |       17 |       3 |         4 |    57 |
| 043C'@480     |       32 |       18 |       8 |         1 |    61 |

Picks:

1. **055@1150** — high late_defensive + low-possession, very low unclear_loss.
2. **055v2@1000** — elevated **poor_conversion** (6/50) is distinctive;
   runner-up in every other bucket.
3. **053Acont@430** — higher **unclear_loss** share (9/47 ≈ 19%),
   + has PBRS blood so loss-attribution distribution differs.

Net effect: 055 anchors late_def + low_poss; 055v2 covers poor_conversion;
053Acont covers unclear_loss. 031B is a tempting 4th for unclear (12/63),
but we keep it to 3 members to stay within inference budget.

All three are team_ray; equal weights.
"""
from cs8803drl.deployment.trained_team_ensemble_next_agent import (
    CKPT_055_1150,
    TeamEnsembleNextAgent,
)

# Paths for the 2 "non-canonical" members (not in the top helper constants):
CKPT_055V2_1000 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_a1e7d_00000_0_2026-04-20_14-27-48/"
    "checkpoint_001000/checkpoint-1000"
)
CKPT_053ACONT_430 = (
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
    "053Acont_iter200_to_500_20260419_194712/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_23ae7_00000_0_2026-04-19_19-47-35/"
    "checkpoint_000430/checkpoint-430"
)


_MEMBERS = [
    ("team_ray", CKPT_055_1150, 1.0),
    ("team_ray", CKPT_055V2_1000, 1.0),
    ("team_ray", CKPT_053ACONT_430, 1.0),
]


class Agent(TeamEnsembleNextAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
