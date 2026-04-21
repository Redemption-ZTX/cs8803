"""074F — Weighted deploy-time ensemble with 055v2_extend@1750 NEW SOTA anchor.

Members (weighted prob-avg by deploy-time 1000ep+ WR):
- 055v2_extend@1750 (weight 0.5): combined 4000ep 0.9155, H2H vs 055 = 0.538 sig.
- 055@1150        (weight 0.3): prior SOTA, combined 2000ep 0.907.
- 055v2@1000      (weight 0.2): combined 3000ep 0.909 tied.

Motivation: 074A-E uniform averaging failed because members all 0.89-0.91 near-
ceiling and same-family correlated. 074F addresses this by (a) dominant-weighting
the truly-stronger 1750 member, (b) using all distill-family members (avoids the
arch-mix regression of 074B).
"""
from cs8803drl.deployment.trained_team_ensemble_next_agent import TeamEnsembleNextAgent, CKPT_055_1150

CKPT_055V2_1000 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_a1e7d_00000_0_2026-04-20_14-27-48/"
    "checkpoint_001000/checkpoint-1000"
)

CKPT_055V2_EXTEND_1750 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "055v2_extend_resume_1210_to_2000_20260421_030743/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/"
    "checkpoint_001750/checkpoint-1750"
)

_MEMBERS = [
    ("team_ray", CKPT_055V2_EXTEND_1750, 0.5),
    ("team_ray", CKPT_055_1150, 0.3),
    ("team_ray", CKPT_055V2_1000, 0.2),
]


class Agent(TeamEnsembleNextAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
