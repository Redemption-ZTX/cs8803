"""074G — Weighted ensemble with TRUE orthogonal-reward member (081 aggressive).

Members:
- 055v2_extend@1750 (weight 0.5): NEW SOTA, combined 4000ep 0.9155 (v2-shape distill family).
- 081_aggressive@970 (weight 0.3): orthogonal-reward specialist, 0.826 baseline single-shot
  but qualitatively different action distribution (no defense, no possession, shoot-spam).
- 055@1150 (weight 0.2): prior SOTA, distill-family stabilizer.

Hypothesis: 081's failure modes (low_possession likely 80%+) are non-overlapping with
the v2-shape family's (late_defensive_collapse + possession_stolen). When ensemble
averages, 1750/055 dominate normal play but 081's vote pulls the ensemble toward more
aggressive offense in stalemate states → potential break of the 0.91 ceiling.
Different from 074F where all 3 members were v2-family correlated.
"""
from cs8803drl.deployment.trained_team_ensemble_next_agent import TeamEnsembleNextAgent, CKPT_055_1150

CKPT_055V2_EXTEND_1750 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "055v2_extend_resume_1210_to_2000_20260421_030743/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_d761e_00000_0_2026-04-21_03-08-04/"
    "checkpoint_001750/checkpoint-1750"
)

CKPT_081_AGGRESSIVE_970 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "081_aggressive_offense_scratch_20260421_184522/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_d3c3b_00000_0_2026-04-21_18-45-42/"
    "checkpoint_000970/checkpoint-970"
)

_MEMBERS = [
    ("team_ray", CKPT_055V2_EXTEND_1750, 0.5),
    ("team_ray", CKPT_081_AGGRESSIVE_970, 0.3),
    ("team_ray", CKPT_055_1150, 0.2),
]


class Agent(TeamEnsembleNextAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
