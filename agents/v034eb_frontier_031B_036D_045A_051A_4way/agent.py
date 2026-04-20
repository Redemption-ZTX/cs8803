"""034eb — 4-way ensemble: 031B + 036D + 045A + 051A.

Maximum diversity variant. Adds 036D (defensive_pin/defend-counter) on top of
034ea {031B, 045A, 051A} for 4-way failure mode coverage.

Member individual 1000ep WR: 031B 0.882, 036D 0.860, 045A 0.867, 051A 0.888
(avg 0.874). Inference cost 4× single model.
"""
from cs8803drl.deployment.ensemble_agent import ProbabilityAveragingMixedEnsembleAgent


_MEMBERS = [
    {
        "kind": "team_ray",
        "checkpoint_path": (
            "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
            "031B_team_cross_attention_scratch_v2_resume1080/"
            "TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/"
            "checkpoint_001220/checkpoint-1220"
        ),
    },
    {
        "kind": "shared_cc",
        "checkpoint_path": (
            "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
            "PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/"
            "MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/"
            "checkpoint_000150/checkpoint-150"
        ),
    },
    {
        "kind": "team_ray",
        "checkpoint_path": (
            "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
            "045A_team_combo_on_031A1040_formal_rerun1/"
            "TeamVsBaselineShapingPPOTrainer_Soccer_be409_00000_0_2026-04-19_05-47-13/"
            "checkpoint_000180/checkpoint-180"
        ),
    },
    {
        "kind": "team_ray",
        "checkpoint_path": (
            "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
            "051A_combo_on_031B_with_051reward_512x512_20260419_110852/"
            "TeamVsBaselineShapingPPOTrainer_Soccer_b9914_00000_0_2026-04-19_11-09-13/"
            "checkpoint_000130/checkpoint-130"
        ),
    },
]


class Agent(ProbabilityAveragingMixedEnsembleAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
