"""034ea — 3-way ensemble: 031B + 045A + 051A.

Hypothesis (snapshot-034 / snapshot-051 §8.6): replacing 036D + 029B (both
defensive_pin/territorial_dominance dominant) with 045A (wasted_possession
55%) + 051A (ultra-defensive turtle, tail_x -1.20) yields more orthogonal
failure-mode coverage than 034e {031B, 036D, 029B}.

Member individual 1000ep WR: 031B 0.882, 045A 0.867, 051A 0.888 (avg 0.879
vs 034e avg 0.863). Probability averaging expected to give +2-3pp over
member mean if failure modes are truly orthogonal.
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
