from cs8803drl.deployment.ensemble_agent import ProbabilityAveragingMixedEnsembleAgent


_CKPT_031B = (
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
    "031B_team_cross_attention_scratch_v2_resume1080/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/"
    "checkpoint_001220/checkpoint-1220"
)


class Agent(ProbabilityAveragingMixedEnsembleAgent):
    def __init__(self, env):
        super().__init__(
            env,
            members=[
                {"kind": "team_ray", "checkpoint_path": _CKPT_031B},
                {"kind": "team_ray", "checkpoint_path": _CKPT_031B},
                {"kind": "team_ray", "checkpoint_path": _CKPT_031B},
            ],
        )
