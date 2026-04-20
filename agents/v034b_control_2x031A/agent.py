from cs8803drl.deployment.ensemble_agent import ProbabilityAveragingMixedEnsembleAgent


_CKPT_031A = (
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
    "031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/"
    "checkpoint_001040/checkpoint-1040"
)


class Agent(ProbabilityAveragingMixedEnsembleAgent):
    def __init__(self, env):
        super().__init__(
            env,
            members=[
                {"kind": "team_ray", "checkpoint_path": _CKPT_031A},
                {"kind": "team_ray", "checkpoint_path": _CKPT_031A},
            ],
        )

