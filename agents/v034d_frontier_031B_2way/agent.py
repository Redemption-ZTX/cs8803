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
]


class Agent(ProbabilityAveragingMixedEnsembleAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
