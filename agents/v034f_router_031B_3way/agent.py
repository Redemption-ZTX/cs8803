from cs8803drl.deployment.ensemble_agent import HeuristicRoutingMixedEnsembleAgent


_MEMBERS = [
    {
        "name": "031B_anchor",
        "kind": "team_ray",
        "role": "anchor",
        "base_weight": 0.50,
        "checkpoint_path": (
            "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
            "031B_team_cross_attention_scratch_v2_resume1080/"
            "TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/"
            "checkpoint_001220/checkpoint-1220"
        ),
    },
    {
        "name": "036D_specialist",
        "kind": "shared_cc",
        "role": "specialist",
        "base_weight": 0.20,
        "checkpoint_path": (
            "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
            "PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/"
            "MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/"
            "checkpoint_000150/checkpoint-150"
        ),
    },
    {
        "name": "029B_stabilizer",
        "kind": "shared_cc",
        "role": "stabilizer",
        "base_weight": 0.30,
        "checkpoint_path": (
            "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
            "PPO_mappo_029B_bwarm170_to_v2_512x512_formal/"
            "MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/"
            "checkpoint_000190/checkpoint-190"
        ),
    },
]


class Agent(HeuristicRoutingMixedEnsembleAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
