from cs8803drl.deployment.ensemble_agent import ProbabilityAveragingSharedCCEnsembleAgent


_CHECKPOINTS = [
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/"
    "MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190",
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/"
    "MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80",
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/"
    "MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100",
]


class Agent(ProbabilityAveragingSharedCCEnsembleAgent):
    def __init__(self, env):
        super().__init__(env, checkpoint_paths=_CHECKPOINTS)

