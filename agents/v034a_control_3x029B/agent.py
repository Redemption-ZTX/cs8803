from cs8803drl.deployment.ensemble_agent import ProbabilityAveragingSharedCCEnsembleAgent


_CKPT_029B = (
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/"
    "MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190"
)


class Agent(ProbabilityAveragingSharedCCEnsembleAgent):
    def __init__(self, env):
        super().__init__(env, checkpoint_paths=[_CKPT_029B, _CKPT_029B, _CKPT_029B])

