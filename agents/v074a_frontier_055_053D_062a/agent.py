"""074A — deploy-time ensemble: 055@1150 + 053Dmirror@670 + 062a@1220.

Member rationale (see snapshot-074-034-next-deploy-time-ensemble.md §2.1):
- ``055@1150`` (team_ray, combined 2000ep 0.907): distillation SOTA anchor.
- ``053Dmirror@670`` (team_ray, single-shot 1000ep 0.902): PBRS-only blood.
- ``062a@1220`` (team_ray, combined 2000ep 0.892): curriculum + no-shape blood.

Equal weights, probability averaging, greedy by default.
"""
from cs8803drl.deployment.trained_team_ensemble_next_agent import (
    CKPT_053DMIRROR_670,
    CKPT_055_1150,
    CKPT_062A_1220,
    TeamEnsembleNextAgent,
)


_MEMBERS = [
    ("team_ray", CKPT_055_1150, 1.0),
    ("team_ray", CKPT_053DMIRROR_670, 1.0),
    ("team_ray", CKPT_062A_1220, 1.0),
]


class Agent(TeamEnsembleNextAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
