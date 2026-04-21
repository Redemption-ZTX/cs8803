"""074C — H2H-least-correlated ensemble: 055@1150 + 056D@1140 + 053Dmirror@670.

Selected by smallest-margin pairwise H2H as a proxy for the most orthogonal
decision boundaries currently covered by the frontier pool (see
snapshot-074C-h2h-orthogonal.md §2 for the full argument).

H2H table quote (rank.md §5.3):
- ``055@1150 vs 056D@1140 = 0.536`` (z=1.61, p=0.054, **NOT sig**,
  two independent 500ep samples identical due to Unity port-seeded
  determinism) → **marginally tied**, the most even frontier pair.
- ``053Dmirror vs *`` direct H2H is **not yet measured** (noted as
  "未测,需要先补 H2H" — see snapshot-074C §4). We substitute the
  structural argument that 053D's PBRS-only blood is orthogonal to
  055/056D's v2 shape lineage, pending H2H fill.

Equal weights. Uses the ``TeamEnsembleNextAgent`` wrapper.
"""
from cs8803drl.deployment.trained_team_ensemble_next_agent import (
    CKPT_053DMIRROR_670,
    CKPT_055_1150,
    CKPT_056D_1140,
    TeamEnsembleNextAgent,
)


_MEMBERS = [
    ("team_ray", CKPT_055_1150, 1.0),
    ("team_ray", CKPT_056D_1140, 1.0),
    ("team_ray", CKPT_053DMIRROR_670, 1.0),
]


class Agent(TeamEnsembleNextAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
