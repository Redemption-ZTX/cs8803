"""074B — architectural-diversity ensemble: 055@1150 + 054M@1230 + 062a@1220.

Member rationale (see snapshot-074B-arch-diversity.md §2):
- ``055@1150`` (team_ray, Siamese cross-attn): distillation SOTA anchor.
- ``054M@1230`` (team_ray, MAT-medium **cross-agent** attention): distinct
  attention axis; single-shot 1000ep 0.889 (peak of pre-extend run). This
  is the only cross-agent-attention (as opposed to cross-field) member in
  the 074 family — its failure modes should differ from Siamese lineage.
- ``062a@1220`` (team_ray, Siamese cross-attn): curriculum + no-shape blood,
  chosen instead of 053Dmirror here to keep 074B "pure architecture axis"
  (both 055 and 054M share v2 shaping, so 062a's no-shape data-distribution
  orthogonality complements the architectural one).

Equal weights. Note: extended 054M checkpoints (iter 1550+) are not yet
post-eval'd as of 2026-04-21 06:10 EDT; using iter 1230 (single-shot peak).
"""
from cs8803drl.deployment.trained_team_ensemble_next_agent import (
    CKPT_054M_1230,
    CKPT_055_1150,
    CKPT_062A_1220,
    TeamEnsembleNextAgent,
)


_MEMBERS = [
    ("team_ray", CKPT_055_1150, 1.0),
    ("team_ray", CKPT_054M_1230, 1.0),
    ("team_ray", CKPT_062A_1220, 1.0),
]


class Agent(TeamEnsembleNextAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
