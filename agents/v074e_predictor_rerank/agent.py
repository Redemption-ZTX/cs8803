"""074E — outcome-predictor-enhanced ensemble (top-K re-rank).

Combines the 074A member set (best-evidence orthogonal frontier triad)
with the calibrated v3 outcome predictor used as the PBRS reward signal
in snapshot-053D. At ``act()`` time, the top-K actions by ensemble
probability are tilted toward higher predicted-win outcomes — but only
when the ensemble's margin between top-1 and top-2 is <0.10 (i.e., the
ensemble is uncertain). See snapshot-074E §2 for the full design rationale.

Environment variables (read at init time):
- ``OUTCOME_RERANK_ENABLE=0`` → disable predictor; behaves as plain 074A.
- ``OUTCOME_RERANK_TOPK=3`` → number of candidate actions considered.
- ``OUTCOME_RERANK_DEVICE=cuda|cpu|auto`` → where to run the predictor.
- ``OUTCOME_RERANK_PREDICTOR_PATH=/abs/path.pt`` → override model weights.
- ``OUTCOME_RERANK_BUFFER=80`` → trajectory window length.

Members reuse 074A:
- ``055@1150`` (distill SOTA anchor)
- ``053Dmirror@670`` (PBRS-only blood — matches predictor training axis)
- ``062a@1220`` (curriculum + no-shape blood)
"""
from cs8803drl.deployment.trained_team_ensemble_next_agent import (
    CKPT_053DMIRROR_670,
    CKPT_055_1150,
    CKPT_062A_1220,
    OutcomePredictorRerankEnsembleAgent,
)


_MEMBERS = [
    ("team_ray", CKPT_055_1150, 1.0),
    ("team_ray", CKPT_053DMIRROR_670, 1.0),
    ("team_ray", CKPT_062A_1220, 1.0),
]


class Agent(OutcomePredictorRerankEnsembleAgent):
    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
