"""074F — Weighted deploy-time ensemble with 055v2_extend@1750 NEW SOTA anchor.

Self-contained package: all 3 member checkpoints + params.pkl bundled inside
this directory, so the dir is zip-ready and portable (no external paths).

Members (weighted prob-avg):
- 055v2_extend@1750 (weight 0.5): combined 4000ep 0.9155, H2H vs 055 = 0.538 sig
- 055@1150         (weight 0.3): prior SOTA, combined 2000ep 0.907
- 055v2@1000       (weight 0.2): combined 3000ep 0.909 tied

Motivation: 074A-E uniform averaging failed because members all 0.89-0.91
near-ceiling and same-family correlated. 074F dominant-weights the truly
stronger 1750 member while keeping the distill-family stack.

Layout:
    v074f_weighted_1750_sota/
        agent.py
        __init__.py
        README.md
        m1_055v2_extend_1750/
            params.pkl
            checkpoint_001750/{checkpoint-1750,checkpoint-1750.tune_metadata}
        m2_055_1150/
            params.pkl
            checkpoint_001150/{checkpoint-1150,checkpoint-1150.tune_metadata}
        m3_055v2_1000/
            params.pkl
            checkpoint_001000/{checkpoint-1000,checkpoint-1000.tune_metadata}

Usage:
    python -m soccer_twos.watch -m agents.v074f_weighted_1750_sota
"""
from pathlib import Path

from cs8803drl.deployment.trained_team_ensemble_next_agent import TeamEnsembleNextAgent

_AGENT_DIR = Path(__file__).resolve().parent

_MEMBER_PATHS = [
    ("m1_055v2_extend_1750", "checkpoint_001750", "checkpoint-1750", 0.5),
    ("m2_055_1150",          "checkpoint_001150", "checkpoint-1150", 0.3),
    ("m3_055v2_1000",        "checkpoint_001000", "checkpoint-1000", 0.2),
]

_MEMBERS = []
for member_dir, ckpt_dir, ckpt_file, weight in _MEMBER_PATHS:
    ckpt_path = _AGENT_DIR / member_dir / ckpt_dir / ckpt_file
    params_path = _AGENT_DIR / member_dir / "params.pkl"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Bundled member checkpoint missing: {ckpt_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Bundled member params.pkl missing: {params_path}")
    _MEMBERS.append(("team_ray", str(ckpt_path), weight))


class Agent(TeamEnsembleNextAgent):
    """Weighted ensemble Agent (074F) — self-contained package."""

    def __init__(self, env):
        super().__init__(env, members=_MEMBERS)
