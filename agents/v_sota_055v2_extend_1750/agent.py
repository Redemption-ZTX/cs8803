"""Project SOTA submission: 055v2_extend @ ckpt 1750 (self-contained).

Architecture: 031B Siamese dual encoder + cross-attention.
Training: scratch-distill from 5-teacher ensemble (055@1150 + 031B@1220 +
045A@180 + 051A@130 + 056D@1140); recursive distill recipe (LR=3e-4)
extended from iter 1216 → 2000.
Reward: v2 shaping.

Performance (verified 2026-04-21):
- Combined 4000ep baseline WR = 0.9155 (CI [0.908, 0.924])
- H2H vs prior SOTA 055@1150 = 0.538 (z=1.70, p=0.045)
- 4 independent samples (1000+2000+1000) all 0.913-0.917 → stable peak

Layout (self-contained, zip-ready):
    v_sota_055v2_extend_1750/
        agent.py
        __init__.py
        README.md
        params.pkl
        checkpoint_001750/
            checkpoint-1750
            checkpoint-1750.tune_metadata

Usage:
    python -m soccer_twos.watch -m agents.v_sota_055v2_extend_1750
"""
import os
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent
CKPT_PATH = str(_AGENT_DIR / "checkpoint_001750" / "checkpoint-1750")

if not Path(CKPT_PATH).exists():
    raise FileNotFoundError(
        f"Bundled checkpoint missing: {CKPT_PATH}\n"
        "Re-package this agent dir with the checkpoint binary included."
    )
if not (_AGENT_DIR / "params.pkl").exists():
    raise FileNotFoundError(
        f"Bundled params.pkl missing: {_AGENT_DIR / 'params.pkl'}\n"
        "Re-package this agent dir with params.pkl from the trial root."
    )

os.environ.setdefault("TRAINED_RAY_CHECKPOINT", CKPT_PATH)

from cs8803drl.deployment.trained_team_ray_agent import TeamRayAgent  # noqa: E402


class Agent(TeamRayAgent):
    """Team-level SOTA agent (055v2_extend@1750), self-contained package."""
    pass
