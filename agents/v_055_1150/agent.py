"""055 @ ckpt 1150 (prior project SOTA, self-contained).

Architecture: 031B Siamese dual encoder + cross-attention (team-level).
Training: scratch-distill from 034E 3-teacher ensemble (031B@1220 + 036D@150 +
029B@190); LR=1e-4, v2 reward shaping.
Performance (verified): combined 2000ep baseline WR = 0.907.

Layout: self-contained, zip-ready.
    v_055_1150/
        agent.py / __init__.py / README.md
        params.pkl
        checkpoint_001150/{checkpoint-1150,checkpoint-1150.tune_metadata}
"""
import os
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent
CKPT_PATH = str(_AGENT_DIR / "checkpoint_001150" / "checkpoint-1150")

if not Path(CKPT_PATH).exists():
    raise FileNotFoundError(f"Bundled checkpoint missing: {CKPT_PATH}")
if not (_AGENT_DIR / "params.pkl").exists():
    raise FileNotFoundError(f"Bundled params.pkl missing: {_AGENT_DIR / 'params.pkl'}")

os.environ.setdefault("TRAINED_RAY_CHECKPOINT", CKPT_PATH)

from cs8803drl.deployment.trained_team_ray_agent import TeamRayAgent  # noqa: E402


class Agent(TeamRayAgent):
    """Team-level prior-SOTA agent (055@1150), self-contained package."""
    pass
