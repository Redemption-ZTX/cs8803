"""031B @ ckpt 1220 (architecture-baseline frontier, self-contained).

Architecture: team-level Siamese dual encoder + 1-head cross-attention
(4 tokens × 64 dim) — the project's foundational architecture step.
Training: scratch with v2 reward shaping (no distill, no warm-start).
Performance (verified): 1000ep baseline WR = 0.882.

Used as the 074F ensemble's anchor architecture and as the warm-start
base for 053D / 062 / 071 / 072 / 073 / 080 lanes.

Layout: self-contained, zip-ready.
"""
import os
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent
CKPT_PATH = str(_AGENT_DIR / "checkpoint_001220" / "checkpoint-1220")

if not Path(CKPT_PATH).exists():
    raise FileNotFoundError(f"Bundled checkpoint missing: {CKPT_PATH}")
if not (_AGENT_DIR / "params.pkl").exists():
    raise FileNotFoundError(f"Bundled params.pkl missing: {_AGENT_DIR / 'params.pkl'}")

os.environ.setdefault("TRAINED_RAY_CHECKPOINT", CKPT_PATH)

from cs8803drl.deployment.trained_team_ray_agent import TeamRayAgent  # noqa: E402


class Agent(TeamRayAgent):
    """Team-level architecture-baseline agent (031B@1220), self-contained."""
    pass
