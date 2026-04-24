"""029B @ ckpt 190 (per-agent SOTA, self-contained).

Architecture: per-agent (336-dim per-agent obs, MAPPO with shared centralized
critic), 512×512 MLP. NOT team-level — each agent gets its own observation.
Training: PBRS handoff from 029A B-warm.
Performance (verified): 1000ep baseline WR ≈ 0.86; one of the project's
strongest per-agent models, used as ensemble member in 034E and 074F lineage.

Layout: self-contained, zip-ready.
"""
import os
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent
CKPT_PATH = str(_AGENT_DIR / "checkpoint_000190" / "checkpoint-190")

if not Path(CKPT_PATH).exists():
    raise FileNotFoundError(f"Bundled checkpoint missing: {CKPT_PATH}")
if not (_AGENT_DIR / "params.pkl").exists():
    raise FileNotFoundError(f"Bundled params.pkl missing: {_AGENT_DIR / 'params.pkl'}")

os.environ.setdefault("TRAINED_RAY_CHECKPOINT", CKPT_PATH)

from cs8803drl.deployment.trained_shared_cc_agent import SharedCCAgent  # noqa: E402


class Agent(SharedCCAgent):
    """Per-agent SOTA agent (029B@190), self-contained package."""
    pass
