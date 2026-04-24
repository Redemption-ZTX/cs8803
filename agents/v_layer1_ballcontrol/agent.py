"""Stone Layered Learning Phase 1 specialist: 101A @ ckpt 460 (self-contained).

Architecture: 031B Siamese dual encoder + cross-attention (same as project SOTA backbone).
Training: scratch 500 iter / 20M steps, vs random opponent only (BASELINE_PROB=0.0),
reward = ball_progress + possession_bonus (no shot / no defense — pure ball-control
specialty). Layer 1 of Stone & Veloso 2000 Layered Learning decomposition.

Performance (verified 2026-04-22):
- Peak baseline 1000ep WR = 0.851 @ ckpt 460 (LATE window, originally missed by
  inline-eval coverage gap; user-caught via supplementary eval)
- vs random (training-distribution) inline = 0.99
- vs 1750 SOTA = sub-frontier expected (this is a SPECIALIST, not generalist)
- §3.1 main vs baseline ≥0.85 HIT (just barely), §3.2 vs random ≥0.95 HIT

Intended use:
- PIPELINE V1 specialist library member (ball-control / Layer 1 role)
- NOT a direct submission candidate (0.851 < 1750 SOTA 0.9155)
- Warm-start source for Stone Layered Phase 2 (pass-decision specialist)

Layout (self-contained, zip-ready):
    v_layer1_ballcontrol/
        agent.py
        __init__.py
        README.md
        params.pkl
        checkpoint_000460/
            checkpoint-460
            checkpoint-460.tune_metadata

Usage:
    python -m soccer_twos.watch -m agents.v_layer1_ballcontrol
"""
import os
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent
CKPT_PATH = str(_AGENT_DIR / "checkpoint_000460" / "checkpoint-460")

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
    """Stone Layered Phase 1 ball-control specialist (101A@460), self-contained package."""
