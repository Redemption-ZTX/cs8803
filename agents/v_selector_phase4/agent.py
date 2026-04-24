"""Phase-conditional selector ensemble (Stone DIR-A simplified, Wave 1).

Inspired by Wang/Stone/Hanna ICRA 2025 ("Reinforcement Learning Within the
Classical Robotics Stack: A Case Study in Robot Soccer", arXiv 2412.09417):
4 specialist sub-policies + heuristic state-based selector. Unlike
`v074f_weighted_1750_sota` (action-space probability averaging), this is a
HARD STATE-CONDITIONAL ROUTER — each step, every agent independently
classifies its current "phase" from its 336-dim observation and routes the
forward call to the best specialist for that phase.

Phase definitions (computed per-agent, per-step):
- NEAR-GOAL    : ball very close (nearest < 0.18) AND ball ahead (centroid > 0)
                 → use the strongest raw policy (1750 SOTA)
- BALL-DUEL    : ball visible AND moderately close (0.18 <= nearest < 0.40)
                 → use a different family policy to break tie patterns (055@1150)
- POSITIONING  : I cannot see ball OR teammate has much better ball view
                 → use per-agent SOTA which has stronger individual ball-control (029B)
- MID-FIELD    : default catch-all
                 → use 1750 SOTA

This is the WAVE-1 placeholder: real specialists for each phase will be
plugged in once 081 (aggressive offense scratch, currently training) and
later DIR-B layered specialists become available.

Layout: self-contained, references sibling agent dirs by relative path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np

from soccer_twos import AgentInterface

_AGENT_DIR = Path(__file__).resolve().parent
_AGENTS_ROOT = _AGENT_DIR.parent
_REPO_ROOT = _AGENTS_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cs8803drl.deployment.ensemble_agent import _TeamRayPolicyHandle, _SharedCCPolicyHandle  # noqa: E402

# Register ALL custom models (mirrors trained_team_ensemble_next_agent.py setup)
# so any team_ray ckpt with cross-attn / distill / ensemble-distill / two-stream
# / per-ray architecture can be loaded by _TeamRayPolicyHandle.
from cs8803drl.branches.team_siamese import (  # noqa: E402
    register_team_siamese_cross_agent_attn_medium_model,
    register_team_siamese_cross_agent_attn_model,
    register_team_siamese_cross_attention_model,
    register_team_siamese_model,
    register_team_siamese_transformer_model,
    register_team_siamese_transformer_mha_model,
    register_team_siamese_transformer_min_model,
)
from cs8803drl.branches.team_siamese_distill import (  # noqa: E402
    register_team_siamese_distill_model,
    register_team_siamese_ensemble_distill_model,
)
from cs8803drl.branches.team_siamese_two_stream import (  # noqa: E402
    register_team_siamese_two_stream_model,
)
from cs8803drl.branches.team_siamese_per_ray import (  # noqa: E402
    register_team_siamese_per_ray_model,
)
from cs8803drl.branches.team_action_aux import register_team_action_aux_model  # noqa: E402

register_team_siamese_model()
register_team_siamese_cross_attention_model()
register_team_siamese_cross_agent_attn_model()
register_team_siamese_cross_agent_attn_medium_model()
register_team_siamese_transformer_model()
register_team_siamese_transformer_mha_model()
register_team_siamese_transformer_min_model()
register_team_siamese_distill_model()
register_team_siamese_ensemble_distill_model()
register_team_siamese_two_stream_model()
register_team_siamese_per_ray_model()
register_team_action_aux_model()

# --------------------------------------------------------------------------
# Ray-perception phase classifier (no extra inference, pure numpy on obs)
# --------------------------------------------------------------------------

_RAY_BLOCK_SIZE = 8
_RAY_TYPE_DIM = 7
_BALL_TAG_INDEX = 0  # confirmed empirically per cs8803drl/branches/expert_coordination.py


def _ball_proxy(obs: np.ndarray) -> Dict[str, float]:
    """Reduce 336-dim ray obs to a few ball-related scalars.

    Returns dict with: visible (0/1), count (frac of rays hitting ball),
    nearest (1.0=far, 0.0=close), centroid (-1=hard left, +1=hard right).
    """
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size == 0 or arr.size % _RAY_BLOCK_SIZE != 0:
        return {"visible": 0.0, "count": 0.0, "nearest": 1.0, "centroid": 0.0}
    blocks = arr.reshape(-1, _RAY_BLOCK_SIZE)
    type_scores = blocks[:, :_RAY_TYPE_DIM]
    distances = np.clip(blocks[:, _RAY_TYPE_DIM], 0.0, 1.0)
    num_rays = max(1, blocks.shape[0])
    ray_positions = np.linspace(-1.0, 1.0, num_rays, dtype=np.float32)
    mask = type_scores[:, _BALL_TAG_INDEX] > 0.5
    if not np.any(mask):
        return {"visible": 0.0, "count": 0.0, "nearest": 1.0, "centroid": 0.0}
    nearest = float(np.min(distances[mask]))
    return {
        "visible": 1.0 if nearest < 0.98 else 0.0,
        "count": float(np.mean(mask)),
        "nearest": nearest,
        "centroid": float(np.mean(ray_positions[mask])),
    }


# Phase IDs
NEAR_GOAL = "near_goal"
BALL_DUEL = "ball_duel"
POSITIONING = "positioning"
MID_FIELD = "mid_field"

# Hyperparameters (selector thresholds) — "default" set used by wave1/wave2/all ablation_*
_NEAR_GOAL_NEAREST = 0.18
_NEAR_GOAL_CENTROID = 0.0
_BALL_DUEL_NEAREST = 0.40
_POSITION_TEAMMATE_MARGIN = 0.10  # teammate's ball score must beat self's by this much

# Wave 3 narrow-trigger thresholds — halved NEAR_GOAL bounds so swap fires much less
# often (minimal damage if specialist is worse than SOTA, only activate on very-confident
# "I'm the striker with forward centering" states). Addresses Wave 2 regression
# root cause per task-queue DIR-A Wave 3 redesign.
_NARROW_NEAR_GOAL_NEAREST = 0.10    # was 0.18 → ~half the distance threshold
_NARROW_NEAR_GOAL_CENTROID = 0.5    # was 0.0 → must be strongly forward-right-biased
_NARROW_BALL_DUEL_NEAREST = 0.40    # unchanged
_NARROW_POSITION_MARGIN = 0.10      # unchanged

# Module-level thresholds, set at import time based on preset. Wave 3 overrides defaults.
_ACTIVE_NEAR_GOAL_NEAREST = _NEAR_GOAL_NEAREST
_ACTIVE_NEAR_GOAL_CENTROID = _NEAR_GOAL_CENTROID
_ACTIVE_BALL_DUEL_NEAREST = _BALL_DUEL_NEAREST
_ACTIVE_POSITION_MARGIN = _POSITION_TEAMMATE_MARGIN


def classify_phase(my_obs: np.ndarray, teammate_obs: Optional[np.ndarray]) -> str:
    """Decide which sub-policy should handle this step for THIS agent."""
    me = _ball_proxy(my_obs)
    if (me["visible"] > 0.5
            and me["nearest"] < _ACTIVE_NEAR_GOAL_NEAREST
            and me["centroid"] > _ACTIVE_NEAR_GOAL_CENTROID):
        return NEAR_GOAL
    if me["visible"] > 0.5 and me["nearest"] < _ACTIVE_BALL_DUEL_NEAREST:
        return BALL_DUEL
    if teammate_obs is not None:
        mate = _ball_proxy(teammate_obs)
        my_score = (1.0 - me["nearest"]) + 0.5 * me["count"]
        mate_score = (1.0 - mate["nearest"]) + 0.5 * mate["count"]
        if mate["visible"] > 0.5 and mate_score > my_score + _ACTIVE_POSITION_MARGIN:
            return POSITIONING
    return MID_FIELD


# --------------------------------------------------------------------------
# Specialist registry — Wave 1 placeholder mapping
# --------------------------------------------------------------------------

# Each tuple: (kind, ckpt_path)
# kind ∈ {"team_ray", "shared_cc"}
_SOTA = (
    "team_ray",
    str(_AGENTS_ROOT / "v_sota_055v2_extend_1750" / "checkpoint_001750" / "checkpoint-1750"),
)
_055_1150 = (
    "team_ray",
    str(_AGENTS_ROOT / "v_055_1150" / "checkpoint_001150" / "checkpoint-1150"),
)
_029B_190 = (
    "shared_cc",
    str(_AGENTS_ROOT / "v_029B_190" / "checkpoint_000190" / "checkpoint-190"),
)
# Wave 2 specialists (Stone DIR-A Wave 3 sub-task lanes + 081 aggressive)
# References ckpt paths directly (no packaging — see user 2026-04-22 directive).
_081_AGGRESSIVE = (
    "team_ray",
    "/storage/ice1/5/1/wsun377/ray_results_scratch/081_aggressive_offense_scratch_20260421_184522/TeamVsBaselineShapingPPOTrainer_Soccer_d3c3b_00000_0_2026-04-21_18-45-42/checkpoint_000970/checkpoint-970",
)
_101A_BALLCONTROL = (
    "team_ray",
    "/storage/ice1/5/1/wsun377/ray_results_scratch/101A_layered_p1_ballcontrol_20260422_014241/TeamVsBaselineShapingPPOTrainer_Soccer_21c17_00000_0_2026-04-22_01-43-04/checkpoint_000460/checkpoint-460",
)
_103A_INTERCEPTOR = (
    "team_ray",
    # 2026-04-22: upgraded from v1 @500 (0.548 baseline) → resume @780 (0.620, +0.072pp)
    "/storage/ice1/5/1/wsun377/ray_results_scratch/103A_interceptor_resume_20260422_102921/TeamVsBaselineShapingPPOTrainer_Soccer_b46b6_00000_0_2026-04-22_10-29-43/checkpoint_000780/checkpoint-780",
)
# 2026-04-23: 103A-warm-distill v2 ckpt 400 — Stone Layered L2 ULTIMATE (combined 5000ep 0.9042 = TIED 1750 SOTA at 0.91 plateau)
# Used to retest DIR-A ablation A-F per snapshot-109 §5: prior Wave 3 verdicts (0/5 specialists improve) were based on weak v1 specialists.
# v2 is SOTA-tier, so swapping it into BALL_DUEL slot tests if heuristic selector framework can extract value when specialist is no longer broken.
_103A_WD_V2_400 = (
    "team_ray",
    "/storage/ice1/5/1/wsun377/ray_results_scratch/103A_warm_distill_v2_bugfix_20260422_202340/TeamVsBaselineShapingPPOTrainer_Soccer_bcfb1_00000_0_2026-04-22_20-24-06/checkpoint_000400/checkpoint-400",
)
_103B_DEFENDER = (
    "team_ray",
    "/storage/ice1/5/1/wsun377/ray_results_scratch/103B_defender_20260422_022803/TeamVsBaselineShapingPPOTrainer_Soccer_795ae_00000_0_2026-04-22_02-28-28/checkpoint_000400/checkpoint-400",
)
_103C_DRIBBLE = (
    "team_ray",
    "/storage/ice1/5/1/wsun377/ray_results_scratch/103C_dribble_20260422_023142/TeamVsBaselineShapingPPOTrainer_Soccer_fb211_00000_0_2026-04-22_02-32-06/checkpoint_000300/checkpoint-300",
)

# --------------------------------------------------------------------------
# Phase-map presets (Wave 1 / Wave 2 / Wave 3 ablation variants)
# --------------------------------------------------------------------------
# Controlled by env var `SELECTOR_PHASE_MAP_PRESET`:
#   wave1                  - original Wave 1 mapping (3 distill-family generalists)
#   wave2                  - Wave 2 mapping (NEAR-GOAL→081, BALL_DUEL→103A, POS/MID→1750) — 2026-04-22 regressed -0.110
#   ablation_A_baseline    - all 4 phases use 1750 SOTA (control)
#   ablation_B_081         - +081 in NEAR-GOAL only (rest = 1750)
#   ablation_C_103A        - +103A in BALL_DUEL only
#   ablation_D_103B        - +103B in POSITIONING only
#   ablation_E_103C        - +103C in MID-FIELD only
#   ablation_F_all         - all 4 specialists active (= wave2 mapping but explicit)
# Default if env var unset = wave2 (current Stone DIR-A active config).
_PRESETS = {
    "wave1": {
        NEAR_GOAL: _SOTA,
        BALL_DUEL: _055_1150,
        POSITIONING: _SOTA,
        MID_FIELD: _SOTA,
    },
    "wave2": {
        NEAR_GOAL: _081_AGGRESSIVE,
        BALL_DUEL: _103A_INTERCEPTOR,
        POSITIONING: _SOTA,
        MID_FIELD: _SOTA,
    },
    "ablation_A_baseline": {
        NEAR_GOAL: _SOTA,
        BALL_DUEL: _SOTA,
        POSITIONING: _SOTA,
        MID_FIELD: _SOTA,
    },
    "ablation_B_081": {
        NEAR_GOAL: _081_AGGRESSIVE,
        BALL_DUEL: _SOTA,
        POSITIONING: _SOTA,
        MID_FIELD: _SOTA,
    },
    "ablation_C_103A": {
        NEAR_GOAL: _SOTA,
        BALL_DUEL: _103A_INTERCEPTOR,
        POSITIONING: _SOTA,
        MID_FIELD: _SOTA,
    },
    "ablation_D_103B": {
        NEAR_GOAL: _SOTA,
        BALL_DUEL: _SOTA,
        POSITIONING: _103B_DEFENDER,
        MID_FIELD: _SOTA,
    },
    "ablation_E_103C": {
        NEAR_GOAL: _SOTA,
        BALL_DUEL: _SOTA,
        POSITIONING: _SOTA,
        MID_FIELD: _103C_DRIBBLE,
    },
    "ablation_F_all": {
        NEAR_GOAL: _081_AGGRESSIVE,
        BALL_DUEL: _103A_INTERCEPTOR,
        POSITIONING: _103B_DEFENDER,
        MID_FIELD: _103C_DRIBBLE,
    },
    # 2026-04-23 snapshot-109 retest: replace BALL_DUEL specialist with 103A-wd v2@400 (SOTA-tier, NOT v1 0.548 broken specialist)
    # Tests: did Wave 3 §7C "0/5 specialists improve" verdict reflect framework limit, or just specialist quality?
    "ablation_v2_103Awd_balldul": {
        NEAR_GOAL: _SOTA,
        BALL_DUEL: _103A_WD_V2_400,
        POSITIONING: _SOTA,
        MID_FIELD: _SOTA,
    },
    # 2026-04-23 snapshot-109 full-v2 retest: 103A-wd v2 in BALL_DUEL + 081 in NEAR-GOAL + 101A in POSITIONING (3 strong specialists)
    "ablation_v2_strong3": {
        NEAR_GOAL: _081_AGGRESSIVE,
        BALL_DUEL: _103A_WD_V2_400,
        POSITIONING: _101A_BALLCONTROL,
        MID_FIELD: _SOTA,
    },
    # Wave 3 narrow-trigger: only NEAR-GOAL uses 081 (strictest threshold), rest all 1750 SOTA.
    # Hypothesis: minimize specialist exposure; 081 only fires on highest-confidence shot-ready
    # states (nearest<0.10 AND centroid>0.5), so even if 081 is worse than 1750, damage bounded.
    # If 081 striker is more accurate on these rare states, small net lift. See task-queue P0.
    "wave3_narrow": {
        NEAR_GOAL: _081_AGGRESSIVE,
        BALL_DUEL: _SOTA,
        POSITIONING: _SOTA,
        MID_FIELD: _SOTA,
    },
}

# Presets that use narrow thresholds (otherwise default thresholds apply).
_NARROW_TRIGGER_PRESETS = {"wave3_narrow"}

_DEFAULT_PRESET = os.environ.get("SELECTOR_PHASE_MAP_PRESET", "wave2").strip()
if _DEFAULT_PRESET not in _PRESETS:
    raise ValueError(
        f"SELECTOR_PHASE_MAP_PRESET={_DEFAULT_PRESET!r} not in {list(_PRESETS.keys())}"
    )
_DEFAULT_PHASE_MAP = _PRESETS[_DEFAULT_PRESET]

# Apply preset-specific thresholds (Wave 3 narrow vs default).
if _DEFAULT_PRESET in _NARROW_TRIGGER_PRESETS:
    _ACTIVE_NEAR_GOAL_NEAREST = _NARROW_NEAR_GOAL_NEAREST
    _ACTIVE_NEAR_GOAL_CENTROID = _NARROW_NEAR_GOAL_CENTROID
    _ACTIVE_BALL_DUEL_NEAREST = _NARROW_BALL_DUEL_NEAREST
    _ACTIVE_POSITION_MARGIN = _NARROW_POSITION_MARGIN
print(
    f"[v_selector_phase4] phase_map preset = {_DEFAULT_PRESET} | "
    f"thresholds: NEAR_GOAL_NEAREST={_ACTIVE_NEAR_GOAL_NEAREST} "
    f"NEAR_GOAL_CENTROID={_ACTIVE_NEAR_GOAL_CENTROID}",
    flush=True,
)


def _validate_specialists(phase_map: Dict[str, Tuple[str, str]]) -> None:
    for phase, (kind, ckpt) in phase_map.items():
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Specialist ckpt missing for phase '{phase}': {ckpt}")


_validate_specialists(_DEFAULT_PHASE_MAP)


# --------------------------------------------------------------------------
# Agent class
# --------------------------------------------------------------------------

class Agent(AgentInterface):
    """4-phase task-conditional selector ensemble (Stone DIR-A Wave 1)."""

    def __init__(self, env: gym.Env, phase_map: Optional[Dict[str, Tuple[str, str]]] = None):
        super().__init__()
        os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        os.environ.setdefault("RAY_GRAFANA_HOST", "")
        os.environ.setdefault("RAY_PROMETHEUS_HOST", "")
        import ray
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            local_mode=True,
            num_cpus=1,
            log_to_driver=False,
        )

        if phase_map is None:
            phase_map = _DEFAULT_PHASE_MAP
        self._phase_map = dict(phase_map)

        # De-duplicate ckpts: load each unique (kind, ckpt) once, share handles.
        unique_specs: Dict[Tuple[str, str], object] = {}
        for kind, ckpt in self._phase_map.values():
            key = (kind, ckpt)
            if key in unique_specs:
                continue
            if kind == "team_ray":
                unique_specs[key] = _TeamRayPolicyHandle(env, ckpt)
            elif kind == "shared_cc":
                unique_specs[key] = _SharedCCPolicyHandle(env, ckpt)
            else:
                raise ValueError(f"Unknown specialist kind: {kind}")
        self._handles_by_key = unique_specs
        self._env_action_space = getattr(env, "action_space", None)

        # Phase usage counters (for post-hoc debugging via get_stats())
        self._phase_count: Dict[str, int] = {p: 0 for p in self._phase_map}
        self._step_count = 0

    def _route_per_agent(
        self,
        my_obs: np.ndarray,
        mate_obs: np.ndarray,
    ) -> np.ndarray:
        phase = classify_phase(my_obs, mate_obs)
        self._phase_count[phase] += 1
        kind, ckpt = self._phase_map[phase]
        handle = self._handles_by_key[(kind, ckpt)]

        # Both handle types expose action_probs(obs=, teammate_obs=) +
        # sample_env_action(probs, greedy). For team_ray the call returns
        # (self_probs, mate_probs, norm) — we only need self_probs since
        # we route per-agent.
        if isinstance(handle, _TeamRayPolicyHandle):
            self_probs, _mate_probs, _norm = handle.action_probs(
                obs=my_obs, teammate_obs=mate_obs
            )
        elif isinstance(handle, _SharedCCPolicyHandle):
            self_probs, _norm = handle.action_probs(
                obs=my_obs, teammate_obs=mate_obs
            )
        else:
            raise RuntimeError(f"Unsupported handle type: {type(handle)}")
        return handle.sample_env_action(self_probs, greedy=True)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        self._step_count += 1
        player_ids = sorted(int(pid) for pid in observation.keys())
        if len(player_ids) != 2:
            raise ValueError(
                f"v_selector_phase4 expects exactly 2 local teammates, got ids={player_ids}"
            )
        a, b = player_ids
        return {
            a: self._route_per_agent(observation[a], observation[b]),
            b: self._route_per_agent(observation[b], observation[a]),
        }

    def get_stats(self) -> Dict[str, float]:
        """Optional: report phase distribution for analysis."""
        if self._step_count == 0:
            return {}
        out = {f"phase_pct_{p}": v / max(1, self._step_count * 2) for p, v in self._phase_count.items()}
        out["total_agent_steps"] = self._step_count * 2
        return out
