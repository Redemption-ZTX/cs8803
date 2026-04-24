"""Mixture-of-Experts Router agent — Stone DIR-G Wave 1 (uniform routing baseline).

This Wave 1 implementation is the **framework** for DIR-G learned routing:
- K frozen expert agents (same set as v_selector_phase4 Wave 1)
- Router function over agent obs → distribution over K experts
- Per-agent, per-step: route to ONE expert via argmax / sample
- Use that expert's action

Wave 1: router = uniform random over K experts (no NN). This serves as
**baseline for Wave 2 (REINFORCE-trained router NN)**. Comparing uniform vs
trained router isolates the value of state-conditional routing — if uniform
already beats single best expert, the gain comes from member diversity (not
routing); if learned router beats uniform, the gain comes from STATE-conditional
specialization.

Differs from v074f (probability averaging at action level): DIR-G picks ONE
expert per step (hard switch like Mixture-of-Experts), not soft averaging.

Differs from v_selector_phase4 (geometric heuristic routing): DIR-G's router
is a learnable function, not hand-coded geometry. Wave 1 is uniform random as
the lower bound; Wave 2 will train.
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

from cs8803drl.deployment.ensemble_agent import (  # noqa: E402
    _TeamRayPolicyHandle,
    _SharedCCPolicyHandle,
)

# Mirror trained_team_ensemble_next_agent registrations so all custom models
# load (cross-attn / distill / ensemble-distill / two-stream / per-ray).
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
# Expert registry (K experts, same as v_selector_phase4 Wave 1 for direct comparison)
# --------------------------------------------------------------------------
# Each tuple: (name, kind, ckpt_path)
# Wave 2 expert pool (2026-04-22): expanded from 3 distill-family generalists to
# 8 experts including 5 Stone sub-task specialists. References ckpt paths
# directly (no packaging needed per user 2026-04-22 directive).
_EXPERTS = [
    # GENERALISTS (Wave 1 — distill-family)
    ("1750_sota", "team_ray",
     str(_AGENTS_ROOT / "v_sota_055v2_extend_1750" / "checkpoint_001750" / "checkpoint-1750")),
    ("055_1150", "team_ray",
     str(_AGENTS_ROOT / "v_055_1150" / "checkpoint_001150" / "checkpoint-1150")),
    ("029B_190", "shared_cc",
     str(_AGENTS_ROOT / "v_029B_190" / "checkpoint_000190" / "checkpoint-190")),
    # SPECIALISTS (Wave 2 — Stone DIR-A Wave 3 + 081 + 101A)
    ("081_aggressive", "team_ray",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/081_aggressive_offense_scratch_20260421_184522/TeamVsBaselineShapingPPOTrainer_Soccer_d3c3b_00000_0_2026-04-21_18-45-42/checkpoint_000970/checkpoint-970"),
    ("101A_ballcontrol", "team_ray",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/101A_layered_p1_ballcontrol_20260422_014241/TeamVsBaselineShapingPPOTrainer_Soccer_21c17_00000_0_2026-04-22_01-43-04/checkpoint_000460/checkpoint-460"),
    # 2026-04-22: upgraded from v1 @500 (0.548) → resume @780 (0.620, +0.072pp)
    ("103A_interceptor", "team_ray",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/103A_interceptor_resume_20260422_102921/TeamVsBaselineShapingPPOTrainer_Soccer_b46b6_00000_0_2026-04-22_10-29-43/checkpoint_000780/checkpoint-780"),
    ("103B_defender", "team_ray",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/103B_defender_20260422_022803/TeamVsBaselineShapingPPOTrainer_Soccer_795ae_00000_0_2026-04-22_02-28-28/checkpoint_000400/checkpoint-400"),
    ("103C_dribble", "team_ray",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/103C_dribble_20260422_023142/TeamVsBaselineShapingPPOTrainer_Soccer_fb211_00000_0_2026-04-22_02-32-06/checkpoint_000300/checkpoint-300"),
]


def _validate(experts: List[Tuple[str, str, str]]) -> None:
    for name, _kind, ckpt in experts:
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Expert '{name}' ckpt missing: {ckpt}")


_validate(_EXPERTS)


# --------------------------------------------------------------------------
# Router (Wave 1 = uniform; Wave 2 will swap in trained NN)
# --------------------------------------------------------------------------

class UniformRouter:
    """Wave 1 baseline: uniformly sample expert each step, no state conditioning."""

    def __init__(self, n_experts: int, seed: Optional[int] = None):
        self._n = int(n_experts)
        self._rng = np.random.default_rng(seed)

    def route(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(0, self._n))


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

class Agent(AgentInterface):
    """K-expert MoE router agent (Stone DIR-G Wave 1, uniform routing)."""

    def __init__(
        self,
        env: gym.Env,
        experts: Optional[List[Tuple[str, str, str]]] = None,
        router: Optional[object] = None,
        router_seed: Optional[int] = 0,
    ):
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

        if experts is None:
            experts = _EXPERTS
        self._expert_names = [name for name, _k, _c in experts]
        self._handles: List[object] = []
        for name, kind, ckpt in experts:
            if kind == "team_ray":
                self._handles.append(_TeamRayPolicyHandle(env, ckpt))
            elif kind == "shared_cc":
                self._handles.append(_SharedCCPolicyHandle(env, ckpt))
            else:
                raise ValueError(f"Unknown expert kind: {kind!r}")

        if router is None:
            router = UniformRouter(len(self._handles), seed=router_seed)
        self._router = router

        # Per-expert usage counter for diagnostics
        self._expert_count = {name: 0 for name in self._expert_names}
        self._step_count = 0

    def _route_per_agent(self, my_obs: np.ndarray, mate_obs: np.ndarray) -> np.ndarray:
        expert_idx = int(self._router.route(my_obs))
        expert_idx = max(0, min(expert_idx, len(self._handles) - 1))
        self._expert_count[self._expert_names[expert_idx]] += 1
        handle = self._handles[expert_idx]
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
                f"v_moe_router_uniform expects exactly 2 local teammates, got ids={player_ids}"
            )
        a, b = player_ids
        return {
            a: self._route_per_agent(observation[a], observation[b]),
            b: self._route_per_agent(observation[b], observation[a]),
        }

    def get_stats(self) -> Dict[str, float]:
        if self._step_count == 0:
            return {}
        out = {f"expert_pct_{name}": v / max(1, self._step_count * 2) for name, v in self._expert_count.items()}
        out["total_agent_steps"] = self._step_count * 2
        return out
