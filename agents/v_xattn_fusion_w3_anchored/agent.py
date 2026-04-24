"""DIR-H W3: 1750-anchored cross-attention expert fusion (deploy-time, no training).

Addresses the Wave 2 regression root cause (snapshot-100 §7B): forcing specialists
into NEAR-GOAL/BALL_DUEL slots pulled baseline WR from 0.875 → 0.765 because
specialist standalone WR (0.205-0.826) dominated ensemble when their slots fired.

**Cross-attention "safe routing"**: instead of hard phase switching, produce a soft
weighted average of K experts' action probabilities where 1750 SOTA gets a strong
bias (+3.0 logit) at init. Specialists only meaningfully contribute if their
randomly-initialized keys happen to align with the query — essentially a noisy
"1750 with small specialist perturbations" at Wave 1 deploy time.

Untrained W3 variant (this lane):
    - Q-projection: small random init (0.1 variance)
    - Keys: small random init
    - Bias[1750]=+3.0, bias[others]=0
    - Expected behavior: softmax weights ≈ [0.75, ~0.035 × 7] for 8 experts
        → output ≈ 0.75 * p(1750) + 0.035 * Σ p(specialist_i)
        → very close to 1750 alone, with ~25% soft perturbation from specialists

Contrast:
    - 074F weighted ensemble: fixed uniform/weighted averaging, no state conditioning
    - v_selector_phase4 wave2: hard phase switch, specialists dominate their slots
    - v_selector_phase4 wave3_narrow: hard phase switch, specialists fire rarely
    - v_moe_router_uniform: uniform random routing to ONE expert per step
    - THIS (v_xattn_fusion_w3_anchored): SOFT state-conditional fusion, 1750-biased

W2 variant (trained, separate lane v_xattn_fusion_w2_trained) is the follow-up if
this untrained baseline shows the fusion structure doesn't hurt SOTA.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
import torch

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
from cs8803drl.branches.xattn_fusion_nn import XAttnFusionNN  # noqa: E402

# Register all custom model classes (mirror trained_team_ensemble_next_agent).
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
# Expert registry — 8 experts (same as v_moe_router_uniform for direct A/B)
# Order matters: index 0 = 1750 (anchor).
# --------------------------------------------------------------------------
_EXPERTS: List[Tuple[str, str, str]] = [
    ("1750_sota", "team_ray",
     str(_AGENTS_ROOT / "v_sota_055v2_extend_1750" / "checkpoint_001750" / "checkpoint-1750")),
    ("055_1150", "team_ray",
     str(_AGENTS_ROOT / "v_055_1150" / "checkpoint_001150" / "checkpoint-1150")),
    ("029B_190", "shared_cc",
     str(_AGENTS_ROOT / "v_029B_190" / "checkpoint_000190" / "checkpoint-190")),
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

_ANCHOR_IDX = 0  # 1750 SOTA at index 0
_ANCHOR_BIAS = float(os.environ.get("XATTN_ANCHOR_BIAS", "3.0"))
_D_KEY = int(os.environ.get("XATTN_D_KEY", "64"))
_SEED = int(os.environ.get("XATTN_SEED", "0"))


def _validate(experts: List[Tuple[str, str, str]]) -> None:
    for name, _kind, ckpt in experts:
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Expert '{name}' ckpt missing: {ckpt}")


_validate(_EXPERTS)


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

class Agent(AgentInterface):
    """K-expert cross-attention fusion agent (Stone DIR-H W3, anchored deploy-time)."""

    def __init__(
        self,
        env: gym.Env,
        experts: Optional[List[Tuple[str, str, str]]] = None,
        anchor_bias: float = _ANCHOR_BIAS,
        d_key: int = _D_KEY,
        seed: int = _SEED,
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

        # Build cross-attention fusion module (untrained, anchor biased)
        torch.manual_seed(int(seed))
        self._fusion = XAttnFusionNN(
            obs_dim=336,
            n_experts=len(self._handles),
            d_key=int(d_key),
            anchor_idx=_ANCHOR_IDX,
            anchor_bias=float(anchor_bias),
            trainable=False,
        )
        self._fusion.eval()

        # Per-expert weighted exposure counter (weights sum per step)
        self._expert_weight_sum: Dict[str, float] = {name: 0.0 for name in self._expert_names}
        self._step_count = 0

        # Action-space helper: use first team_ray handle's sample_env_action()
        self._sampler_handle = self._handles[0]

    def _fuse_probs(self, my_obs: np.ndarray, mate_obs: np.ndarray) -> np.ndarray:
        """Compute fused action probs via weighted average over experts."""
        # 1) Get per-expert action probs (Discrete(27))
        per_expert_probs = []
        for handle in self._handles:
            if isinstance(handle, _TeamRayPolicyHandle):
                self_probs, _mate_probs, _norm = handle.action_probs(
                    obs=my_obs, teammate_obs=mate_obs
                )
            elif isinstance(handle, _SharedCCPolicyHandle):
                self_probs, _norm = handle.action_probs(
                    obs=my_obs, teammate_obs=mate_obs
                )
            else:
                raise RuntimeError(f"Unsupported handle: {type(handle)}")
            per_expert_probs.append(np.asarray(self_probs, dtype=np.float64))

        per_expert = np.stack(per_expert_probs, axis=0)  # (K, 27)

        # 2) Compute fusion weights from my_obs
        with torch.no_grad():
            obs_t = torch.from_numpy(np.asarray(my_obs, dtype=np.float32))
            weights = self._fusion(obs_t).cpu().numpy()  # (K,)

        # 3) Track weight accumulation
        for name, w in zip(self._expert_names, weights):
            self._expert_weight_sum[name] += float(w)

        # 4) Weighted sum
        fused = (weights[:, None] * per_expert).sum(axis=0)  # (27,)
        fused_sum = float(fused.sum())
        if fused_sum > 0:
            fused = fused / fused_sum
        return fused

    def _route_per_agent(self, my_obs: np.ndarray, mate_obs: np.ndarray) -> np.ndarray:
        fused = self._fuse_probs(my_obs, mate_obs)
        # 2026-04-22 fix per audit SUSP-2: greedy=True can flip argmax when fusion smears
        # 1750's strong-action prob across specialists' competing actions. Use stochastic
        # sample from fused distribution — preserves the fusion's intended probability mass.
        # Override via XATTN_GREEDY=1 env var if needed for ablation.
        greedy = os.environ.get("XATTN_GREEDY", "0").strip().lower() in {"1", "true", "yes"}
        return self._sampler_handle.sample_env_action(fused, greedy=greedy)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        self._step_count += 1
        player_ids = sorted(int(pid) for pid in observation.keys())
        if len(player_ids) != 2:
            raise ValueError(
                f"v_xattn_fusion_w3_anchored expects exactly 2 local teammates, got ids={player_ids}"
            )
        a, b = player_ids
        return {
            a: self._route_per_agent(observation[a], observation[b]),
            b: self._route_per_agent(observation[b], observation[a]),
        }

    def get_stats(self) -> Dict[str, float]:
        if self._step_count == 0:
            return {}
        # Per-expert average weight across all agent-steps
        denom = max(1, self._step_count * 2)
        out = {
            f"expert_avg_weight_{name}": v / denom
            for name, v in self._expert_weight_sum.items()
        }
        out["total_agent_steps"] = float(self._step_count * 2)
        return out
