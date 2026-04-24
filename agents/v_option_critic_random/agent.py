"""Option-Critic over frozen experts — Stone DIR-E Wave 1 (random-init NN).

Bacon, Harb, Precup 2017 Option-Critic = end-to-end learning of:
  - intra-option policies π_k(a|s)
  - termination function β_k(s) ∈ [0, 1]
  - selector over options π_Ω(k|s)

This Wave 1 is the **frozen-options variant**: the K intra-option policies are
already trained (= our K packaged agents = experts), so we only need the
termination + selector NN (~30k params total, trainable in <1h).

Wave 1 = random-init NN (no training yet), serves as sanity-check baseline.
The expected behavior is "switch options at random ~50% chance per step
because sigmoid(N(0, ε)) ≈ 0.5". That's similar to DIR-G uniform router but
with TEMPORAL ABSTRACTION (when β=False, stick with current option).

Wave 2 will train the NN via REINFORCE / PPO over episode returns.

DIFFERENCES:
- vs v074f (prob averaging): DIR-E hard-switches between full policies
- vs v_selector_phase4 (geometric heuristic): DIR-E learns the selector
- vs v_moe_router_uniform (uniform per-step): DIR-E has temporal stickiness
  (commit to option until β fires) — explicit option/macro-action structure
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np

import torch
import torch.nn as nn

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
# Frozen experts (same set as DIR-A / DIR-G for direct comparison)
# --------------------------------------------------------------------------

# Wave 2 expert pool (2026-04-22): expanded 3 → 8 with Stone sub-task specialists.
_EXPERTS = [
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
    ("103A_interceptor", "team_ray",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/103A_interceptor_20260422_022803/TeamVsBaselineShapingPPOTrainer_Soccer_78989_00000_0_2026-04-22_02-28-27/checkpoint_000500/checkpoint-500"),
    ("103B_defender", "team_ray",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/103B_defender_20260422_022803/TeamVsBaselineShapingPPOTrainer_Soccer_795ae_00000_0_2026-04-22_02-28-28/checkpoint_000400/checkpoint-400"),
    ("103C_dribble", "team_ray",
     "/storage/ice1/5/1/wsun377/ray_results_scratch/103C_dribble_20260422_023142/TeamVsBaselineShapingPPOTrainer_Soccer_fb211_00000_0_2026-04-22_02-32-06/checkpoint_000300/checkpoint-300"),
]

for name, _kind, ckpt in _EXPERTS:
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Expert '{name}' ckpt missing: {ckpt}")


# --------------------------------------------------------------------------
# Option-Critic head NN: termination + selector
# --------------------------------------------------------------------------

class OptionCriticHead(nn.Module):
    """Small NN: input obs (336) + current_option (K-dim one-hot) →
    output (K logits for next option, 1 logit for termination).

    Wave 1: randomly initialized → behaves close to "random selector + ~50%
    termination per step". Wave 2 will train this head.
    """

    def __init__(self, obs_dim: int = 336, n_options: int = 3, hidden: int = 64):
        super().__init__()
        self._n = int(n_options)
        in_dim = obs_dim + n_options
        self._trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self._selector_head = nn.Linear(hidden, n_options)
        self._termination_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, current_option: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs: (336,) tensor. Returns (selector_logits[K], termination_prob in [0,1])."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        cur_oh = torch.zeros((obs.shape[0], self._n), device=obs.device)
        cur_oh[:, current_option] = 1.0
        x = torch.cat([obs, cur_oh], dim=-1)
        h = self._trunk(x)
        return self._selector_head(h), torch.sigmoid(self._termination_head(h)).squeeze(-1)

    def n_options(self) -> int:
        return self._n


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

class Agent(AgentInterface):
    """Option-Critic over K frozen experts (Stone DIR-E Wave 1, random-init NN)."""

    def __init__(
        self,
        env: gym.Env,
        experts: Optional[List[Tuple[str, str, str]]] = None,
        head_seed: int = 0,
        sample_options: bool = True,
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

        torch.manual_seed(int(head_seed))
        self._head = OptionCriticHead(obs_dim=336, n_options=len(self._handles), hidden=64)
        self._head.eval()
        # Per-agent current option index (defaults to 0 = first expert)
        self._current_option: Dict[int, int] = {}
        self._sample = bool(sample_options)
        self._rng = np.random.default_rng(int(head_seed))

        # Diagnostics
        self._terminate_count = 0
        self._step_count = 0
        self._option_count = {name: 0 for name in self._expert_names}

    def _select_option(
        self,
        my_pid: int,
        my_obs: np.ndarray,
    ) -> int:
        """Sample option via head: terminate? if so, select new; else keep current."""
        cur = self._current_option.get(my_pid, 0)
        with torch.no_grad():
            obs_t = torch.from_numpy(np.asarray(my_obs, dtype=np.float32))
            logits, term_prob = self._head.forward(obs_t, current_option=cur)
            term_prob_v = float(term_prob.item())
            terminate = (self._rng.random() < term_prob_v) if self._sample else (term_prob_v > 0.5)
            if terminate or my_pid not in self._current_option:
                self._terminate_count += 1
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                if self._sample:
                    new_opt = int(self._rng.choice(len(probs), p=probs))
                else:
                    new_opt = int(np.argmax(probs))
                self._current_option[my_pid] = new_opt
                return new_opt
            return cur

    def _route_per_agent(
        self,
        my_pid: int,
        my_obs: np.ndarray,
        mate_obs: np.ndarray,
    ) -> np.ndarray:
        opt = self._select_option(my_pid, my_obs)
        opt = max(0, min(opt, len(self._handles) - 1))
        self._option_count[self._expert_names[opt]] += 1
        handle = self._handles[opt]
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
                f"v_option_critic_random expects exactly 2 local teammates, got ids={player_ids}"
            )
        a, b = player_ids
        return {
            a: self._route_per_agent(a, observation[a], observation[b]),
            b: self._route_per_agent(b, observation[b], observation[a]),
        }

    def get_stats(self) -> Dict[str, float]:
        if self._step_count == 0:
            return {}
        out = {f"option_pct_{name}": v / max(1, self._step_count * 2)
               for name, v in self._option_count.items()}
        out["termination_rate"] = self._terminate_count / max(1, self._step_count * 2)
        out["total_agent_steps"] = self._step_count * 2
        return out
