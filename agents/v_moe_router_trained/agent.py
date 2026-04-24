"""DIR-G Wave 2 — MoE Router with REINFORCE-trained NN selector.

Wave 2 of Stone DIR-G. Identical infrastructure to v_moe_router_uniform but
the per-step expert selection is driven by a NN trained via REINFORCE on
episode returns (`scripts/research/train_moe_router_reinforce.py`). Loads
trained router weights from `agents/v_moe_router_trained/router_weights.pt`.

If `router_weights.pt` is missing (training not yet run), this falls back to
uniform routing identical to v_moe_router_uniform — useful as a soft
graceful degradation.

Differs from v_moe_router_uniform: state-conditional learned routing, not
random uniform. Should beat uniform (Wave 1 = 0.900) iff the trained router
extracts state-conditional signal — that's the key Wave 2 hypothesis.
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
from cs8803drl.branches.moe_router_nn import RouterNN  # noqa: E402

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


# Mirror the trainer's default expert pool. Auto-include 081/103 specialists
# if their packages exist (post-Wave-1 specialist library expansion).
def _default_experts() -> List[Tuple[str, str, str]]:
    base = [
        ("1750_sota", "team_ray",
         str(_AGENTS_ROOT / "v_sota_055v2_extend_1750" / "checkpoint_001750" / "checkpoint-1750")),
        ("055_1150", "team_ray",
         str(_AGENTS_ROOT / "v_055_1150" / "checkpoint_001150" / "checkpoint-1150")),
        ("029B_190", "shared_cc",
         str(_AGENTS_ROOT / "v_029B_190" / "checkpoint_000190" / "checkpoint-190")),
    ]
    optional = [
        ("081_aggressive", "team_ray",
         _AGENTS_ROOT / "v_081_aggressive" / "checkpoint" / "checkpoint"),
        ("103A_interceptor", "team_ray",
         _AGENTS_ROOT / "v_103A_interceptor" / "checkpoint" / "checkpoint"),
        ("103B_defender", "team_ray",
         _AGENTS_ROOT / "v_103B_defender" / "checkpoint" / "checkpoint"),
        ("103C_dribble", "team_ray",
         _AGENTS_ROOT / "v_103C_dribble" / "checkpoint" / "checkpoint"),
    ]
    for name, kind, path in optional:
        if path.exists():
            base.append((name, kind, str(path)))
    return base


_WEIGHTS_PATH = _AGENT_DIR / "router_weights.pt"


class Agent(AgentInterface):
    """Wave 2 MoE Router agent: trained NN selector over K frozen experts."""

    def __init__(
        self,
        env: gym.Env,
        experts: Optional[List[Tuple[str, str, str]]] = None,
        weights_path: Optional[Path] = None,
        greedy: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        import ray
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            local_mode=True,
            num_cpus=1,
            log_to_driver=False,
        )

        if experts is None:
            experts = _default_experts()
        self._expert_names = [name for name, _k, _c in experts]
        self._handles: List[object] = []
        for name, kind, ckpt in experts:
            if not Path(ckpt).exists():
                raise FileNotFoundError(f"Expert '{name}' ckpt missing: {ckpt}")
            if kind == "team_ray":
                self._handles.append(_TeamRayPolicyHandle(env, ckpt))
            elif kind == "shared_cc":
                self._handles.append(_SharedCCPolicyHandle(env, ckpt))
            else:
                raise ValueError(f"Unknown expert kind: {kind!r}")

        # Build router NN. If trained weights exist on disk, load them; else
        # fall back to fresh-init (= roughly uniform).
        wp = Path(weights_path) if weights_path is not None else _WEIGHTS_PATH
        self._router = RouterNN(obs_dim=336, n_experts=len(self._handles), hidden=64)
        self._loaded_weights = False
        self._weights_path = wp
        if wp.exists():
            ckpt = torch.load(str(wp), map_location="cpu")
            n_in_ckpt = int(ckpt.get("n_experts", len(self._handles)))
            if n_in_ckpt != len(self._handles):
                print(
                    f"[v_moe_router_trained] WARNING: weights file expects {n_in_ckpt} experts "
                    f"but {len(self._handles)} loaded — using fresh-init router instead.",
                    flush=True,
                )
            else:
                self._router.load_state_dict(ckpt["router_state_dict"])
                self._loaded_weights = True
                print(
                    f"[v_moe_router_trained] loaded router weights from {wp} "
                    f"(episode {ckpt.get('episode', '?')}, train WR last50 = {ckpt.get('win_rate_last50', '?')})",
                    flush=True,
                )
        self._router.eval()
        self._greedy = bool(greedy)
        self._rng = np.random.default_rng(seed)

        self._expert_count = {name: 0 for name in self._expert_names}
        self._step_count = 0

    def _route(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            logits = self._router(obs_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        if self._greedy:
            return int(np.argmax(probs))
        return int(self._rng.choice(len(probs), p=probs))

    def _route_per_agent(self, my_obs: np.ndarray, mate_obs: np.ndarray) -> np.ndarray:
        idx = self._route(my_obs)
        idx = max(0, min(idx, len(self._handles) - 1))
        self._expert_count[self._expert_names[idx]] += 1
        handle = self._handles[idx]
        if isinstance(handle, _TeamRayPolicyHandle):
            self_probs, _mate, _norm = handle.action_probs(obs=my_obs, teammate_obs=mate_obs)
        else:
            self_probs, _norm = handle.action_probs(obs=my_obs, teammate_obs=mate_obs)
        return handle.sample_env_action(self_probs, greedy=True)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        self._step_count += 1
        ids = sorted(int(p) for p in observation.keys())
        if len(ids) != 2:
            raise ValueError(f"v_moe_router_trained needs 2 teammates, got {ids}")
        a, b = ids
        return {
            a: self._route_per_agent(observation[a], observation[b]),
            b: self._route_per_agent(observation[b], observation[a]),
        }

    def get_stats(self) -> Dict[str, float]:
        if self._step_count == 0:
            return {}
        out = {f"expert_pct_{name}": v / max(1, self._step_count * 2)
               for name, v in self._expert_count.items()}
        out["loaded_weights"] = float(self._loaded_weights)
        out["weights_path"] = str(self._weights_path)
        out["total_agent_steps"] = self._step_count * 2
        return out
