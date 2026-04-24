"""State/action bottleneck wrappers for Stone DIR-A specialists per Wang/Stone/Hanna 2025.

snapshot-110 design: implement Wang Table I per-specialist obs/action bottleneck
without changing model architecture. Two gym wrappers:

- **ObsBottleneckWrapper**: zero out non-relevant ray channels in 336-d obs.
  Each ray block (8 features = 7 type one-hot + 1 distance) is kept only if its
  detected ray-type is in the allowed set AND its ray index (0-13) is in the
  allowed set. Others get type=zeros + distance=1.0 ("not visible / max far").

- **ActionBottleneckWrapper**: replace specific MultiDiscrete action dims with
  fixed values before passing to env. Policy still outputs MultiDiscrete([3,3,3])
  but the constrained dims are deterministic (policy learns the others matter,
  fixed dims contribute zero gradient signal in expectation).

Both wrappers are config-driven via env vars in launcher (see snapshot-110 §2.4
for example). Wrapper composition order in env factory:

    RewardShapingWrapper → ScenarioResetWrapper → ObsBottleneckWrapper → ActionBottleneckWrapper → soccer_twos.make()

(ObsBottleneck applied after scenario init so state is correctly placed before
masking; ActionBottleneck applied at step time between policy output and env.)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import gym
import numpy as np

from cs8803drl.branches.obs_summary import RAY_BLOCK_SIZE, RAY_TYPE_DIM


# Soccer-Twos ray-cast constants (per snapshot-082/083 design + obs_summary)
RAYS_PER_FRAME = 14   # 11 forward + 3 back
N_FRAMES = 3          # 3 stacked frames
TOTAL_RAYS = RAYS_PER_FRAME * N_FRAMES   # 42
EXPECTED_OBS_DIM = TOTAL_RAYS * RAY_BLOCK_SIZE  # 336


def _parse_int_csv(value: str) -> List[int]:
    """Parse 'a,b,c' → [a, b, c]. Empty string → [] (sentinel for 'unrestricted')."""
    value = (value or "").strip()
    if not value:
        return []
    out: List[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_int_keyval_csv(value: str) -> Dict[int, int]:
    """Parse 'k1:v1,k2:v2' → {k1: v1, k2: v2}."""
    value = (value or "").strip()
    if not value:
        return {}
    out: Dict[int, int] = {}
    for tok in value.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        out[int(k.strip())] = int(v.strip())
    return out


class ObsBottleneckWrapper(gym.Wrapper):
    """Mask non-relevant ray channels in 336-d obs (snapshot-110 §2.2.1).

    Args:
        env: underlying env (must produce 336-d ray-cast obs per agent)
        ray_type_mask: List[int] of ray-type indices to KEEP (0=ball, 1=teammate,
                       2=opp, 4=opp_goal, 5=own_goal — empirically per snapshot-103/107).
                       Empty list = no type filtering (keep all types).
        ray_index_mask: Optional List[int] (0..13) of ray INDICES to keep within each frame.
                       Forward 11 = indices 0..10; back 3 = indices 11..13.
                       None = all 14 rays kept.
        masked_distance: Value to set distance to when ray is masked out. Default 1.0
                         ("max far / not visible") which collides with genuine "out of range"
                         obs. Set to -1.0 if you want a unique "masked" sentinel (but encoder
                         was trained on [0,1] distance, so -1.0 may cause gradient surprise).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        ray_type_mask: Optional[Iterable[int]] = None,
        ray_index_mask: Optional[Iterable[int]] = None,
        masked_distance: float = 1.0,
    ):
        super().__init__(env)
        self._type_mask: Optional[Set[int]] = (
            set(int(t) for t in ray_type_mask) if ray_type_mask else None
        )
        self._index_mask: Optional[Set[int]] = (
            set(int(i) for i in ray_index_mask) if ray_index_mask else None
        )
        self._masked_distance = float(masked_distance)

    def _mask_one(self, arr: np.ndarray) -> np.ndarray:
        flat = np.asarray(arr, dtype=np.float32).reshape(-1).copy()
        if flat.size == 0 or flat.size % RAY_BLOCK_SIZE != 0:
            return flat
        n_blocks = flat.size // RAY_BLOCK_SIZE
        blocks = flat.reshape(n_blocks, RAY_BLOCK_SIZE)
        for i in range(n_blocks):
            ray_idx = i % RAYS_PER_FRAME
            type_scores = blocks[i, :RAY_TYPE_DIM]
            # Type-relevant if any kept type detected (one-hot > 0.5)
            if self._type_mask is None:
                type_relevant = True
            else:
                type_relevant = False
                for t in self._type_mask:
                    if 0 <= t < RAY_TYPE_DIM and type_scores[t] > 0.5:
                        type_relevant = True
                        break
            # Index-relevant per ray-index-within-frame
            index_relevant = (self._index_mask is None) or (ray_idx in self._index_mask)
            if not (type_relevant and index_relevant):
                blocks[i, :RAY_TYPE_DIM] = 0.0
                blocks[i, RAY_TYPE_DIM] = self._masked_distance
        return blocks.reshape(-1)

    def _mask_obs(self, obs):
        if isinstance(obs, dict):
            return {aid: self._mask_one(o) for aid, o in obs.items()}
        return self._mask_one(obs)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._mask_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._mask_obs(obs), reward, done, info


class ActionBottleneckWrapper(gym.Wrapper):
    """Constrain MultiDiscrete action dims to fixed values (snapshot-110 §2.2.2).

    Args:
        env: underlying env (MultiDiscrete([3,3,3]) action per agent expected)
        free_dims: List[int] of dim indices the policy controls (0=forward, 1=rotate, 2=kick).
                   Empty = all dims free (no-op wrapper).
        fixed_values: Dict[dim_idx → int_value] for constrained dims.
                      e.g., {0: 2, 1: 1} means forward fixed=2 (always forward), rotate fixed=1 (no turn).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        free_dims: Optional[Iterable[int]] = None,
        fixed_values: Optional[Dict[int, int]] = None,
    ):
        super().__init__(env)
        self._free_dims: Set[int] = set(int(d) for d in (free_dims or []))
        self._fixed_values: Dict[int, int] = {
            int(k): int(v) for k, v in (fixed_values or {}).items()
        }

    def _constrain_one(self, act) -> np.ndarray:
        arr = np.asarray(act, dtype=np.int64).reshape(-1).copy()
        for dim, val in self._fixed_values.items():
            if 0 <= dim < arr.size:
                arr[dim] = val
        return arr

    def _constrain_actions(self, action):
        if isinstance(action, dict):
            return {aid: self._constrain_one(a) for aid, a in action.items()}
        return self._constrain_one(action)

    def step(self, action):
        return self.env.step(self._constrain_actions(action))


def parse_obs_bottleneck_env_vars() -> Optional[Dict]:
    """Read OBS_BOTTLENECK_RAY_TYPES + OBS_BOTTLENECK_RAY_INDICES env vars → kwargs dict.

    Returns None if no bottleneck configured (env var unset or empty).
    """
    import os
    types_csv = os.environ.get("OBS_BOTTLENECK_RAY_TYPES", "").strip()
    indices_csv = os.environ.get("OBS_BOTTLENECK_RAY_INDICES", "").strip()
    if not types_csv and not indices_csv:
        return None
    masked_distance = float(os.environ.get("OBS_BOTTLENECK_MASKED_DISTANCE", "1.0"))
    return {
        "ray_type_mask": _parse_int_csv(types_csv) if types_csv else None,
        "ray_index_mask": _parse_int_csv(indices_csv) if indices_csv else None,
        "masked_distance": masked_distance,
    }


def parse_action_bottleneck_env_vars() -> Optional[Dict]:
    """Read ACTION_BOTTLENECK_FREE_DIMS + ACTION_BOTTLENECK_FIXED env vars → kwargs.

    Returns None if no bottleneck configured.
    """
    import os
    free_csv = os.environ.get("ACTION_BOTTLENECK_FREE_DIMS", "").strip()
    fixed_csv = os.environ.get("ACTION_BOTTLENECK_FIXED", "").strip()
    if not free_csv and not fixed_csv:
        return None
    return {
        "free_dims": _parse_int_csv(free_csv) if free_csv else None,
        "fixed_values": _parse_int_keyval_csv(fixed_csv) if fixed_csv else None,
    }
