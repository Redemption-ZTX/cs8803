"""FrozenSharedCCPolicy: per-env adapter that lets a frozen shared-CC/MAPPO
checkpoint serve as the opponent for `team_vs_policy` env training.

Unlike `FrozenTeamPolicy`, this adapter does not emit a joint action. Instead it
loads a per-agent centralized-critic policy and services the two sequential
team1 calls made by `TeamVsPolicyWrapper.step()`:

* FIRST call (agent 2): build CC obs from `obs_2 + last_obs_3_from_prev_step`
* SECOND call (agent 3): build CC obs from `obs_3 + current_obs_2`

This mirrors the 1-frame-stale teammate context approximation already accepted
for `FrozenTeamPolicy`, but keeps the per-agent action semantics intact.
"""

from __future__ import annotations

import os
import pickle
import threading
from typing import Dict, Optional

import gym
import numpy as np

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls

from cs8803drl.branches.shared_central_critic import (
    SHARED_CC_POLICY_ID,
    build_cc_obs_space,
    register_shared_cc_model,
)
from cs8803drl.branches.teammate_aux_head import register_shared_cc_teammate_aux_model
from cs8803drl.core.checkpoint_utils import infer_action_dim_from_checkpoint, load_policy_weights

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:
        ActionFlattener = None


class _DummyMultiAgentEnv(MultiAgentEnv):
    def __init__(self, cc_obs_space, raw_obs_space, action_space):
        super().__init__()
        zero_raw = np.zeros(raw_obs_space.shape, dtype=np.float32)
        self.action_space = action_space
        self._reset_obs = {
            0: {
                "own_obs": zero_raw.copy(),
                "teammate_obs": zero_raw.copy(),
                "teammate_action": 0,
            },
            1: {
                "own_obs": zero_raw.copy(),
                "teammate_obs": zero_raw.copy(),
                "teammate_action": 0,
            },
            2: zero_raw.copy(),
            3: zero_raw.copy(),
        }

    def reset(self):
        return self._reset_obs

    def step(self, action_dict):
        obs = self._reset_obs
        rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        dones = {0: True, 1: True, 2: True, 3: True, "__all__": True}
        infos = {0: {}, 1: {}, 2: {}, 3: {}}
        return obs, rewards, dones, infos


def _unflatten_discrete_to_multidiscrete(flat, nvec):
    flat = int(flat)
    out = np.zeros((len(nvec),), dtype=np.int64)
    for i in range(len(nvec) - 1, -1, -1):
        base = int(nvec[i])
        if base <= 0:
            out[i] = 0
            continue
        out[i] = flat % base
        flat //= base
    return out


def _unwrap_action(action):
    current = action
    while isinstance(current, (list, tuple)) and len(current) == 1:
        current = current[0]
    if isinstance(current, tuple) and len(current) >= 1:
        return current[0]
    return current


class FrozenSharedCCPolicy:
    def __init__(self, checkpoint_path: str, *, obs_space: gym.Space, action_space: gym.Space):
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(
                "FrozenSharedCCPolicy expects per-agent obs_space to be Box, "
                f"got {type(obs_space)!r}"
            )
        if not isinstance(action_space, gym.spaces.MultiDiscrete):
            raise TypeError(
                "FrozenSharedCCPolicy expects per-agent action_space to be MultiDiscrete, "
                f"got {type(action_space)!r}"
            )

        self._checkpoint_path = os.path.abspath(checkpoint_path)
        self._obs_space = obs_space
        self._env_action_space = action_space
        self._per_agent_obs_dim = int(np.prod(obs_space.shape))
        self._per_agent_action_dim = int(np.asarray(action_space.nvec, dtype=np.int64).reshape(-1).shape[0])
        self._cached_obs_3 = np.zeros(self._per_agent_obs_dim, dtype=np.float32)
        self._current_obs_2 = np.zeros(self._per_agent_obs_dim, dtype=np.float32)
        self._call_parity = 0

        self._shared_policy = None
        self._cc_preprocessor = None
        self._action_flattener = None
        self._policy_load_lock = threading.Lock()

    def _load_policy(self):
        if self._shared_policy is not None:
            return self._shared_policy
        with self._policy_load_lock:
            if self._shared_policy is not None:
                return self._shared_policy

            ray.init(ignore_reinit_error=True, include_dashboard=False)

            config_dir = os.path.dirname(self._checkpoint_path)
            config_path = os.path.join(config_dir, "params.pkl")
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")
            if not os.path.exists(config_path):
                raise ValueError(
                    f"FrozenSharedCCPolicy: could not find params.pkl near checkpoint: "
                    f"{self._checkpoint_path}"
                )

            with open(config_path, "rb") as handle:
                config = pickle.load(handle)
            config["num_workers"] = 0
            config["num_gpus"] = 0

            obs_space = self._obs_space
            env_action_space = self._env_action_space
            cc_obs_space = build_cc_obs_space(obs_space, env_action_space)
            self._cc_preprocessor = ModelCatalog.get_preprocessor_for_space(cc_obs_space)
            trainer_action_space = env_action_space
            inferred_action_dim = infer_action_dim_from_checkpoint(self._checkpoint_path)
            if inferred_action_dim is not None:
                trainer_action_space = gym.spaces.Discrete(int(inferred_action_dim))
            elif isinstance(env_action_space, gym.spaces.MultiDiscrete):
                trainer_action_space = gym.spaces.Discrete(
                    int(np.prod(env_action_space.nvec))
                )

            if ActionFlattener is not None and isinstance(env_action_space, gym.spaces.MultiDiscrete):
                try:
                    self._action_flattener = ActionFlattener(env_action_space.nvec)
                except Exception:
                    self._action_flattener = None

            register_shared_cc_model()
            register_shared_cc_teammate_aux_model()

            env_name = f"DummyEnvFrozenSharedCC_{abs(hash(self._checkpoint_path))}"
            tune.registry.register_env(
                env_name,
                lambda *_: _DummyMultiAgentEnv(cc_obs_space, obs_space, trainer_action_space),
            )
            config["env"] = env_name
            config["env_config"] = {}
            if "multiagent" in config and "policies" in config["multiagent"]:
                config["multiagent"]["policies"][SHARED_CC_POLICY_ID] = (
                    None,
                    cc_obs_space,
                    trainer_action_space,
                    {},
                )

            cls = get_trainable_cls("PPO")
            trainer = cls(env=config["env"], config=config)
            self._shared_policy = load_policy_weights(
                self._checkpoint_path,
                trainer,
                SHARED_CC_POLICY_ID,
            )
        return self._shared_policy

    def reset_episode(self) -> None:
        self._cached_obs_3 = np.zeros(self._per_agent_obs_dim, dtype=np.float32)
        self._current_obs_2 = np.zeros(self._per_agent_obs_dim, dtype=np.float32)
        self._call_parity = 0

    def _compute_env_action(self, *, own_obs: np.ndarray, teammate_obs: np.ndarray) -> np.ndarray:
        policy = self._load_policy()
        cc_obs: Dict[str, np.ndarray] = {
            "own_obs": own_obs,
            "teammate_obs": teammate_obs,
            "teammate_action": 0,
        }
        flat_cc_obs = self._cc_preprocessor.transform(cc_obs)
        action = _unwrap_action(policy.compute_single_action(flat_cc_obs, explore=False))
        if isinstance(action, np.ndarray):
            if action.ndim == 1 and action.size == self._per_agent_action_dim:
                return action.astype(np.int64, copy=False)
            if action.shape == ():
                action = int(action.item())
        if isinstance(action, (list, tuple)):
            arr = np.asarray(action)
            if arr.ndim == 1 and arr.size == self._per_agent_action_dim:
                return arr.astype(np.int64, copy=False)
            if arr.shape == ():
                action = int(arr.item())
        if isinstance(action, (np.integer, int)):
            flat = int(action)
        else:
            flat = int(action)
        if self._action_flattener is not None:
            return np.asarray(self._action_flattener.lookup_action(flat), dtype=np.int64)
        return _unflatten_discrete_to_multidiscrete(flat, np.asarray(self._env_action_space.nvec))

    def __call__(self, obs, *args, **kwargs) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_arr.size != self._per_agent_obs_dim:
            raise ValueError(
                "FrozenSharedCCPolicy: per-agent obs has unexpected size "
                f"{obs_arr.size}, expected {self._per_agent_obs_dim}."
            )

        if self._call_parity == 0:
            self._current_obs_2 = obs_arr.copy()
            action_2 = self._compute_env_action(
                own_obs=obs_arr,
                teammate_obs=self._cached_obs_3,
            )
            self._call_parity = 1
            return action_2

        action_3 = self._compute_env_action(
            own_obs=obs_arr,
            teammate_obs=self._current_obs_2,
        )
        self._cached_obs_3 = obs_arr.copy()
        self._call_parity = 0
        return action_3
