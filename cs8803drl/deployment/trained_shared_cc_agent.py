import os
import pickle
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_PYTHONPATH = os.environ.get("PYTHONPATH", "")
_PYTHONPATH_ENTRIES = [entry for entry in _PYTHONPATH.split(os.pathsep) if entry]
if REPO_ROOT not in _PYTHONPATH_ENTRIES:
    os.environ["PYTHONPATH"] = REPO_ROOT if not _PYTHONPATH else REPO_ROOT + os.pathsep + _PYTHONPATH

try:
    import sitecustomize as _project_sitecustomize  # noqa: F401
except Exception:
    _project_sitecustomize = None

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface
from cs8803drl.core.checkpoint_utils import infer_action_dim_from_checkpoint, load_policy_weights
from cs8803drl.branches.shared_central_critic import SHARED_CC_POLICY_ID, build_cc_obs_space, register_shared_cc_model
from cs8803drl.branches.teammate_aux_head import register_shared_cc_teammate_aux_model
from cs8803drl.branches.per_agent_distill import register_per_agent_distill_model

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:
        ActionFlattener = None


ALGORITHM = "PPO"
_DUMMY_ENV_NAME = f"DummyEnvTrainedSharedCCAgent_{os.getpid()}"


def _unwrap_action(action):
    a = action
    while isinstance(a, (list, tuple)) and len(a) == 1:
        a = a[0]
    if isinstance(a, tuple) and len(a) >= 1:
        return a[0]
    return a


def _coerce_int_action(action):
    a = _unwrap_action(action)
    if isinstance(a, np.ndarray) and a.shape == ():
        a = a.item()
    if isinstance(a, (np.integer, int)):
        return int(a)
    if isinstance(a, (list, tuple, np.ndarray)):
        arr = np.asarray(a)
        if arr.shape == ():
            return int(arr.item())
        if arr.size >= 1 and arr.ndim == 1:
            return int(arr.reshape(-1)[0])
    return int(a)


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


class _DummyMultiAgentEnv(MultiAgentEnv):
    def __init__(self, cc_obs_space, raw_obs_space, action_space):
        super().__init__()
        self.cc_obs_space = cc_obs_space
        self.raw_obs_space = raw_obs_space
        self.action_space = action_space
        zero_raw = np.zeros(raw_obs_space.shape, dtype=np.float32)
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


def _default_checkpoint_path():
    for env_name in ("TRAINED_RAY_CHECKPOINT", "TRAINED_SHARED_CC_CHECKPOINT"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise ValueError(
        "Missing TRAINED_RAY_CHECKPOINT env var. Example: "
        "TRAINED_RAY_CHECKPOINT=/path/to/checkpoint-20"
    )


class SharedCCAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        checkpoint_path = _default_checkpoint_path()
        inferred_action_dim = infer_action_dim_from_checkpoint(checkpoint_path)

        os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        os.environ.setdefault("RAY_GRAFANA_HOST", "")
        os.environ.setdefault("RAY_PROMETHEUS_HOST", "")
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            local_mode=True,
            num_cpus=1,
            log_to_driver=False,
        )

        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            raise ValueError("Could not find params.pkl near checkpoint.")

        with open(config_path, "rb") as f:
            config = pickle.load(f)
        config["num_workers"] = 0
        config["num_gpus"] = 0

        obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        if obs_space is None or act_space is None:
            raise ValueError("Env must expose observation_space and action_space.")

        cc_obs_space = build_cc_obs_space(obs_space, act_space)
        self._cc_preprocessor = ModelCatalog.get_preprocessor_for_space(cc_obs_space)
        if inferred_action_dim is not None:
            act_space = gym.spaces.Discrete(int(inferred_action_dim))
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            act_space = gym.spaces.Discrete(int(np.prod(act_space.nvec)))

        register_shared_cc_model()
        register_shared_cc_teammate_aux_model()
        register_per_agent_distill_model()
        tune.registry.register_env(
            _DUMMY_ENV_NAME, lambda *_: _DummyMultiAgentEnv(cc_obs_space, obs_space, act_space)
        )
        config["env"] = _DUMMY_ENV_NAME
        config["env_config"] = {}
        if "multiagent" in config and "policies" in config["multiagent"]:
            config["multiagent"]["policies"][SHARED_CC_POLICY_ID] = (
                None,
                cc_obs_space,
                act_space,
                {},
            )

        self._env_action_space = getattr(env, "action_space", None)
        self._action_flattener = None
        if ActionFlattener is not None and isinstance(self._env_action_space, gym.spaces.MultiDiscrete):
            try:
                self._action_flattener = ActionFlattener(self._env_action_space.nvec)
            except Exception:
                self._action_flattener = None

        cls = get_trainable_cls(ALGORITHM)
        trainer = cls(env=config["env"], config=config)
        self._trainer = trainer
        self._shared_policy = load_policy_weights(checkpoint_path, trainer, SHARED_CC_POLICY_ID)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        zero_obs = None
        for player_id, obs in observation.items():
            if zero_obs is None:
                zero_obs = np.zeros_like(np.asarray(obs, dtype=np.float32).reshape(-1))
            mate_id = 1 if int(player_id) == 0 else 0
            mate_obs = np.asarray(observation.get(mate_id, zero_obs), dtype=np.float32).reshape(-1)
            cc_obs = {
                "own_obs": np.asarray(obs, dtype=np.float32).reshape(-1),
                "teammate_obs": mate_obs,
                "teammate_action": 0,
            }
            flat_cc_obs = self._cc_preprocessor.transform(cc_obs)
            action = _unwrap_action(self._shared_policy.compute_single_action(flat_cc_obs))

            if isinstance(self._env_action_space, gym.spaces.MultiDiscrete):
                if isinstance(action, (list, tuple, np.ndarray)):
                    arr = np.asarray(action)
                    if arr.ndim == 1 and arr.size == len(self._env_action_space.nvec):
                        actions[player_id] = arr.astype(np.int64)
                        continue

                flat = _coerce_int_action(action)
                if self._action_flattener is not None:
                    actions[player_id] = np.asarray(
                        self._action_flattener.lookup_action(int(flat)), dtype=np.int64
                    )
                else:
                    actions[player_id] = _unflatten_discrete_to_multidiscrete(
                        flat, np.asarray(self._env_action_space.nvec)
                    )
            else:
                actions[player_id] = _coerce_int_action(action)
        return actions


Agent = SharedCCAgent
