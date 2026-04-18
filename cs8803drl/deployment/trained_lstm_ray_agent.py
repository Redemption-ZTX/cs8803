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
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface
from cs8803drl.core.checkpoint_utils import infer_action_dim_from_checkpoint, load_policy_weights

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:
        ActionFlattener = None


ALGORITHM = "PPO"
POLICY_NAME = "default_policy"
_DUMMY_ENV_NAME = f"DummyEnvTrainedLSTMRayAgent_{os.getpid()}"
_KICKOFF_RESETS = 4
_KICKOFF_ABS_TOL = 1e-4
_KICKOFF_MEAN_ABS_THRESHOLD = 1e-3


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


class _DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        raise RuntimeError("Dummy env should never be stepped")

    def step(self, action):
        raise RuntimeError("Dummy env should never be stepped")


def _default_checkpoint_path():
    for env_name in ("TRAINED_RAY_CHECKPOINT", "TRAINED_LSTM_RAY_CHECKPOINT"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise ValueError(
        "Missing TRAINED_RAY_CHECKPOINT env var. Example: "
        "TRAINED_RAY_CHECKPOINT=/path/to/checkpoint-20"
    )


class LSTMRayAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        checkpoint_path = _default_checkpoint_path()
        inferred_action_dim = infer_action_dim_from_checkpoint(checkpoint_path)

        self._kickoff_templates = self._sample_kickoff_templates(env)

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

        if inferred_action_dim is not None:
            act_space = gym.spaces.Discrete(int(inferred_action_dim))
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            act_space = gym.spaces.Discrete(int(np.prod(act_space.nvec)))

        tune.registry.register_env(
            _DUMMY_ENV_NAME, lambda *_: _DummyGymEnv(obs_space, act_space)
        )
        config["env"] = _DUMMY_ENV_NAME
        config["env_config"] = {}

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
        self.policy = load_policy_weights(checkpoint_path, trainer, POLICY_NAME)
        self._state = {
            0: self.policy.get_initial_state(),
            1: self.policy.get_initial_state(),
        }

    def _sample_kickoff_templates(self, env):
        templates = []
        if not hasattr(env, "reset"):
            return templates
        try:
            for _ in range(_KICKOFF_RESETS):
                obs = env.reset()
                templates.append(
                    (
                        np.asarray(obs[0], dtype=np.float32).reshape(-1),
                        np.asarray(obs[1], dtype=np.float32).reshape(-1),
                    )
                )
                templates.append(
                    (
                        np.asarray(obs[2], dtype=np.float32).reshape(-1),
                        np.asarray(obs[3], dtype=np.float32).reshape(-1),
                    )
                )
        except Exception:
            return templates
        return templates

    def _is_kickoff_obs(self, observation):
        if not self._kickoff_templates:
            return False
        if 0 not in observation or 1 not in observation:
            return False
        obs0 = np.asarray(observation[0], dtype=np.float32).reshape(-1)
        obs1 = np.asarray(observation[1], dtype=np.float32).reshape(-1)
        for tmpl0, tmpl1 in self._kickoff_templates:
            if obs0.shape != tmpl0.shape or obs1.shape != tmpl1.shape:
                continue
            if np.allclose(obs0, tmpl0, atol=_KICKOFF_ABS_TOL) and np.allclose(obs1, tmpl1, atol=_KICKOFF_ABS_TOL):
                return True
            mean_abs = 0.5 * (float(np.mean(np.abs(obs0 - tmpl0))) + float(np.mean(np.abs(obs1 - tmpl1))))
            if mean_abs <= _KICKOFF_MEAN_ABS_THRESHOLD:
                return True
        return False

    def _reset_state(self):
        self._state = {
            0: self.policy.get_initial_state(),
            1: self.policy.get_initial_state(),
        }

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        if self._is_kickoff_obs(observation):
            self._reset_state()

        actions = {}
        for player_id, obs in observation.items():
            state_in = self._state.get(int(player_id), self.policy.get_initial_state())
            action, state_out, _ = self.policy.compute_single_action(
                np.asarray(obs, dtype=np.float32),
                state=state_in,
                explore=False,
            )
            self._state[int(player_id)] = state_out
            action = _unwrap_action(action)

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


Agent = LSTMRayAgent
