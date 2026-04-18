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
_DEFAULT_BASE_CHECKPOINT = (
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
    "PPO_continue_ckpt160_cpu32_20260408_183648/"
    "PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/"
    "checkpoint_000225/checkpoint-225"
)


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


def _specialist_checkpoint():
    value = os.environ.get("TRAINED_RAY_CHECKPOINT", "").strip()
    if value:
        return value
    raise ValueError("Missing TRAINED_RAY_CHECKPOINT for specialist policy.")


def _teammate_checkpoint():
    for name in ("TEAMMATE_BASE_CHECKPOINT", "TEAMMATE_CHECKPOINT"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return _DEFAULT_BASE_CHECKPOINT


def _load_params_config(checkpoint_path):
    config_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(f"Could not find params.pkl near checkpoint: {checkpoint_path}")
    with open(config_path, "rb") as handle:
        config = pickle.load(handle)
    config["num_workers"] = 0
    config["num_gpus"] = 0
    return config


def _build_discrete_action_space(checkpoint_path, env_action_space):
    inferred_action_dim = infer_action_dim_from_checkpoint(checkpoint_path)
    if inferred_action_dim is not None:
        return gym.spaces.Discrete(int(inferred_action_dim))
    if isinstance(env_action_space, gym.spaces.MultiDiscrete):
        return gym.spaces.Discrete(int(np.prod(env_action_space.nvec)))
    return env_action_space


def _load_policy(checkpoint_path, env, *, env_name):
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    if obs_space is None or act_space is None:
        raise ValueError("Env must expose observation_space and action_space.")

    config = _load_params_config(checkpoint_path)
    act_space = _build_discrete_action_space(checkpoint_path, act_space)
    tune.registry.register_env(env_name, lambda *_: _DummyGymEnv(obs_space, act_space))
    config["env"] = env_name
    config["env_config"] = {}

    cls = get_trainable_cls(ALGORITHM)
    trainer = cls(env=config["env"], config=config)
    policy = load_policy_weights(checkpoint_path, trainer, POLICY_NAME)
    return trainer, policy


class FixedTeammateAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        specialist_checkpoint = _specialist_checkpoint()
        teammate_checkpoint = _teammate_checkpoint()

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

        self._env_action_space = getattr(env, "action_space", None)
        self._action_flattener = None
        if ActionFlattener is not None and isinstance(self._env_action_space, gym.spaces.MultiDiscrete):
            try:
                self._action_flattener = ActionFlattener(self._env_action_space.nvec)
            except Exception:
                self._action_flattener = None

        self._specialist_local_agent_id = int(os.environ.get("SPECIALIST_LOCAL_AGENT_ID", "0"))

        specialist_env_name = f"DummyEnvFixedTeammateSpecialist_{os.getpid()}"
        teammate_env_name = f"DummyEnvFixedTeammateTeammate_{os.getpid()}"
        self._specialist_trainer, self._specialist_policy = _load_policy(
            specialist_checkpoint,
            env,
            env_name=specialist_env_name,
        )
        if os.path.abspath(teammate_checkpoint) == os.path.abspath(specialist_checkpoint):
            self._teammate_trainer = self._specialist_trainer
            self._teammate_policy = self._specialist_policy
        else:
            self._teammate_trainer, self._teammate_policy = _load_policy(
                teammate_checkpoint,
                env,
                env_name=teammate_env_name,
            )

    def _policy_for_player(self, player_id):
        if int(player_id) == self._specialist_local_agent_id:
            return self._specialist_policy
        return self._teammate_policy

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            policy = self._policy_for_player(player_id)
            action = _unwrap_action(policy.compute_single_action(obs, explore=False))

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


Agent = FixedTeammateAgent
