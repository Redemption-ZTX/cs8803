import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

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
from cs8803drl.branches.expert_coordination import (
    DualExpertCoordinator,
    default_attack_checkpoint,
    default_defense_checkpoint,
)

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:
        ActionFlattener = None


ALGORITHM = "PPO"
POLICY_NAME = "default_policy"


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


def _load_single_player_policy(checkpoint_path, env, *, env_name: str):
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    if obs_space is None or act_space is None:
        raise ValueError("Env must expose observation_space and action_space.")

    config = _load_params_config(checkpoint_path)
    discrete_action_space = _build_discrete_action_space(checkpoint_path, act_space)
    tune.registry.register_env(env_name, lambda *_: _DummyGymEnv(obs_space, discrete_action_space))
    config["env"] = env_name
    config["env_config"] = {}

    cls = get_trainable_cls(ALGORITHM)
    trainer = cls(env=config["env"], config=config)
    policy = load_policy_weights(checkpoint_path, trainer, POLICY_NAME)
    return trainer, policy


class DualExpertAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        attack_checkpoint = default_attack_checkpoint()
        defense_checkpoint = default_defense_checkpoint()
        if not attack_checkpoint and not defense_checkpoint:
            raise ValueError(
                "Missing attack/defense checkpoints. Set ATTACK_EXPERT_CHECKPOINT and/or "
                "DEFENSE_EXPERT_CHECKPOINT, or provide TRAINED_RAY_CHECKPOINT for the active expert."
            )
        if not attack_checkpoint:
            attack_checkpoint = defense_checkpoint
        if not defense_checkpoint:
            defense_checkpoint = attack_checkpoint

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

        self._attack_checkpoint = attack_checkpoint
        self._defense_checkpoint = defense_checkpoint

        attack_env_name = f"DummyEnvDualExpertAttack_{os.getpid()}"
        defense_env_name = f"DummyEnvDualExpertDefense_{os.getpid()}"
        self._attack_trainer, self._attack_policy = _load_single_player_policy(
            attack_checkpoint,
            env,
            env_name=attack_env_name,
        )
        if os.path.abspath(defense_checkpoint) == os.path.abspath(attack_checkpoint):
            self._defense_trainer = self._attack_trainer
            self._defense_policy = self._attack_policy
        else:
            self._defense_trainer, self._defense_policy = _load_single_player_policy(
                defense_checkpoint,
                env,
                env_name=defense_env_name,
            )

        self._coordinator = DualExpertCoordinator(
            default_attacker_id=int(os.environ.get("COORD_DEFAULT_ATTACKER_ID", "0")),
            switch_margin=float(os.environ.get("COORD_SWITCH_MARGIN", "0.08")),
            switch_cooldown=int(os.environ.get("COORD_SWITCH_COOLDOWN", "4")),
        )

    def _policy_for_player(self, player_id, attacker_id):
        if int(player_id) == int(attacker_id):
            return self._attack_policy
        return self._defense_policy

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        obs_map = {
            int(player_id): np.asarray(obs, dtype=np.float32).reshape(-1)
            for player_id, obs in observation.items()
        }
        if not obs_map:
            return {}

        attacker_id = self._coordinator.choose_attacker(obs_map)
        actions = {}
        for player_id, obs in obs_map.items():
            policy = self._policy_for_player(player_id, attacker_id)
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


Agent = DualExpertAgent
