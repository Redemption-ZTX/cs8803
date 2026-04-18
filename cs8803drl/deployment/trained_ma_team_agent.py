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
from ray.rllib import MultiAgentEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface
from cs8803drl.core.checkpoint_utils import load_policy_weights


ALGORITHM = "PPO"
POLICY_NAME = "default"
_DUMMY_ENV_NAME = "DummyEnvTrainedMATeamAgent"


class _DummyMultiAgentEnv(MultiAgentEnv):
    def __init__(self, team_obs_space, team_action_space):
        self.observation_space = team_obs_space
        self.action_space = team_action_space

    def reset(self):
        zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return {0: zero_obs.copy(), 1: zero_obs.copy()}

    def step(self, action_dict):
        zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs = {0: zero_obs.copy(), 1: zero_obs.copy()}
        reward = {0: 0.0, 1: 0.0}
        done = {"__all__": True, 0: True, 1: True}
        info = {0: {}, 1: {}}
        return obs, reward, done, info


def _default_checkpoint_path():
    checkpoint = os.environ.get("TRAINED_RAY_CHECKPOINT", "").strip()
    if checkpoint:
        return checkpoint
    raise ValueError("Missing TRAINED_RAY_CHECKPOINT env var for shared multiagent team policy.")


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


def _build_team_obs_space(raw_obs_space):
    low = np.asarray(raw_obs_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(raw_obs_space.high, dtype=np.float32).reshape(-1)
    return gym.spaces.Box(
        low=np.concatenate([low, low], axis=0),
        high=np.concatenate([high, high], axis=0),
        dtype=np.float32,
    )


def _build_team_action_space(raw_action_space):
    if not isinstance(raw_action_space, gym.spaces.MultiDiscrete):
        raise ValueError(
            "trained_ma_team_agent expects env.action_space to be MultiDiscrete, "
            f"got {type(raw_action_space)!r}"
        )
    nvec = np.asarray(raw_action_space.nvec, dtype=np.int64).reshape(-1)
    return gym.spaces.MultiDiscrete(np.concatenate([nvec, nvec], axis=0))


def _normalize_team_action(action, expected_dim):
    if isinstance(action, tuple) and len(action) >= 1:
        action = action[0]
    arr = np.asarray(action)
    if arr.shape == (expected_dim,):
        return arr.astype(np.int64)
    if arr.size == expected_dim:
        return arr.reshape(expected_dim).astype(np.int64)
    raise ValueError(
        f"Shared multiagent team policy returned action with shape {arr.shape}, expected {expected_dim} values."
    )


class MATeamAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        checkpoint_path = _default_checkpoint_path()
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

        raw_obs_space = getattr(env, "observation_space", None)
        raw_action_space = getattr(env, "action_space", None)
        if raw_obs_space is None or raw_action_space is None:
            raise ValueError("Env must expose observation_space and action_space.")

        self._team_obs_space = _build_team_obs_space(raw_obs_space)
        self._team_action_space = _build_team_action_space(raw_action_space)
        self._player_action_dim = int(len(np.asarray(raw_action_space.nvec).reshape(-1)))

        config = _load_params_config(checkpoint_path)
        tune.registry.register_env(
            _DUMMY_ENV_NAME,
            lambda *_: _DummyMultiAgentEnv(self._team_obs_space, self._team_action_space),
        )
        config["env"] = _DUMMY_ENV_NAME
        config["env_config"] = {}

        cls = get_trainable_cls(ALGORITHM)
        trainer = cls(env=config["env"], config=config)
        self._trainer = trainer
        self.policy = load_policy_weights(checkpoint_path, trainer, POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        player_ids = sorted(int(pid) for pid in observation.keys())
        if len(player_ids) != 2:
            raise ValueError(
                "trained_ma_team_agent expects exactly 2 local teammates in observation, "
                f"got ids={player_ids}"
            )

        team_obs = np.concatenate(
            [
                np.asarray(observation[player_ids[0]], dtype=np.float32).reshape(-1),
                np.asarray(observation[player_ids[1]], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        team_action = _normalize_team_action(
            self.policy.compute_single_action(team_obs, explore=False),
            expected_dim=self._player_action_dim * 2,
        )

        return {
            player_ids[0]: team_action[: self._player_action_dim].astype(np.int64),
            player_ids[1]: team_action[self._player_action_dim :].astype(np.int64),
        }

