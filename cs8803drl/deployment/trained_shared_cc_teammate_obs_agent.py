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
from cs8803drl.core.obs_teammate import (
    augment_observation_dict_with_decoded_teammate_state,
    build_teammate_obs_space,
    fit_observation_state_decoder,
    parse_teammate_state_scale,
    normalized_episode_time,
)
from cs8803drl.branches.shared_central_critic import (
    SHARED_CC_POLICY_ID,
    build_cc_obs_space,
    register_shared_cc_model,
)
from cs8803drl.branches.teammate_aux_head import register_shared_cc_teammate_aux_model
from cs8803drl.deployment.trained_shared_cc_agent import (
    ALGORITHM,
    _DummyMultiAgentEnv,
    _coerce_int_action,
    _default_checkpoint_path,
    _unflatten_discrete_to_multidiscrete,
    _unwrap_action,
)

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:
        ActionFlattener = None


class SharedCCTeammateObsAgent(AgentInterface):
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

        raw_obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        if raw_obs_space is None or act_space is None:
            raise ValueError("Env must expose observation_space and action_space.")

        decoder_samples = int(os.environ.get("TRAINED_SHARED_CC_DECODER_SAMPLES", "256"))
        self._include_time = str(os.environ.get("TRAINED_SHARED_CC_INCLUDE_TIME", "")).strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
        self._state_scale = parse_teammate_state_scale(
            os.environ.get("TRAINED_SHARED_CC_TEAMMATE_STATE_SCALE", "").strip()
        )
        self._time_max_steps = max(
            1,
            int(os.environ.get("TRAINED_SHARED_CC_TIME_MAX_STEPS", "1500")),
        )
        self._episode_step = 0
        kickoff_obs = env.reset()
        self._kickoff_templates = (
            {
                0: np.asarray(kickoff_obs[0], dtype=np.float32).reshape(-1),
                1: np.asarray(kickoff_obs[1], dtype=np.float32).reshape(-1),
            },
            {
                0: np.asarray(kickoff_obs[2], dtype=np.float32).reshape(-1),
                1: np.asarray(kickoff_obs[3], dtype=np.float32).reshape(-1),
            },
        )
        self._decoder = fit_observation_state_decoder(
            env,
            num_samples=decoder_samples,
            state_scale=self._state_scale,
        )
        self._raw_obs_space = raw_obs_space
        self._aug_obs_space = build_teammate_obs_space(
            raw_obs_space,
            include_time=self._include_time,
        )
        cc_obs_space = build_cc_obs_space(self._aug_obs_space, act_space)
        self._cc_preprocessor = ModelCatalog.get_preprocessor_for_space(cc_obs_space)

        trainer_action_space = act_space
        if inferred_action_dim is not None:
            trainer_action_space = gym.spaces.Discrete(int(inferred_action_dim))
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            trainer_action_space = gym.spaces.Discrete(int(np.prod(act_space.nvec)))

        register_shared_cc_model()
        register_shared_cc_teammate_aux_model()
        env_name = f"DummyEnvTrainedSharedCCTeammateObsAgent_{os.getpid()}"
        tune.registry.register_env(
            env_name,
            lambda *_: _DummyMultiAgentEnv(cc_obs_space, self._aug_obs_space, trainer_action_space),
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

    def _looks_like_kickoff(self, observation: Dict[int, np.ndarray]) -> bool:
        if not isinstance(observation, dict) or set(observation.keys()) != {0, 1}:
            return False
        current = {
            0: np.asarray(observation[0], dtype=np.float32).reshape(-1),
            1: np.asarray(observation[1], dtype=np.float32).reshape(-1),
        }
        for template in self._kickoff_templates:
            if all(
                current[k].shape == template[k].shape and np.allclose(current[k], template[k], atol=1e-5)
                for k in (0, 1)
            ):
                return True
        return False

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        if self._looks_like_kickoff(observation):
            self._episode_step = 0
        time_tail = (
            normalized_episode_time(episode_step=self._episode_step, max_steps=self._time_max_steps)
            if self._include_time
            else None
        )
        augmented_obs = augment_observation_dict_with_decoded_teammate_state(
            observation,
            self._decoder,
            include_time=self._include_time,
            normalized_time=time_tail,
        )
        zero_obs = None
        for player_id, obs in augmented_obs.items():
            if zero_obs is None:
                zero_obs = np.zeros_like(np.asarray(obs, dtype=np.float32).reshape(-1))
            mate_id = 1 if int(player_id) == 0 else 0
            mate_obs = np.asarray(augmented_obs.get(mate_id, zero_obs), dtype=np.float32).reshape(-1)
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
        self._episode_step += 1
        return actions


Agent = SharedCCTeammateObsAgent
