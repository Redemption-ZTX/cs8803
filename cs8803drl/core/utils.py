from random import uniform as randfloat

import gym
import os
import pickle
import random
import sys
import types
import numpy as np
from typing import Any, Dict
from cs8803drl.core.checkpoint_utils import infer_action_dim_from_checkpoint, load_policy_weights
from cs8803drl.core.obs_teammate import (
    augment_observation_dict_with_teammate_state,
    build_teammate_obs_space,
    extract_player_state_by_agent,
    parse_teammate_state_scale,
    normalized_episode_time,
)

from cs8803drl.core.soccer_info import (
    aggregate_scalar_shaping,
    compute_event_shaping,
    compute_shaping_components,
    compute_terminal_shaping,
    extract_ball_position,
    extract_player_positions,
)
from cs8803drl.branches.expert_coordination import sample_expert_scenario
from cs8803drl.branches.obs_summary import append_ray_summary_features, build_summary_obs_space


if not hasattr(np, "bool"):
    np.bool = bool


try:
    import cv2 as _cv2  # noqa: F401
except Exception:
    _cv2_stub = types.ModuleType("cv2")

    class _OclStub:
        @staticmethod
        def setUseOpenCL(_flag):
            return None

    def _noop(*_args, **_kwargs):
        return None

    _cv2_stub.ocl = _OclStub()
    _cv2_stub.setNumThreads = _noop
    _cv2_stub.setUseOptimized = _noop

    def __getattr__(_name):
        return _noop

    _cv2_stub.__getattr__ = __getattr__
    sys.modules["cv2"] = _cv2_stub

from ray.rllib import MultiAgentEnv
import soccer_twos
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.base_env import BaseEnv


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class RoleTokenObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, token_by_agent: Dict[int, Any]):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                "RoleTokenObsWrapper expects a flat Box observation_space, "
                f"got {type(env.observation_space)!r}"
            )
        self._token_by_agent = {
            int(agent_id): np.asarray(token, dtype=np.float32).reshape(-1)
            for agent_id, token in (token_by_agent or {}).items()
        }
        if not self._token_by_agent:
            raise ValueError("RoleTokenObsWrapper requires a non-empty token_by_agent map.")

        token_dims = {token.shape[0] for token in self._token_by_agent.values()}
        if len(token_dims) != 1:
            raise ValueError("All role tokens must have the same dimensionality.")
        self._token_dim = int(next(iter(token_dims)))

        base_space = env.observation_space
        low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        token_low = np.zeros((self._token_dim,), dtype=np.float32)
        token_high = np.ones((self._token_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, token_low], axis=0),
            high=np.concatenate([high, token_high], axis=0),
            dtype=np.float32,
        )

    def _token_for_agent(self, agent_id: int) -> np.ndarray:
        token = self._token_by_agent.get(int(agent_id))
        if token is None:
            return np.zeros((self._token_dim,), dtype=np.float32)
        return token

    def _augment_obs(self, obs):
        if not isinstance(obs, dict):
            return obs

        out = {}
        for agent_id, value in obs.items():
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            token = self._token_for_agent(agent_id)
            out[agent_id] = np.concatenate([arr, token], axis=0).astype(np.float32, copy=False)
        return out

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._augment_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._augment_obs(obs), reward, done, info


class RewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        time_penalty: float = 0.001,
        ball_progress_scale: float = 0.01,
        goal_proximity_scale: float = 0.0,
        goal_proximity_gamma: float = 0.99,
        goal_center_x: float = 15.0,
        goal_center_y: float = 0.0,
        event_shot_reward: float = 0.0,
        event_tackle_reward: float = 0.0,
        event_clearance_reward: float = 0.0,
        event_cooldown_steps: int = 10,
        shot_x_threshold: float = 10.0,
        shot_ball_dx_min: float = 0.5,
        clearance_from_x: float = -8.0,
        clearance_to_x: float = -4.0,
        opponent_progress_penalty_scale: float = 0.0,
        possession_dist: float = 1.25,
        possession_bonus: float = 0.002,
        progress_requires_possession: bool = False,
        deep_zone_outer_threshold: float = 0.0,
        deep_zone_outer_penalty: float = 0.0,
        deep_zone_inner_threshold: float = 0.0,
        deep_zone_inner_penalty: float = 0.0,
        defensive_survival_threshold: float = 0.0,
        defensive_survival_bonus: float = 0.0,
        fast_loss_threshold_steps: int = 0,
        fast_loss_penalty_per_step: float = 0.0,
        ball_progress_scale_by_agent: Dict[int, float] = None,
        opponent_progress_penalty_scale_by_agent: Dict[int, float] = None,
        possession_bonus_by_agent: Dict[int, float] = None,
        deep_zone_outer_penalty_by_agent: Dict[int, float] = None,
        deep_zone_inner_penalty_by_agent: Dict[int, float] = None,
        role_binding_mode: str = "",
        role_binding_team_agent_ids=(0, 1),
        ball_progress_scale_by_role: Dict[str, float] = None,
        opponent_progress_penalty_scale_by_role: Dict[str, float] = None,
        possession_bonus_by_role: Dict[str, float] = None,
        deep_zone_outer_penalty_by_role: Dict[str, float] = None,
        deep_zone_inner_penalty_by_role: Dict[str, float] = None,
        scalar_reward_agent_ids=(0, 1),
        debug_info: bool = False,
    ):
        super().__init__(env)
        self._time_penalty = float(time_penalty)
        self._ball_progress_scale = float(ball_progress_scale)
        self._goal_proximity_scale = float(goal_proximity_scale)
        self._goal_proximity_gamma = float(goal_proximity_gamma)
        self._goal_center_x = float(goal_center_x)
        self._goal_center_y = float(goal_center_y)
        self._event_shot_reward = float(event_shot_reward)
        self._event_tackle_reward = float(event_tackle_reward)
        self._event_clearance_reward = float(event_clearance_reward)
        self._event_cooldown_steps = int(event_cooldown_steps)
        self._shot_x_threshold = float(shot_x_threshold)
        self._shot_ball_dx_min = float(shot_ball_dx_min)
        self._clearance_from_x = float(clearance_from_x)
        self._clearance_to_x = float(clearance_to_x)
        self._opponent_progress_penalty_scale = float(opponent_progress_penalty_scale)
        self._possession_dist = float(possession_dist)
        self._possession_bonus = float(possession_bonus)
        self._progress_requires_possession = bool(progress_requires_possession)
        self._deep_zone_outer_threshold = float(deep_zone_outer_threshold)
        self._deep_zone_outer_penalty = float(deep_zone_outer_penalty)
        self._deep_zone_inner_threshold = float(deep_zone_inner_threshold)
        self._deep_zone_inner_penalty = float(deep_zone_inner_penalty)
        self._defensive_survival_threshold = float(defensive_survival_threshold)
        self._defensive_survival_bonus = float(defensive_survival_bonus)
        self._fast_loss_threshold_steps = int(fast_loss_threshold_steps)
        self._fast_loss_penalty_per_step = float(fast_loss_penalty_per_step)
        self._ball_progress_scale_by_agent = self._coerce_agent_scale_map(ball_progress_scale_by_agent)
        self._opponent_progress_penalty_scale_by_agent = self._coerce_agent_scale_map(
            opponent_progress_penalty_scale_by_agent
        )
        self._possession_bonus_by_agent = self._coerce_agent_scale_map(possession_bonus_by_agent)
        self._deep_zone_outer_penalty_by_agent = self._coerce_agent_scale_map(
            deep_zone_outer_penalty_by_agent
        )
        self._deep_zone_inner_penalty_by_agent = self._coerce_agent_scale_map(
            deep_zone_inner_penalty_by_agent
        )
        self._role_binding_mode = str(role_binding_mode or "").strip().lower()
        self._role_binding_team_agent_ids = tuple(
            int(agent_id) for agent_id in (role_binding_team_agent_ids or ())
        )
        self._ball_progress_scale_by_role = self._coerce_role_scale_map(ball_progress_scale_by_role)
        self._opponent_progress_penalty_scale_by_role = self._coerce_role_scale_map(
            opponent_progress_penalty_scale_by_role
        )
        self._possession_bonus_by_role = self._coerce_role_scale_map(possession_bonus_by_role)
        self._deep_zone_outer_penalty_by_role = self._coerce_role_scale_map(
            deep_zone_outer_penalty_by_role
        )
        self._deep_zone_inner_penalty_by_role = self._coerce_role_scale_map(
            deep_zone_inner_penalty_by_role
        )
        self._scalar_reward_agent_ids = tuple(int(agent_id) for agent_id in scalar_reward_agent_ids)
        self._debug_info = bool(debug_info)
        self._prev_ball_x = None
        self._prev_ball_pos = None
        self._prev_possessing_team = None
        self._event_last_trigger_steps = {}
        self._episode_steps = 0
        self._role_by_agent = None

    def reset(self, **kwargs):
        self._prev_ball_x = None
        self._prev_ball_pos = None
        self._prev_possessing_team = None
        self._event_last_trigger_steps = {}
        self._episode_steps = 0
        self._role_by_agent = None
        return self.env.reset(**kwargs)

    @staticmethod
    def _extract_ball_pos(info):
        return extract_ball_position(info)

    @staticmethod
    def _extract_player_positions(info):
        return extract_player_positions(info)

    @staticmethod
    def _coerce_agent_scale_map(value):
        if not isinstance(value, dict):
            return None
        out = {}
        for agent_id, scale in value.items():
            try:
                out[int(agent_id)] = float(scale)
            except Exception:
                continue
        return out or None

    @staticmethod
    def _coerce_role_scale_map(value):
        if not isinstance(value, dict):
            return None
        out = {}
        for role_name, scale in value.items():
            key = str(role_name).strip().lower()
            if not key:
                continue
            try:
                out[key] = float(scale)
            except Exception:
                continue
        return out or None

    def _resolve_role_binding(self, info):
        if self._role_binding_mode != "spawn_depth":
            return None
        if len(self._role_binding_team_agent_ids) != 2:
            return None
        player_positions = extract_player_positions(info)
        if not isinstance(player_positions, dict):
            return None
        try:
            aid0, aid1 = self._role_binding_team_agent_ids
            pos0 = player_positions[int(aid0)]
            pos1 = player_positions[int(aid1)]
        except Exception:
            return None

        p0 = np.asarray(pos0, dtype=np.float32).reshape(-1)
        p1 = np.asarray(pos1, dtype=np.float32).reshape(-1)
        if p0.shape[0] < 2 or p1.shape[0] < 2:
            return None

        candidates = [
            (int(aid0), float(p0[0]), float(p0[1])),
            (int(aid1), float(p1[0]), float(p1[1])),
        ]
        mean_x = sum(item[1] for item in candidates) / float(len(candidates))

        if mean_x <= 0.0:
            striker = max(candidates, key=lambda item: (item[1], -abs(item[2]), -item[0]))[0]
            defender = min(candidates, key=lambda item: (item[1], abs(item[2]), item[0]))[0]
        else:
            striker = min(candidates, key=lambda item: (item[1], abs(item[2]), item[0]))[0]
            defender = max(candidates, key=lambda item: (item[1], -abs(item[2]), -item[0]))[0]

        if striker == defender:
            return None
        return {
            int(striker): "striker",
            int(defender): "defender",
        }

    @staticmethod
    def _merge_role_scale_map(agent_map, role_map, role_by_agent):
        merged = dict(agent_map or {})
        if not isinstance(role_map, dict) or not isinstance(role_by_agent, dict):
            return merged or None
        for agent_id, role_name in role_by_agent.items():
            role_key = str(role_name).strip().lower()
            if role_key not in role_map:
                continue
            merged[int(agent_id)] = float(role_map[role_key])
        return merged or None

    @staticmethod
    def _apply_additive_reward(reward, add_by_agent):
        if isinstance(reward, dict):
            for k, v in add_by_agent.items():
                if k in reward:
                    reward[k] = float(reward[k]) + float(v)
            return reward

        # list/np.ndarray (common in soccer_twos default wrappers)
        if isinstance(reward, (list, tuple, np.ndarray)):
            r = np.asarray(reward, dtype=np.float32).copy()
            for k, v in add_by_agent.items():
                if isinstance(k, int) and 0 <= k < len(r):
                    r[k] += float(v)
            # return same type as input for compatibility
            return r if isinstance(reward, np.ndarray) else r.tolist()

        return reward

    @staticmethod
    def _is_scalar_reward(reward):
        return isinstance(reward, (int, float, np.generic)) and not isinstance(reward, bool)

    @staticmethod
    def _episode_done(done):
        if isinstance(done, dict):
            if "__all__" in done:
                return bool(done["__all__"])
            return bool(max(done.values())) if done else False
        return bool(done)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._episode_steps += 1

        add = {}
        scalar_delta = 0.0
        prev_ball_x = self._prev_ball_x
        if self._role_binding_mode and self._role_by_agent is None:
            self._role_by_agent = self._resolve_role_binding(info)

        ball_progress_scale_by_agent = self._merge_role_scale_map(
            self._ball_progress_scale_by_agent,
            self._ball_progress_scale_by_role,
            self._role_by_agent,
        )
        opponent_progress_penalty_scale_by_agent = self._merge_role_scale_map(
            self._opponent_progress_penalty_scale_by_agent,
            self._opponent_progress_penalty_scale_by_role,
            self._role_by_agent,
        )
        possession_bonus_by_agent = self._merge_role_scale_map(
            self._possession_bonus_by_agent,
            self._possession_bonus_by_role,
            self._role_by_agent,
        )
        deep_zone_outer_penalty_by_agent = self._merge_role_scale_map(
            self._deep_zone_outer_penalty_by_agent,
            self._deep_zone_outer_penalty_by_role,
            self._role_by_agent,
        )
        deep_zone_inner_penalty_by_agent = self._merge_role_scale_map(
            self._deep_zone_inner_penalty_by_agent,
            self._deep_zone_inner_penalty_by_role,
            self._role_by_agent,
        )

        # 1) time penalty
        if self._time_penalty != 0:
            if self._is_scalar_reward(reward):
                scalar_delta -= self._time_penalty
            elif isinstance(reward, dict):
                add.update({k: -self._time_penalty for k in reward.keys()})
            elif isinstance(reward, (list, tuple, np.ndarray)):
                add.update({i: -self._time_penalty for i in range(len(reward))})

        shaping_add, shaping_debug = compute_shaping_components(
            info,
            prev_ball_x,
            prev_ball_pos=self._prev_ball_pos,
            ball_progress_scale=self._ball_progress_scale,
            goal_proximity_scale=self._goal_proximity_scale,
            goal_proximity_gamma=self._goal_proximity_gamma,
            goal_center_x=self._goal_center_x,
            goal_center_y=self._goal_center_y,
            opponent_progress_penalty_scale=self._opponent_progress_penalty_scale,
            possession_dist=self._possession_dist,
            possession_bonus=self._possession_bonus,
            progress_requires_possession=self._progress_requires_possession,
            deep_zone_outer_threshold=self._deep_zone_outer_threshold,
            deep_zone_outer_penalty=self._deep_zone_outer_penalty,
            deep_zone_inner_threshold=self._deep_zone_inner_threshold,
            deep_zone_inner_penalty=self._deep_zone_inner_penalty,
            defensive_survival_threshold=self._defensive_survival_threshold,
            defensive_survival_bonus=self._defensive_survival_bonus,
            ball_progress_scale_by_agent=ball_progress_scale_by_agent,
            opponent_progress_penalty_scale_by_agent=opponent_progress_penalty_scale_by_agent,
            possession_bonus_by_agent=possession_bonus_by_agent,
            deep_zone_outer_penalty_by_agent=deep_zone_outer_penalty_by_agent,
            deep_zone_inner_penalty_by_agent=deep_zone_inner_penalty_by_agent,
        )
        for agent_id, delta in shaping_add.items():
            add[agent_id] = add.get(agent_id, 0.0) + delta

        current_ball_pos = shaping_debug.get("ball_pos")
        event_add, event_debug, updated_triggers = compute_event_shaping(
            episode_steps=self._episode_steps,
            ball_x=None if current_ball_pos is None else float(current_ball_pos[0]),
            prev_ball_x=prev_ball_x,
            ball_dx=shaping_debug.get("ball_dx"),
            possessing_team=shaping_debug.get("possessing_team"),
            prev_possessing_team=self._prev_possessing_team,
            possession_confirmed=bool(shaping_debug.get("possession_confirmed")),
            event_shot_reward=self._event_shot_reward,
            event_tackle_reward=self._event_tackle_reward,
            event_clearance_reward=self._event_clearance_reward,
            event_cooldown_steps=self._event_cooldown_steps,
            shot_x_threshold=self._shot_x_threshold,
            shot_ball_dx_min=self._shot_ball_dx_min,
            clearance_from_x=self._clearance_from_x,
            clearance_to_x=self._clearance_to_x,
            last_trigger_steps=self._event_last_trigger_steps,
        )
        for agent_id, delta in event_add.items():
            add[agent_id] = add.get(agent_id, 0.0) + delta
        self._event_last_trigger_steps.update(updated_triggers)

        terminal_debug = {}
        if self._episode_done(done):
            terminal_add, terminal_debug = compute_terminal_shaping(
                info,
                episode_steps=self._episode_steps,
                fast_loss_threshold_steps=self._fast_loss_threshold_steps,
                fast_loss_penalty_per_step=self._fast_loss_penalty_per_step,
            )
            for agent_id, delta in terminal_add.items():
                add[agent_id] = add.get(agent_id, 0.0) + delta

        ball_pos = current_ball_pos
        if ball_pos is not None:
            self._prev_ball_x = float(ball_pos[0])
            self._prev_ball_pos = (float(ball_pos[0]), float(ball_pos[1]))
        possessing_team = shaping_debug.get("possessing_team")
        if bool(shaping_debug.get("possession_confirmed")) and possessing_team in (0, 1):
            self._prev_possessing_team = int(possessing_team)

        if self._is_scalar_reward(reward):
            scalar_delta += aggregate_scalar_shaping(
                add,
                controlled_agent_ids=self._scalar_reward_agent_ids,
            )
            reward = float(reward) + float(scalar_delta)
        elif add:
            reward = self._apply_additive_reward(reward, add)

        if self._debug_info and isinstance(info, dict):
            info["_reward_shaping"] = {
                **shaping_debug,
                **event_debug,
                **terminal_debug,
                "applied_reward": dict(add),
                "scalar_reward_delta": float(scalar_delta),
                "episode_steps": int(self._episode_steps),
                "role_binding_mode": self._role_binding_mode or None,
                "role_by_agent": dict(self._role_by_agent or {}),
            }

        return obs, reward, done, info


class RaySummaryObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                "RaySummaryObsWrapper expects a flat Box observation_space, "
                f"got {type(env.observation_space)!r}"
            )
        self.observation_space = build_summary_obs_space(env.observation_space)

    def _augment_obs(self, obs):
        if isinstance(obs, dict):
            return {
                agent_id: append_ray_summary_features(value)
                for agent_id, value in obs.items()
            }
        return append_ray_summary_features(obs)

    def reset(self, **kwargs):
        return self._augment_obs(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._augment_obs(obs), reward, done, info


class TeammateStateObsWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        include_time: bool = False,
        max_steps: int = 1500,
        state_scale=None,
    ):
        super().__init__(env)
        self._include_time = bool(include_time)
        self._max_steps = max(1, int(max_steps))
        self._episode_steps = 0
        self._state_scale = parse_teammate_state_scale(state_scale)
        self._debug_remaining = max(
            0, int(os.environ.get("TEAMMATE_WRAPPER_DEBUG_STEPS", "0") or "0")
        )
        self.observation_space = build_teammate_obs_space(
            env.observation_space,
            include_time=self._include_time,
        )

    def _maybe_debug_dump(self, info_by_agent, state_by_agent, augmented):
        if self._debug_remaining <= 0:
            return
        self._debug_remaining -= 1
        top_keys = list(info_by_agent.keys()) if isinstance(info_by_agent, dict) else type(info_by_agent)
        print(f"[wrapper] info top keys: {top_keys}", flush=True)
        if isinstance(info_by_agent, dict) and 0 in info_by_agent:
            payload0 = info_by_agent.get(0)
            payload0_keys = list(payload0.keys()) if isinstance(payload0, dict) else type(payload0)
            print(f"[wrapper] info[0].keys(): {payload0_keys}", flush=True)
        print(f"[wrapper] state_by_agent: {state_by_agent}", flush=True)
        tail_dim = 4 + (1 if self._include_time else 0)
        if isinstance(augmented, dict):
            for agent_id, obs in augmented.items():
                tail = np.asarray(obs, dtype=np.float32).reshape(-1)[-tail_dim:]
                print(f"[wrapper] agent {agent_id} tail: {tail}", flush=True)

    def _augment_obs(self, obs, info_by_agent=None):
        state_by_agent = extract_player_state_by_agent(
            info_by_agent,
            state_scale=self._state_scale,
        )
        time_tail = (
            normalized_episode_time(episode_step=self._episode_steps, max_steps=self._max_steps)
            if self._include_time
            else None
        )
        augmented = augment_observation_dict_with_teammate_state(
            obs,
            state_by_agent,
            include_time=self._include_time,
            normalized_time=time_tail,
        )
        self._maybe_debug_dump(info_by_agent, state_by_agent, augmented)
        return augmented

    def reset(self, **kwargs):
        self._episode_steps = 0
        obs = self.env.reset(**kwargs)
        return self._augment_obs(obs, info_by_agent=None)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._episode_steps += 1
        return self._augment_obs(obs, info_by_agent=info), reward, done, info


class ScenarioResetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, mode: str):
        super().__init__(env)
        self._mode = str(mode or "").strip().lower()
        if not self._mode:
            raise ValueError("ScenarioResetWrapper requires a non-empty mode.")

    def _find_env_channel(self):
        cur = self.env
        for _ in range(16):
            env_channel = getattr(cur, "env_channel", None)
            if env_channel is not None:
                return env_channel
            if not hasattr(cur, "env"):
                break
            cur = cur.env
        return None

    def reset(self, **kwargs):
        env_channel = self._find_env_channel()
        if env_channel is None:
            raise RuntimeError("Could not locate env_channel for scenario reset.")
        env_channel.set_parameters(**sample_expert_scenario(self._mode))
        return self.env.reset(**kwargs)


_BASELINE_POLICY_CACHE = None
_CHECKPOINT_POLICY_CACHE = {}


class _DummyPolicyEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        raise RuntimeError("Dummy policy env should never be stepped")

    def step(self, action):
        raise RuntimeError("Dummy policy env should never be stepped")


def _get_baseline_policy():
    global _BASELINE_POLICY_CACHE
    if _BASELINE_POLICY_CACHE is not None:
        return _BASELINE_POLICY_CACHE

    from ceia_baseline_agent.agent_ray import CHECKPOINT_PATH

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    config_dir = os.path.dirname(CHECKPOINT_PATH)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError("Could not find baseline params.pkl")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    config["num_workers"] = 0
    config["num_gpus"] = 0

    # Dummy env: only needed so RLlib can build/restore the policy.
    # Baseline checkpoints may include multiagent policies; RLlib requires BaseEnv/MultiAgentEnv.
    tune.registry.register_env("DummyEnvBaseline", lambda *_: BaseEnv())
    config["env"] = "DummyEnvBaseline"

    cls = get_trainable_cls("PPO")
    agent = cls(env=config["env"], config=config)
    agent.restore(CHECKPOINT_PATH)

    # Baseline self-play checkpoints use policy id "default".
    policy = agent.get_policy("default")
    if policy is None:
        policy = agent.get_policy("default_policy")
    if policy is None:
        raise ValueError("Could not find baseline policy in restored trainer")

    _BASELINE_POLICY_CACHE = policy
    return policy


def _get_checkpoint_policy(checkpoint_path, *, observation_space, action_space):
    checkpoint_path = os.path.abspath(checkpoint_path)
    cache_key = (
        checkpoint_path,
        tuple(getattr(observation_space, "shape", ()) or ()),
        repr(action_space),
    )
    if cache_key in _CHECKPOINT_POLICY_CACHE:
        return _CHECKPOINT_POLICY_CACHE[cache_key]

    ray.init(ignore_reinit_error=True, include_dashboard=False)

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

    inferred_action_dim = infer_action_dim_from_checkpoint(checkpoint_path)
    trainer_action_space = action_space
    if inferred_action_dim is not None:
        trainer_action_space = gym.spaces.Discrete(int(inferred_action_dim))
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        trainer_action_space = gym.spaces.Discrete(int(np.prod(action_space.nvec)))

    env_name = f"DummyEnvCheckpointPolicy_{abs(hash(cache_key))}"
    tune.registry.register_env(
        env_name,
        lambda *_: _DummyPolicyEnv(observation_space, trainer_action_space),
    )
    config["env"] = env_name
    config["env_config"] = {}

    cls = get_trainable_cls("PPO")
    trainer = cls(env=config["env"], config=config)
    policy = load_policy_weights(checkpoint_path, trainer, "default_policy")
    _CHECKPOINT_POLICY_CACHE[cache_key] = policy
    return policy


def _make_mixed_opponent_policy(*, baseline_prob: float, action_space):
    act_n = int(getattr(action_space, "n", 0) or 0)

    def opponent_policy(obs, *args, **kwargs):
        # 10% random
        if random.random() > float(baseline_prob):
            if act_n > 0:
                return random.randrange(act_n)
            return 0

        policy = _get_baseline_policy()
        inv_lookup = kwargs.get("_inv_lookup")

        # soccer_twos may call opponent_policy with either a numpy obs or a dict of obs.
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                a, *_ = policy.compute_single_action(v)
                if isinstance(a, np.ndarray) and a.size == 1:
                    a = int(a.reshape(-1)[0])
                elif isinstance(a, (np.integer,)):
                    a = int(a)
                elif inv_lookup is not None and isinstance(a, (list, tuple, np.ndarray)):
                    key = tuple(np.asarray(a).reshape(-1).tolist())
                    if key in inv_lookup:
                        a = int(inv_lookup[key])
                out[k] = a
            return out

        a, *_ = policy.compute_single_action(obs)
        if isinstance(a, np.ndarray) and a.size == 1:
            return int(a.reshape(-1)[0])
        if isinstance(a, (np.integer,)):
            return int(a)
        if inv_lookup is not None and isinstance(a, (list, tuple, np.ndarray)):
            key = tuple(np.asarray(a).reshape(-1).tolist())
            if key in inv_lookup:
                return int(inv_lookup[key])
        return a

    return opponent_policy


def _make_checkpoint_action_policy(*, checkpoint_path: str, observation_space, action_space):
    policy = _get_checkpoint_policy(
        checkpoint_path,
        observation_space=observation_space,
        action_space=action_space,
    )

    def teammate_policy(obs, *args, **kwargs):
        action, *_ = policy.compute_single_action(obs, explore=False)
        if isinstance(action, np.ndarray) and action.size == 1:
            return int(action.reshape(-1)[0])
        if isinstance(action, (np.integer, int)):
            return int(action)
        return action

    return teammate_policy


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    shaping_cfg = None
    if isinstance(env_config, dict) and env_config.get("reward_shaping"):
        shaping_cfg = env_config.get("reward_shaping")
        # do not forward wrapper config into soccer_twos.make
        env_config = {k: v for k, v in env_config.items() if k != "reward_shaping"}

    opponent_mix = None
    if isinstance(env_config, dict) and env_config.get("opponent_mix"):
        opponent_mix = env_config.get("opponent_mix")
        # do not forward wrapper config into soccer_twos.make
        env_config = {k: v for k, v in env_config.items() if k != "opponent_mix"}

    role_token_cfg = None
    if isinstance(env_config, dict) and env_config.get("role_token_obs"):
        role_token_cfg = env_config.get("role_token_obs")
        env_config = {k: v for k, v in env_config.items() if k != "role_token_obs"}

    teammate_state_obs_cfg = False
    if isinstance(env_config, dict) and env_config.get("teammate_state_obs"):
        teammate_state_obs_cfg = env_config.get("teammate_state_obs")
        env_config = {k: v for k, v in env_config.items() if k != "teammate_state_obs"}

    ray_summary_cfg = False
    if isinstance(env_config, dict) and env_config.get("ray_summary_obs"):
        ray_summary_cfg = bool(env_config.get("ray_summary_obs"))
        env_config = {k: v for k, v in env_config.items() if k != "ray_summary_obs"}

    scenario_reset_cfg = None
    if isinstance(env_config, dict) and env_config.get("scenario_reset"):
        scenario_reset_cfg = str(env_config.get("scenario_reset") or "").strip()
        env_config = {k: v for k, v in env_config.items() if k != "scenario_reset"}

    teammate_checkpoint = None
    if isinstance(env_config, dict) and env_config.get("teammate_checkpoint"):
        teammate_checkpoint = str(env_config.get("teammate_checkpoint") or "").strip()
        env_config = {k: v for k, v in env_config.items() if k != "teammate_checkpoint"}

    # Inject mixed opponent policy (baseline/random) for team_vs_policy.
    if (
        isinstance(env_config, dict)
        and env_config.get("variation") == soccer_twos.EnvType.team_vs_policy
        and (opponent_mix or teammate_checkpoint)
    ):
        baseline_prob = float(opponent_mix.get("baseline_prob", 0.9)) if isinstance(opponent_mix, dict) else 0.9
        env_config = dict(env_config)
        # We need the action space for random sampling; create a temporary env to inspect it.
        tmp_env = soccer_twos.make(**env_config)
        inv_lookup = None
        flattener = getattr(tmp_env, "_flattener", None)
        if flattener is not None and hasattr(flattener, "action_lookup"):
            inv_lookup = {tuple(v): k for k, v in flattener.action_lookup.items()}

        base_opponent = _make_mixed_opponent_policy(
            baseline_prob=baseline_prob,
            action_space=tmp_env.action_space,
        )

        def opponent_policy(obs, *args, **kwargs):
            return base_opponent(obs, *args, _inv_lookup=inv_lookup, **kwargs)

        if opponent_mix:
            env_config["opponent_policy"] = opponent_policy
        if teammate_checkpoint:
            env_config["teammate_policy"] = _make_checkpoint_action_policy(
                checkpoint_path=teammate_checkpoint,
                observation_space=tmp_env.observation_space,
                action_space=tmp_env.action_space,
            )
        env = tmp_env
    else:
        env = soccer_twos.make(**env_config)

    if scenario_reset_cfg:
        env = ScenarioResetWrapper(env, mode=scenario_reset_cfg)
    if ray_summary_cfg:
        env = RaySummaryObsWrapper(env)
    if teammate_state_obs_cfg:
        if isinstance(teammate_state_obs_cfg, dict):
            env = TeammateStateObsWrapper(
                env,
                include_time=bool(teammate_state_obs_cfg.get("include_time", False)),
                max_steps=int(teammate_state_obs_cfg.get("max_steps", 1500)),
                state_scale=teammate_state_obs_cfg.get("state_scale"),
            )
        else:
            env = TeammateStateObsWrapper(env)
    if role_token_cfg:
        token_by_agent = role_token_cfg if isinstance(role_token_cfg, dict) else {}
        env = RoleTokenObsWrapper(env, token_by_agent=token_by_agent)
    if shaping_cfg:
        if shaping_cfg is True:
            env = RewardShapingWrapper(env)
        elif isinstance(shaping_cfg, dict):
            env = RewardShapingWrapper(env, **shaping_cfg)
        else:
            env = RewardShapingWrapper(env)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
