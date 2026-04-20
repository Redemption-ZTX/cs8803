from random import uniform as randfloat

import gym
import os
import pickle
import random
import sys
import types
import numpy as np
from typing import Any, Dict, List
from cs8803drl.core.checkpoint_utils import infer_action_dim_from_checkpoint, load_policy_weights
from cs8803drl.core.obs_teammate import (
    augment_observation_dict_with_teammate_state,
    build_teammate_obs_space,
    extract_player_state_by_agent,
    parse_teammate_state_scale,
    normalized_episode_time,
)

from cs8803drl.core.soccer_info import (
    TEAM0_AGENT_IDS,
    TEAM1_AGENT_IDS,
    aggregate_scalar_shaping,
    compute_event_shaping,
    compute_shaping_components,
    compute_specialist_outcome_override,
    compute_team_coordination_potentials,
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
        team_spacing_scale: float = 0.0,
        team_coverage_scale: float = 0.0,
        team_potential_gamma: float = 0.99,
        team_near_ball_threshold: float = 3.0,
        team_spacing_min: float = 2.0,
        team_spacing_max: float = 6.0,
        scalar_reward_agent_ids=(0, 1),
        specialist_mode: str = "none",
        fast_win_threshold: int = 100,
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
        self._team_spacing_scale = float(team_spacing_scale)
        self._team_coverage_scale = float(team_coverage_scale)
        self._team_potential_gamma = float(team_potential_gamma)
        self._team_near_ball_threshold = float(team_near_ball_threshold)
        self._team_spacing_min = float(team_spacing_min)
        self._team_spacing_max = float(team_spacing_max)
        self._scalar_reward_agent_ids = tuple(int(agent_id) for agent_id in scalar_reward_agent_ids)
        self._controlled_team_id = self._infer_controlled_team_id(self._scalar_reward_agent_ids)
        self._specialist_mode = str(specialist_mode or "none").strip().lower()
        self._fast_win_threshold = int(fast_win_threshold)
        self._debug_info = bool(debug_info)
        self._prev_ball_x = None
        self._prev_ball_pos = None
        self._prev_possessing_team = None
        self._prev_team_coord_total = None
        self._event_last_trigger_steps = {}
        self._episode_steps = 0
        self._role_by_agent = None

    def reset(self, **kwargs):
        self._prev_ball_x = None
        self._prev_ball_pos = None
        self._prev_possessing_team = None
        self._prev_team_coord_total = None
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

    @staticmethod
    def _infer_controlled_team_id(agent_ids):
        ids = tuple(int(agent_id) for agent_id in (agent_ids or ()))
        if ids and all(agent_id in TEAM0_AGENT_IDS for agent_id in ids):
            return 0
        if ids and all(agent_id in TEAM1_AGENT_IDS for agent_id in ids):
            return 1
        return None

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
        team_coord_debug = {}
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
        specialist_debug = {}
        if self._episode_done(done):
            terminal_add, terminal_debug = compute_terminal_shaping(
                info,
                episode_steps=self._episode_steps,
                fast_loss_threshold_steps=self._fast_loss_threshold_steps,
                fast_loss_penalty_per_step=self._fast_loss_penalty_per_step,
            )
            for agent_id, delta in terminal_add.items():
                add[agent_id] = add.get(agent_id, 0.0) + delta

            if self._specialist_mode != "none":
                specialist_add, specialist_debug = compute_specialist_outcome_override(
                    info,
                    episode_steps=self._episode_steps,
                    mode=self._specialist_mode,
                    fast_win_threshold=self._fast_win_threshold,
                    controlled_team_id=self._controlled_team_id,
                )
                for agent_id, delta in specialist_add.items():
                    add[agent_id] = add.get(agent_id, 0.0) + delta

        if self._team_spacing_scale != 0.0 or self._team_coverage_scale != 0.0:
            team_coord_phi, team_coord_debug = compute_team_coordination_potentials(
                info,
                near_ball_threshold=self._team_near_ball_threshold,
                spacing_min=self._team_spacing_min,
                spacing_max=self._team_spacing_max,
            )
            team_coord_total = {}
            team_coord_delta = {}
            prev_team_coord_total = self._prev_team_coord_total or {}
            for team_id in (0, 1):
                phi = team_coord_phi.get(int(team_id), {})
                total = (
                    self._team_spacing_scale * float(phi.get("spacing", 0.0))
                    + self._team_coverage_scale * float(phi.get("coverage", 0.0))
                )
                team_coord_total[int(team_id)] = float(total)
                prev_total = prev_team_coord_total.get(int(team_id))
                if prev_total is None:
                    delta = 0.0
                else:
                    delta = self._team_potential_gamma * float(total) - float(prev_total)
                team_coord_delta[int(team_id)] = float(delta)
                team_coord_debug[f"team_coord_total_team{team_id}"] = float(total)
                team_coord_debug[f"team_coord_delta_team{team_id}"] = float(delta)

            self._prev_team_coord_total = dict(team_coord_total)
            if self._is_scalar_reward(reward):
                controlled_team = self._controlled_team_id
                if controlled_team in team_coord_delta:
                    scalar_delta += float(team_coord_delta[controlled_team])
            else:
                for team_id, agent_ids in ((0, TEAM0_AGENT_IDS), (1, TEAM1_AGENT_IDS)):
                    delta = float(team_coord_delta.get(int(team_id), 0.0))
                    if delta == 0.0:
                        continue
                    shared_delta = delta / float(len(agent_ids))
                    for agent_id in agent_ids:
                        add[agent_id] = add.get(agent_id, 0.0) + shared_delta

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
                **team_coord_debug,
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


def _find_wrapper(env, wrapper_cls, *, max_depth=16):
    """Walk gym wrapper chain (.env links) looking for an instance of
    `wrapper_cls`. Returns the matching wrapper or None."""
    cur = env
    for _ in range(max_depth):
        if isinstance(cur, wrapper_cls):
            return cur
        inner = getattr(cur, "env", None)
        if inner is None or inner is cur:
            return None
        cur = inner
    return None


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


def _coerce_env_action(action, *, action_space, action_flattener=None):
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        if isinstance(action, np.ndarray):
            if action.ndim == 1 and action.size == len(action_space.nvec):
                return action.astype(np.int64, copy=False)
            if action.shape == ():
                action = int(action.item())
        if isinstance(action, (list, tuple)):
            arr = np.asarray(action)
            if arr.ndim == 1 and arr.size == len(action_space.nvec):
                return arr.astype(np.int64, copy=False)
            if arr.shape == ():
                action = int(arr.item())
        if isinstance(action, (np.integer, int)):
            flat = int(action)
            if action_flattener is not None:
                try:
                    return np.asarray(action_flattener.lookup_action(flat), dtype=np.int64)
                except Exception:
                    pass
            return _unflatten_discrete_to_multidiscrete(flat, np.asarray(action_space.nvec))
    if isinstance(action, np.ndarray) and action.shape == ():
        return int(action.item())
    if isinstance(action, (np.integer, int)):
        return int(action)
    return action


def _make_baseline_only_opponent_policy(*, action_space, action_flattener=None):
    def opponent_policy(obs, *args, **kwargs):
        policy = _get_baseline_policy()
        action, *_ = policy.compute_single_action(obs, explore=False)
        return _coerce_env_action(
            action,
            action_space=action_space,
            action_flattener=action_flattener,
        )

    return opponent_policy


class _EpisodeOpponentPoolPolicy:
    """Episode-level opponent sampling over baseline/random/frozen policies."""

    def __init__(self, entries: List[Dict[str, Any]], allow_zero_weight: bool = False):
        """Initialize pool. If allow_zero_weight=True, keep entries with 0 weight
        (they'll get eps probability) so curriculum/dynamic updates can reactivate them.
        """
        normalized_entries = []
        for entry in entries or []:
            weight = float(entry.get("weight", 0.0))
            if weight <= 0 and not allow_zero_weight:
                continue
            policy = entry.get("policy")
            if policy is None:
                continue
            normalized_entries.append(
                {
                    "name": str(entry.get("name") or "opponent"),
                    "weight": max(weight, 0.0),
                    "policy": policy,
                }
            )
        if not normalized_entries:
            raise ValueError("_EpisodeOpponentPoolPolicy requires >=1 entry.")

        weights = np.asarray([entry["weight"] for entry in normalized_entries], dtype=np.float64)
        weights = np.maximum(weights, 1e-9)
        self._entries = normalized_entries
        self._weights = weights / weights.sum()
        self._current_index = None
        self._debug = os.environ.get("FROZEN_OPPONENT_POOL_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def reset_episode(self):
        self._current_index = int(np.random.choice(len(self._entries), p=self._weights))
        current_policy = self._entries[self._current_index]["policy"]
        if hasattr(current_policy, "reset_episode"):
            current_policy.reset_episode()
        if self._debug:
            print(
                "[opponent-pool] selected="
                f"{self._entries[self._current_index]['name']}"
            )

    def __call__(self, obs, *args, **kwargs):
        if self._current_index is None:
            self.reset_episode()
        return self._entries[self._current_index]["policy"](obs, *args, **kwargs)

    def set_pool_weights(self, new_weights):
        """Update entry weights from {name: weight} dict (snapshot-058 curriculum hook).

        Missing names retain their old weights. Negative or zero weights are clamped
        to a tiny eps so the entry still has nonzero selection probability (so we
        don't silently drop entries — caller can set ~0 to effectively disable).
        Re-normalizes weights after update.
        """
        if not isinstance(new_weights, dict):
            return
        for entry in self._entries:
            name = entry["name"]
            if name in new_weights:
                entry["weight"] = max(float(new_weights[name]), 1e-9)
        weights = np.asarray(
            [entry["weight"] for entry in self._entries], dtype=np.float64
        )
        weights = np.maximum(weights, 1e-9)
        self._weights = weights / weights.sum()
        if self._debug:
            print(
                "[opponent-pool] weights updated → "
                + ", ".join(
                    f"{e['name']}={w:.3f}" for e, w in zip(self._entries, self._weights)
                )
            )


def _build_frozen_opponent_policy(*, spec, obs_space, action_space):
    kind = str(spec.get("kind", "")).strip().lower()
    checkpoint_path = str(spec.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise ValueError(f"Missing checkpoint_path in frozen opponent spec: {spec}")

    if kind in {"team_ray", "team", "siamese"}:
        from cs8803drl.core.frozen_team_policy import FrozenTeamPolicy

        return FrozenTeamPolicy(
            checkpoint_path,
            obs_space=obs_space,
            action_space=action_space,
        )
    if kind in {"shared_cc", "per_agent", "mappo"}:
        from cs8803drl.core.frozen_shared_cc_policy import FrozenSharedCCPolicy

        return FrozenSharedCCPolicy(
            checkpoint_path,
            obs_space=obs_space,
            action_space=action_space,
        )
    raise ValueError(f"Unknown frozen opponent kind: {kind}")


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

    # snapshot-046: optional frozen team-level checkpoint as opponent_policy.
    # When set, replaces the mixed baseline/random opponent with a
    # FrozenTeamPolicy adapter that wraps a team-level (Siamese) checkpoint
    # and dispatches per-agent calls into a single joint inference.
    team_opponent_checkpoint = None
    if isinstance(env_config, dict) and env_config.get("team_opponent_checkpoint"):
        team_opponent_checkpoint = str(env_config.get("team_opponent_checkpoint") or "").strip()
        env_config = {k: v for k, v in env_config.items() if k != "team_opponent_checkpoint"}

    if team_opponent_checkpoint and opponent_mix:
        raise ValueError(
            "create_rllib_env: env_config['team_opponent_checkpoint'] and "
            "env_config['opponent_mix'] are mutually exclusive. The team-level "
            "frozen opponent fully replaces the baseline/random mix."
        )

    # Inject explicit opponent / teammate policies for team_vs_policy.
    if (
        isinstance(env_config, dict)
        and env_config.get("variation") == soccer_twos.EnvType.team_vs_policy
        and (opponent_mix or teammate_checkpoint or team_opponent_checkpoint)
    ):
        baseline_prob = (
            float(opponent_mix.get("baseline_prob", 0.9))
            if isinstance(opponent_mix, dict)
            else 0.9
        )
        frozen_opponents = (
            list(opponent_mix.get("frozen_opponents", []))
            if isinstance(opponent_mix, dict)
            else []
        )
        env_config = dict(env_config)
        tmp_env = soccer_twos.make(**env_config)
        from soccer_twos.wrappers import TeamVsPolicyWrapper

        tvp_wrapper = _find_wrapper(tmp_env, TeamVsPolicyWrapper)
        if tvp_wrapper is None:
            raise RuntimeError(
                "create_rllib_env: could not locate TeamVsPolicyWrapper to install "
                "explicit opponent/teammate policies."
            )

        # The TeamVsPolicyWrapper itself exposes JOINT team-level obs/action
        # spaces. For opponent / teammate callables we need the underlying
        # per-agent spaces from the wrapped env.
        base_env = tvp_wrapper.env
        flattener = getattr(base_env, "_flattener", None)
        action_flattener = flattener if hasattr(flattener, "lookup_action") else None

        if teammate_checkpoint:
            tvp_wrapper.set_teammate_policy(
                _make_checkpoint_action_policy(
                    checkpoint_path=teammate_checkpoint,
                    observation_space=base_env.observation_space,
                    action_space=base_env.action_space,
                )
            )

        if team_opponent_checkpoint:
            from cs8803drl.core.frozen_team_policy import FrozenTeamPolicy

            tvp_wrapper.set_opponent_policy(
                FrozenTeamPolicy(
                    team_opponent_checkpoint,
                    obs_space=base_env.observation_space,
                    action_space=base_env.action_space,
                )
            )
        elif opponent_mix:
            curriculum_enabled = bool(opponent_mix.get("curriculum_enabled", False)) if isinstance(opponent_mix, dict) else False
            pool_entries = []
            # When curriculum is enabled, ALWAYS include both baseline+random
            # entries (with eps weight if currently inactive) so curriculum can
            # reactivate them at runtime via set_pool_weights.
            include_baseline = curriculum_enabled or baseline_prob > 0.0
            include_random = curriculum_enabled or (1.0 - baseline_prob) > 0.0
            if include_baseline:
                pool_entries.append(
                    {
                        "name": "baseline",
                        "weight": baseline_prob if baseline_prob > 0.0 else 1e-9,
                        "policy": _make_baseline_only_opponent_policy(
                            action_space=base_env.action_space,
                            action_flattener=action_flattener,
                        ),
                    }
                )
            if frozen_opponents:
                for idx, spec in enumerate(frozen_opponents):
                    pool_entries.append(
                        {
                            "name": str(spec.get("name") or f"frozen_{idx}"),
                            "weight": float(spec.get("weight", 0.0)),
                            "policy": _build_frozen_opponent_policy(
                                spec=spec,
                                obs_space=base_env.observation_space,
                                action_space=base_env.action_space,
                            ),
                        }
                    )
            elif include_random:
                random_weight = max(0.0, 1.0 - baseline_prob)
                pool_entries.append(
                    {
                        "name": "random",
                        "weight": random_weight if random_weight > 0.0 else 1e-9,
                        "policy": lambda *_args, **_kwargs: base_env.action_space.sample(),
                    }
                )

            if len(pool_entries) == 1 and not curriculum_enabled:
                tvp_wrapper.set_opponent_policy(pool_entries[0]["policy"])
                if hasattr(pool_entries[0]["policy"], "reset_episode"):
                    pool_entries[0]["policy"].reset_episode()
            else:
                tvp_wrapper.set_opponent_policy(
                    _EpisodeOpponentPoolPolicy(
                        pool_entries, allow_zero_weight=curriculum_enabled
                    )
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

    # snapshot-036 Path C: optional learned-reward shaping on top of v2 shaping.
    # Applied AFTER RewardShapingWrapper so the learned signal is additive to any
    # existing hand-designed shaping. Takes env_config["learned_reward_shaping"]:
    #   { "model_path": str, "weight": float, "team0_agent_ids": list[int],
    #     "apply_to_team1": bool }
    learned_reward_cfg = env_config.get("learned_reward_shaping", None)
    if learned_reward_cfg:
        if isinstance(learned_reward_cfg, dict) and learned_reward_cfg.get("model_path"):
            try:
                from cs8803drl.imitation.learned_reward_shaping import (
                    LearnedRewardShapingWrapper,
                )
                env = LearnedRewardShapingWrapper(
                    env,
                    reward_model_path=str(learned_reward_cfg["model_path"]),
                    shaping_weight=float(learned_reward_cfg.get("weight", 0.01)),
                    team0_agent_ids=tuple(
                        int(i) for i in learned_reward_cfg.get("team0_agent_ids", (0, 1))
                    ),
                    apply_to_team1=bool(learned_reward_cfg.get("apply_to_team1", False)),
                    warmup_steps=int(learned_reward_cfg.get("warmup_steps", 0)),
                )
            except Exception as exc:
                print(
                    f"[create_rllib_env] LearnedRewardShapingWrapper skipped: {exc!r}"
                )

    # A2 PBRS: per-step ΔV(s) shaping using calibrated outcome predictor (direction_1b_v3).
    # Applied AFTER any learned_reward_shaping so PBRS bonus is additive too.
    # env_config["outcome_pbrs_shaping"] = { "predictor_path": str, "weight": float,
    #   "team0_agent_ids": list[int], "warmup_steps": int, "max_buffer_steps": int }
    outcome_pbrs_cfg = env_config.get("outcome_pbrs_shaping", None)
    if outcome_pbrs_cfg:
        if isinstance(outcome_pbrs_cfg, dict) and outcome_pbrs_cfg.get("predictor_path"):
            try:
                from cs8803drl.imitation.outcome_pbrs_shaping import OutcomePBRSWrapper
                env = OutcomePBRSWrapper(
                    env,
                    predictor_path=str(outcome_pbrs_cfg["predictor_path"]),
                    shaping_weight=float(outcome_pbrs_cfg.get("weight", 0.01)),
                    team0_agent_ids=tuple(
                        int(i) for i in outcome_pbrs_cfg.get("team0_agent_ids", (0, 1))
                    ),
                    warmup_steps=int(outcome_pbrs_cfg.get("warmup_steps", 0)),
                    max_buffer_steps=int(outcome_pbrs_cfg.get("max_buffer_steps", 80)),
                )
            except Exception as exc:
                print(f"[create_rllib_env] OutcomePBRSWrapper skipped: {exc!r}")

    # SNAPSHOT-057 (Tier A3) RND intrinsic motivation wrapper. Applied AFTER
    # all reward-shaping wrappers so intrinsic is additive to env + shaping.
    # env_config["rnd_shaping"] = { "weight": float, "team0_agent_ids": [0,1],
    #   "hidden_dim": int, "embed_dim": int, "lr": float, "train_every_steps": int,
    #   "train_batch_size": int, "warmup_steps": int }
    rnd_cfg = env_config.get("rnd_shaping", None)
    if rnd_cfg:
        if isinstance(rnd_cfg, dict) and rnd_cfg.get("weight"):
            try:
                from cs8803drl.branches.rnd_wrapper import RNDWrapper
                env = RNDWrapper(
                    env,
                    shaping_weight=float(rnd_cfg["weight"]),
                    team0_agent_ids=tuple(
                        int(i) for i in rnd_cfg.get("team0_agent_ids", (0, 1))
                    ),
                    hidden_dim=int(rnd_cfg.get("hidden_dim", 256)),
                    embed_dim=int(rnd_cfg.get("embed_dim", 64)),
                    rnd_lr=float(rnd_cfg.get("lr", 1e-4)),
                    train_every_steps=int(rnd_cfg.get("train_every_steps", 16)),
                    train_batch_size=int(rnd_cfg.get("train_batch_size", 256)),
                    warmup_steps=int(rnd_cfg.get("warmup_steps", 0)),
                    device=str(rnd_cfg.get("device", "cpu")),
                    random_seed=int(rnd_cfg.get("random_seed", 1234)),
                )
            except Exception as exc:
                print(f"[create_rllib_env] RNDWrapper skipped: {exc!r}")

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
