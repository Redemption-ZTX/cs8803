from random import uniform as randfloat

import gym
import os
import pickle
import random
import sys
import types
import numpy as np
from typing import Any, Dict


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


class RewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        time_penalty: float = 0.001,
        ball_progress_scale: float = 0.01,
        opponent_progress_penalty_scale: float = 0.0,
        possession_dist: float = 1.25,
        possession_bonus: float = 0.002,
    ):
        super().__init__(env)
        self._time_penalty = float(time_penalty)
        self._ball_progress_scale = float(ball_progress_scale)
        self._opponent_progress_penalty_scale = float(opponent_progress_penalty_scale)
        self._possession_dist = float(possession_dist)
        self._possession_bonus = float(possession_bonus)
        self._prev_ball_x = None

    def reset(self, **kwargs):
        self._prev_ball_x = None
        return self.env.reset(**kwargs)

    @staticmethod
    def _extract_ball_pos(info):
        """Best-effort extraction of ball position from env info."""
        if info is None:
            return None

        # info may be a dict keyed by agent_id, or a list/tuple indexed by agent.
        if isinstance(info, dict):
            candidates = list(info.values())
        elif isinstance(info, (list, tuple)) and len(info) > 0:
            candidates = list(info)
        else:
            candidates = [info]

        for c in candidates:
            if not isinstance(c, dict):
                continue
            ball = c.get("ball_info") or c.get("ball")
            if isinstance(ball, dict):
                pos = ball.get("position")
            else:
                pos = None
            if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                return float(pos[0]), float(pos[1])
        return None

    @staticmethod
    def _extract_player_positions(info):
        if info is None:
            return {}
        if isinstance(info, dict):
            items = info.items()
        elif isinstance(info, (list, tuple)):
            items = enumerate(info)
        else:
            items = [(0, info)]

        out = {}
        for agent_id, c in items:
            if not isinstance(c, dict):
                continue
            p = c.get("player_info") or c.get("player")
            if isinstance(p, dict):
                pos = p.get("position")
            else:
                pos = None
            if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                out[agent_id] = (float(pos[0]), float(pos[1]))
        return out

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

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        add = {}

        # 1) time penalty
        if self._time_penalty != 0:
            if isinstance(reward, dict):
                add.update({k: -self._time_penalty for k in reward.keys()})
            elif isinstance(reward, (list, tuple, np.ndarray)):
                add.update({i: -self._time_penalty for i in range(len(reward))})

        # 2) ball x progress shaping (team0 wants +x, team1 wants -x)
        ball_pos = self._extract_ball_pos(info)
        if ball_pos is not None and self._ball_progress_scale != 0:
            ball_x, _ = ball_pos
            if self._prev_ball_x is not None:
                dx = ball_x - self._prev_ball_x
                # Assumes agent ids 0,1 are team0; 2,3 are team1 (as in examples)
                for aid in (0, 1):
                    add[aid] = add.get(aid, 0.0) + self._ball_progress_scale * dx
                for aid in (2, 3):
                    add[aid] = add.get(aid, 0.0) - self._ball_progress_scale * dx
            self._prev_ball_x = ball_x

        # 2b) opponent progress penalty: if the team closest to the ball is advancing
        # in its preferred x direction, penalize the defending team.
        if (
            ball_pos is not None
            and self._prev_ball_x is not None
            and self._opponent_progress_penalty_scale != 0
        ):
            ball_x, ball_y = ball_pos
            player_pos = self._extract_player_positions(info)
            if player_pos:
                closest_aid = None
                closest_d = None
                for aid, (px, py) in player_pos.items():
                    d = float(np.hypot(px - ball_x, py - ball_y))
                    if closest_d is None or d < closest_d:
                        closest_d = d
                        closest_aid = int(aid)

                if closest_aid is not None:
                    possessing_team = 0 if closest_aid in (0, 1) else 1
                    defending_team = 1 - possessing_team

                    # Recompute dx from current ball_x and previous ball_x snapshot.
                    dx = float(ball_x - self._prev_ball_x)

                    # Team0 attacks toward +x; Team1 attacks toward -x.
                    if possessing_team == 0:
                        progress = max(dx, 0.0)
                    else:
                        progress = max(-dx, 0.0)

                    if progress > 0:
                        penalty = -self._opponent_progress_penalty_scale * progress
                        if defending_team == 0:
                            for aid in (0, 1):
                                add[aid] = add.get(aid, 0.0) + penalty
                        else:
                            for aid in (2, 3):
                                add[aid] = add.get(aid, 0.0) + penalty

        # 3) possession bonus if close to ball
        if ball_pos is not None and self._possession_bonus != 0:
            ball_x, ball_y = ball_pos
            player_pos = self._extract_player_positions(info)
            for aid, (px, py) in player_pos.items():
                d = float(np.hypot(px - ball_x, py - ball_y))
                if d <= self._possession_dist:
                    add[aid] = add.get(aid, 0.0) + self._possession_bonus

        if add:
            reward = self._apply_additive_reward(reward, add)

        return obs, reward, done, info


_BASELINE_POLICY_CACHE = None


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

    # Inject mixed opponent policy (baseline/random) for team_vs_policy.
    if opponent_mix and isinstance(env_config, dict) and env_config.get("variation") == soccer_twos.EnvType.team_vs_policy:
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

        env_config["opponent_policy"] = opponent_policy
        env = tmp_env
    else:
        env = soccer_twos.make(**env_config)

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
