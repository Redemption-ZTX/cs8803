import argparse
import os
import pickle
from typing import Any, Optional, Tuple

import gym
import numpy as np
import ray
import soccer_twos
from ray import tune
from ray.tune.registry import get_trainable_cls

from utils import create_rllib_env
from checkpoint_utils import (
    unpickle_if_bytes,
    find_nested_key,
    find_torch_state_dict,
    strip_optimizer_state,
    infer_action_dim_from_checkpoint,
    load_policy_weights,
)


ALGORITHM = "PPO"
POLICY_NAME = "default_policy"


class _DummyGymEnv(gym.Env):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        raise RuntimeError("Dummy env")

    def step(self, action):
        raise RuntimeError("Dummy env")


def _extract_score_from_info(info) -> Optional[Tuple[float, float]]:
    if not isinstance(info, dict):
        return None

    for k in (
        "score",
        "scores",
        "team_score",
        "team_scores",
        "goal",
        "goals",
    ):
        if k not in info:
            continue
        v = info.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                return float(v[0]), float(v[1])
            except Exception:
                return None
        if isinstance(v, dict):
            for a, b in (
                ("team0", "team1"),
                ("blue", "orange"),
                ("home", "away"),
                ("left", "right"),
            ):
                if a in v and b in v:
                    try:
                        return float(v[a]), float(v[b])
                    except Exception:
                        return None

    for a, b in (
        ("team0_score", "team1_score"),
        ("blue_score", "orange_score"),
        ("home_score", "away_score"),
    ):
        if a in info and b in info:
            try:
                return float(info[a]), float(info[b])
            except Exception:
                return None

    return None


def _extract_winner_from_info(info) -> Optional[int]:
    if not isinstance(info, dict):
        return None
    for k in ("winner", "winning_team", "result", "game_result", "outcome"):
        if k not in info:
            continue
        v = info.get(k)
        if isinstance(v, str):
            s = v.lower()
            if "team0" in s or "blue" in s or "home" in s or "left" in s:
                return 0
            if "team1" in s or "orange" in s or "away" in s or "right" in s:
                return 1
        if isinstance(v, (int, float)):
            if int(v) in (0, 1):
                return int(v)
    return None


def _resolve_checkpoint_file(path: str) -> str:
    if os.path.isdir(path):
        cands = [p for p in os.listdir(path) if p.startswith("checkpoint-")]
        if not cands:
            raise ValueError(f"No checkpoint-* file found in directory: {path}")
        cands.sort()
        return os.path.join(path, cands[0])
    return path


def load_policy_from_checkpoint(checkpoint_file: str):
    checkpoint_file = os.path.abspath(checkpoint_file)
    with open(checkpoint_file, "rb") as f:
        raw_state = pickle.load(f)

    state = unpickle_if_bytes(raw_state, max_depth=4)

    config_dir = os.path.dirname(checkpoint_file)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    if not os.path.exists(config_path):
        raise ValueError("Could not find params.pkl in either the checkpoint dir or its parent directory.")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    config["num_workers"] = 0
    config["num_gpus"] = 0

    # Create a probe env to infer observation/action spaces.
    env = create_rllib_env(
        {
            "variation": soccer_twos.EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": True,
            "flatten_branched": True,
            "reward_shaping": False,
            "opponent_mix": {"baseline_prob": 1.0},
            "base_port": 19100,
        }
    )
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    try:
        env.close()
    except Exception:
        pass

    tune.registry.register_env(
        "DummyEnv", lambda *_: _DummyGymEnv(obs_space, act_space)
    )
    config["env"] = "DummyEnv"
    config["env_config"] = {}

    cls = get_trainable_cls(ALGORITHM)
    trainer = cls(env=config["env"], config=config)

    policy = load_policy_weights(checkpoint_file, trainer, POLICY_NAME)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--episodes", "-n", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--base_port", "-p", type=int, default=9100)
    args = parser.parse_args()

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

    checkpoint_file = _resolve_checkpoint_file(args.checkpoint)
    policy = load_policy_from_checkpoint(checkpoint_file)

    env = create_rllib_env(
        {
            "variation": soccer_twos.EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": True,
            "flatten_branched": True,
            "reward_shaping": False,
            "opponent_mix": {"baseline_prob": 1.0},
            "base_port": int(args.base_port),
        }
    )

    wins = 0
    ties = 0
    total = 0

    try:
        for _ in range(max(args.episodes, 0)):
            obs = env.reset()
            ep_reward = 0.0
            last_info: Any = None
            for _t in range(args.max_steps):
                act = policy.compute_single_action(obs, explore=False)
                if isinstance(act, tuple) and len(act) >= 1:
                    act = act[0]
                obs, r, done, info = env.step(act)
                ep_reward += float(r)
                last_info = info
                if bool(done):
                    break

            score = _extract_score_from_info(last_info)
            winner = _extract_winner_from_info(last_info)

            if score is not None:
                s0, s1 = score
                if s0 > s1:
                    wins += 1
                elif s0 == s1:
                    ties += 1
            elif winner is not None:
                if winner == 0:
                    wins += 1
            else:
                if ep_reward > 0:
                    wins += 1
                elif ep_reward == 0:
                    ties += 1

            total += 1
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("---- Summary ----")
    print(f"checkpoint: {checkpoint_file}")
    print(f"episodes: {total}")
    print(f"wins: {wins}")
    print(f"ties: {ties}")
    print(f"win_rate: {wins / max(total, 1):.3f}")


if __name__ == "__main__":
    main()
