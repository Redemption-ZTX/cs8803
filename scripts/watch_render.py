#!/usr/bin/env python3
"""Render a Soccer-Twos agent matchup without the initial headless bootstrap.

`python -m soccer_twos.watch` first launches a non-rendered Soccer-Twos binary
just to obtain `observation_space` / `action_space` for agent construction.
On shared login nodes that first bootstrap occasionally times out before the
real watch environment even starts.

This helper sidesteps that step by instantiating agents against a tiny dummy
gym env with the fixed competition spaces:
  - per-agent observation: Box(336,)
  - per-agent action: MultiDiscrete([3, 3, 3])

After the agents are loaded, it launches only the real rendered watch env.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_pythonpath = os.environ.get("PYTHONPATH", "")
if str(REPO_ROOT) not in [p for p in _pythonpath.split(os.pathsep) if p]:
    os.environ["PYTHONPATH"] = (
        str(REPO_ROOT) if not _pythonpath else str(REPO_ROOT) + os.pathsep + _pythonpath
    )

try:
    import sitecustomize as _project_sitecustomize  # noqa: F401
except Exception:
    _project_sitecustomize = None

import gym
import numpy as np
import soccer_twos
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from soccer_twos.package import ROLLOUT_ENV_PATH
from soccer_twos.side_channels import EnvConfigurationChannel
from soccer_twos.utils import DummyEnv, get_agent_class
from soccer_twos.wrappers import EnvChannelWrapper, MultiAgentUnityWrapper


def _build_dummy_env() -> DummyEnv:
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(336,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete(np.asarray([3, 3, 3], dtype=np.int64))
    return DummyEnv(obs_space, action_space)


def _make_watch_env(base_port: int, blue_team_name: str, orange_team_name: str):
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=1, quality_level=5)

    env_channel = EnvConfigurationChannel()
    env_channel.set_parameters(
        blue_team_name=blue_team_name,
        orange_team_name=orange_team_name,
    )

    additional_args = [
        "-screen-width",
        os.environ.get("SOCCER_TWOS_SCREEN_WIDTH", "1280"),
        "-screen-height",
        os.environ.get("SOCCER_TWOS_SCREEN_HEIGHT", "720"),
        "-screen-fullscreen",
        os.environ.get("SOCCER_TWOS_FULLSCREEN", "0"),
    ]
    timeout_wait = int(os.environ.get("SOCCER_TWOS_TIMEOUT_WAIT", "120"))
    log_folder = os.environ.get("SOCCER_TWOS_LOG_FOLDER")

    logging.info(
        "Launching Unity watch env on port %s with args %s (timeout=%ss)",
        base_port,
        additional_args,
        timeout_wait,
    )
    unity_env = UnityEnvironment(
        ROLLOUT_ENV_PATH,
        no_graphics=False,
        base_port=base_port,
        timeout_wait=timeout_wait,
        additional_args=additional_args,
        side_channels=[engine_channel, env_channel],
        log_folder=log_folder if log_folder else None,
    )
    env = MultiAgentUnityWrapper(unity_env)
    env = EnvChannelWrapper(env, env_channel)
    return env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Soccer-Twos matchup without the initial headless bootstrap."
    )
    parser.add_argument("-m", "--agent-module", help="Selfplay agent module")
    parser.add_argument("-m1", "--agent1-module", help="Team 1 agent module")
    parser.add_argument("-m2", "--agent2-module", help="Team 2 agent module")
    parser.add_argument("-p", "--base-port", type=int, default=54115, help="Base communication port")
    return parser.parse_args()


def main() -> int:
    loglevel = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=loglevel)
    args = _parse_args()

    if args.agent_module:
        agent1_module_name = args.agent_module
        agent2_module_name = args.agent_module
    elif args.agent1_module and args.agent2_module:
        agent1_module_name = args.agent1_module
        agent2_module_name = args.agent2_module
    else:
        raise ValueError("Must specify selfplay (-m) or team (-m1, -m2) agent modules")

    logging.info("Loading %s as blue team", agent1_module_name)
    agent1_module = importlib.import_module(agent1_module_name)
    logging.info("Loading %s as orange team", agent2_module_name)
    agent2_module = importlib.import_module(agent2_module_name)

    dummy_env = _build_dummy_env()
    agent1 = get_agent_class(agent1_module)(dummy_env)
    agent2 = get_agent_class(agent2_module)(dummy_env)

    logging.info("%s name is %s", agent1_module_name, agent1.name)
    logging.info("%s name is %s", agent2_module_name, agent2.name)

    env = _make_watch_env(
        base_port=args.base_port,
        blue_team_name=agent1.name,
        orange_team_name=agent2.name,
    )

    obs = env.reset()
    team0_reward = 0
    team1_reward = 0
    while True:
        agent1_actions = agent1.act({0: obs[0], 1: obs[1]})
        agent2_actions = agent2.act({0: obs[2], 1: obs[3]})
        actions = {
            0: agent1_actions[0],
            1: agent1_actions[1],
            2: agent2_actions[0],
            3: agent2_actions[1],
        }

        obs, reward, done, _info = env.step(actions)
        team0_reward += reward[0] + reward[1]
        team1_reward += reward[2] + reward[3]
        if max(done.values()):
            logging.info("Total Reward: %s x %s", team0_reward, team1_reward)
            team0_reward = 0
            team1_reward = 0
            env.reset()


if __name__ == "__main__":
    raise SystemExit(main())
