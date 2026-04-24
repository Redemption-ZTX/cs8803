"""Phase-specific match eval — wraps soccer_twos.make with ScenarioResetWrapper.

Methodology fix (per user 2026-04-22): standalone baseline WR is not a fair
metric for sub-task specialists (e.g., 103A INTERCEPTOR can't solo-finish so
0.548 WR doesn't reflect its DUEL-phase capability). Instead, evaluate each
specialist in **its training scenario init** — measures how well it does what
it was trained for.

Companion to `cs8803drl/evaluation/evaluate_matches.py` but adds:
  --scenario-reset {attack_expert,defense_expert,interceptor_subtask,defender_subtask,dribble_subtask}
which applies `ScenarioResetWrapper` to env.reset(). Every episode starts in
the specified scenario init (matches the specialist's training distribution).

Usage:
    python -m scripts.research.eval_in_scenario \\
        --m1 agents.v_103A_interceptor \\
        --m2 ceia_baseline_agent \\
        --scenario-reset interceptor_subtask \\
        -n 200 --base-port 62005

Or vs SOTA (specialist's phase value verification):
    python -m scripts.research.eval_in_scenario \\
        --m1 agents.v_103A_interceptor \\
        --m2 agents.v_sota_055v2_extend_1750 \\
        --scenario-reset interceptor_subtask \\
        -n 200 --base-port 62015
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gym  # noqa: F401  (required for soccer_twos)
import numpy as np
import soccer_twos

from cs8803drl.core.utils import ScenarioResetWrapper
from cs8803drl.evaluation.evaluate_matches import (
    _team_local_obs,
    _team_reward,
    _remap_team_actions,
    adapt_actions_to_env,
    episode_done,
    load_agent,
    make_env_with_retry,
)


def _supported_scenarios() -> Tuple[str, ...]:
    return (
        "attack_expert",
        "defense_expert",
        "interceptor_subtask",
        "defender_subtask",
        "dribble_subtask",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m1",
        "--team0_module",
        dest="team0_module",
        required=True,
        help="Python module for team0 agent (e.g., agents.v_selector_phase4)",
    )
    parser.add_argument(
        "--m2",
        "--team1_module",
        dest="team1_module",
        required=True,
        help="Python module for team1 agent (e.g., ceia_baseline_agent or agents.v_sota_055v2_extend_1750)",
    )
    parser.add_argument(
        "--scenario-reset",
        type=str,
        default="",
        choices=("",) + _supported_scenarios(),
        help="Scenario init mode for ScenarioResetWrapper. Empty = default env (no scenario init).",
    )
    parser.add_argument("-n", "--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("-p", "--base-port", type=int, default=62005)
    parser.add_argument(
        "--save-log",
        type=str,
        default="",
        help="Optional output log file (in addition to stdout).",
    )
    args = parser.parse_args()

    print(f"=== eval_in_scenario ===")
    print(f"  team0_module:   {args.team0_module}")
    print(f"  team1_module:   {args.team1_module}")
    print(f"  scenario_reset: {args.scenario_reset or '(none, default env)'}")
    print(f"  episodes:       {args.episodes}")
    print(f"  base_port:      {args.base_port}")

    env, port_used = make_env_with_retry(base_port=args.base_port, max_tries=20)
    print(f"  env initialized on port {port_used}")
    if args.scenario_reset:
        env = ScenarioResetWrapper(env, mode=args.scenario_reset)
        print(f"  ScenarioResetWrapper applied: mode={args.scenario_reset}")

    agent0 = load_agent(args.team0_module, env)
    print(f"  team0 agent loaded: {args.team0_module}")

    agent1 = load_agent(args.team1_module, env)
    print(f"  team1 agent loaded: {args.team1_module}")

    wins0 = 0
    wins1 = 0
    ties = 0
    all_steps = []

    start = time.time()

    for ep in range(args.episodes):
        obs = env.reset()
        if ep % 2 == 0:
            team0_ids = (0, 1)
            team1_ids = (2, 3)
        else:
            team0_ids = (2, 3)
            team1_ids = (0, 1)

        team0_reward_total = 0.0
        team1_reward_total = 0.0
        final_team0_step_reward = 0.0
        final_team1_step_reward = 0.0
        total_steps = 0

        for step_idx in range(args.max_steps):
            obs0 = _team_local_obs(obs, team0_ids)
            obs1 = _team_local_obs(obs, team1_ids)
            act0 = _remap_team_actions(agent0.act(obs0), team0_ids)
            act1 = _remap_team_actions(agent1.act(obs1), team1_ids)

            act0 = adapt_actions_to_env(act0, env)
            act1 = adapt_actions_to_env(act1, env)

            action: Dict[int, Any] = {}
            action.update(act0)
            action.update(act1)

            obs, reward, done, info = env.step(action)
            final_team0_step_reward = _team_reward(reward, team0_ids)
            final_team1_step_reward = _team_reward(reward, team1_ids)
            team0_reward_total += final_team0_step_reward
            team1_reward_total += final_team1_step_reward
            total_steps = step_idx + 1
            if episode_done(done):
                break

        if final_team0_step_reward > 0.05:
            wins0 += 1
            outcome = "team0_win"
        elif final_team1_step_reward > 0.05:
            wins1 += 1
            outcome = "team1_win"
        else:
            ties += 1
            outcome = "tie"
        all_steps.append(total_steps)
        if (ep + 1) % 50 == 0 or ep < 5:
            elapsed = time.time() - start
            print(
                f"  ep {ep+1:4d}/{args.episodes}  outcome={outcome}  "
                f"steps={total_steps:4d}  W-L-T={wins0}-{wins1}-{ties}  "
                f"WR_so_far={wins0/(ep+1):.3f}  elapsed={elapsed:.0f}s",
                flush=True,
            )

    print()
    print("---- Summary (eval_in_scenario) ----")
    print(f"team0_module: {args.team0_module}")
    print(f"team1_module: {args.team1_module}")
    print(f"scenario_reset: {args.scenario_reset or '(none)'}")
    print(f"episodes: {args.episodes}")
    print(f"team0_wins: {wins0}")
    print(f"team1_wins: {wins1}")
    print(f"ties: {ties}")
    print(f"team0_win_rate: {wins0/args.episodes:.4f}")
    print(f"team0_non_loss_rate: {(wins0 + ties)/args.episodes:.4f}")
    arr = np.asarray(all_steps)
    print(
        f"episode_steps: mean={arr.mean():.1f} median={int(np.median(arr))} "
        f"p75={int(np.percentile(arr, 75))} min={arr.min()} max={arr.max()}"
    )

    if args.save_log:
        Path(args.save_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_log, "w") as f:
            f.write(
                f"team0_module={args.team0_module}\n"
                f"team1_module={args.team1_module}\n"
                f"scenario_reset={args.scenario_reset}\n"
                f"episodes={args.episodes}\n"
                f"team0_wins={wins0}\n"
                f"team1_wins={wins1}\n"
                f"ties={ties}\n"
                f"team0_win_rate={wins0/args.episodes:.4f}\n"
                f"episode_steps_mean={arr.mean():.1f}\n"
            )
        print(f"  saved log → {args.save_log}", flush=True)


if __name__ == "__main__":
    main()
