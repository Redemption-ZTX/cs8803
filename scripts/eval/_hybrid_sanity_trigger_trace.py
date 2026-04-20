"""SNAPSHOT-048 §6 step 3 — R1 sanity check: trigger trace + ball_x consistency.

Runs the hybrid evaluator's per-step machinery for 2 episodes (one per side
configuration) with verbose printing. Validates:
  1. extract_team0_ball_x returns sensible values per step (not all 0/inf/NaN)
  2. Sign convention: when team0 controls a player on positive-x side, ball_x
     gets flipped so "negative = our half" stays consistent
  3. Trigger fire rates: alpha ~15-30%, beta ~5-15%
  4. Side-swap (ep 0 = team0 as blue, ep 1 = team0 as orange) gives similar
     fire rates -> sign-flipping logic works

Usage:
    /home/hice1/wsun377/.conda/envs/soccertwos/bin/python \
        scripts/eval/_hybrid_sanity_trigger_trace.py \
        --student-module cs8803drl.deployment.trained_team_ray_agent \
        --student-checkpoint <path>/checkpoint-1040 \
        --base-port 50201
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs8803drl.deployment._hybrid.switching import (
    TRIGGER_REGISTRY,
    BETA_WINDOW_LEN,
    extract_team0_ball_x,
)
from cs8803drl.evaluation.evaluate_matches import (
    adapt_actions_to_env,
    episode_done,
    load_agent,
    make_env_with_retry,
    _remap_team_actions,
    _team_local_obs,
    _team_reward,
)
from cs8803drl.evaluation.failure_cases import infer_team0_outcome


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--student-module", required=True)
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--base-port", type=int, default=50201)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--print-stride", type=int, default=20, help="Print one row per N steps")
    p.add_argument("--episodes", type=int, default=2, help="Total episodes to run (alternating sides)")
    p.add_argument("--verbose-eps", type=int, default=2, help="Print per-step trace for first N episodes; rest stats-only")
    args = p.parse_args()

    os.environ["TRAINED_RAY_CHECKPOINT"] = args.student_checkpoint

    env, port = make_env_with_retry(base_port=args.base_port)
    print(f"[env] connected port={port}")
    student = load_agent(args.student_module, env)
    team1_agent = load_agent("ceia_baseline_agent", env)

    alpha = TRIGGER_REGISTRY["alpha"]
    beta = TRIGGER_REGISTRY["beta"]

    def _new_acc():
        return {
            "alpha_fires": 0, "beta_fires": 0, "total_steps": 0,
            "episodes": 0, "ball_x_all": [],
            # outcome-conditional: how many fires happened in WINNING vs LOSING episodes
            "win_eps": 0, "loss_eps": 0, "tie_eps": 0,
            "alpha_fires_in_wins": 0, "beta_fires_in_wins": 0, "steps_in_wins": 0,
            "alpha_fires_in_losses": 0, "beta_fires_in_losses": 0, "steps_in_losses": 0,
        }

    aggregate = {"blue": _new_acc(), "orange": _new_acc()}

    for ep in range(args.episodes):
        team0_ids = (0, 1) if ep % 2 == 0 else (2, 3)
        team1_ids = (2, 3) if ep % 2 == 0 else (0, 1)
        side = "blue" if ep % 2 == 0 else "orange"
        verbose = ep < args.verbose_eps

        recent_ball_x: deque = deque(maxlen=BETA_WINDOW_LEN)
        ball_x_samples: List[float] = []
        alpha_fires = 0
        beta_fires = 0
        total_steps = 0

        obs = env.reset()
        info: Dict[Any, Any] = {}
        team0_reward_total = 0.0
        team1_reward_total = 0.0
        final_team0_step_reward = 0.0
        final_team1_step_reward = 0.0

        if verbose:
            print()
            print(f"=== EP {ep} (team0 as {side}) ===")
            print(f"  step | ball_x_world | ball_x_signed | alpha | beta | mean(window)")
            print(f"  -----+--------------+---------------+-------+------+-------------")

        for step_idx in range(args.max_steps):
            obs0 = _team_local_obs(obs, team0_ids)
            obs1 = _team_local_obs(obs, team1_ids)

            ball_x = extract_team0_ball_x(info, team0_ids) if info else 0.0
            recent_ball_x.append(ball_x)
            ball_x_samples.append(ball_x)

            fired_alpha = alpha(ball_x, list(recent_ball_x))
            fired_beta = beta(ball_x, list(recent_ball_x))
            if fired_alpha:
                alpha_fires += 1
            if fired_beta:
                beta_fires += 1
            total_steps += 1

            # Print stride
            if verbose and step_idx % args.print_stride == 0:
                ball_x_world_raw = 0.0
                if info and team0_ids[0] in info:
                    bp = info[team0_ids[0]].get("ball_info", {}).get("position")
                    if bp is not None and len(bp) > 0:
                        ball_x_world_raw = float(bp[0])
                window_mean = sum(recent_ball_x) / max(len(recent_ball_x), 1)
                print(
                    f"  {step_idx:4d} | {ball_x_world_raw:+12.3f} | {ball_x:+13.3f} | "
                    f"{'YES' if fired_alpha else '   '}   | {'YES' if fired_beta else '   '}  | {window_mean:+.3f}"
                )

            # Always run student on team0 — this trace is for trigger inspection only.
            act0 = _remap_team_actions(student.act(obs0), team0_ids)
            act1 = _remap_team_actions(team1_agent.act(obs1), team1_ids)
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
            if episode_done(done):
                break

        outcome, _score, _winner = infer_team0_outcome(
            final_info=info if isinstance(info, dict) else {},
            final_team0_step_reward=final_team0_step_reward,
            final_team1_step_reward=final_team1_step_reward,
            cumulative_team0_reward=team0_reward_total,
            cumulative_team1_reward=team1_reward_total,
        )

        if verbose and ball_x_samples:
            print()
            print(f"  [stats] steps={total_steps}")
            print(f"  [stats] ball_x_signed: mean={statistics.mean(ball_x_samples):+.3f}  median={statistics.median(ball_x_samples):+.3f}  min={min(ball_x_samples):+.3f}  max={max(ball_x_samples):+.3f}")
            print(f"  [stats] alpha fires: {alpha_fires}/{total_steps}  ({alpha_fires/max(total_steps,1):.1%})")
            print(f"  [stats] beta  fires: {beta_fires}/{total_steps}  ({beta_fires/max(total_steps,1):.1%})")

        side_acc = aggregate[side]
        side_acc["alpha_fires"] += alpha_fires
        side_acc["beta_fires"] += beta_fires
        side_acc["total_steps"] += total_steps
        side_acc["episodes"] += 1
        side_acc["ball_x_all"].extend(ball_x_samples)
        if outcome == "team0_win":
            side_acc["win_eps"] += 1
            side_acc["alpha_fires_in_wins"] += alpha_fires
            side_acc["beta_fires_in_wins"] += beta_fires
            side_acc["steps_in_wins"] += total_steps
        elif outcome == "team1_win":
            side_acc["loss_eps"] += 1
            side_acc["alpha_fires_in_losses"] += alpha_fires
            side_acc["beta_fires_in_losses"] += beta_fires
            side_acc["steps_in_losses"] += total_steps
        else:
            side_acc["tie_eps"] += 1

        if verbose:
            print(f"  [outcome] {outcome}  cum_reward team0={team0_reward_total:.3f} team1={team1_reward_total:.3f}")

    try:
        env.close()
    except Exception:
        pass

    print()
    print("=" * 72)
    print(f"AGGREGATE STATS over {args.episodes} episodes")
    print("=" * 72)
    overall_alpha = 0
    overall_beta = 0
    overall_steps = 0
    overall_wins = 0
    overall_losses = 0
    overall_ties = 0
    for side, acc in aggregate.items():
        n_eps = acc["episodes"]
        n_steps = acc["total_steps"]
        if n_steps == 0:
            continue
        ball = acc["ball_x_all"]
        win_pct = acc["win_eps"] / max(n_eps, 1)
        in_wins = acc["steps_in_wins"]
        in_losses = acc["steps_in_losses"]
        alpha_in_wins_rate = acc["alpha_fires_in_wins"] / max(in_wins, 1)
        alpha_in_losses_rate = acc["alpha_fires_in_losses"] / max(in_losses, 1)
        beta_in_wins_rate = acc["beta_fires_in_wins"] / max(in_wins, 1)
        beta_in_losses_rate = acc["beta_fires_in_losses"] / max(in_losses, 1)

        print(
            f"\n  {side:6} ({n_eps} eps, {n_steps} steps, W={acc['win_eps']} L={acc['loss_eps']} T={acc['tie_eps']}, WR={win_pct:.2f}):"
            f"\n    ball_x_signed:  mean={statistics.mean(ball):+.3f}  median={statistics.median(ball):+.3f}  range=[{min(ball):+.2f}, {max(ball):+.2f}]"
            f"\n    alpha fires:    {acc['alpha_fires']}/{n_steps}  ({acc['alpha_fires']/n_steps:.1%})  | in wins: {alpha_in_wins_rate:.1%} | in losses: {alpha_in_losses_rate:.1%}"
            f"\n    beta  fires:    {acc['beta_fires']}/{n_steps}  ({acc['beta_fires']/n_steps:.1%})  | in wins: {beta_in_wins_rate:.1%} | in losses: {beta_in_losses_rate:.1%}"
        )
        overall_alpha += acc["alpha_fires"]
        overall_beta += acc["beta_fires"]
        overall_steps += n_steps
        overall_wins += acc["win_eps"]
        overall_losses += acc["loss_eps"]
        overall_ties += acc["tie_eps"]

    overall_n = overall_wins + overall_losses + overall_ties
    print(
        f"\n  OVERALL ({overall_n} eps, {overall_steps} steps, W={overall_wins} L={overall_losses} T={overall_ties}, WR={overall_wins/max(overall_n,1):.2f}):"
        f"\n    alpha fires:    {overall_alpha}/{overall_steps}  ({overall_alpha/max(overall_steps,1):.1%})"
        f"\n    beta  fires:    {overall_beta}/{overall_steps}  ({overall_beta/max(overall_steps,1):.1%})"
    )
    print()
    print("=" * 72)
    print("R1 sanity interpretation:")
    print("=" * 72)
    print("  - OVERALL alpha ~10-25%, beta ~3-12% (target band for window-based design)")
    print("  - OUTCOME-CONDITIONAL: alpha/beta fires_in_losses >> fires_in_wins is GOOD")
    print("    (means trigger correctly catches losing episodes, not noise)")
    print("  - Blue vs orange asymmetry can be sample noise OR genuine; check WR per side")
    print("    (if blue WR ≈ orange WR but fire rates very different: trigger has side bias)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
