"""SNAPSHOT-048 hybrid evaluator: student plays + baseline takes over on trigger.

Architecture:
  - Loads a STUDENT agent from --student-module (existing wrapper, e.g.
    cs8803drl.deployment.trained_team_ray_agent reading TRAINED_RAY_CHECKPOINT)
  - Loads the project BASELINE policy via cs8803drl.core.utils._get_baseline_policy
    (returns a Ray RLlib policy with .compute_single_action(obs))
  - Per env step: pulls ball_x from info dict, applies the trigger function;
    on fire, baseline controls BOTH team0 agents this step (one act per agent);
    otherwise student controls team0. Team1 is the project baseline opponent
    (loaded the same way as in the official evaluator).

This evaluator is used for the 6 conditions in snapshot-048 §2.3:
    --student-module ... --student-checkpoint ... --trigger {none,alpha,beta}

Output (stdout, plus optional --json-out):
    win_rate, ties, swap_pct, side-split WR, mean episode length

Usage:
    python -m cs8803drl.evaluation.evaluate_hybrid \
        --student-module cs8803drl.deployment.trained_team_ray_agent \
        --student-checkpoint <path>/checkpoint-1040 \
        --trigger alpha --episodes 1000 --base-port 50001
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from cs8803drl.deployment._hybrid.switching import (
    TriggerState,
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


def _load_baseline_callable():
    """Load the project baseline as a per-agent callable: obs_336 -> action_3."""
    from cs8803drl.core.utils import _get_baseline_policy  # noqa: WPS437

    policy = _get_baseline_policy()

    def call(obs: np.ndarray):
        action = policy.compute_single_action(obs, explore=False)
        if isinstance(action, tuple):
            action = action[0]
        return action

    return call


def _baseline_act_for_team(baseline_call, team_local_obs: Dict[int, Any]) -> Dict[int, Any]:
    """Run baseline for each local team0 player. Returns local-id action dict."""
    out: Dict[int, Any] = {}
    for local_id, obs in team_local_obs.items():
        out[local_id] = baseline_call(np.asarray(obs, dtype=np.float32))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid eval: student + teacher takeover on trigger")
    p.add_argument("--student-module", required=True, help="Module path for student wrapper, e.g. cs8803drl.deployment.trained_team_ray_agent")
    p.add_argument("--student-checkpoint", required=True, help="Path to student checkpoint (sets TRAINED_RAY_CHECKPOINT)")
    p.add_argument("--trigger", default="none", choices=["none", "alpha", "beta"])
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--base-port", type=int, default=50001)
    p.add_argument("--json-out", default=None, help="Optional path to write JSON summary")
    p.add_argument("--print-every", type=int, default=50, help="Progress print interval (episodes)")
    p.add_argument("--takeover-module", default=None, help="Optional module path for takeover teacher (e.g. cs8803drl.deployment.trained_shared_cc_agent for 036D). If unset, uses project baseline.")
    p.add_argument("--takeover-checkpoint", default=None, help="Optional checkpoint path for takeover teacher. If unset, uses project baseline.")
    return p.parse_args()


def _make_team1_baseline_agent(env):
    """Load the project baseline as the team1 opponent agent (same as official eval).

    The baseline ships as ceia_baseline_agent module; reuse load_agent path
    so the env / Ray setup mirrors evaluate_matches.
    """
    return load_agent("ceia_baseline_agent", env)


def main() -> int:
    args = _parse_args()

    # Communicate student checkpoint to the wrapper via the env var convention.
    os.environ["TRAINED_RAY_CHECKPOINT"] = args.student_checkpoint

    print(f"[setup] student_module        = {args.student_module}")
    print(f"[setup] student_checkpoint    = {args.student_checkpoint}")
    print(f"[setup] trigger               = {args.trigger}")
    print(f"[setup] episodes              = {args.episodes}")
    print(f"[setup] base_port             = {args.base_port}")
    print()

    env, used_port = make_env_with_retry(base_port=args.base_port)
    print(f"[env] connected, port={used_port}")

    student = load_agent(args.student_module, env)
    print(f"[student] loaded {type(student).__name__}")

    team1_agent = _make_team1_baseline_agent(env)
    print(f"[team1] loaded baseline opponent agent {type(team1_agent).__name__}")

    teacher_agent = None
    baseline_call = None
    if args.takeover_module and args.takeover_checkpoint:
        # Cross-student DAGGER probe: load teacher agent (full Agent class)
        # and use teacher.act(team_local_obs) for takeover.
        os.environ["TRAINED_RAY_CHECKPOINT"] = args.takeover_checkpoint
        os.environ["TRAINED_SHARED_CC_CHECKPOINT"] = args.takeover_checkpoint
        os.environ["TRAINED_TEAM_OPPONENT_CHECKPOINT"] = args.takeover_checkpoint
        teacher_agent = load_agent(args.takeover_module, env)
        print(f"[teacher-takeover] loaded {type(teacher_agent).__name__} from {args.takeover_checkpoint}")
        # Restore the student's env var so any lazy re-loads point to student
        os.environ["TRAINED_RAY_CHECKPOINT"] = args.student_checkpoint
    else:
        baseline_call = _load_baseline_callable()
        print("[baseline-takeover] project baseline policy loaded for trigger fires")
    print()

    wins0 = 0
    wins1 = 0
    ties = 0
    side_split = {
        "team0_as_blue":   {"win": 0, "loss": 0, "tie": 0},
        "team0_as_orange": {"win": 0, "loss": 0, "tie": 0},
    }
    episode_steps: List[int] = []
    swap_pcts: List[float] = []
    aggregate_swap_steps = 0
    aggregate_total_steps = 0

    started = time.time()

    for ep in range(args.episodes):
        if ep % 2 == 0:
            team0_ids = (0, 1)
            team1_ids = (2, 3)
            side_key = "team0_as_blue"
        else:
            team0_ids = (2, 3)
            team1_ids = (0, 1)
            side_key = "team0_as_orange"

        trigger_state = TriggerState(args.trigger)

        obs = env.reset()
        info: Dict[Any, Any] = {}
        final_team0_step_reward = 0.0
        final_team1_step_reward = 0.0
        team0_reward_total = 0.0
        team1_reward_total = 0.0
        total_steps = 0

        for step_idx in range(args.max_steps):
            obs0 = _team_local_obs(obs, team0_ids)
            obs1 = _team_local_obs(obs, team1_ids)

            # Decide team0 action based on trigger.
            ball_x = extract_team0_ball_x(info, team0_ids) if info else 0.0
            fired = trigger_state.step(ball_x)

            if fired:
                if teacher_agent is not None:
                    # Teacher agent (.act takes dict of local ids, returns dict of local ids)
                    act0_local = teacher_agent.act(obs0)
                else:
                    act0_local = _baseline_act_for_team(baseline_call, obs0)
            else:
                act0_local = student.act(obs0)

            act1_local = team1_agent.act(obs1)

            act0 = _remap_team_actions(act0_local, team0_ids)
            act1 = _remap_team_actions(act1_local, team1_ids)
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

        outcome, _score, _winner = infer_team0_outcome(
            final_info=info if isinstance(info, dict) else {},
            final_team0_step_reward=final_team0_step_reward,
            final_team1_step_reward=final_team1_step_reward,
            cumulative_team0_reward=team0_reward_total,
            cumulative_team1_reward=team1_reward_total,
        )

        if outcome == "team0_win":
            wins0 += 1
            side_split[side_key]["win"] += 1
        elif outcome == "team1_win":
            wins1 += 1
            side_split[side_key]["loss"] += 1
        else:
            ties += 1
            side_split[side_key]["tie"] += 1

        episode_steps.append(total_steps)
        swap_pcts.append(trigger_state.swap_pct())
        aggregate_swap_steps += trigger_state.swap_count
        aggregate_total_steps += trigger_state.total_steps

        if (ep + 1) % args.print_every == 0:
            elapsed = time.time() - started
            running_wr = wins0 / max(ep + 1, 1)
            print(
                f"[progress] ep={ep + 1}/{args.episodes}  WR={running_wr:.3f}  ties={ties}  "
                f"mean_swap_pct={statistics.mean(swap_pcts):.3f}  elapsed={elapsed:.0f}s"
            )

    n = len(episode_steps)
    win_rate = wins0 / max(n, 1)
    loss_rate = wins1 / max(n, 1)
    tie_rate = ties / max(n, 1)
    overall_swap_pct = aggregate_swap_steps / max(aggregate_total_steps, 1)

    def _side_wr(d):
        total = d["win"] + d["loss"] + d["tie"]
        return d["win"] / max(total, 1)

    summary = {
        "student_module": args.student_module,
        "student_checkpoint": args.student_checkpoint,
        "trigger": args.trigger,
        "episodes": n,
        "wins": wins0,
        "losses": wins1,
        "ties": ties,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "tie_rate": tie_rate,
        "overall_swap_pct": overall_swap_pct,
        "mean_per_episode_swap_pct": statistics.mean(swap_pcts) if swap_pcts else 0.0,
        "side_team0_as_blue":   {**side_split["team0_as_blue"],   "wr": _side_wr(side_split["team0_as_blue"])},
        "side_team0_as_orange": {**side_split["team0_as_orange"], "wr": _side_wr(side_split["team0_as_orange"])},
        "episode_steps_mean": statistics.mean(episode_steps) if episode_steps else 0.0,
        "episode_steps_median": statistics.median(episode_steps) if episode_steps else 0.0,
        "elapsed_seconds": time.time() - started,
    }

    print()
    print("=" * 72)
    print("HYBRID EVAL SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        print(f"  {k:30} = {v}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[json] wrote {args.json_out}")

    try:
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
