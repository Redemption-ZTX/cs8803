#!/usr/bin/env python3
"""Dump full per-step trajectories from N episodes for snapshot-036
learned reward shaping (Path C).

For each episode, persist:
  - obs_a0 (T, 336), obs_a1 (T, 336)
  - act_a0 (T,), act_a1 (T,)
  - meta json: outcome, primary_label, labels (multi-label), metrics, etc.

Filename pattern:  ``{prefix_}ep{idx:05d}_{outcome}_{primary_label}.npz``

Example::

    python scripts/eval/dump_trajectories.py \\
        --team0-module cs8803drl.deployment.trained_shared_cc_agent \\
        --team1-module ceia_baseline_agent \\
        --episodes 100 \\
        --save-dir docs/experiments/artifacts/trajectories/036_029B_vs_baseline \\
        --filename-prefix 029B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# allow running as a script from anywhere
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if not hasattr(np, "bool"):
    np.bool = bool

from cs8803drl.core.utils import RewardShapingWrapper  # noqa: E402
from cs8803drl.evaluation.evaluate_matches import (  # noqa: E402
    _team_local_obs,
    _team_reward,
    _remap_team_actions,
    adapt_actions_to_env,
    episode_done,
    load_agent,
    make_env_with_retry,
)
from cs8803drl.evaluation.failure_cases import infer_team0_outcome  # noqa: E402
from cs8803drl.imitation.trajectory_dumper import (  # noqa: E402
    TrajectoryRecorder,
    save_trajectory,
)


# action coercion lives in TrajectoryRecorder._action_to_multidiscrete; here we
# just pass the raw agent output through.


def _local_action_raw(act_dict: Dict[int, Any], local_id: int) -> Any:
    """Get the (pre-adapt) raw action for one team-local slot.

    Returned object can be int (Discrete) or array-like (MultiDiscrete).
    TrajectoryRecorder normalizes to MultiDiscrete([3,3,3]) internally.
    """
    if local_id not in act_dict:
        return 0
    return act_dict[local_id]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump per-step trajectories with failure-bucket labels."
    )
    parser.add_argument("--team0-module", required=True)
    parser.add_argument("--team1-module", default="ceia_baseline_agent")
    parser.add_argument("--episodes", "-n", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--base-port", type=int, default=8500)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--filename-prefix", default="")
    parser.add_argument(
        "--trace-stride",
        type=int,
        default=5,
        help="Sub-sampling stride for the failure-recorder's sampled_steps.",
    )
    parser.add_argument(
        "--trace-tail-steps",
        type=int,
        default=40,
        help="Tail buffer length for failure-recorder metrics.",
    )
    parser.add_argument(
        "--trained-checkpoint",
        default="",
        help="If provided, set TRAINED_RAY_CHECKPOINT env var so the deployment "
        "wrapper loads this checkpoint.",
    )
    parser.add_argument(
        "--reward-shaping-debug",
        action="store_true",
        default=True,
        help="Wrap env with RewardShapingWrapper(debug_info=True) so "
        "possessing_team / progress_toward_goal are available for the "
        "failure-bucket classifier. Default: on (needed for stratified labels).",
    )
    parser.add_argument(
        "--no-reward-shaping-debug",
        action="store_false",
        dest="reward_shaping_debug",
        help="Disable the RewardShapingWrapper injection (failure metrics then become None).",
    )
    args = parser.parse_args()

    if args.trained_checkpoint:
        os.environ["TRAINED_RAY_CHECKPOINT"] = args.trained_checkpoint

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    env, used_port = make_env_with_retry(base_port=args.base_port)
    if used_port != args.base_port:
        print(
            f"[dump_trajectories] base_port {args.base_port} in use; "
            f"using base_port={used_port}"
        )

    if args.reward_shaping_debug:
        # CRITICAL: explicit zero-shaping wrapper.
        # We need debug_info=True for failure-bucket metrics (possessing_team,
        # progress_toward_goal), but we must NOT apply v2 shaping rewards on top
        # of the env reward — otherwise dumped per-step / cumulative rewards
        # would contain the v2 shaping bonus which would be misleading for any
        # downstream consumer expecting raw env reward.
        env = RewardShapingWrapper(
            env,
            time_penalty=0.0,
            ball_progress_scale=0.0,
            possession_bonus=0.0,
            opponent_progress_penalty_scale=0.0,
            deep_zone_outer_penalty=0.0,
            deep_zone_inner_penalty=0.0,
            defensive_survival_bonus=0.0,
            fast_loss_penalty_per_step=0.0,
            goal_proximity_scale=0.0,
            event_shot_reward=0.0,
            event_tackle_reward=0.0,
            event_clearance_reward=0.0,
            debug_info=True,
        )
        print(
            "[dump_trajectories] wrapped env with RewardShapingWrapper(all_zero, debug_info=True) — "
            "zero shaping so dumped rewards are pure sparse env reward; debug_info=True populates "
            "possessing_team / progress_toward_goal for failure-bucket classifier."
        )

    agent0 = load_agent(args.team0_module, env)
    agent1 = load_agent(args.team1_module, env)

    counts = {"team0_win": 0, "team1_win": 0, "tie": 0}
    bucket_counts: Dict[str, int] = {}
    saved_paths = []

    for ep in range(args.episodes):
        obs = env.reset()
        if ep % 2 == 0:
            team0_ids = (0, 1)
            team1_ids = (2, 3)
        else:
            team0_ids = (2, 3)
            team1_ids = (0, 1)

        recorder = TrajectoryRecorder(
            trace_stride=int(args.trace_stride),
            tail_steps=int(args.trace_tail_steps),
        )
        team0_reward_total = 0.0
        team1_reward_total = 0.0
        final_team0_step_reward = 0.0
        final_team1_step_reward = 0.0
        total_steps = 0
        info: Any = {}

        for step_idx in range(args.max_steps):
            obs0_local = _team_local_obs(obs, team0_ids)
            obs1_local = _team_local_obs(obs, team1_ids)

            pre_obs_a0 = obs0_local[0]
            pre_obs_a1 = obs0_local[1]

            act0_dict = agent0.act(obs0_local)
            act1_dict = agent1.act(obs1_local)

            pre_act_a0 = _local_action_raw(act0_dict, 0)
            pre_act_a1 = _local_action_raw(act0_dict, 1)

            act0 = _remap_team_actions(act0_dict, team0_ids)
            act1 = _remap_team_actions(act1_dict, team1_ids)
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

            recorder.record_step(
                step_index=step_idx,
                obs_a0=pre_obs_a0,
                obs_a1=pre_obs_a1,
                act_a0=pre_act_a0,
                act_a1=pre_act_a1,
                reward=reward,
                info=info,
                team0_ids=team0_ids,
                team1_ids=team1_ids,
            )

            total_steps = step_idx + 1
            if episode_done(done):
                break

        outcome, final_score, final_winner = infer_team0_outcome(
            final_info=info if isinstance(info, dict) else {},
            final_team0_step_reward=final_team0_step_reward,
            final_team1_step_reward=final_team1_step_reward,
            cumulative_team0_reward=team0_reward_total,
            cumulative_team1_reward=team1_reward_total,
        )
        counts[outcome] += 1

        record = recorder.build_trajectory(
            episode_index=ep,
            team0_module=args.team0_module,
            team1_module=args.team1_module,
            outcome=outcome,
            final_score=final_score,
            final_winner=final_winner,
            cumulative_team0_reward=team0_reward_total,
            cumulative_team1_reward=team1_reward_total,
            final_team0_step_reward=final_team0_step_reward,
            final_team1_step_reward=final_team1_step_reward,
            total_steps=total_steps,
        )
        record["_recording_config"] = {
            "shaping_disabled": True,
            "debug_info_only": bool(args.reward_shaping_debug),
            "trained_checkpoint": args.trained_checkpoint or None,
            "action_encoding": "MultiDiscrete([3,3,3]) stored as (T, 3) int8",
            "reward_semantics": "cumulative_team*_reward is pure sparse env reward (shaping=0)",
        }
        primary = record.get("primary_label") or "unknown"
        bucket_counts[primary] = bucket_counts.get(primary, 0) + 1

        npz_path = save_trajectory(record, save_dir, args.filename_prefix)
        saved_paths.append(npz_path)

        print(
            f"Ep {ep:3d}: {outcome:<10s} primary={primary:<28s} "
            f"steps={total_steps:3d}  -> {npz_path.name}"
        )

    print()
    print("---- Summary ----")
    print(f"team0_module:  {args.team0_module}")
    print(f"team1_module:  {args.team1_module}")
    print(f"episodes:      {args.episodes}")
    print(f"team0_wins:    {counts['team0_win']}")
    print(f"team1_wins:    {counts['team1_win']}")
    print(f"ties:          {counts['tie']}")
    win_rate = counts["team0_win"] / max(args.episodes, 1)
    print(f"team0_win_rate:{win_rate:.3f}")
    print(f"primary_label distribution:")
    for label, cnt in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<32s} {cnt:4d} ({cnt/args.episodes:.1%})")
    print(f"saved {len(saved_paths)} files to {save_dir}")


if __name__ == "__main__":
    main()
