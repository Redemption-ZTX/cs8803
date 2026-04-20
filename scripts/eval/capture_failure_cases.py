#!/usr/bin/env python
"""Run a checkpoint-vs-opponent evaluation and persist structured failure traces."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H100_PYTHON = Path.home() / ".venvs" / "soccertwos_h100" / "bin" / "python"


def _default_python() -> str:
    if DEFAULT_H100_PYTHON.exists():
        return str(DEFAULT_H100_PYTHON)
    return sys.executable


def _resolve_opponent(name: str) -> str:
    mapping = {
        "baseline": "ceia_baseline_agent",
        "random": "example_player_agent.agent_random",
    }
    return mapping.get(name, name)


def _parse_bool_arg(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--team0-module",
        default="cs8803drl.deployment.trained_team_ray_agent",
    )
    parser.add_argument("--opponent", default="baseline")
    parser.add_argument("-n", "--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--base-port", type=int, default=61205)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--save-mode",
        choices=("losses", "nonwins", "all", "wins", "fast_wins", "ties"),
        default="losses",
    )
    parser.add_argument(
        "--fast-win-threshold",
        type=int,
        default=100,
        help=(
            "Episode step threshold below which a team0 win counts as a 'fast' win. "
            "Forwarded to evaluate_matches.py; required when --save-mode=fast_wins."
        ),
    )
    parser.add_argument("--max-saved-episodes", type=int, default=50)
    parser.add_argument("--trace-stride", type=int, default=5)
    parser.add_argument("--trace-tail-steps", type=int, default=40)
    parser.add_argument("--reward-shaping-debug", action="store_true")
    parser.add_argument("--time-penalty", type=float, default=0.001)
    parser.add_argument("--ball-progress-scale", type=float, default=0.01)
    parser.add_argument("--goal-proximity-scale", type=float, default=0.0)
    parser.add_argument("--goal-proximity-gamma", type=float, default=0.99)
    parser.add_argument("--goal-center-x", type=float, default=15.0)
    parser.add_argument("--goal-center-y", type=float, default=0.0)
    parser.add_argument("--opponent-progress-penalty-scale", type=float, default=0.0)
    parser.add_argument("--possession-dist", type=float, default=1.25)
    parser.add_argument("--possession-bonus", type=float, default=0.002)
    parser.add_argument(
        "--progress-requires-possession",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool_arg,
    )
    parser.add_argument("--deep-zone-outer-threshold", type=float, default=0.0)
    parser.add_argument("--deep-zone-outer-penalty", type=float, default=0.0)
    parser.add_argument("--deep-zone-inner-threshold", type=float, default=0.0)
    parser.add_argument("--deep-zone-inner-penalty", type=float, default=0.0)
    parser.add_argument("--defensive-survival-threshold", type=float, default=0.0)
    parser.add_argument("--defensive-survival-bonus", type=float, default=0.0)
    parser.add_argument("--fast-loss-threshold-steps", type=int, default=0)
    parser.add_argument("--fast-loss-penalty-per-step", type=float, default=0.0)
    parser.add_argument("--event-shot-reward", type=float, default=0.0)
    parser.add_argument("--event-tackle-reward", type=float, default=0.0)
    parser.add_argument("--event-clearance-reward", type=float, default=0.0)
    parser.add_argument("--event-cooldown-steps", type=int, default=10)
    parser.add_argument("--python-bin", default=_default_python())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"
    if args.checkpoint:
        env["TRAINED_RAY_CHECKPOINT"] = str(Path(args.checkpoint).resolve())

    cmd = [
        args.python_bin,
        "-m",
        "cs8803drl.evaluation.evaluate_matches",
        "-m1",
        args.team0_module,
        "-m2",
        _resolve_opponent(args.opponent),
        "-n",
        str(int(args.episodes)),
        "--max_steps",
        str(int(args.max_steps)),
        "--base_port",
        str(int(args.base_port)),
        "--save-episodes-dir",
        str(Path(args.save_dir).resolve()),
        "--save-mode",
        args.save_mode,
        "--max-saved-episodes",
        str(int(args.max_saved_episodes)),
        "--trace-stride",
        str(int(args.trace_stride)),
        "--trace-tail-steps",
        str(int(args.trace_tail_steps)),
        "--time-penalty",
        str(float(args.time_penalty)),
        "--ball-progress-scale",
        str(float(args.ball_progress_scale)),
        "--goal-proximity-scale",
        str(float(args.goal_proximity_scale)),
        "--goal-proximity-gamma",
        str(float(args.goal_proximity_gamma)),
        "--goal-center-x",
        str(float(args.goal_center_x)),
        "--goal-center-y",
        str(float(args.goal_center_y)),
        "--opponent-progress-penalty-scale",
        str(float(args.opponent_progress_penalty_scale)),
        "--possession-dist",
        str(float(args.possession_dist)),
        "--possession-bonus",
        str(float(args.possession_bonus)),
    ]
    if args.progress_requires_possession:
        cmd.append("--progress-requires-possession")
    cmd += [
        "--deep-zone-outer-threshold",
        str(float(args.deep_zone_outer_threshold)),
        "--deep-zone-outer-penalty",
        str(float(args.deep_zone_outer_penalty)),
        "--deep-zone-inner-threshold",
        str(float(args.deep_zone_inner_threshold)),
        "--deep-zone-inner-penalty",
        str(float(args.deep_zone_inner_penalty)),
        "--defensive-survival-threshold",
        str(float(args.defensive_survival_threshold)),
        "--defensive-survival-bonus",
        str(float(args.defensive_survival_bonus)),
        "--fast-loss-threshold-steps",
        str(int(args.fast_loss_threshold_steps)),
        "--fast-loss-penalty-per-step",
        str(float(args.fast_loss_penalty_per_step)),
        "--event-shot-reward",
        str(float(args.event_shot_reward)),
        "--event-tackle-reward",
        str(float(args.event_tackle_reward)),
        "--event-clearance-reward",
        str(float(args.event_clearance_reward)),
        "--event-cooldown-steps",
        str(int(args.event_cooldown_steps)),
        "--fast-win-threshold",
        str(int(args.fast_win_threshold)),
    ]
    if args.reward_shaping_debug:
        cmd.append("--reward-shaping-debug")

    raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT), env=env))


if __name__ == "__main__":
    main()
