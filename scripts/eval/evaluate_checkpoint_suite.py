#!/usr/bin/env python
"""Run a small evaluation suite across multiple checkpoints and opponents."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H100_PYTHON = Path.home() / ".venvs" / "soccertwos_h100" / "bin" / "python"
SUMMARY_PATTERN = re.compile(
    r"^(team0_module|team1_module|episodes|team0_wins|team1_wins|ties|team0_win_rate):\s*(.+)$"
)
DEFAULT_OPPONENTS = (
    ("random", "example_player_agent.agent_random"),
    ("baseline", "ceia_baseline_agent"),
)


def _default_python() -> str:
    if DEFAULT_H100_PYTHON.exists():
        return str(DEFAULT_H100_PYTHON)
    return sys.executable


def _parse_iterations(raw: str) -> List[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("No checkpoint iterations provided.")
    return values


def _resolve_trial_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / "progress.csv").exists():
        return path

    trials = sorted(
        child for child in path.iterdir() if child.is_dir() and child.name.startswith("PPO_Soccer_")
    )
    if len(trials) == 1:
        return trials[0]
    if not trials:
        raise FileNotFoundError(f"No PPO_Soccer_* trial directory found under: {path}")
    raise ValueError(f"Multiple trial directories found under {path}; pass the exact one you want.")


def _resolve_checkpoint_file(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = sorted(child for child in path.iterdir() if child.name.startswith("checkpoint-"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* file found in directory: {path}")
    return candidates[0]


def _checkpoint_dirs_from_run(run_dir: Path, iterations: Iterable[int]) -> List[Path]:
    trial_dir = _resolve_trial_dir(run_dir)
    checkpoints = []
    for iteration in iterations:
        checkpoint_dir = trial_dir / f"checkpoint_{int(iteration):06d}"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
        checkpoints.append(checkpoint_dir)
    return checkpoints


def _parse_summary(lines: List[str]) -> dict:
    summary = {}
    in_summary = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---- Summary ----":
            in_summary = True
            continue
        if not in_summary:
            continue

        match = SUMMARY_PATTERN.match(stripped)
        if match:
            summary[match.group(1)] = match.group(2)
    return summary


def _run_one(
    *,
    python_bin: str,
    checkpoint_path: Path,
    team0_module: str,
    opponent_name: str,
    opponent_module: str,
    episodes: int,
    base_port: int,
) -> dict:
    checkpoint_file = _resolve_checkpoint_file(checkpoint_path)
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"
    env["TRAINED_RAY_CHECKPOINT"] = str(checkpoint_file)
    env.setdefault("RAY_DISABLE_DASHBOARD", "1")
    env.setdefault("RAY_USAGE_STATS_ENABLED", "0")

    cmd = [
        python_bin,
        "-m",
        "cs8803drl.evaluation.evaluate_matches",
        "-m1",
        team0_module,
        "-m2",
        opponent_module,
        "-n",
        str(episodes),
        "--base_port",
        str(base_port),
    ]

    print("")
    print(f"=== Evaluating {checkpoint_path.name} vs {opponent_name} ===")
    print(f"checkpoint: {checkpoint_file}")
    print(f"command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)

    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Evaluation failed for {checkpoint_path.name} vs {opponent_name} with exit code {return_code}."
        )

    summary = _parse_summary(lines)
    if not summary:
        raise RuntimeError(f"Could not parse summary for {checkpoint_path.name} vs {opponent_name}.")

    summary["checkpoint"] = str(checkpoint_file)
    summary["checkpoint_name"] = checkpoint_path.name
    summary["opponent"] = opponent_name
    return summary


def _print_recap(results: List[dict]) -> None:
    print("")
    print("=== Suite Recap ===")
    for result in results:
        wins = result.get("team0_wins", "?")
        losses = result.get("team1_wins", "?")
        ties = result.get("ties", "?")
        win_rate = result.get("team0_win_rate", "?")
        print(
            f"{result['checkpoint_name']} vs {result['opponent']}: "
            f"win_rate={win_rate} ({wins}W-{losses}L-{ties}T)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        help="Run directory containing a single PPO_Soccer_* trial, or the trial directory itself.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Explicit checkpoint directory or checkpoint file. Repeat to evaluate multiple checkpoints.",
    )
    parser.add_argument(
        "--iterations",
        default="160,188",
        help="Comma-separated checkpoint iterations when using --run-dir. Default: 160,188",
    )
    parser.add_argument("-n", "--episodes", type=int, default=20, help="Episodes per matchup.")
    parser.add_argument(
        "--base-port",
        type=int,
        default=63105,
        help="Base port for the first matchup. Later runs add offsets automatically.",
    )
    parser.add_argument(
        "--python-bin",
        default=_default_python(),
        help="Python interpreter used to run cs8803drl.evaluation.evaluate_matches.",
    )
    parser.add_argument(
        "--team0-module",
        default="cs8803drl.deployment.trained_ray_agent",
        help="Python module for the evaluated team0 agent. Default: cs8803drl.deployment.trained_ray_agent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoints: List[Path] = []
    if args.run_dir:
        checkpoints.extend(_checkpoint_dirs_from_run(Path(args.run_dir), _parse_iterations(args.iterations)))
    checkpoints.extend(_resolve_checkpoint_file(Path(path)) for path in args.checkpoint)

    if not checkpoints:
        raise ValueError("Pass either --run-dir or at least one --checkpoint.")

    results = []
    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        for opponent_index, (opponent_name, opponent_module) in enumerate(DEFAULT_OPPONENTS):
            port = int(args.base_port) + checkpoint_index * 100 + opponent_index * 10
            results.append(
                _run_one(
                    python_bin=args.python_bin,
                    checkpoint_path=checkpoint_path,
                    team0_module=args.team0_module,
                    opponent_name=opponent_name,
                    opponent_module=opponent_module,
                    episodes=int(args.episodes),
                    base_port=port,
                )
            )

    _print_recap(results)


if __name__ == "__main__":
    main()
