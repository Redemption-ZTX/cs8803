#!/usr/bin/env python
"""Wrapper around ``soccer_twos.evaluate`` with a clearer H2H recap.

This keeps the raw evaluator output intact, then appends a team0-perspective
summary that makes three things explicit:
1. the overall H2H result is the main number,
2. blue/orange splits are diagnostics only, and
3. the distance from 0.500 is often more informative than the raw win rate.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H100_PYTHON = Path.home() / ".venvs" / "soccertwos_h100" / "bin" / "python"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def _default_python() -> str:
    if DEFAULT_H100_PYTHON.exists():
        return str(DEFAULT_H100_PYTHON)
    return sys.executable


def _parse_scalar(raw: str):
    text = raw.strip()
    try:
        value = float(text)
    except ValueError:
        return text
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def _extract_policy_sections(output_text: str):
    lines = output_text.replace("\r", "\n").splitlines()
    in_policies = False
    current_policy = None
    current_section = None
    policies = {}

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if not in_policies:
            if stripped == "policies:":
                in_policies = True
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent == 2 and stripped.endswith(":"):
            current_policy = stripped[:-1]
            current_section = None
            policies.setdefault(
                current_policy,
                {
                    "overall": {},
                    "blue_team": {},
                    "orange_team": {},
                },
            )
            continue

        if current_policy is None:
            continue

        if indent == 4 and stripped in {"blue_team:", "orange_team:"}:
            current_section = stripped[:-1]
            continue

        if indent == 4 and stripped.startswith("policy_"):
            key, value = stripped.split(":", 1)
            policies[current_policy]["overall"][key] = _parse_scalar(value)
            current_section = None
            continue

        if indent == 6 and current_section and stripped.startswith("policy_"):
            key, value = stripped.split(":", 1)
            policies[current_policy][current_section][key] = _parse_scalar(value)
            continue

        if indent == 0:
            break

    return policies


def _get_int(mapping: dict, key: str) -> int:
    return int(mapping.get(key, 0))


def _get_float(mapping: dict, key: str) -> float:
    return float(mapping.get(key, 0.0))


def _print_team_block(prefix: str, stats: dict) -> None:
    wins = _get_int(stats, "policy_wins")
    losses = _get_int(stats, "policy_losses")
    draws = _get_int(stats, "policy_draws")
    total = _get_int(stats, "policy_total_games")
    win_rate = _get_float(stats, "policy_win_rate")
    edge_vs_even = win_rate - 0.5

    print(f"{prefix}_overall_record: {wins}W-{losses}L-{draws}T")
    print(f"{prefix}_overall_games: {total}")
    print(f"{prefix}_overall_win_rate: {win_rate:.3f}")
    print(f"{prefix}_edge_vs_even: {edge_vs_even:+.3f}")
    print(f"{prefix}_net_wins_minus_losses: {wins - losses:+d}")


def _print_side_block(side_name: str, stats: dict) -> None:
    prefix = f"policy_{side_name}_team"
    wins = _get_int(stats, f"{prefix}_wins")
    losses = _get_int(stats, f"{prefix}_losses")
    draws = _get_int(stats, f"{prefix}_draws")
    total = _get_int(stats, f"{prefix}_total_games")
    win_rate = _get_float(stats, f"{prefix}_win_rate")

    print(f"team0_{side_name}_record: {wins}W-{losses}L-{draws}T")
    print(f"team0_{side_name}_games: {total}")
    print(f"team0_{side_name}_win_rate: {win_rate:.3f}")


def _print_recap(*, output_text: str, team0_module: str, team1_module: str, episodes: int) -> None:
    policies = _extract_policy_sections(output_text)
    team0 = policies.get(team0_module)
    team1 = policies.get(team1_module)

    if not team0 or not team1:
        print("---- H2H Recap ----")
        print("recap_status: unavailable")
        print("recap_note: could not parse policy blocks from raw evaluator output")
        return

    blue = team0.get("blue_team", {})
    orange = team0.get("orange_team", {})
    side_gap = _get_float(blue, "policy_blue_team_win_rate") - _get_float(
        orange, "policy_orange_team_win_rate"
    )

    print("---- H2H Recap ----")
    print(f"team0_module: {team0_module}")
    print(f"team1_module: {team1_module}")
    print(f"episodes: {episodes}")
    _print_team_block("team0", team0["overall"])
    _print_team_block("team1", team1["overall"])
    _print_side_block("blue", blue)
    _print_side_block("orange", orange)
    print(f"team0_side_gap_blue_minus_orange: {side_gap:+.3f}")
    print("reading_note: interpret team0_overall_* as the H2H result; blue/orange_* are side-split diagnostics only.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run soccer_twos.evaluate and append a clearer team0-perspective H2H recap."
    )
    parser.add_argument("-m1", "--team0-module", required=True)
    parser.add_argument("-m2", "--team1-module", required=True)
    parser.add_argument("-e", "--episodes", type=int, default=500)
    parser.add_argument("-p", "--base-port", type=int, default=63105)
    parser.add_argument("--python-bin", default=_default_python())
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("RAY_DISABLE_DASHBOARD", "1")
    env.setdefault("RAY_USAGE_STATS_ENABLED", "0")

    cmd = [
        args.python_bin,
        "-m",
        "soccer_twos.evaluate",
        "-m1",
        args.team0_module,
        "-m2",
        args.team1_module,
        "-e",
        str(int(args.episodes)),
        "-p",
        str(int(args.base_port)),
        *passthrough,
    ]

    collected = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        collected.append(line)
    return_code = proc.wait()
    output_text = "".join(collected)
    if return_code != 0:
        raise SystemExit(return_code)

    print("")
    _print_recap(
        output_text=output_text,
        team0_module=args.team0_module,
        team1_module=args.team1_module,
        episodes=int(args.episodes),
    )


if __name__ == "__main__":
    main()
