#!/usr/bin/env python
"""Coarse-to-fine checkpoint scan using the official soccer_twos evaluator."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.evaluate_official_suite import (  # noqa: E402
    _default_python,
    _normalize_opponents,
    _run_one,
    _safe_base_port,
)


def _iter_checkpoints(run_dir: Path):
    for path in sorted(run_dir.rglob("checkpoint_*")):
        if not path.is_dir():
            continue
        try:
            iteration = int(path.name.split("_", 1)[1])
        except Exception:
            continue
        yield iteration, path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run dir to scan. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--team0-module",
        default="cs8803drl.deployment.trained_ray_agent",
        help="Team0 agent module. Default: cs8803drl.deployment.trained_ray_agent",
    )
    parser.add_argument(
        "--opponents",
        default="baseline",
        help="Comma-separated opponents. Default: baseline",
    )
    parser.add_argument("-n", "--episodes", type=int, default=20)
    parser.add_argument("--step", type=int, default=10, help="Only evaluate checkpoints where iteration %% step == 0.")
    parser.add_argument("--start", type=int, default=None, help="Minimum checkpoint iteration to include.")
    parser.add_argument("--end", type=int, default=None, help="Maximum checkpoint iteration to include.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top rows to print in ranking.")
    parser.add_argument("--base-port", type=int, default=65105)
    parser.add_argument("--python-bin", default=_default_python())
    parser.add_argument("--save-csv", default="", help="Optional CSV path for all scan results.")
    parser.add_argument("--save-logs-dir", default="", help="Optional directory to save raw evaluator logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if int(args.episodes) < 2:
        raise SystemExit(
            "Official soccer_twos.evaluate requires at least 2 episodes because its summary "
            "expects both blue and orange side stats."
        )
    opponents = _normalize_opponents(args.opponents)
    save_logs_dir = Path(args.save_logs_dir).resolve() if args.save_logs_dir else None
    if save_logs_dir is not None:
        save_logs_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    seen = set()
    for raw_run_dir in args.run_dir:
        run_dir = Path(raw_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        for iteration, checkpoint_dir in _iter_checkpoints(run_dir):
            if args.start is not None and iteration < int(args.start):
                continue
            if args.end is not None and iteration > int(args.end):
                continue
            if int(args.step) > 0 and iteration % int(args.step) != 0:
                continue
            key = str(checkpoint_dir)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((iteration, checkpoint_dir))

    candidates.sort(key=lambda item: (item[0], str(item[1])))
    if not candidates:
        raise RuntimeError("No checkpoints matched the requested scan window.")

    port_index = 0
    results = []
    for iteration, checkpoint_dir in candidates:
        for opponent_label, opponent_module in opponents:
            current_port = _safe_base_port(int(args.base_port), port_index)
            port_index += 1
            print(f"=== Official Scan: it {iteration} vs {opponent_label} ===")
            checkpoint_file, metrics, output = _run_one(
                checkpoint=str(checkpoint_dir),
                team0_module=args.team0_module,
                opponent_label=opponent_label,
                opponent_module=opponent_module,
                episodes=int(args.episodes),
                base_port=current_port,
                python_bin=args.python_bin,
            )
            row = {
                "iteration": iteration,
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_file": checkpoint_file,
                "opponent": opponent_label,
                "episodes": int(args.episodes),
                "wins": int(metrics["wins"]),
                "losses": int(metrics["losses"]),
                "draws": int(metrics["draws"]),
                "win_rate": float(metrics["win_rate"]),
            }
            results.append(row)
            print(
                f"win_rate={row['win_rate']:.3f} "
                f"({row['wins']}W-{row['losses']}L-{row['draws']}T)"
            )
            print("")

            if save_logs_dir is not None:
                safe_name = f"it{iteration:04d}_vs_{opponent_label}.log"
                (save_logs_dir / safe_name).write_text(output)

    if args.save_csv:
        csv_path = Path(args.save_csv).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "iteration",
                    "checkpoint_dir",
                    "checkpoint_file",
                    "opponent",
                    "episodes",
                    "wins",
                    "losses",
                    "draws",
                    "win_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

    print("=== Official Scan Ranking ===")
    ranked = sorted(results, key=lambda row: (-row["win_rate"], row["iteration"], row["checkpoint_file"]))
    for row in ranked[: max(int(args.top_k), 1)]:
        print(
            f"it {row['iteration']:>4} | {row['opponent']:<8} | "
            f"win_rate={row['win_rate']:.3f} "
            f"({row['wins']}W-{row['losses']}L-{row['draws']}T) | "
            f"{row['checkpoint_file']}"
        )


if __name__ == "__main__":
    main()
