#!/usr/bin/env python
"""Backfill checkpoint evaluations for an existing training run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs8803drl.training.train_ray_team_vs_random_shaping import (  # noqa: E402
    _append_eval_row,
    _evaluate_checkpoint_once,
    _failed_eval_row,
    _find_checkpoint_dirs,
    _load_eval_rows,
    _parse_eval_opponents,
    _safe_eval_base_port,
    _select_best_eval,
    _write_best_eval_summary,
)


DEFAULT_H100_PYTHON = Path.home() / ".venvs" / "soccertwos_h100" / "bin" / "python"


def _default_python() -> str:
    if DEFAULT_H100_PYTHON.exists():
        return str(DEFAULT_H100_PYTHON)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. ray_results/<run_name>")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Only evaluate checkpoints whose iteration is divisible by this interval. Default: 5",
    )
    parser.add_argument("-n", "--episodes", type=int, default=20, help="Episodes per matchup.")
    parser.add_argument("--max-steps", type=int, default=1500, help="Max steps per episode.")
    parser.add_argument("--base-port", type=int, default=64005, help="Base port for evaluations.")
    parser.add_argument(
        "--min-iteration",
        type=int,
        default=0,
        help="Only evaluate checkpoints with iteration >= this value. Default: 0",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=0,
        help="Only evaluate checkpoints with iteration <= this value. Default: disabled",
    )
    parser.add_argument(
        "--opponents",
        default="baseline",
        help="Comma-separated opponents: baseline,random or explicit module names. Default: baseline",
    )
    parser.add_argument(
        "--team0-module",
        default="cs8803drl.deployment.trained_role_agent",
        help="Team0 agent module used for evaluation. Default: cs8803drl.deployment.trained_role_agent",
    )
    parser.add_argument(
        "--python-bin",
        default=_default_python(),
        help="Python interpreter used to run -m cs8803drl.evaluation.evaluate_matches.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry rows previously recorded with status=failed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    eval_csv = run_dir / "checkpoint_eval.csv"
    eval_log_dir = run_dir / "checkpoint_eval_logs"
    rows = _load_eval_rows(eval_csv)
    seen_ok = {
        (row.get("checkpoint_dir"), row.get("opponent"))
        for row in rows
        if row.get("checkpoint_dir") and row.get("opponent") and row.get("status") == "ok"
    }
    seen_failed = {
        (row.get("checkpoint_dir"), row.get("opponent"))
        for row in rows
        if row.get("checkpoint_dir") and row.get("opponent") and row.get("status") == "failed"
    }
    seen = seen_ok | (set() if args.retry_failed else seen_failed)

    opponents = _parse_eval_opponents(args.opponents)
    checkpoint_dirs = _find_checkpoint_dirs(run_dir)
    pending = []
    for checkpoint_dir in checkpoint_dirs:
        iteration = int(checkpoint_dir.name.split("_", 1)[1])
        if iteration < int(args.min_iteration):
            continue
        if int(args.max_iteration) > 0 and iteration > int(args.max_iteration):
            continue
        if args.eval_interval > 0 and iteration % int(args.eval_interval) != 0:
            continue
        for opponent_name, opponent_module in opponents:
            key = (str(checkpoint_dir), opponent_name)
            if key not in seen:
                pending.append((checkpoint_dir, iteration, opponent_name, opponent_module))

    if not pending:
        print("No pending checkpoint evaluations.")
    else:
        next_port_offset = 0
        for checkpoint_dir, iteration, opponent_name, opponent_module in pending:
            checkpoint_name = checkpoint_dir.name
            log_path = eval_log_dir / f"{checkpoint_name}_{opponent_name}.log"
            try:
                result = _evaluate_checkpoint_once(
                    checkpoint_dir=checkpoint_dir,
                    team0_module=args.team0_module,
                    opponent_name=opponent_name,
                    opponent_module=opponent_module,
                    episodes=int(args.episodes),
                    max_steps=int(args.max_steps),
                    base_port=_safe_eval_base_port(int(args.base_port), next_port_offset),
                    python_bin=args.python_bin,
                    log_dir=eval_log_dir,
                )
                next_port_offset += 1
                result["timestamp"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
                _append_eval_row(eval_csv, result)
                rows.append(result)
                print(
                    f"[backfill-eval] it {iteration} | {opponent_name} "
                    f"{result['wins']}W-{result['losses']}L-{result['ties']}T "
                    f"(win_rate={result['win_rate']:.3f})"
                )
            except Exception as exc:
                failed_row = _failed_eval_row(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_iteration=iteration,
                    opponent_name=opponent_name,
                    episodes=int(args.episodes),
                    log_path=log_path,
                )
                _append_eval_row(eval_csv, failed_row)
                rows.append(failed_row)
                print(f"[backfill-eval] it {iteration} | {opponent_name} FAILED: {exc}")

    best_eval = _select_best_eval(rows)
    if best_eval:
        _write_best_eval_summary(run_dir, best_eval)
        baseline = best_eval.get("baseline") or {}
        print("")
        print("Best Eval Checkpoint")
        print(f"checkpoint: {best_eval.get('checkpoint_file') or best_eval.get('checkpoint_dir')}")
        print(
            "baseline: "
            f"{float(baseline.get('win_rate', 0.0)):.3f} "
            f"({baseline.get('wins', '?')}W-{baseline.get('losses', '?')}L-{baseline.get('ties', '?')}T)"
        )
        print(f"iteration: {best_eval.get('checkpoint_iteration', '?')}")


if __name__ == "__main__":
    main()
