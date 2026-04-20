#!/usr/bin/env python3
"""Pick checkpoints for 1000ep eval per the project doctrine.

Doctrine (per docs/architecture/engineering-standards.md "Checkpoint 选模规则"
and SNAPSHOT-027):
    top 5% by training-internal baseline WR + ties at the 5% cutoff + ±1
    ckpt window around each selected ckpt.

This guards against late-window noise spikes (the "ckpt-830 = 0.900 internal
but 0.758 official" trap from snapshot-027) by widening the candidate pool
beyond a single argmax.

Usage:
    python scripts/eval/pick_top_ckpts.py <run_dir> [--opponent baseline]
    -> prints space-separated ckpt iterations on one line

Exits non-zero if no checkpoint_eval.csv is found.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


VALID_METRICS = ("win_rate", "non_loss_rate", "fast_win_rate")


def load_evals(run_dir: Path, opponent: str, metric: str = "win_rate"):
    """Load (iter, metric_value) tuples from checkpoint_eval.csv. Iter is int.

    `metric` selects which column to use as the ranking value; defaults to
    `win_rate` for the standard grading lane. Specialist lanes (snapshot-044)
    use `fast_win_rate` (spear) or `non_loss_rate` (shield).
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}")
    csv_path = run_dir / "checkpoint_eval.csv"
    if not csv_path.exists():
        print(f"checkpoint_eval.csv not found in {run_dir}", file=sys.stderr)
        sys.exit(1)
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("opponent", "").strip() != opponent:
                continue
            if row.get("status", "").strip() != "ok":
                continue
            try:
                it = int(row["checkpoint_iteration"])
                wr = float(row[metric])
            except (KeyError, ValueError):
                continue
            rows.append((it, wr))
    return rows


def pick(evals, top_pct: float = 5.0, neighbor_window: int = 1):
    """Return sorted list of ckpt iterations.

    1. Sort all evals by WR desc.
    2. Cutoff at the top `top_pct` percent (always >= 1 ckpt).
    3. Include all ties at the cutoff WR.
    4. For each selected ckpt, also include ckpts at ±neighbor_window steps
       in the *iteration* sequence (continuity, not WR proximity).
    """
    if not evals:
        return []
    by_wr = sorted(evals, key=lambda r: r[1], reverse=True)
    n = len(by_wr)
    cutoff_count = max(1, math.ceil(n * top_pct / 100.0))
    cutoff_wr = by_wr[cutoff_count - 1][1]

    selected_iters = {it for it, wr in by_wr if wr >= cutoff_wr}

    # Add ±N neighbors in the iter sequence
    sorted_iters = sorted(it for it, _ in evals)
    iter_set = set(sorted_iters)
    iter_to_idx = {it: i for i, it in enumerate(sorted_iters)}
    expanded = set(selected_iters)
    for it in selected_iters:
        idx = iter_to_idx[it]
        for delta in range(-neighbor_window, neighbor_window + 1):
            j = idx + delta
            if 0 <= j < len(sorted_iters):
                expanded.add(sorted_iters[j])
    return sorted(expanded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to ray_results/<run> directory")
    parser.add_argument("--opponent", default="baseline")
    parser.add_argument(
        "--metric",
        choices=VALID_METRICS,
        default="win_rate",
        help=(
            "Column from checkpoint_eval.csv to rank by. Default win_rate. "
            "Specialist lanes use fast_win_rate (spear) or non_loss_rate (shield)."
        ),
    )
    parser.add_argument("--top-pct", type=float, default=5.0)
    parser.add_argument("--window", type=int, default=1, help="±N iter neighbors")
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="If >0, cap total returned ckpts (keep highest-WR-first)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print human-readable summary instead of one-line iter list",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    evals = load_evals(run_dir, args.opponent, metric=args.metric)
    chosen = pick(evals, top_pct=args.top_pct, neighbor_window=args.window)

    if args.max and len(chosen) > args.max:
        wr_by_iter = dict(evals)
        chosen = sorted(
            chosen,
            key=lambda it: wr_by_iter.get(it, 0.0),
            reverse=True,
        )[: args.max]
        chosen.sort()

    if args.summary:
        wr_by_iter = dict(evals)
        print(
            f"opponent={args.opponent} metric={args.metric} n_evals={len(evals)} "
            f"top_pct={args.top_pct} window=±{args.window} "
            f"selected={len(chosen)}"
        )
        for it in chosen:
            print(f"  ckpt {it:5d}  wr={wr_by_iter.get(it, float('nan')):.3f}")
    else:
        print(" ".join(str(it) for it in chosen))


if __name__ == "__main__":
    main()
