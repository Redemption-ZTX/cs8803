#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cs8803drl.core.checkpoint_utils import resolve_checkpoint_file
from cs8803drl.core.training_plots import (
    summarize_training_progress,
    write_training_loss_curve_png,
    _read_csv_rows_lossy,
    _filter_monotonic_progress_rows,
)


def _row_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    return _read_csv_rows_lossy(csv_path)


def _trial_progress_segments(run_root: Path) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    for progress_csv in sorted(run_root.glob("*/progress.csv")):
        rows = _filter_monotonic_progress_rows(_load_csv_rows(progress_csv))
        if not rows:
            continue
        first_row = rows[0]
        last_row = rows[-1]
        segments.append(
            {
                "trial_dir": str(progress_csv.parent),
                "progress_csv": str(progress_csv),
                "first_iteration": _row_int(first_row, "training_iteration", 0),
                "last_iteration": _row_int(last_row, "training_iteration", 0),
                "first_timesteps": _row_int(first_row, "timesteps_total", 0),
                "last_timesteps": _row_int(last_row, "timesteps_total", 0),
            }
        )
    segments.sort(key=lambda seg: (int(seg["first_iteration"]), int(seg["last_iteration"])))
    return segments


def _checkpoint_iteration(checkpoint_dir: Path) -> int:
    name = checkpoint_dir.name
    if not name.startswith("checkpoint_"):
        return -1
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return -1


def _find_checkpoint_dir(run_root: Path, iteration: int) -> Optional[Path]:
    matches = [
        path
        for path in run_root.glob("*/checkpoint_*")
        if path.is_dir() and _checkpoint_iteration(path) == iteration
    ]
    if not matches:
        return None
    return sorted(matches)[-1]


def _find_latest_checkpoint(run_root: Path) -> Optional[Path]:
    candidates = [path for path in run_root.glob("*/checkpoint_*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_checkpoint_iteration(path), str(path)))


def _select_best_eval(eval_rows: List[Dict[str, str]]) -> Optional[Dict[str, object]]:
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in eval_rows:
        if (row.get("status") or "").strip() not in ("", "ok"):
            continue
        checkpoint_dir = (row.get("checkpoint_dir") or "").strip()
        opponent = (row.get("opponent") or "").strip()
        if not checkpoint_dir or not opponent:
            continue
        grouped.setdefault(checkpoint_dir, {})[opponent] = row

    best: Optional[Dict[str, object]] = None
    best_key: Optional[Tuple[float, float, int]] = None
    for checkpoint_dir, result_map in grouped.items():
        baseline = result_map.get("baseline")
        if baseline is None:
            continue
        random_row = result_map.get("random")
        key = (
            _row_float(baseline, "win_rate", 0.0),
            _row_float(random_row, "win_rate", -1.0) if random_row else -1.0,
            _row_int(baseline, "checkpoint_iteration", -1),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_file": baseline.get("checkpoint_file") or checkpoint_dir,
                "checkpoint_iteration": _row_int(baseline, "checkpoint_iteration", -1),
                "baseline": baseline,
                "random": random_row,
            }
    return best


def _build_summary_lines(run_root: Path) -> List[str]:
    progress_summary = summarize_training_progress(run_root)
    if not progress_summary:
        raise RuntimeError(f"No readable progress rows found under {run_root}")

    best_row = progress_summary.get("best_row")
    final_row = progress_summary.get("final_row")
    progress_csvs = progress_summary.get("progress_csvs") or []
    segments = _trial_progress_segments(run_root)

    eval_csv = run_root / "checkpoint_eval.csv"
    eval_rows = _load_csv_rows(eval_csv) if eval_csv.exists() else []
    best_eval = _select_best_eval(eval_rows)

    best_reward_iteration = _row_int(best_row, "training_iteration", 0) if isinstance(best_row, dict) else 0
    best_reward_checkpoint_dir = _find_checkpoint_dir(run_root, best_reward_iteration) if best_reward_iteration > 0 else None
    latest_checkpoint_dir = _find_latest_checkpoint(run_root)
    merged_curve_path = write_training_loss_curve_png(run_root, output_name="training_loss_curve_merged.png")

    lines = [
        "Merged Training Summary",
        f"  run_dir:               {run_root}",
        f"  progress_csv_count:    {len(progress_csvs)}",
        f"  trial_segment_count:   {len(segments)}",
    ]

    if best_row is not None:
        lines.append(
            "  best_reward_mean:      "
            f"{_row_float(best_row, 'episode_reward_mean', 0.0):+0.4f} @ iteration {best_reward_iteration}"
        )
    else:
        lines.append("  best_reward_mean:      unavailable")

    if best_reward_checkpoint_dir is not None:
        lines.append(
            "  best_reward_checkpoint: "
            f"{resolve_checkpoint_file(str(best_reward_checkpoint_dir))}"
        )
    else:
        lines.append("  best_reward_checkpoint: unavailable")

    if final_row is not None:
        lines.append(
            "  final_iteration:       "
            f"{_row_int(final_row, 'training_iteration', 0)}"
        )
        lines.append(
            "  final_timesteps:       "
            f"{_row_int(final_row, 'timesteps_total', 0):,}"
        )
        lines.append(
            "  final_reward_mean:     "
            f"{_row_float(final_row, 'episode_reward_mean', 0.0):+0.4f}"
        )
    else:
        lines.append("  final_iteration:       unavailable")
        lines.append("  final_timesteps:       unavailable")
        lines.append("  final_reward_mean:     unavailable")

    if latest_checkpoint_dir is not None:
        lines.append(f"  final_checkpoint:      {resolve_checkpoint_file(str(latest_checkpoint_dir))}")
    else:
        lines.append("  final_checkpoint:      unavailable")

    if best_eval:
        baseline = best_eval.get("baseline") or {}
        random_row = best_eval.get("random") or {}
        lines.append(f"  best_eval_checkpoint:  {best_eval.get('checkpoint_file')}")
        lines.append(
            "  best_eval_baseline:    "
            f"{_row_float(baseline, 'win_rate', 0.0):.3f} "
            f"({_row_int(baseline, 'wins', 0)}W-{_row_int(baseline, 'losses', 0)}L-{_row_int(baseline, 'ties', 0)}T) "
            f"@ iteration {_row_int(baseline, 'checkpoint_iteration', -1)}"
        )
        if random_row:
            lines.append(
                "  best_eval_random:      "
                f"{_row_float(random_row, 'win_rate', 0.0):.3f} "
                f"({_row_int(random_row, 'wins', 0)}W-{_row_int(random_row, 'losses', 0)}L-{_row_int(random_row, 'ties', 0)}T)"
            )
        lines.append(f"  eval_results_csv:      {eval_csv}")
    else:
        lines.append("  best_eval_checkpoint:  unavailable")

    if merged_curve_path:
        lines.append(f"  loss_curve_file:       {merged_curve_path}")

    if segments:
        lines.append("  segments:")
        for idx, seg in enumerate(segments, start=1):
            lines.append(
                "    "
                f"{idx}. {seg['trial_dir']} | "
                f"it {seg['first_iteration']}->{seg['last_iteration']} | "
                f"steps {int(seg['first_timesteps']):,}->{int(seg['last_timesteps']):,}"
            )

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a merged summary for a resumed Ray run root.")
    parser.add_argument("--run-dir", required=True, help="Run root under ray_results/")
    parser.add_argument(
        "--output",
        default="merged_training_summary.txt",
        help="Output filename written under the run root (default: merged_training_summary.txt)",
    )
    args = parser.parse_args()

    run_root = Path(args.run_dir).resolve()
    if not run_root.exists():
        raise SystemExit(f"Run root does not exist: {run_root}")

    lines = _build_summary_lines(run_root)
    text = "\n".join(lines) + "\n"
    print(text, end="")

    output_path = run_root / args.output
    output_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
