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
    write_training_history_png,
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


def _trial_progress_segments_many(run_roots: List[Path]) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    for run_root in run_roots:
        for seg in _trial_progress_segments(run_root):
            seg = dict(seg)
            seg["run_root"] = str(run_root)
            segments.append(seg)
    segments.sort(
        key=lambda seg: (
            int(seg["first_timesteps"]),
            int(seg["last_timesteps"]),
            int(seg["first_iteration"]),
            int(seg["last_iteration"]),
            str(seg["trial_dir"]),
        )
    )
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


def _find_checkpoint_dir_in_trial(trial_dir: Path, iteration: int) -> Optional[Path]:
    matches = [
        path
        for path in trial_dir.glob("checkpoint_*")
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


def _find_latest_checkpoint_many(run_roots: List[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    for run_root in run_roots:
        candidates.extend(path for path in run_root.glob("*/checkpoint_*") if path.is_dir())
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


def _load_progress_rows_many(run_roots: List[Path]) -> Tuple[List[Dict[str, str]], List[str]]:
    merged_rows: List[Dict[str, str]] = []
    progress_csvs: List[str] = []
    for run_root in run_roots:
        for progress_csv in sorted(run_root.glob("*/progress.csv")):
            rows = _filter_monotonic_progress_rows(_load_csv_rows(progress_csv))
            if not rows:
                continue
            progress_csvs.append(str(progress_csv))
            for row in rows:
                row_copy = dict(row)
                row_copy["_trial_dir"] = str(progress_csv.parent)
                row_copy["_progress_csv"] = str(progress_csv)
                row_copy["_run_root"] = str(run_root)
                merged_rows.append(row_copy)

    merged_rows.sort(
        key=lambda row: (
            _row_int(row, "timesteps_total", 0),
            _row_int(row, "training_iteration", 0),
            row.get("_progress_csv", ""),
        )
    )
    return merged_rows, progress_csvs


def _load_eval_rows_many(run_roots: List[Path]) -> List[Dict[str, str]]:
    eval_rows: List[Dict[str, str]] = []
    for run_root in run_roots:
        eval_csv = run_root / "checkpoint_eval.csv"
        if not eval_csv.exists():
            continue
        for row in _load_csv_rows(eval_csv):
            row_copy = dict(row)
            row_copy["_eval_csv"] = str(eval_csv)
            row_copy["_run_root"] = str(run_root)
            eval_rows.append(row_copy)
    return eval_rows


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


def _build_summary_lines_many(run_roots: List[Path], output_root: Path) -> List[str]:
    merged_rows, progress_csvs = _load_progress_rows_many(run_roots)
    if not merged_rows:
        joined = ", ".join(str(root) for root in run_roots)
        raise RuntimeError(f"No readable progress rows found under run roots: {joined}")

    best_row = max(
        merged_rows,
        key=lambda row: (
            _row_float(row, "episode_reward_mean", float("-inf")),
            _row_int(row, "training_iteration", -1),
            _row_int(row, "timesteps_total", -1),
        ),
    )
    final_row = max(
        merged_rows,
        key=lambda row: (
            _row_int(row, "timesteps_total", -1),
            _row_int(row, "training_iteration", -1),
            row.get("_progress_csv", ""),
        ),
    )
    segments = _trial_progress_segments_many(run_roots)
    eval_rows = _load_eval_rows_many(run_roots)
    best_eval = _select_best_eval(eval_rows)

    best_reward_iteration = _row_int(best_row, "training_iteration", 0)
    best_trial_dir = Path(best_row["_trial_dir"])
    best_reward_checkpoint_dir = (
        _find_checkpoint_dir_in_trial(best_trial_dir, best_reward_iteration)
        if best_reward_iteration > 0
        else None
    )
    latest_checkpoint_dir = _find_latest_checkpoint_many(run_roots)
    merged_curve_path = write_training_history_png(
        progress_rows=merged_rows,
        eval_rows=eval_rows,
        output_path=output_root / "training_loss_curve_merged.png",
        title="TRAINING LOSS CURVE (MERGED RUN ROOTS)",
    )

    lines = [
        "Merged Training Summary",
        f"  merged_run_roots:      {len(run_roots)}",
    ]
    for idx, run_root in enumerate(run_roots, start=1):
        lines.append(f"    {idx}. {run_root}")

    lines.extend(
        [
            f"  progress_csv_count:    {len(progress_csvs)}",
            f"  trial_segment_count:   {len(segments)}",
            "  best_reward_mean:      "
            f"{_row_float(best_row, 'episode_reward_mean', 0.0):+0.4f} @ iteration {best_reward_iteration}",
        ]
    )

    if best_reward_checkpoint_dir is not None:
        lines.append(
            "  best_reward_checkpoint: "
            f"{resolve_checkpoint_file(str(best_reward_checkpoint_dir))}"
        )
    else:
        lines.append("  best_reward_checkpoint: unavailable")

    lines.extend(
        [
            "  final_iteration:       "
            f"{_row_int(final_row, 'training_iteration', 0)}",
            "  final_timesteps:       "
            f"{_row_int(final_row, 'timesteps_total', 0):,}",
            "  final_reward_mean:     "
            f"{_row_float(final_row, 'episode_reward_mean', 0.0):+0.4f}",
        ]
    )

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
        lines.append("  eval_results_csvs:")
        for run_root in run_roots:
            eval_csv = run_root / "checkpoint_eval.csv"
            if eval_csv.exists():
                lines.append(f"    - {eval_csv}")
    else:
        lines.append("  best_eval_checkpoint:  unavailable")

    if merged_curve_path:
        lines.append(f"  loss_curve_file:       {merged_curve_path}")
    else:
        lines.append("  loss_curve_file:       unavailable (multi-run summary)")

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
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run root under ray_results/. Repeat to merge split resume roots.",
    )
    parser.add_argument(
        "--output",
        default="merged_training_summary.txt",
        help="Output filename written under the output run root (default: merged_training_summary.txt)",
    )
    parser.add_argument(
        "--output-run-dir",
        default="",
        help="Optional run root where the summary file should be written. Defaults to the last --run-dir.",
    )
    args = parser.parse_args()

    run_roots = [Path(path).resolve() for path in args.run_dir]
    for run_root in run_roots:
        if not run_root.exists():
            raise SystemExit(f"Run root does not exist: {run_root}")

    output_root = (
        Path(args.output_run_dir).resolve()
        if args.output_run_dir
        else run_roots[-1]
    )
    if not output_root.exists():
        raise SystemExit(f"Output run root does not exist: {output_root}")

    if len(run_roots) == 1:
        lines = _build_summary_lines(run_roots[0])
    else:
        lines = _build_summary_lines_many(run_roots, output_root)
    text = "\n".join(lines) + "\n"
    print(text, end="")

    output_path = output_root / args.output
    output_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
