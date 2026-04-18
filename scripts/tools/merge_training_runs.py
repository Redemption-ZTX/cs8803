#!/usr/bin/env python3
"""Merge sequential training runs into one continuous history and redraw curves.

Intended for cases where a run was interrupted and later continued from a
checkpoint. The merged output contains:

- progress.csv
- checkpoint_eval.csv
- best_checkpoint_by_eval.txt
- merge_manifest.json
- training_loss_curve_full.png

The script uses the shared PNG renderer and falls back gracefully if plotting
dependencies are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cs8803drl.core.training_plots import write_training_history_png


PANEL_WIDTH = 1400
PANEL_HEIGHT = 1180
LEFT = 90
RIGHT = 40
TOP = 50
PLOT_PANEL_HEIGHT = 220
PANEL_GAP = 30
PLOT_WIDTH = PANEL_WIDTH - LEFT - RIGHT
FONT = 'font-family="monospace" font-size="12"'


def _find_progress_csv(run_dir: Path) -> Path:
    direct = run_dir / "progress.csv"
    if direct.exists():
        return direct

    candidates = sorted(run_dir.glob("*/progress.csv"))
    if not candidates:
        raise FileNotFoundError(f"No progress.csv found under {run_dir}")
    return candidates[-1]


def _find_eval_csv(run_dir: Path) -> Optional[Path]:
    path = run_dir / "checkpoint_eval.csv"
    return path if path.exists() else None


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _row_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge_progress(run_dirs: Sequence[Path]) -> Tuple[List[Dict[str, str]], List[str]]:
    merged: Dict[int, Dict[str, str]] = {}
    fieldnames: List[str] = []
    seen_fields = set()

    for run_dir in run_dirs:
        rows = _load_csv(_find_progress_csv(run_dir))
        for row in rows:
            for key in row.keys():
                if key not in seen_fields:
                    seen_fields.add(key)
                    fieldnames.append(key)
            iteration = _row_int(row, "training_iteration", 0)
            if iteration <= 0:
                continue
            merged[iteration] = row

    ordered_rows = [merged[it] for it in sorted(merged)]
    return ordered_rows, fieldnames


def _merge_eval(run_dirs: Sequence[Path]) -> Tuple[List[Dict[str, str]], List[str]]:
    merged: Dict[Tuple[int, str], Dict[str, str]] = {}
    fieldnames: List[str] = []
    seen_fields = set()

    for run_dir in run_dirs:
        eval_csv = _find_eval_csv(run_dir)
        if eval_csv is None:
            continue
        rows = _load_csv(eval_csv)
        for row in rows:
            for key in row.keys():
                if key not in seen_fields:
                    seen_fields.add(key)
                    fieldnames.append(key)
            checkpoint_iteration = _row_int(row, "checkpoint_iteration", 0)
            opponent = (row.get("opponent") or "").strip()
            if checkpoint_iteration <= 0 or not opponent:
                continue
            merged[(checkpoint_iteration, opponent)] = row

    def _sort_key(item: Dict[str, str]) -> Tuple[int, int, str]:
        opponent = (item.get("opponent") or "").strip()
        opponent_rank = 0 if opponent == "baseline" else 1
        return (_row_int(item, "checkpoint_iteration", 0), opponent_rank, opponent)

    ordered_rows = sorted(merged.values(), key=_sort_key)
    return ordered_rows, fieldnames


def _select_best_eval(rows: Sequence[Dict[str, str]]) -> Optional[Dict[str, Dict[str, str]]]:
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        if (row.get("status") or "").strip() != "ok":
            continue
        checkpoint_file = row.get("checkpoint_file") or row.get("checkpoint_dir")
        if not checkpoint_file:
            continue
        grouped.setdefault(checkpoint_file, {})[(row.get("opponent") or "").strip()] = row

    best = None
    best_key = None
    for checkpoint_file, result_map in grouped.items():
        baseline = result_map.get("baseline")
        if baseline is None:
            continue
        random_row = result_map.get("random")
        baseline_rate = _row_float(baseline, "win_rate", 0.0)
        random_rate = _row_float(random_row, "win_rate", -1.0) if random_row is not None else -1.0
        checkpoint_iteration = _row_int(baseline, "checkpoint_iteration", -1)
        key = (baseline_rate, random_rate, checkpoint_iteration)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "checkpoint_file": checkpoint_file,
                "baseline": baseline,
                "random": random_row,
            }
    return best


def _write_best_eval_summary(out_dir: Path, best_eval: Optional[Dict[str, Dict[str, str]]]) -> None:
    if not best_eval:
        return
    baseline = best_eval["baseline"]
    random_row = best_eval.get("random")

    lines = [
        f"checkpoint_file={best_eval['checkpoint_file']}",
        f"checkpoint_iteration={baseline.get('checkpoint_iteration', '')}",
        f"baseline_win_rate={_row_float(baseline, 'win_rate', 0.0):.3f}",
        f"baseline_record={baseline.get('wins', '?')}W-{baseline.get('losses', '?')}L-{baseline.get('ties', '?')}T",
    ]
    if random_row is not None:
        lines.extend(
            [
                f"random_win_rate={_row_float(random_row, 'win_rate', 0.0):.3f}",
                f"random_record={random_row.get('wins', '?')}W-{random_row.get('losses', '?')}L-{random_row.get('ties', '?')}T",
            ]
        )

    (out_dir / "best_checkpoint_by_eval.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _bounds(series_values: Sequence[Sequence[float]], default: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
    flat: List[float] = []
    for values in series_values:
        flat.extend(_finite(values))
    if not flat:
        return default
    lower = min(flat)
    upper = max(flat)
    if math.isclose(lower, upper):
        pad = max(1e-6, abs(lower) * 0.1, 0.1)
        return lower - pad, upper + pad
    pad = (upper - lower) * 0.1
    return lower - pad, upper + pad


def _x_to_svg(x: float, x_min: float, x_max: float) -> float:
    denom = (x_max - x_min) if x_max != x_min else 1.0
    return LEFT + (float(x) - float(x_min)) / denom * PLOT_WIDTH


def _y_to_svg(y: float, top: float, y_min: float, y_max: float) -> float:
    denom = (y_max - y_min) if y_max != y_min else 1.0
    return top + PLOT_PANEL_HEIGHT - (float(y) - float(y_min)) / denom * PLOT_PANEL_HEIGHT


def _polyline(xs: Sequence[int], ys: Sequence[float], *, top: float, x_min: float, x_max: float, y_min: float, y_max: float) -> str:
    points = []
    for x, y in zip(xs, ys):
        if not (isinstance(y, (int, float)) and math.isfinite(y)):
            continue
        px = _x_to_svg(x, x_min, x_max)
        py = _y_to_svg(y, top, y_min, y_max)
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def _draw_panel(
    *,
    title: str,
    xs: Sequence[int],
    ys_list: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str],
    dashes: Sequence[Optional[str]],
    top: float,
    y_min: float,
    y_max: float,
    x_min: float,
    x_max: float,
    extra_points: Optional[Sequence[Tuple[int, float, str]]] = None,
    overlay_unit_scale: bool = False,
) -> str:
    parts: List[str] = []
    parts.append(f'<rect x="{LEFT}" y="{top}" width="{PLOT_WIDTH}" height="{PLOT_PANEL_HEIGHT}" fill="white" stroke="#cccccc"/>')
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + PLOT_PANEL_HEIGHT * frac
        parts.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + PLOT_WIDTH}" y2="{y:.1f}" stroke="#eeeeee"/>')
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = LEFT + PLOT_WIDTH * frac
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + PLOT_PANEL_HEIGHT}" stroke="#f3f3f3"/>')

    parts.append(f'<text x="{LEFT}" y="{top - 10}" {FONT}>{title}</text>')
    parts.append(f'<text x="10" y="{top + 15}" {FONT}>{y_max:.3f}</text>')
    parts.append(f'<text x="10" y="{top + PLOT_PANEL_HEIGHT:.1f}" {FONT}>{y_min:.3f}</text>')
    if overlay_unit_scale:
        parts.append(f'<text x="{LEFT + PLOT_WIDTH + 10}" y="{top + 15}" {FONT}>1.000</text>')
        parts.append(f'<text x="{LEFT + PLOT_WIDTH + 10}" y="{top + PLOT_PANEL_HEIGHT:.1f}" {FONT}>0.000</text>')

    legend_x = LEFT + 10
    legend_y = top + 20
    for ys, label, color, dash in zip(ys_list, labels, colors, dashes):
        pts = _polyline(xs, ys, top=top, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if pts:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash_attr} points="{pts}"/>')
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="{color}" stroke-width="2"{dash_attr}/>'
            f'<text x="{legend_x + 28}" y="{legend_y + 4}" {FONT}>{label}</text>'
        )
        legend_y += 18

    if extra_points:
        for ex, ey, color in extra_points:
            if not math.isfinite(ey):
                continue
            px = _x_to_svg(ex, x_min, x_max)
            if overlay_unit_scale:
                py = top + PLOT_PANEL_HEIGHT - ey * PLOT_PANEL_HEIGHT
            else:
                py = _y_to_svg(ey, top, y_min, y_max)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}"/>')
    return "\n".join(parts)


def _write_full_png(out_dir: Path, progress_rows: Sequence[Dict[str, str]], eval_rows: Sequence[Dict[str, str]], title: str) -> None:
    write_training_history_png(
        progress_rows=progress_rows,
        eval_rows=eval_rows,
        output_path=out_dir / "training_loss_curve_full.png",
        title=title,
    )
    write_training_history_png(
        progress_rows=progress_rows,
        eval_rows=eval_rows,
        output_path=out_dir / "training_loss_curve.png",
        title=title,
    )


def _merge_run_histories(run_dirs: Sequence[Path], out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_rows, progress_fields = _merge_progress(run_dirs)
    eval_rows, eval_fields = _merge_eval(run_dirs)

    _write_csv(out_dir / "progress.csv", progress_rows, progress_fields)
    if eval_rows and eval_fields:
        _write_csv(out_dir / "checkpoint_eval.csv", eval_rows, eval_fields)

    best_eval = _select_best_eval(eval_rows)
    _write_best_eval_summary(out_dir, best_eval)

    manifest = {
        "title": title,
        "source_runs": [str(path) for path in run_dirs],
        "merged_progress_rows": len(progress_rows),
        "merged_eval_rows": len(eval_rows),
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _write_full_png(out_dir, progress_rows, eval_rows, title=title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Source run directory. Pass multiple times in chronological order.")
    parser.add_argument("--out-dir", required=True, help="Output directory for merged artifacts.")
    parser.add_argument("--title", default="merged training history", help="Title used in merged plots.")
    args = parser.parse_args()

    run_dirs = [Path(run).resolve() for run in args.run]
    out_dir = Path(args.out_dir).resolve()
    _merge_run_histories(run_dirs, out_dir, title=args.title)
    print(out_dir)


if __name__ == "__main__":
    main()
