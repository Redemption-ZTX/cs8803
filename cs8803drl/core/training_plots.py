from __future__ import annotations

import csv
import math
import re
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


WIDTH = 1600
HEIGHT = 1498
LEFT = 90
RIGHT = 40
TOP = 55
BOTTOM = 55
PANEL_GAP = 28
PANEL_HEIGHT = 230
PLOT_WIDTH = WIDTH - LEFT - RIGHT

_METRIC_PATTERN = re.compile(r"^info/learner/([^/]+)/learner_stats/([^/]+)$")
_LOSS_METRICS = ("total_loss", "policy_loss", "vf_loss")
_AUX_METRICS = ("entropy", "kl")
_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#17becf",
    "#bcbd22",
]


_FONT = {
    " ": ["000", "000", "000", "000", "000"],
    "-": ["000", "000", "111", "000", "000"],
    "_": ["000", "000", "000", "000", "111"],
    ".": ["000", "000", "000", "000", "010"],
    ",": ["000", "000", "000", "010", "100"],
    ":": ["000", "010", "000", "010", "000"],
    "/": ["001", "001", "010", "100", "100"],
    "+": ["000", "010", "111", "010", "000"],
    "(": ["001", "010", "010", "010", "001"],
    ")": ["100", "010", "010", "010", "100"],
    "%": ["101", "001", "010", "100", "101"],
    "=": ["000", "111", "000", "111", "000"],
    "?": ["111", "001", "010", "000", "010"],
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "001", "001", "001"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "A": ["111", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"],
    "C": ["111", "100", "100", "100", "111"],
    "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "110", "100", "111"],
    "F": ["111", "100", "110", "100", "100"],
    "G": ["111", "100", "101", "101", "111"],
    "H": ["101", "101", "111", "101", "101"],
    "I": ["111", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "101", "111"],
    "K": ["101", "101", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"],
    "N": ["101", "111", "111", "111", "101"],
    "O": ["111", "101", "101", "101", "111"],
    "P": ["111", "101", "111", "100", "100"],
    "Q": ["111", "101", "101", "111", "001"],
    "R": ["111", "101", "111", "110", "101"],
    "S": ["111", "100", "111", "001", "111"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "010"],
    "W": ["101", "101", "111", "111", "101"],
    "X": ["101", "101", "010", "101", "101"],
    "Y": ["101", "101", "010", "010", "010"],
    "Z": ["111", "001", "010", "100", "111"],
}


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


class _Canvas:
    def __init__(self, width: int, height: int, background: str = "#fafafa") -> None:
        self.width = width
        self.height = height
        bg = _hex_to_rgb(background)
        self.buf = bytearray(bg * (width * height))

    def _idx(self, x: int, y: int) -> int:
        return (y * self.width + x) * 3

    def set_pixel(self, x: int, y: int, color: str) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        idx = self._idx(x, y)
        r, g, b = _hex_to_rgb(color)
        self.buf[idx : idx + 3] = bytes((r, g, b))

    def fill_rect(self, x: int, y: int, w: int, h: int, color: str) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        rgb = bytes(_hex_to_rgb(color))
        row = rgb * (x1 - x0)
        for yy in range(y0, y1):
            idx = self._idx(x0, yy)
            self.buf[idx : idx + len(row)] = row

    def rect(self, x: int, y: int, w: int, h: int, color: str) -> None:
        self.fill_rect(x, y, w, 1, color)
        self.fill_rect(x, y + h - 1, w, 1, color)
        self.fill_rect(x, y, 1, h, color)
        self.fill_rect(x + w - 1, y, 1, h, color)

    def line(self, x0: float, y0: float, x1: float, y1: float, color: str, thickness: int = 2) -> None:
        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy)))
        if steps <= 0:
            self.fill_rect(int(round(x0)) - thickness // 2, int(round(y0)) - thickness // 2, thickness, thickness, color)
            return
        for i in range(steps + 1):
            t = i / steps
            x = int(round(x0 + dx * t))
            y = int(round(y0 + dy * t))
            self.fill_rect(x - thickness // 2, y - thickness // 2, thickness, thickness, color)

    def circle(self, cx: float, cy: float, r: int, color: str) -> None:
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        for yy in range(cy_i - r, cy_i + r + 1):
            for xx in range(cx_i - r, cx_i + r + 1):
                if (xx - cx_i) ** 2 + (yy - cy_i) ** 2 <= r * r:
                    self.set_pixel(xx, yy, color)

    def text(self, x: int, y: int, text: str, color: str = "#222222", scale: int = 3) -> None:
        cursor_x = x
        for ch in text:
            glyph = _FONT.get(ch.upper(), _FONT["?"])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(cursor_x + gx * scale, y + gy * scale, scale, scale, color)
            cursor_x += (len(glyph[0]) + 1) * scale

    def save_png(self, path: Path) -> None:
        def _chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        raw = bytearray()
        row_bytes = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * row_bytes
            raw.extend(self.buf[start : start + row_bytes])

        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(_chunk(b"IHDR", struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)))
        png.extend(_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
        png.extend(_chunk(b"IEND", b""))
        path.write_bytes(bytes(png))


def _row_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


def _scale_x(x: float, x_min: float, x_max: float) -> float:
    denom = (x_max - x_min) if x_max != x_min else 1.0
    return LEFT + (float(x) - float(x_min)) / denom * PLOT_WIDTH


def _scale_y(y: float, top: float, y_min: float, y_max: float) -> float:
    denom = (y_max - y_min) if y_max != y_min else 1.0
    return top + PANEL_HEIGHT - (float(y) - float(y_min)) / denom * PANEL_HEIGHT


def _find_progress_csv(run_root: Path) -> Optional[Path]:
    candidates = _list_progress_csvs(run_root)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _list_progress_csvs(run_root: Path) -> List[Path]:
    run_root = Path(run_root)
    if not run_root.exists():
        return []

    candidates: List[Path] = []
    direct = run_root / "progress.csv"
    if direct.exists():
        candidates.append(direct)

    candidates.extend(sorted(run_root.glob("*/progress.csv")))
    return candidates


def _read_csv_rows_lossy(csv_path: Path) -> List[Dict[str, str]]:
    try:
        raw = csv_path.read_bytes()
    except OSError:
        return []

    text = raw.replace(b"\x00", b"").decode("utf-8", errors="ignore")
    try:
        return list(csv.DictReader(text.splitlines()))
    except csv.Error:
        return []


def _filter_monotonic_progress_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    last_iteration = 0
    last_timesteps = 0
    for row in rows:
        iteration = _row_int(row, "training_iteration", 0)
        timesteps_total = _row_int(row, "timesteps_total", 0)
        if filtered and (iteration < last_iteration or timesteps_total < last_timesteps):
            continue
        filtered.append(dict(row))
        last_iteration = max(last_iteration, iteration)
        last_timesteps = max(last_timesteps, timesteps_total)
    return filtered


def _load_progress_rows(run_root: Path) -> List[Dict[str, str]]:
    merged_rows: List[Dict[str, str]] = []
    for source_index, progress_csv in enumerate(_list_progress_csvs(run_root)):
        rows = _filter_monotonic_progress_rows(_read_csv_rows_lossy(progress_csv))
        for row_index, row in enumerate(rows):
            enriched = dict(row)
            enriched["__source_progress_csv"] = str(progress_csv)
            enriched["__source_order"] = str(source_index)
            enriched["__row_order"] = str(row_index)
            merged_rows.append(enriched)

    if not merged_rows:
        return []

    merged_rows.sort(
        key=lambda row: (
            _row_int(row, "training_iteration", 0),
            _row_int(row, "timesteps_total", 0),
            _row_float(row, "time_total_s", 0.0),
            int(row.get("__source_order", "0")),
            int(row.get("__row_order", "0")),
        )
    )

    rows_without_iteration: List[Dict[str, str]] = []
    deduped_by_iteration: Dict[int, Dict[str, str]] = {}
    for row in merged_rows:
        iteration = _row_int(row, "training_iteration", 0)
        if iteration <= 0:
            rows_without_iteration.append(row)
            continue
        # Later rows win, so resumed trials override older rows at the same iteration.
        deduped_by_iteration[iteration] = row

    return rows_without_iteration + [deduped_by_iteration[it] for it in sorted(deduped_by_iteration)]


def summarize_training_progress(run_root: str | Path) -> Optional[Dict[str, object]]:
    progress_rows = _load_progress_rows(Path(run_root))
    if not progress_rows:
        return None

    best_row: Optional[Dict[str, str]] = None
    best_key: Optional[Tuple[float, int, int, float]] = None
    final_row: Optional[Dict[str, str]] = None
    final_key: Optional[Tuple[int, int, float]] = None

    for row in progress_rows:
        iteration = _row_int(row, "training_iteration", 0)
        timesteps_total = _row_int(row, "timesteps_total", 0)
        elapsed = _row_float(row, "time_total_s", 0.0)

        row_final_key = (iteration, timesteps_total, elapsed)
        if final_key is None or row_final_key > final_key:
            final_key = row_final_key
            final_row = row

        reward = _row_float(row, "episode_reward_mean", float("nan"))
        if not math.isfinite(reward):
            continue
        row_best_key = (reward, iteration, timesteps_total, elapsed)
        if best_key is None or row_best_key > best_key:
            best_key = row_best_key
            best_row = row

    progress_csvs = [str(path) for path in _list_progress_csvs(Path(run_root))]
    return {
        "progress_rows": progress_rows,
        "progress_csvs": progress_csvs,
        "best_row": best_row,
        "final_row": final_row,
    }


def _load_eval_rows(run_root: Path) -> List[Dict[str, str]]:
    eval_csv = run_root / "checkpoint_eval.csv"
    if not eval_csv.exists():
        return []
    with eval_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_plot_data(
    progress_rows: Sequence[Dict[str, str]],
    eval_rows: Sequence[Dict[str, str]],
) -> Optional[Dict[str, object]]:
    if not progress_rows:
        return None

    x_values: List[int] = []
    reward_values: List[float] = []
    series: Dict[Tuple[str, str], List[float]] = {}

    for idx, row in enumerate(progress_rows, start=1):
        iteration = _row_int(row, "training_iteration", idx)
        x_values.append(iteration if iteration > 0 else idx)
        reward_values.append(_row_float(row, "episode_reward_mean", float("nan")))

        for key, raw in row.items():
            match = _METRIC_PATTERN.match(key)
            if not match:
                continue
            policy_id, metric_name = match.groups()
            if metric_name not in _LOSS_METRICS + _AUX_METRICS:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float("nan")
            series.setdefault((policy_id, metric_name), []).append(value)

    if not x_values:
        return None

    eval_points: List[Tuple[int, float]] = []
    for row in eval_rows:
        if (row.get("opponent") or "").strip() != "baseline":
            continue
        it = _row_int(row, "checkpoint_iteration", 0)
        wr = _row_float(row, "win_rate", float("nan"))
        if it > 0 and math.isfinite(wr):
            eval_points.append((it, wr))

    return {
        "x": x_values,
        "reward": reward_values,
        "series": series,
        "eval_points": eval_points,
    }


def _draw_axes(canvas: _Canvas, top: int, title: str, y_min: float, y_max: float, x_min: int, x_max: int) -> None:
    canvas.fill_rect(LEFT, top, PLOT_WIDTH, PANEL_HEIGHT, "#ffffff")
    canvas.rect(LEFT, top, PLOT_WIDTH, PANEL_HEIGHT, "#cfcfcf")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(round(top + PANEL_HEIGHT * frac))
        canvas.line(LEFT, y, LEFT + PLOT_WIDTH, y, "#ececec", thickness=1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(round(LEFT + PLOT_WIDTH * frac))
        canvas.line(x, top, x, top + PANEL_HEIGHT, "#f3f3f3", thickness=1)
        tick_value = int(round(x_min + (x_max - x_min) * frac))
        canvas.text(x - 12, top + PANEL_HEIGHT + 8, str(tick_value), "#666666", scale=2)

    canvas.text(LEFT, top - 26, title, "#111111", scale=3)
    canvas.text(12, top + 4, f"{y_max:.3f}", "#666666", scale=2)
    canvas.text(12, top + PANEL_HEIGHT - 12, f"{y_min:.3f}", "#666666", scale=2)


def _draw_series(
    canvas: _Canvas,
    *,
    xs: Sequence[int],
    ys: Sequence[float],
    top: int,
    y_min: float,
    y_max: float,
    x_min: int,
    x_max: int,
    color: str,
    thickness: int = 2,
) -> None:
    prev = None
    for x, y in zip(xs, ys):
        if not (isinstance(y, (int, float)) and math.isfinite(y)):
            prev = None
            continue
        px = _scale_x(x, x_min, x_max)
        py = _scale_y(y, top, y_min, y_max)
        if prev is not None:
            canvas.line(prev[0], prev[1], px, py, color, thickness=thickness)
        prev = (px, py)


def _draw_eval_points(
    canvas: _Canvas,
    *,
    points: Sequence[Tuple[int, float]],
    top: int,
    x_min: int,
    x_max: int,
    color: str = "#ff7f0e",
) -> None:
    prev = None
    for x, y in points:
        px = _scale_x(x, x_min, x_max)
        py = _scale_y(y, top, 0.0, 1.0)
        if prev is not None:
            canvas.line(prev[0], prev[1], px, py, "#2ca02c", thickness=2)
        canvas.circle(px, py, 4, color)
        prev = (px, py)


def _legend_series(items: Sequence[Tuple[str, str]], top: int) -> int:
    return max(1, len(items)) * 18


def write_training_history_png(
    *,
    progress_rows: Sequence[Dict[str, str]],
    eval_rows: Sequence[Dict[str, str]],
    output_path: Path,
    title: str,
) -> Optional[str]:
    data = _extract_plot_data(progress_rows, eval_rows)
    if not data:
        return None

    xs = data["x"]  # type: ignore[assignment]
    reward = data["reward"]  # type: ignore[assignment]
    series = data["series"]  # type: ignore[assignment]
    eval_points = data["eval_points"]  # type: ignore[assignment]

    x_min = min(xs)
    x_max = max(xs)

    loss_items = sorted((k, v) for k, v in series.items() if k[1] in _LOSS_METRICS)
    entropy_items = sorted((k, v) for k, v in series.items() if k[1] == "entropy")
    kl_items = sorted((k, v) for k, v in series.items() if k[1] == "kl")

    loss_bounds = _bounds([values for _, values in loss_items], default=(-1.0, 1.0))
    entropy_bounds = _bounds([values for _, values in entropy_items], default=(0.0, 1.0))
    kl_bounds = _bounds([values for _, values in kl_items], default=(0.0, 0.1))
    reward_bounds = _bounds([reward], default=(-1.0, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if plt is not None:
        fig, axes = plt.subplots(5, 1, figsize=(16, 15.0), sharex=True)
        fig.suptitle(title)

        ax = axes[0]
        for idx, ((policy_id, metric_name), values) in enumerate(loss_items):
            color = _PALETTE[idx % len(_PALETTE)]
            ax.plot(xs[: len(values)], values, label=f"{policy_id} {metric_name}", color=color, linewidth=1.7)
        ax.set_title("Losses")
        ax.set_ylim(loss_bounds)
        ax.grid(True, alpha=0.25)
        if loss_items:
            ax.legend(loc="upper left", fontsize=8)

        ax = axes[1]
        for idx, ((policy_id, metric_name), values) in enumerate(entropy_items):
            color = _PALETTE[idx % len(_PALETTE)]
            ax.plot(xs[: len(values)], values, label=f"{policy_id} {metric_name}", color=color, linewidth=1.7)
        ax.set_title("Entropy")
        ax.set_ylim(entropy_bounds)
        ax.grid(True, alpha=0.25)
        if entropy_items:
            ax.legend(loc="upper left", fontsize=8)

        ax = axes[2]
        for idx, ((policy_id, metric_name), values) in enumerate(kl_items):
            color = _PALETTE[idx % len(_PALETTE)]
            ax.plot(xs[: len(values)], values, label=f"{policy_id} {metric_name}", color=color, linewidth=1.7)
        ax.set_title("KL")
        ax.set_ylim(kl_bounds)
        ax.grid(True, alpha=0.25)
        if kl_items:
            ax.legend(loc="upper left", fontsize=8)

        ax = axes[3]
        ax.plot(xs[: len(reward)], reward, label="episode_reward_mean", color="#111111", linewidth=2.0)
        ax.set_title("Episode Reward Mean")
        ax.set_ylim(reward_bounds)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)

        ax = axes[4]
        if eval_points:
            eval_x = [x for x, _ in eval_points]
            eval_y = [y for _, y in eval_points]
            ax.plot(eval_x, eval_y, color="#2ca02c", linewidth=1.6)
            ax.scatter(eval_x, eval_y, color="#ff7f0e", s=16, label="baseline checkpoint eval")
        ax.set_title("Baseline Win Rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        if eval_points:
            ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Training Iteration")

        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return str(output_path)

    canvas = _Canvas(WIDTH, HEIGHT, background="#fafafa")
    canvas.text(LEFT, 18, title, "#111111", scale=3)
    canvas.text(LEFT, 36, f"ITER {x_min}..{x_max}", "#555555", scale=2)

    tops = [
        TOP,
        TOP + PANEL_HEIGHT + PANEL_GAP,
        TOP + 2 * (PANEL_HEIGHT + PANEL_GAP),
        TOP + 3 * (PANEL_HEIGHT + PANEL_GAP),
        TOP + 4 * (PANEL_HEIGHT + PANEL_GAP),
    ]

    _draw_axes(canvas, tops[0], "LOSSES", loss_bounds[0], loss_bounds[1], x_min, x_max)
    for idx, ((policy_id, metric_name), values) in enumerate(loss_items):
        color = _PALETTE[idx % len(_PALETTE)]
        _draw_series(
            canvas,
            xs=xs[: len(values)],
            ys=values,
            top=tops[0],
            y_min=loss_bounds[0],
            y_max=loss_bounds[1],
            x_min=x_min,
            x_max=x_max,
            color=color,
        )
        canvas.line(LEFT + 8, tops[0] + 12 + idx * 18, LEFT + 26, tops[0] + 12 + idx * 18, color, thickness=2)
        canvas.text(LEFT + 32, tops[0] + 6 + idx * 18, f"{policy_id} {metric_name}", "#333333", scale=2)

    _draw_axes(canvas, tops[1], "ENTROPY", entropy_bounds[0], entropy_bounds[1], x_min, x_max)
    for idx, ((policy_id, metric_name), values) in enumerate(entropy_items):
        color = _PALETTE[idx % len(_PALETTE)]
        _draw_series(
            canvas,
            xs=xs[: len(values)],
            ys=values,
            top=tops[1],
            y_min=entropy_bounds[0],
            y_max=entropy_bounds[1],
            x_min=x_min,
            x_max=x_max,
            color=color,
        )
        canvas.line(LEFT + 8, tops[1] + 12 + idx * 18, LEFT + 26, tops[1] + 12 + idx * 18, color, thickness=2)
        canvas.text(LEFT + 32, tops[1] + 6 + idx * 18, f"{policy_id} {metric_name}", "#333333", scale=2)

    _draw_axes(canvas, tops[2], "KL", kl_bounds[0], kl_bounds[1], x_min, x_max)
    for idx, ((policy_id, metric_name), values) in enumerate(kl_items):
        color = _PALETTE[idx % len(_PALETTE)]
        _draw_series(
            canvas,
            xs=xs[: len(values)],
            ys=values,
            top=tops[2],
            y_min=kl_bounds[0],
            y_max=kl_bounds[1],
            x_min=x_min,
            x_max=x_max,
            color=color,
        )
        canvas.line(LEFT + 8, tops[2] + 12 + idx * 18, LEFT + 26, tops[2] + 12 + idx * 18, color, thickness=2)
        canvas.text(LEFT + 32, tops[2] + 6 + idx * 18, f"{policy_id} {metric_name}", "#333333", scale=2)

    _draw_axes(canvas, tops[3], "EPISODE REWARD MEAN", reward_bounds[0], reward_bounds[1], x_min, x_max)
    _draw_series(
        canvas,
        xs=xs[: len(reward)],
        ys=reward,
        top=tops[3],
        y_min=reward_bounds[0],
        y_max=reward_bounds[1],
        x_min=x_min,
        x_max=x_max,
        color="#111111",
        thickness=3,
    )
    canvas.line(LEFT + 8, tops[3] + 12, LEFT + 26, tops[3] + 12, "#111111", thickness=3)
    canvas.text(LEFT + 32, tops[3] + 6, "EPISODE_REWARD_MEAN", "#333333", scale=2)

    _draw_axes(canvas, tops[4], "BASELINE WIN RATE", 0.0, 1.0, x_min, x_max)
    if eval_points:
        _draw_eval_points(canvas, points=eval_points, top=tops[4], x_min=x_min, x_max=x_max)
    canvas.line(LEFT + 8, tops[4] + 12, LEFT + 26, tops[4] + 12, "#2ca02c", thickness=2)
    canvas.circle(LEFT + 17, tops[4] + 12, 4, "#ff7f0e")
    canvas.text(LEFT + 32, tops[4] + 6, "BASELINE_CHECKPOINT_EVAL", "#333333", scale=2)

    canvas.text(LEFT, HEIGHT - BOTTOM + 8, "TRAINING ITERATION", "#555555", scale=2)

    canvas.save_png(output_path)
    return str(output_path)


def write_training_loss_curve_png(run_root: str | Path, output_name: str = "training_loss_curve.png") -> Optional[str]:
    run_root = Path(run_root)
    progress_rows = _load_progress_rows(run_root)
    eval_rows = _load_eval_rows(run_root)
    return write_training_history_png(
        progress_rows=progress_rows,
        eval_rows=eval_rows,
        output_path=run_root / output_name,
        title="TRAINING LOSS CURVE",
    )
