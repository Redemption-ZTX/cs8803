#!/usr/bin/env python3
"""Training curves: keep the ORIGINAL inline-eval rolling-mean lines (training
dynamics) and overlay the official 1000-episode evaluator points (rigorous
checkpoint-level validation) as scatter.

Lines   = inline eval (n=50/ckpt) rolling mean — smooth trajectory.
Scatter = official eval (n=1000/ckpt) — validated per-checkpoint WR.
"""
from __future__ import annotations
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RAY = Path("/storage/ice1/5/1/wsun377/ray_results_scratch")
EVAL_DIR = PROJECT / "docs/experiments/artifacts/official-evals"
REPORT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

CSV_S2 = RAY / "055v2_recursive_distill_5teacher_lr3e4_scratch_20260420_142725" / "checkpoint_eval.csv"
CSV_S3 = RAY / "055v2_extend_resume_1210_to_2000_20260421_030743" / "checkpoint_eval.csv"

CKPT_RE = re.compile(r"checkpoint-(\d+) vs baseline: win_rate=([0-9.]+)")


def load_inline(csv_path: Path) -> pd.DataFrame:
    """Inline eval CSV: opponent=baseline, ok rows only."""
    df = pd.read_csv(csv_path)
    df = df[(df["opponent"] == "baseline") & (df["status"] == "ok")].copy()
    df = df.sort_values("checkpoint_iteration").reset_index(drop=True)
    return df[["checkpoint_iteration", "win_rate"]]


def rolling(df: pd.DataFrame, k: int = 5) -> pd.Series:
    return df["win_rate"].rolling(k, center=True, min_periods=1).mean()


def parse_official(*paths: Path) -> list[tuple[int, float]]:
    combined: dict[int, list[float]] = {}
    for p in paths:
        if not p.exists():
            continue
        for m in CKPT_RE.finditer(p.read_text()):
            combined.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    return sorted((it, sum(v) / len(v)) for it, v in combined.items())


def main():
    # --- inline eval (lines) ---
    df2 = load_inline(CSV_S2)   # Stage 2: iter 0-1250
    df3 = load_inline(CSV_S3)   # Stage 3: iter 1210-2000

    # --- official 1000-ep (scatter overlay) ---
    s1_official = parse_official(
        EVAL_DIR / "055_baseline1000.log",
        EVAL_DIR / "055_rerun_1000_1010.log",
    )
    s2_official = parse_official(
        EVAL_DIR / "055v2_baseline1000.log",
        EVAL_DIR / "055v2_baseline_rerun1000.log",
        EVAL_DIR / "055v2_baseline_v3_1000.log",
    )
    s3_official = parse_official(EVAL_DIR / "055v2_extend_baseline1000.log")

    # Keep within stage x-range
    s1_official = [(i, w) for (i, w) in s1_official if i <= 1250]
    s2_official = [(i, w) for (i, w) in s2_official if i <= 1250]
    s3_official = [(i, w) for (i, w) in s3_official if 1250 <= i <= 2000]

    # Typography + palette unified with pipeline / architecture figures.
    # figsize is set to native CoRL text-column width (5.5 in) so that
    # \includegraphics[width=\linewidth]{...} embeds at 1:1 with no scaling
    # — fonts stay at their intended point size.
    plt.rcParams.update({
        "font.family": ["Inter", "Source Sans Pro", "Helvetica Neue", "Arial",
                        "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(5.5, 2.3))

    # Match pipeline palette:
    #   Stage 1 teacher blue  #1F6A85 (Nord frost deep)
    #   Stage 2 student blue  #5E81AC (Nord frost frost-4)
    #   Stage 3 final red     #BF616A (Nord aurora 1)
    C_S1_SCAT = "#1F6A85"
    C_S2      = "#5E81AC"
    C_S3      = "#BF616A"
    C_TARGET  = "#0F172A"
    C_CEIL    = "#94A3B8"

    # --- Only rolling-mean lines from inline eval (scatter removed for clarity) ---
    ax.plot(df2["checkpoint_iteration"], rolling(df2, 5),
            color=C_S2, linewidth=2.1,
            label="Stage 2 · inline win-rate (rolling mean)")
    ax.plot(df3["checkpoint_iteration"], rolling(df3, 5),
            color=C_S3, linewidth=2.1,
            label="Stage 3 · inline win-rate (rolling mean)")

    # Highlight submitted ckpt @1750 with the official 1000-ep measurement
    # (single annotated point — no full scatter).
    target_wr = None
    for it, wr in s3_official:
        if it == 1750:
            target_wr = wr
            break
    if target_wr is not None:
        ax.scatter([1750], [target_wr], s=170, marker="*",
                   color="#BF616A", edgecolor="#7F3338", linewidth=1.1,
                   zorder=6, label="submitted ckpt (official $n{=}1000$)")
        ax.annotate("submitted\nckpt 1750",
                    xy=(1750, target_wr), xytext=(1540, target_wr + 0.035),
                    fontsize=8, ha="right", color="#7F3338",
                    weight="bold",
                    arrowprops=dict(arrowstyle="->", color="#7F3338",
                                    lw=0.7))

    # Reference lines
    ax.axhline(0.9, color=C_TARGET, linestyle=":", linewidth=1.0,
               label=r"assignment target ($\geq$0.9)")
    ax.axhline(0.88, color=C_CEIL, linestyle="--", linewidth=1.0,
               label="031B scratch ceiling (0.88)")
    ax.axvline(1210, color="#cbd5e1", linestyle="-", linewidth=0.8, zorder=1)

    ax.set_xlabel("Training iteration", color="#334155")
    ax.set_ylabel("Win rate vs. Baseline", labelpad=5, color="#334155")
    ax.set_xlim(0, 2050)
    ax.set_ylim(0.30, 0.98)
    ax.grid(True, alpha=0.25, color="#cbd5e1")
    ax.tick_params(colors="#334155")
    ax.legend(loc="lower right", fontsize=7.2, framealpha=0.95, ncol=1,
              edgecolor="#cbd5e1")

    plt.tight_layout()
    out = FIG_DIR / "training_curves.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  official pts: S1={len(s1_official)} S2={len(s2_official)} S3={len(s3_official)}")


if __name__ == "__main__":
    main()
