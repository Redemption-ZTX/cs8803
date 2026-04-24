#!/usr/bin/env python3
"""Pipeline diagram — per-model column tabular layout.

Each atomic model (teacher / ensemble / student) gets its own column; ALL text
data (checkpoint, role, WR, stage) lives in the bottom section; vertical rules
separate every column; stage labels belong to a single column each (no spans).

Columns (left to right):
  1  031B@1220      teacher, scratch 031B + v2 shaping
  2  045A@180       teacher, learned reward head
  3  051A@130       teacher, PBRS predictor
  4  034E           deploy-time 3-ensemble  (aggregates 1..3)
  5  Student 055    Stage-1 output, distilled from 034E
  6  056D@1140      teacher, LR-diverse variant (3e-4)
  7  5-teacher pool deploy-time 5-ensemble  (aggregates 1..3 + 5 + 6)
  8  Student 055v2  Stage-2 output, distilled from pool
  9  Our agent      Stage-3 output, extend past teacher horizon
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Role palette
C_TEACH   = "#1F6A85"   # teal — frozen teacher
C_TEACH_B = "#0F4352"
C_NEW     = "#C07A3E"   # amber — new teacher added in Stage 2
C_NEW_B   = "#7A4820"
C_ENS     = "#D08770"   # coral — deploy-time ensemble
C_ENS_B   = "#8A4E3E"
C_STU     = "#5E81AC"   # frost blue — trained student
C_STU_B   = "#3E557B"
C_FINAL   = "#BF616A"   # red — submitted agent
C_FINAL_B = "#7F3338"

C_ARROW   = "#475569"
C_DISTILL = "#BF616A"
C_RULE    = "#94A3B8"
C_HEAD    = "#F1F5F9"   # header band
C_TEXT    = "#1E293B"
C_HINT    = "#64748B"


def box(ax, cx, cy, w, h, face, edge, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.05",
        linewidth=lw, edgecolor=edge, facecolor=face))


def arrow(ax, x0, y0, x1, y1, *, color=C_ARROW, lw=1.1, mutation_scale=10,
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=lw, color=color,
        connectionstyle=connectionstyle, shrinkA=3, shrinkB=3))


COLUMNS = [
    # (short_name, role_short, ckpt, wr, stage, box_face, box_edge, label_on_box)
    ("031B", "teacher\n(scratch 031B,\nv2 shaping)",       "@1220", "0.880",         "Stage 1", C_TEACH, C_TEACH_B, "031B"),
    ("045A", "teacher\n(learned reward\nhead, 184k W/L)",  "@180",  "0.887",         "Stage 1", C_TEACH, C_TEACH_B, "045A"),
    ("051A", "teacher\n(PBRS predictor\nfrom failures)",   "@130",  "0.888",         "Stage 1", C_TEACH, C_TEACH_B, "051A"),
    ("034E", "3-model\ndeploy-time\nensemble",             "factor-avg", "0.878",    "Stage 1", C_ENS,   C_ENS_B,   "034E"),
    ("055",  "Stage-1 student\n(KL distill from\n034E)",    "@1150", "0.907",        "Stage 1", C_STU,   C_STU_B,   "Student\n055"),
    ("056D", "teacher\n(LR-diverse\n$3e{-}4$ variant)",     "@1140", "0.891",        "Stage 2", C_NEW,   C_NEW_B,   "056D"),
    ("pool", "5-model\ndeploy-time\npool",                   "factor-avg", "—",      "Stage 2", C_ENS,   C_ENS_B,   "5-teacher\npool"),
    ("055v2","Stage-2 student\n(recursive distill\nfrom pool)", "@1210", "0.909",    "Stage 2", C_STU,   C_STU_B,   "Student\n055v2"),
    ("final","Stage-3 extend\n(training past\nteacher horizon)", "@1750", r"$\mathbf{0.907 \pm 0.004}$", "Stage 3", C_FINAL, C_FINAL_B, "Our\nagent"),
]


def main():
    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    W, H = 24.0, 9.2
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # layout
    LABEL_W = 2.6
    N = len(COLUMNS)
    col_w = (W - LABEL_W) / N
    col_centers = [LABEL_W + col_w * (i + 0.5) for i in range(N)]

    # y rows (from top to bottom)
    Y_DIAG_CENTER = 7.35
    Y_ROW_NAME    = 5.55
    Y_ROW_CKPT    = 4.75
    Y_ROW_ROLE    = 3.50
    Y_ROW_WR      = 2.05
    Y_ROW_STAGE   = 1.05
    Y_BOTTOM      = 0.15

    # Row separator lines (very subtle)
    def hrule(y, alpha=0.6, w=0.7):
        ax.plot([0, W], [y, y], color=C_RULE, linewidth=w, alpha=alpha)

    hrule(6.25, alpha=0.9, w=0.9)   # below diagram
    hrule(Y_ROW_CKPT - 0.45)
    hrule(Y_ROW_ROLE - 0.75)
    hrule(Y_ROW_WR - 0.55)
    hrule(Y_ROW_STAGE - 0.45)

    # Vertical rules
    for i in range(N + 1):
        x = LABEL_W + col_w * i
        ax.plot([x, x], [0.0, 8.85], color=C_RULE, linewidth=0.8)

    # Left label column
    def lbl(y, text, *, weight="bold", fontsize=9.0, italic=False, color=None):
        ax.text(LABEL_W - 0.20, y, text, ha="right", va="center",
                fontsize=fontsize, weight=weight, color=color or C_TEXT,
                style="italic" if italic else "normal")

    lbl(Y_DIAG_CENTER, "Pipeline\nstep", fontsize=10)
    lbl(Y_ROW_NAME, "Model", fontsize=9.2)
    lbl(Y_ROW_CKPT, "Checkpoint", fontsize=9.2)
    lbl(Y_ROW_ROLE, "Role", fontsize=9.2)
    lbl(Y_ROW_WR,  "Baseline WR", fontsize=9.2)
    lbl(Y_ROW_STAGE, "Stage", fontsize=9.2)

    # ---- Diagram row: model boxes ----
    bw = col_w * 0.72
    bh = 1.20
    box_positions = []
    for i, (_, _, _, _, _, face, edge, label) in enumerate(COLUMNS):
        cx = col_centers[i]
        box(ax, cx, Y_DIAG_CENTER, bw, bh, face, edge, lw=1.2)
        ax.text(cx, Y_DIAG_CENTER, label, ha="center", va="center",
                fontsize=8.5, color="white", weight="bold", linespacing=1.15)
        box_positions.append((cx, Y_DIAG_CENTER))

    # ---- Arrows between model boxes ----
    def top_arrow(i_from, i_to, *, color=C_ARROW, lw=1.0, rad=-0.35, lift=0.35):
        """Arrow that curves ABOVE the boxes (for non-adjacent jumps)."""
        x0, _ = box_positions[i_from]
        x1, _ = box_positions[i_to]
        y0 = Y_DIAG_CENTER + bh / 2 + 0.02
        y1 = Y_DIAG_CENTER + bh / 2 + 0.02
        arrow(ax, x0, y0, x1, y1,
              color=color, lw=lw, mutation_scale=8,
              connectionstyle=f"arc3,rad={rad}")

    def side_arrow(i_from, i_to, *, color=C_ARROW, lw=1.0):
        """Horizontal arrow between adjacent columns."""
        x0, y0 = box_positions[i_from]
        x1, y1 = box_positions[i_to]
        arrow(ax, x0 + bw / 2, y0, x1 - bw / 2, y1,
              color=color, lw=lw)

    # Stage 1 flow: 3 teachers -> 034E  (top curved arrows so they don't
    # pass through intermediate boxes)
    top_arrow(0, 3, lw=0.9, rad=-0.42, lift=0.45)
    top_arrow(1, 3, lw=0.9, rad=-0.36)
    side_arrow(2, 3, lw=0.9)          # 051A adjacent, use straight
    side_arrow(3, 4, color=C_DISTILL, lw=1.5)  # 034E -> 055

    # Stage 2 flow: teachers feed pool
    side_arrow(5, 6, lw=1.0)          # 056D -> pool
    top_arrow(4, 6, color=C_HINT, lw=0.9, rad=-0.32)  # 055 recycled -> pool
    # Original 3 teachers also contribute (shown as one rolled-up label)
    top_arrow(0, 6, color=C_HINT, lw=0.7, rad=-0.50)
    top_arrow(1, 6, color=C_HINT, lw=0.7, rad=-0.45)
    top_arrow(2, 6, color=C_HINT, lw=0.7, rad=-0.40)

    # pool -> 055v2  + 055v2 -> Our agent
    side_arrow(6, 7, color=C_DISTILL, lw=1.5)
    side_arrow(7, 8, color=C_DISTILL, lw=1.5)

    # Distillation edge labels (above boxes)
    ax.text((col_centers[3] + col_centers[4]) / 2, Y_DIAG_CENTER + bh / 2 + 0.30,
            "KL distill\n" + r"$\alpha\!:\!0.05\to0$",
            ha="center", va="bottom", fontsize=7.3, color=C_DISTILL,
            style="italic", linespacing=1.1)
    ax.text((col_centers[6] + col_centers[7]) / 2, Y_DIAG_CENTER + bh / 2 + 0.30,
            "recursive\ndistill",
            ha="center", va="bottom", fontsize=7.3, color=C_DISTILL,
            style="italic", linespacing=1.1)
    ax.text((col_centers[7] + col_centers[8]) / 2, Y_DIAG_CENTER + bh / 2 + 0.30,
            "extend training\n" + r"iter $1210\!\to\!2000$",
            ha="center", va="bottom", fontsize=7.3, color=C_DISTILL,
            style="italic", linespacing=1.1)

    # ---- Data rows ----
    for i, (name, role, ckpt, wr, stage, _, _, _) in enumerate(COLUMNS):
        cx = col_centers[i]
        # Name
        ax.text(cx, Y_ROW_NAME, name, ha="center", va="center",
                fontsize=9.2, color=C_TEXT, weight="bold",
                family="monospace")
        # Checkpoint
        ax.text(cx, Y_ROW_CKPT, ckpt, ha="center", va="center",
                fontsize=8.8, color=C_TEXT,
                style="italic" if ckpt == "factor-avg" else "normal")
        # Role (small multi-line)
        ax.text(cx, Y_ROW_ROLE, role, ha="center", va="center",
                fontsize=7.2, color=C_HINT, linespacing=1.15)
        # WR
        ax.text(cx, Y_ROW_WR, wr, ha="center", va="center",
                fontsize=9.0, color=C_TEXT, weight="bold")
        # Stage (no span; one per column)
        stage_color = {"Stage 1": "#1F6A85", "Stage 2": "#C07A3E",
                       "Stage 3": "#BF616A"}[stage]
        ax.text(cx, Y_ROW_STAGE, stage, ha="center", va="center",
                fontsize=8.5, color=stage_color, weight="bold")

    plt.tight_layout()
    out = FIG_DIR / "pipeline.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
