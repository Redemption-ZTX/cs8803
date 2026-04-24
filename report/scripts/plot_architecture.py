#!/usr/bin/env python3
"""Architecture diagram — refined academic palette (Nord-inspired).

Output: report/figures/architecture.pdf
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Nord-inspired palette (more editorial / publication feel)
C_OBS    = "#ECEFF4"   # snow-storm #3 (near-white grey)
C_ENC    = "#88C0D0"   # frost #2 (light cyan)
C_ATTN   = "#D08770"   # aurora #3 (warm coral)
C_MERGE  = "#A3BE8C"   # aurora #4 (muted sage)
C_POL    = "#BF616A"   # aurora #1 (red)
C_VAL    = "#B48EAD"   # aurora #5 (purple)
C_EDGE   = "#2E3440"   # polar-night #1 (deep indigo)
C_BORDER = "#2E3440"
C_TEXT   = "#2E3440"
C_HINT   = "#4C566A"   # polar-night #4 (mid grey-blue)
C_ACCENT = "#5E81AC"   # frost #4 (accent blue)


def rbox(ax, x, y, w, h, face, text, *, border=C_BORDER, fontsize=9,
         weight="normal", txt_color=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.07",
        linewidth=1.2, edgecolor=border, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=txt_color or C_TEXT, weight=weight, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, *, color=None, mutation_scale=11, lw=1.2):
    color = color or C_EDGE
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle="-|>", mutation_scale=mutation_scale,
                        linewidth=lw, color=color,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    ax.set_xlim(0, 11.8)
    ax.set_ylim(-0.15, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Sizes
    box_w_io = 1.55
    box_w_enc = 1.65
    box_w_attn = 1.85
    box_w_merge = 1.85
    box_w_head = 1.70
    box_h = 0.75

    y_top = 2.30
    y_bot = 0.55

    # Column 1: inputs
    x1 = 0.20
    rbox(ax, x1, y_top, box_w_io, box_h, C_OBS,
         r"obs$_0$" + "\n" + r"$\mathbb{R}^{336}$", fontsize=9)
    rbox(ax, x1, y_bot, box_w_io, box_h, C_OBS,
         r"obs$_1$" + "\n" + r"$\mathbb{R}^{336}$", fontsize=9)

    # Column 2: Siamese encoder (two copies, tied)
    x2 = x1 + box_w_io + 0.55
    rbox(ax, x2, y_top, box_w_enc, box_h, C_ENC,
         "Siamese MLP\n" + r"$256 \!\to\! 256$", fontsize=9,
         txt_color="#2E3440")
    rbox(ax, x2, y_bot, box_w_enc, box_h, C_ENC,
         "Siamese MLP\n" + r"$256 \!\to\! 256$", fontsize=9,
         txt_color="#2E3440")
    # Tied-weights dashed connector
    ax.plot([x2 + box_w_enc / 2, x2 + box_w_enc / 2],
            [y_top, y_bot + box_h],
            linestyle=(0, (2.5, 2)), color=C_HINT, linewidth=1.0)
    ax.text(x2 + box_w_enc / 2 + 0.08, (y_top + y_bot + box_h) / 2,
            "weights\ntied", fontsize=7.5, color=C_HINT,
            ha="left", va="center", style="italic", linespacing=1.0)

    # Column 3: cross-agent attention (centered block)
    x3 = x2 + box_w_enc + 0.70
    y_mid = (y_top + y_bot + box_h) / 2
    y_attn = y_mid - box_h / 2
    rbox(ax, x3, y_attn, box_w_attn, box_h, C_ATTN,
         "Cross-agent\n" + r"attention $(4 \times 64)$", fontsize=9,
         txt_color="#2E3440", weight="bold")

    # Column 4: merge
    x4 = x3 + box_w_attn + 0.55
    rbox(ax, x4, y_attn, box_w_merge, box_h, C_MERGE,
         "Merge\n" + r"$512 \!\to\! 256 \!\to\! 128$", fontsize=9,
         txt_color="#2E3440")

    # Column 5: heads
    x5 = x4 + box_w_merge + 0.60
    rbox(ax, x5, y_top, box_w_head, box_h, C_POL,
         "Policy head\n" + r"logits $[3,3,3]^{\times 2}$", fontsize=8.5,
         txt_color="white", weight="bold")
    rbox(ax, x5, y_bot, box_w_head, box_h, C_VAL,
         "Value head\n" + r"$V(s) \in \mathbb{R}$", fontsize=8.5,
         txt_color="white", weight="bold")

    # Arrows (dark navy for publication feel)
    for (ax0, ay0, ax1, ay1) in [
        (x1 + box_w_io, y_top + box_h / 2, x2, y_top + box_h / 2),
        (x1 + box_w_io, y_bot + box_h / 2, x2, y_bot + box_h / 2),
        (x2 + box_w_enc, y_top + box_h / 2, x3, y_mid + 0.12),
        (x2 + box_w_enc, y_bot + box_h / 2, x3, y_mid - 0.12),
        (x3 + box_w_attn, y_mid, x4, y_mid),
        (x4 + box_w_merge, y_mid + 0.08, x5, y_top + box_h / 2),
        (x4 + box_w_merge, y_mid - 0.08, x5, y_bot + box_h / 2),
    ]:
        arrow(ax, ax0, ay0, ax1, ay1)

    # Figure footnote
    ax.text(0.20, 3.50,
            r"Team-level Siamese + cross-agent attention  $\cdot$  $\sim$0.46 M parameters  $\cdot$  value head shares encoder",
            fontsize=9, color=C_HINT, style="italic")

    plt.tight_layout()
    out = FIG_DIR / "architecture.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
