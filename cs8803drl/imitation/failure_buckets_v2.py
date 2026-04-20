"""Failure-bucket v2 classifier (snapshot-036 §12 redesign).

v1 lived in ``cs8803drl.evaluation.failure_cases.classify_failure`` and
suffered from two issues exposed by the 313-episode Stage-1 dump:

  1. Thresholds (e.g. ``mean_ball_x < -0.15``) were written as if metric
     values were normalized; in fact the soccer field is in raw Unity
     units (~ ±15). Almost any leftward bias triggered ``territory_loss``
     and ``late_defensive_collapse`` simultaneously, so the two heads
     learned redundant signals.

  2. ``low_possession`` (poss < 0.35) only fired on **7.1%** of the
     SOTA target policy 029B's losses — the head had almost no negative
     samples to learn from. Real losses for strong policies look like
     "high possession + failed conversion", not "no ball".

v2 keeps the *spirit* (possession matters; defensive pinning matters)
but replaces single-direction thresholds with discriminative ones
calibrated on the actual W vs L metric distributions:

    W eps median team0_poss = 0.483
    L eps median team0_poss = 0.588   ← high possession is a LOSS signal

So instead of ``low_possession``, v2 has two complementary heads:

  - ``wasted_possession``  — team0_poss > 0.55 AND L (the strong-policy failure mode)
  - ``possession_stolen``  — team0_poss < 0.35 AND L (the weak-policy failure mode)

Pure function: only depends on the ``metrics`` dict already saved in
``.meta.json`` by the trajectory dumper. No re-recording required —
trainers can re-label existing dumps in memory.

For aggregate validation see snapshot-036 §12.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# v2 thresholds (raw Unity field units). Keep as constants so they show up in
# checkpoint metadata and are auditable.
DEFENSIVE_PIN_TAIL_X = -3.0          # tail_mean_ball_x < this → ball pinned in own half
TERRITORIAL_DOMINANCE_MEAN_X = -1.5  # mean_ball_x < this → ball lived in our half
WASTED_POSSESSION_RATIO = 0.55       # team0_poss > this AND L → had ball, didn't convert
POSSESSION_STOLEN_RATIO = 0.35       # team0_poss < this AND L → couldn't keep ball
PROGRESS_DEFICIT_GAP = 3.0           # (team1_prog - team0_prog) > this → opp made more progress


# Map v2 head name → list of bucket labels (each head can match multiple labels;
# here it's 1-1 but kept as tuple for compatibility with v1's structure).
HEAD_TO_LABELS_V2: Dict[str, Tuple[str, ...]] = {
    "defensive_pin":         ("defensive_pin",),
    "territorial_dominance": ("territorial_dominance",),
    "wasted_possession":     ("wasted_possession",),
    "possession_stolen":     ("possession_stolen",),
    "progress_deficit":      ("progress_deficit",),
}
HEAD_NAMES_V2: Tuple[str, ...] = tuple(HEAD_TO_LABELS_V2.keys())


def classify_failure_v2(metrics: Optional[Dict[str, Any]], outcome: Optional[str]) -> Dict[str, Any]:
    """Compute v2 failure-bucket multi-label set for one episode.

    Args:
        metrics: dict from .meta.json["metrics"]. Tolerates missing keys.
        outcome: one of "team0_win", "team1_win", "tie".

    Returns:
        ``{"primary_label": str, "labels": List[str]}``. For wins/ties the
        labels mirror v1 ("win" / "tie") so downstream code that filters on
        outcome works unchanged.
    """
    if outcome == "team0_win":
        return {"primary_label": "win", "labels": ["win"]}
    if outcome == "tie":
        return {"primary_label": "tie", "labels": ["tie"]}

    metrics = metrics or {}
    labels: List[str] = []

    mbx = metrics.get("mean_ball_x")
    tbx = metrics.get("tail_mean_ball_x")
    poss = metrics.get("team0_possession_ratio")
    t0p = float(metrics.get("team0_progress_toward_goal", 0.0) or 0.0)
    t1p = float(metrics.get("team1_progress_toward_goal", 0.0) or 0.0)

    if tbx is not None and float(tbx) < DEFENSIVE_PIN_TAIL_X:
        labels.append("defensive_pin")
    if mbx is not None and float(mbx) < TERRITORIAL_DOMINANCE_MEAN_X:
        labels.append("territorial_dominance")
    if poss is not None and float(poss) > WASTED_POSSESSION_RATIO:
        labels.append("wasted_possession")
    if poss is not None and float(poss) < POSSESSION_STOLEN_RATIO:
        labels.append("possession_stolen")
    if (t1p - t0p) > PROGRESS_DEFICIT_GAP:
        labels.append("progress_deficit")

    if not labels:
        labels.append("unclear_loss")

    return {"primary_label": labels[0], "labels": labels}


def thresholds_dict() -> Dict[str, float]:
    """Return active thresholds for logging in checkpoint metadata."""
    return {
        "defensive_pin_tail_x": DEFENSIVE_PIN_TAIL_X,
        "territorial_dominance_mean_x": TERRITORIAL_DOMINANCE_MEAN_X,
        "wasted_possession_ratio": WASTED_POSSESSION_RATIO,
        "possession_stolen_ratio": POSSESSION_STOLEN_RATIO,
        "progress_deficit_gap": PROGRESS_DEFICIT_GAP,
    }
