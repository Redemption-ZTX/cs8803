"""Trigger functions and ball-position helpers for SNAPSHOT-048 hybrid eval.

Trigger semantics (2026-04-19 recalibrated after R1 sanity):

  Original draft used instantaneous thresholds (`ball_x < -2`) which fire
  whenever ball briefly crosses midfield — typically 30-40% of steps even for
  a winning student. That conflates "ball in our half right now" with
  "student is in defensive trouble", and the test ends up measuring
  "baseline does 30% of the work" rather than "DAGGER selective help".

  Real defensive_pin failure mode (per v2 buckets, snapshot-036 §12) is
  a TAIL-WINDOW mean phenomenon, not instantaneous: the ball stays deep
  in our half for many consecutive steps. We pivot both triggers to
  window-based semantics matching that definition.

  alpha (loose): window_mean(40) < -3.0   — direct match to v2 DEFENSIVE_PIN_TAIL_X
  beta  (tight): window_mean(40) < -5.0   — only when window strongly pinned

  Both require >= MIN_WINDOW_SAMPLES samples before firing (avoid spurious
  fires at episode start when window is empty / dominated by initial 0.0).

Snapshot-049 verified info["ball_info"]["position"] exposes accurate
absolute world coordinates; evaluator pulls ball_x from there.

Convention: ball_x_signed < 0 means ball is in team0's defensive half.
For team0_ids=(2,3) (orange side, spawn at +x), the world coordinate is
flipped so "negative = our half" stays consistent.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, List


# Trigger thresholds (snapshot-048 §2.2 recalibrated)
ALPHA_WINDOW_MEAN_THRESHOLD = -3.0   # matches v2 DEFENSIVE_PIN_TAIL_X (failure_buckets_v2.py:43)
BETA_WINDOW_MEAN_THRESHOLD = -5.0    # tighter: only sustained deep pin
BETA_WINDOW_LEN = 40                 # tail window length (matches v2 EpisodeFailureRecorder.tail_steps default)
MIN_WINDOW_SAMPLES = 5               # need this many samples before window trigger can fire


TriggerFn = Callable[[float, List[float]], bool]


def trigger_none(_ball_x: float, _hist: List[float]) -> bool:
    """No swap; student plays alone. Used for C0 baseline conditions."""
    return False


def trigger_alpha(_ball_x: float, hist: List[float]) -> bool:
    """Window-based loose trigger: tail mean deep in our half (matches v2 defensive_pin)."""
    if len(hist) < MIN_WINDOW_SAMPLES:
        return False
    window_mean = sum(hist) / len(hist)
    return window_mean < ALPHA_WINDOW_MEAN_THRESHOLD


def trigger_beta(_ball_x: float, hist: List[float]) -> bool:
    """Window-based tight trigger: tail mean strongly pinned in our half."""
    if len(hist) < MIN_WINDOW_SAMPLES:
        return False
    window_mean = sum(hist) / len(hist)
    return window_mean < BETA_WINDOW_MEAN_THRESHOLD


TRIGGER_REGISTRY: dict = {
    "none": trigger_none,
    "alpha": trigger_alpha,
    "beta": trigger_beta,
}


class TriggerState:
    """Per-episode mutable state for a trigger function."""

    def __init__(self, trigger_name: str):
        if trigger_name not in TRIGGER_REGISTRY:
            raise ValueError(
                f"Unknown trigger '{trigger_name}'; must be one of {sorted(TRIGGER_REGISTRY)}"
            )
        self.trigger_name = trigger_name
        self.trigger_fn: TriggerFn = TRIGGER_REGISTRY[trigger_name]
        self.recent_ball_x: Deque[float] = deque(maxlen=BETA_WINDOW_LEN)
        self.swap_count = 0
        self.total_steps = 0

    def reset_episode(self) -> None:
        self.recent_ball_x.clear()

    def step(self, ball_x: float) -> bool:
        """Update history, decide whether to swap this step. Returns True if baseline takes over."""
        self.recent_ball_x.append(float(ball_x))
        self.total_steps += 1
        fired = bool(self.trigger_fn(float(ball_x), list(self.recent_ball_x)))
        if fired:
            self.swap_count += 1
        return fired

    def swap_pct(self) -> float:
        return self.swap_count / max(self.total_steps, 1)


def extract_team0_ball_x(info: dict, team0_ids) -> float:
    """Pull ball_x from the env info dict, signed so that NEGATIVE means
    ball is in team0's defensive half regardless of which side team0 spawns on.

    Convention from snapshot-022 §6.4 + snapshot-013:
      - global agent IDs 0/1 = blue side, spawn at NEGATIVE x (-9.03)
      - global agent IDs 2/3 = orange side, spawn at POSITIVE x (+6.45)
      - "team0 defensive half" therefore depends on whether team0 is blue or orange

    Side determination uses team0_ids parity (a TEAM-LEVEL invariant), not
    per-step player_x position — a player chasing the ball can momentarily
    cross midfield and have opposite-sign x without changing which half is
    its team's defensive half.
    """
    team0_id = int(team0_ids[0])
    if team0_id not in info:
        return 0.0
    a_info = info[team0_id]
    if not isinstance(a_info, dict):
        return 0.0
    ball_info = a_info.get("ball_info") or {}
    ball_pos = ball_info.get("position")
    if ball_pos is None or len(ball_pos) < 1:
        return 0.0
    ball_x_world = float(ball_pos[0])

    # team0_ids = (0, 1) -> blue, defensive half is negative-x, no flip needed
    # team0_ids = (2, 3) -> orange, defensive half is positive-x, flip sign
    is_orange = team0_id >= 2
    if is_orange:
        return -ball_x_world
    return ball_x_world
