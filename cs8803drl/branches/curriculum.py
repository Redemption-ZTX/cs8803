"""Opponent-strength curriculum for team_vs_policy training.

Phase-based schedule of opponent_mix weights over training iterations:
  Phase 1 (warmup): baseline_prob=0.0 → 100% random opponent
  Phase 2 (transition): baseline_prob=0.3 → mostly random + some baseline
  Phase 3 (consolidate): baseline_prob=0.7 → mostly baseline
  Phase 4 (final): baseline_prob=1.0 → 100% baseline

Implementation: callback-driven dynamic weight update across workers.
  - `CurriculumOpponentPolicy` wraps `_EpisodeOpponentPoolPolicy` with `set_pool_weights()`
  - `CurriculumPhaseScheduler` parses CURRICULUM_PHASES env var, computes
    current weights given training iter
  - `CurriculumUpdateCallback` (in train script) calls
    `trainer.workers.foreach_worker(...foreach_env...update_curriculum_weights)`
    after each train iter, synchronizing all envs to current phase

Snapshot reference: SNAPSHOT-058 (Tier A4).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_curriculum_phases(spec: str) -> List[Tuple[int, float]]:
    """Parse CURRICULUM_PHASES env var format: "0:0.0,200:0.3,500:0.7,1000:1.0".
    Returns list of (start_iter, baseline_prob) tuples sorted by start_iter.
    Phase i is active for iter in [phases[i].start_iter, phases[i+1].start_iter).
    """
    phases: List[Tuple[int, float]] = []
    spec = (spec or "").strip()
    if not spec:
        return []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid curriculum phase spec '{token}': expect 'iter:prob'")
        iter_part, prob_part = token.split(":", 1)
        start_iter = int(iter_part.strip())
        baseline_prob = float(prob_part.strip())
        if not 0.0 <= baseline_prob <= 1.0:
            raise ValueError(
                f"Invalid curriculum phase '{token}': baseline_prob must be in [0,1]"
            )
        phases.append((start_iter, baseline_prob))
    if not phases:
        return []
    phases.sort(key=lambda t: t[0])
    if phases[0][0] != 0:
        raise ValueError(
            "First curriculum phase must start at iter 0, got "
            f"{phases[0][0]}"
        )
    return phases


class CurriculumPhaseScheduler:
    """Stateless schedule from training iter → opponent_mix weights."""

    def __init__(self, phases: List[Tuple[int, float]]):
        if not phases:
            raise ValueError("CurriculumPhaseScheduler requires >= 1 phase")
        self._phases = list(phases)
        # Phase iter thresholds for quick lookup
        self._thresholds = [p[0] for p in self._phases]

    def baseline_prob_for_iter(self, train_iter: int) -> float:
        """Return baseline_prob for the active phase at given training iter."""
        active = self._phases[0]
        for phase in self._phases:
            if train_iter >= phase[0]:
                active = phase
            else:
                break
        return float(active[1])

    def compute_weights_dict(self, train_iter: int) -> Dict[str, float]:
        """Return {entry_name: weight} for current phase. Used by CurriculumOpponentPolicy."""
        b_prob = self.baseline_prob_for_iter(train_iter)
        return {
            "baseline": float(b_prob),
            "random": float(1.0 - b_prob),
        }


class AdaptiveCurriculumPhaseScheduler(CurriculumPhaseScheduler):
    """Reward-gated phase transition (SNAPSHOT-062 Tier A4.1).

    Extends CurriculumPhaseScheduler with two new gates:
      1. `gate_rewards[i]`: minimum episode_reward_mean required to advance
         from phase i → phase i+1 (len = num_phases - 1).
      2. `min_phase_iters`: at least N iters must pass between advances
         (anti-thrash).
      3. `max_phase_wait`: hard fallback — if train_iter >= next_phase.start_iter
         + max_phase_wait, force advance even if reward gate fails (anti-stall).

    Semantics: a phase boundary is crossed when ALL of:
      (a) train_iter >= next_phase.start_iter
      (b) recent_reward >= gate_rewards[current_phase_idx], OR
          (a') train_iter >= next_phase.start_iter + max_phase_wait (fallback)
      (c) train_iter - last_advance_iter >= min_phase_iters

    Use `try_advance(iter, recent_reward)` instead of `baseline_prob_for_iter`.
    """

    def __init__(
        self,
        phases: List[Tuple[int, float]],
        gate_rewards: List[float],
        min_phase_iters: int = 50,
        max_phase_wait: int = 200,
    ):
        super().__init__(phases)
        if len(gate_rewards) != max(0, len(phases) - 1):
            raise ValueError(
                f"gate_rewards length {len(gate_rewards)} must equal num_phases-1 "
                f"({len(phases) - 1}) — one gate per transition"
            )
        self._gate_rewards = list(gate_rewards)
        self._min_phase_iters = int(min_phase_iters)
        self._max_phase_wait = int(max_phase_wait)
        self._current_phase_idx = 0
        self._last_advance_iter = 0
        self._gate_triggered_fallback = False  # diagnostic: was last advance forced?

    def try_advance(self, train_iter: int, recent_reward: float) -> float:
        """Returns baseline_prob for the current (possibly updated) phase.

        Called per on_train_result. Phase advances happen in-place if conditions met.
        """
        # No next phase → stay at final
        next_idx = self._current_phase_idx + 1
        if next_idx >= len(self._phases):
            return float(self._phases[self._current_phase_idx][1])

        next_phase = self._phases[next_idx]
        iter_ok = train_iter >= next_phase[0]
        cooldown_ok = (train_iter - self._last_advance_iter) >= self._min_phase_iters
        fallback_trigger = train_iter >= (next_phase[0] + self._max_phase_wait)
        reward_ok = recent_reward >= self._gate_rewards[self._current_phase_idx]

        # Advance if: time condition + cooldown + (reward OR fallback)
        if iter_ok and cooldown_ok and (reward_ok or fallback_trigger):
            self._current_phase_idx = next_idx
            self._last_advance_iter = train_iter
            self._gate_triggered_fallback = fallback_trigger and not reward_ok

        return float(self._phases[self._current_phase_idx][1])

    @property
    def current_phase_idx(self) -> int:
        return self._current_phase_idx

    @property
    def last_advance_iter(self) -> int:
        return self._last_advance_iter

    @property
    def gate_triggered_fallback(self) -> bool:
        return self._gate_triggered_fallback


def parse_gate_rewards(spec: str) -> List[float]:
    """Parse CURRICULUM_GATE_REWARDS env var like '-0.5,0.0,0.5' → [-0.5, 0.0, 0.5]."""
    spec = (spec or "").strip()
    if not spec:
        return []
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def update_env_curriculum_weights(env, new_weights: Dict[str, float]) -> bool:
    """Walk env wrapper chain, find opponent_policy with set_pool_weights, update.
    Returns True if a CurriculumOpponentPolicy was found and updated, False otherwise.

    Used by CurriculumUpdateCallback via trainer.workers.foreach_worker(foreach_env(...)).
    """
    import gym

    cur = env
    visited = 0
    while cur is not None and visited < 32:
        visited += 1
        # Check both .opponent_policy and TeamVsPolicyWrapper-style storage
        op_policy = getattr(cur, "opponent_policy", None)
        if op_policy is not None and hasattr(op_policy, "set_pool_weights"):
            op_policy.set_pool_weights(new_weights)
            return True
        # Walk to next wrapped env
        cur = getattr(cur, "env", None)
        if not isinstance(cur, gym.Wrapper) and not hasattr(cur, "env"):
            break
    return False
