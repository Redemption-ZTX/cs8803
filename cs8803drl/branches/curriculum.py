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
