"""In-memory trajectory buffer for snapshot-039 adaptive reward refresh.

Collects per-episode `(obs_a0/a1, act_a0/a1, outcome, labels)` tuples online
via the Ray callback, in a format compatible with `learned_reward_trainer`'s
`_build_sample_table` — but with outcome labels derived from the runtime
episode.

The buffer is bounded (FIFO by insertion order) and partitions episodes
by W/L for easy balanced sampling when refreshing the discriminator.

Compact design to avoid Ray cross-worker serialization overhead:
- no full trace stored, only per-step (obs, action) arrays
- per-episode stored as np.float32 obs (T, 336) and np.int8 action (T, 3)
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


class EpisodeRecord:
    """One captured rollout episode."""
    __slots__ = ("obs_a0", "obs_a1", "act_a0", "act_a1", "outcome", "labels", "steps")

    def __init__(
        self,
        obs_a0: np.ndarray,
        obs_a1: np.ndarray,
        act_a0: np.ndarray,
        act_a1: np.ndarray,
        outcome: str,
        labels: List[str],
        steps: int,
    ):
        self.obs_a0 = obs_a0
        self.obs_a1 = obs_a1
        self.act_a0 = act_a0
        self.act_a1 = act_a1
        self.outcome = outcome
        self.labels = list(labels)
        self.steps = int(steps)


class TrajectoryBuffer:
    """Thread-safe bounded buffer of episodes, partitioned by W/L.

    `max_per_class` caps memory (default 200 W + 200 L ≈ 40 MB @ ~30 step eps).
    Oldest episodes are evicted first (FIFO within class).
    """

    def __init__(self, max_per_class: int = 200):
        self._max = int(max_per_class)
        self._w: Deque[EpisodeRecord] = deque(maxlen=self._max)
        self._l: Deque[EpisodeRecord] = deque(maxlen=self._max)
        self._lock = threading.Lock()

    def add(self, record: EpisodeRecord) -> None:
        with self._lock:
            if record.outcome == "team0_win":
                self._w.append(record)
            elif record.outcome == "team1_win":
                self._l.append(record)
            # ties silently discarded

    def sample_pairs(self, n_pairs: int, rng: Optional[np.random.Generator] = None) -> List[Tuple[EpisodeRecord, EpisodeRecord]]:
        """Return n_pairs of (W, L) episode tuples, sampled with replacement."""
        if rng is None:
            rng = np.random.default_rng()
        with self._lock:
            w_list = list(self._w)
            l_list = list(self._l)
        if not w_list or not l_list:
            return []
        wi = rng.integers(0, len(w_list), size=n_pairs)
        li = rng.integers(0, len(l_list), size=n_pairs)
        return [(w_list[int(i)], l_list[int(j)]) for i, j in zip(wi, li)]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {"W": len(self._w), "L": len(self._l)}

    def clear(self) -> None:
        with self._lock:
            self._w.clear()
            self._l.clear()

    def all_episodes(self) -> List[EpisodeRecord]:
        with self._lock:
            return list(self._w) + list(self._l)
