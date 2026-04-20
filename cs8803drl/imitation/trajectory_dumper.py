"""Trajectory dumper for snapshot-036 learned reward shaping (Path C).

Records full per-step `(obs, action, reward, info)` for one episode plus
the failure-bucket multi-label set computed by ``classify_failure``.
Persists each episode as one ``.npz`` (arrays) plus one ``.meta.json``
(scalar metadata) so a downstream reward model trainer can:

  - load arrays cheaply with ``numpy.load``
  - filter trajectories by primary_label / labels / outcome
  - apply per-step γ-decayed labels (W/L contrastive)

Used by ``scripts/eval/dump_trajectories.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from cs8803drl.evaluation.failure_cases import (
    EpisodeFailureRecorder,
    classify_failure,
)
from cs8803drl.core.soccer_info import extract_ball_position


# Soccer-Twos per-agent action space = MultiDiscrete([3, 3, 3])
_MULTIDISCRETE_NDIM = 3
_MULTIDISCRETE_BASE = 3


def _action_to_multidiscrete(action: Any) -> np.ndarray:
    """Coerce an agent's per-agent action into a MultiDiscrete int8 array of shape (3,).

    Storing the canonical MultiDiscrete representation (instead of a flat int)
    sidesteps the encoding-mismatch hazard between custom encodings and
    ``gym_unity.ActionFlattener`` used by RLlib internally.

    If the input is already a flat int / Discrete int, we unflatten using the
    project canonical encoding (little-endian base-3, matching
    ``_unflatten_discrete_to_multidiscrete`` in trained_shared_cc_agent.py).
    """
    if isinstance(action, (int, np.integer)):
        flat = int(action)
        out = np.zeros(_MULTIDISCRETE_NDIM, dtype=np.int8)
        for i in range(_MULTIDISCRETE_NDIM):
            out[i] = flat % _MULTIDISCRETE_BASE
            flat //= _MULTIDISCRETE_BASE
        return out
    if isinstance(action, (list, tuple)):
        action = np.asarray(action)
    if isinstance(action, np.ndarray):
        if action.ndim == 0:
            return _action_to_multidiscrete(int(action))
        arr = action.flatten()[:_MULTIDISCRETE_NDIM].astype(np.int8)
        if arr.size < _MULTIDISCRETE_NDIM:
            padded = np.zeros(_MULTIDISCRETE_NDIM, dtype=np.int8)
            padded[: arr.size] = arr
            return padded
        return arr
    try:
        return _action_to_multidiscrete(int(action))
    except Exception:
        return np.zeros(_MULTIDISCRETE_NDIM, dtype=np.int8)


class TrajectoryRecorder:
    """Per-episode recorder that combines full trajectory dump
    with failure-bucket metric tracking.

    Usage::

        recorder = TrajectoryRecorder()
        for step in episode:
            obs0 = ...                # before env.step()
            action = agent.act(obs0)  # before env.step()
            obs, reward, done, info = env.step(action)
            recorder.record_step(
                step_index=step, ...
                obs_a0=pre_step_obs[0], obs_a1=pre_step_obs[1],
                act_a0=action_int_0, act_a1=action_int_1,
                reward=reward, info=info,
                team0_ids=team0_ids, team1_ids=team1_ids,
            )

        traj = recorder.build_trajectory(episode_index=ep, outcome=outcome, ...)
        save_trajectory(traj, save_dir, filename_hint="run_label")
    """

    def __init__(self, *, trace_stride: int = 5, tail_steps: int = 40):
        self._failure_recorder = EpisodeFailureRecorder(
            trace_stride=trace_stride, tail_steps=tail_steps
        )
        self._steps: List[Dict[str, Any]] = []

    def record_step(
        self,
        *,
        step_index: int,
        obs_a0: np.ndarray,
        obs_a1: np.ndarray,
        act_a0: Any,
        act_a1: Any,
        reward: Any,
        info: Any,
        team0_ids: Iterable[int],
        team1_ids: Iterable[int],
    ) -> None:
        # explicit .copy() defends against env reusing internal buffers
        ball_pos = extract_ball_position(info)
        ball_xy = (
            np.asarray(ball_pos, dtype=np.float32)
            if ball_pos is not None
            else np.array([np.nan, np.nan], dtype=np.float32)
        )
        self._steps.append(
            {
                "step": int(step_index),
                "obs_a0": np.asarray(obs_a0, dtype=np.float32).reshape(-1).copy(),
                "obs_a1": np.asarray(obs_a1, dtype=np.float32).reshape(-1).copy(),
                "act_a0": _action_to_multidiscrete(act_a0),
                "act_a1": _action_to_multidiscrete(act_a1),
                "ball_xy": ball_xy,
            }
        )
        # forward to failure recorder so we get classify_failure() labels later
        self._failure_recorder.record_step(
            step_index=step_index,
            reward=reward,
            info=info,
            team0_ids=team0_ids,
            team1_ids=team1_ids,
        )

    def build_trajectory(
        self,
        *,
        episode_index: int,
        team0_module: str,
        team1_module: str,
        outcome: str,
        final_score: Optional[List[float]],
        final_winner: Optional[int],
        cumulative_team0_reward: float,
        cumulative_team1_reward: float,
        final_team0_step_reward: float,
        final_team1_step_reward: float,
        total_steps: int,
    ) -> Dict[str, Any]:
        # use EpisodeFailureRecorder to compute metrics + classify_failure
        failure_record = self._failure_recorder.build_episode_record(
            episode_index=episode_index,
            team0_module=team0_module,
            team1_module=team1_module,
            outcome=outcome,
            final_score=final_score,
            final_winner=final_winner,
            cumulative_team0_reward=cumulative_team0_reward,
            cumulative_team1_reward=cumulative_team1_reward,
            final_team0_step_reward=final_team0_step_reward,
            final_team1_step_reward=final_team1_step_reward,
            total_steps=total_steps,
        )

        classification = classify_failure(failure_record)

        T = len(self._steps)
        if T == 0:
            obs_a0 = np.zeros((0, 0), dtype=np.float32)
            obs_a1 = np.zeros((0, 0), dtype=np.float32)
            act_a0 = np.zeros((0, _MULTIDISCRETE_NDIM), dtype=np.int8)
            act_a1 = np.zeros((0, _MULTIDISCRETE_NDIM), dtype=np.int8)
            ball_xy = np.zeros((0, 2), dtype=np.float32)
        else:
            obs_a0 = np.stack([s["obs_a0"] for s in self._steps])
            obs_a1 = np.stack([s["obs_a1"] for s in self._steps])
            act_a0 = np.stack([s["act_a0"] for s in self._steps])  # (T, 3)
            act_a1 = np.stack([s["act_a1"] for s in self._steps])  # (T, 3)
            ball_xy = np.stack([s["ball_xy"] for s in self._steps])  # (T, 2)

        return {
            # episode metadata
            "episode_index": int(episode_index),
            "team0_module": team0_module,
            "team1_module": team1_module,
            "outcome": outcome,
            "primary_label": classification.get("primary_label"),
            "labels": classification.get("labels", []),
            "metrics": failure_record.get("metrics", {}),
            "final_score": final_score,
            "final_winner": final_winner,
            "cumulative_team0_reward": float(cumulative_team0_reward),
            "cumulative_team1_reward": float(cumulative_team1_reward),
            "total_steps": int(total_steps),
            # full trajectory arrays:
            #   obs_a0 / obs_a1: (T, 336) float32 — pre-step egocentric obs per agent
            #   act_a0 / act_a1: (T, 3) int8 — MultiDiscrete([3,3,3]) per-agent action
            #     (storing MultiDiscrete avoids any encoding mismatch with
            #      gym_unity.ActionFlattener at deploy / reward-query time)
            "obs_a0": obs_a0,
            "obs_a1": obs_a1,
            "act_a0": act_a0,
            "act_a1": act_a1,
            "ball_xy": ball_xy,  # (T, 2) raw ball world coords from info dict
        }


def save_trajectory(
    record: Dict[str, Any],
    save_dir: Path,
    filename_hint: str = "",
) -> Path:
    """Persist one trajectory: arrays in .npz (compressed), metadata in .meta.json.

    Filename pattern: ``{hint_}ep{index:05d}_{outcome}_{primary_label}.npz``

    Returns the path to the .npz file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ep = record["episode_index"]
    outcome = record["outcome"]
    primary = record.get("primary_label") or "unknown"
    hint_part = f"{filename_hint}_" if filename_hint else ""
    base = f"{hint_part}ep{ep:05d}_{outcome}_{primary}"
    npz_path = save_dir / f"{base}.npz"
    json_path = save_dir / f"{base}.meta.json"

    np.savez_compressed(
        str(npz_path),
        obs_a0=record["obs_a0"],
        obs_a1=record["obs_a1"],
        act_a0=record["act_a0"],
        act_a1=record["act_a1"],
        ball_xy=record.get("ball_xy", np.zeros((0, 2), dtype=np.float32)),
    )

    meta = {
        k: v
        for k, v in record.items()
        if k not in ("obs_a0", "obs_a1", "act_a0", "act_a1", "ball_xy")
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    return npz_path
