from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from cs8803drl.core.soccer_info import (
    TEAM0_AGENT_IDS,
    TEAM1_AGENT_IDS,
    extract_ball_position,
    extract_player_positions,
    extract_score_from_info,
    extract_winner_from_info,
)


def _to_float_pair(value: Optional[Tuple[float, float]]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(value[0]), float(value[1])]


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("._") or "episode"


def _mean(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values]
    if not values:
        return None
    return float(sum(values) / len(values))


def _team_centroid(
    positions: Dict[int, Tuple[float, float]], agent_ids: Iterable[int]
) -> Optional[List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for agent_id in agent_ids:
        if int(agent_id) not in positions:
            continue
        x, y = positions[int(agent_id)]
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        return None
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_json_safe(v) for v in value]
    try:
        return float(value)
    except Exception:
        return str(value)


def infer_team0_outcome(
    *,
    final_info: Any,
    final_team0_step_reward: float,
    final_team1_step_reward: float,
    cumulative_team0_reward: float,
    cumulative_team1_reward: float,
) -> Tuple[str, Optional[List[float]], Optional[int]]:
    score = extract_score_from_info(final_info)
    winner = extract_winner_from_info(final_info)

    if score is not None:
        score_list = [float(score[0]), float(score[1])]
        if score[0] > score[1]:
            return "team0_win", score_list, winner
        if score[1] > score[0]:
            return "team1_win", score_list, winner
        return "tie", score_list, winner

    if winner is not None:
        return ("team0_win" if int(winner) == 0 else "team1_win"), None, int(winner)

    if final_team0_step_reward > final_team1_step_reward:
        return "team0_win", None, None
    if final_team1_step_reward > final_team0_step_reward:
        return "team1_win", None, None

    if cumulative_team0_reward > cumulative_team1_reward:
        return "team0_win", None, None
    if cumulative_team1_reward > cumulative_team0_reward:
        return "team1_win", None, None

    return "tie", None, None


def classify_failure(summary: Dict[str, Any]) -> Dict[str, Any]:
    outcome = summary.get("outcome")
    if outcome == "team0_win":
        return {"primary_label": "win", "labels": ["win"]}
    if outcome == "tie":
        return {"primary_label": "tie", "labels": ["tie"]}

    metrics = summary.get("metrics", {}) or {}
    labels: List[str] = []

    mean_ball_x = metrics.get("mean_ball_x")
    tail_mean_ball_x = metrics.get("tail_mean_ball_x")
    team0_possession_ratio = metrics.get("team0_possession_ratio")
    team0_progress = float(metrics.get("team0_progress_toward_goal", 0.0) or 0.0)
    team1_progress = float(metrics.get("team1_progress_toward_goal", 0.0) or 0.0)

    if tail_mean_ball_x is not None and float(tail_mean_ball_x) < -0.45:
        labels.append("late_defensive_collapse")
    if mean_ball_x is not None and float(mean_ball_x) < -0.15:
        labels.append("territory_loss")
    if team0_possession_ratio is not None and float(team0_possession_ratio) < 0.35:
        labels.append("low_possession")
    if team0_possession_ratio is not None and float(team0_possession_ratio) > 0.55:
        labels.append("poor_conversion")
    if team1_progress > max(team0_progress * 1.35, 0.25):
        labels.append("opponent_forward_progress")

    if not labels:
        labels.append("unclear_loss")

    return {"primary_label": labels[0], "labels": labels}


class EpisodeFailureRecorder:
    def __init__(self, *, trace_stride: int = 5, tail_steps: int = 40):
        self._trace_stride = max(int(trace_stride), 1)
        self._tail_steps = max(int(tail_steps), 1)
        self._sampled_steps: List[Dict[str, Any]] = []
        self._tail_steps_buf: Deque[Dict[str, Any]] = deque(maxlen=self._tail_steps)
        self._ball_x_values: List[float] = []
        self._possession_counts = {0: 0, 1: 0}
        self._progress_by_team = {0: 0.0, 1: 0.0}
        self._shaping_reward_by_team = {0: 0.0, 1: 0.0}

    def record_step(
        self,
        *,
        step_index: int,
        reward: Any,
        info: Any,
        team0_ids: Iterable[int],
        team1_ids: Iterable[int],
    ) -> None:
        ball_pos = extract_ball_position(info)
        player_positions = extract_player_positions(info)
        score = extract_score_from_info(info)
        winner = extract_winner_from_info(info)
        shaping = info.get("_reward_shaping") if isinstance(info, dict) else None

        step_team0_reward = _team_reward(reward, team0_ids)
        step_team1_reward = _team_reward(reward, team1_ids)

        if ball_pos is not None:
            self._ball_x_values.append(float(ball_pos[0]))

        if isinstance(shaping, dict):
            possessing_team = shaping.get("possessing_team")
            if possessing_team in (0, 1):
                self._possession_counts[int(possessing_team)] += 1
                progress = float(shaping.get("progress_toward_goal", 0.0) or 0.0)
                self._progress_by_team[int(possessing_team)] += progress

            applied_reward = shaping.get("applied_reward")
            if isinstance(applied_reward, dict):
                for agent_id, delta in applied_reward.items():
                    try:
                        aid = int(agent_id)
                        val = float(delta)
                    except Exception:
                        continue
                    if aid in TEAM0_AGENT_IDS:
                        self._shaping_reward_by_team[0] += val
                    elif aid in TEAM1_AGENT_IDS:
                        self._shaping_reward_by_team[1] += val

        snapshot = {
            "step": int(step_index),
            "team0_step_reward": float(step_team0_reward),
            "team1_step_reward": float(step_team1_reward),
            "score": _to_float_pair(score),
            "winner": None if winner is None else int(winner),
            "ball_pos": _to_float_pair(ball_pos),
            "team0_centroid": _team_centroid(player_positions, team0_ids),
            "team1_centroid": _team_centroid(player_positions, team1_ids),
            "reward_shaping": _json_safe(shaping),
        }

        if step_index % self._trace_stride == 0:
            self._sampled_steps.append(snapshot)
        self._tail_steps_buf.append(snapshot)

    def build_episode_record(
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
        tail_steps = list(self._tail_steps_buf)
        tail_ball_x = [
            float(step["ball_pos"][0])
            for step in tail_steps
            if isinstance(step.get("ball_pos"), list) and len(step["ball_pos"]) >= 1
        ]
        total_possession = self._possession_counts[0] + self._possession_counts[1]
        metrics = {
            "mean_ball_x": _mean(self._ball_x_values),
            "tail_mean_ball_x": _mean(tail_ball_x),
            "team0_possession_ratio": (
                float(self._possession_counts[0] / total_possession)
                if total_possession > 0
                else None
            ),
            "team1_possession_ratio": (
                float(self._possession_counts[1] / total_possession)
                if total_possession > 0
                else None
            ),
            "team0_progress_toward_goal": float(self._progress_by_team[0]),
            "team1_progress_toward_goal": float(self._progress_by_team[1]),
            "team0_shaping_reward_sum": float(self._shaping_reward_by_team[0]),
            "team1_shaping_reward_sum": float(self._shaping_reward_by_team[1]),
        }

        record = {
            "episode_index": int(episode_index),
            "team0_module": str(team0_module),
            "team1_module": str(team1_module),
            "outcome": str(outcome),
            "final_score": _json_safe(final_score),
            "final_winner": None if final_winner is None else int(final_winner),
            "steps": int(total_steps),
            "cumulative_team0_reward": float(cumulative_team0_reward),
            "cumulative_team1_reward": float(cumulative_team1_reward),
            "final_team0_step_reward": float(final_team0_step_reward),
            "final_team1_step_reward": float(final_team1_step_reward),
            "metrics": metrics,
            "sampled_trace": self._sampled_steps,
            "tail_trace": tail_steps,
        }
        record["classification"] = classify_failure(record)
        return record


def save_episode_record(record: Dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classification = record.get("classification", {}) or {}
    primary = _safe_name(classification.get("primary_label", "episode"))
    outcome = _safe_name(record.get("outcome", "episode"))
    episode_index = int(record.get("episode_index", 0))
    path = out_dir / f"episode_{episode_index:04d}_{outcome}_{primary}.json"
    path.write_text(json.dumps(_json_safe(record), indent=2, sort_keys=True))
    return path


def _team_reward(reward: Any, team_ids: Iterable[int]) -> float:
    total = 0.0
    if isinstance(reward, dict):
        for team_id in team_ids:
            if int(team_id) in reward:
                total += float(reward[int(team_id)])
        return total

    try:
        for team_id in team_ids:
            total += float(reward[int(team_id)])
    except Exception:
        pass
    return total
