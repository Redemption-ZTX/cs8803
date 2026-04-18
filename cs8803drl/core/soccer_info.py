"""Helpers for parsing Soccer-Twos info payloads and reward shaping signals.

This module is intentionally dependency-light so we can validate parsing and
shaping logic without requiring the Unity environment or RLlib runtime.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple


TEAM0_AGENT_IDS = (0, 1)
TEAM1_AGENT_IDS = (2, 3)
BALL_KEYS = ("ball_info", "ball", "ball_state", "ball_position")
PLAYER_KEYS = ("player_info", "player", "player_state")
SCORE_KEYS = ("score", "scores", "team_score", "team_scores", "goal", "goals")
WINNER_KEYS = ("winner", "winning_team", "result", "game_result", "outcome")


def _coerce_agent_id(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _coerce_xy(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, dict):
        if "position" in value:
            return _coerce_xy(value["position"])
        if "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"])
            except Exception:
                return None
        return None

    if not isinstance(value, (str, bytes, bytearray)):
        try:
            if len(value) >= 2:
                return float(value[0]), float(value[1])
        except Exception:
            pass

    return None


def _iter_dicts(obj: Any, *, max_depth: int = 5) -> Iterable[Dict[str, Any]]:
    if max_depth < 0:
        return
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value, max_depth=max_depth - 1)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_dicts(value, max_depth=max_depth - 1)


def extract_ball_position(info: Any) -> Optional[Tuple[float, float]]:
    for mapping in _iter_dicts(info):
        for key in BALL_KEYS:
            if key not in mapping:
                continue
            pos = _coerce_xy(mapping.get(key))
            if pos is not None:
                return pos
    return None


def extract_player_positions(info: Any) -> Dict[int, Tuple[float, float]]:
    positions: Dict[int, Tuple[float, float]] = {}

    if isinstance(info, dict):
        for key in PLAYER_KEYS:
            if key in info:
                pos = _coerce_xy(info.get(key))
                if pos is not None:
                    # single_player=True commonly exposes only the controlled player
                    positions[0] = pos
                    return positions

    if isinstance(info, dict):
        for agent_id, payload in info.items():
            coerced = _coerce_agent_id(agent_id)
            if coerced is None or not isinstance(payload, dict):
                continue
            for key in PLAYER_KEYS:
                if key not in payload:
                    continue
                pos = _coerce_xy(payload.get(key))
                if pos is not None:
                    positions[coerced] = pos
                    break

    if positions:
        return positions

    for mapping in _iter_dicts(info):
        agent_id = _coerce_agent_id(mapping.get("agent_id"))
        if agent_id is None:
            continue
        for key in PLAYER_KEYS:
            if key not in mapping:
                continue
            pos = _coerce_xy(mapping.get(key))
            if pos is not None:
                positions[agent_id] = pos
                break

    return positions


def extract_score_from_info(info: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(info, dict):
        return None

    for key in SCORE_KEYS:
        if key not in info:
            continue
        value = info.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except Exception:
                return None
        if isinstance(value, dict):
            for left, right in (
                ("team0", "team1"),
                ("blue", "orange"),
                ("home", "away"),
                ("left", "right"),
            ):
                if left in value and right in value:
                    try:
                        return float(value[left]), float(value[right])
                    except Exception:
                        return None

    for left, right in (
        ("team0_score", "team1_score"),
        ("blue_score", "orange_score"),
        ("home_score", "away_score"),
    ):
        if left in info and right in info:
            try:
                return float(info[left]), float(info[right])
            except Exception:
                return None

    return None


def extract_winner_from_info(info: Any) -> Optional[int]:
    if not isinstance(info, dict):
        return None

    for key in WINNER_KEYS:
        if key not in info:
            continue
        value = info.get(key)
        if isinstance(value, str):
            lower = value.lower()
            if any(token in lower for token in ("team0", "blue", "home", "left")):
                return 0
            if any(token in lower for token in ("team1", "orange", "away", "right")):
                return 1
        if isinstance(value, (int, float)):
            winner = int(value)
            if winner in (0, 1):
                return winner

    return None


def infer_terminal_winner(info: Any) -> Optional[int]:
    winner = extract_winner_from_info(info)
    if winner is not None:
        return winner

    score = extract_score_from_info(info)
    if score is None:
        return None

    team0_score, team1_score = score
    if team0_score > team1_score:
        return 0
    if team1_score > team0_score:
        return 1
    return None


def compute_shaping_components(
    info: Any,
    prev_ball_x: Optional[float],
    *,
    prev_ball_pos: Optional[Tuple[float, float]] = None,
    ball_progress_scale: float,
    goal_proximity_scale: float = 0.0,
    goal_proximity_gamma: float = 0.99,
    goal_center_x: float = 15.0,
    goal_center_y: float = 0.0,
    opponent_progress_penalty_scale: float,
    possession_dist: float,
    possession_bonus: float,
    progress_requires_possession: bool = False,
    deep_zone_outer_threshold: float = 0.0,
    deep_zone_outer_penalty: float = 0.0,
    deep_zone_inner_threshold: float = 0.0,
    deep_zone_inner_penalty: float = 0.0,
    defensive_survival_threshold: float = 0.0,
    defensive_survival_bonus: float = 0.0,
    ball_progress_scale_by_agent: Optional[Dict[int, float]] = None,
    opponent_progress_penalty_scale_by_agent: Optional[Dict[int, float]] = None,
    possession_bonus_by_agent: Optional[Dict[int, float]] = None,
    deep_zone_outer_penalty_by_agent: Optional[Dict[int, float]] = None,
    deep_zone_inner_penalty_by_agent: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """Return per-agent shaping rewards plus lightweight debug metadata."""
    add: Dict[int, float] = {}
    ball_pos = extract_ball_position(info)
    player_positions = extract_player_positions(info)

    debug: Dict[str, Any] = {
        "ball_found": ball_pos is not None,
        "ball_pos": ball_pos,
        "player_count": len(player_positions),
        "prev_ball_x": prev_ball_x,
        "ball_dx": None,
        "prev_ball_pos": prev_ball_pos,
        "closest_agent_id": None,
        "closest_agent_distance": None,
        "possessing_team": None,
        "progress_toward_goal": None,
        "progress_reward_gated": bool(progress_requires_possession),
        "progress_reward_applied": False,
        "goal_proximity_reward_applied": False,
        "goal_proximity_team0_delta": None,
        "goal_proximity_team1_delta": None,
        "deep_zone_team0_outer": False,
        "deep_zone_team0_inner": False,
        "deep_zone_team1_outer": False,
        "deep_zone_team1_inner": False,
        "defensive_survival_team0_active": False,
        "defensive_survival_team1_active": False,
    }

    if ball_pos is None:
        return add, debug

    ball_x, ball_y = ball_pos
    dx = None if prev_ball_x is None else float(ball_x - prev_ball_x)
    debug["ball_dx"] = dx

    if prev_ball_pos is not None and goal_proximity_scale != 0.0:
        try:
            prev_x, prev_y = float(prev_ball_pos[0]), float(prev_ball_pos[1])
            curr_x, curr_y = float(ball_x), float(ball_y)
            team0_goal = (float(goal_center_x), float(goal_center_y))
            team1_goal = (-float(goal_center_x), float(goal_center_y))

            phi0_prev = -math.hypot(prev_x - team0_goal[0], prev_y - team0_goal[1])
            phi0_curr = -math.hypot(curr_x - team0_goal[0], curr_y - team0_goal[1])
            phi1_prev = -math.hypot(prev_x - team1_goal[0], prev_y - team1_goal[1])
            phi1_curr = -math.hypot(curr_x - team1_goal[0], curr_y - team1_goal[1])

            team0_delta = float(goal_proximity_scale) * (
                float(goal_proximity_gamma) * phi0_curr - phi0_prev
            )
            team1_delta = float(goal_proximity_scale) * (
                float(goal_proximity_gamma) * phi1_curr - phi1_prev
            )
            debug["goal_proximity_reward_applied"] = True
            debug["goal_proximity_team0_delta"] = team0_delta
            debug["goal_proximity_team1_delta"] = team1_delta

            for agent_id in TEAM0_AGENT_IDS:
                add[agent_id] = add.get(agent_id, 0.0) + team0_delta
            for agent_id in TEAM1_AGENT_IDS:
                add[agent_id] = add.get(agent_id, 0.0) + team1_delta
        except Exception:
            pass

    closest_agent_id = None
    closest_distance = None
    possessing_team = None
    progress_toward_goal = None
    possession_confirmed = False
    if dx is not None and player_positions:
        for agent_id, (player_x, player_y) in player_positions.items():
            distance = math.hypot(player_x - ball_x, player_y - ball_y)
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest_agent_id = agent_id

        if closest_agent_id is not None:
            possessing_team = 0 if closest_agent_id in TEAM0_AGENT_IDS else 1
            progress_toward_goal = max(dx, 0.0) if possessing_team == 0 else max(-dx, 0.0)
            possession_confirmed = (
                closest_distance is not None
                and float(closest_distance) <= float(possession_dist)
            )

    debug["closest_agent_id"] = closest_agent_id
    debug["closest_agent_distance"] = (
        None if closest_distance is None else float(closest_distance)
    )
    debug["possessing_team"] = possessing_team
    debug["progress_toward_goal"] = progress_toward_goal
    debug["possession_confirmed"] = bool(possession_confirmed)

    has_ball_progress_overrides = bool(
        ball_progress_scale_by_agent
        and any(float(v) != 0.0 for v in ball_progress_scale_by_agent.values())
    )
    apply_progress_reward = (
        dx is not None
        and (ball_progress_scale != 0 or has_ball_progress_overrides)
        and (
            not progress_requires_possession
            or (possessing_team is not None and possession_confirmed)
        )
    )
    if apply_progress_reward:
        for agent_id in TEAM0_AGENT_IDS:
            scale = float(
                ball_progress_scale_by_agent.get(agent_id, ball_progress_scale)
            ) if ball_progress_scale_by_agent is not None else float(ball_progress_scale)
            add[agent_id] = add.get(agent_id, 0.0) + scale * dx
        for agent_id in TEAM1_AGENT_IDS:
            scale = float(
                ball_progress_scale_by_agent.get(agent_id, ball_progress_scale)
            ) if ball_progress_scale_by_agent is not None else float(ball_progress_scale)
            add[agent_id] = add.get(agent_id, 0.0) - scale * dx
        debug["progress_reward_applied"] = True

    has_opp_progress_overrides = bool(
        opponent_progress_penalty_scale_by_agent
        and any(float(v) != 0.0 for v in opponent_progress_penalty_scale_by_agent.values())
    )
    if (
        dx is not None
        and possessing_team is not None
        and progress_toward_goal is not None
        and progress_toward_goal > 0
        and (opponent_progress_penalty_scale != 0 or has_opp_progress_overrides)
    ):
        defending_team = 1 - possessing_team
        target_agents = TEAM0_AGENT_IDS if defending_team == 0 else TEAM1_AGENT_IDS
        for agent_id in target_agents:
            scale = float(
                opponent_progress_penalty_scale_by_agent.get(
                    agent_id, opponent_progress_penalty_scale
                )
            ) if opponent_progress_penalty_scale_by_agent is not None else float(
                opponent_progress_penalty_scale
            )
            penalty = -scale * progress_toward_goal
            add[agent_id] = add.get(agent_id, 0.0) + penalty

    has_possession_overrides = bool(
        possession_bonus_by_agent
        and any(float(v) != 0.0 for v in possession_bonus_by_agent.values())
    )
    if (possession_bonus != 0 or has_possession_overrides) and player_positions:
        for agent_id, (player_x, player_y) in player_positions.items():
            distance = math.hypot(player_x - ball_x, player_y - ball_y)
            if distance <= possession_dist:
                bonus = float(
                    possession_bonus_by_agent.get(agent_id, possession_bonus)
                ) if possession_bonus_by_agent is not None else float(possession_bonus)
                add[agent_id] = add.get(agent_id, 0.0) + bonus

    survival_threshold = float(defensive_survival_threshold)
    survival_bonus = float(defensive_survival_bonus)
    if survival_bonus != 0.0 and possessing_team is not None and possession_confirmed:
        if survival_threshold < 0.0:
            team0_under_pressure = ball_x < survival_threshold
            team1_under_pressure = ball_x > abs(survival_threshold)
        else:
            team0_under_pressure = ball_x < survival_threshold
            team1_under_pressure = ball_x > survival_threshold

        if possessing_team == 1 and team0_under_pressure:
            debug["defensive_survival_team0_active"] = True
            per_agent_bonus = abs(survival_bonus) / float(len(TEAM0_AGENT_IDS))
            for agent_id in TEAM0_AGENT_IDS:
                add[agent_id] = add.get(agent_id, 0.0) + per_agent_bonus

        if possessing_team == 0 and team1_under_pressure:
            debug["defensive_survival_team1_active"] = True
            per_agent_bonus = abs(survival_bonus) / float(len(TEAM1_AGENT_IDS))
            for agent_id in TEAM1_AGENT_IDS:
                add[agent_id] = add.get(agent_id, 0.0) + per_agent_bonus

    outer_threshold = float(deep_zone_outer_threshold)
    outer_penalty = float(deep_zone_outer_penalty)
    inner_threshold = float(deep_zone_inner_threshold)
    inner_penalty = float(deep_zone_inner_penalty)
    has_outer_overrides = bool(
        deep_zone_outer_penalty_by_agent
        and any(float(v) != 0.0 for v in deep_zone_outer_penalty_by_agent.values())
    )
    has_inner_overrides = bool(
        deep_zone_inner_penalty_by_agent
        and any(float(v) != 0.0 for v in deep_zone_inner_penalty_by_agent.values())
    )
    if outer_penalty != 0.0 or inner_penalty != 0.0 or has_outer_overrides or has_inner_overrides:
        # Team0 defends the negative-x goal; team1 defends the positive-x goal.
        if (outer_penalty != 0.0 or has_outer_overrides) and outer_threshold < 0.0 and ball_x < outer_threshold:
            debug["deep_zone_team0_outer"] = True
            for agent_id in TEAM0_AGENT_IDS:
                penalty = float(
                    deep_zone_outer_penalty_by_agent.get(agent_id, outer_penalty)
                ) if deep_zone_outer_penalty_by_agent is not None else float(outer_penalty)
                add[agent_id] = add.get(agent_id, 0.0) - abs(penalty)
        if (inner_penalty != 0.0 or has_inner_overrides) and inner_threshold < 0.0 and ball_x < inner_threshold:
            debug["deep_zone_team0_inner"] = True
            for agent_id in TEAM0_AGENT_IDS:
                penalty = float(
                    deep_zone_inner_penalty_by_agent.get(agent_id, inner_penalty)
                ) if deep_zone_inner_penalty_by_agent is not None else float(inner_penalty)
                add[agent_id] = add.get(agent_id, 0.0) - abs(penalty)

        mirror_outer = abs(outer_threshold)
        mirror_inner = abs(inner_threshold)
        if (outer_penalty != 0.0 or has_outer_overrides) and mirror_outer > 0.0 and ball_x > mirror_outer:
            debug["deep_zone_team1_outer"] = True
            for agent_id in TEAM1_AGENT_IDS:
                penalty = float(
                    deep_zone_outer_penalty_by_agent.get(agent_id, outer_penalty)
                ) if deep_zone_outer_penalty_by_agent is not None else float(outer_penalty)
                add[agent_id] = add.get(agent_id, 0.0) - abs(penalty)
        if (inner_penalty != 0.0 or has_inner_overrides) and mirror_inner > 0.0 and ball_x > mirror_inner:
            debug["deep_zone_team1_inner"] = True
            for agent_id in TEAM1_AGENT_IDS:
                penalty = float(
                    deep_zone_inner_penalty_by_agent.get(agent_id, inner_penalty)
                ) if deep_zone_inner_penalty_by_agent is not None else float(inner_penalty)
                add[agent_id] = add.get(agent_id, 0.0) - abs(penalty)

    return add, debug


def compute_terminal_shaping(
    info: Any,
    *,
    episode_steps: int,
    fast_loss_threshold_steps: int = 0,
    fast_loss_penalty_per_step: float = 0.0,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    add: Dict[int, float] = {}
    debug: Dict[str, Any] = {
        "terminal_winner": None,
        "fast_loss_penalty_team0": 0.0,
        "fast_loss_penalty_team1": 0.0,
    }

    threshold_steps = int(fast_loss_threshold_steps)
    penalty_per_step = float(fast_loss_penalty_per_step)
    if threshold_steps <= 0 or penalty_per_step <= 0.0:
        return add, debug

    winner = infer_terminal_winner(info)
    debug["terminal_winner"] = winner
    if winner not in (0, 1):
        return add, debug

    steps = int(episode_steps)
    if steps >= threshold_steps:
        return add, debug

    shortfall = float(threshold_steps - steps)
    team_penalty = abs(penalty_per_step) * shortfall
    if winner == 0:
        per_agent_penalty = team_penalty / float(len(TEAM1_AGENT_IDS))
        debug["fast_loss_penalty_team1"] = team_penalty
        for agent_id in TEAM1_AGENT_IDS:
            add[agent_id] = add.get(agent_id, 0.0) - per_agent_penalty
    else:
        per_agent_penalty = team_penalty / float(len(TEAM0_AGENT_IDS))
        debug["fast_loss_penalty_team0"] = team_penalty
        for agent_id in TEAM0_AGENT_IDS:
            add[agent_id] = add.get(agent_id, 0.0) - per_agent_penalty

    return add, debug


def compute_event_shaping(
    *,
    episode_steps: int,
    ball_x: Optional[float],
    prev_ball_x: Optional[float],
    ball_dx: Optional[float],
    possessing_team: Optional[int],
    prev_possessing_team: Optional[int],
    possession_confirmed: bool,
    event_shot_reward: float = 0.0,
    event_tackle_reward: float = 0.0,
    event_clearance_reward: float = 0.0,
    event_cooldown_steps: int = 10,
    shot_x_threshold: float = 10.0,
    shot_ball_dx_min: float = 0.5,
    clearance_from_x: float = -8.0,
    clearance_to_x: float = -4.0,
    last_trigger_steps: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[int, float], Dict[str, Any], Dict[str, int]]:
    add: Dict[int, float] = {}
    debug: Dict[str, Any] = {
        "event_shot_team0": False,
        "event_shot_team1": False,
        "event_tackle_team0": False,
        "event_tackle_team1": False,
        "event_clearance_team0": False,
        "event_clearance_team1": False,
    }
    updated_triggers: Dict[str, int] = {}

    cooldown_steps = max(int(event_cooldown_steps), 0)
    trigger_steps = dict(last_trigger_steps or {})

    def _cooldown_ready(key: str) -> bool:
        if cooldown_steps <= 0:
            return True
        last_step = int(trigger_steps.get(key, -10**9))
        return int(episode_steps) - last_step >= cooldown_steps

    def _trigger_team(event_key: str, team_id: int, reward: float) -> None:
        if reward == 0.0:
            return
        key = f"{event_key}_team{team_id}"
        if not _cooldown_ready(key):
            return
        target_agents = TEAM0_AGENT_IDS if team_id == 0 else TEAM1_AGENT_IDS
        for agent_id in target_agents:
            add[agent_id] = add.get(agent_id, 0.0) + float(reward)
        debug[key] = True
        updated_triggers[key] = int(episode_steps)

    # shot_on_goal surrogate: ball already in attacking third and still moving toward goal.
    if ball_x is not None and ball_dx is not None:
        if float(ball_x) > float(shot_x_threshold) and float(ball_dx) > float(shot_ball_dx_min):
            _trigger_team("event_shot", 0, float(event_shot_reward))
        if float(ball_x) < -abs(float(shot_x_threshold)) and float(ball_dx) < -abs(float(shot_ball_dx_min)):
            _trigger_team("event_shot", 1, float(event_shot_reward))

    # tackle: confirmed possession flips from opponent to us.
    if possession_confirmed and possessing_team in (0, 1) and prev_possessing_team in (0, 1):
        if int(possessing_team) != int(prev_possessing_team):
            _trigger_team("event_tackle", int(possessing_team), float(event_tackle_reward))

    # clearance: ball leaves deep defensive zone into safer territory.
    if ball_x is not None and prev_ball_x is not None:
        team0_from = float(clearance_from_x)
        team0_to = float(clearance_to_x)
        team1_from = abs(team0_from)
        team1_to = abs(team0_to)
        if float(prev_ball_x) < team0_from and float(ball_x) > team0_to:
            _trigger_team("event_clearance", 0, float(event_clearance_reward))
        if float(prev_ball_x) > team1_from and float(ball_x) < team1_to:
            _trigger_team("event_clearance", 1, float(event_clearance_reward))

    return add, debug, updated_triggers


def aggregate_scalar_shaping(add_by_agent: Dict[int, float], *, controlled_agent_ids: Iterable[int]) -> float:
    total = 0.0
    for agent_id in controlled_agent_ids:
        total += float(add_by_agent.get(int(agent_id), 0.0))
    return total
