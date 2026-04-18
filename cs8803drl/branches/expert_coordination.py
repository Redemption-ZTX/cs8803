from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, Optional

import numpy as np

from cs8803drl.branches.obs_summary import RAY_BLOCK_SIZE, RAY_TYPE_DIM


ATTACK_SCENARIO = "attack_expert"
DEFENSE_SCENARIO = "defense_expert"

# Empirically identified from controlled side-channel probes:
# type 0 reacts strongly when the ball is placed directly in front of the agent.
BALL_TYPE_INDEX = 0


def _sample_uniform(range_or_value):
    if isinstance(range_or_value, (list, tuple)) and len(range_or_value) == 2:
        return random.uniform(float(range_or_value[0]), float(range_or_value[1]))
    return float(range_or_value)


def _sample_position(x_range, y_range):
    return [_sample_uniform(x_range), _sample_uniform(y_range)]


def sample_expert_scenario(mode: str) -> Dict[str, Any]:
    mode = (mode or "").strip().lower()
    if mode not in {ATTACK_SCENARIO, DEFENSE_SCENARIO}:
        raise ValueError(f"Unknown scenario mode: {mode!r}")

    if mode == ATTACK_SCENARIO:
        player0 = _sample_position((-12.5, -7.5), (-2.0, 2.0))
        player1 = _sample_position((-15.0, -10.5), (-2.5, 2.5))
        ball = [
            min(2.0, player0[0] + _sample_uniform((1.0, 4.0))),
            player0[1] + _sample_uniform((-1.0, 1.0)),
        ]
        opp2 = _sample_position((1.0, 7.0), (-2.5, 2.5))
        opp3 = _sample_position((4.0, 10.0), (-2.5, 2.5))
    else:
        # Put the controlled player under pressure near the home goal so that
        # "good" behavior means recovering and clearing the ball to the right.
        player0 = _sample_position((-15.5, -12.5), (-2.0, 2.0))
        player1 = _sample_position((-9.0, -4.0), (-2.5, 2.5))
        ball = [
            min(-8.0, player0[0] + _sample_uniform((0.5, 2.5))),
            player0[1] + _sample_uniform((-1.0, 1.0)),
        ]
        opp2 = [
            min(1.0, ball[0] + _sample_uniform((0.5, 2.5))),
            ball[1] + _sample_uniform((-1.0, 1.0)),
        ]
        opp3 = _sample_position((-2.0, 4.0), (-2.5, 2.5))

    return {
        "players_states": {
            0: {"position": player0, "rotation_y": 0.0},
            1: {"position": player1, "rotation_y": 0.0},
            2: {"position": opp2, "rotation_y": 180.0},
            3: {"position": opp3, "rotation_y": 180.0},
        },
        "ball_state": {
            "position": ball,
            "velocity": [0.0, 0.0],
        },
    }


def extract_ball_proxy(observation: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(observation, dtype=np.float32).reshape(-1)
    if arr.size == 0 or arr.size % RAY_BLOCK_SIZE != 0:
        return {
            "visible": 0.0,
            "count": 0.0,
            "nearest": 1.0,
            "centroid": 0.0,
            "score": 0.0,
        }

    blocks = arr.reshape(-1, RAY_BLOCK_SIZE)
    type_scores = blocks[:, :RAY_TYPE_DIM]
    distances = np.clip(blocks[:, RAY_TYPE_DIM], 0.0, 1.0)
    ray_positions = np.linspace(-1.0, 1.0, blocks.shape[0], dtype=np.float32)

    mask = type_scores[:, BALL_TYPE_INDEX] > 0.5
    if not np.any(mask):
        return {
            "visible": 0.0,
            "count": 0.0,
            "nearest": 1.0,
            "centroid": 0.0,
            "score": 0.0,
        }

    count = float(np.mean(mask))
    nearest = float(np.min(distances[mask]))
    centroid = float(np.mean(ray_positions[mask]))
    score = (1.0 - nearest) + 0.5 * count
    visible = 1.0 if nearest < 0.98 else 0.0
    return {
        "visible": visible,
        "count": count,
        "nearest": nearest,
        "centroid": centroid,
        "score": score,
    }


class DualExpertCoordinator:
    def __init__(
        self,
        *,
        default_attacker_id: int = 0,
        switch_margin: float = 0.08,
        switch_cooldown: int = 6,
    ):
        self.default_attacker_id = int(default_attacker_id)
        self.switch_margin = float(switch_margin)
        self.switch_cooldown = int(switch_cooldown)
        self._attacker_id: Optional[int] = None
        self._cooldown_left = 0

    def reset(self):
        self._attacker_id = None
        self._cooldown_left = 0

    def choose_attacker(self, observation: Dict[int, np.ndarray]) -> int:
        local_ids = sorted(int(k) for k in observation.keys())
        if len(local_ids) != 2:
            return self.default_attacker_id if self.default_attacker_id in local_ids else local_ids[0]

        proxies = {
            agent_id: extract_ball_proxy(observation[agent_id])
            for agent_id in local_ids
        }

        if self._attacker_id is None or self._attacker_id not in local_ids:
            visible_ids = [aid for aid in local_ids if proxies[aid]["visible"] > 0.5]
            if visible_ids:
                self._attacker_id = max(visible_ids, key=lambda aid: proxies[aid]["score"])
            else:
                self._attacker_id = self.default_attacker_id if self.default_attacker_id in local_ids else local_ids[0]
            self._cooldown_left = self.switch_cooldown
            return int(self._attacker_id)

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return int(self._attacker_id)

        current_id = int(self._attacker_id)
        other_id = local_ids[0] if local_ids[1] == current_id else local_ids[1]
        current_score = proxies[current_id]["score"]
        other_score = proxies[other_id]["score"]
        current_visible = proxies[current_id]["visible"] > 0.5
        other_visible = proxies[other_id]["visible"] > 0.5

        should_switch = False
        if other_visible and not current_visible:
            should_switch = True
        elif other_visible and current_visible and other_score > current_score + self.switch_margin:
            should_switch = True

        if should_switch:
            self._attacker_id = int(other_id)
            self._cooldown_left = self.switch_cooldown

        return int(self._attacker_id)


def default_attack_checkpoint():
    return os.environ.get("ATTACK_EXPERT_CHECKPOINT", "").strip() or os.environ.get(
        "TRAINED_RAY_CHECKPOINT", ""
    ).strip()


def default_defense_checkpoint():
    return os.environ.get("DEFENSE_EXPERT_CHECKPOINT", "").strip() or os.environ.get(
        "TRAINED_RAY_CHECKPOINT", ""
    ).strip()
