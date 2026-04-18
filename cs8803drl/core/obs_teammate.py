from __future__ import annotations

from typing import Any, Dict, Optional

import gym
import numpy as np


TEAMMATE_STATE_DIM = 4
TIME_OBS_DIM = 1
TEAMMATE_OBS_DIM = TEAMMATE_STATE_DIM + TIME_OBS_DIM

_TEAMMATE_ID_BY_AGENT = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
}


def teammate_id_for_agent(agent_id: int) -> Optional[int]:
    return _TEAMMATE_ID_BY_AGENT.get(int(agent_id))


def parse_teammate_state_scale(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) != TEAMMATE_STATE_DIM:
            raise ValueError(
                f"Teammate state scale must provide {TEAMMATE_STATE_DIM} comma-separated values, "
                f"got {value!r}"
            )
        arr = np.asarray([float(part) for part in parts], dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] != TEAMMATE_STATE_DIM:
            raise ValueError(
                f"Teammate state scale must have {TEAMMATE_STATE_DIM} elements, "
                f"got shape {arr.shape!r}"
            )
    if np.any(arr <= 0):
        raise ValueError(f"Teammate state scale must be strictly positive, got {arr!r}")
    return arr


def build_teammate_obs_space(base_space: gym.Space, *, include_time: bool = False) -> gym.spaces.Box:
    if not isinstance(base_space, gym.spaces.Box):
        raise TypeError(
            "Teammate obs expansion expects a flat Box observation_space, "
            f"got {type(base_space)!r}"
        )
    low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
    tail_dim = TEAMMATE_STATE_DIM + (TIME_OBS_DIM if include_time else 0)
    extra_low = np.full((tail_dim,), -np.inf, dtype=np.float32)
    extra_high = np.full((tail_dim,), np.inf, dtype=np.float32)
    return gym.spaces.Box(
        low=np.concatenate([low, extra_low], axis=0),
        high=np.concatenate([high, extra_high], axis=0),
        dtype=np.float32,
    )


def extract_own_player_state(info: Any, *, state_scale: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if not isinstance(info, dict):
        return None
    player_info = info.get("player_info")
    if not isinstance(player_info, dict):
        return None

    position = player_info.get("position")
    velocity = player_info.get("velocity")
    try:
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        vel = np.asarray(velocity, dtype=np.float32).reshape(-1)
    except Exception:
        return None

    if pos.shape[0] < 2 or vel.shape[0] < 2:
        return None
    state = np.asarray([pos[0], pos[1], vel[0], vel[1]], dtype=np.float32)
    if state_scale is not None:
        scale = np.asarray(state_scale, dtype=np.float32).reshape(-1)
        if scale.shape[0] != TEAMMATE_STATE_DIM:
            raise ValueError(
                f"state_scale must have {TEAMMATE_STATE_DIM} elements, got {scale.shape!r}"
            )
        state = state / scale
    return state


def normalized_episode_time(*, episode_step: int, max_steps: int) -> np.ndarray:
    steps = max(1, int(max_steps))
    frac = float(max(0, int(episode_step))) / float(steps)
    frac = min(max(frac, 0.0), 1.0)
    return np.asarray([frac], dtype=np.float32)


def extract_player_state_by_agent(
    info_by_agent: Any,
    *,
    state_scale: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if not isinstance(info_by_agent, dict):
        return out
    for agent_id, payload in info_by_agent.items():
        if not isinstance(agent_id, (int, np.integer)):
            continue
        state = extract_own_player_state(payload, state_scale=state_scale)
        if state is None:
            continue
        out[int(agent_id)] = state
    return out


def augment_observation_dict_with_teammate_state(
    observation: Any,
    teammate_state_by_agent: Dict[int, np.ndarray],
    *,
    include_time: bool = False,
    normalized_time: Optional[np.ndarray] = None,
) -> Any:
    if not isinstance(observation, dict):
        return observation

    out = {}
    state_zero = np.zeros((TEAMMATE_STATE_DIM,), dtype=np.float32)
    time_tail = (
        np.asarray(normalized_time, dtype=np.float32).reshape(-1)
        if normalized_time is not None
        else np.zeros((TIME_OBS_DIM,), dtype=np.float32)
    )
    if time_tail.shape[0] != TIME_OBS_DIM:
        time_tail = np.zeros((TIME_OBS_DIM,), dtype=np.float32)
    for agent_id, value in observation.items():
        obs = np.asarray(value, dtype=np.float32).reshape(-1)
        teammate_id = teammate_id_for_agent(int(agent_id))
        teammate_state = (
            teammate_state_by_agent.get(int(teammate_id))
            if teammate_id is not None
            else None
        )
        tail = (
            np.asarray(teammate_state, dtype=np.float32).reshape(-1)
            if teammate_state is not None
            else state_zero
        )
        if tail.shape[0] != TEAMMATE_STATE_DIM:
            tail = state_zero
        if include_time:
            tail = np.concatenate([tail, time_tail], axis=0)
        out[agent_id] = np.concatenate([obs, tail], axis=0).astype(np.float32, copy=False)
    return out


def fit_observation_state_decoder(
    env: gym.Env,
    *,
    num_samples: int = 256,
    state_scale: Optional[np.ndarray] = None,
) -> Optional[Dict[str, np.ndarray]]:
    if num_samples <= 0:
        return None

    obs = env.reset()
    features = []
    targets = []
    max_rollout_steps = max(int(num_samples) * 2, 64)
    steps = 0

    while len(features) < int(num_samples) and steps < max_rollout_steps:
        if not isinstance(obs, dict) or not obs:
            break
        action = {agent_id: env.action_space.sample() for agent_id in obs.keys()}
        obs, _reward, done, info = env.step(action)
        state_by_agent = extract_player_state_by_agent(info, state_scale=state_scale)
        for agent_id, value in obs.items():
            state = state_by_agent.get(int(agent_id))
            if state is None:
                continue
            features.append(np.asarray(value, dtype=np.float32).reshape(-1))
            targets.append(state)
            if len(features) >= int(num_samples):
                break
        steps += 1
        episode_done = bool(done.get("__all__")) if isinstance(done, dict) else bool(done)
        if episode_done:
            obs = env.reset()

    if not features:
        return None

    x = np.stack(features).astype(np.float32)
    y = np.stack(targets).astype(np.float32)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    coeffs, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    weights = np.asarray(coeffs[:-1], dtype=np.float32)
    bias = np.asarray(coeffs[-1], dtype=np.float32)
    return {
        "weights": weights,
        "bias": bias,
        "obs_dim": np.asarray([x.shape[1]], dtype=np.int64),
    }


def decode_player_state_from_observation(
    observation: Any,
    decoder: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    if not decoder:
        return np.zeros((TEAMMATE_STATE_DIM,), dtype=np.float32)
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    weights = np.asarray(decoder.get("weights"), dtype=np.float32)
    bias = np.asarray(decoder.get("bias"), dtype=np.float32).reshape(-1)
    if weights.ndim != 2 or bias.shape[0] != TEAMMATE_STATE_DIM:
        return np.zeros((TEAMMATE_STATE_DIM,), dtype=np.float32)
    if obs.shape[0] != weights.shape[0]:
        return np.zeros((TEAMMATE_STATE_DIM,), dtype=np.float32)
    return (obs @ weights + bias).astype(np.float32, copy=False)


def augment_observation_dict_with_decoded_teammate_state(
    observation: Any,
    decoder: Optional[Dict[str, np.ndarray]],
    *,
    include_time: bool = False,
    normalized_time: Optional[np.ndarray] = None,
) -> Any:
    if not isinstance(observation, dict):
        return observation

    decoded_by_agent = {
        int(agent_id): decode_player_state_from_observation(value, decoder)
        for agent_id, value in observation.items()
    }
    return augment_observation_dict_with_teammate_state(
        observation,
        decoded_by_agent,
        include_time=include_time,
        normalized_time=normalized_time,
    )
