"""A2 PBRS reward shaping using v3 calibrated outcome predictor.

Per-step PBRS reward = λ * (V(s_t+1) - V(s_t))
where V(s_t) = P(team0_win | obs sequence up to step t), from calibrated transformer.

Calibrated predictor (direction_1b_v3, val_acc 0.835, per-prefix gap 0.015→0.240
across t=5→50) trained with random prefix truncation augmentation, so V(s)
evolves meaningfully with prefix length (NOT just full-trajectory classifier).

Key difference vs LearnedRewardShapingWrapper:
- v2: multi-head bucket reward, single-step state input → scalar shaping
- v3 (this): per-step trajectory transformer → V(s) value function → PBRS ΔV
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import gym
import numpy as np
import torch
import torch.nn as nn


class _OutcomePredictor(nn.Module):
    """Same arch as direction_1b_v3 (4-layer transformer + per-step P(W) head)."""
    def __init__(self, obs_dim: int = 672, d_model: int = 384, n_layers: int = 4, n_heads: int = 6, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(obs_dim, d_model)
        self.proj_dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, 200, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=1024,
                                            batch_first=True, dropout=dropout, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.proj_dropout(self.proj(x)) + self.pos_embed[:, :T]
        kpm = (mask < 0.5)
        h = self.enc(h, src_key_padding_mask=kpm)
        return self.head(h).squeeze(-1)


def _load_predictor(ckpt_path: str, device: str) -> _OutcomePredictor:
    state = torch.load(ckpt_path, map_location=device)
    model = _OutcomePredictor().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


class OutcomePBRSWrapper(gym.Wrapper):
    """Per-step PBRS using calibrated v3 outcome predictor.

    Per env: maintain trajectory buffer of concat(obs[0], obs[1]) per step.
    On step: feed buffer (or its prefix) to predictor → V(s) = sigmoid(mean per-step logit).
    PBRS reward = λ * (V_t+1 - V_t), added to team0 agent rewards.

    Args:
        env: base soccer env (multi-agent dict reward + obs).
        predictor_path: path to .pt file with model state_dict (best_outcome_predictor_v3_calibrated.pt).
        shaping_weight: λ multiplier on ΔV. Default 0.01 (v2 was 0.003 for multi-head).
        team0_agent_ids: which agent ids in dict belong to team0 (these get the bonus).
        device: 'auto' picks cuda if available; CPU is fine for single-env latency.
        warmup_steps: skip shaping for first N env.step()s (per env wrapper).
        max_buffer_steps: cap trajectory buffer length to this (default 80).
    """

    OBS_DIM_PER_AGENT = 336
    CONCAT_OBS_DIM = OBS_DIM_PER_AGENT * 2

    def __init__(
        self,
        env: gym.Env,
        *,
        predictor_path: str,
        shaping_weight: float = 0.01,
        team0_agent_ids: Iterable[int] = (0, 1),
        device: str = "auto",
        warmup_steps: int = 0,
        max_buffer_steps: int = 80,
    ):
        super().__init__(env)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model = _load_predictor(predictor_path, device)
        self._shaping_weight = float(shaping_weight)
        self._team0_agent_ids = tuple(int(i) for i in team0_agent_ids)
        self._warmup_steps_remaining = max(0, int(warmup_steps))
        self._warmup_steps_initial = self._warmup_steps_remaining
        self._max_buffer_steps = int(max_buffer_steps)

        # Per-env state (single env per wrapper instance)
        self._traj_buffer: List[np.ndarray] = []  # list of (CONCAT_OBS_DIM,) arrays
        self._v_prev: Optional[float] = None  # V(s_t) from previous step

        print(f"[outcome-pbrs-shaping] loaded {Path(predictor_path).name}")
        print(f"  config: shaping_weight={self._shaping_weight} "
              f"team0_ids={self._team0_agent_ids} warmup_steps={self._warmup_steps_initial} "
              f"max_buffer_steps={self._max_buffer_steps} device={device}")

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._traj_buffer.clear()
        self._v_prev = None
        # Append initial obs to buffer
        concat = self._concat_team0_obs(obs)
        if concat is not None:
            self._traj_buffer.append(concat)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Append new obs to buffer
        concat = self._concat_team0_obs(obs)
        if concat is not None:
            self._traj_buffer.append(concat)
            # Cap buffer to last max_buffer_steps to bound transformer cost
            if len(self._traj_buffer) > self._max_buffer_steps:
                self._traj_buffer = self._traj_buffer[-self._max_buffer_steps:]

        bonus = 0.0
        if self._warmup_steps_remaining > 0:
            self._warmup_steps_remaining -= 1
        elif len(self._traj_buffer) >= 2:
            # Compute V(s_t+1) using current buffer
            v_curr = self._compute_v()
            if self._v_prev is None:
                # First V computation: no ΔV yet (need at least 2 V values)
                self._v_prev = v_curr
            else:
                bonus = self._shaping_weight * (v_curr - self._v_prev)
                self._v_prev = v_curr

        # Apply bonus to team0 agents in reward dict
        if isinstance(reward, dict) and bonus != 0.0:
            new_reward = dict(reward)
            for aid in self._team0_agent_ids:
                if aid in new_reward:
                    new_reward[aid] = float(new_reward[aid]) + float(bonus)
            reward = new_reward
        elif isinstance(reward, (list, tuple, np.ndarray)) and bonus != 0.0:
            new_reward = list(float(r) for r in reward)
            for aid in self._team0_agent_ids:
                if 0 <= int(aid) < len(new_reward):
                    new_reward[int(aid)] += float(bonus)
            reward = type(reward)(new_reward) if not isinstance(reward, np.ndarray) else np.asarray(new_reward)
        elif bonus != 0.0:
            reward = float(reward) + float(bonus) * len(self._team0_agent_ids)

        # Surface bonus per-agent for diagnostics (per RLlib multi-agent contract)
        if isinstance(info, dict) and bonus != 0.0:
            for aid in self._team0_agent_ids:
                agent_info = info.get(aid)
                if isinstance(agent_info, dict):
                    agent_info["_outcome_pbrs_bonus"] = float(bonus)
                    agent_info["_outcome_pbrs_v"] = float(self._v_prev or 0.0)

        # Reset trajectory state on episode end
        is_done = done.get("__all__", False) if isinstance(done, dict) else bool(done)
        if is_done:
            self._traj_buffer.clear()
            self._v_prev = None

        return obs, reward, done, info

    def _concat_team0_obs(self, obs: Any) -> Optional[np.ndarray]:
        """Build (CONCAT_OBS_DIM,) array = concat(obs[team0_id_0], obs[team0_id_1])."""
        if not isinstance(obs, dict):
            return None
        obs0_id, obs1_id = self._team0_agent_ids[0], self._team0_agent_ids[1]
        a0 = obs.get(obs0_id)
        a1 = obs.get(obs1_id)
        if a0 is None or a1 is None:
            return None
        a0 = np.asarray(a0, dtype=np.float32).reshape(-1)
        a1 = np.asarray(a1, dtype=np.float32).reshape(-1)
        if a0.shape[0] != self.OBS_DIM_PER_AGENT or a1.shape[0] != self.OBS_DIM_PER_AGENT:
            return None
        return np.concatenate([a0, a1], axis=0)  # (672,)

    def _compute_v(self) -> float:
        """Run predictor on current trajectory buffer, return V = mean per-step P(W)."""
        T = len(self._traj_buffer)
        if T == 0:
            return 0.5
        seq = np.stack(self._traj_buffer, axis=0).astype(np.float32)  # (T, 672)
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(self._device)  # (1, T, 672)
        mask_t = torch.ones(1, T, device=self._device)
        with torch.no_grad():
            logits = self._model(seq_t, mask_t)  # (1, T)
            ep_logit = logits.mean().item()
            v = float(torch.sigmoid(torch.tensor(ep_logit)).item())
        return v
