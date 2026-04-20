"""RND env wrapper: per-step intrinsic reward + periodic predictor training.

Designed for use within `cs8803drl.core.utils.create_rllib_env`. One RNDModule
lives per env instance (per worker process). Each step:
  * Concat team0 obs (2 agents × 336 = 672) into one observation
  * Compute intrinsic reward = ||target(obs) - predictor(obs)||^2 (normalized)
  * Add intrinsic to team0 agents' rewards (scaled by `shaping_weight`)
  * Buffer obs for periodic predictor update

Every `train_every_steps` env steps, run one predictor optimizer step on the
buffered obs. This converges per-worker predictor over time so intrinsic
shrinks for visited states.

Snapshot reference: SNAPSHOT-057 (Tier A3).
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import gym
import numpy as np

from cs8803drl.branches.rnd_module import RNDModule


class RNDWrapper(gym.Wrapper):
    OBS_DIM_PER_AGENT = 336
    CONCAT_OBS_DIM = OBS_DIM_PER_AGENT * 2

    def __init__(
        self,
        env: gym.Env,
        *,
        shaping_weight: float = 0.01,
        team0_agent_ids: Iterable[int] = (0, 1),
        hidden_dim: int = 256,
        embed_dim: int = 64,
        rnd_lr: float = 1e-4,
        train_every_steps: int = 16,
        train_batch_size: int = 256,
        warmup_steps: int = 0,
        device: str = "cpu",
        random_seed: int = 1234,
    ):
        super().__init__(env)
        self._shaping_weight = float(shaping_weight)
        self._team0_agent_ids = tuple(int(i) for i in team0_agent_ids)
        self._train_every = max(1, int(train_every_steps))
        self._train_batch_size = max(8, int(train_batch_size))
        self._warmup_remaining = int(warmup_steps)
        self._step_counter = 0

        self._rnd = RNDModule(
            obs_dim=self.CONCAT_OBS_DIM,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            lr=rnd_lr,
            device=device,
            random_seed=random_seed,
        )

        # Buffer of raw concat obs awaiting predictor training
        self._obs_buffer: list[np.ndarray] = []
        self._max_buffer = self._train_batch_size * 4  # cap buffer growth

        print(
            f"[rnd-shaping] init shaping_weight={self._shaping_weight} "
            f"hidden={hidden_dim} embed={embed_dim} lr={rnd_lr} "
            f"train_every={self._train_every} batch={self._train_batch_size} "
            f"warmup={warmup_steps} device={device}"
        )

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        concat_obs = self._concat_team0_obs(obs)
        bonus = 0.0
        if concat_obs is not None:
            self._obs_buffer.append(concat_obs)
            if len(self._obs_buffer) > self._max_buffer:
                self._obs_buffer = self._obs_buffer[-self._max_buffer:]

            if self._warmup_remaining > 0:
                self._warmup_remaining -= 1
            else:
                # Compute intrinsic for this step
                intrinsic = self._rnd.compute_intrinsic(concat_obs)
                bonus = self._shaping_weight * float(intrinsic)

            # Periodic predictor training
            self._step_counter += 1
            if (
                self._step_counter % self._train_every == 0
                and len(self._obs_buffer) >= self._train_batch_size
            ):
                # Sample most recent batch (FIFO) for predictor update
                batch = np.stack(self._obs_buffer[-self._train_batch_size:], axis=0)
                self._rnd.train_step(batch)

        # Apply bonus to team0 agents
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
            reward = (
                type(reward)(new_reward)
                if not isinstance(reward, np.ndarray)
                else np.asarray(new_reward)
            )

        # Surface bonus per-agent for diagnostics
        if isinstance(info, dict) and bonus != 0.0:
            for aid in self._team0_agent_ids:
                agent_info = info.get(aid)
                if isinstance(agent_info, dict):
                    agent_info["_rnd_bonus"] = float(bonus)

        return obs, reward, done, info

    def _concat_team0_obs(self, obs: Any) -> Optional[np.ndarray]:
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
        return np.concatenate([a0, a1], axis=0)
