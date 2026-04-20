"""RND (Random Network Distillation, Burda et al. 2019) intrinsic-motivation module.

Self-contained PyTorch module that:
  * holds a frozen random target network + learnable predictor network
  * computes per-step intrinsic reward = ||target(obs) - predictor(obs)||^2
  * maintains running obs normalization + running intrinsic-reward normalization
  * trains the predictor with a single-optimizer self-contained step API

Designed to live inside an env wrapper (one instance per worker process),
so per-worker predictor copies diverge but approximate the same fixed target.

Snapshot reference: SNAPSHOT-057 (Tier A3).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


class RunningNormalizer:
    """Online mean/std tracker (Welford's algorithm) for a scalar or vector."""

    def __init__(self, shape, eps: float = 1e-6):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape):
            batch_count = 1
            batch_mean = x
            batch_var = np.zeros_like(x)
        else:
            batch_count = x.shape[0]
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-8))


class RNDModule(nn.Module):
    """RND with trainable predictor + frozen target + normalization.

    Forward returns intrinsic reward (normalized, clipped). Call
    `train_step(obs_batch)` to take one predictor gradient step.
    """

    def __init__(
        self,
        obs_dim: int = 672,
        hidden_dim: int = 256,
        embed_dim: int = 64,
        lr: float = 1e-4,
        obs_clip: float = 5.0,
        intrinsic_clip: float = 5.0,
        device: str = "cpu",
        random_seed: int = 1234,
    ):
        super().__init__()
        self._device = torch.device(device)

        # Fixed, deterministic target across workers (use manual_seed for reproducibility)
        torch.manual_seed(random_seed)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        for m in self.target.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for p in self.target.parameters():
            p.requires_grad = False

        # Trainable predictor (per-worker divergent init)
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.target.to(self._device).eval()
        self.predictor.to(self._device)

        self._optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        self._obs_norm = RunningNormalizer(obs_dim)
        self._intrinsic_norm = RunningNormalizer(1)
        self._obs_clip = float(obs_clip)
        self._intrinsic_clip = float(intrinsic_clip)
        self._obs_dim = int(obs_dim)

    def _normalize_obs(self, obs_t: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self._obs_norm.mean, dtype=obs_t.dtype, device=obs_t.device)
        std = torch.as_tensor(self._obs_norm.std, dtype=obs_t.dtype, device=obs_t.device)
        return ((obs_t - mean) / (std + 1e-6)).clamp(-self._obs_clip, self._obs_clip)

    @torch.no_grad()
    def compute_intrinsic(self, obs: np.ndarray) -> np.ndarray:
        """Compute normalized intrinsic reward for obs (B, D) or (D,). Returns (B,) or scalar."""
        obs_np = np.asarray(obs, dtype=np.float32)
        single = obs_np.ndim == 1
        if single:
            obs_np = obs_np[None, :]
        # Update obs normalizer first (uses this batch)
        self._obs_norm.update(obs_np)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self._device)
        obs_n = self._normalize_obs(obs_t)
        t_out = self.target(obs_n)
        p_out = self.predictor(obs_n)
        raw_intrinsic = ((t_out - p_out) ** 2).mean(dim=-1)
        raw_np = raw_intrinsic.detach().cpu().numpy()
        # Update + apply intrinsic normalizer
        self._intrinsic_norm.update(raw_np.reshape(-1, 1))
        int_std = float(self._intrinsic_norm.std.reshape(-1)[0])
        normalized = raw_np / max(int_std, 1e-6)
        normalized = np.clip(normalized, -self._intrinsic_clip, self._intrinsic_clip)
        if single:
            return float(normalized[0])
        return normalized.astype(np.float32)

    def train_step(self, obs_batch: np.ndarray) -> float:
        """One predictor gradient step on obs_batch. Returns loss value."""
        obs_np = np.asarray(obs_batch, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self._device)
        obs_n = self._normalize_obs(obs_t)
        with torch.no_grad():
            t_out = self.target(obs_n)
        p_out = self.predictor(obs_n)
        loss = ((t_out - p_out) ** 2).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return float(loss.detach().cpu().item())
