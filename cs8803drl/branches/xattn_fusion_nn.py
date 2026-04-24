"""Cross-attention fusion module for DIR-H ensemble routing.

A lightweight attention mechanism that mixes K frozen experts' action
probabilities via learnable (or static-anchored) state-conditional weights.

Design:
    - Q: projection of agent obs (336-dim) to d-dim query vector
    - K: per-expert learnable key matrix (K × d)
    - b: per-expert learnable bias (K,) — used for W3 anchoring (1750 gets +3.0 init)
    - Scores: Q @ K^T + b (shape: K)
    - Weights: softmax(Scores / sqrt(d))  (shape: K, sum = 1)
    - Output: Σ_i weight[i] * expert_i_probs(obs) — a fused Discrete(27) distribution

For the deploy-time W3 variant (no training), Q-projection is fixed (small random
init) and only the `b` bias steers the anchor: b[0]=+3 for 1750 ensures fusion
is dominated by 1750 at baseline, other experts contribute marginally unless
their keys happen to match the query — i.e. "safe routing" per task-queue P1.

For REINFORCE training (DIR-H W2 fully learned), all params (Q-projection +
keys + bias) become trainable. Training pipeline is a fork of train_moe_router_reinforce.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class XAttnFusionNN(nn.Module):
    """Per-step soft fusion across K frozen experts.

    forward(obs) → (K,) weights summing to 1.
    """

    def __init__(
        self,
        obs_dim: int = 336,
        n_experts: int = 8,
        d_key: int = 64,
        anchor_idx: Optional[int] = 0,
        anchor_bias: float = 3.0,
        trainable: bool = False,
    ):
        super().__init__()
        self._obs_dim = int(obs_dim)
        self._n = int(n_experts)
        self._d = int(d_key)
        self._anchor_idx = anchor_idx
        self._trainable = bool(trainable)

        # Q-projection: obs → d_key
        self._q_proj = nn.Linear(obs_dim, d_key, bias=False)
        # Keys: K × d
        self._keys = nn.Parameter(torch.randn(n_experts, d_key) * 0.1)
        # Bias per expert (anchoring)
        init_bias = torch.zeros(n_experts)
        if anchor_idx is not None and 0 <= anchor_idx < n_experts:
            init_bias[anchor_idx] = anchor_bias
        self._bias = nn.Parameter(init_bias)
        # Temperature (softer/sharper) — fixed at sqrt(d) standard transformer scale
        self._scale = 1.0 / math.sqrt(d_key)

        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs shape (obs_dim,) or (B, obs_dim) → weights shape (K,) or (B, K)."""
        squeeze = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze = True
        q = self._q_proj(obs)                       # (B, d)
        scores = q @ self._keys.t() * self._scale   # (B, K)
        scores = scores + self._bias.unsqueeze(0)    # (B, K)
        weights = torch.softmax(scores, dim=-1)      # (B, K)
        if squeeze:
            weights = weights.squeeze(0)             # (K,)
        return weights

    def n_experts(self) -> int:
        return self._n
