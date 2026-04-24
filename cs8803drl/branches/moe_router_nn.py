"""Shared RouterNN definition for DIR-G MoE Router (Wave 2 trained variant).

Used by both `scripts/research/train_moe_router_reinforce.py` (training) and
`agents/v_moe_router_trained/agent.py` (deployment) to keep the architecture
identical between training and inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RouterNN(nn.Module):
    """Small MLP: 336 → hidden → hidden → K_experts logits."""

    def __init__(self, obs_dim: int = 336, n_experts: int = 3, hidden: int = 64):
        super().__init__()
        self._obs_dim = int(obs_dim)
        self._n = int(n_experts)
        self._hidden = int(hidden)
        self._trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self._trunk(obs)

    def n_experts(self) -> int:
        return self._n
