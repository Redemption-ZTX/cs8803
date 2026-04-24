"""VDN (Value Decomposition Network) variant of 031B Siamese cross-attention.

Sunehag et al. 2017 "Value-Decomposition Networks for Cooperative Multi-Agent
Learning". Architecturally enforces the joint critic to factorize as

    V_team(s) = V_0(s_0) + V_1(s_1)

where V_k uses ONLY agent k's own observation (no cross-attention features for
the value path). This forces the per-agent value heads to learn each agent's
INDIVIDUAL contribution to the team return, which gives PPO's advantage
calculation explicit per-agent credit signal.

NOTE on actor vs critic decomposition:
- Actor (logits) STILL uses the merged cross-attention feature (centralized
  policy) — same as 031B parent.
- Critic (value) is the only thing changed to be decomposed.

This is "centralized actor + decomposed critic" — the simplest VDN variant
that fits inside our existing PPO + team-level joint-action stack. A pure
VDN would also decentralize the actor (one policy per agent), but that
requires a multi-agent policy mapping change and is left to a future variant.

vs 031B:
- Same encoder, same cross-attention (action policy still centralized)
- Different value path: per-agent (no cross-attn) summed → joint V
- Forces value decomposition that 031B's single value head cannot express
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

try:
    from ray.rllib.models import ModelCatalog
except Exception:  # pragma: no cover
    ModelCatalog = None
try:
    from ray.tune.registry import _global_registry, RLLIB_MODEL
except Exception:  # pragma: no cover
    _global_registry = None
    RLLIB_MODEL = None

from cs8803drl.branches.team_siamese import SiameseCrossAttentionTeamTorchModel


TEAM_SIAMESE_VDN_MODEL_NAME = "team_siamese_vdn_model"


class SiameseVDNTeamTorchModel(SiameseCrossAttentionTeamTorchModel):
    """031B Siamese cross-attention with VDN-style decomposed critic.

    Inherits the encoder + cross-attention + actor logits from 031B.
    Replaces the single value head with per-agent value heads whose sum
    is the joint critic value.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        # Per-agent value heads. Each takes ONLY that agent's encoder feature
        # (NOT the cross-attention output) so the per-agent value depends only
        # on the agent's own observation — Sunehag 2017 §3.1 setup.
        self.value_layer_agent0 = nn.Linear(self._encoder_out, 1)
        self.value_layer_agent1 = nn.Linear(self._encoder_out, 1)

        # Optional bias parameter (shared scalar) for joint V. Helps numerical
        # stability when both per-agent values converge to ~0 during early
        # training (otherwise V floor is exactly 0 which can break advantage
        # normalization). Initialized to 0.
        self.value_bias = nn.Parameter(torch.zeros(1))

        # The parent's `self.value_layer` (single head on merged features) is
        # left in place but unused. We override `value_function()` to ignore it.
        self._vdn_metrics = {
            "vdn_v_agent0_mean": 0.0,
            "vdn_v_agent1_mean": 0.0,
            "vdn_v_agent_diff_abs_mean": 0.0,
        }

    def forward(self, input_dict, state, seq_lens):
        """Same as parent but compute decomposed per-agent value from
        encoder features (not the merged feature)."""
        obs = input_dict["obs_flat"].float()
        obs0 = obs[:, : self._half_obs_dim]
        obs1 = obs[:, self._half_obs_dim :]

        feat0 = self.shared_encoder(obs0)
        feat1 = self.shared_encoder(obs1)

        batch_size = feat0.shape[0]
        tokens0 = feat0.view(batch_size, self._n_tokens, self._head_dim)
        tokens1 = feat1.view(batch_size, self._n_tokens, self._head_dim)

        attn0_out, attn0_weights = self._attend(tokens0, tokens1)
        attn1_out, attn1_weights = self._attend(tokens1, tokens0)

        attn0_flat = attn0_out.reshape(batch_size, self._encoder_out)
        attn1_flat = attn1_out.reshape(batch_size, self._encoder_out)

        merged_input = torch.cat([feat0, attn0_flat, feat1, attn1_flat], dim=1)
        merged = self.merge_mlp(merged_input)

        # Actor: still uses the merged feature (centralized policy)
        logits = self.logits_layer(merged)

        # VDN critic: per-agent value heads on each agent's OWN encoder feature
        v0 = self.value_layer_agent0(feat0).squeeze(1)
        v1 = self.value_layer_agent1(feat1).squeeze(1)
        # Joint V = V_0 + V_1 + bias
        self._value_out = v0 + v1 + self.value_bias

        with torch.no_grad():
            cos = nn.functional.cosine_similarity(feat0, feat1, dim=1).mean()
            ent0 = -(attn0_weights * (attn0_weights + 1e-9).log()).sum(dim=-1)
            ent1 = -(attn1_weights * (attn1_weights + 1e-9).log()).sum(dim=-1)
            ent_all = torch.cat([ent0, ent1], dim=1)
            self._metrics = {
                "encoder_cos_sim_mean": float(cos.detach().cpu().item()),
                "attention_entropy_mean": float(ent_all.mean().detach().cpu().item()),
                "attention_entropy_min": float(ent_all.min().detach().cpu().item()),
            }
            self._vdn_metrics = {
                "vdn_v_agent0_mean": float(v0.mean().detach().cpu().item()),
                "vdn_v_agent1_mean": float(v1.mean().detach().cpu().item()),
                "vdn_v_agent_diff_abs_mean": float(
                    (v0 - v1).abs().mean().detach().cpu().item()
                ),
            }

        return logits, state

    def metrics(self) -> Dict[str, float]:
        merged = dict(self._metrics)
        merged.update(self._vdn_metrics)
        return merged


def register_team_siamese_vdn_model():
    try:
        if ModelCatalog is not None:
            ModelCatalog.register_custom_model(
                TEAM_SIAMESE_VDN_MODEL_NAME, SiameseVDNTeamTorchModel
            )
            return
    except (AttributeError, ValueError):
        pass
    if _global_registry is not None and RLLIB_MODEL is not None:
        try:
            _global_registry.register(
                RLLIB_MODEL, TEAM_SIAMESE_VDN_MODEL_NAME, SiameseVDNTeamTorchModel
            )
        except ValueError:
            pass
