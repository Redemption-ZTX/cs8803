import math
from typing import Dict, Sequence

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import RLLIB_MODEL, _global_registry


torch, nn = try_import_torch()


TEAM_SIAMESE_MODEL_NAME = "team_siamese_model"
TEAM_SIAMESE_CROSS_ATTENTION_MODEL_NAME = "team_siamese_cross_attention_model"
TEAM_SIAMESE_TRANSFORMER_MODEL_NAME = "team_siamese_transformer_model"
TEAM_SIAMESE_TRANSFORMER_MIN_MODEL_NAME = "team_siamese_transformer_min_model"
TEAM_SIAMESE_TRANSFORMER_MHA_MODEL_NAME = "team_siamese_transformer_mha_model"
TEAM_SIAMESE_CROSS_AGENT_ATTN_MODEL_NAME = "team_siamese_cross_agent_attn_model"


def _parse_hiddens(values: Sequence[int], default):
    if not values:
        return tuple(int(v) for v in default)
    return tuple(int(v) for v in values)


class SiameseTeamTorchModel(TorchModelV2, nn.Module):
    """Two-branch shared encoder for team-level 672-dim observations.

    The first half of the flat team observation is treated as player 0, the
    second half as player 1. A shared encoder processes both halves, then a
    merge MLP produces policy/value features.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_dim = int(np.product(obs_space.shape))
        if obs_dim % 2 != 0:
            raise ValueError(
                f"SiameseTeamTorchModel expects an even obs dim, got {obs_dim}."
            )
        self._half_obs_dim = int(obs_dim // 2)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        encoder_hiddens = _parse_hiddens(custom_cfg.get("encoder_hiddens"), default=(256, 256))
        merge_hiddens = _parse_hiddens(custom_cfg.get("merge_hiddens"), default=(256, 128))

        encoder_layers = []
        in_size = self._half_obs_dim
        for hidden in encoder_hiddens:
            encoder_layers.append(nn.Linear(in_size, int(hidden)))
            encoder_layers.append(nn.ReLU())
            in_size = int(hidden)
        self.shared_encoder = nn.Sequential(*encoder_layers)

        merge_layers = []
        merge_in = int(encoder_hiddens[-1]) * 2
        for hidden in merge_hiddens:
            merge_layers.append(nn.Linear(merge_in, int(hidden)))
            merge_layers.append(nn.ReLU())
            merge_in = int(hidden)
        self.merge_mlp = nn.Sequential(*merge_layers)

        self.logits_layer = nn.Linear(merge_in, int(num_outputs))
        self.value_layer = nn.Linear(merge_in, 1)

        self._value_out = None
        self._metrics = {
            "encoder_cos_sim_mean": 0.0,
        }

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs0 = obs[:, : self._half_obs_dim]
        obs1 = obs[:, self._half_obs_dim :]

        feat0 = self.shared_encoder(obs0)
        feat1 = self.shared_encoder(obs1)
        merged = self.merge_mlp(torch.cat([feat0, feat1], dim=1))

        logits = self.logits_layer(merged)
        self._value_out = self.value_layer(merged).squeeze(1)

        with torch.no_grad():
            cos = nn.functional.cosine_similarity(feat0, feat1, dim=1).mean()
            self._metrics = {
                "encoder_cos_sim_mean": float(cos.detach().cpu().item()),
            }

        return logits, state

    def value_function(self):
        if self._value_out is None:
            raise ValueError("value_function() called before forward().")
        return self._value_out

    def metrics(self) -> Dict[str, float]:
        return dict(self._metrics)


def register_team_siamese_model():
    try:
        ModelCatalog.register_custom_model(TEAM_SIAMESE_MODEL_NAME, SiameseTeamTorchModel)
    except AttributeError:
        _global_registry.register(RLLIB_MODEL, TEAM_SIAMESE_MODEL_NAME, SiameseTeamTorchModel)
    except ValueError:
        pass


class SiameseCrossAttentionTeamTorchModel(TorchModelV2, nn.Module):
    """031-B: Siamese encoder + bidirectional cross-agent attention.

    Reshapes each agent's encoder output `(batch, encoder_hidden)` into
    `(batch, n_tokens, head_dim)` so that attention over a teammate's tokens
    is non-degenerate (a single-token softmax would always yield 1.0).
    `n_tokens * head_dim` must equal `encoder_hidden`.

    Q/K/V projections are shared across both agents (siamese for the attention
    block as well — agent identity is symmetric). Bidirectional: agent 0
    attends to agent 1 and vice versa.

    Final merge input is the residual concat
        `[feat0, attn0, feat1, attn1]`
    so the architecture degrades gracefully to a wider 031A if attention
    output collapses to zero.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_dim = int(np.product(obs_space.shape))
        if obs_dim % 2 != 0:
            raise ValueError(
                f"SiameseCrossAttentionTeamTorchModel expects an even obs dim, got {obs_dim}."
            )
        self._half_obs_dim = int(obs_dim // 2)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        encoder_hiddens = _parse_hiddens(custom_cfg.get("encoder_hiddens"), default=(256, 256))
        merge_hiddens = _parse_hiddens(custom_cfg.get("merge_hiddens"), default=(256, 128))
        n_tokens = int(custom_cfg.get("attention_n_tokens", 4))
        head_dim = int(custom_cfg.get("attention_head_dim", 64))

        encoder_out = int(encoder_hiddens[-1])
        if n_tokens * head_dim != encoder_out:
            raise ValueError(
                "Cross-attention shape mismatch: "
                f"n_tokens ({n_tokens}) * head_dim ({head_dim}) = {n_tokens * head_dim} "
                f"!= encoder_hidden ({encoder_out}). "
                "Set TEAM_CROSS_ATTENTION_TOKENS and TEAM_CROSS_ATTENTION_DIM "
                "so their product equals the last encoder hidden size."
            )
        self._n_tokens = n_tokens
        self._head_dim = head_dim
        self._encoder_out = encoder_out

        encoder_layers = []
        in_size = self._half_obs_dim
        for hidden in encoder_hiddens:
            encoder_layers.append(nn.Linear(in_size, int(hidden)))
            encoder_layers.append(nn.ReLU())
            in_size = int(hidden)
        self.shared_encoder = nn.Sequential(*encoder_layers)

        self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(head_dim, head_dim, bias=False)

        merge_layers = []
        merge_in = encoder_out * 4
        for hidden in merge_hiddens:
            merge_layers.append(nn.Linear(merge_in, int(hidden)))
            merge_layers.append(nn.ReLU())
            merge_in = int(hidden)
        self.merge_mlp = nn.Sequential(*merge_layers)

        self.logits_layer = nn.Linear(merge_in, int(num_outputs))
        self.value_layer = nn.Linear(merge_in, 1)

        self._value_out = None
        self._metrics = {
            "encoder_cos_sim_mean": 0.0,
            "attention_entropy_mean": 0.0,
            "attention_entropy_min": 0.0,
        }

    def _attend(self, src_tokens, tgt_tokens):
        """src_tokens query, tgt_tokens provide K, V. Return (out, attn_weights).

        Shapes:
            src_tokens, tgt_tokens: (batch, n_tokens, head_dim)
            out:                    (batch, n_tokens, head_dim)
            attn_weights:           (batch, n_tokens, n_tokens) softmax over dim=-1
        """
        q = self.q_proj(src_tokens)
        k = self.k_proj(tgt_tokens)
        v = self.v_proj(tgt_tokens)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self._head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out, attn

    def forward(self, input_dict, state, seq_lens):
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

        logits = self.logits_layer(merged)
        self._value_out = self.value_layer(merged).squeeze(1)

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

        return logits, state

    def value_function(self):
        if self._value_out is None:
            raise ValueError("value_function() called before forward().")
        return self._value_out

    def metrics(self) -> Dict[str, float]:
        return dict(self._metrics)


def register_team_siamese_cross_attention_model():
    try:
        ModelCatalog.register_custom_model(
            TEAM_SIAMESE_CROSS_ATTENTION_MODEL_NAME,
            SiameseCrossAttentionTeamTorchModel,
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL,
            TEAM_SIAMESE_CROSS_ATTENTION_MODEL_NAME,
            SiameseCrossAttentionTeamTorchModel,
        )
    except ValueError:
        pass


class SiameseTransformerMinTeamTorchModel(SiameseCrossAttentionTeamTorchModel):
    """031C-min: keep 031B attention, add shared FFN/residual/norm refinement.

    This is intentionally the smallest interpretable step from 031B:
    - keep the existing hand-written single-head token attention
    - keep the existing 1024-dim merge topology `[feat0, z0, feat1, z1]`
    - only refine the attended feature with a transformer-style FFN block
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        ffn_hidden = int(custom_cfg.get("transformer_ffn_hidden", 512))
        activation = str(custom_cfg.get("transformer_ffn_activation", "gelu")).strip().lower()
        norm_style = str(custom_cfg.get("transformer_norm", "postnorm")).strip().lower()
        if norm_style not in {"postnorm", "prenorm"}:
            raise ValueError(
                "transformer_norm must be one of {'postnorm', 'prenorm'}, "
                f"got {norm_style!r}."
            )

        if activation == "gelu":
            act_layer = nn.GELU()
        elif activation == "relu":
            act_layer = nn.ReLU()
        else:
            raise ValueError(
                "transformer_ffn_activation must be one of {'gelu', 'relu'}, "
                f"got {activation!r}."
            )

        self._transformer_norm = norm_style
        self._transformer_ffn = nn.Sequential(
            nn.Linear(self._encoder_out, ffn_hidden),
            act_layer,
            nn.Linear(ffn_hidden, self._encoder_out),
        )
        self._transformer_ln_attn = nn.LayerNorm(self._encoder_out)
        self._transformer_ln_ffn = nn.LayerNorm(self._encoder_out)

    def _refine_attended_feature(self, feat, attn_flat):
        residual = feat + attn_flat
        if self._transformer_norm == "prenorm":
            refined = residual + self._transformer_ffn(self._transformer_ln_attn(residual))
            return self._transformer_ln_ffn(refined)

        attn_normed = self._transformer_ln_attn(residual)
        refined = attn_normed + self._transformer_ffn(attn_normed)
        return self._transformer_ln_ffn(refined)

    def forward(self, input_dict, state, seq_lens):
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

        trans0 = self._refine_attended_feature(feat0, attn0_flat)
        trans1 = self._refine_attended_feature(feat1, attn1_flat)

        merged_input = torch.cat([feat0, trans0, feat1, trans1], dim=1)
        merged = self.merge_mlp(merged_input)

        logits = self.logits_layer(merged)
        self._value_out = self.value_layer(merged).squeeze(1)

        with torch.no_grad():
            cos = nn.functional.cosine_similarity(feat0, feat1, dim=1).mean()
            ent0 = -(attn0_weights * (attn0_weights + 1e-9).log()).sum(dim=-1)
            ent1 = -(attn1_weights * (attn1_weights + 1e-9).log()).sum(dim=-1)
            ent_all = torch.cat([ent0, ent1], dim=1)
            delta0 = (trans0 - feat0).norm(dim=1)
            delta1 = (trans1 - feat1).norm(dim=1)
            delta_all = torch.cat([delta0, delta1], dim=0)
            self._metrics = {
                "encoder_cos_sim_mean": float(cos.detach().cpu().item()),
                "attention_entropy_mean": float(ent_all.mean().detach().cpu().item()),
                "attention_entropy_min": float(ent_all.min().detach().cpu().item()),
                "transformer_delta_norm_mean": float(delta_all.mean().detach().cpu().item()),
            }

        return logits, state


def register_team_siamese_transformer_min_model():
    try:
        ModelCatalog.register_custom_model(
            TEAM_SIAMESE_TRANSFORMER_MIN_MODEL_NAME,
            SiameseTransformerMinTeamTorchModel,
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL,
            TEAM_SIAMESE_TRANSFORMER_MIN_MODEL_NAME,
            SiameseTransformerMinTeamTorchModel,
        )
    except ValueError:
        pass


class SiameseTransformerMhaTeamTorchModel(SiameseTransformerMinTeamTorchModel):
    """031C-mha: swap 031B's hand-written token attention for true MHA.

    This keeps the `031C-min` FFN/residual/norm block and the existing
    1024-dim merge topology, so the principal architecture delta is the
    attention mechanism itself.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        attention_heads = int(custom_cfg.get("attention_num_heads", 4))
        if attention_heads <= 0:
            raise ValueError(f"attention_num_heads must be positive, got {attention_heads}.")
        if self._head_dim % attention_heads != 0:
            raise ValueError(
                "Multi-head attention shape mismatch: "
                f"attention_head_dim ({self._head_dim}) must be divisible by "
                f"attention_num_heads ({attention_heads})."
            )

        # Remove the single-head projections inherited from 031B; 031C-mha should
        # pay only for the actual MHA parameters we intend to evaluate.
        del self.q_proj
        del self.k_proj
        del self.v_proj

        self._attention_num_heads = attention_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=self._head_dim,
            num_heads=attention_heads,
            dropout=0.0,
            bias=False,
        )

    def _attend(self, src_tokens, tgt_tokens):
        src_tokens_t = src_tokens.transpose(0, 1)
        tgt_tokens_t = tgt_tokens.transpose(0, 1)
        out, attn = self.mha(
            src_tokens_t,
            tgt_tokens_t,
            tgt_tokens_t,
            need_weights=True,
        )
        return out.transpose(0, 1), attn

    def metrics(self) -> Dict[str, float]:
        metrics = super().metrics()
        metrics["attention_num_heads"] = float(self._attention_num_heads)
        return metrics


def register_team_siamese_transformer_mha_model():
    try:
        ModelCatalog.register_custom_model(
            TEAM_SIAMESE_TRANSFORMER_MHA_MODEL_NAME,
            SiameseTransformerMhaTeamTorchModel,
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL,
            TEAM_SIAMESE_TRANSFORMER_MHA_MODEL_NAME,
            SiameseTransformerMhaTeamTorchModel,
        )
    except ValueError:
        pass


class SiameseTransformerTeamTorchModel(SiameseTransformerMhaTeamTorchModel):
    """Full 031C: true MHA + FFN/residual/norm + transformer-style merge.

    Relative to 031C-mha, this changes the joint merge topology from
    `[feat0, z0, feat1, z1] -> 1024` down to `[z0, z1] -> 512`, matching the
    original full snapshot intent.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        merge_hiddens = _parse_hiddens(custom_cfg.get("merge_hiddens"), default=(256, 128))

        merge_layers = []
        merge_in = self._encoder_out * 2
        for hidden in merge_hiddens:
            merge_layers.append(nn.Linear(merge_in, int(hidden)))
            merge_layers.append(nn.ReLU())
            merge_in = int(hidden)
        self.merge_mlp = nn.Sequential(*merge_layers)
        self.logits_layer = nn.Linear(merge_in, int(num_outputs))
        self.value_layer = nn.Linear(merge_in, 1)

    def forward(self, input_dict, state, seq_lens):
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

        trans0 = self._refine_attended_feature(feat0, attn0_flat)
        trans1 = self._refine_attended_feature(feat1, attn1_flat)

        merged_input = torch.cat([trans0, trans1], dim=1)
        merged = self.merge_mlp(merged_input)

        logits = self.logits_layer(merged)
        self._value_out = self.value_layer(merged).squeeze(1)

        with torch.no_grad():
            cos = nn.functional.cosine_similarity(feat0, feat1, dim=1).mean()
            ent0 = -(attn0_weights * (attn0_weights + 1e-9).log()).sum(dim=-1)
            ent1 = -(attn1_weights * (attn1_weights + 1e-9).log()).sum(dim=-1)
            ent_all = torch.cat([ent0, ent1], dim=1)
            delta0 = (trans0 - feat0).norm(dim=1)
            delta1 = (trans1 - feat1).norm(dim=1)
            delta_all = torch.cat([delta0, delta1], dim=0)
            self._metrics = {
                "encoder_cos_sim_mean": float(cos.detach().cpu().item()),
                "attention_entropy_mean": float(ent_all.mean().detach().cpu().item()),
                "attention_entropy_min": float(ent_all.min().detach().cpu().item()),
                "transformer_delta_norm_mean": float(delta_all.mean().detach().cpu().item()),
                "attention_num_heads": float(self._attention_num_heads),
            }

        return logits, state


def register_team_siamese_transformer_model():
    try:
        ModelCatalog.register_custom_model(
            TEAM_SIAMESE_TRANSFORMER_MODEL_NAME,
            SiameseTransformerTeamTorchModel,
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL,
            TEAM_SIAMESE_TRANSFORMER_MODEL_NAME,
            SiameseTransformerTeamTorchModel,
        )
    except ValueError:
        pass


# ============================================================================
# 054: MAT-min cross-AGENT attention
# ============================================================================
# Subclass adds a cross-AGENT attention block AFTER 031B's within-agent
# token attention, BEFORE merge. Each agent's encoded feature attends over
# both agents' features (2-token attention, single-head, no FFN, no LayerNorm).
#
# Lessons from 052 (REGRESSION -8 ~ -11pp): refinement block (FFN + LayerNorm)
# hurts on small PPO model. MAT-min strictly avoids:
#   - NO FFN (no over-param)
#   - NO LayerNorm (no PPO gradient conflict)
#   - NO merge topology change (preserve 1024 layout)
#
# Just adds a small cross-agent attention residual (~80K params, ~5% increase).
class SiameseCrossAgentAttnTeamTorchModel(SiameseCrossAttentionTeamTorchModel):
    """031B + cross-AGENT attention block (MAT-min for 2v2 soccer).

    Difference from 031B (within-agent token attention only):
      Step 3 NEW: per-agent attended feature → cross-agent attention layer
        Q/K/V over 2 agents (Q,K dim 64, V dim = encoder_out 256)
        attn = softmax(Q @ K.T / sqrt(64))  # (B, 2, 2)
        cross_agent = attn @ V  # (B, 2, encoder_out)
        final_aN = attended_aN + cross_agent[:, N]   # residual, NO LN

    Merge layout SAME as 031B (1024 dim concat), uses final_aN where 031B uses attn_aN.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        custom_cfg = dict(model_config.get("custom_model_config") or {})
        # cross-agent attention dim (Q/K projection dim, V keeps encoder_out)
        cross_agent_attn_dim = int(custom_cfg.get("cross_agent_attn_dim", 64))
        self._cross_agent_attn_dim = cross_agent_attn_dim

        # Q, K project encoder_out → cross_agent_attn_dim (small, prevents over-param)
        # V projects encoder_out → encoder_out (preserves dim for residual addition)
        self._ca_q_proj = nn.Linear(self._encoder_out, cross_agent_attn_dim, bias=False)
        self._ca_k_proj = nn.Linear(self._encoder_out, cross_agent_attn_dim, bias=False)
        self._ca_v_proj = nn.Linear(self._encoder_out, self._encoder_out, bias=False)
        # Init V to small scale so residual starts near identity (graceful degrade to 031B)
        nn.init.zeros_(self._ca_v_proj.weight)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs0 = obs[:, : self._half_obs_dim]
        obs1 = obs[:, self._half_obs_dim:]

        feat0 = self.shared_encoder(obs0)
        feat1 = self.shared_encoder(obs1)

        batch_size = feat0.shape[0]
        tokens0 = feat0.view(batch_size, self._n_tokens, self._head_dim)
        tokens1 = feat1.view(batch_size, self._n_tokens, self._head_dim)

        # 031B within-agent attention
        attn0_out, attn0_weights = self._attend(tokens0, tokens1)
        attn1_out, attn1_weights = self._attend(tokens1, tokens0)
        attended0 = attn0_out.reshape(batch_size, self._encoder_out)  # (B, 256)
        attended1 = attn1_out.reshape(batch_size, self._encoder_out)  # (B, 256)

        # NEW: cross-AGENT attention block
        # Stack 2 agents as tokens: (B, 2, encoder_out)
        agent_stack = torch.stack([attended0, attended1], dim=1)
        q = self._ca_q_proj(agent_stack)  # (B, 2, ca_dim)
        k = self._ca_k_proj(agent_stack)  # (B, 2, ca_dim)
        v = self._ca_v_proj(agent_stack)  # (B, 2, encoder_out)
        ca_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self._cross_agent_attn_dim)
        ca_weights = torch.softmax(ca_scores, dim=-1)  # (B, 2, 2)
        ca_out = torch.matmul(ca_weights, v)  # (B, 2, encoder_out)

        # Residual add (no LayerNorm, no FFN) — graceful degrade to 031B if ca→0
        final0 = attended0 + ca_out[:, 0]
        final1 = attended1 + ca_out[:, 1]

        # 031B merge layout (use final instead of attended)
        merged_input = torch.cat([feat0, final0, feat1, final1], dim=1)
        merged = self.merge_mlp(merged_input)

        logits = self.logits_layer(merged)
        self._value_out = self.value_layer(merged).squeeze(1)

        with torch.no_grad():
            cos = nn.functional.cosine_similarity(feat0, feat1, dim=1).mean()
            ent0 = -(attn0_weights * (attn0_weights + 1e-9).log()).sum(dim=-1)
            ent1 = -(attn1_weights * (attn1_weights + 1e-9).log()).sum(dim=-1)
            ca_ent = -(ca_weights * (ca_weights + 1e-9).log()).sum(dim=-1)
            ca_delta_norm = (ca_out[:, 0].norm(dim=-1) + ca_out[:, 1].norm(dim=-1)) / 2.0
            self._metrics = {
                "encoder_cos_sim_mean": float(cos.detach().cpu().item()),
                "attention_entropy_mean": float(torch.cat([ent0, ent1], dim=1).mean().detach().cpu().item()),
                "cross_agent_attn_entropy_mean": float(ca_ent.mean().detach().cpu().item()),
                "cross_agent_residual_norm_mean": float(ca_delta_norm.mean().detach().cpu().item()),
            }

        return logits, state


def register_team_siamese_cross_agent_attn_model():
    try:
        ModelCatalog.register_custom_model(
            TEAM_SIAMESE_CROSS_AGENT_ATTN_MODEL_NAME,
            SiameseCrossAgentAttnTeamTorchModel,
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL,
            TEAM_SIAMESE_CROSS_AGENT_ATTN_MODEL_NAME,
            SiameseCrossAgentAttnTeamTorchModel,
        )
    except ValueError:
        pass
