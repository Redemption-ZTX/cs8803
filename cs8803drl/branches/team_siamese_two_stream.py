"""082: Hierarchical two-stream Siamese encoder for 2v2 Soccer.

Splits each agent's 336-dim observation into two semantic streams processed by
separate encoders, then concatenates the per-agent features before the existing
031B within-agent cross-attention + merge pipeline.

Motivation
----------
031B treats the 336-dim per-agent obs as a monolithic vector fed into a single
MLP. The observation, however, is a ray-perception bundle with a clear stride
structure (stacked frames / ray sensors). Forcing one dense layer to disentangle
temporal vs sensor structure limits feature quality. A hierarchical encoder
gives explicit inductive bias: one sub-encoder focuses on the "self-state /
current-frame" slice, a wider sub-encoder on the historical / environmental
slice.

Observation split (discovered via soccer_twos upstream inspection)
------------------------------------------------------------------
The compiled SoccerTwos binary exposes `observation_space = Box(336,)`. The
wrapper at `site-packages/soccer_twos/wrappers.py:172` hardcodes this shape,
and downstream `_obs[:336]` is pure ray perception — position/velocity/ball
state is delivered separately via the info dict, not the obs.

ML-Agents default for this compiled env: RayPerceptionSensor with
`tags=7, distance=1 → 8 features per ray`, and stacking 3 frames. That gives
336 = 3 stacked frames × 112 features per frame. Within each frame: a forward
sensor bank (88) + a back sensor bank (24).

Since we cannot crack the Unity binary to surface true field semantics, we use
a temporal hierarchy as the cleanest defensible split:
  - "self_slice" (default: last 112 dims) ≈ most recent frame's ray view.
    This is the agent's *current egocentric snapshot*.
  - "env_slice"  (default: first 224 dims) ≈ the two older stacked frames —
    historical perception used for motion / velocity inference.

Both slice widths and encoder hidden sizes are env-var-configurable, so later
snapshots can rerun with a different split (e.g., front-sensor vs back-sensor)
without a code change.

Architecture
------------
Per agent:
  obs_336 → split [env_slice, self_slice]
    self_stream:  self_dim  → MLP(self_hiddens)   → self_feat   (e.g. 64)
    env_stream:   env_dim   → MLP(env_hiddens)    → env_feat    (e.g. 128)
    concat → agent_feat = [self_feat, env_feat]  (e.g. 192)

Two agents → concat(agent_feat_0, agent_feat_1) → 031B cross-attention + merge:
  tokens = agent_feat.view(B, n_tokens, head_dim)
  attn_self = softmax(Q·Kᵀ/√d) · V over teammate tokens (bidirectional siamese)
  merge_input = [feat0, attn0, feat1, attn1] → merge MLP → policy/value heads

Cross-attention is retained because prior snapshots (031B +2pp over 031A)
established attention as a load-bearing component; this snapshot isolates the
encoder change.
"""

import math
from typing import Dict, Sequence

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import RLLIB_MODEL, _global_registry


torch, nn = try_import_torch()


TEAM_SIAMESE_TWO_STREAM_MODEL_NAME = "team_siamese_two_stream_model"


def _parse_hiddens(values: Sequence[int], default):
    if not values:
        return tuple(int(v) for v in default)
    return tuple(int(v) for v in values)


class SiameseTwoStreamTeamTorchModel(TorchModelV2, nn.Module):
    """Hierarchical two-stream Siamese encoder + 031B cross-attention.

    Splits each agent's obs into `self_slice` (tail, small, default 112 dim) and
    `env_slice` (head, larger, default 224 dim), encodes them separately, and
    concatenates before the existing within-agent cross-attention pipeline.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_dim = int(np.product(obs_space.shape))
        if obs_dim % 2 != 0:
            raise ValueError(
                f"SiameseTwoStreamTeamTorchModel expects an even obs dim, got {obs_dim}."
            )
        self._half_obs_dim = int(obs_dim // 2)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        # self_dim defaults to the last (336 // 3) = 112 dims, representing the
        # most recent stacked ray frame. env_dim defaults to the remainder.
        default_self_dim = max(1, self._half_obs_dim // 3)
        self_dim = int(custom_cfg.get("self_slice_dim", default_self_dim))
        if self_dim <= 0 or self_dim >= self._half_obs_dim:
            raise ValueError(
                f"self_slice_dim must be in (0, {self._half_obs_dim}), got {self_dim}."
            )
        self._self_dim = self_dim
        self._env_dim = int(self._half_obs_dim - self_dim)

        # Default hidden sizes chosen so self_out + env_out = 256 = 4 * 64
        # (matches 031B's default TEAM_CROSS_ATTENTION_TOKENS * _DIM so attention
        # shape check passes out of the box). Users tuning these must keep the
        # last-layer sum aligned with n_tokens * head_dim.
        self_hiddens = _parse_hiddens(custom_cfg.get("self_hiddens"), default=(64, 64))
        env_hiddens = _parse_hiddens(custom_cfg.get("env_hiddens"), default=(192, 192))
        merge_hiddens = _parse_hiddens(custom_cfg.get("merge_hiddens"), default=(256, 128))

        self_layers = []
        in_size = self._self_dim
        for hidden in self_hiddens:
            self_layers.append(nn.Linear(in_size, int(hidden)))
            self_layers.append(nn.ReLU())
            in_size = int(hidden)
        self.self_encoder = nn.Sequential(*self_layers)
        self_out = int(self_hiddens[-1])

        env_layers = []
        in_size = self._env_dim
        for hidden in env_hiddens:
            env_layers.append(nn.Linear(in_size, int(hidden)))
            env_layers.append(nn.ReLU())
            in_size = int(hidden)
        self.env_encoder = nn.Sequential(*env_layers)
        env_out = int(env_hiddens[-1])

        encoder_out = self_out + env_out
        self._encoder_out = encoder_out

        n_tokens = int(custom_cfg.get("attention_n_tokens", 4))
        head_dim = int(custom_cfg.get("attention_head_dim", 64))
        if n_tokens * head_dim != encoder_out:
            raise ValueError(
                "Two-stream encoder_out does not match attention shape: "
                f"self_out ({self_out}) + env_out ({env_out}) = {encoder_out}, "
                f"but n_tokens ({n_tokens}) * head_dim ({head_dim}) = {n_tokens * head_dim}. "
                "Adjust TEAM_SIAMESE_SELF_HIDDENS / TEAM_SIAMESE_ENV_HIDDENS so their "
                "last-layer sum equals TEAM_CROSS_ATTENTION_TOKENS * TEAM_CROSS_ATTENTION_DIM."
            )
        self._n_tokens = n_tokens
        self._head_dim = head_dim

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
            "self_stream_norm_mean": 0.0,
            "env_stream_norm_mean": 0.0,
        }

    def _encode_agent(self, half_obs):
        # self_slice is the tail of the obs (most recent frame heuristic);
        # env_slice is the head (historical frames heuristic).
        env_slice = half_obs[:, : self._env_dim]
        self_slice = half_obs[:, self._env_dim :]
        self_feat = self.self_encoder(self_slice)
        env_feat = self.env_encoder(env_slice)
        return torch.cat([self_feat, env_feat], dim=1), self_feat, env_feat

    def _attend(self, src_tokens, tgt_tokens):
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

        feat0, self0, env0 = self._encode_agent(obs0)
        feat1, self1, env1 = self._encode_agent(obs1)

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
            self_norm = 0.5 * (self0.norm(dim=1).mean() + self1.norm(dim=1).mean())
            env_norm = 0.5 * (env0.norm(dim=1).mean() + env1.norm(dim=1).mean())
            self._metrics = {
                "encoder_cos_sim_mean": float(cos.detach().cpu().item()),
                "attention_entropy_mean": float(ent_all.mean().detach().cpu().item()),
                "self_stream_norm_mean": float(self_norm.detach().cpu().item()),
                "env_stream_norm_mean": float(env_norm.detach().cpu().item()),
            }

        return logits, state

    def value_function(self):
        if self._value_out is None:
            raise ValueError("value_function() called before forward().")
        return self._value_out

    def metrics(self) -> Dict[str, float]:
        return dict(self._metrics)


def register_team_siamese_two_stream_model():
    try:
        ModelCatalog.register_custom_model(
            TEAM_SIAMESE_TWO_STREAM_MODEL_NAME, SiameseTwoStreamTeamTorchModel
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL, TEAM_SIAMESE_TWO_STREAM_MODEL_NAME, SiameseTwoStreamTeamTorchModel
        )
    except ValueError:
        pass
