"""083: Per-Ray Attention Siamese encoder for 2v2 Soccer (DIR ③).

Replaces the monolithic 336 → MLP(256, 256) per-agent encoder with a
**token-per-ray** encoder: the 336-dim observation is reshaped as
(3 frames, 14 rays, 8 features), each ray is collapsed across frames into a
single 24-dim token (3 frames × 8 features), 14 ray tokens are embedded and
refined by a small Transformer-style self-attention stack, and the refined
tokens are pooled back to a 256-dim agent feature. The per-agent feature is
then concatenated with the teammate's feature and fed through the *same*
031B cross-attention + merge + policy/value head pipeline.

Motivation (see snapshot-083-per-ray-attention.md):
---------------------------------------------------
- 031B / 076 / 055 / 082 all tied in [0.89, 0.92] and saturated.
- 082 already separated temporal stacks but kept the MLP (no spatial
  inductive bias across rays).
- The raw observation has an obvious spatial structure: **14 independent
  ray casts, each carrying the same 8-feature schema (7 tag one-hots +
  1 distance)**. A monolithic MLP is forced to discover ray-level locality
  from scratch; a per-ray attention layer gets it for free (shared ray
  embedding + pairwise interactions across rays).
- This is the minimal encoder revision that changes the **feature-
  extraction primitive** (not just width or stream split).

Observation layout (same as snapshot-082 discovery)
---------------------------------------------------
soccer_twos wraps the compiled Unity env to expose
`observation_space = Box(336,)`, with `_obs[:336]` being the complete
policy input (position/rotation/velocity/ball state are info-dict only).
Unity ML-Agents RayPerceptionSensor default for the compiled env:
    tags=7, distance=1 → 8 features / ray
    11 forward rays + 3 back rays = 14 rays / frame → 112 features / frame
    3 stacked frames → 336 total.
Stacked frames are laid out contiguously: obs[0:112], obs[112:224],
obs[224:336] correspond to frame t-2, t-1, t.

Frame handling (Option A variant)
---------------------------------
This implementation picks **Option A with per-ray temporal collapse**:
reshape to (3, 14, 8), permute to (14, 3, 8), then flatten the temporal axis
into the feature dimension → (14 rays, 24 features). Each ray token then
carries its own 3-frame history *inside* the feature vector, and self-
attention runs over the 14 spatial tokens within a single pass. This is
cheaper than Option B (42 tokens), simpler than Option C (hierarchical),
and matches the pre-registered defaults `TEAM_PER_RAY_N_RAYS=14` /
`TEAM_PER_RAY_FEAT_DIM=24` cleanly. Note that temporal structure is
preserved but not *attended to* across frames — this is a deliberate
lane-scope restriction; if 083 HITs, 083-follow-up can explore Option C
(attend-within-frame, then attend-across-frames).

Pooling
-------
Three pool modes are supported; default `mean` matches Set-Transformer
practice for small token counts:
    - `mean`   : token-wise mean over the 14 rays.
    - `cls`    : prepend a learnable [CLS] token, attend jointly, return the
                 CLS slot after the last attention layer.
    - `attn`   : a learned query attends over the 14 refined tokens, returns
                 a single pooled vector (simplified Perceiver-IO-style).

Output size
-----------
The pooled vector is projected (linear + ReLU) to 256-dim to match
`encoder_hiddens[-1]=256`, so that all downstream 031B shape constraints
(cross-attention `n_tokens * head_dim == encoder_hidden`, merge concat of
`[feat0, attn0, feat1, attn1] = 1024`) remain unchanged. This keeps the
merge + cross-attn + policy/value heads **identical** to 031B and isolates
the encoder change as the only architectural delta.
"""

import math
from typing import Dict, Sequence

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import RLLIB_MODEL, _global_registry


torch, nn = try_import_torch()


TEAM_SIAMESE_PER_RAY_ATTN_MODEL_NAME = "team_siamese_per_ray_attn_model"


def _parse_hiddens(values: Sequence[int], default):
    if not values:
        return tuple(int(v) for v in default)
    return tuple(int(v) for v in values)


class _PerRayAttentionBlock(nn.Module):
    """Minimal pre-LN Transformer encoder block over ray tokens.

    Single-file implementation so we stay Ray 1.4 / PyTorch-era compatible
    without needing to thread a full `nn.TransformerEncoderLayer` config. The
    per-block shape is `(batch, n_tokens, embed_dim)` in/out.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"per-ray embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, embed_dim),
        )
        # Zero-init the last FFN linear so block ≈ identity at start
        # (graceful degrade: early training sees the shared ray embedding
        # directly, attention refinement ramps up).
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, tokens: "torch.Tensor") -> "torch.Tensor":
        # pre-LN attention sub-block
        tokens_ln = self.norm_attn(tokens)
        # nn.MultiheadAttention expects (seq, batch, dim)
        tokens_ln_t = tokens_ln.transpose(0, 1)
        attn_out, _ = self.mha(
            tokens_ln_t, tokens_ln_t, tokens_ln_t, need_weights=False
        )
        tokens = tokens + attn_out.transpose(0, 1)
        # pre-LN FFN sub-block
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        return tokens


class SiamesePerRayAttnTeamTorchModel(TorchModelV2, nn.Module):
    """083: Per-Ray Attention Siamese encoder + 031B cross-attention / merge.

    Per-agent flow:
      obs_336 → reshape (n_frames=3, n_rays=14, ray_feat=8)
             → permute+flatten → (n_rays=14, feat_dim=24)
             → linear embed per ray → (n_rays=14, embed_dim=64)
             + positional embedding (learned per-ray)
             → [PerRayAttentionBlock × N] → (n_rays=14, embed_dim=64)
             → pool (mean / cls / attn) → (embed_dim=64)
             → project_out (Linear 64 → 256 + ReLU) → agent_feat_256

    Two agents' 256-d features then go through the **unchanged** 031B
    cross-attention + merge + logits/value heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_dim = int(np.product(obs_space.shape))
        if obs_dim % 2 != 0:
            raise ValueError(
                f"SiamesePerRayAttnTeamTorchModel expects an even obs dim, got {obs_dim}."
            )
        self._half_obs_dim = int(obs_dim // 2)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        # Per-ray encoder knobs (task-pre-registered defaults)
        n_rays = int(custom_cfg.get("per_ray_n_rays", 14))
        ray_feat_dim = int(custom_cfg.get("per_ray_feat_dim", 24))
        embed_dim = int(custom_cfg.get("per_ray_embed_dim", 64))
        attn_layers = int(custom_cfg.get("per_ray_attn_layers", 2))
        attn_heads = int(custom_cfg.get("per_ray_attn_heads", 4))
        ffn_hidden = int(custom_cfg.get("per_ray_ffn_hidden", 128))
        pool_mode = str(custom_cfg.get("per_ray_pool", "mean")).strip().lower()
        if pool_mode not in {"mean", "cls", "attn"}:
            raise ValueError(
                f"per_ray_pool must be one of {{mean, cls, attn}}, got {pool_mode!r}."
            )

        # Downstream (031B) knobs — kept identical to preserve merge topology.
        encoder_hiddens = _parse_hiddens(custom_cfg.get("encoder_hiddens"), default=(256, 256))
        merge_hiddens = _parse_hiddens(custom_cfg.get("merge_hiddens"), default=(256, 128))
        n_tokens = int(custom_cfg.get("attention_n_tokens", 4))
        head_dim = int(custom_cfg.get("attention_head_dim", 64))

        encoder_out = int(encoder_hiddens[-1])
        if n_tokens * head_dim != encoder_out:
            raise ValueError(
                "Per-ray model cross-attn shape mismatch: "
                f"n_tokens ({n_tokens}) * head_dim ({head_dim}) = {n_tokens * head_dim} "
                f"!= encoder_out ({encoder_out})."
            )

        # Validate observation reshape math (soft-guard against upstream obs shape drifts).
        if n_rays * ray_feat_dim != self._half_obs_dim:
            raise ValueError(
                "Per-ray reshape mismatch: "
                f"n_rays ({n_rays}) * ray_feat_dim ({ray_feat_dim}) = {n_rays * ray_feat_dim} "
                f"!= half_obs_dim ({self._half_obs_dim}). "
                "For soccer_twos default 336-dim per-agent obs, use 14 × 24 = 336."
            )

        self._n_rays = n_rays
        self._ray_feat_dim = ray_feat_dim
        self._embed_dim = embed_dim
        self._attn_layers = attn_layers
        self._attn_heads = attn_heads
        self._ffn_hidden = ffn_hidden
        self._pool_mode = pool_mode
        self._encoder_out = encoder_out
        self._n_tokens = n_tokens
        self._head_dim = head_dim

        # ------------ Per-ray encoder (siamese / shared across agents) ------------
        self.ray_embed = nn.Linear(ray_feat_dim, embed_dim)
        # Learned positional embedding per-ray; ray index carries meaning
        # (e.g. forward vs back sensor banks), so absolute position is informative.
        self.pos_embed = nn.Parameter(torch.zeros(1, n_rays, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Optional [CLS] token for `cls` pooling
        if pool_mode == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.attn_blocks = nn.ModuleList(
            [
                _PerRayAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=attn_heads,
                    ffn_hidden=ffn_hidden,
                )
                for _ in range(attn_layers)
            ]
        )
        self.attn_norm_out = nn.LayerNorm(embed_dim)

        # Optional learned query for `attn` pooling
        if pool_mode == "attn":
            self.attn_pool_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.attn_pool_query, std=0.02)
            self.attn_pool_layer = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_heads,
                bias=True,
            )
        else:
            self.attn_pool_query = None
            self.attn_pool_layer = None

        # Project pooled embed_dim → encoder_out (256) + ReLU.
        self.project_out = nn.Sequential(
            nn.Linear(embed_dim, encoder_out),
            nn.ReLU(),
        )

        # ------------ 031B cross-attention (shared Q/K/V) ------------
        self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(head_dim, head_dim, bias=False)

        # ------------ 031B merge MLP + heads ------------
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
        self._metrics: Dict[str, float] = {
            "encoder_cos_sim_mean": 0.0,
            "per_ray_token_norm_mean": 0.0,
            "attention_entropy_mean": 0.0,
        }

    # ---------------- Per-agent encoding helpers ----------------
    def _reshape_obs_to_rays(self, obs_half: "torch.Tensor") -> "torch.Tensor":
        """(B, half_obs_dim) → (B, n_rays, ray_feat_dim).

        Default interpretation: half_obs_dim = n_rays * ray_feat_dim = 14 * 24 = 336,
        where each ray token carries its own 3-frame temporal history.

        We reshape the flat 336 obs as (B, 3 frames, 14 rays, 8 feats), permute to
        (B, 14 rays, 3 frames, 8 feats), then flatten the last two dims. This
        groups the same ray across frames into one (frame-aware) 24-d token.
        """
        batch = obs_half.shape[0]
        # Default path: 336 = 3 × 14 × 8; both defaults align to this.
        # We derive n_frames = ray_feat_dim // 8 if ray_feat_dim is a multiple
        # of 8; otherwise we fall back to a plain (n_rays, ray_feat_dim) reshape
        # and trust the caller.
        if self._ray_feat_dim % 8 == 0 and self._ray_feat_dim >= 8:
            n_frames = self._ray_feat_dim // 8
            ray_inner_feat = 8
            # (B, n_frames, n_rays, ray_inner_feat)
            reshaped = obs_half.view(batch, n_frames, self._n_rays, ray_inner_feat)
            # (B, n_rays, n_frames, ray_inner_feat)
            permuted = reshaped.permute(0, 2, 1, 3).contiguous()
            # (B, n_rays, ray_feat_dim)
            return permuted.view(batch, self._n_rays, self._ray_feat_dim)
        # Fallback (no temporal collapse): treat flat obs as already ray-major.
        return obs_half.view(batch, self._n_rays, self._ray_feat_dim)

    def _encode_agent(self, obs_half: "torch.Tensor") -> "torch.Tensor":
        """(B, half_obs_dim) → (B, encoder_out=256) agent feature."""
        rays = self._reshape_obs_to_rays(obs_half)  # (B, 14, 24)
        tokens = self.ray_embed(rays)               # (B, 14, 64)
        tokens = tokens + self.pos_embed            # broadcast over batch

        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, 15, 64)

        for block in self.attn_blocks:
            tokens = block(tokens)
        tokens = self.attn_norm_out(tokens)

        if self._pool_mode == "mean":
            pooled = tokens.mean(dim=1)
        elif self._pool_mode == "cls":
            pooled = tokens[:, 0, :]
        else:  # "attn"
            q = self.attn_pool_query.expand(tokens.shape[0], -1, -1)
            # MHA API: (seq, batch, dim)
            q_t = q.transpose(0, 1)
            kv_t = tokens.transpose(0, 1)
            pooled_t, _ = self.attn_pool_layer(q_t, kv_t, kv_t, need_weights=False)
            pooled = pooled_t.transpose(0, 1).squeeze(1)

        agent_feat = self.project_out(pooled)  # (B, 256)
        return agent_feat, tokens

    # ---------------- 031B cross-attention (same as SiameseCrossAttentionTeamTorchModel) ----------------
    def _cross_attend(self, src_tokens, tgt_tokens):
        q = self.q_proj(src_tokens)
        k = self.k_proj(tgt_tokens)
        v = self.v_proj(tgt_tokens)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self._head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out, attn

    # ---------------- Forward ----------------
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs0 = obs[:, : self._half_obs_dim]
        obs1 = obs[:, self._half_obs_dim :]

        feat0, ray_tokens0 = self._encode_agent(obs0)  # (B, 256), (B, n_rays[+1], embed_dim)
        feat1, ray_tokens1 = self._encode_agent(obs1)

        batch_size = feat0.shape[0]
        cross_tokens0 = feat0.view(batch_size, self._n_tokens, self._head_dim)
        cross_tokens1 = feat1.view(batch_size, self._n_tokens, self._head_dim)

        attn0_out, attn0_weights = self._cross_attend(cross_tokens0, cross_tokens1)
        attn1_out, attn1_weights = self._cross_attend(cross_tokens1, cross_tokens0)

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
            token_norm = torch.cat(
                [
                    ray_tokens0.norm(dim=-1).mean(dim=1),
                    ray_tokens1.norm(dim=-1).mean(dim=1),
                ],
                dim=0,
            )
            self._metrics = {
                "encoder_cos_sim_mean": float(cos.detach().cpu().item()),
                "attention_entropy_mean": float(ent_all.mean().detach().cpu().item()),
                "per_ray_token_norm_mean": float(token_norm.mean().detach().cpu().item()),
                "per_ray_attn_layers": float(self._attn_layers),
                "per_ray_n_rays": float(self._n_rays),
            }

        return logits, state

    def value_function(self):
        if self._value_out is None:
            raise ValueError("value_function() called before forward().")
        return self._value_out

    def metrics(self) -> Dict[str, float]:
        return dict(self._metrics)


def register_team_siamese_per_ray_model():
    try:
        ModelCatalog.register_custom_model(
            TEAM_SIAMESE_PER_RAY_ATTN_MODEL_NAME,
            SiamesePerRayAttnTeamTorchModel,
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL,
            TEAM_SIAMESE_PER_RAY_ATTN_MODEL_NAME,
            SiamesePerRayAttnTeamTorchModel,
        )
    except ValueError:
        pass
