import itertools
from typing import Dict, List, Sequence

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import RLLIB_MODEL, _global_registry

from cs8803drl.branches.team_siamese import (
    SiameseCrossAttentionTeamTorchModel,
    SiameseTeamTorchModel,
)
from cs8803drl.core.checkpoint_utils import extract_torch_weights_from_checkpoint


torch, nn = try_import_torch()


TEAM_SIAMESE_DISTILL_MODEL_NAME = "team_siamese_distill_model"
TEAM_SIAMESE_ENSEMBLE_DISTILL_MODEL_NAME = "team_siamese_ensemble_distill_model"


def _parse_action_spec(action_space):
    nvec = np.asarray(getattr(action_space, "nvec", []), dtype=np.int64).reshape(-1)
    if nvec.size == 0 or nvec.size % 2 != 0:
        raise ValueError(
            "TeamSiameseDistill model expects an even-length MultiDiscrete action space, "
            f"got nvec={nvec!r}"
        )
    if not np.all(nvec == nvec[0]):
        raise ValueError(
            "TeamSiameseDistill currently expects equal-sized discrete factors, "
            f"got nvec={nvec!r}"
        )
    joint_dims = int(nvec.size)
    agent_dims = int(joint_dims // 2)
    classes = int(nvec[0])
    return joint_dims, agent_dims, classes


def _flat_action_tuples(action_dims, classes):
    return list(itertools.product(*[range(classes) for _ in range(action_dims)]))


class _FrozenPerAgentTeacher(nn.Module):
    """Lightweight frozen 029B actor reconstructed from checkpoint weights."""

    def __init__(self, checkpoint_path, *, policy_id="shared_cc_policy"):
        super().__init__()
        weights = extract_torch_weights_from_checkpoint(checkpoint_path, policy_name=policy_id)

        hidden0_w = torch.as_tensor(
            np.asarray(weights["action_model._hidden_layers.0._model.0.weight"])
        ).float()
        hidden0_b = torch.as_tensor(
            np.asarray(weights["action_model._hidden_layers.0._model.0.bias"])
        ).float()
        hidden1_w = torch.as_tensor(
            np.asarray(weights["action_model._hidden_layers.1._model.0.weight"])
        ).float()
        hidden1_b = torch.as_tensor(
            np.asarray(weights["action_model._hidden_layers.1._model.0.bias"])
        ).float()
        logits_w = torch.as_tensor(
            np.asarray(weights["action_model._logits._model.0.weight"])
        ).float()
        logits_b = torch.as_tensor(
            np.asarray(weights["action_model._logits._model.0.bias"])
        ).float()

        obs_dim = int(hidden0_w.shape[1])
        hidden0 = int(hidden0_w.shape[0])
        hidden1 = int(hidden1_w.shape[0])
        action_dim = int(logits_w.shape[0])

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden0),
            nn.ReLU(),
            nn.Linear(hidden0, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, action_dim),
        )
        with torch.no_grad():
            self.net[0].weight.copy_(hidden0_w)
            self.net[0].bias.copy_(hidden0_b)
            self.net[2].weight.copy_(hidden1_w)
            self.net[2].bias.copy_(hidden1_b)
            self.net[4].weight.copy_(logits_w)
            self.net[4].bias.copy_(logits_b)

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, obs):
        return self.net(obs)


class SiameseTeamDistillTorchModel(SiameseTeamTorchModel):
    """031A-style Siamese student with factor-wise KL to frozen 029B teacher."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        teacher_checkpoint = (custom_cfg.get("teacher_checkpoint") or "").strip()
        if not teacher_checkpoint:
            raise ValueError(
                "TEAM_DISTILL_KL requires custom_model_config.teacher_checkpoint."
            )
        teacher_policy_id = (custom_cfg.get("teacher_policy_id") or "shared_cc_policy").strip()

        self._joint_action_dims, self._agent_action_dims, self._factor_classes = _parse_action_spec(
            action_space
        )
        teacher_action_dim = int(self._factor_classes ** self._agent_action_dims)

        tuples = _flat_action_tuples(self._agent_action_dims, self._factor_classes)
        if len(tuples) != teacher_action_dim:
            raise ValueError(
                "Teacher flat action size mismatch for factorized KL: "
                f"{len(tuples)} != {teacher_action_dim}"
            )
        marginal_matrix = torch.zeros(
            self._agent_action_dims,
            self._factor_classes,
            teacher_action_dim,
            dtype=torch.float32,
        )
        for flat_idx, action_tuple in enumerate(tuples):
            for dim_idx, cls_idx in enumerate(action_tuple):
                marginal_matrix[dim_idx, cls_idx, flat_idx] = 1.0
        self.register_buffer("_teacher_marginal_matrix", marginal_matrix)

        self.teacher_model = _FrozenPerAgentTeacher(
            teacher_checkpoint, policy_id=teacher_policy_id
        )
        self._teacher_obs_dim = int(self.teacher_model.net[0].in_features)
        if self._teacher_obs_dim != self._half_obs_dim:
            raise ValueError(
                "Teacher/student obs mismatch: "
                f"teacher expects {self._teacher_obs_dim}, student half obs is {self._half_obs_dim}"
            )

        self._distill_alpha_init = float(custom_cfg.get("distill_alpha_init", 0.02))
        self._distill_alpha_final = float(custom_cfg.get("distill_alpha_final", 0.0))
        self._distill_decay_updates = max(1, int(custom_cfg.get("distill_decay_updates", 16000)))
        self._distill_temperature = float(custom_cfg.get("distill_temperature", 1.0))
        self._distill_updates = 0

        self._student_factor_logits = None
        self._last_obs0 = None
        self._last_obs1 = None
        self._distill_metrics = {
            "distill_kl": 0.0,
            "distill_alpha": self._distill_alpha_init,
            "distill_teacher_entropy": 0.0,
        }

    def _current_alpha(self):
        progress = min(1.0, float(self._distill_updates) / float(self._distill_decay_updates))
        return (1.0 - progress) * self._distill_alpha_init + progress * self._distill_alpha_final

    def _teacher_factor_probs(self, obs):
        teacher_logits = self.teacher_model(obs)
        temp = max(self._distill_temperature, 1e-6)
        teacher_probs = torch.softmax(teacher_logits / temp, dim=-1)
        factor_probs = torch.einsum(
            "bk,dck->bdc", teacher_probs, self._teacher_marginal_matrix.to(teacher_probs.device)
        )
        teacher_entropy = -(teacher_probs * (teacher_probs + 1e-9).log()).sum(dim=-1).mean()
        return factor_probs, teacher_entropy

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_obs0 = obs[:, : self._half_obs_dim]
        self._last_obs1 = obs[:, self._half_obs_dim :]
        logits, state_out = super().forward(input_dict, state, seq_lens)
        self._student_factor_logits = logits.reshape(
            -1, self._joint_action_dims, self._factor_classes
        )
        return logits, state_out

    def custom_loss(self, policy_loss, loss_inputs):
        if (
            self._student_factor_logits is None
            or self._last_obs0 is None
            or self._last_obs1 is None
        ):
            return policy_loss

        with torch.no_grad():
            teacher_probs0, teacher_entropy0 = self._teacher_factor_probs(self._last_obs0)
            teacher_probs1, teacher_entropy1 = self._teacher_factor_probs(self._last_obs1)
            teacher_probs = torch.cat([teacher_probs0, teacher_probs1], dim=1)
            teacher_entropy = 0.5 * (teacher_entropy0 + teacher_entropy1)

        temp = max(self._distill_temperature, 1e-6)
        student_log_probs = torch.log_softmax(self._student_factor_logits / temp, dim=-1)
        kl_per_factor = teacher_probs * (
            (teacher_probs + 1e-9).log() - student_log_probs
        )
        kl = kl_per_factor.sum(dim=-1).mean() * (temp ** 2)
        alpha = self._current_alpha()

        self._distill_metrics = {
            "distill_kl": float(kl.detach().cpu().item()),
            "distill_alpha": float(alpha),
            "distill_teacher_entropy": float(teacher_entropy.detach().cpu().item()),
        }
        self._distill_updates += 1

        distill_term = alpha * kl
        if isinstance(policy_loss, (list, tuple)):
            return [loss_ + distill_term for loss_ in policy_loss]
        return policy_loss + distill_term

    def metrics(self) -> Dict[str, float]:
        metrics = super().metrics()
        metrics.update(self._distill_metrics)
        return metrics


def register_team_siamese_distill_model():
    try:
        ModelCatalog.register_custom_model(
            TEAM_SIAMESE_DISTILL_MODEL_NAME, SiameseTeamDistillTorchModel
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL, TEAM_SIAMESE_DISTILL_MODEL_NAME, SiameseTeamDistillTorchModel
        )
    except ValueError:
        pass


def _detect_team_arch_from_weights(weights: Dict[str, np.ndarray]) -> str:
    """Auto-detect team-level model architecture from checkpoint weight keys.

    Returns one of:
      - "cross_attention" : 031B-arch (q_proj/k_proj/v_proj present)
      - "siamese_only"    : 031A-arch (no attention)
    """
    has_attn = any(k in weights for k in ("q_proj.weight", "k_proj.weight", "v_proj.weight"))
    return "cross_attention" if has_attn else "siamese_only"


def _build_frozen_team_siamese_from_checkpoint(checkpoint_path: str, obs_space, action_space):
    """Build a frozen team-level Siamese teacher (031A or 031B-arch) from a checkpoint.

    Auto-detects architecture from weight keys. The teacher is set to eval mode
    with all parameters frozen. Output of forward is (B, num_outputs) factored
    logits matching the team-level MultiDiscrete action space.
    """
    weights = extract_torch_weights_from_checkpoint(checkpoint_path, policy_name="default_policy")
    arch = _detect_team_arch_from_weights(weights)

    # Determine num_outputs from logits_layer shape
    logits_w = np.asarray(weights["logits_layer.weight"])
    num_outputs = int(logits_w.shape[0])

    # Determine encoder hidden sizes from shared_encoder weights
    enc0_w = np.asarray(weights["shared_encoder.0.weight"])  # (h0, half_obs_dim)
    enc2_w = np.asarray(weights["shared_encoder.2.weight"])  # (h1, h0)
    encoder_hiddens = (int(enc0_w.shape[0]), int(enc2_w.shape[0]))

    # Determine merge hidden sizes from merge_mlp weights
    merge0_w = np.asarray(weights["merge_mlp.0.weight"])  # (m0, merge_in)
    merge2_w = np.asarray(weights["merge_mlp.2.weight"])  # (m1, m0)
    merge_hiddens = (int(merge0_w.shape[0]), int(merge2_w.shape[0]))

    # Build appropriate model class with reconstructed config
    custom_cfg = {
        "encoder_hiddens": encoder_hiddens,
        "merge_hiddens": merge_hiddens,
    }
    if arch == "cross_attention":
        # Determine n_tokens, head_dim from q_proj shape (square head_dim x head_dim)
        q_w = np.asarray(weights["q_proj.weight"])
        head_dim = int(q_w.shape[0])
        encoder_out = encoder_hiddens[-1]
        if encoder_out % head_dim != 0:
            raise ValueError(
                f"Cross-attn head_dim ({head_dim}) does not divide encoder output "
                f"dim ({encoder_out})."
            )
        n_tokens = encoder_out // head_dim
        custom_cfg["attention_n_tokens"] = n_tokens
        custom_cfg["attention_head_dim"] = head_dim
        model = SiameseCrossAttentionTeamTorchModel(
            obs_space, action_space, num_outputs,
            {"custom_model_config": custom_cfg}, "frozen_teacher_cross_attn",
        )
    else:
        model = SiameseTeamTorchModel(
            obs_space, action_space, num_outputs,
            {"custom_model_config": custom_cfg}, "frozen_teacher_siamese",
        )

    # Load weights (convert np arrays to torch tensors)
    state_dict = {k: torch.as_tensor(np.asarray(v)).float() for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(
            f"Unexpected keys when loading frozen teacher from {checkpoint_path}: {unexpected}"
        )
    if missing:
        # Some keys may be metric buffers; warn but don't fail
        non_buffer_missing = [k for k in missing if not k.startswith("_")]
        if non_buffer_missing:
            raise ValueError(
                f"Missing keys when loading frozen teacher from {checkpoint_path}: "
                f"{non_buffer_missing}"
            )

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


class _FrozenTeamEnsembleTeacher(nn.Module):
    """Frozen ensemble of N team-level teachers. Returns averaged factor probs.

    Each member outputs (B, num_outputs) factored logits where num_outputs =
    sum(action_space.nvec) = joint_action_dims * factor_classes. For
    SoccerTwos team-level: 6 dims * 3 classes = 18 logits, reshaped to
    (B, 6, 3) factored probs per teacher, then averaged across teachers.

    Output: (B, joint_action_dims, factor_classes) ensemble factor probs.
    """

    def __init__(
        self,
        checkpoint_paths: List[str],
        obs_space,
        action_space,
        joint_action_dims: int,
        factor_classes: int,
    ):
        super().__init__()
        if not checkpoint_paths:
            raise ValueError("Ensemble teacher needs at least one checkpoint path")
        self.teachers = nn.ModuleList([
            _build_frozen_team_siamese_from_checkpoint(p, obs_space, action_space)
            for p in checkpoint_paths
        ])
        self._joint_action_dims = int(joint_action_dims)
        self._factor_classes = int(factor_classes)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, joint_obs: "torch.Tensor") -> "torch.Tensor":
        """Returns (B, joint_action_dims, factor_classes) averaged factor probs."""
        input_dict = {"obs_flat": joint_obs, "obs": joint_obs}
        all_factor_probs = []
        for teacher in self.teachers:
            logits, _ = teacher.forward(input_dict, [], None)  # (B, num_outputs)
            factor_logits = logits.reshape(-1, self._joint_action_dims, self._factor_classes)
            factor_probs = torch.softmax(factor_logits, dim=-1)
            all_factor_probs.append(factor_probs)
        avg_factor_probs = torch.stack(all_factor_probs, dim=0).mean(dim=0)
        return avg_factor_probs


class SiameseTeamEnsembleDistillTorchModel(SiameseCrossAttentionTeamTorchModel):
    """031B-arch student with KL distillation from a TEAM-LEVEL ENSEMBLE of teachers.

    Differs from SiameseTeamDistillTorchModel (which distills from a single
    per-agent flat teacher) in two ways:
      1. Student is 031B-arch (cross-attention), not plain Siamese.
      2. Teacher is N team-level checkpoints; KL is computed against the
         per-factor average of each teacher's softmaxed factored logits.

    Used for Tier S1 distillation of 034E (031B + 045A + 051A) ensemble.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        ensemble_paths_str = (custom_cfg.get("teacher_ensemble_checkpoints") or "").strip()
        if not ensemble_paths_str:
            raise ValueError(
                "TEAM_DISTILL_ENSEMBLE_KL requires "
                "custom_model_config.teacher_ensemble_checkpoints (comma-separated)."
            )
        ensemble_paths = [p.strip() for p in ensemble_paths_str.split(",") if p.strip()]
        if len(ensemble_paths) < 2:
            raise ValueError(
                f"Ensemble distillation requires >= 2 teacher checkpoints, "
                f"got {len(ensemble_paths)}."
            )

        joint_action_dims, _, factor_classes = _parse_action_spec(action_space)
        self._joint_action_dims = joint_action_dims
        self._factor_classes = factor_classes

        self.teacher_model = _FrozenTeamEnsembleTeacher(
            ensemble_paths, obs_space, action_space,
            joint_action_dims=joint_action_dims, factor_classes=factor_classes,
        )

        self._distill_alpha_init = float(custom_cfg.get("distill_alpha_init", 0.05))
        self._distill_alpha_final = float(custom_cfg.get("distill_alpha_final", 0.0))
        self._distill_decay_updates = max(1, int(custom_cfg.get("distill_decay_updates", 8000)))
        self._distill_temperature = float(custom_cfg.get("distill_temperature", 1.0))
        self._distill_updates = 0

        self._student_factor_logits = None
        self._last_obs = None
        self._distill_metrics = {
            "distill_kl": 0.0,
            "distill_alpha": self._distill_alpha_init,
            "distill_teacher_entropy": 0.0,
            "distill_n_teachers": float(len(ensemble_paths)),
        }

    def _current_alpha(self) -> float:
        progress = min(1.0, float(self._distill_updates) / float(self._distill_decay_updates))
        return (1.0 - progress) * self._distill_alpha_init + progress * self._distill_alpha_final

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_obs = obs
        logits, state_out = super().forward(input_dict, state, seq_lens)
        self._student_factor_logits = logits.reshape(
            -1, self._joint_action_dims, self._factor_classes
        )
        return logits, state_out

    def custom_loss(self, policy_loss, loss_inputs):
        if self._student_factor_logits is None or self._last_obs is None:
            return policy_loss

        with torch.no_grad():
            ensemble_factor_probs = self.teacher_model(self._last_obs)
            teacher_entropy = -(
                ensemble_factor_probs * (ensemble_factor_probs + 1e-9).log()
            ).sum(dim=-1).mean()

        temp = max(self._distill_temperature, 1e-6)
        student_log_probs = torch.log_softmax(self._student_factor_logits / temp, dim=-1)
        kl_per_factor = ensemble_factor_probs * (
            (ensemble_factor_probs + 1e-9).log() - student_log_probs
        )
        kl = kl_per_factor.sum(dim=-1).mean() * (temp ** 2)
        alpha = self._current_alpha()

        self._distill_metrics["distill_kl"] = float(kl.detach().cpu().item())
        self._distill_metrics["distill_alpha"] = float(alpha)
        self._distill_metrics["distill_teacher_entropy"] = float(
            teacher_entropy.detach().cpu().item()
        )
        self._distill_updates += 1

        distill_term = alpha * kl
        if isinstance(policy_loss, (list, tuple)):
            return [loss_ + distill_term for loss_ in policy_loss]
        return policy_loss + distill_term

    def metrics(self) -> Dict[str, float]:
        metrics = super().metrics()
        metrics.update(self._distill_metrics)
        return metrics


def register_team_siamese_ensemble_distill_model():
    try:
        ModelCatalog.register_custom_model(
            TEAM_SIAMESE_ENSEMBLE_DISTILL_MODEL_NAME, SiameseTeamEnsembleDistillTorchModel
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL, TEAM_SIAMESE_ENSEMBLE_DISTILL_MODEL_NAME,
            SiameseTeamEnsembleDistillTorchModel,
        )
    except ValueError:
        pass
