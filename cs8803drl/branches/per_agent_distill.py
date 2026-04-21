"""Per-agent student distillation from a team-level teacher ensemble (DIR-B).

This module implements the snapshot-077 DIR-B lane:
  * Student = per-agent shared-policy (shared centralized-critic arch, Discrete(27)
    flat action). One policy shared across agents 0/1.
  * Teacher = ensemble of team-level (672-dim obs, MultiDiscrete([3]*6)) Siamese
    + cross-attention models (same as 055).
  * Distillation loss is the KL between the student's per-agent factored marginal
    (3 factors x 3 classes over the flat 27-dim action) and the team teacher's
    factors 0-2 (slot-0 convention).

Slot-0 convention
-----------------
The per-agent env (``EnvType.multiagent_player``) dispatches ``own_obs`` and
``teammate_obs`` for both agents through the shared policy. At distill time we
always feed the team teacher ``cat(own_obs, teammate_obs)`` -- i.e. the calling
agent always occupies slot 0, its teammate slot 1. We then take the teacher's
first 3 factor-probs (factors 0,1,2 = agent-0 of the joint policy) as the KL
target. The student, which shares weights across agents, therefore learns a
single "this-agent-from-teacher's-slot-0-view" response. This is consistent and
does not bias agent 0 vs agent 1.

A `replace_all`-style symmetric variant (average of slot-0 and slot-1 views) is
left for a follow-up; the asymmetric slot-0 convention is the simplest
minimally-invasive baseline.

Related files
-------------
  * ``cs8803drl/branches/shared_central_critic.py`` — base per-agent model
  * ``cs8803drl/branches/team_siamese_distill.py`` — team-level ensemble teacher
  * ``cs8803drl/training/train_ray_mappo_vs_baseline.py`` — per-agent trainer
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import RLLIB_MODEL, _global_registry

from cs8803drl.branches.team_siamese_distill import (
    _build_frozen_team_siamese_from_checkpoint,
    _parse_action_spec,
)
from cs8803drl.branches.shared_central_critic import (
    _ensure_modelcatalog_tf_compat,
)


torch, nn = try_import_torch()


PER_AGENT_DISTILL_MODEL_NAME = "per_agent_distill_model"

# Per-agent action factorisation (SoccerTwos): 3 discrete factors x 3 classes
# each, flattened to Discrete(27). Factor decomposition is (f0,f1,f2) with
# f_i in {0,1,2}. See ``soccer_twos.wrappers.ActionFlattener``.
_DEFAULT_PER_AGENT_FACTORS = 3
_DEFAULT_PER_AGENT_CLASSES = 3


def _build_per_agent_marginal_matrix(num_factors: int, num_classes: int) -> torch.Tensor:
    """Return a (num_factors, num_classes, flat_dim) 0/1 marginalisation tensor.

    Mapping follows ``itertools.product(range(num_classes), repeat=num_factors)``
    in row-major order, matching ``soccer_twos.wrappers.ActionFlattener``'s
    enumeration of joint actions when ``nvec == (3,3,3)``.
    """
    flat_dim = int(num_classes ** num_factors)
    tuples = list(
        itertools.product(*[range(num_classes) for _ in range(num_factors)])
    )
    if len(tuples) != flat_dim:
        raise ValueError(
            f"Flat action count mismatch: got {len(tuples)}, expected {flat_dim}"
        )
    mat = torch.zeros(num_factors, num_classes, flat_dim, dtype=torch.float32)
    for flat_idx, action_tuple in enumerate(tuples):
        for dim_idx, cls_idx in enumerate(action_tuple):
            mat[dim_idx, cls_idx, flat_idx] = 1.0
    return mat


def _infer_team_action_space(single_agent_factors: int, single_agent_classes: int):
    """Construct a dummy MultiDiscrete 6-factor space for teacher loader."""
    import gym

    nvec = np.asarray(
        [single_agent_classes] * (2 * single_agent_factors), dtype=np.int64
    )
    return gym.spaces.MultiDiscrete(nvec)


def _infer_team_obs_space(per_agent_obs_dim: int):
    import gym

    dim = 2 * int(per_agent_obs_dim)
    low = -np.inf * np.ones((dim,), dtype=np.float32)
    high = np.inf * np.ones((dim,), dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


class _FrozenTeamEnsemblePerAgentTeacher(nn.Module):
    """Wraps N team-level Siamese(+cross-attn) teachers into a per-agent KL target.

    forward(cat_obs) expects (B, 2 * per_agent_obs_dim) = (B, 672) joint team
    obs with slot-0 = the focal agent, slot-1 = its teammate. Returns a
    ``(B, num_factors, num_classes)`` tensor of ensemble-averaged factor probs
    for agent 0 (factors 0..2 of the team-level output).
    """

    def __init__(
        self,
        checkpoint_paths: List[str],
        per_agent_obs_dim: int,
        *,
        num_factors: int = _DEFAULT_PER_AGENT_FACTORS,
        num_classes: int = _DEFAULT_PER_AGENT_CLASSES,
        teacher_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        if not checkpoint_paths:
            raise ValueError("Per-agent distill teacher needs >=1 checkpoint path.")
        team_action_space = _infer_team_action_space(num_factors, num_classes)
        team_obs_space = _infer_team_obs_space(per_agent_obs_dim)

        self.teachers = nn.ModuleList([
            _build_frozen_team_siamese_from_checkpoint(
                p, team_obs_space, team_action_space
            )
            for p in checkpoint_paths
        ])

        if teacher_weights is None:
            n = len(self.teachers)
            teacher_weights = [1.0 / n] * n
        else:
            if len(teacher_weights) != len(self.teachers):
                raise ValueError(
                    f"teacher_weights length {len(teacher_weights)} != "
                    f"teacher count {len(self.teachers)}"
                )
            total = sum(teacher_weights)
            if total <= 0:
                raise ValueError("teacher_weights must sum to a positive value")
            teacher_weights = [float(w) / total for w in teacher_weights]
        self._teacher_weights_list = list(teacher_weights)

        # Full joint factor count on the team teacher = 2 * num_factors
        self._joint_action_dims = 2 * int(num_factors)
        self._num_factors = int(num_factors)
        self._num_classes = int(num_classes)
        self._per_agent_obs_dim = int(per_agent_obs_dim)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @property
    def per_agent_obs_dim(self) -> int:
        return self._per_agent_obs_dim

    def forward(self, joint_obs: "torch.Tensor") -> "torch.Tensor":
        """Returns (B, num_factors, num_classes) ensemble-averaged factor probs
        for the slot-0 agent (factors 0..num_factors-1 of the team teacher)."""
        input_dict = {"obs_flat": joint_obs, "obs": joint_obs}
        all_agent0_factor_probs = []
        for teacher in self.teachers:
            logits, _ = teacher.forward(input_dict, [], None)  # (B, 18) for 6x3
            factor_logits = logits.reshape(
                -1, self._joint_action_dims, self._num_classes
            )
            factor_probs = torch.softmax(factor_logits, dim=-1)  # (B, 6, 3)
            # Slice factors 0..num_factors-1 = agent-0's factors
            agent0_factor_probs = factor_probs[:, : self._num_factors, :]
            all_agent0_factor_probs.append(agent0_factor_probs)
        stacked = torch.stack(all_agent0_factor_probs, dim=0)  # (N, B, f, c)
        weights = torch.tensor(
            self._teacher_weights_list, dtype=stacked.dtype, device=stacked.device
        ).view(-1, 1, 1, 1)
        return (stacked * weights).sum(dim=0)


class PerAgentSharedPolicyDistillTorchModel(TorchModelV2, nn.Module):
    """Per-agent shared-policy student with KL distill from a team-level teacher.

    The student is the standard centralized-critic per-agent model (FC action
    head on ``own_obs``, FC value head on flattened obs). We add a distill term
    to the custom loss that targets the team-level teacher's slot-0 factor probs
    on the joint obs ``cat(own_obs, teammate_obs)``.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        original = (
            obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        )
        own_obs_space = original.spaces["own_obs"]
        teammate_obs_space = original.spaces["teammate_obs"]
        self._own_obs_dim = int(np.prod(own_obs_space.shape))
        self._teammate_obs_dim = int(np.prod(teammate_obs_space.shape))
        if self._own_obs_dim != self._teammate_obs_dim:
            raise ValueError(
                "PerAgentSharedPolicyDistillTorchModel expects own_obs and "
                f"teammate_obs of equal dim, got {self._own_obs_dim} vs "
                f"{self._teammate_obs_dim}"
            )

        self.action_model = TorchFC(
            own_obs_space, action_space, num_outputs, model_config, name + "_action"
        )
        self.value_model = TorchFC(
            obs_space, action_space, 1, model_config, name + "_vf"
        )
        self._value_out = None

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        ensemble_paths_str = (
            custom_cfg.get("teacher_ensemble_checkpoints") or ""
        ).strip()
        if not ensemble_paths_str:
            raise ValueError(
                "PER_AGENT_STUDENT_DISTILL requires "
                "custom_model_config.teacher_ensemble_checkpoints (comma-sep)."
            )
        ensemble_paths = [
            p.strip() for p in ensemble_paths_str.split(",") if p.strip()
        ]
        if not ensemble_paths:
            raise ValueError(
                "Per-agent distill needs >=1 teacher checkpoint; none parsed."
            )

        teacher_weights_str = (custom_cfg.get("teacher_weights") or "").strip()
        if teacher_weights_str:
            try:
                teacher_weights = [
                    float(w.strip()) for w in teacher_weights_str.split(",")
                ]
            except ValueError as e:
                raise ValueError(
                    f"Malformed teacher_weights '{teacher_weights_str}': {e}"
                )
        else:
            teacher_weights = None

        self._num_factors = int(
            custom_cfg.get("distill_num_factors", _DEFAULT_PER_AGENT_FACTORS)
        )
        self._num_classes = int(
            custom_cfg.get("distill_num_classes", _DEFAULT_PER_AGENT_CLASSES)
        )
        expected_flat = self._num_classes ** self._num_factors
        if int(num_outputs) != expected_flat:
            raise ValueError(
                f"Student expects num_outputs={expected_flat} for "
                f"{self._num_factors}x{self._num_classes} factorisation, got "
                f"{int(num_outputs)}."
            )

        self.teacher_model = _FrozenTeamEnsemblePerAgentTeacher(
            ensemble_paths,
            per_agent_obs_dim=self._own_obs_dim,
            num_factors=self._num_factors,
            num_classes=self._num_classes,
            teacher_weights=teacher_weights,
        )

        marginal_matrix = _build_per_agent_marginal_matrix(
            self._num_factors, self._num_classes
        )
        self.register_buffer("_student_marginal_matrix", marginal_matrix)

        self._distill_alpha_init = float(custom_cfg.get("distill_alpha_init", 0.05))
        self._distill_alpha_final = float(custom_cfg.get("distill_alpha_final", 0.0))
        self._distill_decay_updates = max(
            1, int(custom_cfg.get("distill_decay_updates", 8000))
        )
        self._distill_temperature = float(
            custom_cfg.get("distill_temperature", 1.0)
        )
        self._distill_updates = 0

        # Caches for custom_loss (filled in forward).
        self._student_flat_logits = None
        self._last_own_obs = None
        self._last_teammate_obs = None
        self._distill_metrics = {
            "distill_kl": 0.0,
            "distill_alpha": self._distill_alpha_init,
            "distill_teacher_entropy": 0.0,
            "distill_n_teachers": float(len(ensemble_paths)),
        }

    def _current_alpha(self) -> float:
        progress = min(
            1.0, float(self._distill_updates) / float(self._distill_decay_updates)
        )
        return (
            (1.0 - progress) * self._distill_alpha_init
            + progress * self._distill_alpha_final
        )

    def forward(self, input_dict, state, seq_lens):
        self._value_out, _ = self.value_model(
            {"obs": input_dict["obs_flat"]}, state, seq_lens
        )
        # The shared-CC observer packs {own_obs, teammate_obs, teammate_action}
        # into the Dict obs; RLlib routes both flattened and dict forms through
        # input_dict. Grab the dict form for the per-component obs.
        obs_dict = input_dict["obs"]
        own_obs = obs_dict["own_obs"].float()
        teammate_obs = obs_dict["teammate_obs"].float()
        self._last_own_obs = own_obs
        self._last_teammate_obs = teammate_obs

        logits, state_out = self.action_model(
            {"obs": own_obs}, state, seq_lens
        )
        self._student_flat_logits = logits
        return logits, state_out

    def value_function(self):
        return torch.reshape(self._value_out, [-1])

    def custom_loss(self, policy_loss, loss_inputs):
        if (
            self._student_flat_logits is None
            or self._last_own_obs is None
            or self._last_teammate_obs is None
        ):
            return policy_loss

        temp = max(self._distill_temperature, 1e-6)

        # Student: flat 27 logits -> softmax to flat probs -> marginalise to
        # (B, num_factors, num_classes) factor probs.
        student_flat_probs = torch.softmax(
            self._student_flat_logits / temp, dim=-1
        )
        student_factor_probs = torch.einsum(
            "bk,dck->bdc",
            student_flat_probs,
            self._student_marginal_matrix.to(student_flat_probs.device),
        )
        student_log_probs = torch.log(student_factor_probs + 1e-9)

        with torch.no_grad():
            joint_obs = torch.cat(
                [self._last_own_obs, self._last_teammate_obs], dim=-1
            )
            teacher_factor_probs = self.teacher_model(joint_obs)  # (B, f, c)
            teacher_entropy = -(
                teacher_factor_probs * (teacher_factor_probs + 1e-9).log()
            ).sum(dim=-1).mean()

        kl_per_factor = teacher_factor_probs * (
            (teacher_factor_probs + 1e-9).log() - student_log_probs
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
        return dict(self._distill_metrics)


def register_per_agent_distill_model():
    _ensure_modelcatalog_tf_compat()
    try:
        ModelCatalog.register_custom_model(
            PER_AGENT_DISTILL_MODEL_NAME, PerAgentSharedPolicyDistillTorchModel
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL,
            PER_AGENT_DISTILL_MODEL_NAME,
            PerAgentSharedPolicyDistillTorchModel,
        )
    except ValueError:
        pass
