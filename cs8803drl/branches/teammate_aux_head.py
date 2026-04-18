import os
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

from cs8803drl.branches.shared_central_critic import (
    FillInTeammateActions,
    SHARED_CC_POLICY_ID,
    SharedCentralCriticTorchModel,
    teammate_id,
)
from cs8803drl.core.obs_teammate import TEAMMATE_STATE_DIM, extract_own_player_state

torch, nn = try_import_torch()


TEAMMATE_AUX_MODEL_NAME = "shared_cc_teammate_aux_model"
TEAMMATE_AUX_LABEL = "teammate_aux_label"
TEAMMATE_AUX_VALID = "teammate_aux_valid"
DEFAULT_AUX_SCALE = (15.0, 7.0, 5.0, 5.0)


def _parse_scale(value: Optional[Iterable[float]]) -> np.ndarray:
    if value is None:
        return np.asarray(DEFAULT_AUX_SCALE, dtype=np.float32)
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if len(pieces) != TEAMMATE_STATE_DIM:
            raise ValueError(
                f"AUX_TEAMMATE_LABEL_SCALE must provide {TEAMMATE_STATE_DIM} comma-separated values."
            )
        return np.asarray([float(piece) for piece in pieces], dtype=np.float32)
    arr = np.asarray(list(value), dtype=np.float32).reshape(-1)
    if arr.shape[0] != TEAMMATE_STATE_DIM:
        raise ValueError(f"aux label scale must have length {TEAMMATE_STATE_DIM}, got {arr.shape[0]}")
    return arr


class FillInTeammateActionsAndAuxLabels(FillInTeammateActions):
    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs,
    ):
        super().on_postprocess_trajectory(
            worker=worker,
            episode=episode,
            agent_id=agent_id,
            policy_id=policy_id,
            policies=policies,
            postprocessed_batch=postprocessed_batch,
            original_batches=original_batches,
            **kwargs,
        )
        if policy_id != SHARED_CC_POLICY_ID:
            return

        batch_count = int(postprocessed_batch.count)
        labels = np.zeros((batch_count, TEAMMATE_STATE_DIM), dtype=np.float32)
        valid = np.zeros((batch_count,), dtype=np.float32)

        teammate = teammate_id(agent_id)
        if teammate in original_batches:
            _, teammate_batch = original_batches[teammate]
            teammate_infos = teammate_batch.get(SampleBatch.INFOS)
            if teammate_infos is not None:
                limit = min(batch_count, len(teammate_infos))
                for idx in range(limit):
                    state = extract_own_player_state(teammate_infos[idx])
                    if state is None:
                        continue
                    labels[idx] = state
                    valid[idx] = 1.0

        postprocessed_batch[TEAMMATE_AUX_LABEL] = labels
        postprocessed_batch[TEAMMATE_AUX_VALID] = valid


class SharedCCWithTeammateAuxModel(SharedCentralCriticTorchModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        aux_hidden = int(custom_cfg.get("aux_hidden_size", 128))
        aux_weight = float(custom_cfg.get("aux_weight", 0.1))
        scale = _parse_scale(custom_cfg.get("aux_label_scale"))

        original = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        own_obs_space = original.spaces["own_obs"]
        own_obs_dim = int(np.product(own_obs_space.shape))
        fcnet_hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        feature_dim = int(fcnet_hiddens[-1]) if fcnet_hiddens else own_obs_dim

        self.aux_weight = aux_weight
        self.aux_head = nn.Sequential(
            nn.Linear(feature_dim, aux_hidden),
            nn.ReLU(),
            nn.Linear(aux_hidden, aux_hidden),
            nn.ReLU(),
            nn.Linear(aux_hidden, TEAMMATE_STATE_DIM),
        )
        self.register_buffer("aux_label_scale", torch.as_tensor(scale, dtype=torch.float32))
        self._aux_pred = None
        self._aux_metrics = {
            "aux_train_mse_norm": 0.0,
            "aux_valid_frac": 0.0,
            "aux_train_mae_x_field": 0.0,
            "aux_train_mae_y_field": 0.0,
            "aux_train_mae_vx_field": 0.0,
            "aux_train_mae_vy_field": 0.0,
        }

    def forward(self, input_dict, state, seq_lens):
        logits, state_out = super().forward(input_dict, state, seq_lens)
        features = getattr(self.action_model, "_features", None)
        if features is None:
            raise ValueError("SharedCCWithTeammateAuxModel requires action_model._features from TorchFC.")
        self._aux_pred = self.aux_head(features)
        return logits, state_out

    def custom_loss(self, policy_loss, loss_inputs):
        if self._aux_pred is None:
            return policy_loss

        labels = loss_inputs.get(TEAMMATE_AUX_LABEL)
        valid = loss_inputs.get(TEAMMATE_AUX_VALID)
        if labels is None or valid is None:
            return policy_loss

        device = self._aux_pred.device
        labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        valid = torch.as_tensor(valid, dtype=torch.float32, device=device).reshape(-1)
        if labels.ndim != 2 or labels.shape[-1] != TEAMMATE_STATE_DIM:
            return policy_loss

        scale = self.aux_label_scale.to(device=device, dtype=torch.float32)
        labels_norm = labels / scale
        pred_norm = self._aux_pred

        valid_count = torch.clamp(valid.sum(), min=1.0)
        diff_norm = pred_norm - labels_norm
        mse_norm = (diff_norm.pow(2) * valid.unsqueeze(1)).sum() / (valid_count * float(TEAMMATE_STATE_DIM))
        mae_field = (diff_norm.abs() * scale.unsqueeze(0) * valid.unsqueeze(1)).sum(dim=0) / valid_count
        aux_loss = self.aux_weight * mse_norm

        self._aux_metrics = {
            "aux_train_mse_norm": float(mse_norm.detach().cpu().item()),
            "aux_valid_frac": float((valid.mean()).detach().cpu().item()),
            "aux_train_mae_x_field": float(mae_field[0].detach().cpu().item()),
            "aux_train_mae_y_field": float(mae_field[1].detach().cpu().item()),
            "aux_train_mae_vx_field": float(mae_field[2].detach().cpu().item()),
            "aux_train_mae_vy_field": float(mae_field[3].detach().cpu().item()),
        }

        if isinstance(policy_loss, (list, tuple)):
            return [loss_ + aux_loss for loss_ in policy_loss]
        return policy_loss + aux_loss

    def metrics(self) -> Dict[str, float]:
        return dict(self._aux_metrics)


def register_shared_cc_teammate_aux_model():
    try:
        ModelCatalog.register_custom_model(TEAMMATE_AUX_MODEL_NAME, SharedCCWithTeammateAuxModel)
    except ValueError:
        pass

