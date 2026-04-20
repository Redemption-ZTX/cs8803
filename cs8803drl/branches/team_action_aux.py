from typing import Dict

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import RLLIB_MODEL, _global_registry


torch, nn = try_import_torch()


TEAM_ACTION_AUX_MODEL_NAME = "team_action_aux_model"
TEAM_ACTION_AUX_SYMMETRIC_MODEL_NAME = "team_action_aux_symmetric_model"


def _parse_action_spec(action_space):
    nvec = np.asarray(getattr(action_space, "nvec", []), dtype=np.int64).reshape(-1)
    if nvec.size == 0 or nvec.size % 2 != 0:
        raise ValueError(
            "TeamActionAux models expect an even-length MultiDiscrete action space, "
            f"got nvec={nvec!r}"
        )
    if not np.all(nvec == nvec[0]):
        raise ValueError(
            "TeamActionAux models currently expect equal-sized discrete factors, "
            f"got nvec={nvec!r}"
        )
    joint_dims = int(nvec.size)
    agent_dims = int(joint_dims // 2)
    classes = int(nvec[0])
    return joint_dims, agent_dims, classes


class TeamActionAuxTorchModel(TorchFC):
    """Flat team-level FCNet + auxiliary action-prediction head.

    The main policy/value path stays identical to RLlib's default TorchFC so
    warm-start from vanilla team-level checkpoints can still reuse the trunk,
    logits, and value weights. The aux head reads the shared trunk features and
    predicts the second local agent's 3-way action factors.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        aux_hidden = int(custom_cfg.get("aux_hidden_size", 256))
        aux_weight = float(custom_cfg.get("aux_weight", 0.05))
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )

        self.aux_weight = aux_weight
        self._joint_action_dims, self._target_dims, self._target_classes = _parse_action_spec(
            action_space
        )
        self._target_start = self._target_dims
        feature_dim = int(hiddens[-1]) if hiddens else int(np.product(obs_space.shape))

        self.aux_head = nn.Sequential(
            nn.Linear(feature_dim, aux_hidden),
            nn.ReLU(),
            nn.Linear(aux_hidden, aux_hidden),
            nn.ReLU(),
            nn.Linear(aux_hidden, self._target_dims * self._target_classes),
        )

        self._aux_logits = None
        self._aux_metrics = {
            "aux_action_loss": 0.0,
            "aux_action_acc_mean": 0.0,
        }
        for idx in range(self._target_dims):
            self._aux_metrics[f"aux_action_acc_dim{idx}"] = 0.0

    def forward(self, input_dict, state, seq_lens):
        logits, state_out = super().forward(input_dict, state, seq_lens)
        features = getattr(self, "_features", None)
        if features is None:
            raise ValueError("TeamActionAuxTorchModel requires TorchFC._features to be populated.")
        aux_logits = self.aux_head(features)
        self._aux_logits = aux_logits.reshape(-1, self._target_dims, self._target_classes)
        return logits, state_out

    def custom_loss(self, policy_loss, loss_inputs):
        if self._aux_logits is None:
            return policy_loss

        actions = loss_inputs.get(SampleBatch.ACTIONS)
        if actions is None:
            return policy_loss

        device = self._aux_logits.device
        actions = torch.as_tensor(actions, device=device)
        if actions.ndim == 1:
            if actions.numel() % self._joint_action_dims != 0:
                return policy_loss
            actions = actions.reshape(-1, self._joint_action_dims)
        if actions.ndim != 2 or actions.shape[1] < self._target_start + self._target_dims:
            return policy_loss

        target = actions[:, self._target_start : self._target_start + self._target_dims].long()
        flat_logits = self._aux_logits.reshape(-1, self._target_classes)
        flat_target = target.reshape(-1)
        ce = nn.functional.cross_entropy(flat_logits, flat_target)

        with torch.no_grad():
            pred = self._aux_logits.argmax(dim=-1)
            per_dim_acc = (pred == target).float().mean(dim=0)
            metrics = {
                "aux_action_loss": float(ce.detach().cpu().item()),
                "aux_action_acc_mean": float(per_dim_acc.mean().detach().cpu().item()),
            }
            for idx in range(self._target_dims):
                metrics[f"aux_action_acc_dim{idx}"] = float(per_dim_acc[idx].detach().cpu().item())
            self._aux_metrics = metrics

        aux_term = self.aux_weight * ce
        if isinstance(policy_loss, (list, tuple)):
            return [loss_ + aux_term for loss_ in policy_loss]
        return policy_loss + aux_term

    def metrics(self) -> Dict[str, float]:
        return dict(self._aux_metrics)


class TeamActionAuxSymmetricTorchModel(TorchFC):
    """Flat team-level FCNet + symmetric bidirectional action-prediction heads.

    This keeps the warm-start-compatible main TorchFC trunk/logits/value path
    unchanged, but predicts both local agents' action factors so the auxiliary
    signal is no longer one-sided.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_cfg = dict(model_config.get("custom_model_config") or {})
        aux_hidden = int(custom_cfg.get("aux_hidden_size", 256))
        aux_weight = float(custom_cfg.get("aux_weight", 0.05))
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )

        self.aux_weight = aux_weight
        self._joint_action_dims, self._agent_action_dims, self._target_classes = _parse_action_spec(
            action_space
        )
        feature_dim = int(hiddens[-1]) if hiddens else int(np.product(obs_space.shape))

        def _build_head():
            return nn.Sequential(
                nn.Linear(feature_dim, aux_hidden),
                nn.ReLU(),
                nn.Linear(aux_hidden, aux_hidden),
                nn.ReLU(),
                nn.Linear(aux_hidden, self._agent_action_dims * self._target_classes),
            )

        self.aux_head_agent0 = _build_head()
        self.aux_head_agent1 = _build_head()

        self._aux_logits_agent0 = None
        self._aux_logits_agent1 = None
        self._aux_metrics = {
            "aux_action_loss": 0.0,
            "aux_action_loss_agent0": 0.0,
            "aux_action_loss_agent1": 0.0,
            "aux_action_acc_mean": 0.0,
            "aux_action_acc_agent0": 0.0,
            "aux_action_acc_agent1": 0.0,
        }
        for idx in range(self._joint_action_dims):
            self._aux_metrics[f"aux_action_acc_dim{idx}"] = 0.0

    def forward(self, input_dict, state, seq_lens):
        logits, state_out = super().forward(input_dict, state, seq_lens)
        features = getattr(self, "_features", None)
        if features is None:
            raise ValueError(
                "TeamActionAuxSymmetricTorchModel requires TorchFC._features to be populated."
            )
        self._aux_logits_agent0 = self.aux_head_agent0(features).reshape(
            -1, self._agent_action_dims, self._target_classes
        )
        self._aux_logits_agent1 = self.aux_head_agent1(features).reshape(
            -1, self._agent_action_dims, self._target_classes
        )
        return logits, state_out

    def custom_loss(self, policy_loss, loss_inputs):
        if self._aux_logits_agent0 is None or self._aux_logits_agent1 is None:
            return policy_loss

        actions = loss_inputs.get(SampleBatch.ACTIONS)
        if actions is None:
            return policy_loss

        device = self._aux_logits_agent0.device
        actions = torch.as_tensor(actions, device=device)
        if actions.ndim == 1:
            if actions.numel() % self._joint_action_dims != 0:
                return policy_loss
            actions = actions.reshape(-1, self._joint_action_dims)
        if actions.ndim != 2 or actions.shape[1] != self._joint_action_dims:
            return policy_loss

        target_agent0 = actions[:, : self._agent_action_dims].long()
        target_agent1 = actions[:, self._agent_action_dims :].long()

        flat_logits_agent0 = self._aux_logits_agent0.reshape(-1, self._target_classes)
        flat_logits_agent1 = self._aux_logits_agent1.reshape(-1, self._target_classes)
        ce_agent0 = nn.functional.cross_entropy(flat_logits_agent0, target_agent0.reshape(-1))
        ce_agent1 = nn.functional.cross_entropy(flat_logits_agent1, target_agent1.reshape(-1))
        ce = 0.5 * (ce_agent0 + ce_agent1)

        with torch.no_grad():
            pred_agent0 = self._aux_logits_agent0.argmax(dim=-1)
            pred_agent1 = self._aux_logits_agent1.argmax(dim=-1)
            per_dim_acc_agent0 = (pred_agent0 == target_agent0).float().mean(dim=0)
            per_dim_acc_agent1 = (pred_agent1 == target_agent1).float().mean(dim=0)
            per_dim_acc = torch.cat([per_dim_acc_agent0, per_dim_acc_agent1], dim=0)
            metrics = {
                "aux_action_loss": float(ce.detach().cpu().item()),
                "aux_action_loss_agent0": float(ce_agent0.detach().cpu().item()),
                "aux_action_loss_agent1": float(ce_agent1.detach().cpu().item()),
                "aux_action_acc_mean": float(per_dim_acc.mean().detach().cpu().item()),
                "aux_action_acc_agent0": float(per_dim_acc_agent0.mean().detach().cpu().item()),
                "aux_action_acc_agent1": float(per_dim_acc_agent1.mean().detach().cpu().item()),
            }
            for idx in range(self._joint_action_dims):
                metrics[f"aux_action_acc_dim{idx}"] = float(per_dim_acc[idx].detach().cpu().item())
            self._aux_metrics = metrics

        aux_term = self.aux_weight * ce
        if isinstance(policy_loss, (list, tuple)):
            return [loss_ + aux_term for loss_ in policy_loss]
        return policy_loss + aux_term

    def metrics(self) -> Dict[str, float]:
        return dict(self._aux_metrics)


def register_team_action_aux_model():
    try:
        ModelCatalog.register_custom_model(TEAM_ACTION_AUX_MODEL_NAME, TeamActionAuxTorchModel)
    except AttributeError:
        # RLlib 1.4's register_custom_model unconditionally touches tf.keras
        # even in torch-only environments. Fall back to the underlying
        # registry path so custom torch models remain usable.
        _global_registry.register(RLLIB_MODEL, TEAM_ACTION_AUX_MODEL_NAME, TeamActionAuxTorchModel)
    except ValueError:
        pass
    try:
        ModelCatalog.register_custom_model(
            TEAM_ACTION_AUX_SYMMETRIC_MODEL_NAME, TeamActionAuxSymmetricTorchModel
        )
    except AttributeError:
        _global_registry.register(
            RLLIB_MODEL, TEAM_ACTION_AUX_SYMMETRIC_MODEL_NAME, TeamActionAuxSymmetricTorchModel
        )
    except ValueError:
        pass
