import os
import math
from itertools import product

import gym
import numpy as np
import torch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

from cs8803drl.core.checkpoint_utils import extract_torch_weights_from_checkpoint, load_torch_state_dict
from cs8803drl.branches.imitation_bc import load_bc_checkpoint
from cs8803drl.branches.role_specialization import BASELINE_POLICY_ID, FrozenBaselinePolicy

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:  # pragma: no cover
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:  # pragma: no cover
        ActionFlattener = None


torch, nn = try_import_torch()

SHARED_CC_POLICY_ID = "shared_cc_policy"
SHARED_CC_MODEL_NAME = "shared_cc_model"
DEFAULT_TEAM0_AGENT_IDS = (0, 1)
DEFAULT_TEAM1_AGENT_IDS = (2, 3)


def _ensure_modelcatalog_tf_compat():
    import ray.rllib.models.catalog as catalog_mod

    if getattr(catalog_mod, "tf", None) is None:
        class _TFShim:
            class keras:
                class Model:
                    pass

        catalog_mod.tf = _TFShim


def team0_agent_ids():
    raw = os.environ.get("CC_TEAM0_AGENT_IDS", "")
    if not raw.strip():
        return DEFAULT_TEAM0_AGENT_IDS
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise ValueError("CC_TEAM0_AGENT_IDS must contain exactly two ids, e.g. '0,1'.")
    return tuple(int(piece) for piece in pieces)


def team1_agent_ids():
    raw = os.environ.get("CC_TEAM1_AGENT_IDS", "")
    if not raw.strip():
        return DEFAULT_TEAM1_AGENT_IDS
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise ValueError("CC_TEAM1_AGENT_IDS must contain exactly two ids, e.g. '2,3'.")
    return tuple(int(piece) for piece in pieces)


def teammate_id(agent_id):
    team0 = team0_agent_ids()
    team1 = team1_agent_ids()
    aid = int(agent_id)
    if aid == int(team0[0]):
        return int(team0[1])
    if aid == int(team0[1]):
        return int(team0[0])
    if aid == int(team1[0]):
        return int(team1[1])
    if aid == int(team1[1]):
        return int(team1[0])
    raise ValueError(f"Unknown teammate mapping for agent_id={agent_id}.")


def shared_cc_policy_mapping_fn(agent_id, *args, **kwargs):
    if int(agent_id) in set(team0_agent_ids()):
        return SHARED_CC_POLICY_ID
    return BASELINE_POLICY_ID


def _flat_obs_space(space):
    low = np.asarray(space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(space.high, dtype=np.float32).reshape(-1)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def build_cc_obs_space(obs_space, action_space):
    own_obs_space = _flat_obs_space(obs_space)
    if hasattr(action_space, "n"):
        teammate_action_space = gym.spaces.Discrete(int(action_space.n))
    else:
        teammate_action_space = gym.spaces.Discrete(int(np.prod(action_space.nvec)))
    return gym.spaces.Dict(
        {
            "own_obs": own_obs_space,
            "teammate_obs": own_obs_space,
            "teammate_action": teammate_action_space,
        }
    )


def shared_cc_observer(agent_obs, **kwargs):
    observed = {}
    team0 = set(team0_agent_ids())
    for aid, obs in agent_obs.items():
        aid_int = int(aid)
        if aid_int in team0:
            mate_id = teammate_id(aid_int)
            observed[aid_int] = {
                "own_obs": np.asarray(obs, dtype=np.float32).reshape(-1),
                "teammate_obs": np.asarray(agent_obs[mate_id], dtype=np.float32).reshape(-1),
                "teammate_action": 0,
            }
        else:
            observed[aid_int] = agent_obs[aid_int]
    return observed


def shared_cc_observer_all(agent_obs, **kwargs):
    observed = {}
    for aid, obs in agent_obs.items():
        aid_int = int(aid)
        mate_id = teammate_id(aid_int)
        observed[aid_int] = {
            "own_obs": np.asarray(obs, dtype=np.float32).reshape(-1),
            "teammate_obs": np.asarray(agent_obs[mate_id], dtype=np.float32).reshape(-1),
            "teammate_action": 0,
        }
    return observed


class FillInTeammateActions(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._action_dim = int(os.environ.get("CC_ACTION_DIM", "27"))
        self._eye = np.eye(self._action_dim, dtype=np.float32)

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
        if policy_id != SHARED_CC_POLICY_ID:
            return

        cur_obs = postprocessed_batch.get(SampleBatch.CUR_OBS)
        if cur_obs is None or len(cur_obs.shape) != 2 or cur_obs.shape[1] < self._action_dim:
            return

        teammate = teammate_id(agent_id)
        if teammate not in original_batches:
            cur_obs[:, -self._action_dim :] = 0.0
            return

        _, teammate_batch = original_batches[teammate]
        teammate_actions = np.asarray(teammate_batch[SampleBatch.ACTIONS]).reshape(-1)
        limit = min(len(teammate_actions), cur_obs.shape[0])
        cur_obs[:, -self._action_dim :] = 0.0
        if limit <= 0:
            return
        encoded = self._eye[teammate_actions[:limit].astype(np.int64)]
        cur_obs[:limit, -self._action_dim :] = encoded


class SharedCentralCriticTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        original = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        own_obs_space = original.spaces["own_obs"]
        self.action_model = TorchFC(own_obs_space, action_space, num_outputs, model_config, name + "_action")
        self.value_model = TorchFC(obs_space, action_space, 1, model_config, name + "_vf")
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        self._value_out, _ = self.value_model({"obs": input_dict["obs_flat"]}, state, seq_lens)
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        return torch.reshape(self._value_out, [-1])


def register_shared_cc_model():
    _ensure_modelcatalog_tf_compat()
    try:
        ModelCatalog.register_custom_model(SHARED_CC_MODEL_NAME, SharedCentralCriticTorchModel)
    except ValueError:
        pass


def _flat_action_tuples(action_nvec):
    action_nvec = tuple(int(v) for v in action_nvec)
    flat_dim = int(math.prod(action_nvec))
    if ActionFlattener is not None:
        flattener = ActionFlattener(np.asarray(action_nvec, dtype=np.int64))
        return [tuple(int(x) for x in np.asarray(flattener.lookup_action(i)).reshape(-1).tolist()) for i in range(flat_dim)]
    return [tuple(int(x) for x in combo) for combo in product(*[range(v) for v in action_nvec])]


def _to_torch_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr)


def _to_target_tensor(value, target_tensor):
    source_tensor = _to_torch_tensor(value)
    return source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)


def warmstart_shared_cc_policy(trainer, checkpoint_path, *, policy_id=SHARED_CC_POLICY_ID):
    checkpoint_path = (checkpoint_path or "").strip()
    if not checkpoint_path:
        return {"copied": 0, "adapted": 0, "skipped": 0}

    source = extract_torch_weights_from_checkpoint(checkpoint_path, policy_name=policy_id)
    policy = trainer.get_policy(policy_id)
    if policy is None:
        raise ValueError(f"Could not find target policy {policy_id!r}.")

    model = getattr(policy, "model", policy)
    target_state = model.state_dict()
    merged_state = dict(target_state)

    copied = 0
    adapted = 0
    skipped = 0

    # CC -> CC warm-start: if the source checkpoint already carries shared-cc keys
    # (`action_model.*` / `value_model.*`), copy matching tensors directly.
    source_looks_like_cc = any(
        isinstance(key, str)
        and (key.startswith("action_model.") or key.startswith("value_model."))
        for key in source.keys()
    )
    if source_looks_like_cc:
        for target_key, target_tensor in target_state.items():
            if target_key not in source:
                skipped += 1
                continue
            source_tensor = _to_target_tensor(source[target_key], target_tensor)
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                skipped += 1
                continue
            merged_state[target_key] = source_tensor
            copied += 1
        model.load_state_dict(merged_state, strict=False)
        trainer.workers.sync_weights()
        return {
            "copied": copied,
            "adapted": adapted,
            "skipped": skipped,
        }

    value_first_key = "value_model._hidden_layers.0._model.0.weight"
    source_first_key = "_hidden_layers.0._model.0.weight"
    source_obs_dim = int(source[source_first_key].shape[1]) if source_first_key in source else 0

    direct_mapping = {
        "action_model._hidden_layers.0._model.0.weight": "_hidden_layers.0._model.0.weight",
        "action_model._hidden_layers.0._model.0.bias": "_hidden_layers.0._model.0.bias",
        "action_model._hidden_layers.1._model.0.weight": "_hidden_layers.1._model.0.weight",
        "action_model._hidden_layers.1._model.0.bias": "_hidden_layers.1._model.0.bias",
        "action_model._logits._model.0.weight": "_logits._model.0.weight",
        "action_model._logits._model.0.bias": "_logits._model.0.bias",
        "action_model._value_branch._model.0.weight": "_value_branch._model.0.weight",
        "action_model._value_branch._model.0.bias": "_value_branch._model.0.bias",
        "value_model._hidden_layers.0._model.0.bias": "_hidden_layers.0._model.0.bias",
        "value_model._hidden_layers.1._model.0.weight": "_hidden_layers.1._model.0.weight",
        "value_model._hidden_layers.1._model.0.bias": "_hidden_layers.1._model.0.bias",
        "value_model._logits._model.0.weight": "_value_branch._model.0.weight",
        "value_model._logits._model.0.bias": "_value_branch._model.0.bias",
        "value_model._value_branch._model.0.weight": "_value_branch._model.0.weight",
        "value_model._value_branch._model.0.bias": "_value_branch._model.0.bias",
    }

    for target_key, source_key in direct_mapping.items():
        if target_key not in target_state or source_key not in source:
            skipped += 1
            continue
        source_tensor = _to_target_tensor(source[source_key], target_state[target_key])
        if tuple(source_tensor.shape) != tuple(target_state[target_key].shape):
            skipped += 1
            continue
        merged_state[target_key] = source_tensor
        copied += 1

    if value_first_key in target_state and source_first_key in source:
        target_tensor = target_state[value_first_key]
        source_tensor = _to_target_tensor(source[source_first_key], target_tensor)
        if target_tensor.ndim == 2 and source_tensor.ndim == 2 and target_tensor.shape[0] == source_tensor.shape[0]:
            padded = target_tensor.detach().clone()
            padded.zero_()
            padded[:, : source_obs_dim] = source_tensor
            teammate_end = min(target_tensor.shape[1], source_obs_dim * 2)
            padded[:, source_obs_dim:teammate_end] = source_tensor[:, : max(0, teammate_end - source_obs_dim)]
            merged_state[value_first_key] = padded
            adapted += 1
        else:
            skipped += 1

    model.load_state_dict(merged_state, strict=False)
    trainer.workers.sync_weights()
    return {
        "copied": copied,
        "adapted": adapted,
        "skipped": skipped,
    }


def warmstart_shared_cc_policy_from_bc_player(
    trainer,
    checkpoint_path,
    *,
    policy_id=SHARED_CC_POLICY_ID,
):
    checkpoint_path = (checkpoint_path or "").strip()
    if not checkpoint_path:
        return {"copied": 0, "adapted": 0, "skipped": 0}

    bc_model, metadata = load_bc_checkpoint(checkpoint_path, map_location="cpu")
    format_name = str(metadata.get("format", ""))
    if format_name != "bc_player_policy_v1":
        raise ValueError(
            "BC_WARMSTART_CHECKPOINT must point to a player-level BC checkpoint "
            f"(expected format=bc_player_policy_v1, got {format_name!r})."
        )

    policy = trainer.get_policy(policy_id)
    if policy is None:
        raise ValueError(f"Could not find target policy {policy_id!r}.")

    model = getattr(policy, "model", policy)
    target_state = model.state_dict()
    merged_state = dict(target_state)
    source_state = bc_model.state_dict()

    copied = 0
    adapted = 0
    skipped = 0

    direct_mapping = {
        "action_model._hidden_layers.0._model.0.weight": "backbone.0.weight",
        "action_model._hidden_layers.0._model.0.bias": "backbone.0.bias",
        "action_model._hidden_layers.1._model.0.weight": "backbone.2.weight",
        "action_model._hidden_layers.1._model.0.bias": "backbone.2.bias",
        "value_model._hidden_layers.0._model.0.bias": "backbone.0.bias",
        "value_model._hidden_layers.1._model.0.weight": "backbone.2.weight",
        "value_model._hidden_layers.1._model.0.bias": "backbone.2.bias",
    }

    for target_key, source_key in direct_mapping.items():
        if target_key not in target_state or source_key not in source_state:
            skipped += 1
            continue
        source_tensor = _to_target_tensor(source_state[source_key], target_state[target_key])
        if tuple(source_tensor.shape) != tuple(target_state[target_key].shape):
            skipped += 1
            continue
        merged_state[target_key] = source_tensor
        copied += 1

    actor_first_key = "action_model._hidden_layers.0._model.0.weight"
    critic_first_key = "value_model._hidden_layers.0._model.0.weight"
    source_first_key = "backbone.0.weight"
    source_obs_dim = int(metadata.get("obs_dim", 0))

    if actor_first_key in target_state and source_first_key in source_state:
        source_tensor = _to_target_tensor(source_state[source_first_key], target_state[actor_first_key])
        if tuple(source_tensor.shape) == tuple(target_state[actor_first_key].shape):
            merged_state[actor_first_key] = source_tensor
            copied += 1
        else:
            skipped += 1

    if critic_first_key in target_state and source_first_key in source_state:
        target_tensor = target_state[critic_first_key]
        source_tensor = _to_target_tensor(source_state[source_first_key], target_tensor)
        if (
            target_tensor.ndim == 2
            and source_tensor.ndim == 2
            and target_tensor.shape[0] == source_tensor.shape[0]
            and source_obs_dim > 0
        ):
            padded = target_tensor.detach().clone()
            padded.zero_()
            own_end = min(target_tensor.shape[1], source_obs_dim)
            padded[:, :own_end] = source_tensor[:, :own_end]
            teammate_end = min(target_tensor.shape[1], source_obs_dim * 2)
            if teammate_end > source_obs_dim:
                padded[:, source_obs_dim:teammate_end] = source_tensor[:, : teammate_end - source_obs_dim]
            merged_state[critic_first_key] = padded
            adapted += 1
        else:
            skipped += 1

    logits_key = "action_model._logits._model.0.weight"
    logits_bias_key = "action_model._logits._model.0.bias"
    action_nvec = tuple(int(v) for v in metadata.get("action_nvec", []))
    if action_nvec and logits_key in target_state and logits_bias_key in target_state:
        head_weight_keys = [f"heads.{idx}.weight" for idx in range(len(action_nvec))]
        head_bias_keys = [f"heads.{idx}.bias" for idx in range(len(action_nvec))]
        if all(key in source_state for key in head_weight_keys + head_bias_keys):
            head_weights = [
                _to_target_tensor(source_state[key], target_state[logits_key])
                for key in head_weight_keys
            ]
            head_biases = [
                _to_target_tensor(source_state[key], target_state[logits_bias_key])
                for key in head_bias_keys
            ]
            flat_actions = _flat_action_tuples(action_nvec)
            if (
                len(flat_actions) == target_state[logits_key].shape[0]
                and all(w.ndim == 2 for w in head_weights)
                and all(w.shape[1] == target_state[logits_key].shape[1] for w in head_weights)
            ):
                logits_weight = target_state[logits_key].detach().clone()
                logits_bias = target_state[logits_bias_key].detach().clone()
                for flat_idx, branch_action in enumerate(flat_actions):
                    weight_row = torch.zeros_like(logits_weight[flat_idx])
                    bias_value = torch.zeros_like(logits_bias[flat_idx])
                    for branch_idx, action_idx in enumerate(branch_action):
                        weight_row = weight_row + head_weights[branch_idx][int(action_idx)]
                        bias_value = bias_value + head_biases[branch_idx][int(action_idx)]
                    logits_weight[flat_idx] = weight_row
                    logits_bias[flat_idx] = bias_value
                merged_state[logits_key] = logits_weight
                merged_state[logits_bias_key] = logits_bias
                adapted += 2
            else:
                skipped += 2
        else:
            skipped += 2

    model.load_state_dict(merged_state, strict=False)
    trainer.workers.sync_weights()
    return {
        "copied": copied,
        "adapted": adapted,
        "skipped": skipped,
    }


def load_shared_cc_policy_from_checkpoint(
    trainer,
    checkpoint_path,
    *,
    target_policy_id,
    source_policy_id=SHARED_CC_POLICY_ID,
):
    checkpoint_path = (checkpoint_path or "").strip()
    if not checkpoint_path:
        return {"loaded": False}

    policy = trainer.get_policy(target_policy_id)
    if policy is None:
        raise ValueError(f"Could not find target policy {target_policy_id!r}.")

    weights = extract_torch_weights_from_checkpoint(checkpoint_path, policy_name=source_policy_id)
    load_torch_state_dict(policy, weights)
    trainer.workers.sync_weights()
    return {"loaded": True}
