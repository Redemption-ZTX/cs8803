import os

import numpy as np
import torch

from cs8803drl.core.checkpoint_utils import extract_torch_weights_from_checkpoint
from cs8803drl.branches.role_specialization import BASELINE_POLICY_ID, FrozenBaselinePolicy


SHARED_POLICY_ID = "shared_policy"
DEFAULT_TEAM0_AGENT_IDS = (0, 1)
DEFAULT_TEAM1_AGENT_IDS = (2, 3)


def team0_agent_ids():
    raw = os.environ.get("SHARED_TEAM0_AGENT_IDS", "")
    if not raw.strip():
        return DEFAULT_TEAM0_AGENT_IDS
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise ValueError("SHARED_TEAM0_AGENT_IDS must contain exactly two ids, e.g. '0,1'.")
    return tuple(int(piece) for piece in pieces)


def team1_agent_ids():
    raw = os.environ.get("SHARED_TEAM1_AGENT_IDS", "")
    if not raw.strip():
        return DEFAULT_TEAM1_AGENT_IDS
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise ValueError("SHARED_TEAM1_AGENT_IDS must contain exactly two ids, e.g. '2,3'.")
    return tuple(int(piece) for piece in pieces)


def shared_policy_mapping_fn(agent_id, *args, **kwargs):
    if int(agent_id) in set(team0_agent_ids()):
        return SHARED_POLICY_ID
    return BASELINE_POLICY_ID


def build_role_token_map():
    team0 = team0_agent_ids()
    team1 = team1_agent_ids()
    return {
        int(team0[0]): [1.0, 0.0],
        int(team0[1]): [0.0, 1.0],
        int(team1[0]): [1.0, 0.0],
        int(team1[1]): [0.0, 1.0],
    }


def local_role_token(player_id):
    return np.asarray([1.0, 0.0] if int(player_id) == 0 else [0.0, 1.0], dtype=np.float32)


def _to_torch_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr)


def warmstart_shared_policy(trainer, checkpoint_path, *, policy_id=SHARED_POLICY_ID):
    source_weights = extract_torch_weights_from_checkpoint(checkpoint_path, policy_name="default_policy")
    policy = trainer.get_policy(policy_id)
    if policy is None:
        raise ValueError(f"Could not find target shared policy {policy_id!r}.")

    model = getattr(policy, "model", policy)
    target_state = model.state_dict()
    merged_state = dict(target_state)

    copied = 0
    adapted = 0
    skipped = 0

    for key, target_tensor in target_state.items():
        if key not in source_weights:
            skipped += 1
            continue

        source_tensor = _to_torch_tensor(source_weights[key]).to(dtype=target_tensor.dtype)
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            merged_state[key] = source_tensor
            copied += 1
            continue

        # Expand the first hidden layer to accept the extra role-token dims by
        # zero-padding the new input columns.
        if (
            key.endswith("_hidden_layers.0._model.0.weight")
            and source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] < target_tensor.shape[1]
        ):
            padded = target_tensor.detach().clone()
            padded.zero_()
            padded[:, : source_tensor.shape[1]] = source_tensor
            merged_state[key] = padded
            adapted += 1
            continue

        skipped += 1

    model.load_state_dict(merged_state, strict=False)
    trainer.workers.sync_weights()
    return {
        "copied": copied,
        "adapted": adapted,
        "skipped": skipped,
    }
