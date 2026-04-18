import numpy as np
import torch

from cs8803drl.core.checkpoint_utils import extract_torch_weights_from_checkpoint


LSTM_POLICY_ID = "default_policy"


def _to_torch_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr)


def _mapped_source_key(target_key):
    if target_key.startswith("_logits_branch."):
        return target_key.replace("_logits_branch.", "_logits.", 1)
    return target_key


def warmstart_lstm_policy(trainer, checkpoint_path, *, policy_name=LSTM_POLICY_ID):
    source_weights = extract_torch_weights_from_checkpoint(checkpoint_path, policy_name="default_policy")
    policy = trainer.get_policy(policy_name)
    if policy is None:
        raise ValueError(f"Could not find target policy {policy_name!r}.")

    model = getattr(policy, "model", policy)
    target_state = model.state_dict()
    merged_state = dict(target_state)

    copied = 0
    skipped = 0

    for target_key, target_tensor in target_state.items():
        source_key = _mapped_source_key(target_key)
        if source_key not in source_weights:
            skipped += 1
            continue

        source_tensor = _to_torch_tensor(source_weights[source_key]).to(dtype=target_tensor.dtype)
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            skipped += 1
            continue

        merged_state[target_key] = source_tensor
        copied += 1

    model.load_state_dict(merged_state, strict=False)
    trainer.workers.sync_weights()
    return {
        "copied": copied,
        "skipped": skipped,
    }
