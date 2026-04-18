import numpy as np
import torch

from cs8803drl.core.checkpoint_utils import extract_torch_weights_from_checkpoint


SUMMARY_POLICY_ID = "default_policy"
RAY_BLOCK_SIZE = 8
RAY_TYPE_DIM = 7
SUMMARY_FEATURE_DIM = RAY_TYPE_DIM * 4


def ray_summary_feature_dim():
    return SUMMARY_FEATURE_DIM


def append_ray_summary_features(observation):
    arr = np.asarray(observation, dtype=np.float32).reshape(-1)
    extra = np.zeros((SUMMARY_FEATURE_DIM,), dtype=np.float32)

    if arr.size == 0 or arr.size % RAY_BLOCK_SIZE != 0:
        return np.concatenate([arr, extra], axis=0).astype(np.float32, copy=False)

    blocks = arr.reshape(-1, RAY_BLOCK_SIZE)
    type_scores = blocks[:, :RAY_TYPE_DIM]
    distances = np.clip(blocks[:, RAY_TYPE_DIM], 0.0, 1.0)
    num_rays = max(1, blocks.shape[0])
    ray_positions = np.linspace(-1.0, 1.0, num_rays, dtype=np.float32)

    counts = np.zeros((RAY_TYPE_DIM,), dtype=np.float32)
    nearest = np.ones((RAY_TYPE_DIM,), dtype=np.float32)
    mean_dist = np.ones((RAY_TYPE_DIM,), dtype=np.float32)
    centroid = np.zeros((RAY_TYPE_DIM,), dtype=np.float32)

    for type_idx in range(RAY_TYPE_DIM):
        mask = type_scores[:, type_idx] > 0.5
        if not np.any(mask):
            continue
        counts[type_idx] = float(np.sum(mask)) / float(num_rays)
        nearest[type_idx] = float(np.min(distances[mask]))
        mean_dist[type_idx] = float(np.mean(distances[mask]))
        centroid[type_idx] = float(np.mean(ray_positions[mask]))

    extra = np.concatenate([counts, nearest, mean_dist, centroid], axis=0).astype(np.float32, copy=False)
    return np.concatenate([arr, extra], axis=0).astype(np.float32, copy=False)


def build_summary_obs_space(obs_space):
    low = np.asarray(obs_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(obs_space.high, dtype=np.float32).reshape(-1)

    count_low = np.zeros((RAY_TYPE_DIM,), dtype=np.float32)
    count_high = np.ones((RAY_TYPE_DIM,), dtype=np.float32)
    nearest_low = np.zeros((RAY_TYPE_DIM,), dtype=np.float32)
    nearest_high = np.ones((RAY_TYPE_DIM,), dtype=np.float32)
    mean_low = np.zeros((RAY_TYPE_DIM,), dtype=np.float32)
    mean_high = np.ones((RAY_TYPE_DIM,), dtype=np.float32)
    centroid_low = -np.ones((RAY_TYPE_DIM,), dtype=np.float32)
    centroid_high = np.ones((RAY_TYPE_DIM,), dtype=np.float32)

    extra_low = np.concatenate([count_low, nearest_low, mean_low, centroid_low], axis=0)
    extra_high = np.concatenate([count_high, nearest_high, mean_high, centroid_high], axis=0)

    import gym

    return gym.spaces.Box(
        low=np.concatenate([low, extra_low], axis=0),
        high=np.concatenate([high, extra_high], axis=0),
        dtype=np.float32,
    )


def _to_torch_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr)


def warmstart_summary_policy(trainer, checkpoint_path, *, policy_name=SUMMARY_POLICY_ID):
    source_weights = extract_torch_weights_from_checkpoint(checkpoint_path, policy_name="default_policy")
    policy = trainer.get_policy(policy_name)
    if policy is None:
        raise ValueError(f"Could not find target policy {policy_name!r}.")

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
