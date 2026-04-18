"""
Canonical checkpoint parsing utilities for Ray RLlib 1.4 pickle-based checkpoints.

This is the single source of truth for checkpoint loading logic.
All training scripts and evaluation tools should import from here.

Agent submission modules (agent_*/agent.py) contain a COPY of these functions
because they must be self-contained for zip submission. When modifying functions
here, sync the copies using:
    cp checkpoint_utils.py agent_performance/agent.py  # then edit to keep only needed parts
    cp checkpoint_utils.py agent_reward_mod/agent.py

See docs/architecture/code-audit-000.md § 6.1 for the duplication analysis.
"""

import os
import pickle
import shutil

import numpy as np


# ---------------------------------------------------------------------------
# Low-level unpickling
# ---------------------------------------------------------------------------

def resolve_checkpoint_file(path):
    """Accept either checkpoint file or checkpoint directory."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.startswith("checkpoint-")
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* file found in directory: {path}")
    return candidates[0]

def unpickle_if_bytes(obj, *, max_depth=3):
    """Recursively unpickle bytes/bytearray objects up to max_depth."""
    cur = obj
    for _ in range(max_depth):
        if isinstance(cur, (bytes, bytearray)):
            try:
                cur = pickle.loads(cur)
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, ModuleNotFoundError):
                # Some RLlib checkpoint fields are raw binary payloads rather than
                # nested pickle blobs. Leave them untouched so restore/sanitization
                # can continue operating on the surrounding structure.
                break
            continue
        break
    return cur


def find_nested_key(obj, key, *, max_depth=6):
    """Search for a key in a nested dict/list structure, unpickling as needed."""
    obj = unpickle_if_bytes(obj, max_depth=max_depth)
    if max_depth <= 0:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return unpickle_if_bytes(obj[key], max_depth=max_depth)
        for v in obj.values():
            found = find_nested_key(v, key, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found = find_nested_key(v, key, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Optimizer state stripping
# ---------------------------------------------------------------------------

def strip_optimizer_state(obj):
    """Remove optimizer state from checkpoint data.

    Handles np.ndarray with dtype=object (common source of restore failures
    with newer NumPy/PyTorch) by replacing them with empty float32 arrays.

    This is the most complete version — ndarray check comes first to prevent
    recursion into object arrays.
    """
    # Must come before dict/list check: object-dtype ndarrays can look iterable
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.asarray([], dtype=np.float32)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {
                "optimizer_variables",
                "optimizer_state",
                "optim_state",
                "_optimizer_variables",
            }:
                out[k] = []
            else:
                out[k] = strip_optimizer_state(v)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(strip_optimizer_state(v) for v in obj)
    return obj


def drop_optimizer_state_keys(weights):
    """Remove top-level optimizer bookkeeping entries from weight mappings."""
    if not isinstance(weights, dict):
        return weights

    blocked = {
        "optimizer_variables",
        "optimizer_state",
        "optim_state",
        "_optimizer_variables",
    }
    return {
        k: v
        for k, v in weights.items()
        if not (isinstance(k, str) and (k in blocked or k.startswith("_optimizer")))
    }


def load_torch_state_dict(policy, weights):
    """Load a torch-style state_dict into a policy model, skipping bad entries."""
    import torch

    if not isinstance(weights, dict):
        raise ValueError("Expected a dict-like weight mapping.")

    torch_weights = {}
    for k, v in weights.items():
        if not isinstance(k, str):
            continue
        if k in {
            "optimizer_variables",
            "optimizer_state",
            "optim_state",
            "_optimizer_variables",
        } or k.startswith("_optimizer"):
            continue

        value = unpickle_if_bytes(v, max_depth=3)
        if isinstance(value, torch.Tensor):
            torch_weights[k] = value
            continue
        if isinstance(value, dict):
            continue

        try:
            arr = np.asarray(value)
        except Exception:
            continue

        if arr.dtype == object:
            try:
                arr = arr.astype(np.float32)
            except Exception:
                continue

        if arr.dtype.kind not in "biufc":
            continue

        torch_weights[k] = torch.from_numpy(arr)

    if not torch_weights:
        raise ValueError("Could not construct a valid torch state_dict from checkpoint.")

    target = getattr(policy, "model", policy)
    target.load_state_dict(torch_weights, strict=False)


# ---------------------------------------------------------------------------
# Torch state dict detection
# ---------------------------------------------------------------------------

def looks_like_torch_state_dict(d):
    """Heuristic: does this dict look like a PyTorch model state_dict?"""
    if not isinstance(d, dict) or not d:
        return False
    keys = [k for k in d.keys() if isinstance(k, str)]
    if not keys:
        return False
    hits = sum(k.endswith(".weight") or k.endswith(".bias") for k in keys[:200])
    if hits < 2:
        return False
    checked = 0
    for k in keys[:200]:
        if not (k.endswith(".weight") or k.endswith(".bias")):
            continue
        v = unpickle_if_bytes(d.get(k))
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        checked += 1
        if arr.dtype == object:
            return False
        if checked >= 3:
            break
    return True


def find_torch_state_dict(obj, *, max_depth=8):
    """Recursively search for a torch-like state_dict in nested checkpoint data."""
    obj = unpickle_if_bytes(obj, max_depth=3)
    if max_depth <= 0:
        return None
    if isinstance(obj, dict):
        if looks_like_torch_state_dict(obj):
            return obj
        for v in obj.values():
            found = find_torch_state_dict(v, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found = find_torch_state_dict(v, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Checkpoint sanitization (for tune.run restore)
# ---------------------------------------------------------------------------

def sanitize_checkpoint_for_restore(checkpoint_path):
    """Clean optimizer state from a checkpoint so it can be restored by tune.run.

    Creates a '-sanitized' copy alongside the original. Idempotent — skips
    if sanitized version already exists and looks valid.

    Returns the path to the sanitized checkpoint.
    """
    checkpoint_path = resolve_checkpoint_file(checkpoint_path)
    base = os.path.basename(checkpoint_path)
    sanitized_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"{base}-sanitized",
    )
    meta_src = checkpoint_path + ".tune_metadata"
    meta_dst = sanitized_path + ".tune_metadata"

    if os.path.exists(sanitized_path):
        try:
            with open(sanitized_path, "rb") as f:
                existing = pickle.load(f)
            ok = True
            if isinstance(existing, dict) and "worker" in existing:
                existing_worker = existing["worker"]
                if not isinstance(existing_worker, (bytes, bytearray)):
                    ok = False
                else:
                    try:
                        decoded = pickle.loads(existing_worker)
                    except Exception:
                        decoded = None
                    # A valid RLlib checkpoint keeps worker as a binary payload.
                    # If unpickling it produces another bytes object, we previously
                    # double-wrapped the payload and must rebuild the sanitized file.
                    if isinstance(decoded, (bytes, bytearray)):
                        ok = False
        except Exception:
            ok = False

        if ok:
            if os.path.exists(meta_src) and not os.path.exists(meta_dst):
                shutil.copyfile(meta_src, meta_dst)
            return sanitized_path

    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)

    state = unpickle_if_bytes(state, max_depth=4)

    if isinstance(state, dict) and "worker" in state:
        raw_worker = state.get("worker")
        worker = unpickle_if_bytes(raw_worker, max_depth=6)
        if isinstance(worker, (bytes, bytearray)):
            # Some RLlib checkpoints store worker state as an opaque binary blob
            # rather than a nested pickle object. Keep that payload untouched;
            # wrapping it again would break restore by turning bytes into
            # "pickled bytes".
            state["worker"] = raw_worker
        else:
            worker = strip_optimizer_state(worker)
            state["worker"] = pickle.dumps(worker)
    else:
        state = strip_optimizer_state(state)

    with open(sanitized_path, "wb") as f:
        pickle.dump(state, f)

    if os.path.exists(meta_src) and not os.path.exists(meta_dst):
        shutil.copyfile(meta_src, meta_dst)

    return sanitized_path


# ---------------------------------------------------------------------------
# High-level: load policy weights from checkpoint
# ---------------------------------------------------------------------------

def load_policy_weights(checkpoint_path, trainer, policy_name="default_policy"):
    """Load policy weights from a Ray 1.4 checkpoint into a trainer's policy.

    Handles the full complexity of RLlib 1.4 pickle format:
    - Nested worker/state/policy_states structure
    - Multiple unpickling layers
    - Torch state_dict detection as fallback
    - Object-dtype ndarray sanitization

    Returns the loaded policy object.
    """
    checkpoint_path = resolve_checkpoint_file(checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        raw_state = pickle.load(f)

    state = unpickle_if_bytes(raw_state)
    worker_state = unpickle_if_bytes(state.get("worker", {})) if isinstance(state, dict) else {}
    worker_state_state = None
    if isinstance(worker_state, dict) and "state" in worker_state:
        worker_state_state = unpickle_if_bytes(worker_state.get("state"), max_depth=6)

    # Find policy states
    policy_states = None
    for source in [worker_state, worker_state_state]:
        if source is not None:
            policy_states = find_nested_key(source, "policy_states")
            if policy_states is not None:
                break
    if policy_states is None:
        policy_states = {}

    # Prefer an exact source policy match when present. This matters for
    # multi-policy checkpoints such as attacker/defender role-specialized runs.
    source_policy_id = policy_name
    if policy_states:
        if policy_name in policy_states:
            source_policy_id = policy_name
        else:
            source_policy_id = next(iter(policy_states.keys()))

    policy = trainer.get_policy(policy_name) or trainer.get_policy(source_policy_id)
    if policy is None:
        raise ValueError(
            f"Could not find target policy (tried {policy_name!r} and {source_policy_id!r})."
        )

    # Extract and load weights
    p_state = unpickle_if_bytes(policy_states.get(source_policy_id, {}))
    p_state = strip_optimizer_state(p_state)

    weights = None
    for key in ("weights", "model", "state_dict"):
        if isinstance(p_state, dict) and key in p_state:
            w = unpickle_if_bytes(p_state[key])
            if isinstance(w, dict):
                weights = w
                break
    if weights is None and isinstance(p_state, dict) and p_state:
        weights = p_state
    weights = unpickle_if_bytes(weights)
    weights = drop_optimizer_state_keys(weights)

    if weights is None:
        # Last resort: scan for torch state_dict
        candidate = None
        if worker_state_state is not None:
            candidate = find_torch_state_dict(worker_state_state)
        if candidate is None:
            candidate = find_torch_state_dict(worker_state)
        if candidate is None:
            raise ValueError("Could not find policy weights in checkpoint.")
        candidate = drop_optimizer_state_keys(candidate)
        load_torch_state_dict(policy, candidate)
    else:
        try:
            policy.set_weights(weights)
        except Exception:
            load_torch_state_dict(policy, weights)

    return policy


def infer_action_dim_from_checkpoint(checkpoint_path):
    """Infer the action space dimension from checkpoint logits layer shape."""
    checkpoint_path = resolve_checkpoint_file(checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        raw_state = pickle.load(f)
    state = unpickle_if_bytes(raw_state)

    worker_state = unpickle_if_bytes(state.get("worker", {})) if isinstance(state, dict) else {}
    worker_state_state = None
    if isinstance(worker_state, dict) and "state" in worker_state:
        worker_state_state = unpickle_if_bytes(worker_state.get("state"), max_depth=6)

    candidate_sd = None
    if worker_state_state is not None:
        candidate_sd = find_torch_state_dict(worker_state_state)
    if candidate_sd is None:
        candidate_sd = find_torch_state_dict(worker_state)

    if not isinstance(candidate_sd, dict):
        return None

    for k, v in candidate_sd.items():
        if not isinstance(k, str):
            continue
        if not (k.endswith(".weight") or k.endswith(".bias")):
            continue
        if "logits" not in k and "_logits" not in k:
            continue
        try:
            arr = np.asarray(unpickle_if_bytes(v))
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] > 1:
            return int(arr.shape[0])

    return None


def extract_torch_weights_from_checkpoint(checkpoint_path, policy_name="default_policy"):
    """Extract a torch-style weight mapping from a Ray 1.4 checkpoint.

    Prefers an exact policy_state match when available, otherwise falls back to
    recursively scanning for a torch-like state_dict.
    """
    checkpoint_path = resolve_checkpoint_file(checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        raw_state = pickle.load(f)

    state = unpickle_if_bytes(raw_state)
    worker_state = unpickle_if_bytes(state.get("worker", {})) if isinstance(state, dict) else {}
    worker_state_state = None
    if isinstance(worker_state, dict) and "state" in worker_state:
        worker_state_state = unpickle_if_bytes(worker_state.get("state"), max_depth=6)

    policy_states = None
    for source in [worker_state, worker_state_state]:
        if source is not None:
            policy_states = find_nested_key(source, "policy_states")
            if policy_states is not None:
                break
    if policy_states is None:
        policy_states = {}

    source_policy_id = policy_name
    if policy_states:
        if policy_name in policy_states:
            source_policy_id = policy_name
        else:
            source_policy_id = next(iter(policy_states.keys()))

    p_state = unpickle_if_bytes(policy_states.get(source_policy_id, {}))
    p_state = strip_optimizer_state(p_state)

    weights = None
    for key in ("weights", "model", "state_dict"):
        if isinstance(p_state, dict) and key in p_state:
            w = unpickle_if_bytes(p_state[key])
            if isinstance(w, dict):
                weights = w
                break
    if weights is None and isinstance(p_state, dict) and p_state:
        weights = p_state
    weights = drop_optimizer_state_keys(unpickle_if_bytes(weights))

    if weights is None:
        candidate = None
        if worker_state_state is not None:
            candidate = find_torch_state_dict(worker_state_state)
        if candidate is None:
            candidate = find_torch_state_dict(worker_state)
        if candidate is None:
            raise ValueError("Could not find torch weights in checkpoint.")
        weights = drop_optimizer_state_keys(candidate)

    return weights
