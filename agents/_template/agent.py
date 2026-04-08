"""
Agent template — copy this directory to create a new agent version.

Checkpoint loading uses self-contained copies of functions from checkpoint_utils.py.
If checkpoint_utils.py is updated, sync changes here.

See docs/architecture/code-audit-000.md § 5.4 for context.
"""

import os
import pickle
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:
    ActionFlattener = None


ALGORITHM = "PPO"
POLICY_NAME = "default_policy"
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoint():
    """Find the checkpoint file in this agent's directory."""
    env_path = os.environ.get("AGENT_CHECKPOINT", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path

    direct = os.path.join(_AGENT_DIR, "checkpoint")
    if os.path.exists(direct):
        return direct

    for f in sorted(os.listdir(_AGENT_DIR)):
        if f.startswith("checkpoint-"):
            return os.path.join(_AGENT_DIR, f)

    raise FileNotFoundError(
        f"No checkpoint found in {_AGENT_DIR}. "
        f"Place a checkpoint file here or set AGENT_CHECKPOINT env var."
    )


def _find_params_pkl(checkpoint_path):
    config_dir = os.path.dirname(checkpoint_path)
    for candidate in [
        os.path.join(config_dir, "params.pkl"),
        os.path.join(config_dir, "..", "params.pkl"),
        os.path.join(_AGENT_DIR, "params.pkl"),
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find params.pkl near {checkpoint_path} or in {_AGENT_DIR}"
    )


# ---------------------------------------------------------------------------
# Checkpoint parsing (copied from checkpoint_utils.py — keep in sync)
# ---------------------------------------------------------------------------

def _unpickle_if_bytes(obj, *, max_depth=3):
    cur = obj
    for _ in range(max_depth):
        if isinstance(cur, (bytes, bytearray)):
            cur = pickle.loads(cur)
            continue
        break
    return cur


def _find_nested_key(obj, key, *, max_depth=6):
    obj = _unpickle_if_bytes(obj, max_depth=max_depth)
    if max_depth <= 0:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return _unpickle_if_bytes(obj[key], max_depth=max_depth)
        for v in obj.values():
            found = _find_nested_key(v, key, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_nested_key(v, key, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def _looks_like_torch_state_dict(d):
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
        v = _unpickle_if_bytes(d.get(k))
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


def _find_torch_state_dict(obj, *, max_depth=8):
    obj = _unpickle_if_bytes(obj, max_depth=3)
    if max_depth <= 0:
        return None
    if isinstance(obj, dict):
        if _looks_like_torch_state_dict(obj):
            return obj
        for v in obj.values():
            found = _find_torch_state_dict(v, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_torch_state_dict(v, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def _strip_optimizer_state(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.asarray([], dtype=np.float32)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {
                "optimizer_variables", "optimizer_state",
                "optim_state", "_optimizer_variables",
            }:
                out[k] = []
            else:
                out[k] = _strip_optimizer_state(v)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(_strip_optimizer_state(v) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _unwrap_action(action):
    a = action
    while isinstance(a, (list, tuple)) and len(a) == 1:
        a = a[0]
    if isinstance(a, tuple) and len(a) >= 1:
        return a[0]
    return a


def _coerce_int_action(action):
    a = _unwrap_action(action)
    if isinstance(a, np.ndarray) and a.shape == ():
        a = a.item()
    if isinstance(a, (np.integer, int)):
        return int(a)
    if isinstance(a, (list, tuple, np.ndarray)):
        arr = np.asarray(a)
        if arr.shape == ():
            return int(arr.item())
        if arr.size >= 1 and arr.ndim == 1:
            return int(arr.reshape(-1)[0])
    return int(a)


def _unflatten_discrete_to_multidiscrete(flat, nvec):
    flat = int(flat)
    out = np.zeros((len(nvec),), dtype=np.int64)
    for i in range(len(nvec) - 1, -1, -1):
        base = int(nvec[i])
        if base <= 0:
            out[i] = 0
            continue
        out[i] = flat % base
        flat //= base
    return out


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

class _DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        raise RuntimeError("Dummy env")

    def step(self, action):
        raise RuntimeError("Dummy env")


def _load_weights_into_policy(policy, p_state, worker_state, worker_state_state):
    p_state = _strip_optimizer_state(p_state)
    weights = None
    for key in ("weights", "model", "state_dict"):
        if isinstance(p_state, dict) and key in p_state:
            w = _unpickle_if_bytes(p_state[key])
            if isinstance(w, dict):
                weights = w
                break
    if weights is None and isinstance(p_state, dict) and p_state:
        weights = p_state
    weights = _unpickle_if_bytes(weights)

    if weights is None:
        candidate = None
        if worker_state_state is not None:
            candidate = _find_torch_state_dict(worker_state_state)
        if candidate is None:
            candidate = _find_torch_state_dict(worker_state)
        if candidate is None:
            raise ValueError("Could not find policy weights in checkpoint.")
        import torch
        torch_weights = {}
        for k, v in candidate.items():
            if isinstance(v, torch.Tensor):
                torch_weights[k] = v
            else:
                arr = np.asarray(v)
                if arr.dtype == object:
                    try:
                        arr = arr.astype(np.float32)
                    except Exception:
                        continue
                torch_weights[k] = torch.from_numpy(arr)
        policy.model.load_state_dict(torch_weights, strict=False)
        return

    try:
        policy.set_weights(weights)
    except Exception:
        import torch
        torch_weights = {}
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                torch_weights[k] = v
            else:
                arr = np.asarray(v)
                if arr.dtype == object:
                    try:
                        arr = arr.astype(np.float32)
                    except Exception:
                        continue
                torch_weights[k] = torch.from_numpy(arr)
        policy.model.load_state_dict(torch_weights, strict=False)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RayAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        checkpoint_path = _find_checkpoint()
        params_path = _find_params_pkl(checkpoint_path)

        # Infer action dim from checkpoint
        with open(checkpoint_path, "rb") as f:
            raw_state = pickle.load(f)
        state = _unpickle_if_bytes(raw_state)
        worker_state = _unpickle_if_bytes(state.get("worker", {})) if isinstance(state, dict) else {}
        worker_state_state = None
        if isinstance(worker_state, dict) and "state" in worker_state:
            worker_state_state = _unpickle_if_bytes(worker_state.get("state"), max_depth=6)

        candidate_sd = None
        for src in [worker_state_state, worker_state]:
            if src is not None:
                candidate_sd = _find_torch_state_dict(src)
                if candidate_sd is not None:
                    break

        inferred_action_dim = None
        if isinstance(candidate_sd, dict):
            for k, v in candidate_sd.items():
                if not isinstance(k, str):
                    continue
                if not (k.endswith(".weight") or k.endswith(".bias")):
                    continue
                if "logits" not in k and "_logits" not in k:
                    continue
                try:
                    arr = np.asarray(_unpickle_if_bytes(v))
                except Exception:
                    continue
                if arr.ndim == 2 and arr.shape[0] > 1:
                    inferred_action_dim = int(arr.shape[0])
                    break

        # Init Ray
        os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        os.environ.setdefault("RAY_GRAFANA_HOST", "")
        os.environ.setdefault("RAY_PROMETHEUS_HOST", "")
        ray.init(
            ignore_reinit_error=True, include_dashboard=False,
            local_mode=True, num_cpus=1, log_to_driver=False,
        )

        with open(params_path, "rb") as f:
            config = pickle.load(f)
        config["num_workers"] = 0
        config["num_gpus"] = 0

        obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        if obs_space is None or act_space is None:
            raise ValueError("Env must expose observation_space and action_space.")

        if inferred_action_dim is not None:
            act_space = gym.spaces.Discrete(int(inferred_action_dim))
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            act_space = gym.spaces.Discrete(int(np.prod(act_space.nvec)))

        tune.registry.register_env("DummyEnv", lambda *_: _DummyGymEnv(obs_space, act_space))
        config["env"] = "DummyEnv"
        config["env_config"] = {}

        self._env_action_space = getattr(env, "action_space", None)
        self._action_flattener = None
        if ActionFlattener is not None and isinstance(self._env_action_space, gym.spaces.MultiDiscrete):
            try:
                self._action_flattener = ActionFlattener(self._env_action_space.nvec)
            except Exception:
                self._action_flattener = None

        cls = get_trainable_cls(ALGORITHM)
        trainer = cls(env=config["env"], config=config)

        # Load weights
        state = _unpickle_if_bytes(raw_state)
        worker_state = _unpickle_if_bytes(state.get("worker", {})) if isinstance(state, dict) else {}
        worker_state_state = None
        if isinstance(worker_state, dict) and "state" in worker_state:
            worker_state_state = _unpickle_if_bytes(worker_state.get("state"), max_depth=6)

        policy_states = None
        for source in [worker_state, worker_state_state]:
            if source is not None:
                policy_states = _find_nested_key(source, "policy_states")
                if policy_states is not None:
                    break
        if policy_states is None:
            policy_states = {}

        policy_id = next(iter(policy_states.keys()), POLICY_NAME) if policy_states else POLICY_NAME
        policy = trainer.get_policy(policy_id) or trainer.get_policy(POLICY_NAME)
        if policy is None:
            raise ValueError(f"Could not find policy (tried {policy_id!r} and {POLICY_NAME!r}).")

        p_state = _unpickle_if_bytes(policy_states.get(policy_id, {}))
        _load_weights_into_policy(policy, p_state, worker_state, worker_state_state)
        self.policy = policy

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            a = self.policy.compute_single_action(observation[player_id])
            a = _unwrap_action(a)

            if isinstance(self._env_action_space, gym.spaces.MultiDiscrete):
                if isinstance(a, (list, tuple, np.ndarray)):
                    arr = np.asarray(a)
                    if arr.ndim == 1 and arr.size == len(self._env_action_space.nvec):
                        actions[player_id] = arr.astype(np.int64)
                        continue
                flat = _coerce_int_action(a)
                if self._action_flattener is not None:
                    actions[player_id] = np.asarray(
                        self._action_flattener.lookup_action(int(flat)), dtype=np.int64
                    )
                else:
                    actions[player_id] = _unflatten_discrete_to_multidiscrete(
                        flat, np.asarray(self._env_action_space.nvec)
                    )
            else:
                actions[player_id] = _coerce_int_action(a)
        return actions
