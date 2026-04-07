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


def _unwrap_action(action):
    a = action
    while isinstance(a, (list, tuple)) and len(a) == 1:
        a = a[0]
    if isinstance(a, tuple) and len(a) >= 1:
        return a[0]
    return a


def _coerce_int_action(action) -> int:
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


def _unflatten_discrete_to_multidiscrete(flat: int, nvec: np.ndarray) -> np.ndarray:
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


def _unpickle_if_bytes(obj, *, max_depth: int = 3):
    cur = obj
    for _ in range(max_depth):
        if isinstance(cur, (bytes, bytearray)):
            cur = pickle.loads(cur)
            continue
        break
    return cur


def _find_nested_key(obj, key: str, *, max_depth: int = 6):
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


def _looks_like_torch_state_dict(d: dict) -> bool:
    if not d:
        return False
    # Heuristic: torch state_dict keys commonly end with ".weight"/".bias".
    keys = [k for k in d.keys() if isinstance(k, str)]
    if not keys:
        return False
    hits = sum(k.endswith(".weight") or k.endswith(".bias") for k in keys[:200])
    if hits < 2:
        return False

    # Also ensure the corresponding values look like numeric arrays/tensors (not object dtype).
    checked = 0
    for k in keys[:200]:
        if not (k.endswith(".weight") or k.endswith(".bias")):
            continue
        v = d.get(k)
        v = _unpickle_if_bytes(v)
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


def _find_torch_state_dict(obj, *, max_depth: int = 8):
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
                out[k] = _strip_optimizer_state(v)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(_strip_optimizer_state(v) for v in obj)
    return obj


class _DummyGymEnv(gym.Env):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        raise RuntimeError("Dummy env should never be stepped")

    def step(self, action):
        raise RuntimeError("Dummy env should never be stepped")


def _default_checkpoint_path() -> str:
    base = os.environ.get("TRAINED_RAY_CHECKPOINT", "").strip()
    if base:
        return base

    raise ValueError(
        "Missing TRAINED_RAY_CHECKPOINT env var. Example: "
        "TRAINED_RAY_CHECKPOINT=/path/to/checkpoint-20"
    )


class RayAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        checkpoint_path = _default_checkpoint_path()

        # Load checkpoint early so we can infer action dimension for the flattened action space.
        with open(checkpoint_path, "rb") as f:
            raw_state = pickle.load(f)
        state = _unpickle_if_bytes(raw_state)

        worker_state = state.get("worker", {}) if isinstance(state, dict) else {}
        worker_state = _unpickle_if_bytes(worker_state)
        worker_state_state = None
        if isinstance(worker_state, dict) and "state" in worker_state:
            worker_state_state = _unpickle_if_bytes(worker_state.get("state"), max_depth=6)

        # Try to find a torch-like state_dict to infer logits output dim.
        candidate_sd = None
        if worker_state_state is not None:
            candidate_sd = _find_torch_state_dict(worker_state_state)
        if candidate_sd is None:
            candidate_sd = _find_torch_state_dict(worker_state)

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

        os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        os.environ.setdefault("RAY_GRAFANA_HOST", "")
        os.environ.setdefault("RAY_PROMETHEUS_HOST", "")

        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            local_mode=True,
            num_cpus=1,
            log_to_driver=False,
        )

        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or its parent directory."
            )

        config["num_workers"] = 0
        config["num_gpus"] = 0

        obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        if obs_space is None or act_space is None:
            raise ValueError(
                "Evaluation env must expose observation_space and action_space."
            )

        # Build the policy with a flattened Discrete action space, matching the training setup.
        # If we inferred the dimension from the checkpoint (preferred), use it.
        if inferred_action_dim is not None:
            act_space = gym.spaces.Discrete(int(inferred_action_dim))
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            act_space = gym.spaces.Discrete(int(np.prod(act_space.nvec)))

        tune.registry.register_env(
            "DummyEnv", lambda *_: _DummyGymEnv(obs_space, act_space)
        )
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
        agent = cls(env=config["env"], config=config)

        # For evaluation, we only need policy weights. RLlib checkpoints also contain optimizer
        # state, which can fail to convert under newer NumPy/PyTorch combos.
        state = _unpickle_if_bytes(raw_state)

        worker_state = state.get("worker", {}) if isinstance(state, dict) else {}
        worker_state = _unpickle_if_bytes(worker_state)

        # RLlib 1.4 commonly stores most of the worker data under worker['state'].
        worker_state_state = None
        if isinstance(worker_state, dict) and "state" in worker_state:
            worker_state_state = _unpickle_if_bytes(worker_state.get("state"), max_depth=6)

        policy_states = None
        if isinstance(worker_state, dict) and "policy_states" in worker_state:
            policy_states = _unpickle_if_bytes(worker_state.get("policy_states"))
        if policy_states is None:
            # RLlib 1.4 checkpoints often store policy states under worker['state'].
            policy_states = _find_nested_key(worker_state, "policy_states")
        if policy_states is None and worker_state_state is not None:
            policy_states = _find_nested_key(worker_state_state, "policy_states")

        if policy_states is None:
            policy_states = {}

        policy_id = None
        if isinstance(policy_states, dict) and policy_states:
            policy_id = next(iter(policy_states.keys()))

        if policy_id is None:
            policy_id = POLICY_NAME

        policy = agent.get_policy(policy_id)
        if policy is None:
            policy = agent.get_policy(POLICY_NAME)

        if policy is None:
            raise ValueError(f"Could not find policy in restored trainer (tried {policy_id!r} and {POLICY_NAME!r}).")

        p_state = policy_states.get(policy_id, {}) if isinstance(policy_states, dict) else {}
        p_state = _unpickle_if_bytes(p_state)
        p_state = _strip_optimizer_state(p_state)

        weights = None
        if isinstance(p_state, dict) and "weights" in p_state:
            w = _unpickle_if_bytes(p_state["weights"])
            if isinstance(w, dict):
                weights = w
        elif isinstance(p_state, dict) and "model" in p_state:
            w = _unpickle_if_bytes(p_state["model"])
            if isinstance(w, dict):
                weights = w
        elif isinstance(p_state, dict) and "state_dict" in p_state:
            w = _unpickle_if_bytes(p_state["state_dict"])
            if isinstance(w, dict):
                weights = w
        elif isinstance(p_state, dict) and p_state:
            weights = p_state

        weights = _unpickle_if_bytes(weights)

        if weights is None:
            # Last resort: scan the checkpoint for a torch-like state_dict.
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
            self.policy = policy
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
