"""Self-contained submission package for 055v2_extend@1750.

This module intentionally avoids imports from the project repository so the
folder can be copied into a fresh clone and zipped on its own.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

# Compatibility patch for older ml-agents stacks used by soccer_twos.
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "object"):
    np.object = object  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "str"):
    np.str = str  # type: ignore[attr-defined]

import gym
import torch
import torch.nn as nn

from soccer_twos import AgentInterface


_AGENT_DIR = Path(__file__).resolve().parent
_CHECKPOINT_PATH = _AGENT_DIR / "checkpoint_001750" / "checkpoint-1750"
_POLICY_NAME = "default_policy"


def _resolve_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    candidates = sorted(child for child in path.iterdir() if child.name.startswith("checkpoint-"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* file found in directory: {path}")
    return candidates[0]


def _unpickle_if_bytes(obj, *, max_depth: int = 3):
    cur = obj
    for _ in range(max_depth):
        if isinstance(cur, (bytes, bytearray)):
            try:
                cur = pickle.loads(cur)
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, ModuleNotFoundError):
                break
            continue
        break
    return cur


def _find_nested_key(obj, key, *, max_depth: int = 6):
    obj = _unpickle_if_bytes(obj, max_depth=max_depth)
    if max_depth <= 0:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return _unpickle_if_bytes(obj[key], max_depth=max_depth)
        for value in obj.values():
            found = _find_nested_key(value, key, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_nested_key(value, key, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def _strip_optimizer_state(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.asarray([], dtype=np.float32)
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in {
                "optimizer_variables",
                "optimizer_state",
                "optim_state",
                "_optimizer_variables",
            }:
                out[key] = []
            else:
                out[key] = _strip_optimizer_state(value)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(_strip_optimizer_state(value) for value in obj)
    return obj


def _drop_optimizer_state_keys(weights):
    if not isinstance(weights, dict):
        return weights
    blocked = {
        "optimizer_variables",
        "optimizer_state",
        "optim_state",
        "_optimizer_variables",
    }
    return {
        key: value
        for key, value in weights.items()
        if not (
            isinstance(key, str)
            and (key in blocked or key.startswith("_optimizer"))
        )
    }


def _looks_like_torch_state_dict(obj) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    keys = [key for key in obj.keys() if isinstance(key, str)]
    if not keys:
        return False
    hits = sum(key.endswith(".weight") or key.endswith(".bias") for key in keys[:200])
    return hits >= 2


def _find_torch_state_dict(obj, *, max_depth: int = 8):
    obj = _unpickle_if_bytes(obj, max_depth=3)
    if max_depth <= 0:
        return None
    if isinstance(obj, dict):
        if _looks_like_torch_state_dict(obj):
            return obj
        for value in obj.values():
            found = _find_torch_state_dict(value, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_torch_state_dict(value, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def _extract_torch_weights_from_checkpoint(checkpoint_path: Path, policy_name: str = _POLICY_NAME):
    checkpoint_file = _resolve_checkpoint_file(checkpoint_path)
    with checkpoint_file.open("rb") as handle:
        raw_state = pickle.load(handle)

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

    source_policy_id = policy_name
    if policy_states:
        if policy_name in policy_states:
            source_policy_id = policy_name
        else:
            source_policy_id = next(iter(policy_states.keys()))

    policy_state = _unpickle_if_bytes(policy_states.get(source_policy_id, {}))
    policy_state = _strip_optimizer_state(policy_state)

    weights = None
    for key in ("weights", "model", "state_dict"):
        if isinstance(policy_state, dict) and key in policy_state:
            candidate = _unpickle_if_bytes(policy_state[key])
            if isinstance(candidate, dict):
                weights = candidate
                break
    if weights is None and isinstance(policy_state, dict) and policy_state:
        weights = policy_state
    weights = _drop_optimizer_state_keys(_unpickle_if_bytes(weights))

    if weights is None:
        candidate = None
        if worker_state_state is not None:
            candidate = _find_torch_state_dict(worker_state_state)
        if candidate is None:
            candidate = _find_torch_state_dict(worker_state)
        if candidate is None:
            raise ValueError("Could not find torch weights in checkpoint.")
        weights = _drop_optimizer_state_keys(candidate)

    return weights


def _student_weights_only(weights: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in weights.items()
        if isinstance(key, str) and not key.startswith("teacher_model.")
    }


def _infer_architecture(weights: Dict[str, object]) -> Dict[str, int]:
    shared_layer_names = sorted(
        key for key in weights.keys()
        if key.startswith("shared_encoder.") and key.endswith(".weight")
    )
    merge_layer_names = sorted(
        key for key in weights.keys()
        if key.startswith("merge_mlp.") and key.endswith(".weight")
    )
    if not shared_layer_names or not merge_layer_names:
        raise ValueError("Checkpoint does not contain expected shared_encoder/merge_mlp weights.")

    encoder_hiddens = [int(np.asarray(weights[key]).shape[0]) for key in shared_layer_names]
    merge_hiddens = [int(np.asarray(weights[key]).shape[0]) for key in merge_layer_names]
    half_obs_dim = int(np.asarray(weights[shared_layer_names[0]]).shape[1])
    head_dim = int(np.asarray(weights["q_proj.weight"]).shape[0])
    encoder_out = int(encoder_hiddens[-1])
    if encoder_out % head_dim != 0:
        raise ValueError(
            f"Encoder output {encoder_out} is not divisible by attention head dim {head_dim}."
        )
    n_tokens = int(encoder_out // head_dim)
    num_outputs = int(np.asarray(weights["logits_layer.weight"]).shape[0])
    return {
        "half_obs_dim": half_obs_dim,
        "encoder_out": encoder_out,
        "n_tokens": n_tokens,
        "head_dim": head_dim,
        "num_outputs": num_outputs,
        "encoder_hiddens_0": int(encoder_hiddens[0]),
        "encoder_hiddens_1": int(encoder_hiddens[1]),
        "merge_hiddens_0": int(merge_hiddens[0]),
        "merge_hiddens_1": int(merge_hiddens[1]),
    }


class _SiameseCrossAttentionPolicy(nn.Module):
    def __init__(
        self,
        *,
        half_obs_dim: int,
        encoder_hiddens: tuple[int, int],
        merge_hiddens: tuple[int, int],
        n_tokens: int,
        head_dim: int,
        num_outputs: int,
    ):
        super().__init__()
        encoder_out = int(encoder_hiddens[-1])
        if n_tokens * head_dim != encoder_out:
            raise ValueError("Cross-attention shape mismatch.")

        self._half_obs_dim = int(half_obs_dim)
        self._n_tokens = int(n_tokens)
        self._head_dim = int(head_dim)
        self._encoder_out = encoder_out

        self.shared_encoder = nn.Sequential(
            nn.Linear(self._half_obs_dim, int(encoder_hiddens[0])),
            nn.ReLU(),
            nn.Linear(int(encoder_hiddens[0]), int(encoder_hiddens[1])),
            nn.ReLU(),
        )

        self.q_proj = nn.Linear(self._head_dim, self._head_dim, bias=False)
        self.k_proj = nn.Linear(self._head_dim, self._head_dim, bias=False)
        self.v_proj = nn.Linear(self._head_dim, self._head_dim, bias=False)

        merge_in = self._encoder_out * 4
        self.merge_mlp = nn.Sequential(
            nn.Linear(merge_in, int(merge_hiddens[0])),
            nn.ReLU(),
            nn.Linear(int(merge_hiddens[0]), int(merge_hiddens[1])),
            nn.ReLU(),
        )
        self.logits_layer = nn.Linear(int(merge_hiddens[1]), int(num_outputs))
        self.value_layer = nn.Linear(int(merge_hiddens[1]), 1)

    def _attend(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(src_tokens)
        k = self.k_proj(tgt_tokens)
        v = self.v_proj(tgt_tokens)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self._head_dim)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        obs0 = joint_obs[:, : self._half_obs_dim]
        obs1 = joint_obs[:, self._half_obs_dim :]

        feat0 = self.shared_encoder(obs0)
        feat1 = self.shared_encoder(obs1)

        batch_size = feat0.shape[0]
        tokens0 = feat0.view(batch_size, self._n_tokens, self._head_dim)
        tokens1 = feat1.view(batch_size, self._n_tokens, self._head_dim)

        attn0_flat = self._attend(tokens0, tokens1).reshape(batch_size, self._encoder_out)
        attn1_flat = self._attend(tokens1, tokens0).reshape(batch_size, self._encoder_out)

        merged = self.merge_mlp(torch.cat([feat0, attn0_flat, feat1, attn1_flat], dim=1))
        return self.logits_layer(merged)


def _load_inference_model(checkpoint_path: Path) -> _SiameseCrossAttentionPolicy:
    raw_weights = _extract_torch_weights_from_checkpoint(checkpoint_path)
    weights = _student_weights_only(raw_weights)
    arch = _infer_architecture(weights)
    model = _SiameseCrossAttentionPolicy(
        half_obs_dim=arch["half_obs_dim"],
        encoder_hiddens=(arch["encoder_hiddens_0"], arch["encoder_hiddens_1"]),
        merge_hiddens=(arch["merge_hiddens_0"], arch["merge_hiddens_1"]),
        n_tokens=arch["n_tokens"],
        head_dim=arch["head_dim"],
        num_outputs=arch["num_outputs"],
    )

    state_dict = {}
    for key, value in weights.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().float()
        else:
            arr = np.asarray(_unpickle_if_bytes(value))
            if arr.dtype == object:
                try:
                    arr = arr.astype(np.float32)
                except Exception:
                    continue
            tensor = torch.from_numpy(arr).float()
        state_dict[key] = tensor

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("teacher_model.")]
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys after teacher stripping: {unexpected}")
    required_missing = [key for key in missing if not key.startswith("_")]
    if required_missing:
        raise ValueError(f"Missing required model keys: {required_missing}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class Agent(AgentInterface):
    """Self-contained team-level inference wrapper for 055v2_extend@1750."""

    def __init__(self, env: gym.Env):
        super().__init__()

        checkpoint_path = _resolve_checkpoint_file(_CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Bundled checkpoint missing: {checkpoint_path}")

        raw_action_space = getattr(env, "action_space", None)
        if raw_action_space is None or not isinstance(raw_action_space, gym.spaces.MultiDiscrete):
            raise ValueError(
                "CS8803DRL_AGENT expects env.action_space to be gym.spaces.MultiDiscrete."
            )
        raw_nvec = np.asarray(raw_action_space.nvec, dtype=np.int64).reshape(-1)
        if raw_nvec.size == 0 or not np.all(raw_nvec == raw_nvec[0]):
            raise ValueError(f"Unexpected action space nvec: {raw_nvec!r}")

        self._player_action_dim = int(raw_nvec.size)
        self._factor_classes = int(raw_nvec[0])
        self._joint_action_dim = self._player_action_dim * 2
        self._expected_num_outputs = self._joint_action_dim * self._factor_classes

        self._device = torch.device("cpu")
        self._model = _load_inference_model(checkpoint_path).to(self._device)
        actual_num_outputs = int(self._model.logits_layer.weight.shape[0])
        if actual_num_outputs != self._expected_num_outputs:
            raise ValueError(
                "Checkpoint/action-space mismatch: "
                f"model outputs {actual_num_outputs} logits, expected {self._expected_num_outputs}."
            )

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        player_ids = sorted(int(player_id) for player_id in observation.keys())
        if len(player_ids) != 2:
            raise ValueError(
                "CS8803DRL_AGENT expects exactly 2 local teammates in observation, "
                f"got ids={player_ids}"
            )

        joint_obs = np.concatenate(
            [
                np.asarray(observation[player_ids[0]], dtype=np.float32).reshape(-1),
                np.asarray(observation[player_ids[1]], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        obs_tensor = torch.from_numpy(joint_obs).to(self._device).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(obs_tensor).reshape(
                self._joint_action_dim, self._factor_classes
            )
            team_action = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)

        return {
            player_ids[0]: team_action[: self._player_action_dim].copy(),
            player_ids[1]: team_action[self._player_action_dim :].copy(),
        }
