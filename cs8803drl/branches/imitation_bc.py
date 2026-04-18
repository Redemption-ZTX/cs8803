from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def _as_int_list(values: Iterable[int]) -> List[int]:
    return [int(v) for v in values]


def _resolve_activation(name: str):
    key = str(name or "relu").strip().lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported BC activation: {name!r}")
    return _ACTIVATIONS[key]


class BCTeamPolicy(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_nvec: Sequence[int],
        fcnet_hiddens: Sequence[int] = (512, 512),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_nvec = _as_int_list(action_nvec)
        self.fcnet_hiddens = _as_int_list(fcnet_hiddens)
        self.activation = str(activation)

        act_cls = _resolve_activation(self.activation)
        layers: List[nn.Module] = []
        in_dim = self.obs_dim
        for hidden_dim in self.fcnet_hiddens:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(act_cls())
            in_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.heads = nn.ModuleList(nn.Linear(in_dim, int(branch_n)) for branch_n in self.action_nvec)

    def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(obs)
        return [head(features) for head in self.heads]

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        return torch.stack([branch.argmax(dim=-1) for branch in logits], dim=-1)


def checkpoint_model_path(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_file():
        return path
    for candidate in (
        path / "model.pt",
        path / "checkpoint.pt",
        path / "model.pth",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate BC model file under {path}")


def checkpoint_metadata_path(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_dir():
        candidate = path / "metadata.json"
        if candidate.exists():
            return candidate
    model_path = checkpoint_model_path(path)
    candidate = model_path.with_name("metadata.json")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not locate BC metadata.json near {path}")


def save_bc_checkpoint(
    *,
    checkpoint_dir: Path,
    model: BCTeamPolicy,
    metadata: Dict,
) -> Tuple[Path, Path]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.pt"
    metadata_path = checkpoint_dir / "metadata.json"

    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_bc_checkpoint(path: Path, *, map_location="cpu") -> Tuple[BCTeamPolicy, Dict]:
    model_path = checkpoint_model_path(path)
    payload = torch.load(model_path, map_location=map_location)
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = json.loads(checkpoint_metadata_path(path).read_text(encoding="utf-8"))

    model = BCTeamPolicy(
        obs_dim=int(metadata["obs_dim"]),
        action_nvec=metadata["action_nvec"],
        fcnet_hiddens=metadata.get("fcnet_hiddens", [512, 512]),
        activation=metadata.get("activation", "relu"),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def _to_target_tensor(source_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(source_tensor, torch.Tensor):
        tensor = source_tensor.detach()
    else:
        tensor = torch.as_tensor(source_tensor)
    return tensor.to(dtype=target_tensor.dtype, device=target_tensor.device)


def _merge_tensor_prefix(
    *,
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    source_tensor = _to_target_tensor(source_tensor, target_tensor)
    if tuple(source_tensor.shape) == tuple(target_tensor.shape):
        return source_tensor, "copied"

    if source_tensor.ndim != target_tensor.ndim:
        return None, None

    if any(int(s) > int(t) for s, t in zip(source_tensor.shape, target_tensor.shape)):
        return None, None

    merged = target_tensor.detach().clone()
    if source_tensor.ndim == 1:
        merged[: source_tensor.shape[0]] = source_tensor
        return merged, "adapted"

    if source_tensor.ndim == 2:
        merged[: source_tensor.shape[0], : source_tensor.shape[1]] = source_tensor
        return merged, "adapted"

    return None, None


def warmstart_team_level_policy_from_bc(
    trainer,
    checkpoint_path,
    *,
    policy_name: str = "default_policy",
):
    checkpoint_path = (checkpoint_path or "").strip()
    if not checkpoint_path:
        return {"copied": 0, "adapted": 0, "skipped": 0}

    bc_model, metadata = load_bc_checkpoint(checkpoint_path, map_location="cpu")
    format_name = str(metadata.get("format", ""))
    if format_name != "bc_team_policy_v1":
        raise ValueError(
            "BC_WARMSTART_CHECKPOINT must point to a team-level BC checkpoint "
            f"(expected format=bc_team_policy_v1, got {format_name!r})."
        )

    policy = trainer.get_policy(policy_name)
    if policy is None:
        raise ValueError(f"Could not find target policy {policy_name!r}.")

    model = getattr(policy, "model", policy)
    target_state = model.state_dict()
    merged_state = dict(target_state)
    source_state = bc_model.state_dict()

    copied = 0
    adapted = 0
    skipped = 0

    backbone_mapping = {
        "_hidden_layers.0._model.0.weight": "backbone.0.weight",
        "_hidden_layers.0._model.0.bias": "backbone.0.bias",
        "_hidden_layers.1._model.0.weight": "backbone.2.weight",
        "_hidden_layers.1._model.0.bias": "backbone.2.bias",
    }

    for target_key, source_key in backbone_mapping.items():
        if target_key not in target_state or source_key not in source_state:
            skipped += 1
            continue
        merged_tensor, status = _merge_tensor_prefix(
            target_tensor=target_state[target_key],
            source_tensor=source_state[source_key],
        )
        if merged_tensor is None:
            skipped += 1
            continue
        merged_state[target_key] = merged_tensor
        if status == "copied":
            copied += 1
        else:
            adapted += 1

    # When actor/critic do not share layers, seed the value-side hidden stack
    # from the same BC backbone so PPO does not start with a fully random critic.
    value_mapping = {
        "_value_branch_separate.0._model.0.weight": "backbone.0.weight",
        "_value_branch_separate.0._model.0.bias": "backbone.0.bias",
        "_value_branch_separate.1._model.0.weight": "backbone.2.weight",
        "_value_branch_separate.1._model.0.bias": "backbone.2.bias",
    }
    for target_key, source_key in value_mapping.items():
        if target_key not in target_state or source_key not in source_state:
            continue
        merged_tensor, status = _merge_tensor_prefix(
            target_tensor=target_state[target_key],
            source_tensor=source_state[source_key],
        )
        if merged_tensor is None:
            skipped += 1
            continue
        merged_state[target_key] = merged_tensor
        if status == "copied":
            copied += 1
        else:
            adapted += 1

    logits_weight_key = "_logits._model.0.weight"
    logits_bias_key = "_logits._model.0.bias"
    action_nvec = tuple(int(v) for v in metadata.get("action_nvec", []))
    head_weight_keys = [f"heads.{idx}.weight" for idx in range(len(action_nvec))]
    head_bias_keys = [f"heads.{idx}.bias" for idx in range(len(action_nvec))]

    if (
        action_nvec
        and logits_weight_key in target_state
        and logits_bias_key in target_state
        and all(key in source_state for key in head_weight_keys + head_bias_keys)
    ):
        merged_weight, weight_status = _merge_tensor_prefix(
            target_tensor=target_state[logits_weight_key],
            source_tensor=torch.cat([source_state[key] for key in head_weight_keys], dim=0),
        )
        merged_bias, bias_status = _merge_tensor_prefix(
            target_tensor=target_state[logits_bias_key],
            source_tensor=torch.cat([source_state[key] for key in head_bias_keys], dim=0),
        )
        if merged_weight is None or merged_bias is None:
            skipped += 2
        else:
            merged_state[logits_weight_key] = merged_weight
            merged_state[logits_bias_key] = merged_bias
            copied += int(weight_status == "copied") + int(bias_status == "copied")
            adapted += int(weight_status == "adapted") + int(bias_status == "adapted")
    else:
        skipped += 2

    model.load_state_dict(merged_state, strict=False)
    try:
        trainer.workers.sync_weights()
    except Exception:
        pass
    return {
        "copied": copied,
        "adapted": adapted,
        "skipped": skipped,
    }
