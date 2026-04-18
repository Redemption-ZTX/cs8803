import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_PYTHONPATH = os.environ.get("PYTHONPATH", "")
_PYTHONPATH_ENTRIES = [entry for entry in _PYTHONPATH.split(os.pathsep) if entry]
if REPO_ROOT not in _PYTHONPATH_ENTRIES:
    os.environ["PYTHONPATH"] = REPO_ROOT if not _PYTHONPATH else REPO_ROOT + os.pathsep + _PYTHONPATH

try:
    import sitecustomize as _project_sitecustomize  # noqa: F401
except Exception:
    _project_sitecustomize = None

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cs8803drl.branches.imitation_bc import BCTeamPolicy, save_bc_checkpoint

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_RUN_NAME = "BC_team_baseline_selfplay"
DEFAULT_LOCAL_DIR = "./ray_results"
DEFAULT_FCNET_HIDDENS = (512, 512)
DEFAULT_BATCH_SIZE = 4096
DEFAULT_EPOCHS = 30
DEFAULT_CHECKPOINT_FREQ = 5


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else int(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)


def _env_layers(name: str, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _find_manifest(path_value: str) -> Tuple[Path, Dict]:
    path = Path(path_value).resolve()
    if path.is_file():
        manifest_path = path
    else:
        manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find manifest.json from DATASET_DIR={path_value}")
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_manifests() -> List[Tuple[Path, Dict]]:
    dataset_dirs_raw = os.environ.get("DATASET_DIRS", "").strip()
    if dataset_dirs_raw:
        dataset_entries = [piece.strip() for piece in dataset_dirs_raw.split(",") if piece.strip()]
        if not dataset_entries:
            raise ValueError("DATASET_DIRS was set but no dataset paths were parsed.")
        return [_find_manifest(entry) for entry in dataset_entries]

    dataset_dir = os.environ.get("DATASET_DIR", "").strip()
    if not dataset_dir:
        raise ValueError("Either DATASET_DIR or DATASET_DIRS is required for BC training.")
    return [_find_manifest(dataset_dir)]


def _load_team_dataset(dataset_root: Path, *, max_samples: int = 0, seed: int = 0) -> Dict[str, np.ndarray]:
    team_dir = dataset_root / "team"
    shard_paths = sorted(team_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No team shards found under {team_dir}")

    rng = np.random.default_rng(int(seed))
    if max_samples > 0:
        shard_paths = list(shard_paths)
        rng.shuffle(shard_paths)

    obs_parts = []
    action_parts = []
    episode_parts = []
    step_parts = []
    side_parts = []
    collected = 0
    for shard_path in shard_paths:
        with np.load(shard_path) as shard:
            shard_size = int(shard["obs"].shape[0])
            if max_samples > 0:
                remaining = int(max_samples - collected)
                if remaining <= 0:
                    break
                if remaining < shard_size:
                    take_idx = np.sort(rng.choice(shard_size, size=remaining, replace=False))
                else:
                    take_idx = slice(None)
            else:
                take_idx = slice(None)

            obs_chunk = np.asarray(shard["obs"][take_idx], dtype=np.float32)
            action_chunk = np.asarray(shard["action"][take_idx], dtype=np.int64)
            episode_chunk = np.asarray(shard["episode"][take_idx], dtype=np.int64)
            step_chunk = np.asarray(shard["step"][take_idx], dtype=np.int64)
            side_chunk = np.asarray(shard["side"][take_idx], dtype=np.int64)

        obs_parts.append(obs_chunk)
        action_parts.append(action_chunk)
        episode_parts.append(episode_chunk)
        step_parts.append(step_chunk)
        side_parts.append(side_chunk)
        collected += int(obs_chunk.shape[0])

        if max_samples > 0 and collected >= max_samples:
            break

    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "action": np.concatenate(action_parts, axis=0),
        "episode": np.concatenate(episode_parts, axis=0),
        "step": np.concatenate(step_parts, axis=0),
        "side": np.concatenate(side_parts, axis=0),
    }


def _load_team_datasets(
    dataset_roots: List[Path],
    *,
    max_samples: int = 0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    if not dataset_roots:
        raise ValueError("dataset_roots must not be empty.")

    arrays_by_root: List[Dict[str, np.ndarray]] = []
    episode_offset = 0
    for dataset_idx, dataset_root in enumerate(dataset_roots):
        arrays = _load_team_dataset(dataset_root, max_samples=0, seed=seed + dataset_idx)
        arrays = {
            key: np.asarray(value).copy()
            for key, value in arrays.items()
        }
        arrays["episode"] = arrays["episode"] + int(episode_offset)
        episode_offset = int(arrays["episode"].max()) + 1 if len(arrays["episode"]) else episode_offset
        arrays_by_root.append(arrays)

    merged = {
        key: np.concatenate([arrays[key] for arrays in arrays_by_root], axis=0)
        for key in ("obs", "action", "episode", "step", "side")
    }

    if max_samples > 0 and len(merged["obs"]) > max_samples:
        rng = np.random.default_rng(int(seed))
        take_idx = np.sort(rng.choice(len(merged["obs"]), size=int(max_samples), replace=False))
        merged = {key: value[take_idx] for key, value in merged.items()}

    return merged


def _split_train_val_indices(episode: np.ndarray, side: np.ndarray, *, val_fraction: float, seed: int):
    group_id = episode.astype(np.int64) * 2 + side.astype(np.int64)
    unique_groups = np.unique(group_id)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique_groups)

    if val_fraction <= 0:
        train_mask = np.ones_like(group_id, dtype=bool)
        val_mask = np.zeros_like(group_id, dtype=bool)
        return train_mask, val_mask

    val_groups = max(1, int(round(len(unique_groups) * float(val_fraction))))
    val_group_set = set(int(v) for v in unique_groups[:val_groups].tolist())
    val_mask = np.asarray([int(g) in val_group_set for g in group_id], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask


def _write_curve_png(run_dir: Path, rows: List[Dict[str, float]]) -> Optional[Path]:
    if plt is None or not rows:
        return None

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_exact = [float(row["train_exact_match"]) for row in rows]
    val_exact = [float(row["val_exact_match"]) for row in rows]
    train_branch = [float(row["train_branch_acc"]) for row in rows]
    val_branch = [float(row["val_branch_acc"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(epochs, train_loss, label="train_loss", color="#1f77b4")
    axes[0].plot(epochs, val_loss, label="val_loss", color="#d62728")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("BC Team Policy Training")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_exact, label="train_exact", color="#2ca02c")
    axes[1].plot(epochs, val_exact, label="val_exact", color="#ff7f0e")
    axes[1].plot(epochs, train_branch, label="train_branch", color="#17becf")
    axes[1].plot(epochs, val_branch, label="val_branch", color="#9467bd")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    out_path = run_dir / "training_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _evaluate(model: BCTeamPolicy, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    branch_correct = 0
    exact_correct = 0
    num_branches = len(model.action_nvec)
    with torch.no_grad():
        for obs, action in loader:
            obs = obs.to(device)
            action = action.to(device)
            logits = model(obs)
            loss = sum(nn.functional.cross_entropy(logits[i], action[:, i]) for i in range(num_branches)) / num_branches

            pred = torch.stack([branch.argmax(dim=-1) for branch in logits], dim=-1)
            branch_correct += int((pred == action).sum().item())
            exact_correct += int((pred == action).all(dim=1).sum().item())
            batch_size = int(obs.shape[0])
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size

    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "branch_acc": branch_correct / float(denom * num_branches),
        "exact_match": exact_correct / float(denom),
        "samples": total_samples,
    }


def main() -> None:
    manifests = _resolve_manifests()
    manifest_paths = [manifest_path for manifest_path, _ in manifests]
    dataset_roots = [manifest_path.parent for manifest_path, _ in manifests]
    run_name = os.environ.get("RUN_NAME", "").strip() or f"{DEFAULT_RUN_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
    local_dir = Path(os.environ.get("LOCAL_DIR", DEFAULT_LOCAL_DIR)).resolve()
    run_dir = local_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_size = _env_int("BATCH_SIZE", DEFAULT_BATCH_SIZE)
    epochs = _env_int("EPOCHS", DEFAULT_EPOCHS)
    checkpoint_freq = _env_int("CHECKPOINT_FREQ", DEFAULT_CHECKPOINT_FREQ)
    seed = _env_int("SEED", 42)
    val_fraction = _env_float("VAL_FRACTION", 0.10)
    lr = _env_float("LR", 3e-4)
    weight_decay = _env_float("WEIGHT_DECAY", 1e-5)
    num_loader_workers = _env_int("NUM_LOADER_WORKERS", 4)
    fcnet_hiddens = _env_layers("FCNET_HIDDENS", DEFAULT_FCNET_HIDDENS)
    activation = os.environ.get("FCNET_ACTIVATION", "relu").strip() or "relu"
    max_samples = _env_int("MAX_SAMPLES", 0)
    num_gpus = _env_int("NUM_GPUS", 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if int(num_gpus) > 0 and torch.cuda.is_available() else "cpu")

    print("")
    print("BC Training Configuration")
    print(f"  run_dir:              {run_dir}")
    print(f"  dataset_count:        {len(dataset_roots)}")
    for idx, (manifest_path, dataset_root) in enumerate(zip(manifest_paths, dataset_roots), start=1):
        print(f"  dataset[{idx}]_dir:     {dataset_root}")
        print(f"  dataset[{idx}]_manifest:{manifest_path}")
    print(f"  device:               {device}")
    print(f"  batch_size:           {batch_size}")
    print(f"  epochs:               {epochs}")
    print(f"  checkpoint_freq:      {checkpoint_freq}")
    print(f"  val_fraction:         {val_fraction:.2f}")
    print(f"  lr:                   {lr}")
    print(f"  weight_decay:         {weight_decay}")
    print(f"  model_hidden:         {fcnet_hiddens}")
    print(f"  activation:           {activation}")
    print(f"  max_samples:          {max_samples if max_samples > 0 else 'disabled'}")
    print("")

    arrays = _load_team_datasets(dataset_roots, max_samples=max_samples, seed=seed)
    obs = arrays["obs"]
    action = arrays["action"]
    episode = arrays["episode"]
    side = arrays["side"]
    print(f"  loaded_samples:       {len(obs)}")

    train_mask, val_mask = _split_train_val_indices(episode, side, val_fraction=val_fraction, seed=seed)
    if not train_mask.any():
        raise ValueError("Train split is empty.")
    if not val_mask.any():
        raise ValueError("Validation split is empty.")

    action_nvec = [int(action[:, i].max()) + 1 for i in range(action.shape[1])]
    model = BCTeamPolicy(
        obs_dim=int(obs.shape[1]),
        action_nvec=action_nvec,
        fcnet_hiddens=fcnet_hiddens,
        activation=activation,
    ).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(obs[train_mask]).float(),
        torch.from_numpy(action[train_mask]).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(obs[val_mask]).float(),
        torch.from_numpy(action[val_mask]).long(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_loader_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_loader_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rows: List[Dict[str, float]] = []
    best_val_exact = -1.0
    best_epoch = 0
    best_checkpoint_dir = None

    progress_csv = run_dir / "progress.csv"
    with progress_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_branch_acc",
                "train_exact_match",
                "val_loss",
                "val_branch_acc",
                "val_exact_match",
                "train_samples",
                "val_samples",
                "elapsed_s",
            ],
        )
        writer.writeheader()

        started_at = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_samples = 0
            branch_correct = 0
            exact_correct = 0
            num_branches = len(action_nvec)

            for batch_obs, batch_action in train_loader:
                batch_obs = batch_obs.to(device, non_blocking=True)
                batch_action = batch_action.to(device, non_blocking=True)

                logits = model(batch_obs)
                loss = sum(
                    nn.functional.cross_entropy(logits[i], batch_action[:, i])
                    for i in range(num_branches)
                ) / num_branches

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                pred = torch.stack([branch.argmax(dim=-1) for branch in logits], dim=-1)
                branch_correct += int((pred == batch_action).sum().item())
                exact_correct += int((pred == batch_action).all(dim=1).sum().item())
                batch_count = int(batch_obs.shape[0])
                total_samples += batch_count
                total_loss += float(loss.item()) * batch_count

            train_metrics = {
                "loss": total_loss / max(total_samples, 1),
                "branch_acc": branch_correct / float(max(total_samples * num_branches, 1)),
                "exact_match": exact_correct / float(max(total_samples, 1)),
                "samples": total_samples,
            }
            val_metrics = _evaluate(model, val_loader, device)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_branch_acc": train_metrics["branch_acc"],
                "train_exact_match": train_metrics["exact_match"],
                "val_loss": val_metrics["loss"],
                "val_branch_acc": val_metrics["branch_acc"],
                "val_exact_match": val_metrics["exact_match"],
                "train_samples": train_metrics["samples"],
                "val_samples": val_metrics["samples"],
                "elapsed_s": time.time() - started_at,
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={row['train_loss']:.4f} "
                f"train_exact={row['train_exact_match']:.3f} "
                f"val_loss={row['val_loss']:.4f} "
                f"val_exact={row['val_exact_match']:.3f} "
                f"val_branch={row['val_branch_acc']:.3f}"
            )

            is_best = float(val_metrics["exact_match"]) > float(best_val_exact)
            should_checkpoint = is_best or (epoch % checkpoint_freq == 0) or (epoch == epochs)
            if should_checkpoint:
                checkpoint_dir = run_dir / f"checkpoint_{epoch:06d}"
                metadata = {
                    "format": "bc_team_policy_v1",
                    "obs_dim": int(obs.shape[1]),
                    "action_nvec": action_nvec,
                    "player_action_dim": int(action.shape[1] // 2),
                    "fcnet_hiddens": list(fcnet_hiddens),
                    "activation": activation,
                    "dataset_manifest": str(manifest_paths[0]),
                    "dataset_root": str(dataset_roots[0]),
                    "dataset_manifests": [str(path) for path in manifest_paths],
                    "dataset_roots": [str(path) for path in dataset_roots],
                    "train_samples": int(train_metrics["samples"]),
                    "val_samples": int(val_metrics["samples"]),
                    "epoch": int(epoch),
                    "train_loss": float(train_metrics["loss"]),
                    "train_branch_acc": float(train_metrics["branch_acc"]),
                    "train_exact_match": float(train_metrics["exact_match"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_branch_acc": float(val_metrics["branch_acc"]),
                    "val_exact_match": float(val_metrics["exact_match"]),
                }
                save_bc_checkpoint(checkpoint_dir=checkpoint_dir, model=model, metadata=metadata)
                if is_best:
                    best_val_exact = float(val_metrics["exact_match"])
                    best_epoch = int(epoch)
                    best_checkpoint_dir = checkpoint_dir

    curve_path = _write_curve_png(run_dir, rows)

    final_checkpoint_dir = run_dir / f"checkpoint_{epochs:06d}"
    summary = {
        "run_dir": str(run_dir),
        "manifest_json": str(manifest_paths[0]),
        "dataset_root": str(dataset_roots[0]),
        "manifest_jsons": [str(path) for path in manifest_paths],
        "dataset_roots": [str(path) for path in dataset_roots],
        "best_epoch": int(best_epoch),
        "best_val_exact_match": float(best_val_exact),
        "best_checkpoint_dir": str(best_checkpoint_dir) if best_checkpoint_dir else None,
        "final_checkpoint_dir": str(final_checkpoint_dir),
        "curve_path": str(curve_path) if curve_path else None,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("Training Summary")
    print(f"  run_dir:             {run_dir}")
    print(f"  dataset_manifests:   {', '.join(str(path) for path in manifest_paths)}")
    print(f"  best_epoch:          {best_epoch}")
    print(f"  best_val_exact:      {best_val_exact:.4f}")
    print(f"  best_checkpoint:     {best_checkpoint_dir}")
    print(f"  final_checkpoint:    {final_checkpoint_dir}")
    if curve_path:
        print(f"  curve_file:          {curve_path}")


if __name__ == "__main__":
    main()
