import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

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
from cs8803drl.training.train_bc_team_policy import (
    _env_float,
    _env_int,
    _env_layers,
    _evaluate,
    _find_manifest,
    _split_train_val_indices,
    _write_curve_png,
)


DEFAULT_RUN_NAME = "BC_player_baseline_selfplay"
DEFAULT_LOCAL_DIR = "./ray_results"
DEFAULT_FCNET_HIDDENS = (512, 512)
DEFAULT_BATCH_SIZE = 4096
DEFAULT_EPOCHS = 30
DEFAULT_CHECKPOINT_FREQ = 5


def _load_player_dataset(dataset_root: Path, *, max_samples: int = 0, seed: int = 0) -> Dict[str, np.ndarray]:
    player_dir = dataset_root / "player"
    shard_paths = sorted(player_dir.glob("shard_*.npz"))
    derive_from_team = False
    if not shard_paths:
        team_dir = dataset_root / "team"
        shard_paths = sorted(team_dir.glob("shard_*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No player or team shards found under {dataset_root}")
        derive_from_team = True

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
            effective_shard_size = shard_size * 2 if derive_from_team else shard_size
            if max_samples > 0:
                remaining = int(max_samples - collected)
                if remaining <= 0:
                    break
                if remaining < effective_shard_size:
                    if derive_from_team:
                        take_idx = np.sort(rng.choice(effective_shard_size, size=remaining, replace=False))
                    else:
                        take_idx = np.sort(rng.choice(shard_size, size=remaining, replace=False))
                else:
                    take_idx = slice(None)
            else:
                take_idx = slice(None)

            if derive_from_team:
                team_obs = np.asarray(shard["obs"], dtype=np.float32)
                team_action = np.asarray(shard["action"], dtype=np.int64)
                half_obs = int(team_obs.shape[1] // 2)
                half_action = int(team_action.shape[1] // 2)
                derived_obs = np.concatenate([team_obs[:, :half_obs], team_obs[:, half_obs:]], axis=0)
                derived_action = np.concatenate([team_action[:, :half_action], team_action[:, half_action:]], axis=0)
                derived_episode = np.repeat(np.asarray(shard["episode"], dtype=np.int64), 2)
                derived_step = np.repeat(np.asarray(shard["step"], dtype=np.int64), 2)
                derived_side = np.repeat(np.asarray(shard["side"], dtype=np.int64), 2)

                obs_chunk = np.asarray(derived_obs[take_idx], dtype=np.float32)
                action_chunk = np.asarray(derived_action[take_idx], dtype=np.int64)
                episode_chunk = np.asarray(derived_episode[take_idx], dtype=np.int64)
                step_chunk = np.asarray(derived_step[take_idx], dtype=np.int64)
                side_chunk = np.asarray(derived_side[take_idx], dtype=np.int64)
            else:
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


def main() -> None:
    dataset_dir = os.environ.get("DATASET_DIR", "").strip()
    if not dataset_dir:
        raise ValueError("DATASET_DIR is required for BC player training.")

    manifest_path, manifest = _find_manifest(dataset_dir)
    dataset_root = manifest_path.parent
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
    print("BC Player Training Configuration")
    print(f"  run_dir:              {run_dir}")
    print(f"  dataset_dir:          {dataset_root}")
    print(f"  manifest_json:        {manifest_path}")
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

    arrays = _load_player_dataset(dataset_root, max_samples=max_samples, seed=seed)
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
                    "format": "bc_player_policy_v1",
                    "obs_dim": int(obs.shape[1]),
                    "action_nvec": action_nvec,
                    "flat_action_dim": int(np.prod(action_nvec)),
                    "fcnet_hiddens": list(fcnet_hiddens),
                    "activation": activation,
                    "dataset_manifest": str(manifest_path),
                    "dataset_root": str(dataset_root),
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
        "manifest_json": str(manifest_path),
        "dataset_root": str(dataset_root),
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
    print(f"  dataset_manifest:    {manifest_path}")
    print(f"  best_epoch:          {best_epoch}")
    print(f"  best_val_exact:      {best_val_exact:.4f}")
    print(f"  best_checkpoint:     {best_checkpoint_dir}")
    print(f"  final_checkpoint:    {final_checkpoint_dir}")
    if curve_path:
        print(f"  curve_file:          {curve_path}")


if __name__ == "__main__":
    main()
