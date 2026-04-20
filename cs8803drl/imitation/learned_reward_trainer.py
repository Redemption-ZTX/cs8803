"""Multi-head reward model training for snapshot-036 Path C (§3.6).

Loads trajectory .npz + .meta.json from ``scripts/eval/dump_trajectories.py``
output directories, builds per-head training labels from outcome +
multi-label failure bucket, and trains a shared-encoder + 5-head binary
classifier reward model.

Per-head formulation (§3.6.2):
    head k ∈ {late_def, low_poss, poor_conv, opp_fwd, territory}
    positive class: (s, a) from W trajectory (outcome=team0_win)
    negative class: (s, a) from L trajectory whose multi-label set contains k
    skipped:        (s, a) from L trajectory without label k, OR tie

Output:
    Trained checkpoint at ``<out_dir>/reward_model.pt`` plus
    ``metadata.json`` describing head names, input dims, γ, etc.

Can be used later by PPO integration wrapper (to be written) to produce
shaping reward ``r = sum_k w_k * head_k(s, a)`` at training time.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from cs8803drl.imitation.failure_buckets_v2 import (
    HEAD_NAMES_V2,
    HEAD_TO_LABELS_V2,
    classify_failure_v2,
    thresholds_dict as v2_thresholds_dict,
)


# ---------------------------------------------------------------------------
# Head / label mapping (v1 — original buckets from
# cs8803drl.evaluation.failure_cases.classify_failure)
# ---------------------------------------------------------------------------

HEAD_TO_LABELS: Dict[str, Tuple[str, ...]] = {
    "late_def": ("late_defensive_collapse",),
    "low_poss": ("low_possession",),
    "poor_conv": ("poor_conversion",),
    "opp_fwd": ("opponent_forward_progress",),
    "territory": ("territory_loss",),
}
HEAD_NAMES: Tuple[str, ...] = tuple(HEAD_TO_LABELS.keys())


def _head_matches(head: str, episode_labels: List[str], head_to_labels: Dict[str, Tuple[str, ...]]) -> bool:
    """Does this episode's multi-label set include any label mapped to `head`?"""
    targets = head_to_labels[head]
    return any(lbl in episode_labels for lbl in targets)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

OBS_DIM = 336
ACTION_MD_NDIM = 3
ACTION_MD_BASE = 3
# one-hot of MultiDiscrete([3, 3, 3]) = 3 + 3 + 3 = 9 dims
ACTION_ONEHOT_DIM = ACTION_MD_NDIM * ACTION_MD_BASE


class MultiHeadRewardModel(nn.Module):
    """Shared encoder + N binary-classifier heads.

    Each head outputs a scalar in ℝ (interpreted as logit for positive class = W).
    During deployment as shaping reward, we take tanh() to bound in (-1, +1) or just
    use raw scalar.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        use_action: bool = True,
        hidden_dims: Tuple[int, ...] = (256, 256),
        head_hidden: int = 128,
        head_names: Tuple[str, ...] = HEAD_NAMES,
    ):
        super().__init__()
        self.use_action = use_action
        self.head_names = tuple(head_names)
        input_dim = obs_dim + (ACTION_ONEHOT_DIM if use_action else 0)

        # shared encoder
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.shared_dim = prev

        # heads: (shared_dim → head_hidden → 1)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.shared_dim, head_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(head_hidden, 1),
                )
                for name in self.head_names
            }
        )

    def _action_one_hot(self, action_md: torch.Tensor) -> torch.Tensor:
        # action_md: (B, 3) int ∈ [0, 3)
        parts = []
        for i in range(ACTION_MD_NDIM):
            parts.append(F.one_hot(action_md[:, i].long(), num_classes=ACTION_MD_BASE).float())
        return torch.cat(parts, dim=-1)  # (B, 9)

    def forward(self, obs: torch.Tensor, action_md: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.use_action:
            assert action_md is not None, "use_action=True requires action_md"
            act_feat = self._action_one_hot(action_md)
            x = torch.cat([obs, act_feat], dim=-1)
        else:
            x = obs
        feat = self.encoder(x)
        return {name: head(feat).squeeze(-1) for name, head in self.heads.items()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SampleTable:
    """Flat table of training samples extracted from trajectory files.

    Per (s, a) sample, we store:
      - obs: (336,) float32
      - action_md: (3,) int8
      - head_signs: (num_heads,) float32 — label sign ∈ {-1, 0, +1} per head
      - head_masks: (num_heads,) bool   — True if this sample contributes to head's loss
      - weight: float — γ^(T-1-t) temporal credit
    """

    obs: np.ndarray           # (N, 336)
    action_md: np.ndarray     # (N, 3)
    head_signs: np.ndarray    # (N, H)
    head_masks: np.ndarray    # (N, H) bool
    weights: np.ndarray       # (N,)


def _build_sample_table(
    traj_dirs: List[Path],
    gamma: float = 0.95,
    max_episodes_per_dir: Optional[int] = None,
    verbose: bool = True,
    *,
    head_to_labels: Dict[str, Tuple[str, ...]] = HEAD_TO_LABELS,
    classifier_fn: Optional[callable] = None,
) -> SampleTable:
    """Walk trajectory .npz/.meta.json files and build flat training table.

    Args:
        head_to_labels: mapping of head name → labels it's positive on.
            Defaults to v1 mapping. Pass HEAD_TO_LABELS_V2 for v2 buckets.
        classifier_fn: optional ``(metrics_dict, outcome) -> {labels: [...]}``
            callable used to RE-derive labels from the saved metrics, ignoring
            ``meta["labels"]``. Required for v2 (since dumps were saved with
            v1 labels). If None, trust ``meta["labels"]`` (v1 path).
    """
    head_names = tuple(head_to_labels.keys())
    num_heads = len(head_names)
    obs_chunks: List[np.ndarray] = []
    act_chunks: List[np.ndarray] = []
    sign_chunks: List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    weight_chunks: List[np.ndarray] = []

    ep_count_by_outcome: Dict[str, int] = defaultdict(int)
    label_episode_counts: Dict[str, int] = defaultdict(int)

    for traj_dir in traj_dirs:
        npz_paths = sorted(traj_dir.glob("*.npz"))
        if max_episodes_per_dir is not None:
            npz_paths = npz_paths[:max_episodes_per_dir]
        if verbose:
            print(f"  scanning {traj_dir}: {len(npz_paths)} episodes")
        for npz_path in npz_paths:
            meta_path = npz_path.with_name(npz_path.name.replace(".npz", ".meta.json"))
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            outcome = meta.get("outcome")
            if classifier_fn is not None:
                cls = classifier_fn(meta.get("metrics") or {}, outcome) or {}
                labels = cls.get("labels", []) or []
            else:
                labels = meta.get("labels", []) or []
            ep_count_by_outcome[outcome] += 1
            for lbl in labels:
                label_episode_counts[lbl] += 1

            # per-head sign & mask for the WHOLE episode
            head_sign = np.zeros(num_heads, dtype=np.float32)
            head_mask = np.zeros(num_heads, dtype=bool)
            if outcome == "team0_win":
                # W: positive on all heads
                head_sign[:] = +1.0
                head_mask[:] = True
            elif outcome == "team1_win":
                # L: negative only on heads whose labels match this episode
                for h_idx, h in enumerate(head_names):
                    if _head_matches(h, labels, head_to_labels):
                        head_sign[h_idx] = -1.0
                        head_mask[h_idx] = True
                # heads not matched → mask=False (skipped for this head's loss)
            else:
                # tie: skip entirely
                continue

            if not head_mask.any():
                # L with NO bucket labels (e.g., pure unclear_loss) → skip
                continue

            data = np.load(npz_path)
            obs_a0 = data["obs_a0"]  # (T, 336)
            obs_a1 = data["obs_a1"]
            act_a0 = data["act_a0"]  # (T, 3)
            act_a1 = data["act_a1"]
            T = obs_a0.shape[0]
            if T == 0:
                continue

            # γ^(T-1-t) decayed weights
            weights = np.power(gamma, np.arange(T - 1, -1, -1, dtype=np.float32))  # (T,)

            # stack BOTH agents' samples (per-agent reward model sees either slot)
            for obs_arr, act_arr in ((obs_a0, act_a0), (obs_a1, act_a1)):
                obs_chunks.append(obs_arr.astype(np.float32))
                act_chunks.append(act_arr.astype(np.int8))
                sign_chunks.append(np.broadcast_to(head_sign, (T, num_heads)).copy())
                mask_chunks.append(np.broadcast_to(head_mask, (T, num_heads)).copy())
                weight_chunks.append(weights.copy())

    if not obs_chunks:
        raise RuntimeError("No trajectory samples collected. Check traj_dirs.")

    table = SampleTable(
        obs=np.concatenate(obs_chunks, axis=0),
        action_md=np.concatenate(act_chunks, axis=0),
        head_signs=np.concatenate(sign_chunks, axis=0),
        head_masks=np.concatenate(mask_chunks, axis=0),
        weights=np.concatenate(weight_chunks, axis=0),
    )

    if verbose:
        print()
        print(f"  [build] table size: {table.obs.shape[0]} samples")
        print(f"  [build] outcome episode counts: {dict(ep_count_by_outcome)}")
        print(f"  [build] label episode counts:")
        for lbl, cnt in sorted(label_episode_counts.items(), key=lambda x: -x[1]):
            print(f"    {lbl:<32s} {cnt:5d}")
        print(f"  [build] per-head sample count (mask=True):")
        for h_idx, h in enumerate(head_names):
            n_pos = int(((table.head_masks[:, h_idx]) & (table.head_signs[:, h_idx] > 0)).sum())
            n_neg = int(((table.head_masks[:, h_idx]) & (table.head_signs[:, h_idx] < 0)).sum())
            print(f"    {h:<22s} W_pos={n_pos:6d}  L_neg={n_neg:6d}")

    return table


class RewardDataset(Dataset):
    def __init__(self, table: SampleTable):
        self._table = table

    def __len__(self) -> int:
        return int(self._table.obs.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._table.obs[idx]).float(),
            torch.from_numpy(self._table.action_md[idx]).long(),
            torch.from_numpy(self._table.head_signs[idx]).float(),
            torch.from_numpy(self._table.head_masks[idx]).bool(),
            float(self._table.weights[idx]),
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# snapshot-039: Bradley-Terry pairwise loss + online refresh entry
# ---------------------------------------------------------------------------


def _bradley_terry_loss(
    r_W: torch.Tensor,
    r_L: torch.Tensor,
) -> torch.Tensor:
    """`-log σ(r(τ_W) - r(τ_L))` averaged over pairs.

    Both inputs should be shape (n_pairs,) scalars representing
    trajectory-aggregated rewards for the W and L trajectory in each pair.
    """
    diff = r_W - r_L  # (n_pairs,)
    # numerically stable: log σ(x) = -softplus(-x)
    return F.softplus(-diff).mean()


def refresh_reward_model_online(
    model: "MultiHeadRewardModel",
    offline_table: SampleTable,
    online_pairs: List[Tuple["EpisodeRecord", "EpisodeRecord"]],
    *,
    head_to_labels: Dict[str, Tuple[str, ...]],
    classifier_fn: Optional[callable] = None,
    loss: str = "bce",
    steps: int = 2000,
    batch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.95,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """Fine-tune reward model on offline pool + online pairs.

    Used by AdaptiveRewardCallback to refresh D during PPO training
    (snapshot-039 mechanism A). Does NOT retrain from scratch; continues
    from current model weights.

    Args:
        offline_table: pre-built SampleTable from Stage-1 trajectories.
        online_pairs: list of (W_episode, L_episode) tuples from recent rollouts.
            For BCE loss: each episode is unrolled into (s,a) samples with
            per-head labels (same as offline). For B-T loss: each pair gives
            one (r_W, r_L) comparison using trajectory-mean reward per head.
        head_to_labels: mapping of head name → set of matching failure-bucket labels.
        classifier_fn: if provided, re-derive labels from metrics for online eps.
        loss: "bce" (same as Stage 2) or "bt" (Bradley-Terry pairwise).
        steps: how many SGD steps to fine-tune.

    Returns metrics dict: {"train_loss_final", "online_pairs_used", "total_steps"}.
    """
    from torch.utils.data import DataLoader as _DL, Dataset as _DS, random_split as _rs  # noqa

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    head_names = tuple(head_to_labels.keys())

    # Build per-step sample table for online eps by calling _build_sample_table
    # flow but on raw EpisodeRecords. We inline a minimal version:
    if online_pairs:
        online_obs, online_act, online_sign, online_mask, online_w = [], [], [], [], []
        num_heads = len(head_names)
        for (w_ep, l_ep) in online_pairs:
            for ep, outcome in [(w_ep, "team0_win"), (l_ep, "team1_win")]:
                head_sign = np.zeros(num_heads, dtype=np.float32)
                head_mask = np.zeros(num_heads, dtype=bool)
                if outcome == "team0_win":
                    head_sign[:] = +1.0
                    head_mask[:] = True
                    labels = ["win"]
                else:
                    labels = list(ep.labels)
                    for h_idx, h in enumerate(head_names):
                        if any(l in labels for l in head_to_labels[h]):
                            head_sign[h_idx] = -1.0
                            head_mask[h_idx] = True
                if not head_mask.any():
                    continue
                T = ep.obs_a0.shape[0]
                if T == 0:
                    continue
                weights = np.power(gamma, np.arange(T - 1, -1, -1, dtype=np.float32))
                for obs_arr, act_arr in ((ep.obs_a0, ep.act_a0), (ep.obs_a1, ep.act_a1)):
                    online_obs.append(obs_arr.astype(np.float32))
                    online_act.append(act_arr.astype(np.int8))
                    online_sign.append(np.broadcast_to(head_sign, (T, num_heads)).copy())
                    online_mask.append(np.broadcast_to(head_mask, (T, num_heads)).copy())
                    online_w.append(weights.copy())
        if online_obs:
            online_obs = np.concatenate(online_obs, axis=0)
            online_act = np.concatenate(online_act, axis=0)
            online_sign = np.concatenate(online_sign, axis=0)
            online_mask = np.concatenate(online_mask, axis=0)
            online_w = np.concatenate(online_w, axis=0)
            # Concatenate with offline table
            combined_obs = np.concatenate([offline_table.obs, online_obs], axis=0)
            combined_act = np.concatenate([offline_table.action_md, online_act], axis=0)
            combined_sign = np.concatenate([offline_table.head_signs, online_sign], axis=0)
            combined_mask = np.concatenate([offline_table.head_masks, online_mask], axis=0)
            combined_w = np.concatenate([offline_table.weights, online_w], axis=0)
        else:
            combined_obs = offline_table.obs
            combined_act = offline_table.action_md
            combined_sign = offline_table.head_signs
            combined_mask = offline_table.head_masks
            combined_w = offline_table.weights
    else:
        combined_obs = offline_table.obs
        combined_act = offline_table.action_md
        combined_sign = offline_table.head_signs
        combined_mask = offline_table.head_masks
        combined_w = offline_table.weights

    n_samples = combined_obs.shape[0]
    running_loss = 0.0
    last_loss = 0.0
    for step in range(steps):
        idx = np.random.randint(0, n_samples, size=batch_size)
        ob = torch.from_numpy(combined_obs[idx]).float().to(device)
        ac = torch.from_numpy(combined_act[idx]).long().to(device)
        sg = torch.from_numpy(combined_sign[idx]).float().to(device)
        mk = torch.from_numpy(combined_mask[idx]).bool().to(device)
        wt = torch.from_numpy(combined_w[idx]).float().to(device)

        out = model(ob, ac)
        total_loss = torch.zeros((), device=device)
        for h_idx, h in enumerate(head_names):
            if loss == "bce":
                lh, _ = _masked_head_loss(out[h], sg[:, h_idx], mk[:, h_idx], wt)
                total_loss = total_loss + lh
            elif loss == "bt":
                # B-T on head-k: treat +1 sign as "preferred"; apply softplus-style
                # contrastive loss between pos and neg samples in this batch
                mask_f = mk[:, h_idx].float()
                pos_mask = (sg[:, h_idx] > 0) & mk[:, h_idx]
                neg_mask = (sg[:, h_idx] < 0) & mk[:, h_idx]
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pos_mean = (out[h] * pos_mask.float() * wt).sum() / (pos_mask.float() * wt).sum().clamp_min(1e-6)
                    neg_mean = (out[h] * neg_mask.float() * wt).sum() / (neg_mask.float() * wt).sum().clamp_min(1e-6)
                    lh = F.softplus(-(pos_mean - neg_mean))
                    total_loss = total_loss + lh
            else:
                raise ValueError(f"unknown loss: {loss}")

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        last_loss = float(total_loss.detach().cpu())
        running_loss += last_loss

    model.eval()
    return {
        "train_loss_final": last_loss,
        "train_loss_running": running_loss / max(steps, 1),
        "total_steps": steps,
        "n_pairs_used": len(online_pairs),
    }


def _masked_head_loss(
    head_logits: torch.Tensor,
    head_sign: torch.Tensor,
    head_mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BCE-style loss for one head, with per-sample masking + γ temporal weighting.

    Positive class (sign=+1): target=1, loss = -log σ(logit)
    Negative class (sign=-1): target=0, loss = -log(1 - σ(logit))
    Masked samples (mask=False): contribute 0 to loss.

    Returns (weighted_mean_loss, effective_batch_size).
    """
    # target: 1 if sign>0, 0 if sign<0, arbitrary if mask=False
    target = (head_sign > 0).float()
    per_sample_loss = F.binary_cross_entropy_with_logits(
        head_logits, target, reduction="none"
    )  # (B,)
    # apply mask + temporal weight
    mask_f = head_mask.float()
    weighted = per_sample_loss * mask_f * sample_weight
    denom = (mask_f * sample_weight).sum().clamp_min(1e-6)
    return weighted.sum() / denom, mask_f.sum()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    table: SampleTable,
    out_dir: Path,
    *,
    use_action: bool = True,
    hidden_dims: Tuple[int, ...] = (256, 256),
    head_hidden: int = 128,
    batch_size: int = 256,
    n_epochs: int = 10,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    head_to_labels: Dict[str, Tuple[str, ...]] = HEAD_TO_LABELS,
    label_version: str = "v1",
    extra_metadata: Optional[Dict] = None,
) -> Path:
    """Train multi-head reward model.

    head_to_labels controls which heads are produced. label_version + extra_metadata
    are recorded in the checkpoint config so the inference wrapper knows what
    bucket scheme it's using (and the wrapper auto-loads head_names from config).
    """
    head_names = tuple(head_to_labels.keys())
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device}  out_dir={out_dir}")
    print(f"[train] label_version={label_version}  heads={head_names}")

    dataset = RewardDataset(table)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MultiHeadRewardModel(
        obs_dim=OBS_DIM,
        use_action=use_action,
        hidden_dims=hidden_dims,
        head_hidden=head_hidden,
        head_names=head_names,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {param_count:,}  use_action={use_action}")
    print(f"[train] samples: train={n_train}  val={n_val}")

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(n_epochs):
        model.train()
        t_start = time.time()
        running_loss_per_head = {h: 0.0 for h in head_names}
        running_count = 0
        for obs, act, signs, masks, weights in train_loader:
            obs = obs.to(device)
            act = act.to(device)
            signs = signs.to(device)
            masks = masks.to(device)
            weights = weights.float().to(device)

            out = model(obs, act if use_action else None)
            total_loss = torch.zeros((), device=device)
            for h_idx, h in enumerate(head_names):
                loss_h, _ = _masked_head_loss(out[h], signs[:, h_idx], masks[:, h_idx], weights)
                total_loss = total_loss + loss_h
                running_loss_per_head[h] += float(loss_h.detach().cpu())

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running_count += 1

        # val
        model.eval()
        val_loss_per_head = {h: 0.0 for h in head_names}
        val_count = 0
        val_correct_per_head = {h: 0 for h in head_names}
        val_masked_per_head = {h: 0 for h in head_names}
        with torch.no_grad():
            for obs, act, signs, masks, weights in val_loader:
                obs = obs.to(device)
                act = act.to(device)
                signs = signs.to(device)
                masks = masks.to(device)
                weights = weights.float().to(device)
                out = model(obs, act if use_action else None)
                for h_idx, h in enumerate(head_names):
                    loss_h, _ = _masked_head_loss(out[h], signs[:, h_idx], masks[:, h_idx], weights)
                    val_loss_per_head[h] += float(loss_h.cpu())
                    preds = (out[h] > 0).float()
                    target = (signs[:, h_idx] > 0).float()
                    mask_f = masks[:, h_idx].float()
                    correct = ((preds == target).float() * mask_f).sum()
                    val_correct_per_head[h] += int(correct.cpu())
                    val_masked_per_head[h] += int(mask_f.sum().cpu())
                val_count += 1

        avg_val_loss = sum(v / max(val_count, 1) for v in val_loss_per_head.values()) / len(head_names)
        elapsed = time.time() - t_start

        print(
            f"[epoch {epoch+1:2d}/{n_epochs}] "
            f"tr_loss(mean/head)={sum(v/max(running_count,1) for v in running_loss_per_head.values())/len(head_names):.4f}  "
            f"val_loss(mean/head)={avg_val_loss:.4f}  "
            f"elapsed={elapsed:.1f}s"
        )
        for h in head_names:
            n_mask = val_masked_per_head[h]
            acc = val_correct_per_head[h] / max(n_mask, 1)
            print(
                f"  {h:<22s}  val_loss={val_loss_per_head[h]/max(val_count,1):.4f}  "
                f"acc={acc:.3f}  (n_mask={n_mask})"
            )

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_epoch = epoch
            ckpt_path = out_dir / "reward_model.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "obs_dim": OBS_DIM,
                        "use_action": use_action,
                        "hidden_dims": list(hidden_dims),
                        "head_hidden": head_hidden,
                        "head_names": list(head_names),
                        "action_md_ndim": ACTION_MD_NDIM,
                        "action_md_base": ACTION_MD_BASE,
                        "label_version": label_version,
                    },
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                ckpt_path,
            )
            print(f"  → saved best checkpoint (val_loss={best_val:.4f}) to {ckpt_path.name}")

    # final metadata
    meta_payload = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "n_epochs": n_epochs,
        "n_train": n_train,
        "n_val": n_val,
        "label_version": label_version,
        "head_names": list(head_names),
        "head_to_labels": {k: list(v) for k, v in head_to_labels.items()},
        "model_config": {
            "obs_dim": OBS_DIM,
            "use_action": use_action,
            "hidden_dims": list(hidden_dims),
            "head_hidden": head_hidden,
        },
    }
    if extra_metadata:
        meta_payload["classifier_thresholds"] = extra_metadata
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta_payload, indent=2))
    print(f"[train] done. best val_loss={best_val:.4f} @ epoch {best_epoch+1}")
    print(f"[train] checkpoint: {out_dir / 'reward_model.pt'}")
    print(f"[train] metadata:   {meta_path}")
    return out_dir / "reward_model.pt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train multi-head reward model from dumped trajectories.")
    parser.add_argument(
        "--traj-dir",
        nargs="+",
        required=True,
        help="One or more directories containing dumped .npz + .meta.json (from dump_trajectories.py).",
    )
    parser.add_argument("--out-dir", required=True, help="Where to save reward_model.pt + metadata.json.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Temporal credit decay.")
    parser.add_argument("--use-action", type=int, default=1, help="0=state-only reward, 1=state+action.")
    parser.add_argument("--hidden", type=str, default="256,256", help="Comma-separated hidden sizes for encoder.")
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes-per-dir", type=int, default=None)
    parser.add_argument(
        "--label-version",
        choices=["v1", "v2"],
        default="v2",
        help="Failure-bucket scheme. v1 = original 5 buckets from "
             "failure_cases.classify_failure (low_poss/late_def/...). "
             "v2 = data-driven redesign in failure_buckets_v2 "
             "(wasted_possession/possession_stolen/defensive_pin/...). Default v2.",
    )
    args = parser.parse_args()

    traj_dirs = [Path(d).expanduser().resolve() for d in args.traj_dir]
    out_dir = Path(args.out_dir).expanduser().resolve()
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    if args.label_version == "v2":
        head_to_labels = HEAD_TO_LABELS_V2
        classifier_fn = classify_failure_v2
        extra_metadata = v2_thresholds_dict()
        print(f"[cli] label_version=v2  thresholds={extra_metadata}")
    else:
        head_to_labels = HEAD_TO_LABELS
        classifier_fn = None  # trust meta["labels"]
        extra_metadata = None
        print("[cli] label_version=v1  (using meta['labels'] as-is)")

    print("[build] scanning trajectory directories...")
    table = _build_sample_table(
        traj_dirs,
        gamma=args.gamma,
        max_episodes_per_dir=args.max_episodes_per_dir,
        verbose=True,
        head_to_labels=head_to_labels,
        classifier_fn=classifier_fn,
    )

    train(
        table,
        out_dir,
        use_action=bool(args.use_action),
        hidden_dims=hidden,
        head_hidden=args.head_hidden,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        head_to_labels=head_to_labels,
        label_version=args.label_version,
        extra_metadata=extra_metadata,
    )


if __name__ == "__main__":
    main()
