#!/usr/bin/env python3
"""Direction 1.b prototype: sequence model over per-step state → predict W/L outcome.

Uses 051 trajectory dumps (2000 episodes, full per-step). Train Transformer
encoder over obs sequence, output per-step P(team0_win). Reports val acc + per-step
P(W) curve to verify outcome IS predictable from state sequence.

If val acc >> 50%, this validates Direction 1.b paradigm.
Subsequent step (not done here): wire as PBRS shaping into PPO callback.
"""
import json
import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = 'docs/experiments/artifacts/trajectories/051_strong_vs_strong'
OUT_DIR = 'docs/experiments/artifacts/v3_dataset/direction_1b'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[1b] device={DEVICE}', flush=True)


class TrajectoryDataset(Dataset):
    """Loads (obs_a0, obs_a1, act_a0, act_a1, outcome) from 051 trajectory npz.
    Concatenates obs_a0 + obs_a1 → (T, 672). outcome from sister meta.json."""

    def __init__(self, npz_files):
        self.items = []
        for npz_path in npz_files:
            meta_path = npz_path.replace('.npz', '.meta.json')
            try:
                meta = json.load(open(meta_path))
                outcome = meta.get('outcome')
                if outcome not in ('team0_win', 'team1_win'):
                    continue
                d = np.load(npz_path)
                obs0, obs1 = d['obs_a0'], d['obs_a1']
                T = obs0.shape[0]
                if T < 5 or T > 200:
                    continue
                seq = np.concatenate([obs0, obs1], axis=1).astype(np.float32)  # (T, 672)
                label = 1.0 if outcome == 'team0_win' else 0.0
                self.items.append({'seq': seq, 'label': label, 'T': T,
                                   'team0_module': meta.get('team0_module', ''),
                                   'team1_module': meta.get('team1_module', '')})
            except Exception:
                continue

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, max_T=80):
    """Pad sequences to max_T. Returns (seq[B,Tmax,D], mask[B,Tmax], label[B])."""
    B = len(batch)
    D = batch[0]['seq'].shape[1]
    Tmax = min(max_T, max(b['T'] for b in batch))
    seq = np.zeros((B, Tmax, D), dtype=np.float32)
    mask = np.zeros((B, Tmax), dtype=np.float32)
    label = np.zeros(B, dtype=np.float32)
    for i, b in enumerate(batch):
        T = min(b['T'], Tmax)
        seq[i, :T] = b['seq'][:T]
        mask[i, :T] = 1.0
        label[i] = b['label']
    return (torch.from_numpy(seq), torch.from_numpy(mask), torch.from_numpy(label))


class OutcomePredictor(nn.Module):
    """Small Transformer encoder + per-step P(W) head."""

    def __init__(self, obs_dim=672, d_model=256, n_layers=2, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(obs_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 200, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=512,
                                            batch_first=True, dropout=0.1, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T) 1=valid
        B, T, D = x.shape
        h = self.proj(x) + self.pos_embed[:, :T]
        # transformer expects key_padding_mask: True=ignore
        kpm = (mask < 0.5)  # (B, T)
        h = self.enc(h, src_key_padding_mask=kpm)
        logits = self.head(h).squeeze(-1)  # (B, T)
        return logits


def main():
    t0 = time.time()
    npz_files = sorted(glob.glob(f'{DATA_DIR}/*/*.npz'))
    print(f'[1b] found {len(npz_files)} npz files', flush=True)

    # Train/val split BY DIRECTORY (so val is held-out source)
    by_dir = {}
    for f in npz_files:
        d = os.path.basename(os.path.dirname(f))
        by_dir.setdefault(d, []).append(f)
    sources = sorted(by_dir.keys())
    print(f'[1b] sources ({len(sources)}): {sources}', flush=True)
    # Val: 2 sources (1 forward, 1 reverse) random pick
    np.random.seed(0)
    val_sources = list(np.random.choice(sources, 2, replace=False))
    train_sources = [s for s in sources if s not in val_sources]
    print(f'[1b] train sources ({len(train_sources)}): {train_sources}', flush=True)
    print(f'[1b] val sources ({len(val_sources)}): {val_sources}', flush=True)

    train_files = [f for s in train_sources for f in by_dir[s]]
    val_files = [f for s in val_sources for f in by_dir[s]]
    print(f'[1b] loading {len(train_files)} train + {len(val_files)} val', flush=True)

    train_ds = TrajectoryDataset(train_files)
    val_ds = TrajectoryDataset(val_files)
    print(f'[1b] dataset built in {time.time()-t0:.1f}s: train={len(train_ds)} val={len(val_ds)}', flush=True)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=2)

    model = OutcomePredictor().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    print(f'[1b] model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

    best_val_acc = 0
    history = []
    for epoch in range(15):
        # Train
        model.train()
        train_loss = 0; train_acc = 0; train_n = 0
        for seq, mask, label in train_loader:
            seq, mask, label = seq.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            opt.zero_grad()
            logits = model(seq, mask)  # (B, T)
            # Per-step BCE: broadcast label to all valid steps
            label_bc = label.unsqueeze(1).expand_as(logits)
            losses = bce(logits, label_bc)  # (B, T)
            loss = (losses * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * mask.sum().item()
            # Acc: take mean logit over valid steps for each episode, threshold 0
            with torch.no_grad():
                ep_logits = (logits * mask).sum(1) / mask.sum(1).clamp(min=1)
                pred = (ep_logits > 0).float()
                train_acc += (pred == label).sum().item()
                train_n += label.shape[0]
        train_loss /= max(train_n, 1)
        train_acc /= max(train_n, 1)

        # Val
        model.eval()
        val_loss = 0; val_acc = 0; val_n = 0
        per_step_pw = []  # for early/mid/late P(W) curve
        with torch.no_grad():
            for seq, mask, label in val_loader:
                seq, mask, label = seq.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
                logits = model(seq, mask)
                label_bc = label.unsqueeze(1).expand_as(logits)
                losses = bce(logits, label_bc)
                vloss = (losses * mask).sum() / mask.sum().clamp(min=1)
                val_loss += vloss.item() * mask.sum().item()
                ep_logits = (logits * mask).sum(1) / mask.sum(1).clamp(min=1)
                pred = (ep_logits > 0).float()
                val_acc += (pred == label).sum().item()
                val_n += label.shape[0]
                # Per-step P(W) collection
                pw = torch.sigmoid(logits).cpu().numpy()
                m = mask.cpu().numpy()
                lab = label.cpu().numpy()
                for i in range(pw.shape[0]):
                    Ti = int(m[i].sum())
                    if Ti < 10: continue
                    early = float(pw[i, :Ti // 3].mean())
                    mid = float(pw[i, Ti // 3:2 * Ti // 3].mean())
                    late = float(pw[i, 2 * Ti // 3:Ti].mean())
                    per_step_pw.append((lab[i], early, mid, late))
        val_loss /= max(val_n, 1)
        val_acc /= max(val_n, 1)

        # Per-step curve for W vs L val episodes
        wins = [t for t in per_step_pw if t[0] > 0.5]
        losses_list = [t for t in per_step_pw if t[0] < 0.5]
        if wins and losses_list:
            wm = (np.mean([t[1] for t in wins]), np.mean([t[2] for t in wins]), np.mean([t[3] for t in wins]))
            lm = (np.mean([t[1] for t in losses_list]), np.mean([t[2] for t in losses_list]), np.mean([t[3] for t in losses_list]))
            wm_str = f'{wm[0]:.2f}/{wm[1]:.2f}/{wm[2]:.2f}'
            lm_str = f'{lm[0]:.2f}/{lm[1]:.2f}/{lm[2]:.2f}'
        else:
            wm_str = lm_str = 'N/A'

        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': val_loss, 'val_acc': val_acc,
                        'val_W_PW_e/m/l': wm_str, 'val_L_PW_e/m/l': lm_str})
        print(f'[ep {epoch:2d}] train_loss={train_loss:.4f} acc={train_acc:.3f}  '
              f'val_loss={val_loss:.4f} acc={val_acc:.3f}  '
              f'val W early/mid/late P(W)={wm_str}  L early/mid/late P(W)={lm_str}', flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': model.state_dict(), 'val_acc': val_acc, 'epoch': epoch},
                       f'{OUT_DIR}/best_outcome_predictor.pt')

    # Final report
    print(f'\n[1b] best val_acc = {best_val_acc:.3f} ({best_val_acc*100:.1f}%)', flush=True)
    print(f'[1b] benchmark random = 0.500, useful threshold = 0.55-0.60', flush=True)

    if best_val_acc >= 0.60:
        verdict = 'STRONG SIGNAL — Direction 1.b validated, worth wiring as PBRS reward'
    elif best_val_acc >= 0.55:
        verdict = 'MARGINAL SIGNAL — Direction 1.b borderline, expand data + bigger model'
    else:
        verdict = 'NO SIGNAL — per-step state alone does not predict outcome'
    print(f'[1b] verdict: {verdict}', flush=True)

    with open(f'{OUT_DIR}/training_history.json', 'w') as f:
        json.dump({'history': history, 'best_val_acc': best_val_acc, 'verdict': verdict}, f, indent=2)
    print(f'[1b] saved to {OUT_DIR}/', flush=True)


if __name__ == '__main__':
    main()
