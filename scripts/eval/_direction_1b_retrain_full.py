#!/usr/bin/env python3
"""Direction 1.b retrain — expanded dataset (15000 ep) + 4-layer transformer + dropout.

Replaces the prototype (78.8% val acc on 2000 ep) with full v3all dump dataset.
"""
import json, os, glob, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = 'docs/experiments/artifacts/trajectories/v3_all_30pair'
OUT_DIR = 'docs/experiments/artifacts/v3_dataset/direction_1b_v2'
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[1b-v2] device={DEVICE}', flush=True)


class TrajectoryDataset(Dataset):
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
                seq = np.concatenate([obs0, obs1], axis=1).astype(np.float32)
                label = 1.0 if outcome == 'team0_win' else 0.0
                self.items.append({'seq': seq, 'label': label, 'T': T})
            except Exception:
                continue

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def collate(batch, max_T=120):
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
    return torch.from_numpy(seq), torch.from_numpy(mask), torch.from_numpy(label)


class OutcomePredictorV2(nn.Module):
    """4-layer transformer encoder + per-step P(W) head + dropout."""
    def __init__(self, obs_dim=672, d_model=384, n_layers=4, n_heads=6, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(obs_dim, d_model)
        self.proj_dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, 200, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=1024,
                                            batch_first=True, dropout=dropout, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x, mask):
        B, T, D = x.shape
        h = self.proj_dropout(self.proj(x)) + self.pos_embed[:, :T]
        kpm = (mask < 0.5)
        h = self.enc(h, src_key_padding_mask=kpm)
        return self.head(h).squeeze(-1)


def main():
    t0 = time.time()
    npz_files = sorted(glob.glob(f'{DATA_DIR}/*/*.npz'))
    print(f'[1b-v2] found {len(npz_files)} npz files', flush=True)

    by_dir = {}
    for f in npz_files:
        d = os.path.basename(os.path.dirname(f))
        by_dir.setdefault(d, []).append(f)
    sources = sorted(by_dir.keys())
    print(f'[1b-v2] {len(sources)} sources', flush=True)

    np.random.seed(0)
    val_sources = list(np.random.choice(sources, 5, replace=False))
    train_sources = [s for s in sources if s not in val_sources]
    print(f'[1b-v2] train ({len(train_sources)}) | val ({len(val_sources)})', flush=True)
    print(f'[1b-v2] val sources: {val_sources}', flush=True)

    train_files = [f for s in train_sources for f in by_dir[s]]
    val_files = [f for s in val_sources for f in by_dir[s]]
    print(f'[1b-v2] loading {len(train_files)} train + {len(val_files)} val', flush=True)

    train_ds = TrajectoryDataset(train_files)
    val_ds = TrajectoryDataset(val_files)
    print(f'[1b-v2] dataset built in {time.time()-t0:.1f}s: train={len(train_ds)} val={len(val_ds)}', flush=True)

    train_loader = DataLoader(train_ds, batch_size=48, shuffle=True, collate_fn=collate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=96, shuffle=False, collate_fn=collate, num_workers=2)

    model = OutcomePredictorV2().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    print(f'[1b-v2] model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

    best_val_acc = 0
    history = []
    for epoch in range(20):
        model.train()
        train_loss = 0; train_acc = 0; train_n = 0
        for seq, mask, label in train_loader:
            seq, mask, label = seq.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            opt.zero_grad()
            logits = model(seq, mask)
            label_bc = label.unsqueeze(1).expand_as(logits)
            losses = bce(logits, label_bc)
            loss = (losses * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * mask.sum().item()
            with torch.no_grad():
                ep_logits = (logits * mask).sum(1) / mask.sum(1).clamp(min=1)
                pred = (ep_logits > 0).float()
                train_acc += (pred == label).sum().item()
                train_n += label.shape[0]
        train_loss /= max(train_n, 1)
        train_acc /= max(train_n, 1)

        model.eval()
        val_loss = 0; val_acc = 0; val_n = 0
        per_step_pw = []
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
                pw = torch.sigmoid(logits).cpu().numpy()
                m = mask.cpu().numpy()
                lab = label.cpu().numpy()
                for i in range(pw.shape[0]):
                    Ti = int(m[i].sum())
                    if Ti < 10: continue
                    early = float(pw[i, :Ti // 3].mean())
                    late = float(pw[i, 2 * Ti // 3:Ti].mean())
                    per_step_pw.append((lab[i], early, late))
        val_loss /= max(val_n, 1)
        val_acc /= max(val_n, 1)

        wins = [t for t in per_step_pw if t[0] > 0.5]
        losses_list = [t for t in per_step_pw if t[0] < 0.5]
        if wins and losses_list:
            wm = np.mean([t[1] for t in wins]), np.mean([t[2] for t in wins])
            lm = np.mean([t[1] for t in losses_list]), np.mean([t[2] for t in losses_list])
            wm_str = f'{wm[0]:.2f}/{wm[1]:.2f}'
            lm_str = f'{lm[0]:.2f}/{lm[1]:.2f}'
        else:
            wm_str = lm_str = 'N/A'

        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': val_loss, 'val_acc': val_acc,
                        'val_W_PW_e/l': wm_str, 'val_L_PW_e/l': lm_str})
        print(f'[ep {epoch:2d}] tr_loss={train_loss:.4f} tr_acc={train_acc:.3f}  '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.3f}  '
              f'W e/l={wm_str}  L e/l={lm_str}', flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': model.state_dict(), 'val_acc': val_acc, 'epoch': epoch},
                       f'{OUT_DIR}/best_outcome_predictor_v2.pt')

    print(f'\n[1b-v2] best val_acc = {best_val_acc:.3f} ({best_val_acc*100:.1f}%)', flush=True)
    print(f'[1b-v2] vs prototype 78.8% (2000 ep): delta = {(best_val_acc - 0.788)*100:+.1f}pp', flush=True)
    if best_val_acc >= 0.85:
        verdict = 'STRONG — Direction 1.b validated, ready to wire as reward'
    elif best_val_acc >= 0.78:
        verdict = 'CONFIRMED — same as prototype, more data not needed; size limited by inherent task'
    elif best_val_acc >= 0.70:
        verdict = 'WEAK — even more data didn\'t help, paradigm has ceiling'
    else:
        verdict = 'COLLAPSE — more data hurt, need investigation'
    print(f'[1b-v2] verdict: {verdict}', flush=True)
    with open(f'{OUT_DIR}/training_history.json', 'w') as f:
        json.dump({'history': history, 'best_val_acc': best_val_acc, 'verdict': verdict}, f, indent=2)


if __name__ == '__main__':
    main()
