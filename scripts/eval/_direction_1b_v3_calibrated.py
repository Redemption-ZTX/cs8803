#!/usr/bin/env python3
"""Direction 1.b v3 (A3): calibrated outcome predictor via random prefix truncation.

Key fix vs v2: each training sample is a random PREFIX of trajectory (not full).
Forces model to produce well-calibrated P(W | prefix[0..t]) for ALL t, not just
when seeing full episode. Solves the prefix-test miscalibration where v2 gave
biased P(W)~0.84 at prefix=5 regardless of true outcome.

Same model arch as v2 (4-layer transformer, 384 dim, dropout 0.2), same data
(15000 episodes), but training-time augmentation differs.
"""
import json, os, glob, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = 'docs/experiments/artifacts/trajectories/v3_all_30pair'
OUT_DIR = 'docs/experiments/artifacts/v3_dataset/direction_1b_v3'
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[1b-v3] device={DEVICE}', flush=True)


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


def collate_random_prefix(batch, max_T=120, train=True):
    """Random prefix truncation per item (train mode). Eval = full sequence."""
    B = len(batch)
    D = batch[0]['seq'].shape[1]
    rng = np.random.default_rng()
    if train:
        # Per-item random t ∈ [5, T_full]
        ts = []
        for b in batch:
            t_max = min(b['T'], max_T)
            ts.append(rng.integers(5, t_max + 1))
        Tmax = max(ts)
    else:
        Tmax = min(max_T, max(b['T'] for b in batch))
        ts = [min(b['T'], Tmax) for b in batch]

    seq = np.zeros((B, Tmax, D), dtype=np.float32)
    mask = np.zeros((B, Tmax), dtype=np.float32)
    label = np.zeros(B, dtype=np.float32)
    for i, b in enumerate(batch):
        T = int(ts[i])
        seq[i, :T] = b['seq'][:T]
        mask[i, :T] = 1.0
        label[i] = b['label']
    return torch.from_numpy(seq), torch.from_numpy(mask), torch.from_numpy(label), ts


def collate_train(batch): return collate_random_prefix(batch, train=True)
def collate_eval_full(batch): return collate_random_prefix(batch, train=False)


class OutcomePredictorV3(nn.Module):
    """Same arch as v2: 4-layer transformer + per-step P(W) head."""
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
    print(f'[1b-v3] found {len(npz_files)} npz files', flush=True)

    by_dir = {}
    for f in npz_files:
        d = os.path.basename(os.path.dirname(f))
        by_dir.setdefault(d, []).append(f)
    sources = sorted(by_dir.keys())
    np.random.seed(0)
    val_sources = list(np.random.choice(sources, 5, replace=False))
    train_sources = [s for s in sources if s not in val_sources]
    print(f'[1b-v3] train ({len(train_sources)}) | val ({len(val_sources)})', flush=True)

    train_files = [f for s in train_sources for f in by_dir[s]]
    val_files = [f for s in val_sources for f in by_dir[s]]
    train_ds = TrajectoryDataset(train_files)
    val_ds = TrajectoryDataset(val_files)
    print(f'[1b-v3] dataset built in {time.time()-t0:.1f}s: train={len(train_ds)} val={len(val_ds)}', flush=True)

    train_loader = DataLoader(train_ds, batch_size=48, shuffle=True, collate_fn=collate_train, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=96, shuffle=False, collate_fn=collate_eval_full, num_workers=2)

    model = OutcomePredictorV3().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    print(f'[1b-v3] model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

    PREFIX_LENS_VAL = [5, 10, 20, 30, 40, 50]
    sigmoid = nn.Sigmoid()

    best_val_acc = 0
    history = []
    for epoch in range(20):
        model.train()
        train_loss = 0; train_acc = 0; train_n = 0
        for seq, mask, label, _ in train_loader:
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
        val_acc_full = 0; val_n_full = 0
        # Per-prefix-length val acc to check calibration
        val_pw_per_prefix = {L: {'W': [], 'L': []} for L in PREFIX_LENS_VAL}

        with torch.no_grad():
            for seq, mask, label, ts in val_loader:
                seq, mask, label = seq.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
                logits = model(seq, mask)  # full sequence
                ep_logits = (logits * mask).sum(1) / mask.sum(1).clamp(min=1)
                pred = (ep_logits > 0).float()
                val_acc_full += (pred == label).sum().item()
                val_n_full += label.shape[0]

                # Per-prefix evaluation: re-run model on shorter prefixes
                for L in PREFIX_LENS_VAL:
                    if L >= seq.shape[1]: continue
                    prefix_seq = seq[:, :L]
                    prefix_mask = mask[:, :L]
                    plogits = model(prefix_seq, prefix_mask)
                    p_logits_ep = plogits.mean(1)
                    pw = sigmoid(p_logits_ep).cpu().numpy()
                    lab = label.cpu().numpy()
                    for i in range(len(lab)):
                        if lab[i] > 0.5: val_pw_per_prefix[L]['W'].append(float(pw[i]))
                        else: val_pw_per_prefix[L]['L'].append(float(pw[i]))

        val_acc_full /= max(val_n_full, 1)

        # Per-prefix-length stats
        prefix_str = ' '.join(
            f't{L}:W={np.mean(val_pw_per_prefix[L]["W"]):.2f}/L={np.mean(val_pw_per_prefix[L]["L"]):.2f}'
            for L in PREFIX_LENS_VAL if val_pw_per_prefix[L]['W'] and val_pw_per_prefix[L]['L']
        )
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_acc_full': val_acc_full, 'prefix_pw': {L: {
                            'W': float(np.mean(val_pw_per_prefix[L]['W'])) if val_pw_per_prefix[L]['W'] else None,
                            'L': float(np.mean(val_pw_per_prefix[L]['L'])) if val_pw_per_prefix[L]['L'] else None,
                        } for L in PREFIX_LENS_VAL}})
        print(f'[ep {epoch:2d}] tr_loss={train_loss:.4f} tr_acc={train_acc:.3f}  val_acc_full={val_acc_full:.3f}  {prefix_str}', flush=True)

        if val_acc_full > best_val_acc:
            best_val_acc = val_acc_full
            torch.save({'model': model.state_dict(), 'val_acc': val_acc_full, 'epoch': epoch},
                       f'{OUT_DIR}/best_outcome_predictor_v3_calibrated.pt')

    print(f'\n[1b-v3] best val_acc (full seq) = {best_val_acc:.3f}', flush=True)
    print(f'[1b-v3] vs v2 0.938 = {(best_val_acc - 0.938)*100:+.1f}pp (calibration may cost some full-seq acc)', flush=True)

    # Final calibration check
    final_prefix_gaps = {}
    for L in PREFIX_LENS_VAL:
        w_mean = np.mean(val_pw_per_prefix[L]['W']) if val_pw_per_prefix[L]['W'] else None
        l_mean = np.mean(val_pw_per_prefix[L]['L']) if val_pw_per_prefix[L]['L'] else None
        if w_mean and l_mean:
            final_prefix_gaps[L] = w_mean - l_mean
    print(f'\n=== Final per-prefix W vs L gap (calibration quality) ===')
    for L, gap in final_prefix_gaps.items():
        print(f'  prefix={L}: gap = {gap:+.3f}')

    avg_gap = float(np.mean(list(final_prefix_gaps.values()))) if final_prefix_gaps else 0
    print(f'\nAverage gap: {avg_gap:.3f}')
    if avg_gap >= 0.30:
        verdict = 'CALIBRATED — A2 PBRS viable, ΔV gives meaningful signal at all prefix lens'
    elif avg_gap >= 0.15:
        verdict = 'PARTIALLY CALIBRATED — usable, but signal weaker at short prefixes'
    else:
        verdict = 'STILL MISCALIBRATED — may need different training strategy'
    print(f'Verdict: {verdict}', flush=True)

    with open(f'{OUT_DIR}/training_history.json', 'w') as f:
        json.dump({'history': history, 'best_val_acc': best_val_acc,
                   'final_prefix_gaps': final_prefix_gaps,
                   'avg_gap': avg_gap, 'verdict': verdict}, f, indent=2)


if __name__ == '__main__':
    main()
