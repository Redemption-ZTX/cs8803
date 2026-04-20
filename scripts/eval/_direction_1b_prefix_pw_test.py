#!/usr/bin/env python3
"""Test: Does P(W | prefix[0..t]) evolve as t grows? Determines A1 vs A2 viability.

Loads best_outcome_predictor_v2.pt. For each val episode, evaluates prefix at
t=5/10/20/30/40/50 step lengths. Measures:
  - mean P(W) at each prefix length
  - within-episode spread (max - min P(W) across prefix lengths)
  - correlation: P(W|prefix=t) vs prefix length

If within-episode spread > 0.15 → PBRS A2 viable (signal evolves)
If within-episode spread < 0.05 → A2 zero-signal, must use A1 direct reward
"""
import json, os, glob, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = 'docs/experiments/artifacts/trajectories/v3_all_30pair'
OUT_DIR = 'docs/experiments/artifacts/v3_dataset/direction_1b_v2'
CKPT = f'{OUT_DIR}/best_outcome_predictor_v2.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[prefix-test] device={DEVICE}', flush=True)


# Same model class as retrain
class OutcomePredictorV2(nn.Module):
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
    print(f'[prefix-test] loading checkpoint {CKPT}', flush=True)
    state = torch.load(CKPT, map_location=DEVICE)
    model = OutcomePredictorV2().to(DEVICE)
    model.load_state_dict(state['model'])
    model.eval()
    print(f'[prefix-test] loaded model val_acc={state["val_acc"]:.3f}', flush=True)

    # Load val sources (same seed → same val split)
    npz_files = sorted(glob.glob(f'{DATA_DIR}/*/*.npz'))
    by_dir = {}
    for f in npz_files:
        d = os.path.basename(os.path.dirname(f))
        by_dir.setdefault(d, []).append(f)
    sources = sorted(by_dir.keys())
    np.random.seed(0)
    val_sources = list(np.random.choice(sources, 5, replace=False))
    val_files = [f for s in val_sources for f in by_dir[s]]
    print(f'[prefix-test] val sources: {val_sources}', flush=True)
    print(f'[prefix-test] {len(val_files)} val files', flush=True)

    # Subsample for speed
    np.random.seed(1)
    val_files_sub = list(np.random.choice(val_files, min(2000, len(val_files)), replace=False))
    print(f'[prefix-test] using {len(val_files_sub)} subsample', flush=True)

    PREFIX_LENS = [5, 10, 20, 30, 40, 50]
    sigmoid = nn.Sigmoid()

    # Per-episode: outcome + P(W) at each prefix len
    records = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(val_files_sub), bs):
            batch_files = val_files_sub[i:i + bs]
            for f in batch_files:
                meta = json.load(open(f.replace('.npz', '.meta.json')))
                outcome = meta.get('outcome')
                if outcome not in ('team0_win', 'team1_win'): continue
                d = np.load(f)
                obs0, obs1 = d['obs_a0'], d['obs_a1']
                T = obs0.shape[0]
                if T < 5 or T > 200: continue
                full_seq = np.concatenate([obs0, obs1], axis=1).astype(np.float32)
                label = 1.0 if outcome == 'team0_win' else 0.0

                # Per prefix len: feed only prefix [0..t] + mask
                pw_per_t = {}
                for L in PREFIX_LENS:
                    if L > T: continue
                    seq = full_seq[:L]  # (L, D)
                    seq_t = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # (1, L, D)
                    mask_t = torch.ones(1, L, device=DEVICE)
                    logits = model(seq_t, mask_t)  # (1, L)
                    # Take MEAN logit over the prefix as ep-level prediction at this prefix length
                    ep_logit = logits.mean().item()
                    pw_per_t[L] = float(sigmoid(torch.tensor(ep_logit)).item())
                if len(pw_per_t) >= 3:
                    records.append({'label': label, 'T_full': T, 'pw': pw_per_t})

    print(f'\n[prefix-test] {len(records)} episodes processed', flush=True)

    # Aggregate
    print(f'\n=== Mean P(W) at each prefix length (W vs L episodes) ===')
    print(f'{"prefix_len":>11s}  {"mean_W":>8s}  {"mean_L":>8s}  {"gap":>6s}  {"n":>5s}')
    for L in PREFIX_LENS:
        w_pw = [r['pw'][L] for r in records if r['label'] > 0.5 and L in r['pw']]
        l_pw = [r['pw'][L] for r in records if r['label'] < 0.5 and L in r['pw']]
        if not w_pw or not l_pw: continue
        mw = float(np.mean(w_pw)); ml = float(np.mean(l_pw))
        print(f'{L:>11d}  {mw:>8.3f}  {ml:>8.3f}  {mw-ml:>6.3f}  {len(w_pw)+len(l_pw):>5d}')

    # Within-episode spread (across prefix lengths)
    print(f'\n=== Within-episode P(W) spread (max - min across prefix lengths) ===')
    print('(Higher spread → P(W) actually evolves with prefix → A2 PBRS viable)')
    spreads_W = []; spreads_L = []
    for r in records:
        pws = list(r['pw'].values())
        if len(pws) < 3: continue
        spread = max(pws) - min(pws)
        if r['label'] > 0.5: spreads_W.append(spread)
        else: spreads_L.append(spread)
    if spreads_W and spreads_L:
        print(f'  W eps: mean spread = {np.mean(spreads_W):.3f}  median = {np.median(spreads_W):.3f}  90pct = {np.percentile(spreads_W, 90):.3f}')
        print(f'  L eps: mean spread = {np.mean(spreads_L):.3f}  median = {np.median(spreads_L):.3f}  90pct = {np.percentile(spreads_L, 90):.3f}')

    avg_spread = float(np.mean(spreads_W + spreads_L))
    print(f'\n=== Verdict ===')
    print(f'Average within-episode P(W) spread: {avg_spread:.3f}')
    if avg_spread > 0.15:
        verdict = 'A2 (PBRS ΔV) VIABLE — P(W) evolves meaningfully across prefix; ΔV gives signal'
    elif avg_spread > 0.05:
        verdict = 'A2 BORDERLINE — small signal, A1 likely better; consider hybrid'
    else:
        verdict = 'A2 ZERO-SIGNAL — P(W) static across prefix; MUST use A1 direct reward'
    print(f'Verdict: {verdict}', flush=True)

    with open(f'{OUT_DIR}/prefix_pw_test_result.json', 'w') as f:
        json.dump({
            'avg_spread': avg_spread,
            'verdict': verdict,
            'n_records': len(records),
            'spreads_W_stats': {'mean': float(np.mean(spreads_W)), 'median': float(np.median(spreads_W))},
            'spreads_L_stats': {'mean': float(np.mean(spreads_L)), 'median': float(np.median(spreads_L))},
        }, f, indent=2)


if __name__ == '__main__':
    main()
