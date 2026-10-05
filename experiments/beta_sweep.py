"""
beta_sweep.py
=============
Runs the OrderedVAE across a grid of beta_end values x seeds, and selects
a single beta* as the knee of the reconstruction-loss-vs-beta curve --
using the SAME line-distance geometry as detect_elbow (vae_core.py),
rather than hand-picking beta=0.5 / beta=4.0 as separate "soft" / "hard"
regimes.

This replaces the earlier one-off soft (beta=0.5) and hard (beta=4.0) runs
with a single sweep that answers "why this beta?" with a concrete,
reproducible criterion.

Run:
    python experiments/beta_sweep.py

Requires token_embeddings.pt in the repository folder, created by
extract_embeddings_and_vae.py (embeddings are not re-extracted here).
"""

import os, sys
# Repo root, so scripts work from any working directory and find vae_core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from vae_core import (
    OrderedVAE, train_epoch, get_beta, compute_latent_variance,
    eval_held_out_recon, detect_elbow, detect_knee_convex_decreasing,
    mean_ci, DEVICE,
)

CFG = {
    'cache_path'   : os.path.join(ROOT, 'token_embeddings.pt'),   # reuse existing cache
    'd_input'      : 768,
    'd_hidden'     : 512,
    'd_latent'     : 64,
    'batch_size'   : 256,
    'epochs'       : 40,
    'lr'           : 1e-3,
    'weight_decay' : 1e-4,
    'beta_start'   : 0.0,
    # Denser grid than the original two-point {0.5, 4.0} comparison,
    # log-spaced since beta acts multiplicatively on KL pressure.
    # Extended with 3.0, 5.0, 6.0 to densify specifically around the
    # existing beta*=4.0 knee estimate, per the paper's stated limitation
    # that a denser grid there would sharpen the knee estimate. These three
    # new values reuse the existing 5-seed protocol; the original 7 betas x
    # 5 seeds = 35 runs are left untouched and will be skipped via the
    # results-cache resume logic in main() below -- only the 3 new betas x
    # 5 seeds = 15 new runs will actually execute.
    'betas'        : [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    'seeds'        : [42, 123, 7, 2024, 99],  # same 5 seeds as the original soft/hard runs
    'results_path' : os.path.join(RESULTS, 'beta_sweep_results.json'),
}

print(f"Device: {DEVICE}")


def run_single(embeddings, beta_end, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(embeddings.float())
    n_train = int(0.9 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False)

    model = OrderedVAE(CFG['d_input'], CFG['d_hidden'], CFG['d_latent']).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'],
                                   weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])

    for epoch in range(1, CFG['epochs'] + 1):
        beta = get_beta(epoch, CFG['epochs'], CFG['beta_start'], beta_end)
        train_epoch(model, train_loader, optimizer, beta)
        scheduler.step()

    variances       = compute_latent_variance(model, val_loader)
    elbow_k, _      = detect_elbow(variances)
    held_out_recon  = eval_held_out_recon(model, val_loader)

    return elbow_k, held_out_recon


def main():
    if not os.path.exists(CFG['cache_path']):
        raise FileNotFoundError(
            f"{CFG['cache_path']} not found. Run extract_embeddings_and_vae.py "
            f"once first (or copy its cached token_embeddings.pt here) so this "
            f"sweep reuses the same embeddings as your earlier soft/hard runs."
        )
    embeddings = torch.load(CFG['cache_path'], map_location='cpu')
    mean = embeddings.mean(0, keepdim=True)
    std  = embeddings.std(0, keepdim=True).clamp(min=1e-8)
    embeddings = (embeddings - mean) / std
    print(f"Loaded {embeddings.shape[0]:,} embeddings.")

    results = []
    if os.path.exists(CFG['results_path']):
        with open(CFG['results_path']) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} (beta, seed) run(s) already done.")
    done = {(r['beta'], r['seed']) for r in results}

    total_runs = len(CFG['betas']) * len(CFG['seeds'])
    print(f"Sweeping {len(CFG['betas'])} betas x {len(CFG['seeds'])} seeds "
          f"= {total_runs} runs ({CFG['epochs']} epochs each).")

    for beta in CFG['betas']:
        for seed in CFG['seeds']:
            if (beta, seed) in done:
                print(f"  beta={beta} seed={seed}: cached, skipping.")
                continue
            elbow_k, recon = run_single(embeddings, beta, seed)
            results.append({'beta': beta, 'seed': seed,
                             'elbow_k': elbow_k, 'held_out_recon': recon})
            with open(CFG['results_path'], 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  beta={beta:>5.2f}  seed={seed:>5d}  "
                  f"elbow_k={elbow_k:>3d}  held_out_recon={recon:.4f}")

    # ── Aggregate across seeds, per beta ──────────────────────────────────
    betas_unique = sorted(set(r['beta'] for r in results))
    mean_recon, mean_k, ci_k = [], [], []
    for b in betas_unique:
        ks     = [r['elbow_k'] for r in results if r['beta'] == b]
        losses = [r['held_out_recon'] for r in results if r['beta'] == b]
        m, lo, hi = mean_ci(ks)
        mean_k.append(m)
        ci_k.append(m - lo)
        mean_recon.append(float(np.mean(losses)))

    betas_arr = np.array(betas_unique)
    recon_arr = np.array(mean_recon)

    knee_idx = detect_knee_convex_decreasing(betas_arr, recon_arr)
    beta_star = float(betas_arr[knee_idx])
    k_star_mean = mean_k[knee_idx]
    k_star_ci = ci_k[knee_idx]

    print("\n" + "=" * 60)
    print("  RATE-DISTORTION CURVE (beta -> held-out recon loss, elbow_k)")
    print("=" * 60)
    for b, r, k, ci in zip(betas_arr, recon_arr, mean_k, ci_k):
        marker = "   <-- beta*" if b == beta_star else ""
        print(f"  beta={b:>5.2f}  held_out_recon={r:.4f}  "
              f"elbow_k={k:.1f} +/- {ci:.1f}{marker}")

    print(f"\n  Selected beta* = {beta_star}")
    print(f"  VAE dimensionality estimate at beta*: "
          f"{k_star_mean:.1f} [{k_star_mean - k_star_ci:.1f}, {k_star_mean + k_star_ci:.1f}]")
    print("=" * 60)

    summary = {
        'betas': betas_arr.tolist(),
        'mean_held_out_recon': recon_arr.tolist(),
        'mean_elbow_k': mean_k,
        'elbow_k_ci_halfwidth': ci_k,
        'beta_star': beta_star,
        'k_star_mean': k_star_mean,
        'k_star_ci_halfwidth': k_star_ci,
    }
    with open(os.path.join(RESULTS, 'beta_sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved -> beta_sweep_summary.json")


if __name__ == '__main__':
    main()
