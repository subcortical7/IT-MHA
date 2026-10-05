"""
verify_pruning_slicing.py
=========================
Checks that build_pruned_model() in pruning_baseline_sst2.py is exact:
a pruned k=10 model must produce the same outputs as the full k=32 model
with the 22 discarded heads masked out (head_mask), before any fine-tuning.

Uses randomly initialised weights, since the check is about the slicing
code, not about a trained model. Runs in a few seconds on CPU.

Run (from the same folder as pruning_baseline_sst2.py):
    python checks/verify_pruning_slicing.py
"""

import os, sys
# Repo root, so scripts work from any working directory and find vae_core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')

import numpy as np
import torch
from experiments import pruning_baseline_sst2 as pb

N_TRIALS = 5
torch.manual_seed(0)
rng = np.random.default_rng(0)
cfg = pb.CFG

max_err = 0.0
for trial in range(N_TRIALS):
    full = pb.TinyTransformer(
        vocab_size=cfg['vocab_size'], d_model=cfg['d_model'],
        n_heads=cfg['n_heads_full'], n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'], max_seq_len=cfg['max_seq_len'],
        n_classes=cfg['n_classes'], dropout=cfg['dropout'],
    ).to(pb.DEVICE).eval()

    # A random choice of kept heads per layer, as the importance step would produce.
    kept = [sorted(rng.choice(cfg['n_heads_full'], cfg['n_heads_pruned'],
                              replace=False).tolist())
            for _ in range(cfg['n_layers'])]

    pruned = pb.build_pruned_model(full, kept, cfg).eval()

    # Mask the discarded heads in the full model.
    for l, heads in enumerate(kept):
        m = torch.zeros(cfg['n_heads_full'])
        m[heads] = 1.0
        full.layers[l].attn.head_mask = m

    # Random token batch with padding at the end of each sequence.
    B, T = 8, 64
    ids = torch.randint(1, cfg['vocab_size'], (B, T), device=pb.DEVICE)
    mask = torch.ones(B, T, dtype=torch.long, device=pb.DEVICE)
    for b in range(B):
        L = int(rng.integers(5, T))
        ids[b, L:] = 0
        mask[b, L:] = 0

    with torch.no_grad():
        out_full = full(ids, mask)
        out_pruned = pruned(ids, mask)

    err = (out_full - out_pruned).abs().max().item()
    max_err = max(max_err, err)
    print(f"trial {trial}: max |difference| = {err:.2e}")

print(f"\nLargest difference over {N_TRIALS} trials: {max_err:.2e}")
print("PASS: pruned model matches masked full model."
      if max_err < 1e-5 else
      "FAIL: outputs differ -- check build_pruned_model().")
