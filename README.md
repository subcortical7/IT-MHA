# Choosing the Number of Attention Heads from Intrinsic Dimension

Code and results for the paper **"Choosing the Number of Attention Heads from Intrinsic Dimension: A Small-Scale Study"** by Arjun Boya, Charan Sai Bandaru, Masthanaiah Pillimetla and Kavana Chidanandappa.

Paper: *arXiv link coming soon*

## What this is

Transformers usually get their head count from convention. We test a simple alternative: estimate the intrinsic dimension of token representations without labels, then use roughly that many heads.

- A nested-dropout VAE on DistilBERT embeddings of WikiText-2 gives an estimate of **k\* = 9.2** (95% CI [8.6, 9.8]), with the KL weight chosen at the knee of the VAE's rate–distortion curve.
- In a small 4-layer transformer trained from scratch, **10 heads match 32 heads** on SST-2 and MRPC (paired TOST over 10 seeds, ±1 point margin) with 69% fewer attention parameters.
- Training 10 heads from the start slightly beats training 32 and pruning to 10.

The study is small: two tasks, one architecture, no pretrained models. See the paper's limitations section before relying on this.

## Setup

```bash
git clone https://github.com/subcortical7/IT-MHA.git
cd IT-MHA
pip install -r requirements.txt
```

The experiments ran on PyTorch 2.13 with CUDA 12.6 on a single NVIDIA GPU. For a GPU build, install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/).

## Layout

```
├── vae_core.py                     shared VAE model and elbow/knee detectors
├── extract_embeddings_and_vae.py   embedding extraction + single-β VAE runs
├── experiments/                    scripts behind the paper's tables and figures
├── checks/                         sanity checks (elbow detector, pruning slicing)
└── results/                        JSON results from the paper's runs; plots are written here too
```

## Files

| File | What it does | Paper |
|---|---|---|
| `vae_core.py` | Shared VAE model, training loop, elbow and knee detectors | §4.1 |
| `extract_embeddings_and_vae.py` | Extracts DistilBERT embeddings of WikiText-2 and caches them as `token_embeddings.pt` | §4.1 |
| `experiments/beta_sweep.py` | Trains the VAE over 10 KL weights × 5 seeds and picks β\* | Table 2, Figure 1 |
| `checks/synthetic_elbow_check.py` | Tests the elbow detector on synthetic spectra with known elbows | Appendix A.2 |
| `experiments/sst2_multiseed_k10_vs_k32.py` | 10 vs 32 heads on SST-2, 10 paired seeds | Table 1 |
| `experiments/mrpc_k10_vs_k32.py` | 10 vs 32 heads on MRPC, 10 paired seeds | Table 1 |
| `experiments/pruning_baseline_sst2.py` | Trains 32 heads, prunes to 10 by ablation importance, fine-tunes | Table 3 |
| `checks/verify_pruning_slicing.py` | Checks the pruned model matches the masked full model | §4.3 |

Result files in `results/` (one entry per run, including training curves and timings):

| File | Contents |
|---|---|
| `sst2_multiseed_results.json` | 20 SST-2 runs |
| `mrpc_multiseed_results.json` | 20 MRPC runs |
| `pruning_baseline_results.json` | 10 pruning runs |
| `beta_sweep_results.json` | 50 VAE runs |
| `beta_sweep_summary.json` | Per-β averages, β\* and k\* |

## Reproducing the results

Run the scripts in this order. Paths are resolved relative to the repository, so they work from any working directory:

```bash
python extract_embeddings_and_vae.py               # creates token_embeddings.pt (~180 MB)
python experiments/beta_sweep.py                   # Table 2, Figure 1
python checks/synthetic_elbow_check.py             # Appendix A.2 (~1 s)
python experiments/sst2_multiseed_k10_vs_k32.py    # Table 1, SST-2
python experiments/mrpc_k10_vs_k32.py              # Table 1, MRPC
python experiments/pruning_baseline_sst2.py        # Table 3 (reads the SST-2 results)
python checks/verify_pruning_slicing.py            # pruning check (~seconds)
```

**The scripts resume from their result files.** Each one skips any run already recorded in its JSON file in `results/`. Because our results are included, running a script as-is only recomputes the statistics from them. To retrain from scratch, rename or delete the corresponding JSON file first.

`extract_embeddings_and_vae.py` also trains a set of VAEs at a single β after caching the embeddings; only the cached embeddings are needed for `beta_sweep.py`. `token_embeddings.pt` is not included in the repository because of its size.

Recorded training time was about 5.2 GPU-hours in total (SST-2 2.55 h, MRPC 0.24 h, pruning baseline 2.41 h); the VAE sweep was not timed.

## Citation

```bibtex
@article{boya2026heads,
  title   = {Choosing the Number of Attention Heads from Intrinsic Dimension: A Small-Scale Study},
  author  = {Boya, Arjun and Bandaru, Charan Sai and Pillimetla, Masthanaiah and Chidanandappa, Kavana},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
