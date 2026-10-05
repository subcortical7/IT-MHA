"""
Multi-Seed Validation: k=10 (VAE elbow) vs k=32 (standard) on SST-2
=====================================================================
Extends sst2_head_comparison.py. Runs BOTH head counts across N seeds
(same seed list for each, so comparisons are paired), then reports:
  - mean +/- std, 95% CI (t-distribution) for best val accuracy per k
  - paired t-test + Cohen's d (paired) on k=10 vs k=32 across seeds
  - per-seed results so you can eyeball outliers before writing them up

Design notes:
  - The train DataLoader is rebuilt fresh inside run_experiment for every
    (k, seed) run, using its own torch.Generator seeded directly with
    `seed` (see make_train_loader()). This is deliberate, not incidental:
    relying on the global RNG (via a bare torch.manual_seed(seed) call)
    would NOT guarantee k=10 and k=32 see the same batch order at "the
    same seed", because Xavier init consumes a different number of random
    draws depending on n_heads -- by the time the loader's RandomSampler
    drew its shuffle permutation, the two arms' global RNG state would
    already have diverged. An explicit, independent generator sidesteps
    that entirely: k=10-seed42 and k=32-seed42 now see identical batch
    order, differing only in n_heads. This is what makes the paired
    t-test in summarize() actually valid.
  - Results are cached to JSON incrementally (after every single run, not
    just at the end), so a crash/timeout partway through a 10-run sweep
    doesn't cost you the runs that already finished. Re-running the script
    skips (n_heads, seed) pairs already present in the cache.
  - No head-redundancy diagnostic. An earlier version of this script
    included a per-head cosine-similarity redundancy hook, but it had two
    separate bugs (flattening across tokens instead of comparing per-token,
    and not applying the padding mask inside the hook's own attention
    recomputation) and, once fixed, wasn't doing narrative work the paper
    needed -- head redundancy was never the mechanism being claimed, the
    accuracy-equivalence result stands on its own. Removed rather than kept
    as a secondary diagnostic to avoid a "known broken, then patched"
    artifact sitting in the paper's evidence trail.

Run:
    pip install torch transformers datasets matplotlib tqdm scipy
    python sst2_multiseed_k10_vs_k32.py
"""

import os, sys
# Repo root, so scripts work from any working directory and find vae_core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')

import os, math, json, time, itertools, traceback, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats as sstats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = {
    'd_model'     : 128,
    'd_ff'        : 512,
    'n_layers'    : 4,
    'max_seq_len' : 128,
    'vocab_size'  : 30522,
    'dropout'     : 0.3,
    'n_classes'   : 2,

    'head_configs': [
        {'n_heads': 10, 'label': 'k=10 (VAE elbow exact)', 'color': '#4f8ef7'},
        {'n_heads': 32, 'label': 'k=32 (standard)',        'color': '#4caf50'},
    ],

    # 5 seeds is the workshop-minimum. Bump to 10 if you want a tighter CI
    # and have the compute budget before Aug 29.
    # 10 seeds for a tighter CI. The original 5 stay first so your existing
    # cached results (sst2_multiseed_results.json) are reused as-is -- only
    # the 10 new (k, seed) combos below will actually train.
    'seeds': [42, 123, 456, 789, 1011, 2024, 31337, 7, 999, 8675309],

    'batch_size'  : 64,
    'epochs'      : 10,
    'patience'    : 3,
    'lr'          : 1e-4,
    'weight_decay': 5e-2,
    'warmup_steps': 500,

    'results_path': os.path.join(RESULTS, 'sst2_multiseed_results.json'),
    'plot_path'   : os.path.join(RESULTS, 'sst2_multiseed_comparison.png'),

    # TOST equivalence margin, in accuracy points. This is a claim you're
    # choosing, not something derived from the data: "I don't care about a
    # true gap smaller than this." 0.01 (1pp) is a reasonable default for
    # SST-2 binary accuracy -- tighten it if you want a stronger claim and
    # have the seed budget to support it (a tighter margin needs a smaller
    # SE, i.e. more seeds, to reach significance).
    'equivalence_margin': 0.01,
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {DEVICE}")
print(f"Seeds  : {CFG['seeds']}  ({len(CFG['seeds'])} per head count, "
      f"{len(CFG['seeds']) * len(CFG['head_configs'])} runs total)")


# ─── 1. Data ─────────────────────────────────────────────────────────────────
def load_sst2():
    """Tokenize once. Returns the tokenized HF datasets, NOT DataLoaders --
    loaders are built fresh per (k, seed) run in run_experiment, see
    make_train_loader() below for why."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("\n[1] Loading SST-2 (stanfordnlp/sst2)...")
    dataset   = load_dataset("stanfordnlp/sst2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch['sentence'],
            truncation=True,
            max_length=CFG['max_seq_len'],
            padding='max_length',
        )

    tokenized = dataset.map(tokenize, batched=True, load_from_cache_file=False)
    tokenized.set_format(type='torch',
                         columns=['input_ids', 'attention_mask', 'label'])

    print(f"  Train: {len(tokenized['train']):,} | "
          f"Val: {len(tokenized['validation']):,}")
    return tokenized


def make_train_loader(tokenized_train, seed):
    """Fresh DataLoader with an EXPLICIT per-seed generator, independent of
    the global RNG.

    Why this matters: Xavier init in TinyTransformer draws a different
    number of random values depending on n_heads (attention weight matrix
    shapes scale with n_heads). If shuffling instead relied on the global
    default generator, then for "the same seed", k=10 and k=32 would each
    call torch.manual_seed(seed) identically, but by the time the loader's
    RandomSampler actually draws its shuffle permutation (after model
    construction has already consumed some number of global RNG draws),
    the global RNG state has diverged between the two arms -- so they'd
    see different batch orders despite sharing a seed. That silently
    breaks the pairing the paired t-test depends on.

    Fix: give the DataLoader its own torch.Generator seeded directly with
    `seed`, so its shuffle sequence is fully decoupled from however many
    draws model init happens to consume. k=10-seed42 and k=32-seed42 now
    see IDENTICAL batch order, differing only in n_heads.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(tokenized_train, batch_size=CFG['batch_size'],
                      shuffle=True, generator=g, num_workers=0)


def make_val_loader(tokenized_val):
    return DataLoader(tokenized_val, batch_size=CFG['batch_size'],
                      shuffle=False, num_workers=0)


# ─── 2. Architecture (unchanged from single-seed script) ────────────────────
class MultiHeadAttention(nn.Module):
    D_K = 16  # fixed per-head dimension, params scale with k

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k     = self.D_K
        self.d_inner = n_heads * self.D_K

        self.W_q = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_k = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_v = nn.Linear(d_model, self.d_inner, bias=False)
        self.W_o = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _  = x.shape
        H, dk    = self.n_heads, self.d_k

        Q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = self.drop(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(B, T, self.d_inner)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.attn(x, mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 d_ff, max_seq_len, n_classes, dropout):
        super().__init__()
        self.emb     = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop    = nn.Dropout(dropout)
        self.layers  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ids, mask=None):
        B, T = ids.shape
        pos  = torch.arange(T, device=ids.device).unsqueeze(0)
        x    = self.drop(self.emb(ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(x[:, 0])

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_attn_params(self):
        return sum(p.numel() for l in self.layers
                   for p in l.attn.parameters())


# ─── 3. Training ─────────────────────────────────────────────────────────────
def make_scheduler(opt, warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_epoch(model, loader, opt, sched):
    model.train()
    loss_sum, correct, total = 0., 0, 0
    for batch in loader:
        ids    = batch['input_ids'].to(DEVICE)
        mask   = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(ids, mask)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        loss_sum += loss.item()
        correct  += (logits.argmax(-1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum, correct, total = 0., 0, 0
    for batch in loader:
        ids    = batch['input_ids'].to(DEVICE)
        mask   = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        logits = model(ids, mask)
        loss   = F.cross_entropy(logits, labels)
        loss_sum += loss.item()
        correct  += (logits.argmax(-1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / len(loader), correct / total


# ─── 4. Single experiment (one k, one seed) ──────────────────────────────────
def run_experiment(n_heads, label, seed, tokenized_train, val_loader):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Built here, AFTER seeding but with its own independent generator, so
    # k=10 and k=32 at this same seed get identical batch order regardless
    # of how many RNG draws model init below consumes. See make_train_loader().
    train_loader = make_train_loader(tokenized_train, seed)

    model = TinyTransformer(
        vocab_size  = CFG['vocab_size'],
        d_model     = CFG['d_model'],
        n_heads     = n_heads,
        n_layers    = CFG['n_layers'],
        d_ff        = CFG['d_ff'],
        max_seq_len = CFG['max_seq_len'],
        n_classes   = CFG['n_classes'],
        dropout     = CFG['dropout'],
    ).to(DEVICE)

    total_p = model.n_params()
    attn_p  = model.n_attn_params()
    steps   = CFG['epochs'] * len(train_loader)

    expected_attn = 4 * CFG['d_model'] * n_heads * 16 * CFG['n_layers']
    print(f"\n  [{label} | seed={seed}]")
    assert attn_p == expected_attn, \
        f"PARAM MISMATCH: got {attn_p}, expected {expected_attn}"

    opt   = torch.optim.AdamW(model.parameters(),
                               lr=CFG['lr'],
                               weight_decay=CFG['weight_decay'])
    sched = make_scheduler(opt, CFG['warmup_steps'], steps)

    hist       = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_acc   = -1.0  # guarantees epoch 1 always saves a checkpoint, even if val_acc is 0
    no_improve = 0
    patience   = CFG['patience']
    ckpt       = f'best_h{n_heads}_s{seed}.pt'
    t0         = time.time()

    for epoch in range(1, CFG['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, sched)
        vl_loss, vl_acc = evaluate(model, val_loader)

        hist['train_loss'].append(tr_loss)
        hist['train_acc'].append(tr_acc)
        hist['val_loss'].append(vl_loss)
        hist['val_acc'].append(vl_acc)

        if vl_acc > best_acc:
            best_acc   = vl_acc
            no_improve = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1

        print(f"    Epoch {epoch:2d}/{CFG['epochs']} | "
              f"Train {tr_acc:.3f} | Val {vl_acc:.3f} | "
              f"Loss {vl_loss:.4f} | patience {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch}.")
            break

    elapsed = time.time() - t0
    try:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                         weights_only=True))
    except Exception as e:
        print(f"    [!] Could not load checkpoint ({e}); "
              f"using final-epoch weights instead of best-epoch weights "
              f"for the returned best_val_acc.")
    if os.path.exists(ckpt):
        os.remove(ckpt)  # checkpoints are large and we only need the metrics

    return {
        'n_heads'        : n_heads,
        'label'          : label,
        'seed'           : seed,
        'total_params'   : total_p,
        'attn_params'    : attn_p,
        'best_val_acc'   : best_acc,
        'final_val_acc'  : hist['val_acc'][-1],
        'train_time_s'   : elapsed,
        'history'        : hist,
    }


# ─── 6. Stats ─────────────────────────────────────────────────────────────────
def mean_ci(values, conf=0.95):
    """Mean, std, and a t-distribution confidence interval for a small sample."""
    arr = np.asarray(values, dtype=float)
    n   = len(arr)
    m   = arr.mean()
    if n < 2:
        return m, 0.0, (m, m)
    sd  = arr.std(ddof=1)
    se  = sd / math.sqrt(n)
    tcrit = sstats.t.ppf((1 + conf) / 2, df=n - 1)
    return m, sd, (m - tcrit * se, m + tcrit * se)


def paired_cohens_d(a, b):
    diffs = np.asarray(a) - np.asarray(b)
    return float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0


def tost_paired(a, b, margin, alpha=0.05):
    """Two One-Sided Tests (TOST) for paired equivalence.

    The paired t-test only tells you what you *failed to reject*: "we
    didn't find a difference." It can't tell you "the difference is
    small" -- a low-power study with noisy data will also fail to reject,
    for the wrong reason. TOST flips the null: instead of H0 = "no
    difference" (which you can only fail to reject), it tests H0 = "the
    true difference is at least as large as +/-margin" against H1 = "the
    true difference is smaller than margin in absolute value". Rejecting
    that H0 is a positive, falsifiable claim of equivalence within the
    margin you chose -- not just an absence of evidence.

    Mechanically this is two one-sided t-tests against the two margin
    boundaries; the overall TOST p-value is the *larger* (weaker) of the
    two one-sided p-values, and equivalence is concluded only if that
    max is still below alpha (i.e. BOTH one-sided tests must pass).

    diff is defined as (a - b), paired by seed/index.
    """
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n     = len(diffs)
    if n < 2:
        return {'diff_mean': float(diffs.mean()) if n else 0.0,
                'p_lower': 1.0, 'p_upper': 1.0, 'p_tost': 1.0,
                'equivalent': False, 'margin': margin, 'n': n}

    mean_diff = diffs.mean()
    se        = diffs.std(ddof=1) / math.sqrt(n)
    df        = n - 1

    if se == 0.0:
        # No variance across seeds at all: equivalence holds iff the
        # (identical) diff is already inside the margin.
        equivalent = abs(mean_diff) < margin
        return {'diff_mean': float(mean_diff), 'p_lower': 0.0 if equivalent else 1.0,
                'p_upper': 0.0 if equivalent else 1.0,
                'p_tost': 0.0 if equivalent else 1.0,
                'equivalent': equivalent, 'margin': margin, 'n': n}

    # H0_lower: true diff <= -margin   vs  H1: true diff > -margin
    t_lower = (mean_diff - (-margin)) / se
    p_lower = 1 - sstats.t.cdf(t_lower, df)

    # H0_upper: true diff >= +margin   vs  H1: true diff < +margin
    t_upper = (mean_diff - margin) / se
    p_upper = sstats.t.cdf(t_upper, df)

    p_tost = max(p_lower, p_upper)
    return {
        'diff_mean' : float(mean_diff),
        'se'        : float(se),
        'p_lower'   : float(p_lower),
        'p_upper'   : float(p_upper),
        'p_tost'    : float(p_tost),
        'equivalent': bool(p_tost < alpha),
        'margin'    : margin,
        'n'         : n,
    }


def summarize(all_results):
    by_k = {}
    for r in all_results:
        by_k.setdefault(r['n_heads'], []).append(r)

    print("\n" + "="*72)
    print("  MULTI-SEED SUMMARY")
    print("="*72)

    stats_by_k = {}
    for k, runs in sorted(by_k.items()):
        accs = [r['best_val_acc'] for r in runs]
        m, sd, ci = mean_ci(accs)
        stats_by_k[k] = {'accs': accs, 'mean': m, 'std': sd, 'ci': ci}
        seeds_str = ", ".join(f"{r['seed']}:{r['best_val_acc']:.4f}" for r in runs)
        print(f"\n  k={k}  (n={len(runs)} seeds)")
        print(f"    per-seed accuracy    : {seeds_str}")
        print(f"    mean±std             : {m:.4f} ± {sd:.4f}")
        print(f"    95% CI               : [{ci[0]:.4f}, {ci[1]:.4f}]")

    if 10 in stats_by_k and 32 in stats_by_k:
        a10 = stats_by_k[10]['accs']
        a32 = stats_by_k[32]['accs']
        # Only valid as a *paired* test if the same seed list was used for
        # both k's and lists are aligned in seed order -- true here since
        # we iterate CFG['seeds'] identically for each k.
        if len(a10) == len(a32):
            tstat, pval = sstats.ttest_rel(a10, a32)
            d = paired_cohens_d(a10, a32)
            print("\n  ── k=10 vs k=32 (paired across seeds) ──")
            print(f"    mean gap (k32 - k10) : {stats_by_k[32]['mean'] - stats_by_k[10]['mean']:.4f}")
            print(f"    paired t-test        : t={tstat:.3f}, p={pval:.4f}")
            print(f"    Cohen's d (paired)   : {d:.3f}")

            # NOTE: a non-significant paired t-test above is absence of
            # evidence for a difference, not evidence of equivalence -- it's
            # equally consistent with "truly equivalent" and "underpowered
            # to detect a real gap". TOST below tests equivalence directly.
            margin = CFG['equivalence_margin']
            tost = tost_paired(a10, a32, margin=margin)
            print(f"\n  ── TOST equivalence test (margin = ±{margin:.3f} acc) ──")
            print(f"    mean diff (k10 - k32) : {tost['diff_mean']:.4f}")
            print(f"    one-sided p (lower)   : {tost['p_lower']:.4f}")
            print(f"    one-sided p (upper)   : {tost['p_upper']:.4f}")
            print(f"    TOST p (max of both)  : {tost['p_tost']:.4f}")
            print(f"    equivalent (p<.05)?   : {tost['equivalent']}")

            if tost['equivalent']:
                verdict = (f"EQUIVALENT within ±{margin:.3f} accuracy "
                           f"(TOST p={tost['p_tost']:.4f}) — positive evidence "
                           f"k=10 matches k=32, not just a failure to find a gap")
            elif pval >= 0.05:
                verdict = ("paired t-test found no significant difference (p >= .05), "
                           "but TOST could NOT confirm equivalence within the chosen "
                           "margin — likely underpowered (n seeds too small) rather "
                           "than evidence against equivalence; consider more seeds "
                           "or a wider margin before claiming equivalence")
            else:
                verdict = ("statistically significant difference (p < .05) — "
                            "re-check whether the gap is practically meaningful")
            print(f"\n    verdict               : {verdict}")

            stats_by_k['_comparison'] = {
                'paired_t': {'t': float(tstat), 'p': float(pval), 'cohens_d': float(d)},
                'tost': tost,
                'verdict': verdict,
            }
        else:
            print("\n  [!] Unequal seed counts for k=10 and k=32 — "
                  "skipping paired test. Re-run missing seeds first.")

    print("="*72)
    return stats_by_k


# ─── 7. Plotting ──────────────────────────────────────────────────────────────
def plot_results(all_results, stats_by_k, save_path):
    BG, PANEL = '#0f1117', '#1a1d27'
    cfgs = {c['n_heads']: c for c in CFG['head_configs']}
    ks   = sorted(k for k in stats_by_k.keys() if isinstance(k, int))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors='#aaa', labelsize=9)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        for s in ['bottom', 'left']: ax.spines[s].set_color('#444')

    # Left: bar with error bars + jittered individual seed points
    ax = axes[0]
    means = [stats_by_k[k]['mean'] for k in ks]
    stds  = [stats_by_k[k]['std']  for k in ks]
    cols  = [cfgs[k]['color'] for k in ks]
    xs    = np.arange(len(ks))
    ax.bar(xs, means, yerr=stds, capsize=6, color=cols, alpha=0.85, width=0.55)
    rng = np.random.default_rng(0)
    for i, k in enumerate(ks):
        accs = stats_by_k[k]['accs']
        jitter = rng.uniform(-0.08, 0.08, size=len(accs))
        ax.scatter(np.full(len(accs), xs[i]) + jitter, accs,
                   color='white', s=28, zorder=5, edgecolors='#000', linewidths=0.4)
    ax.set_xticks(xs); ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel('Best Val Accuracy', color='#aaa', fontsize=9)
    ax.set_title(f'Accuracy Across {len(CFG["seeds"])} Seeds\n(bar = mean ± 1 std, dots = per-seed)',
                color='white', fontsize=11, pad=10)

    comp = stats_by_k.get('_comparison')
    if comp is not None:
        tost = comp['tost']
        tag  = "EQUIVALENT" if tost['equivalent'] else "not confirmed equivalent"
        ax.text(0.5, -0.16,
                f"TOST (±{tost['margin']:.3f}): {tag}, p={tost['p_tost']:.3f}",
                transform=ax.transAxes, ha='center', va='top',
                color='#8fd694' if tost['equivalent'] else '#e0a34f', fontsize=8.5)

    # Right: per-seed validation curves, overlaid
    ax = axes[1]
    for r in all_results:
        col = cfgs[r['n_heads']]['color']
        ep = range(1, len(r['history']['val_acc']) + 1)
        ax.plot(ep, r['history']['val_acc'], color=col, lw=1.3, alpha=0.6)
    handles = [plt.Line2D([0],[0], color=cfgs[k]['color'], lw=2, label=f"k={k}") for k in ks]
    ax.legend(handles=handles, framealpha=0.2, labelcolor='white', fontsize=9)
    ax.set_xlabel('Epoch', color='#aaa', fontsize=9)
    ax.set_ylabel('Val Accuracy', color='#aaa', fontsize=9)
    ax.set_title('Validation Curves, All Seeds', color='white', fontsize=11, pad=10)

    fig.suptitle('Multi-Seed Validation: k=10 (VAE elbow) vs k=32 (standard)',
                color='white', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved -> {save_path}")


# ─── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    tokenized = load_sst2()
    val_loader = make_val_loader(tokenized['validation'])

    all_results = []
    if os.path.exists(CFG['results_path']):
        with open(CFG['results_path']) as f:
            all_results = json.load(f)
        print(f"\nLoaded {len(all_results)} cached run(s) from {CFG['results_path']}")

    done = {(r['n_heads'], r['seed']) for r in all_results}
    todo = [(cfg, seed) for cfg in CFG['head_configs'] for seed in CFG['seeds']
            if (cfg['n_heads'], seed) not in done]

    print(f"\n[2] {len(todo)} run(s) remaining "
          f"({len(done)} already cached)...")

    for cfg, seed in todo:
        try:
            result = run_experiment(cfg['n_heads'], cfg['label'], seed,
                                    tokenized['train'], val_loader)
            all_results.append(result)
            print(f"  DONE k={cfg['n_heads']:2d} seed={seed} | "
                  f"Acc={result['best_val_acc']:.4f}")
        except Exception as e:
            print(f"  [!] FAILED k={cfg['n_heads']} seed={seed}: {e}")
            traceback.print_exc()
            continue
        finally:
            # Save after every run so a crash never loses finished work.
            with open(CFG['results_path'], 'w') as f:
                json.dump(all_results, f, indent=2)

    stats_by_k = summarize(all_results)
    plot_results(all_results, stats_by_k, CFG['plot_path'])


if __name__ == '__main__':
    main()
