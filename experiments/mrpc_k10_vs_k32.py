"""
Multi-Seed Validation: k=10 (VAE elbow) vs k=32 (standard) on MRPC
=====================================================================
Second-dataset generalization check for the SST-2 result.
Same architecture, same paired-seed design, same TOST equivalence test.
MRPC is sentence-PAIR paraphrase detection (GLUE).

Run:
    pip install torch transformers datasets matplotlib scipy
    python mrpc_multiseed_k10_vs_k32.py
"""

import os, sys
# Repo root, so scripts work from any working directory and find vae_core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')

import os, math, json, time, traceback, random
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
    'dropout'     : 0.25,   # was 0.15 -- that setting cured majority-class
                            # collapse but overcorrected: k=10 and k=32 both
                            # hit ~95-97% train acc against ~68-72% val acc
                            # (a ~25pt gap) by epoch 9-11, i.e. severe
                            # overfitting for BOTH configs almost equally.
                            # With a gap that large, a modest val-accuracy
                            # difference between configs is not trustworthy
                            # evidence of a real head-count effect -- it may
                            # just be which random subset each overfit model
                            # happened to memorize well. Raising dropout
                            # partway back (not all the way to 0.3, which
                            # caused collapse) aims for the middle ground:
                            # enough regularization that val accuracy
                            # reflects genuine generalization, not memorization
                            # noise.
    'n_classes'   : 2,

    'head_configs': [
        {'n_heads': 10, 'label': 'k=10 (VAE elbow exact)', 'color': '#4f8ef7'},
        {'n_heads': 32, 'label': 'k=32 (standard)',        'color': '#4caf50'},
    ],

    'seeds': [42, 123, 456, 789, 1011, 2024, 31337, 7, 999, 8675309],

    'batch_size'  : 32,     # was 64 -- MRPC is small (3,668 train examples);
                            # smaller batches mean more gradient updates per
                            # epoch (~115 steps/epoch instead of ~58), giving
                            # the optimizer more chances to escape the
                            # majority-class basin before patience triggers
    'epochs'      : 20,
    'patience'    : 6,
    'lr'          : 2e-4,   # was 1e-4 -- slightly higher peak LR since the
                            # model is training from scratch (BERT tokenizer
                            # only, not pretrained BERT weights) on very
                            # little data; a too-conservative LR can get
                            # stuck at the trivial majority-class solution
    'weight_decay': 3e-2,   # was 1e-2 -- same overcorrection logic as
                            # dropout above; splitting the difference
                            # between the collapse-causing 5e-2 and the
                            # overfit-causing 1e-2

    # NOT a fixed step count. MRPC train is ~3,668 examples -> ~58 steps/epoch
    # -> ~870 total steps at 15 epochs, vs SST-2's ~10,500+. A fixed
    # warmup_steps=500 copied from the SST-2 script would burn ~60-85% of
    # the entire MRPC run just ramping the LR up, starving the model of a
    # real learning-rate plateau before early stopping ever triggers --
    # this is what caused the earlier collapse to majority-class
    # prediction (both k=10 and k=32 stuck at the ~0.684 majority baseline).
    # Scale warmup with the actual step budget instead.
    'warmup_frac' : 0.06,

    'results_path': os.path.join(RESULTS, 'mrpc_multiseed_results.json'),
    'plot_path'   : os.path.join(RESULTS, 'mrpc_multiseed_comparison.png'),

    # MRPC has only 408 validation examples; expect wider CIs.
    # If TOST fails at 0.01, try 0.015 or 0.02 and report transparently.
    'equivalence_margin': 0.01,
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {DEVICE}")
print(f"Seeds  : {CFG['seeds']}  ({len(CFG['seeds'])} per head count, "
      f"{len(CFG['seeds']) * len(CFG['head_configs'])} runs total)")


# ─── 1. Data ─────────────────────────────────────────────────────────────────
def load_mrpc():
    """Load GLUE MRPC and tokenize sentence pairs."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("\n[1] Loading MRPC (GLUE)...")
    
    # Fix for newer datasets/huggingface_hub versions
    try:
        dataset = load_dataset("glue", "mrpc", trust_remote_code=True)
    except Exception:
        dataset = load_dataset("nyu-mll/glue", "mrpc")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        # BERT-style: [CLS] sentence1 [SEP] sentence2 [SEP]
        return tokenizer(
            batch['sentence1'],
            batch['sentence2'],
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
    the global RNG. k=10 and k=32 at the same seed see IDENTICAL batch order."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(tokenized_train, batch_size=CFG['batch_size'],
                      shuffle=True, generator=g, num_workers=0)


def make_val_loader(tokenized_val):
    return DataLoader(tokenized_val, batch_size=CFG['batch_size'],
                      shuffle=False, num_workers=0)


# ─── 2. Architecture ──────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    D_K = 16  # fixed per-head dimension

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


# ─── 4. Single experiment (one k, one seed) ─────────────────────────────────
def run_experiment(n_heads, label, seed, tokenized_train, val_loader):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    warmup_steps = max(10, int(CFG['warmup_frac'] * steps))

    expected_attn = 4 * CFG['d_model'] * n_heads * 16 * CFG['n_layers']
    print(f"\n  [{label} | seed={seed}]")
    print(f"    steps/epoch={len(train_loader)}  total_steps={steps}  "
          f"warmup_steps={warmup_steps} ({warmup_steps/steps:.1%} of run)")
    assert attn_p == expected_attn, \
        f"PARAM MISMATCH: got {attn_p}, expected {expected_attn}"

    opt   = torch.optim.AdamW(model.parameters(),
                               lr=CFG['lr'],
                               weight_decay=CFG['weight_decay'])
    sched = make_scheduler(opt, warmup_steps, steps)

    hist       = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_acc   = -1.0
    no_improve = 0
    patience   = CFG['patience']
    ckpt       = f'best_mrpc_h{n_heads}_s{seed}.pt'
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
    if os.path.exists(ckpt):
        os.remove(ckpt)

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


# ─── 5. Stats ─────────────────────────────────────────────────────────────────
def mean_ci(values, conf=0.95):
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
        equivalent = abs(mean_diff) < margin
        return {'diff_mean': float(mean_diff),
                'p_lower': 0.0 if equivalent else 1.0,
                'p_upper': 0.0 if equivalent else 1.0,
                'p_tost': 0.0 if equivalent else 1.0,
                'equivalent': equivalent, 'margin': margin, 'n': n}

    t_lower = (mean_diff - (-margin)) / se
    p_lower = 1 - sstats.t.cdf(t_lower, df)
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
    print("  MULTI-SEED SUMMARY  (MRPC)")
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
        if len(a10) == len(a32):
            tstat, pval = sstats.ttest_rel(a10, a32)
            d = paired_cohens_d(a10, a32)
            print("\n  ── k=10 vs k=32 (paired across seeds) ──")
            print(f"    mean gap (k32 - k10) : {stats_by_k[32]['mean'] - stats_by_k[10]['mean']:.4f}")
            print(f"    paired t-test        : t={tstat:.3f}, p={pval:.4f}")
            print(f"    Cohen's d (paired)   : {d:.3f}")

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
                           f"(TOST p={tost['p_tost']:.4f})")
            elif pval >= 0.05:
                verdict = ("paired t-test: no significant difference, but "
                           "TOST did NOT confirm equivalence at this margin")
            else:
                verdict = ("statistically significant difference (p < .05)")
            print(f"\n    verdict               : {verdict}")

            stats_by_k['_comparison'] = {
                'paired_t': {'t': float(tstat), 'p': float(pval), 'cohens_d': float(d)},
                'tost': tost,
                'verdict': verdict,
            }
        else:
            print("\n  [!] Unequal seed counts — skipping paired test.")

    print("="*72)
    return stats_by_k


# ─── 6. Plotting ──────────────────────────────────────────────────────────────
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

    fig.suptitle('Multi-Seed Validation on MRPC: k=10 vs k=32',
                color='white', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved -> {save_path}")


# ─── 7. Main ─────────────────────────────────────────────────────────────────
def main():
    tokenized = load_mrpc()
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
                  f"Val Acc={result['best_val_acc']:.4f} | "
                  f"Final Train Acc={result['history']['train_acc'][-1]:.4f}")
        except Exception as e:
            print(f"  [!] FAILED k={cfg['n_heads']} seed={seed}: {e}")
            traceback.print_exc()
            continue
        finally:
            with open(CFG['results_path'], 'w') as f:
                json.dump(all_results, f, indent=2)

    stats_by_k = summarize(all_results)
    plot_results(all_results, stats_by_k, CFG['plot_path'])


if __name__ == '__main__':
    main()