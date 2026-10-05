"""
pruning_baseline_sst2.py
=========================
Pruning baseline for the IT-MHA paper (Week 3 item #3): does the
prospective k=10 head count (chosen up-front, trained from scratch) match
or beat a REACTIVE k=10 arrived at by pruning a full 32-head model down to
10 heads via a standard importance score?

If prospective k=10 matches or beats pruned-k=10, that's a stronger
argument for the paper's method than the accuracy-equivalence result
alone -- it shows the prospective choice isn't just "good enough", it's
competitive with a method that gets to look at a trained 32-head model
before deciding what to keep.

Pipeline, per seed:
  1. Train a full k=32 model on SST-2 (identical protocol to
     sst2_multiseed_k10_vs_k32.py's k=32 arm, so this step's accuracy
     should closely track the k=32 numbers already in the paper).
  2. Compute per-head importance via ablation: for each of the 32 heads
     in each of the 4 layers, zero that head's output and measure the
     resulting increase in held-out validation loss. Larger increase =
     more important head. This is the same spirit as head-pruning
     importance scores in prior work (e.g. Michel et al. 2019), using
     forward-pass ablation rather than gradient-based sensitivity, since
     it's simpler to implement correctly under time pressure and is a
     standard baseline choice in its own right.
  3. Per layer, keep the 10 highest-importance heads, discard the other
     22 -- construct an actual smaller model (real parameter reduction,
     not just masking) by slicing out the kept heads' Q/K/V rows and O
     columns.
  4. Fine-tune the pruned model briefly (fewer epochs, lower LR than the
     original training run -- this is the realistic version of a pruning
     baseline; nobody deploys a freshly pruned model with zero recovery
     fine-tuning).
  5. Evaluate final accuracy. Compare against (a) the original unpruned
     k=32 accuracy from this same run, and (b) the prospective k=10
     accuracy already recorded in sst2_multiseed_results.json, using the
     SAME seed list, so the final comparison is paired.

Design choices and why (see also the ask/answer earlier in this thread):
  - SST-2 only, not MRPC. MRPC required three rounds of hyperparameter
    fixes to avoid training collapse/overfitting; adding a pruning +
    fine-tuning procedure on top of that fragility risks another
    multi-day debugging loop this baseline doesn't need to survive.
  - Ablation-based (forward-pass-only) importance, not gradient-based.
    Simpler, fewer failure modes, and a legitimate standard choice.
  - Per-layer pruning (each of the 4 layers independently keeps its own
    top-10 heads), not a single global top-10-of-128 ranking. This
    matches how head pruning is normally done in the literature and
    guarantees every layer still has heads to attend with.

Run:
    pip install torch transformers datasets matplotlib scipy
    python pruning_baseline_sst2.py

Requires sst2_multiseed_results.json (from sst2_multiseed_k10_vs_k32.py)
in the same directory, to pull the existing prospective-k10 and
original-k32 numbers for the final paired comparison.
"""

import os, sys
# Repo root, so scripts work from any working directory and find vae_core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')

import os, math, json, time, copy, traceback, random
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

    'n_heads_full'   : 32,
    'n_heads_pruned' : 10,   # matches the prospective k=10 for direct comparison

    'seeds': [42, 123, 456, 789, 1011, 2024, 31337, 7, 999, 8675309],

    # Stage 1: train the full k=32 model. Identical to the k=32 arm of
    # sst2_multiseed_k10_vs_k32.py, so this run's k=32 accuracy should
    # closely track what's already reported in the paper.
    'batch_size'   : 64,
    'epochs'       : 10,
    'patience'     : 3,
    'lr'           : 1e-4,
    'weight_decay' : 5e-2,
    'warmup_steps' : 500,

    # Stage 2: fine-tune AFTER pruning. Fewer epochs, lower LR -- this is
    # meant to be a realistic "recover some of what pruning cost you"
    # phase, not a full second training run from scratch.
    'finetune_epochs'      : 5,
    'finetune_patience'    : 2,
    'finetune_lr'          : 2e-5,   # 1/5 of the original LR
    'finetune_weight_decay': 5e-2,
    'finetune_warmup_frac' : 0.05,

    'sst2_results_path'   : os.path.join(RESULTS, 'sst2_multiseed_results.json'),   # existing k=10/k=32 numbers
    'results_path'        : os.path.join(RESULTS, 'pruning_baseline_results.json'),
    'plot_path'            : os.path.join(RESULTS, 'pruning_baseline_comparison.png'),

    'equivalence_margin': 0.01,
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {DEVICE}")
print(f"Seeds  : {CFG['seeds']}  ({len(CFG['seeds'])} runs)")


# ─── 1. Data (identical to sst2_multiseed_k10_vs_k32.py) ────────────────────
def load_sst2():
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
    """Same explicit-generator pattern as sst2_multiseed_k10_vs_k32.py --
    see that script's docstring for why this matters for reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(tokenized_train, batch_size=CFG['batch_size'],
                      shuffle=True, generator=g, num_workers=0)


def make_val_loader(tokenized_val):
    return DataLoader(tokenized_val, batch_size=CFG['batch_size'],
                      shuffle=False, num_workers=0)


# ─── 2. Architecture ──────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    D_K = 16  # fixed per-head dimension, params scale with n_heads

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

        # Ablation hook: when set to a 0/1 tensor of length n_heads, zeros
        # the corresponding heads' contribution before the output
        # projection. None (default) means no ablation -- normal forward
        # pass, used for both regular training/eval AND as the baseline
        # (unablated) loss inside compute_head_importance().
        self.head_mask = None

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

        out = torch.matmul(attn, V)  # (B, H, T, dk)

        if self.head_mask is not None:
            out = out * self.head_mask.to(out.device).view(1, -1, 1, 1)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_inner)
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


# ─── 3. Training / eval ───────────────────────────────────────────────────────
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


def run_training(model, train_loader, val_loader, epochs, patience,
                  lr, weight_decay, warmup_steps, ckpt_path, tag):
    """Shared training loop used for both the initial k=32 training and
    the post-prune fine-tuning stage. Returns (history, best_val_acc)."""
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * len(train_loader)
    sched = make_scheduler(opt, warmup_steps, steps)

    hist       = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc   = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, sched)
        vl_loss, vl_acc = evaluate(model, val_loader)

        hist['train_loss'].append(tr_loss)
        hist['train_acc'].append(tr_acc)
        hist['val_loss'].append(vl_loss)
        hist['val_acc'].append(vl_acc)

        if vl_acc > best_acc:
            best_acc, no_improve = vl_acc, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        print(f"    [{tag}] Epoch {epoch:2d}/{epochs} | "
              f"Train {tr_acc:.3f} | Val {vl_acc:.3f} | "
              f"Loss {vl_loss:.4f} | patience {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"    [{tag}] Early stop at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE,
                                      weights_only=True))
    return hist, best_acc


# ─── 4. Head importance (ablation-based) and pruning ─────────────────────────
@torch.no_grad()
def compute_head_importance(model, val_loader, n_layers, n_heads_full):
    """For each (layer, head), zero that head's contribution and measure
    the increase in held-out val loss relative to the unablated baseline.
    Larger increase = more important head (removing it hurts more)."""
    model.eval()
    baseline_loss, _ = evaluate(model, val_loader)

    importance = np.zeros((n_layers, n_heads_full))
    for l in range(n_layers):
        attn = model.layers[l].attn
        for h in range(n_heads_full):
            mask = torch.ones(n_heads_full)
            mask[h] = 0.0
            attn.head_mask = mask
            ablated_loss, _ = evaluate(model, val_loader)
            importance[l, h] = ablated_loss - baseline_loss
        attn.head_mask = None  # reset before moving to next layer

    return importance, baseline_loss


def build_pruned_attention(orig_attn, selected_heads, d_model, dropout):
    """Construct a smaller MultiHeadAttention with only the selected
    heads' weights, sliced out of the original (real parameter reduction,
    not masking)."""
    m  = len(selected_heads)
    dk = orig_attn.d_k
    new_attn = MultiHeadAttention(d_model, m, dropout)

    with torch.no_grad():
        for name in ['W_q', 'W_k', 'W_v']:
            orig_w = getattr(orig_attn, name).weight.data  # (H*dk, d_model)
            rows = [orig_w[h * dk:(h + 1) * dk, :] for h in selected_heads]
            getattr(new_attn, name).weight.data.copy_(torch.cat(rows, dim=0))

        orig_wo = orig_attn.W_o.weight.data  # (d_model, H*dk)
        cols = [orig_wo[:, h * dk:(h + 1) * dk] for h in selected_heads]
        new_attn.W_o.weight.data.copy_(torch.cat(cols, dim=1))

    return new_attn


def build_pruned_model(orig_model, selected_heads_per_layer, cfg):
    """Build a full TinyTransformer with n_heads=len(selected_heads per
    layer) (assumed equal across layers), copying non-attention weights
    directly and attention weights via build_pruned_attention."""
    n_heads_pruned = len(selected_heads_per_layer[0])
    pruned = TinyTransformer(
        vocab_size  = cfg['vocab_size'],
        d_model     = cfg['d_model'],
        n_heads     = n_heads_pruned,
        n_layers    = cfg['n_layers'],
        d_ff        = cfg['d_ff'],
        max_seq_len = cfg['max_seq_len'],
        n_classes   = cfg['n_classes'],
        dropout     = cfg['dropout'],
    ).to(DEVICE)

    with torch.no_grad():
        pruned.emb.weight.data.copy_(orig_model.emb.weight.data)
        pruned.pos_emb.weight.data.copy_(orig_model.pos_emb.weight.data)
        pruned.head.load_state_dict(orig_model.head.state_dict())

        for l in range(cfg['n_layers']):
            orig_block  = orig_model.layers[l]
            pruned_block = pruned.layers[l]

            pruned_attn = build_pruned_attention(
                orig_block.attn, selected_heads_per_layer[l],
                cfg['d_model'], cfg['dropout'])
            pruned_block.attn.load_state_dict(pruned_attn.state_dict())

            pruned_block.ff.load_state_dict(orig_block.ff.state_dict())
            pruned_block.norm1.load_state_dict(orig_block.norm1.state_dict())
            pruned_block.norm2.load_state_dict(orig_block.norm2.state_dict())

    return pruned


# ─── 5. Single seed's full pipeline ───────────────────────────────────────────
def run_pruning_pipeline(seed, tokenized_train, val_loader):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"\n{'='*70}\n  SEED {seed}\n{'='*70}")

    # ── Stage 1: train the full k=32 model ──────────────────────────────
    train_loader = make_train_loader(tokenized_train, seed)
    full_model = TinyTransformer(
        vocab_size=CFG['vocab_size'], d_model=CFG['d_model'],
        n_heads=CFG['n_heads_full'], n_layers=CFG['n_layers'],
        d_ff=CFG['d_ff'], max_seq_len=CFG['max_seq_len'],
        n_classes=CFG['n_classes'], dropout=CFG['dropout'],
    ).to(DEVICE)

    t0 = time.time()
    ckpt_full = f'pruning_full_s{seed}.pt'
    hist_full, full_val_acc = run_training(
        full_model, train_loader, val_loader,
        epochs=CFG['epochs'], patience=CFG['patience'],
        lr=CFG['lr'], weight_decay=CFG['weight_decay'],
        warmup_steps=CFG['warmup_steps'], ckpt_path=ckpt_full, tag='k=32 full')
    train_time_full = time.time() - t0
    print(f"  Full k=32 model: best_val_acc={full_val_acc:.4f}")

    # ── Stage 2: ablation-based head importance ─────────────────────────
    t0 = time.time()
    importance, baseline_loss = compute_head_importance(
        full_model, val_loader, CFG['n_layers'], CFG['n_heads_full'])
    importance_time = time.time() - t0

    selected_heads_per_layer = []
    for l in range(CFG['n_layers']):
        ranked = np.argsort(-importance[l])  # descending: most important first
        kept = sorted(ranked[:CFG['n_heads_pruned']].tolist())
        selected_heads_per_layer.append(kept)
        print(f"  Layer {l}: kept heads {kept}")

    # ── Stage 3: build the pruned model, check its pre-finetune accuracy ─
    pruned_model = build_pruned_model(full_model, selected_heads_per_layer, CFG)
    pre_ft_loss, pre_ft_acc = evaluate(pruned_model, val_loader)
    print(f"  Pruned model (pre-finetune): val_acc={pre_ft_acc:.4f} "
          f"(vs full k=32: {full_val_acc:.4f})")

    # ── Stage 4: fine-tune the pruned model ──────────────────────────────
    ft_train_loader = make_train_loader(tokenized_train, seed)
    ckpt_pruned = f'pruning_pruned_s{seed}.pt'
    ft_warmup = max(10, int(CFG['finetune_warmup_frac']
                              * CFG['finetune_epochs'] * len(ft_train_loader)))
    t0 = time.time()
    hist_ft, pruned_val_acc = run_training(
        pruned_model, ft_train_loader, val_loader,
        epochs=CFG['finetune_epochs'], patience=CFG['finetune_patience'],
        lr=CFG['finetune_lr'], weight_decay=CFG['finetune_weight_decay'],
        warmup_steps=ft_warmup, ckpt_path=ckpt_pruned, tag='pruned fine-tune')
    finetune_time = time.time() - t0
    print(f"  Pruned model (post-finetune): val_acc={pruned_val_acc:.4f}")

    for p in [ckpt_full, ckpt_pruned]:
        if os.path.exists(p):
            os.remove(p)

    return {
        'seed'                  : seed,
        'full_k32_val_acc'      : full_val_acc,
        'pruned_pre_finetune_acc': pre_ft_acc,
        'pruned_post_finetune_acc': pruned_val_acc,
        'selected_heads_per_layer': selected_heads_per_layer,
        'baseline_loss_before_ablation': baseline_loss,
        'train_time_full_s'     : train_time_full,
        'importance_time_s'     : importance_time,
        'finetune_time_s'       : finetune_time,
        'history_full'          : hist_full,
        'history_finetune'      : hist_ft,
    }


# ─── 6. Stats (identical helpers to sst2_multiseed_k10_vs_k32.py) ───────────
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
        'diff_mean': float(mean_diff), 'se': float(se),
        'p_lower': float(p_lower), 'p_upper': float(p_upper),
        'p_tost': float(p_tost), 'equivalent': bool(p_tost < alpha),
        'margin': margin, 'n': n,
    }


def load_prospective_k10(seeds):
    """Pull the already-trained prospective k=10 (from-scratch) accuracies
    from sst2_multiseed_results.json, for the SAME seeds, so the final
    prospective-vs-pruned comparison is properly paired."""
    if not os.path.exists(CFG['sst2_results_path']):
        print(f"  [!] {CFG['sst2_results_path']} not found -- cannot run "
              f"the prospective-vs-pruned comparison. Copy it into this "
              f"directory to enable it.")
        return None
    with open(CFG['sst2_results_path']) as f:
        existing = json.load(f)
    by_seed = {r['seed']: r['best_val_acc']
               for r in existing if r['n_heads'] == 10}
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        print(f"  [!] Prospective k=10 results missing for seeds {missing} "
              f"in {CFG['sst2_results_path']} -- comparison will be skipped "
              f"or partial.")
        return None
    return [by_seed[s] for s in seeds]


def summarize(all_results):
    seeds = [r['seed'] for r in all_results]
    full_accs   = [r['full_k32_val_acc'] for r in all_results]
    pruned_accs = [r['pruned_post_finetune_acc'] for r in all_results]
    pre_ft_accs = [r['pruned_pre_finetune_acc'] for r in all_results]

    print("\n" + "=" * 72)
    print("  PRUNING BASELINE SUMMARY (SST-2)")
    print("=" * 72)

    for label, accs in [('Full k=32 (this run)', full_accs),
                         ('Pruned k=10, pre-finetune', pre_ft_accs),
                         ('Pruned k=10, post-finetune', pruned_accs)]:
        m, sd, ci = mean_ci(accs)
        print(f"\n  {label}  (n={len(accs)} seeds)")
        print(f"    mean±std : {m:.4f} ± {sd:.4f}")
        print(f"    95% CI   : [{ci[0]:.4f}, {ci[1]:.4f}]")

    prospective = load_prospective_k10(seeds)
    comparison = None
    if prospective is not None:
        m_pro, sd_pro, ci_pro = mean_ci(prospective)
        m_pru, sd_pru, ci_pru = mean_ci(pruned_accs)
        print(f"\n  Prospective k=10 (from sst2_multiseed_results.json, matched seeds)")
        print(f"    mean±std : {m_pro:.4f} ± {sd_pro:.4f}")
        print(f"    95% CI   : [{ci_pro[0]:.4f}, {ci_pro[1]:.4f}]")

        t, p = sstats.ttest_rel(prospective, pruned_accs)
        d = paired_cohens_d(prospective, pruned_accs)
        margin = CFG['equivalence_margin']
        tost = tost_paired(prospective, pruned_accs, margin=margin)

        print(f"\n  ── Prospective k=10 vs Pruned k=10 (paired, n={len(seeds)}) ──")
        print(f"    mean gap (pruned - prospective) : {m_pru - m_pro:.4f}")
        print(f"    paired t-test                   : t={t:.3f}, p={p:.4f}")
        print(f"    Cohen's d (paired)               : {d:.3f}")
        print(f"    TOST (±{margin:.3f})                    : "
              f"p={tost['p_tost']:.4f}, equivalent={tost['equivalent']}")

        if m_pro >= m_pru:
            verdict = ("Prospective k=10 matches or EXCEEDS pruned k=10 -- "
                        "supports choosing head count prospectively rather "
                        "than pruning reactively.")
        elif tost['equivalent']:
            verdict = ("Prospective and pruned k=10 are statistically "
                        "equivalent -- prospective selection costs nothing "
                        "relative to reactive pruning, without requiring a "
                        "trained 32-head model first.")
        else:
            verdict = ("Pruned k=10 significantly exceeds prospective k=10 -- "
                        "report honestly; this would temper the paper's "
                        "practical recommendation.")
        print(f"\n    verdict: {verdict}")

        comparison = {
            'prospective_mean': m_pro, 'pruned_mean': m_pru,
            'paired_t': {'t': float(t), 'p': float(p), 'cohens_d': float(d)},
            'tost': tost, 'verdict': verdict,
        }

    print("=" * 72)
    return {
        'full_accs': full_accs, 'pre_ft_accs': pre_ft_accs,
        'pruned_accs': pruned_accs, 'prospective_accs': prospective,
        'comparison': comparison,
    }


def plot_results(summary, save_path):
    BG, PANEL = '#0f1117', '#1a1d27'
    groups = [('Full\nk=32', summary['full_accs'], '#4caf50')]
    if summary['prospective_accs'] is not None:
        groups.append(('Prospective\nk=10', summary['prospective_accs'], '#4f8ef7'))
    groups.append(('Pruned\nk=10', summary['pruned_accs'], '#e0a34f'))

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#aaa', labelsize=9)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color('#444')

    means = [np.mean(g[1]) for g in groups]
    stds  = [np.std(g[1], ddof=1) for g in groups]
    cols  = [g[2] for g in groups]
    xs    = np.arange(len(groups))
    ax.bar(xs, means, yerr=stds, capsize=6, color=cols, alpha=0.85, width=0.55)

    rng = np.random.default_rng(0)
    for i, (label, accs, _) in enumerate(groups):
        jitter = rng.uniform(-0.08, 0.08, size=len(accs))
        ax.scatter(np.full(len(accs), xs[i]) + jitter, accs,
                   color='white', s=28, zorder=5, edgecolors='#000', linewidths=0.4)

    ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel('Best Val Accuracy', color='#aaa', fontsize=9)
    ax.set_title('Pruning Baseline: SST-2\n(bar = mean ± 1 std, dots = per-seed)',
                color='white', fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    print(f"Plot saved -> {save_path}")


# ─── 7. Main ─────────────────────────────────────────────────────────────────
def main():
    tokenized = load_sst2()
    val_loader = make_val_loader(tokenized['validation'])

    all_results = []
    if os.path.exists(CFG['results_path']):
        with open(CFG['results_path']) as f:
            all_results = json.load(f)
        print(f"\nLoaded {len(all_results)} cached run(s) from {CFG['results_path']}")

    done = {r['seed'] for r in all_results}
    todo = [s for s in CFG['seeds'] if s not in done]
    print(f"\n[2] {len(todo)} seed(s) remaining ({len(done)} already cached)...")

    for seed in todo:
        try:
            result = run_pruning_pipeline(seed, tokenized['train'], val_loader)
            all_results.append(result)
        except Exception as e:
            print(f"  [!] FAILED seed={seed}: {e}")
            traceback.print_exc()
            continue
        finally:
            with open(CFG['results_path'], 'w') as f:
                json.dump(all_results, f, indent=2)

    summary = summarize(all_results)
    plot_results(summary, CFG['plot_path'])


if __name__ == '__main__':
    main()
