"""
Ordered-Variance VAE Experiment
================================
Hypothesis: A VAE trained with nested dropout on token embeddings will produce
a latent variance spectrum with a clear elbow — a natural point where variance
collapses — suggesting an intrinsic dimensionality that can dictate attention heads.

Dataset: WikiText-2 (standard, small, clean prose)
Embeddings: DistilBERT token embeddings (fast, 768-dim, pre-trained)

Run:
    pip install torch transformers datasets matplotlib tqdm scipy
    python extract_embeddings_and_vae.py
"""

import os
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, 'results')

import os
import math
import json
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = {
    # Data
    'max_tokens'    : 60_000,   # how many token embeddings to extract
    'seq_max_len'   : 128,      # max tokens per sentence for DistilBERT
    'cache_path'    : os.path.join(ROOT, 'token_embeddings.pt'),

    # VAE architecture
    'd_input'       : 768,      # DistilBERT hidden dim
    'd_hidden'      : 512,
    'd_latent'      : 64,       # number of ordered latent dims to learn

    # Training
    'batch_size'    : 256,
    'epochs'        : 40,
    'lr'            : 1e-3,
    'weight_decay'  : 1e-4,
    'beta_start'    : 0.0,      # KL annealing: start
    'beta_end'      : 0.5,      # KL annealing: end (reach by epoch 20)
    'seeds'         : [42, 123, 7, 2024, 99],  # 5 seeds for a spread check on elbow_k

    # Output
    'plot_path'     : os.path.join(RESULTS, 'variance_spectrum_soft.png'),
    'model_path'    : os.path.join(ROOT, 'ordered_vae_soft.pt'),
    'results_path'  : os.path.join(RESULTS, 'ordered_vae_multiseed_results.json'),
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


# ─── 1. Extract Token Embeddings ─────────────────────────────────────────────
def extract_embeddings():
    """
    Use DistilBERT to extract contextualized token embeddings from WikiText-2.
    These are richer than static word vectors — each token embedding reflects
    its context, making the VAE task harder and results more meaningful.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    print("\n[1/4] Loading WikiText-2 and DistilBERT...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model     = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    model.eval()

    all_embeddings = []
    total_tokens   = 0

    with torch.no_grad():
        for item in tqdm(dataset, desc="Extracting token embeddings"):
            text = item['text'].strip()
            if len(text) < 20:
                continue

            inputs = tokenizer(
                text,
                return_tensors    = 'pt',
                truncation        = True,
                max_length        = CFG['seq_max_len'],
                padding           = False,
            ).to(DEVICE)

            outputs   = model(**inputs)
            token_emb = outputs.last_hidden_state[0]  # (seq_len, 768)
            # Drop [CLS] and [SEP] tokens
            token_emb = token_emb[1:-1]

            all_embeddings.append(token_emb.cpu().float())
            total_tokens += token_emb.shape[0]

            if total_tokens >= CFG['max_tokens']:
                break

    embeddings = torch.cat(all_embeddings, dim=0)[:CFG['max_tokens']]
    print(f"  Extracted: {embeddings.shape[0]:,} embeddings of dim {embeddings.shape[1]}")
    return embeddings


# ─── 2. VAE with Nested Dropout ──────────────────────────────────────────────
class OrderedVAE(nn.Module):
    """
    VAE where nested dropout forces ordered latent dimensions.

    Nested Dropout (Rippel et al. 2014):
        At each training step, sample k ~ Uniform(1, D_LATENT).
        Zero out dimensions k+1 ... D_LATENT.
        This forces dim 1 to be most informative (it's always active),
        dim 2 to be second most, etc.

    Result: latent dimensions are ranked by importance.
    The variance of each dimension after training reflects its information content.
    """

    def __init__(self, d_input, d_hidden, d_latent):
        super().__init__()
        self.d_latent = d_latent

        # Encoder: input → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
        )
        self.fc_mu     = nn.Linear(d_hidden // 2, d_latent)
        self.fc_logvar = nn.Linear(d_hidden // 2, d_latent)

        # Decoder: z → reconstructed embedding
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_input),
        )

    def encode(self, x):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-4, 4)  # numerical stability
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu  # deterministic at inference

    def nested_dropout(self, z):
        """
        The key mechanism for ordered dimensions.
        Sample k ~ Uniform(1, D), keep only first k dims active.
        Over many steps, dim 1 is always used, dim D rarely used.
        Gradient pressure forces the network to pack info into early dims.
        """
        if not self.training:
            return z
        k    = torch.randint(1, self.d_latent + 1, (1,)).item()
        mask = torch.zeros_like(z)
        mask[:, :k] = 1.0
        return z * mask

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        z          = self.nested_dropout(z)
        x_recon    = self.decoder(z)
        return x_recon, mu, logvar


# ─── 3. Loss Function ────────────────────────────────────────────────────────
def vae_loss(x_recon, x, mu, logvar, beta):
    """
    Standard β-VAE loss with KL annealing.
    β gradually increases to prevent posterior collapse (all dims → prior).
    """
    recon = F.mse_loss(x_recon, x, reduction='mean')
    # KL divergence: -0.5 * mean(1 + logvar - mu² - exp(logvar))
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.item(), kl.item()


# ─── 4. Training ─────────────────────────────────────────────────────────────
def get_beta(epoch, cfg):
    """Linear KL annealing: ramp from beta_start to beta_end over first half of training."""
    warmup = cfg['epochs'] // 2
    if epoch <= warmup:
        return cfg['beta_start'] + (cfg['beta_end'] - cfg['beta_start']) * (epoch / warmup)
    return cfg['beta_end']


def train_epoch(model, loader, optimizer, beta):
    model.train()
    total, recon_sum, kl_sum = 0.0, 0.0, 0.0
    for (x,) in loader:
        x = x.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        x_recon, mu, logvar = model(x)
        loss, recon, kl     = vae_loss(x_recon, x, mu, logvar, beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total     += loss.item()
        recon_sum += recon
        kl_sum    += kl
    n = len(loader)
    return total/n, recon_sum/n, kl_sum/n


# ─── 5. Variance Analysis ────────────────────────────────────────────────────
@torch.no_grad()
def compute_latent_variance(model, loader):
    """
    Pass all data through encoder, collect mu vectors.
    Variance of mu across the dataset per dimension = 
    how much information that dimension carries.
    Low variance → dimension collapsed to prior → unused → prunable.
    """
    model.eval()
    all_mu = []
    for (x,) in loader:
        mu, _ = model.encode(x.to(DEVICE))
        all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu, dim=0)  # (N, D_LATENT)
    return all_mu.var(dim=0).numpy()   # (D_LATENT,)


def detect_elbow(variances):
    """
    Elbow detection via maximum deviation from the diagonal line
    connecting the first and last point of the sorted variance curve.
    This is the 'kneedle' method — finds the point of maximum curvature.
    """
    sorted_var = np.sort(variances)[::-1]
    n          = len(sorted_var)
    # Normalize to [0, 1]
    norm = (sorted_var - sorted_var[-1]) / (sorted_var[0] - sorted_var[-1] + 1e-10)
    # Line from (0,1) to (n-1,0)
    line = np.linspace(1.0, 0.0, n)
    # Variance spectra are convex-decreasing (slow drop, then a steep collapse),
    # so the curve sits BELOW the diagonal in the interior, not above it.
    # The elbow is therefore the point of max distance BELOW the line
    # (line - norm), not above it (norm - line) -- the latter only works for
    # concave-decreasing curves and degenerates to picking an endpoint (k=1 or
    # k=n) via floating-point noise for a convex curve like this one.
    diff   = line - norm
    elbow  = int(np.argmax(diff)) + 1  # 1-indexed
    return elbow, sorted_var


# ─── 6. Plotting ─────────────────────────────────────────────────────────────
def plot_variance_spectrum(variances, elbow_k, history, save_path):
    sorted_var = np.sort(variances)[::-1]
    cumvar     = np.cumsum(sorted_var) / sorted_var.sum() * 100
    dims       = np.arange(1, len(sorted_var) + 1)

    # Dark theme
    BG, PANEL = '#0f1117', '#1a1d27'
    BLUE, RED, GOLD, GREEN = '#4f8ef7', '#ff6b6b', '#ffd700', '#4caf50'

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])]

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#444')

    # ── Plot 1: Variance spectrum (bar) ──────────────────────────────────────
    ax = axes[0]
    ax.bar(dims, sorted_var, color=BLUE, alpha=0.6, width=0.85, label='Variance')
    ax.axvline(x=elbow_k, color=RED, linestyle='--', linewidth=2,
               label=f'Elbow @ k={elbow_k}')
    ax.fill_between([0.5, elbow_k + 0.5], 0, sorted_var.max(),
                    alpha=0.08, color=GREEN, label=f'Active ({elbow_k} heads)')
    ax.fill_between([elbow_k + 0.5, len(sorted_var) + 0.5], 0, sorted_var.max(),
                    alpha=0.08, color=RED, label='Prunable')
    ax.set_xlabel('Latent Dimension (ranked by variance)', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Variance', color='#aaaaaa', fontsize=9)
    ax.set_title('Ordered Variance Spectrum', color='white', fontsize=11, pad=10)
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=8)

    # ── Plot 2: Cumulative explained variance ─────────────────────────────────
    ax = axes[1]
    ax.plot(dims, cumvar, color=BLUE, linewidth=2, label='Cumulative variance')
    ax.axvline(x=elbow_k, color=RED, linestyle='--', linewidth=2,
               label=f'k={elbow_k}')
    ax.axhline(y=cumvar[elbow_k - 1], color=GOLD, linestyle=':', linewidth=1.5,
               label=f'{cumvar[elbow_k - 1]:.1f}% explained')
    ax.fill_between(dims[:elbow_k], 0, cumvar[:elbow_k],
                    alpha=0.15, color=GREEN)
    ax.set_xlabel('k (number of dims used)', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Variance Explained (%)', color='#aaaaaa', fontsize=9)
    ax.set_title('Cumulative Explained Variance', color='white', fontsize=11, pad=10)
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=8)
    ax.set_ylim(0, 105)

    # ── Plot 3: Training loss curves ─────────────────────────────────────────
    ax     = axes[2]
    epochs = range(1, len(history['loss']) + 1)
    ax.plot(epochs, history['loss'],  color=BLUE, linewidth=2, label='Total')
    ax.plot(epochs, history['recon'], color=GOLD, linewidth=2, label='Recon')
    ax.plot(epochs, history['kl'],    color=RED,  linewidth=2, label='KL')
    ax.set_xlabel('Epoch', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Loss', color='#aaaaaa', fontsize=9)
    ax.set_title('Training Loss Curves', color='white', fontsize=11, pad=10)
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=8)

    # ── Plot 4: Variance per dim (log scale + elbow annotation) ──────────────
    ax = axes[3]
    ax.semilogy(dims, sorted_var, color=BLUE, linewidth=2, marker='o',
                markersize=3, alpha=0.8)
    ax.axvline(x=elbow_k, color=RED, linestyle='--', linewidth=2)
    ax.annotate(
        f'Elbow: k={elbow_k}\n{cumvar[elbow_k-1]:.1f}% variance\n'
        f'{(1 - elbow_k/len(dims))*100:.0f}% head reduction',
        xy=(elbow_k, sorted_var[elbow_k - 1]),
        xytext=(elbow_k + len(dims) * 0.1, sorted_var[0] * 0.3),
        color='white', fontsize=9,
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#333', alpha=0.8)
    )
    ax.set_xlabel('Latent Dimension', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Variance (log scale)', color='#aaaaaa', fontsize=9)
    ax.set_title('Log Variance + Elbow Detection', color='white', fontsize=11, pad=10)

    fig.suptitle(
        'Ordered-Variance VAE: Hypothesis Test\n'
        'Does token embedding space have intrinsic dimensionality?',
        color='white', fontsize=13, fontweight='bold', y=1.01
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"\nPlot saved → {save_path}")


# ─── 7. Single-seed run ──────────────────────────────────────────────────────
def run_single_seed(embeddings, seed):
    """
    Train one OrderedVAE with the given seed and return its elbow_k plus
    everything needed for a full-detail plot of this run.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset    = TensorDataset(embeddings.float())
    n_train    = int(0.9 * len(dataset))
    n_val      = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'],
                              shuffle=False, num_workers=0, pin_memory=False)

    model = OrderedVAE(
        d_input  = CFG['d_input'],
        d_hidden = CFG['d_hidden'],
        d_latent = CFG['d_latent'],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG['lr'],
        weight_decay = CFG['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs']
    )

    history = {'loss': [], 'recon': [], 'kl': [], 'active_dims': []}

    for epoch in range(1, CFG['epochs'] + 1):
        beta = get_beta(epoch, CFG)
        loss, recon, kl = train_epoch(model, train_loader, optimizer, beta)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            var = compute_latent_variance(model, val_loader)
            active = int((var > 0.05).sum())
            history['active_dims'].append(active)
            print(f"  Epoch {epoch:3d}/{CFG['epochs']} | "
                  f"β={beta:.2f} | Loss={loss:.4f} | "
                  f"Recon={recon:.4f} | KL={kl:.4f} | "
                  f"Active dims (var>0.05): {active}/{CFG['d_latent']}")
        else:
            history['active_dims'].append(history['active_dims'][-1]
                                          if history['active_dims'] else 0)

        history['loss'].append(loss)
        history['recon'].append(recon)
        history['kl'].append(kl)

    variances = compute_latent_variance(model, val_loader)
    elbow_k, sorted_var = detect_elbow(variances)
    cumvar = np.cumsum(sorted_var) / sorted_var.sum() * 100
    active_final = history['active_dims'][-1]

    result = {
        'seed'            : seed,
        'elbow_k'         : elbow_k,
        'active_dims'     : active_final,
        'variance_top'    : float(sorted_var[0]),
        'variance_bottom' : float(sorted_var[-1]),
        'variance_pct_at_elbow': float(cumvar[elbow_k - 1]),
    }
    return result, model, variances, elbow_k, history


# ─── 8. Multi-seed driver ────────────────────────────────────────────────────
def mean_ci(values, confidence=0.95):
    """95% CI via t-distribution — appropriate for small seed counts."""
    arr = np.array(values, dtype=float)
    n   = len(arr)
    m   = arr.mean()
    if n < 2:
        return m, m, m
    se  = arr.std(ddof=1) / math.sqrt(n)
    h   = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - h, m + h


def main():
    # ── Load or extract embeddings (shared across all seeds) ──────────────────
    if os.path.exists(CFG['cache_path']):
        print(f"\n[1/4] Loading cached embeddings from {CFG['cache_path']}...")
        embeddings = torch.load(CFG['cache_path'], map_location='cpu')
    else:
        embeddings = extract_embeddings()
        torch.save(embeddings, CFG['cache_path'])
        print(f"  Cached to {CFG['cache_path']}")

    mean       = embeddings.mean(0, keepdim=True)
    std        = embeddings.std(0, keepdim=True).clamp(min=1e-8)
    embeddings = (embeddings - mean) / std
    print(f"\n[2/4] {embeddings.shape[0]:,} embeddings ready, "
          f"{len(CFG['seeds'])} seeds queued: {CFG['seeds']}")

    # ── Resume from cache if a previous run was interrupted ────────────────────
    results = []
    if os.path.exists(CFG['results_path']):
        with open(CFG['results_path']) as f:
            results = json.load(f)
        print(f"  Resuming: {len(results)} seed(s) already done.")
    done_seeds = {r['seed'] for r in results}

    best_model, best_variances, best_elbow, best_history = None, None, None, None
    best_seed_dist = float('inf')  # pick the seed closest to the eventual median for the detail plot

    print("\n[3/4] Training across seeds...")
    for seed in CFG['seeds']:
        if seed in done_seeds:
            print(f"  Seed {seed}: cached, skipping.")
            continue
        print(f"\n  --- Seed {seed} ---")
        result, model, variances, elbow_k, history = run_single_seed(embeddings, seed)
        results.append(result)
        with open(CFG['results_path'], 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Seed {seed}: elbow_k={elbow_k}, active_dims={result['active_dims']}")

        # Keep the most recent run around for the detail plot; refined below.
        best_model, best_variances, best_elbow, best_history = model, variances, elbow_k, history

    # ── Stats across seeds ──────────────────────────────────────────────────────
    elbow_ks = [r['elbow_k'] for r in results]
    active_dims = [r['active_dims'] for r in results]
    m_elbow, lo_elbow, hi_elbow = mean_ci(elbow_ks)
    m_active, lo_active, hi_active = mean_ci(active_dims)

    print("\n" + "="*55)
    print("  MULTI-SEED RESULT SUMMARY")
    print("="*55)
    print(f"  Seeds run                : {len(results)}")
    print(f"  elbow_k per seed         : {elbow_ks}")
    print(f"  elbow_k mean [95% CI]    : {m_elbow:.1f} [{lo_elbow:.1f}, {hi_elbow:.1f}]")
    print(f"  elbow_k std              : {np.std(elbow_ks, ddof=1):.2f}")
    print(f"  active_dims mean [95% CI]: {m_active:.1f} [{lo_active:.1f}, {hi_active:.1f}]")
    print("="*55)

    # ── Detail plot + checkpoint for the seed closest to the mean elbow_k ──────
    target_idx = int(np.argmin([abs(r['elbow_k'] - m_elbow) for r in results]))
    representative_seed = results[target_idx]['seed']
    print(f"\n[4/4] Using seed {representative_seed} "
          f"(closest to mean elbow_k) for detail plot + checkpoint.")

    if results[target_idx]['seed'] == CFG['seeds'][-1] and best_model is not None:
        # Already have it in memory from the loop above (only true if it was the last one run)
        model, variances, elbow_k, history = best_model, best_variances, best_elbow, best_history
    else:
        # Retrain just this one seed to get the model/variances/history for plotting
        # (results.json only stores summary stats, not full model state, by design —
        # keeps the cache small and crash-safe).
        result, model, variances, elbow_k, history = run_single_seed(embeddings, representative_seed)

    plot_variance_spectrum(variances, elbow_k, history, CFG['plot_path'])
    torch.save({
        'model_state': model.state_dict(),
        'variances'  : variances,
        'elbow_k'    : elbow_k,
        'seed'       : representative_seed,
        'multiseed_summary': {
            'elbow_ks': elbow_ks, 'mean_ci': [m_elbow, lo_elbow, hi_elbow],
        },
        'config'     : CFG,
    }, CFG['model_path'])
    print(f"Model saved → {CFG['model_path']}")
    print(f"Results saved → {CFG['results_path']}")


if __name__ == '__main__':
    main()
