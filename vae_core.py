"""
vae_core.py
===========
Shared model / training / elbow-detection code, extracted verbatim from
extract_embeddings_and_vae.py so that beta_sweep.py trains with EXACTLY
the same OrderedVAE architecture, training loop, and (already-corrected)
detect_elbow function -- no reimplementation, no drift between the single-
beta scripts you already ran and the sweep.

Only addition relative to the original file: `eval_held_out_recon`, which
the original scripts never needed (they only tracked recon loss on the
training set for the loss-curve plot). The sweep needs held-out
reconstruction loss specifically, to build the rate-distortion curve.
"""

import math
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─── VAE with Nested Dropout (unchanged from extract_embeddings_and_vae.py) ─
class OrderedVAE(nn.Module):
    def __init__(self, d_input, d_hidden, d_latent):
        super().__init__()
        self.d_latent = d_latent

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
        logvar = self.fc_logvar(h).clamp(-4, 4)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def nested_dropout(self, z):
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


def vae_loss(x_recon, x, mu, logvar, beta):
    recon = F.mse_loss(x_recon, x, reduction='mean')
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.item(), kl.item()


def get_beta(epoch, epochs, beta_start, beta_end):
    """Linear KL annealing: ramp from beta_start to beta_end over first half of training."""
    warmup = epochs // 2
    if epoch <= warmup:
        return beta_start + (beta_end - beta_start) * (epoch / warmup)
    return beta_end


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
    return total / n, recon_sum / n, kl_sum / n


@torch.no_grad()
def compute_latent_variance(model, loader):
    model.eval()
    all_mu = []
    for (x,) in loader:
        mu, _ = model.encode(x.to(DEVICE))
        all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu, dim=0)
    return all_mu.var(dim=0).numpy()


@torch.no_grad()
def eval_held_out_recon(model, loader):
    """
    Held-out reconstruction loss (MSE), beta-independent, deterministic
    (eval mode -> no nested dropout, no sampling noise). This is the
    'distortion' axis of the rate-distortion curve used to select beta*.
    Not present in the original scripts, which only logged training recon.
    """
    model.eval()
    total, n = 0.0, 0
    for (x,) in loader:
        x = x.to(DEVICE)
        mu, logvar = model.encode(x)
        x_recon    = model.decoder(mu)  # deterministic: no dropout, use mu directly
        loss       = F.mse_loss(x_recon, x, reduction='mean')
        total += loss.item()
        n     += 1
    return total / n


def detect_elbow(variances):
    """
    Corrected elbow detector (identical to extract_embeddings_and_vae.py).
    Valid for convex-decreasing curves (variance spectra), NOT the standard
    kneedle formula which assumes concave-decreasing curves.
    """
    sorted_var = np.sort(variances)[::-1]
    n          = len(sorted_var)
    norm = (sorted_var - sorted_var[-1]) / (sorted_var[0] - sorted_var[-1] + 1e-10)
    line = np.linspace(1.0, 0.0, n)
    diff  = line - norm
    elbow = int(np.argmax(diff)) + 1
    return elbow, sorted_var


def detect_knee_convex_decreasing(x_vals, y_vals):
    """
    Same line-distance logic as detect_elbow, generalized to arbitrary
    (x, y) convex-decreasing curves rather than assuming unit-spaced
    integer indices. Used to select beta* from the reconstruction-loss-
    vs-beta curve, so beta selection uses the identical geometric
    criterion as the paper's headline elbow-detection fix, rather than a
    second, different heuristic.
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    x0, y0 = x_norm[0], y_norm[0]
    x1, y1 = x_norm[-1], y_norm[-1]
    a = y1 - y0
    b = -(x1 - x0)
    c = x1 * y0 - y1 * x0
    denom = np.sqrt(a**2 + b**2) + 1e-10
    distances = (a * x_norm + b * y_norm + c) / denom
    return int(np.argmax(np.abs(distances)))


def mean_ci(values, confidence=0.95):
    arr = np.array(values, dtype=float)
    n   = len(arr)
    m   = arr.mean()
    if n < 2:
        return m, m, m
    se = arr.std(ddof=1) / math.sqrt(n)
    h  = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - h, m + h
