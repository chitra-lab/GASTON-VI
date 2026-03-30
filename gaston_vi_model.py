#!/usr/bin/env python3
"""
gaston_vi_model.py
Unified GASTON-VI model.

  --z_dim 0         spatial-only baseline: decoder receives only isodepth d_i.
  --z_dim K>0       full model: decoder receives [d_i, z_i] where z_i is a
                    K-dim biological latent from a scVI encoder.

  --n_isodepth 1    single isodepth coordinate (current default behaviour).
  --n_isodepth M>1  M independent (x,y)→d_k spatial encoders; their outputs are
                    concatenated and passed jointly to the decoder.  Each d_k has
                    its own prior N(0,1) and its own KL scaled by --beta.

Data files are supplied as CLI arguments — nothing is hardcoded.

Usage example:
    python gaston_vi_model.py \
        --coords_file  /path/to/coords_mat.npy \
        --counts_file  /path/to/counts_mat_hvg.npz \
        --z_dim        10 \
        --n_isodepth   1 \
        --sigma        0.183959 \
        --num_freqs    16 \
        --beta         0.01 \
        --epochs       175 \
        --repeats      5 \
        --batch_size   2586 \
        --val_frac     0.3 \
        --results_dir  /path/to/results
"""

import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy import sparse
from scvi.nn import Encoder as scviEncoder

# ============================================================
# 0.  ARGS
# ============================================================
parser = argparse.ArgumentParser(description="GASTON-VI unified model")

# -- data --
parser.add_argument("--coords_file", type=str, required=True,
    help="Path to spatial-coordinate .npy file, shape (N, 2)")
parser.add_argument("--counts_file", type=str, required=True,
    help="Path to count matrix (.npz = sparse, .npy = dense float32)")

# -- model --
parser.add_argument("--z_dim", type=int, default=0,
    help="Biological latent dimension (0 = spatial-only baseline)")
parser.add_argument("--n_isodepth", type=int, default=1,
    help="Number of isodepth coordinates.  1 uses a single spatial encoder "
         "(current behaviour).  M>1 trains M independent (x,y)->d_k encoders "
         "whose outputs are concatenated before the decoder.")
parser.add_argument("--sigma", type=float, required=True,
    help="Positional-encoding frequency scale")
parser.add_argument("--num_freqs", type=int, required=True,
    help="Number of positional-encoding frequencies")
parser.add_argument("--beta", type=float, required=True,
    help="KL weight for spatial latent(s) d_i")

# -- training --
parser.add_argument("--epochs",     type=int,   default=175)
parser.add_argument("--repeats",    type=int,   default=1)
parser.add_argument("--batch_size", type=int,   required=True)
parser.add_argument("--val_frac",   type=float, default=0.3,
    help="Fraction of spots held out for validation (0 = no split)")

# -- output --
parser.add_argument("--results_dir", type=str, required=True)

# -- misc --
parser.add_argument("--norm",        type=str,   default="z_score")
parser.add_argument("--noise_level", type=float, default=1.0)

args = parser.parse_args()

z_bio_dim   = args.z_dim
n_isodepth  = args.n_isodepth
sigma       = args.sigma
num_freqs   = args.num_freqs
beta        = args.beta
EPOCHS      = args.epochs
REPEATS     = args.repeats
BATCH_SIZE  = args.batch_size
VAL_FRAC    = args.val_frac

print(f"z_dim={z_bio_dim}, n_isodepth={n_isodepth}, sigma={sigma}, "
      f"num_freqs={num_freqs}, beta={beta}", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

# ============================================================
# 1.  OUTPUT DIRECTORIES
# ============================================================
RESULTS_DIR     = args.results_dir
LOSS_PLOT_DIR   = f"{RESULTS_DIR}/loss_plots"
LATENT_DIR      = f"{RESULTS_DIR}/latents/zdim{z_bio_dim}_isodepth{n_isodepth}"
LATENT_PLOT_DIR = f"{RESULTS_DIR}/latent_plots/zdim{z_bio_dim}_isodepth{n_isodepth}"
MODEL_DIR       = f"{RESULTS_DIR}/models"

for d in [RESULTS_DIR, LOSS_PLOT_DIR, LATENT_DIR, LATENT_PLOT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 2.  LOAD DATA
# ============================================================
print(f"Loading coords from:  {args.coords_file}", flush=True)
print(f"Loading counts from:  {args.counts_file}", flush=True)

S_np = np.load(args.coords_file)  # (N, 2)

ext = os.path.splitext(args.counts_file)[1].lower()
if ext == ".npz":
    A_raw = sparse.load_npz(args.counts_file)
elif ext == ".npy":
    A_raw = sparse.csr_matrix(np.load(args.counts_file).astype(np.float32))
else:
    raise ValueError(f"Unsupported counts extension '{ext}'. Use .npz or .npy.")

print(f"S shape            = {S_np.shape}", flush=True)
print(f"Count matrix shape = {A_raw.shape}", flush=True)
G = A_raw.shape[1]

# ============================================================
# 3.  HELPERS
# ============================================================
def iterate_minibatches(X, S, batch_size, shuffle=True):
    N   = X.shape[0]
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    for start in range(0, N, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], S[batch_idx]


def nb_log_prob(y, mu, theta, eps=1e-8):
    mu = torch.clamp(mu, min=eps)
    return (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y     * (torch.log(mu)    - torch.log(theta + mu))
    )


def nb_loss(model, X, S, batch_size, eps=1e-8):
    """Average per-element NB NLL (evaluation / reconstruction metric)."""
    theta_g = pyro.get_param_store()["theta_g"].detach().to(X.device)
    total, denom = 0.0, 0
    for x, s in iterate_minibatches(X, S, batch_size, shuffle=False):
        x_log1p = torch.log1p(x)
        d_mu    = model.encode_d_mu(s)
        z_mu    = (x.new_zeros(x.shape[0], 0) if model.z_dim == 0
                   else model.bio_encoder(x_log1p)[0])
        l_mu, _ = model.lib_encoder(x_log1p)
        l       = torch.exp(l_mu)
        rho     = model.decoder(d_mu, z_mu)
        mu      = l * rho
        theta   = theta_g.unsqueeze(0).expand_as(mu)
        logp    = nb_log_prob(x, mu, theta, eps=eps)
        total  += (-logp).sum().item()
        denom  += x.numel()
    return total / max(1, denom)


@torch.no_grad()
def extract_latents(model, X, S, batch_size):
    """Return latent arrays over the full dataset.

    For n_isodepth == 1:
        keys: d_mu (N,1), d_sigma (N,1), z_mu (N,z_dim), z_sigma, l_mu, l_sigma

    For n_isodepth > 1:
        keys: d_0_mu (N,1), d_0_sigma, ..., d_{M-1}_mu, d_{M-1}_sigma,
              d_mu (N,M, concatenated), d_sigma (N,M),
              z_mu, z_sigma, l_mu, l_sigma
    """
    buckets = {k: [] for k in (
        ["d_mu", "d_sigma"] +
        ([f"d_{k}_mu" for k in range(model.n_isodepth)] +
         [f"d_{k}_sigma" for k in range(model.n_isodepth)]
         if model.n_isodepth > 1 else []) +
        ["z_mu", "z_sigma", "l_mu", "l_sigma"]
    )}

    model.eval()
    for x, s in iterate_minibatches(X, S, batch_size, shuffle=False):
        x_log1p = torch.log1p(x)

        if model.n_isodepth == 1:
            d_mu, d_sigma = model.spatial_encoder(s)
            buckets["d_mu"].append(d_mu.cpu().numpy())
            buckets["d_sigma"].append(d_sigma.cpu().numpy())
        else:
            dk_mus, dk_sigmas = [], []
            for k, enc in enumerate(model.spatial_encoders):
                d_mu_k, d_sigma_k = enc(s)
                buckets[f"d_{k}_mu"].append(d_mu_k.cpu().numpy())
                buckets[f"d_{k}_sigma"].append(d_sigma_k.cpu().numpy())
                dk_mus.append(d_mu_k)
                dk_sigmas.append(d_sigma_k)
            buckets["d_mu"].append(torch.cat(dk_mus,    dim=-1).cpu().numpy())
            buckets["d_sigma"].append(torch.cat(dk_sigmas, dim=-1).cpu().numpy())

        if model.z_dim > 0:
            z_mu, z_sigma = model.bio_encoder(x_log1p)
            buckets["z_mu"].append(z_mu.cpu().numpy())
            buckets["z_sigma"].append(z_sigma.cpu().numpy())
        else:
            buckets["z_mu"].append(np.zeros((x.shape[0], 0), dtype=np.float32))
            buckets["z_sigma"].append(np.zeros((x.shape[0], 0), dtype=np.float32))

        l_mu, l_sigma = model.lib_encoder(x_log1p)
        buckets["l_mu"].append(l_mu.cpu().numpy())
        buckets["l_sigma"].append(l_sigma.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in buckets.items()}

# ============================================================
# 4.  POSITIONAL ENCODING
# ============================================================
class LogFrequencyPositionalEncoding(nn.Module):
    def __init__(self, in_dim, num_frequencies, sigma):
        super().__init__()
        self.in_dim          = in_dim
        self.num_frequencies = num_frequencies
        self.sigma           = sigma
        j = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("freq_bands", 2 * math.pi * sigma ** (j / num_frequencies))
        print(f"PE: sigma={sigma}, num_freqs={num_frequencies}", flush=True)

    def forward(self, s):
        fb    = self.freq_bands.to(s.device)
        s_exp = s.unsqueeze(-2) * fb.view(1, -1, 1)
        sin   = torch.sin(s_exp)
        cos   = torch.cos(s_exp)
        return torch.cat([s, torch.cat([sin, cos], dim=-1).flatten(-2)], dim=-1)

# ============================================================
# 5.  MODEL COMPONENTS
# ============================================================
class SpatialEncoder(nn.Module):
    """Maps encoded spatial coords → 1-D isodepth (mu, sigma)."""
    def __init__(self, s_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu        = nn.Linear(hidden_dim, 1)
        self.log_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        h = self.net(s)
        return self.mu(h), torch.exp(self.log_sigma(h))


class BiologicalEncoder(nn.Module):
    """scVI encoder: gene expression (log1p) → biological latent z_i."""
    def __init__(self, g_dim, z_dim, n_layers=2, n_hidden=128, dropout_rate=0.1):
        super().__init__()
        self.encoder = scviEncoder(
            n_input=g_dim, n_output=z_dim,
            n_layers=n_layers, n_hidden=n_hidden,
            dropout_rate=dropout_rate, distribution="normal",
        )

    def forward(self, x):
        q_m, q_v, _ = self.encoder(x)
        return q_m, torch.sqrt(q_v)  # (mean, std)


class LibrarySizeEncoder(nn.Module):
    """scVI encoder: gene expression (log1p) → log library size l_i."""
    def __init__(self, g_dim, n_layers=2, n_hidden=128, dropout_rate=0.1):
        super().__init__()
        self.encoder = scviEncoder(
            n_input=g_dim, n_output=1,
            n_layers=n_layers, n_hidden=n_hidden,
            dropout_rate=dropout_rate, distribution="normal",
        )

    def forward(self, x):
        q_m, q_v, _ = self.encoder(x)
        return q_m, torch.sqrt(q_v)


class CombinedDecoder(nn.Module):
    """[d_i, z_i] → gene proportions ρ_i (softmax over G genes).

    d_i has dimension n_isodepth; z_i has dimension z_dim (0 = absent).
    """
    def __init__(self, d_dim, z_dim, g_dim, hidden_dim=128):
        super().__init__()
        in_dim = d_dim + z_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.rho_logits = nn.Linear(hidden_dim, g_dim)

    def forward(self, d, z):
        inp = torch.cat([d, z], dim=-1) if z.shape[-1] > 0 else d
        return torch.softmax(self.rho_logits(self.net(inp)), dim=-1)

# ============================================================
# 6.  FULL MODEL
# ============================================================
class GASTON_VI(nn.Module):
    """
    Unified GASTON-VI model.

    Latents:
        d_i^(k) ~ N(0, 1)  for k = 0 … n_isodepth-1  [isodepth(s)]
        z_i     ~ N(0, I)   (only when z_dim > 0)       [biological]
        l_i     ~ LogNormal(l_mu, l_σ²)                 [library size]

    Decoder:
        ρ_i = softmax( f([ d_i^(0), …, d_i^(K-1), z_i ]) )
        a_ig ~ NB( l_i · ρ_ig, θ_g )

    When n_isodepth=1 the code path is identical to the original single-d model.
    """
    def __init__(self, g_dim, s_dim,
                 n_isodepth=1, z_dim=0,
                 hidden_dim_spatial=20,
                 n_layers_bio=2, n_hidden_bio=128,
                 n_layers_lib=2, n_hidden_lib=128,
                 dropout_rate=0.1,
                 hidden_dim_decoder=20,
                 log_lib_mean=0.0, log_lib_var=1.0,
                 beta=0.01):
        super().__init__()
        self.g_dim      = g_dim
        self.n_isodepth = n_isodepth
        self.z_dim      = z_dim
        self.beta       = beta

        self.register_buffer("log_lib_mean",
                             torch.tensor(log_lib_mean, dtype=torch.float32))
        self.register_buffer("log_lib_var",
                             torch.tensor(log_lib_var,  dtype=torch.float32))

        # ── spatial encoder(s) ─────────────────────────────────────────
        if n_isodepth == 1:
            # Single encoder kept as self.spatial_encoder (original behaviour).
            self.spatial_encoder = SpatialEncoder(s_dim, hidden_dim_spatial)
        else:
            # M independent encoders, one per isodepth coordinate.
            self.spatial_encoders = nn.ModuleList([
                SpatialEncoder(s_dim, hidden_dim_spatial)
                for _ in range(n_isodepth)
            ])
            print(f"n_isodepth={n_isodepth}: {n_isodepth} independent spatial encoders",
                  flush=True)

        # ── biological encoder (optional) ─────────────────────────────
        if z_dim > 0:
            self.bio_encoder = BiologicalEncoder(g_dim, z_dim, n_layers_bio,
                                                 n_hidden_bio, dropout_rate)
        else:
            self.bio_encoder = None
            print("z_dim=0: no BiologicalEncoder (spatial-only baseline)", flush=True)

        self.lib_encoder = LibrarySizeEncoder(g_dim, n_layers_lib,
                                              n_hidden_lib, dropout_rate)
        # decoder input dim = n_isodepth (from d) + z_dim (from z)
        self.decoder = CombinedDecoder(n_isodepth, z_dim, g_dim, hidden_dim_decoder)

    # ------------------------------------------------------------------
    # Convenience: get concatenated isodepth posterior mean
    # ------------------------------------------------------------------
    def encode_d_mu(self, s):
        if self.n_isodepth == 1:
            return self.spatial_encoder(s)[0]         # (N, 1)
        return torch.cat([enc(s)[0] for enc in self.spatial_encoders], dim=-1)  # (N, M)

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    def model(self, x, s, beta=None):
        if beta is None:
            beta = self.beta
        pyro.module("decoder",      self.decoder)
        pyro.module("lib_encoder",  self.lib_encoder)
        if self.z_dim > 0:
            pyro.module("bio_encoder", self.bio_encoder)

        theta_g = pyro.param("theta_g", x.new_ones(x.shape[1]),
                             constraint=dist.constraints.positive)

        with pyro.plate("data", x.shape[0]):
            # ── isodepth latent(s) ─────────────────────────────────────
            if self.n_isodepth == 1:
                with pyro.poutine.scale(scale=beta):
                    d = pyro.sample("d",
                        dist.Normal(x.new_zeros(x.shape[0], 1),
                                    x.new_ones( x.shape[0], 1)).to_event(1))
            else:
                d_parts = []
                with pyro.poutine.scale(scale=beta):
                    for k in range(self.n_isodepth):
                        d_k = pyro.sample(f"d_{k}",
                            dist.Normal(x.new_zeros(x.shape[0], 1),
                                        x.new_ones( x.shape[0], 1)).to_event(1))
                        d_parts.append(d_k)
                d = torch.cat(d_parts, dim=-1)  # (N, n_isodepth)

            # ── biological latent (optional) ───────────────────────────
            if self.z_dim > 0:
                z = pyro.sample("z",
                    dist.Normal(x.new_zeros(x.shape[0], self.z_dim),
                                x.new_ones( x.shape[0], self.z_dim)).to_event(1))
            else:
                z = x.new_zeros(x.shape[0], 0)

            # ── library size ───────────────────────────────────────────
            l_prior_mu    = self.log_lib_mean.expand(x.shape[0], 1)
            l_prior_sigma = self.log_lib_var.sqrt().expand(x.shape[0], 1)
            log_l = pyro.sample("log_l",
                dist.Normal(l_prior_mu, l_prior_sigma).to_event(1))
            l = torch.exp(log_l)

            # ── decode ─────────────────────────────────────────────────
            rho   = self.decoder(d, z)
            mu    = l * rho
            theta = theta_g.unsqueeze(0).expand_as(mu)
            rate  = theta / (mu + 1e-8)
            pyro.sample("obs",
                dist.GammaPoisson(concentration=theta, rate=rate).to_event(1),
                obs=x)

    # ------------------------------------------------------------------
    # guide
    # ------------------------------------------------------------------
    def guide(self, x, s, beta=None):
        if beta is None:
            beta = self.beta
        pyro.module("lib_encoder", self.lib_encoder)
        if self.z_dim > 0:
            pyro.module("bio_encoder", self.bio_encoder)

        x_log1p = torch.log1p(x)

        with pyro.plate("data", x.shape[0]):
            # ── isodepth latent(s) ─────────────────────────────────────
            if self.n_isodepth == 1:
                pyro.module("spatial_encoder", self.spatial_encoder)
                with pyro.poutine.scale(scale=beta):
                    d_mu, d_sigma = self.spatial_encoder(s)
                    pyro.sample("d", dist.Normal(d_mu, d_sigma).to_event(1))
            else:
                pyro.module("spatial_encoders", self.spatial_encoders)
                with pyro.poutine.scale(scale=beta):
                    for k, enc in enumerate(self.spatial_encoders):
                        d_mu_k, d_sigma_k = enc(s)
                        pyro.sample(f"d_{k}",
                                    dist.Normal(d_mu_k, d_sigma_k).to_event(1))

            # ── biological latent (optional) ───────────────────────────
            if self.z_dim > 0:
                z_mu, z_sigma = self.bio_encoder(x_log1p)
                pyro.sample("z", dist.Normal(z_mu, z_sigma).to_event(1))

            # ── library size ───────────────────────────────────────────
            l_mu, l_sigma = self.lib_encoder(x_log1p)
            pyro.sample("log_l", dist.Normal(l_mu, l_sigma).to_event(1))

# ============================================================
# 7.  PLOTTING HELPERS
# ============================================================
def plot_isodepth(S_xy, d_mu, save_path, title="Isodepth"):
    """Plot spots coloured by a single isodepth dimension."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(S_xy[:, 0], S_xy[:, 1],
                    c=d_mu, cmap="viridis",
                    s=1.5, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="d_mu")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_isodepths(S_xy, d_mu_full, save_path, title_prefix="Isodepth"):
    """One subplot per isodepth dimension (d_mu_full shape (N, n_isodepth))."""
    M = d_mu_full.shape[1]
    ncols = min(4, M)
    nrows = math.ceil(M / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 3.5 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()
    for k in range(M):
        ax = axes_flat[k]
        sc = ax.scatter(S_xy[:, 0], S_xy[:, 1],
                        c=d_mu_full[:, k], cmap="viridis",
                        s=1.5, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"d_{k}", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
    for j in range(M, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(title_prefix, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_z_latents(S_xy, z_mu, d_mu_col0, save_path, title_prefix="z latents"):
    """One subplot per z dimension plus a reference isodepth panel."""
    z_dim = z_mu.shape[1]
    if z_dim == 0:
        return

    ncols    = min(4, z_dim + 1)
    nrows    = math.ceil((z_dim + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 3.5 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()

    sc = axes_flat[0].scatter(S_xy[:, 0], S_xy[:, 1],
                               c=d_mu_col0, cmap="viridis",
                               s=1.0, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=axes_flat[0])
    axes_flat[0].set_title("isodepth d_0", fontsize=9)
    axes_flat[0].set_aspect("equal"); axes_flat[0].axis("off")

    for i in range(z_dim):
        ax = axes_flat[i + 1]
        sc = ax.scatter(S_xy[:, 0], S_xy[:, 1],
                        c=z_mu[:, i], cmap="viridis",
                        s=1.0, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"z_{i}", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")

    for j in range(z_dim + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title_prefix, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curve(train_history, val_history, save_path, title="Loss"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_history, linewidth=2, color="steelblue", label="train recon loss")
    if val_history is not None:
        ax.plot(val_history, linewidth=2, color="tomato",
                linestyle="--", label="val recon loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NB NLL (recon loss)")
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================
# 8.  PREPARE DATA
# ============================================================
print("Preparing data tensors...", flush=True)
X_full = torch.tensor(A_raw.toarray(), dtype=torch.float32).to(device)
S_full = torch.tensor(S_np,            dtype=torch.float32).to(device)

N = X_full.shape[0]
torch.manual_seed(0)
perm = torch.randperm(N)

if VAL_FRAC > 0:
    n_val     = max(1, int(N * VAL_FRAC))
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)} ({VAL_FRAC*100:.0f}%)",
          flush=True)
    val_gene    = X_full[val_idx]
    val_spatial = S_full[val_idx]
else:
    train_idx = perm
    val_idx   = None
    print("No validation split (val_frac=0)", flush=True)
    val_gene = val_spatial = None

train_gene    = X_full[train_idx]
train_spatial = S_full[train_idx]

eps = 1e-8
train_spatial_mean = train_spatial.mean(dim=0, keepdim=True)
train_spatial_std  = train_spatial.std(dim=0, unbiased=False,
                                        keepdim=True).clamp_min(eps)

train_spatial_norm = (train_spatial - train_spatial_mean) / train_spatial_std
S_full_norm        = (S_full        - train_spatial_mean) / train_spatial_std

pe_encoder = LogFrequencyPositionalEncoding(in_dim=2,
                                             num_frequencies=num_freqs,
                                             sigma=sigma)
train_spatial_norm = pe_encoder(train_spatial_norm)
S_full_norm        = pe_encoder(S_full_norm)

if val_idx is not None:
    val_spatial_norm = pe_encoder((val_spatial - train_spatial_mean)
                                  / train_spatial_std)
else:
    val_spatial_norm = None

s_dim = train_spatial_norm.shape[1]
print(f"Encoded spatial dim: {s_dim}", flush=True)

lib_sizes    = train_gene.sum(dim=1)
log_lib      = torch.log(lib_sizes + eps)
log_lib_mean = log_lib.mean().item()
log_lib_var  = log_lib.var().item()
print(f"Log-library size: mean={log_lib_mean:.3f}, var={log_lib_var:.3f}", flush=True)

# ============================================================
# 9.  TRAINING LOOP
# ============================================================
print(f"=== BEGIN TRAINING (z_dim={z_bio_dim}, n_isodepth={n_isodepth}) ===",
      flush=True)

all_recon_train_loss = []
all_recon_val_loss   = []
all_elbo_loss        = []

for rep in range(REPEATS):
    seed = 42 + rep
    print(f"\nRepeat {rep+1}/{REPEATS}  (seed={seed})", flush=True)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    vae = GASTON_VI(
        g_dim=G, s_dim=s_dim,
        n_isodepth=n_isodepth, z_dim=z_bio_dim,
        hidden_dim_spatial=20,
        n_layers_bio=2, n_hidden_bio=128,
        n_layers_lib=2, n_hidden_lib=128,
        dropout_rate=0.1,
        hidden_dim_decoder=20,
        log_lib_mean=log_lib_mean, log_lib_var=log_lib_var,
        beta=beta,
    ).to(device)

    svi = SVI(vae.model, vae.guide, Adam({"lr": 1e-3}), loss=Trace_ELBO())

    recon_train_history = []
    recon_val_history   = []
    elbo_history        = []

    for epoch in range(EPOCHS):
        if epoch % 25 == 0:
            print(f"  Epoch {epoch}/{EPOCHS}", flush=True)

        vae.train()
        epoch_loss, num = 0.0, 0
        for gene, spatial in iterate_minibatches(train_gene, train_spatial_norm,
                                                  batch_size=BATCH_SIZE, shuffle=True):
            epoch_loss += svi.step(gene, spatial)
            num += gene.shape[0]
        elbo_history.append(epoch_loss / max(1, num))

        vae.eval()
        with torch.no_grad():
            recon_train = nb_loss(vae, train_gene, train_spatial_norm, BATCH_SIZE)
            recon_train_history.append(recon_train)
            if val_idx is not None:
                recon_val = nb_loss(vae, val_gene, val_spatial_norm, BATCH_SIZE)
                recon_val_history.append(recon_val)

        if epoch % 25 == 0:
            msg = f"  recon_train={recon_train:.5f}"
            if val_idx is not None:
                msg += f"  recon_val={recon_val:.5f}"
            print(msg, flush=True)

    all_recon_train_loss.append(recon_train_history)
    all_elbo_loss.append(elbo_history)
    if val_idx is not None:
        all_recon_val_loss.append(recon_val_history)

    # ── per-repeat loss curve ──────────────────────────────────────────
    plot_loss_curve(
        recon_train_history,
        recon_val_history if val_idx is not None else None,
        save_path=f"{LOSS_PLOT_DIR}/recon_zdim{z_bio_dim}_isodepth{n_isodepth}_rep{rep+1}.png",
        title=f"Recon Loss — z_dim={z_bio_dim}, n_isodepth={n_isodepth}, rep={rep+1}",
    )

    # ── save model ────────────────────────────────────────────────────
    model_path = f"{MODEL_DIR}/model_zdim{z_bio_dim}_isodepth{n_isodepth}_rep{rep+1}.pt"
    torch.save({
        "state_dict":  vae.state_dict(),
        "pyro_params": {k: v.detach().cpu() for k, v in
                        pyro.get_param_store().items()},
        "model_kwargs": {
            "g_dim": G, "s_dim": s_dim,
            "n_isodepth": n_isodepth, "z_dim": z_bio_dim,
            "hidden_dim_spatial": 20,
            "n_layers_bio": 2, "n_hidden_bio": 128,
            "n_layers_lib": 2, "n_hidden_lib": 128,
            "dropout_rate": 0.1, "hidden_dim_decoder": 20,
            "log_lib_mean": log_lib_mean, "log_lib_var": log_lib_var,
            "beta": beta,
        },
        "sigma": sigma, "num_freqs": num_freqs,
        "train_spatial_mean": train_spatial_mean.cpu().numpy(),
        "train_spatial_std":  train_spatial_std.cpu().numpy(),
        "train_idx": train_idx.cpu().numpy(),
        "val_idx":   val_idx.cpu().numpy() if val_idx is not None else None,
        "seed": seed,
    }, model_path)
    print(f"Saved model: {model_path}", flush=True)

    # ── extract latents over ALL spots ────────────────────────────────
    latents = extract_latents(vae, X_full, S_full_norm, BATCH_SIZE)

    rep_latent_dir = f"{LATENT_DIR}/rep{rep+1}"
    os.makedirs(rep_latent_dir, exist_ok=True)
    for key, arr in latents.items():
        npy_path = f"{rep_latent_dir}/{key}.npy"
        np.save(npy_path, arr)
        print(f"Saved latent {key} {arr.shape} → {npy_path}", flush=True)

    # ── latent plots ──────────────────────────────────────────────────
    rep_plot_dir = f"{LATENT_PLOT_DIR}/rep{rep+1}"
    os.makedirs(rep_plot_dir, exist_ok=True)

    if n_isodepth == 1:
        plot_isodepth(
            S_np, latents["d_mu"][:, 0],
            save_path=f"{rep_plot_dir}/isodepth_spatial.png",
            title=f"Isodepth — z_dim={z_bio_dim}, rep={rep+1}",
        )
    else:
        plot_all_isodepths(
            S_np, latents["d_mu"],
            save_path=f"{rep_plot_dir}/isodepths_spatial.png",
            title_prefix=f"Isodepths (n={n_isodepth}) — z_dim={z_bio_dim}, rep={rep+1}",
        )
        # also individual files per isodepth
        for k in range(n_isodepth):
            plot_isodepth(
                S_np, latents[f"d_{k}_mu"][:, 0],
                save_path=f"{rep_plot_dir}/isodepth_d{k}_spatial.png",
                title=f"Isodepth d_{k} — z_dim={z_bio_dim}, rep={rep+1}",
            )

    if z_bio_dim > 0:
        plot_z_latents(
            S_np, latents["z_mu"], latents["d_mu"][:, 0],
            save_path=f"{rep_plot_dir}/z_latents_spatial.png",
            title_prefix=f"z latents — z_dim={z_bio_dim}, rep={rep+1}",
        )

    print(f"Saved latent plots → {rep_plot_dir}", flush=True)

# ============================================================
# 10. AGGREGATE RESULTS
# ============================================================
all_recon_train = np.array(all_recon_train_loss)
avg_recon_train = all_recon_train.mean(axis=0)
std_recon_train = all_recon_train.std(axis=0)

avg_recon_val = std_recon_val = None
if all_recon_val_loss:
    all_recon_val = np.array(all_recon_val_loss)
    avg_recon_val = all_recon_val.mean(axis=0)
    std_recon_val = all_recon_val.std(axis=0)

avg_elbo = np.array(all_elbo_loss).mean(axis=0)

# aggregate loss curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_recon_train, linewidth=2, color="steelblue", label="avg train recon loss")
ax.fill_between(range(EPOCHS),
                avg_recon_train - std_recon_train,
                avg_recon_train + std_recon_train,
                alpha=0.2, color="steelblue")
if avg_recon_val is not None:
    ax.plot(avg_recon_val, linewidth=2, color="tomato", linestyle="--",
            label="avg val recon loss")
    ax.fill_between(range(EPOCHS),
                    avg_recon_val - std_recon_val,
                    avg_recon_val + std_recon_val,
                    alpha=0.2, color="tomato")
ax.set_xlabel("Epoch"); ax.set_ylabel("NB NLL (recon loss)")
ax.set_title(f"Avg Recon Loss — z_dim={z_bio_dim}, n_isodepth={n_isodepth}")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"{LOSS_PLOT_DIR}/recon_zdim{z_bio_dim}_isodepth{n_isodepth}_avg.png",
    dpi=150)
plt.close()

row = {
    "z_dim":                  z_bio_dim,
    "n_isodepth":             n_isodepth,
    "sigma":                  sigma,
    "num_freqs":              num_freqs,
    "beta":                   beta,
    "epochs":                 EPOCHS,
    "repeats":                REPEATS,
    "val_frac":               VAL_FRAC,
    "final_recon_train_loss": float(avg_recon_train[-1]),
    "std_recon_train_loss":   float(std_recon_train[-1]),
    "final_elbo_loss":        float(avg_elbo[-1]),
}
if avg_recon_val is not None:
    row["final_recon_val_loss"] = float(avg_recon_val[-1])
    row["std_recon_val_loss"]   = float(std_recon_val[-1])
    row["train_val_gap"]        = float(avg_recon_val[-1] - avg_recon_train[-1])

csv_path = f"{RESULTS_DIR}/summary_zdim{z_bio_dim}_isodepth{n_isodepth}.csv"
pd.DataFrame([row]).to_csv(csv_path, index=False)
print(f"Saved summary: {csv_path}", flush=True)
print("=== DONE ===", flush=True)
