#!/usr/bin/env python3
"""
GASTON_VI_zdim_sweep.py
Sweeps over z_dim values (including z_dim=0) to find the elbow in
reconstruction loss vs. number of biological latent dimensions.

When z_dim=0 the BiologicalEncoder is removed entirely and the decoder
receives only the spatial isodepth d_i (pure spatial baseline).
"""

import sys, os, time
import numpy as np
import torch, torch.nn as nn
import pyro, pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pandas as pd
import argparse
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scvi.nn import Encoder as scviEncoder

# ============================================================
# 0. PARSE ARGS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
    choices=["kidney", "liver", "tonsil"],
    help="Which dataset to use: kidney, liver, or tonsil")
parser.add_argument("--z_dim", type=int, required=True,
    help="Dimension of biological latent z_i (0 = spatial-only baseline)")
parser.add_argument("--sigma", type=float, required=True,
    help="Positional encoding frequency scale")
parser.add_argument("--num_freqs", type=int, required=True,
    help="Number of positional encoding frequencies")
parser.add_argument("--beta", type=float, required=True,
    help="KL weight for spatial latent d_i")
parser.add_argument("--epochs", type=int, default=175)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--val_frac", type=float, default=0.3,
    help="Fraction of spots held out for validation")
parser.add_argument("--results_dir", type=str, required=True,
    help="Base directory for all output results")
args = parser.parse_args()

z_bio_dim = args.z_dim
sigma     = args.sigma
num_freqs = args.num_freqs
beta      = args.beta
EPOCHS    = args.epochs
REPEATS   = args.repeats
BATCH_SIZE = args.batch_size
VAL_FRAC  = args.val_frac

print(f"dataset={args.dataset}, z_dim={z_bio_dim}, sigma={sigma}, num_freqs={num_freqs}, beta={beta}", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
RESULTS_DIR   = args.results_dir
LOSS_PLOT_DIR = f"{RESULTS_DIR}/loss_plots"
LATENT_DIR    = f"{RESULTS_DIR}/latents/zdim{z_bio_dim}"
LATENT_PLOT_DIR = f"{RESULTS_DIR}/latent_plots/zdim{z_bio_dim}"
MODEL_DIR     = f"{RESULTS_DIR}/models"

for d in [RESULTS_DIR, LOSS_PLOT_DIR, LATENT_DIR, LATENT_PLOT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print(f"Loading {args.dataset} data...", flush=True)

if args.dataset == "kidney":
    ROOT  = "/scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium"
    S_np  = np.load(f"{ROOT}/coords_mat.npy")
    A_raw = sparse.load_npz(f"{ROOT}/counts_mat_hvg5000.npz")

elif args.dataset == "liver":
    ROOT       = "/scratch/uchitra1/datasets/spatial-RNA/merfish/mouse_liver_merfish"
    S_np       = np.load(f"{ROOT}/coords_mat.npy")
    counts_raw = np.load(f"{ROOT}/counts_mat.npy").astype(np.float32)
    print(f"Raw counts shape = {counts_raw.shape}", flush=True)
    gene_means  = counts_raw.mean(axis=0)
    gene_vars   = counts_raw.var(axis=0)
    dispersions = gene_vars / (gene_means + 1e-8)
    hvg_idx     = np.argsort(dispersions)[-5000:]
    A_raw       = sparse.csr_matrix(counts_raw[:, hvg_idx])
    print(f"Selected {len(hvg_idx)} HVGs", flush=True)

elif args.dataset == "tonsil":
    import scanpy as sc
    INPUT_DIR = "/scratch/uchitra1/users/sdharma5_code/gaston/binned_outputs/square_016um"
    try:
        adata = sc.read_h5ad(f"{INPUT_DIR}/adata_with_leiden.h5ad")
    except Exception:
        print("Loading from h5 + parquet...", flush=True)
        adata = sc.read_10x_h5(f"{INPUT_DIR}/filtered_feature_bc_matrix.h5")
        spatial_df = pd.read_parquet(
            f"{INPUT_DIR}/spatial/tissue_positions.parquet"
        ).set_index("barcode")
        spatial_df = spatial_df.loc[adata.obs_names]
        adata.obsm["spatial"] = spatial_df[
            ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        ].values
    S_np       = adata.obsm["spatial"].astype(np.float32)
    counts_raw = adata.X
    if hasattr(counts_raw, "toarray"):
        counts_raw = counts_raw.toarray()
    counts_raw  = np.round(counts_raw).astype(np.float32)
    print(f"Raw counts shape = {counts_raw.shape}", flush=True)
    gene_means  = counts_raw.mean(axis=0)
    gene_vars   = counts_raw.var(axis=0)
    dispersions = gene_vars / (gene_means + 1e-8)
    hvg_idx     = np.argsort(dispersions)[-5000:]
    A_raw       = sparse.csr_matrix(counts_raw[:, hvg_idx])
    print(f"Selected {len(hvg_idx)} HVGs", flush=True)

else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

print(f"S shape = {S_np.shape}", flush=True)
print(f"Count matrix shape = {A_raw.shape}", flush=True)
G = A_raw.shape[1]

# ============================================================
# 2. HELPERS
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
        + y   * (torch.log(mu)    - torch.log(theta + mu))
    )


def nb_loss(model, X, S, batch_size, eps=1e-8):
    """Average per-element NB NLL (evaluation / reconstruction metric)."""
    theta_g = pyro.get_param_store()["theta_g"].detach().to(X.device)
    total, denom = 0.0, 0
    for x, s in iterate_minibatches(X, S, batch_size, shuffle=False):
        x_log1p = torch.log1p(x)
        d_mu, _ = model.spatial_encoder(s)
        z_mu    = x.new_zeros(x.shape[0], 0) if model.z_dim == 0 \
                  else model.bio_encoder(x_log1p)[0]
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
    """
    Run full dataset through encoders and return all latent arrays.
    Returns dict with keys: d_mu, d_sigma, z_mu, z_sigma, l_mu, l_sigma
    z_mu / z_sigma are (N, 0) arrays when z_dim=0.
    """
    all_d_mu, all_d_sigma = [], []
    all_z_mu, all_z_sigma = [], []
    all_l_mu, all_l_sigma = [], []

    model.eval()
    for x, s in iterate_minibatches(X, S, batch_size, shuffle=False):
        x_log1p = torch.log1p(x)

        d_mu, d_sigma = model.spatial_encoder(s)
        all_d_mu.append(d_mu.cpu().numpy())
        all_d_sigma.append(d_sigma.cpu().numpy())

        if model.z_dim > 0:
            z_mu, z_sigma = model.bio_encoder(x_log1p)
            all_z_mu.append(z_mu.cpu().numpy())
            all_z_sigma.append(z_sigma.cpu().numpy())
        else:
            all_z_mu.append(np.zeros((x.shape[0], 0), dtype=np.float32))
            all_z_sigma.append(np.zeros((x.shape[0], 0), dtype=np.float32))

        l_mu, l_sigma = model.lib_encoder(x_log1p)
        all_l_mu.append(l_mu.cpu().numpy())
        all_l_sigma.append(l_sigma.cpu().numpy())

    return {
        "d_mu":    np.concatenate(all_d_mu,    axis=0),
        "d_sigma": np.concatenate(all_d_sigma, axis=0),
        "z_mu":    np.concatenate(all_z_mu,    axis=0),
        "z_sigma": np.concatenate(all_z_sigma, axis=0),
        "l_mu":    np.concatenate(all_l_mu,    axis=0),
        "l_sigma": np.concatenate(all_l_sigma, axis=0),
    }

# ============================================================
# 3. POSITIONAL ENCODING
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
# 4. MODEL COMPONENTS
# ============================================================
class SpatialEncoder(nn.Module):
    def __init__(self, s_dim, d_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu        = nn.Linear(hidden_dim, d_dim)
        self.log_sigma = nn.Linear(hidden_dim, d_dim)

    def forward(self, s):
        h = self.net(s)
        return self.mu(h), torch.exp(self.log_sigma(h))


class BiologicalEncoder(nn.Module):
    def __init__(self, g_dim, z_dim, n_layers=1, n_hidden=128, dropout_rate=0.1):
        super().__init__()
        self.encoder = scviEncoder(
            n_input=g_dim, n_output=z_dim,
            n_layers=n_layers, n_hidden=n_hidden,
            dropout_rate=dropout_rate, distribution="normal",
        )

    def forward(self, x):
        q_m, q_v, _ = self.encoder(x)
        return q_m, torch.sqrt(q_v)


class LibrarySizeEncoder(nn.Module):
    def __init__(self, g_dim, n_layers=1, n_hidden=128, dropout_rate=0.1):
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
    def __init__(self, d_dim, z_dim, g_dim, hidden_dim=128):
        super().__init__()
        in_dim   = d_dim + z_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.rho_logits = nn.Linear(hidden_dim, g_dim)

    def forward(self, d, z):
        h = self.net(torch.cat([d, z], dim=-1)) if z.shape[-1] > 0 else self.net(d)
        return torch.softmax(self.rho_logits(h), dim=-1)

# ============================================================
# 5. FULL MODEL
# ============================================================
class GASTON_scVI_zdim(nn.Module):
    def __init__(self, g_dim, s_dim, d_dim=1, z_dim=10,
                 hidden_dim_spatial=20,
                 n_layers_bio=1, n_hidden_bio=128,
                 n_layers_lib=1, n_hidden_lib=128,
                 dropout_rate=0.1,
                 hidden_dim_decoder=20,
                 log_lib_mean=0.0, log_lib_var=1.0,
                 beta=0.01):
        super().__init__()
        self.g_dim = g_dim
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.beta  = beta

        self.register_buffer("log_lib_mean",
                             torch.tensor(log_lib_mean, dtype=torch.float32))
        self.register_buffer("log_lib_var",
                             torch.tensor(log_lib_var,  dtype=torch.float32))

        self.spatial_encoder = SpatialEncoder(s_dim, d_dim, hidden_dim_spatial)
        self.lib_encoder     = LibrarySizeEncoder(g_dim, n_layers_lib, n_hidden_lib, dropout_rate)
        self.decoder         = CombinedDecoder(d_dim, z_dim, g_dim, hidden_dim_decoder)

        if z_dim > 0:
            self.bio_encoder = BiologicalEncoder(g_dim, z_dim, n_layers_bio, n_hidden_bio, dropout_rate)
        else:
            self.bio_encoder = None
            print("z_dim=0: BiologicalEncoder disabled (spatial-only baseline)", flush=True)

    def model(self, x, s, beta=1e-20):
        pyro.module("decoder", self.decoder)
        theta_g = pyro.param("theta_g", x.new_ones(x.shape[1]),
                             constraint=dist.constraints.positive)

        with pyro.plate("data", x.shape[0]):
            with pyro.poutine.scale(scale=self.beta):
                d = pyro.sample("d",
                    dist.Normal(x.new_zeros(x.shape[0], self.d_dim),
                                x.new_ones( x.shape[0], self.d_dim)).to_event(1))

            if self.z_dim > 0:
                z = pyro.sample("z",
                    dist.Normal(x.new_zeros(x.shape[0], self.z_dim),
                                x.new_ones( x.shape[0], self.z_dim)).to_event(1))
            else:
                z = x.new_zeros(x.shape[0], 0)

            l_prior_mu    = self.log_lib_mean.expand(x.shape[0], 1)
            l_prior_sigma = self.log_lib_var.sqrt().expand(x.shape[0], 1)
            log_l = pyro.sample("log_l",
                dist.Normal(l_prior_mu, l_prior_sigma).to_event(1))
            l = torch.exp(log_l)

            rho   = self.decoder(d, z)
            mu    = l * rho
            theta = theta_g.unsqueeze(0).expand_as(mu)
            rate  = theta / (mu + 1e-8)
            pyro.sample("obs",
                dist.GammaPoisson(concentration=theta, rate=rate).to_event(1),
                obs=x)

    def guide(self, x, s, beta=1e-20):
        pyro.module("spatial_encoder", self.spatial_encoder)
        pyro.module("lib_encoder",     self.lib_encoder)

        x_log1p = torch.log1p(x)

        with pyro.plate("data", x.shape[0]):
            with pyro.poutine.scale(scale=beta):
                d_mu, d_sigma = self.spatial_encoder(s)
                pyro.sample("d", dist.Normal(d_mu, d_sigma).to_event(1))

            if self.z_dim > 0:
                pyro.module("bio_encoder", self.bio_encoder)
                z_mu, z_sigma = self.bio_encoder(x_log1p)
                pyro.sample("z", dist.Normal(z_mu, z_sigma).to_event(1))

            l_mu, l_sigma = self.lib_encoder(x_log1p)
            pyro.sample("log_l", dist.Normal(l_mu, l_sigma).to_event(1))

# ============================================================
# 6. PREPARE DATA
# ============================================================
print("Preparing data tensors...", flush=True)
X_full = torch.tensor(A_raw.toarray(), dtype=torch.float32).to(device)
S_full = torch.tensor(S_np,            dtype=torch.float32).to(device)

N = X_full.shape[0]
torch.manual_seed(0)
perm      = torch.randperm(N)
n_val     = max(1, int(N * VAL_FRAC))
val_idx   = perm[:n_val]
train_idx = perm[n_val:]
print(f"Train spots: {len(train_idx)}, Val spots: {len(val_idx)} ({VAL_FRAC*100:.0f}%)", flush=True)

train_gene    = X_full[train_idx]
train_spatial = S_full[train_idx]
val_gene      = X_full[val_idx]
val_spatial   = S_full[val_idx]

eps = 1e-8
train_spatial_mean = train_spatial.mean(dim=0, keepdim=True)
train_spatial_std  = train_spatial.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)

train_spatial_norm = (train_spatial - train_spatial_mean) / train_spatial_std
val_spatial_norm   = (val_spatial   - train_spatial_mean) / train_spatial_std

pe_encoder = LogFrequencyPositionalEncoding(in_dim=2, num_frequencies=num_freqs, sigma=sigma)
train_spatial_norm = pe_encoder(train_spatial_norm)
val_spatial_norm   = pe_encoder(val_spatial_norm)
# Also encode the full spatial array (for latent extraction over all spots)
S_full_norm = (S_full - train_spatial_mean) / train_spatial_std
S_full_norm = pe_encoder(S_full_norm)

s_dim = train_spatial_norm.shape[1]
print(f"train_spatial_norm.shape: {train_spatial_norm.shape}", flush=True)

lib_sizes    = train_gene.sum(dim=1)
log_lib      = torch.log(lib_sizes + eps)
log_lib_mean = log_lib.mean().item()
log_lib_var  = log_lib.var().item()
print(f"Log-library size: mean={log_lib_mean:.3f}, var={log_lib_var:.3f}", flush=True)
print(f"Train mean expr: {train_gene.mean():.4f}, Val mean expr: {val_gene.mean():.4f}", flush=True)

# ============================================================
# 7. PLOTTING HELPERS
# ============================================================
def plot_isodepth(S_xy, d_mu, save_path, title="Isodepth (d_mu)"):
    """Scatter plot of spots colored by isodepth value."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(S_xy[:, 0], S_xy[:, 1],
                    c=d_mu[:, 0], cmap="viridis",
                    s=1.5, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="d_mu")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_z_latents(S_xy, z_mu, d_mu, save_path, title_prefix="z"):
    """
    One subplot per z dimension, spots colored by that z value.
    Also one subplot coloring by isodepth for reference.
    """
    z_dim = z_mu.shape[1]
    if z_dim == 0:
        return  # nothing to plot

    ncols = min(4, z_dim + 1)
    nrows = math.ceil((z_dim + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # First panel: isodepth for reference
    sc = axes_flat[0].scatter(S_xy[:, 0], S_xy[:, 1],
                               c=d_mu[:, 0], cmap="viridis",
                               s=1.0, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=axes_flat[0])
    axes_flat[0].set_title("isodepth (d_mu)", fontsize=9)
    axes_flat[0].set_aspect("equal")
    axes_flat[0].axis("off")

    # One panel per z dimension
    for i in range(z_dim):
        ax = axes_flat[i + 1]
        sc = ax.scatter(S_xy[:, 0], S_xy[:, 1],
                        c=z_mu[:, i], cmap="viridis",
                        s=1.0, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"z_{i}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")

    # Hide unused panels
    for j in range(z_dim + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title_prefix, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================
# 8. TRAINING LOOP (over repeats)
# ============================================================
print(f"=== BEGIN SWEEP: z_dim={z_bio_dim} ===", flush=True)

all_recon_train_loss = []
all_recon_val_loss   = []
all_elbo_loss        = []

for rep in range(REPEATS):
    seed = 42 + rep
    print(f"\nRepeat {rep+1}/{REPEATS}  (seed={seed})", flush=True)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    vae = GASTON_scVI_zdim(
        g_dim=G, s_dim=s_dim, d_dim=1, z_dim=z_bio_dim,
        hidden_dim_spatial=20,
        n_layers_bio=1, n_hidden_bio=128,
        n_layers_lib=1, n_hidden_lib=128,
        dropout_rate=0.1,
        hidden_dim_decoder=20,
        log_lib_mean=log_lib_mean, log_lib_var=log_lib_var,
        beta=beta,
    ).to(device)

    model_fn = lambda x, s: vae.model(x, s, beta=beta)
    guide_fn = lambda x, s: vae.guide(x, s, beta=beta)
    svi = SVI(model_fn, guide_fn, Adam({"lr": 1e-3}), loss=Trace_ELBO())

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
            recon_val   = nb_loss(vae, val_gene,   val_spatial_norm,   BATCH_SIZE)
        recon_train_history.append(recon_train)
        recon_val_history.append(recon_val)

        if epoch % 25 == 0:
            print(f"  recon_train={recon_train:.5f}  recon_val={recon_val:.5f}", flush=True)

    all_recon_train_loss.append(recon_train_history)
    all_recon_val_loss.append(recon_val_history)
    all_elbo_loss.append(elbo_history)

    # ── per-repeat loss curve ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(recon_train_history, linewidth=2, label="train recon loss")
    ax.plot(recon_val_history,   linewidth=2, label="val recon loss", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NB NLL (recon loss)")
    ax.set_title(f"Train vs Val Recon Loss — z_dim={z_bio_dim}, rep={rep+1}, seed={seed}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{LOSS_PLOT_DIR}/recon_zdim{z_bio_dim}_rep{rep+1}.png", dpi=150)
    plt.close()

    # ── save model (state_dict + Pyro param store) ────────────────────
    model_path = f"{MODEL_DIR}/model_zdim{z_bio_dim}_rep{rep+1}.pt"
    torch.save({
        "state_dict":   vae.state_dict(),
        "pyro_params":  {k: v.detach().cpu() for k, v in pyro.get_param_store().items()},
        "model_kwargs": {
            "g_dim": G, "s_dim": s_dim, "d_dim": 1, "z_dim": z_bio_dim,
            "hidden_dim_spatial": 20,
            "n_layers_bio": 1, "n_hidden_bio": 128,
            "n_layers_lib": 1, "n_hidden_lib": 128,
            "dropout_rate": 0.1, "hidden_dim_decoder": 128,
            "log_lib_mean": log_lib_mean, "log_lib_var": log_lib_var,
            "beta": beta,
        },
        "sigma": sigma, "num_freqs": num_freqs,
        "train_spatial_mean": train_spatial_mean.cpu().numpy(),
        "train_spatial_std":  train_spatial_std.cpu().numpy(),
        "train_idx": train_idx.cpu().numpy(),
        "val_idx":   val_idx.cpu().numpy(),
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

    plot_isodepth(
        S_np, latents["d_mu"],
        save_path=f"{rep_plot_dir}/isodepth_spatial.png",
        title=f"Isodepth — z_dim={z_bio_dim}, rep={rep+1}"
    )

    if z_bio_dim > 0:
        plot_z_latents(
            S_np, latents["z_mu"], latents["d_mu"],
            save_path=f"{rep_plot_dir}/z_latents_spatial.png",
            title_prefix=f"z latents — z_dim={z_bio_dim}, rep={rep+1}"
        )

    print(f"Saved latent plots → {rep_plot_dir}", flush=True)

# ============================================================
# 9. AGGREGATE & SAVE
# ============================================================
all_recon_train = np.array(all_recon_train_loss)
avg_recon_train = all_recon_train.mean(axis=0)
std_recon_train = all_recon_train.std(axis=0)

all_recon_val = np.array(all_recon_val_loss)
avg_recon_val = all_recon_val.mean(axis=0)
std_recon_val = all_recon_val.std(axis=0)

all_elbo  = np.array(all_elbo_loss)
avg_elbo  = all_elbo.mean(axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_recon_train, linewidth=2, color="steelblue", label="train recon loss")
ax.fill_between(range(EPOCHS),
                avg_recon_train - std_recon_train,
                avg_recon_train + std_recon_train,
                alpha=0.2, color="steelblue")
ax.plot(avg_recon_val, linewidth=2, color="tomato", linestyle="--", label="val recon loss")
ax.fill_between(range(EPOCHS),
                avg_recon_val - std_recon_val,
                avg_recon_val + std_recon_val,
                alpha=0.2, color="tomato")
ax.set_xlabel("Epoch"); ax.set_ylabel("NB NLL (recon loss)")
ax.set_title(f"Avg Train vs Val Recon Loss — z_dim={z_bio_dim}")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{LOSS_PLOT_DIR}/recon_zdim{z_bio_dim}_avg.png", dpi=150)
plt.close()

df = pd.DataFrame([{
    "z_dim":                  z_bio_dim,
    "sigma":                  sigma,
    "num_freqs":              num_freqs,
    "beta":                   beta,
    "epochs":                 EPOCHS,
    "repeats":                REPEATS,
    "val_frac":               VAL_FRAC,
    "final_recon_train_loss": float(avg_recon_train[-1]),
    "std_recon_train_loss":   float(std_recon_train[-1]),
    "final_recon_val_loss":   float(avg_recon_val[-1]),
    "std_recon_val_loss":     float(std_recon_val[-1]),
    "final_elbo_loss":        float(avg_elbo[-1]),
    "train_val_gap":          float(avg_recon_val[-1] - avg_recon_train[-1]),
}])
csv_path = f"{RESULTS_DIR}/zdim_sweep_zdim{z_bio_dim}.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}", flush=True)
print("=== DONE ===", flush=True)