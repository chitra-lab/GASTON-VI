#!/usr/bin/env python3
import sys, os, time
import numpy as np
import torch, torch.nn as nn
import pyro, pyro.distributions as dist
from torch.utils.data import Dataset, DataLoader
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from scipy.stats import spearmanr
import pandas as pd
import argparse
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from scvi.nn import Encoder as scviEncoder

# ============================================================
# 0. PARSE SWEEP ARGS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, required=True)
parser.add_argument("--num_freqs", type=int, required=True)
parser.add_argument("--noise_level", type=float, required=True)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--repeats", type=int, default=15)
parser.add_argument("--norm", type=str, required=True)
parser.add_argument("--beta", type=float, required=True,
                    help="Weight on KL for spatial latent d_i")
parser.add_argument("--results_dir", type=str, required=True,
                    help="Base directory for all output results")
parser.add_argument("--batch_size", type=int, required=True,
                    help="Mini-batch size for training")
parser.add_argument("--z_dim", type=int, default=10,
                    help="Dimension of biological latent z_i (default: 10)")
args = parser.parse_args()

sigma = args.sigma
num_freqs = args.num_freqs
noise_level = float(args.noise_level)
EPOCHS = args.epochs
REPEATS = args.repeats
norm = args.norm
beta = args.beta
z_bio_dim = args.z_dim
print(f"Running sigma={sigma}, num_freqs={num_freqs}, beta={beta}, z_dim={z_bio_dim}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}", flush=True)

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
RESULTS_DIR = args.results_dir
ISODEPTH_PLOT_DIR = f"{RESULTS_DIR}/isodepth_plots"
ISODEPTH_NPY_DIR = f"{RESULTS_DIR}/isodepth_npy"
LOSS_PLOT_DIR = f"{RESULTS_DIR}/loss_plots"
VARIANCE_PLOT_DIR = f"{RESULTS_DIR}/variance_plots"
GRID_PLOT_DIR = f"{RESULTS_DIR}/grid/isodepth_plots"
GRID_NPY_DIR = f"{RESULTS_DIR}/grid/isodepth_npy"
BIO_LATENT_PLOT_DIR = f"{RESULTS_DIR}/bio_latent_plots"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ISODEPTH_PLOT_DIR, exist_ok=True)
os.makedirs(ISODEPTH_NPY_DIR, exist_ok=True)
os.makedirs(LOSS_PLOT_DIR, exist_ok=True)
os.makedirs(VARIANCE_PLOT_DIR, exist_ok=True)
os.makedirs(GRID_PLOT_DIR, exist_ok=True)
os.makedirs(GRID_NPY_DIR, exist_ok=True)
os.makedirs(BIO_LATENT_PLOT_DIR, exist_ok=True)

# ============================================================
# 2. LOAD DATA (S, D, A)
# ============================================================
ROOT = "/scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium"
print("Loading data...", flush=True)
S_np = np.load(f"{ROOT}/coords_mat.npy")  # (N, 2)

A_raw = sparse.load_npz(f"{ROOT}/counts_mat_hvg5000.npz")

print(f"S shape = {S_np.shape}", flush=True)
print(f"Count matrix shape = {A_raw.shape}", flush=True)

G = A_raw.shape[1]

# ============================================================
# 3. DATASET
# ============================================================
class SpatialGeneDataset(Dataset):
    def __init__(self, gene_expr, spatial_coords):
        assert len(gene_expr) == len(spatial_coords)
        self.X = gene_expr
        self.S = spatial_coords

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "gene": self.X[idx],
            "spatial": self.S[idx],
        }

def iterate_minibatches(X, S, batch_size, shuffle=True):
    N = X.shape[0]
    idx = torch.randperm(N) if shuffle else torch.arange(N)

    for start in range(0, N, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], S[batch_idx]

# ============================================================
# 4. VAE MODELS
# ============================================================
class SpatialEncoder(nn.Module):
    def __init__(self, s_dim, d_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, d_dim)
        self.log_sigma = nn.Linear(hidden_dim, d_dim)

    def forward(self, s):
        h = self.net(s)
        return self.mu(h), torch.exp(self.log_sigma(h))
        

class BiologicalEncoder(nn.Module):
    """
    scvi.nn.Encoder for q_φ(z_i | a_i, b_i).
    Input: x (log-transformed raw counts).  Output: (mu, sigma) for z_i.
    """
    def __init__(self, g_dim, z_dim=10, n_layers=2, n_hidden=128,
                 dropout_rate=0.1):
        super().__init__()
        self.encoder = scviEncoder(
            n_input=g_dim, n_output=z_dim,
            n_layers=n_layers, n_hidden=n_hidden,
            dropout_rate=dropout_rate, distribution="normal",
        )

    def forward(self, x):
        q_m, q_v, _ = self.encoder(x)  # q_v is variance
        return q_m, torch.sqrt(q_v)    # return (mean, std)


class LibrarySizeEncoder(nn.Module):
    """
    scvi.nn.Encoder for q_τ(l_i | a_i, b_i).
    Input: x (log-transformed raw counts).  Output: (mu, sigma) in log library space.
    """
    def __init__(self, g_dim, n_layers=2, n_hidden=128,
                 dropout_rate=0.1):
        super().__init__()
        self.encoder = scviEncoder(
            n_input=g_dim, n_output=1,
            n_layers=n_layers, n_hidden=n_hidden,
            dropout_rate=dropout_rate, distribution="normal",
        )

    def forward(self, x):
        q_m, q_v, _ = self.encoder(x)  # q_v is variance
        return q_m, torch.sqrt(q_v)    # return (mean, std)


class CombinedDecoder(nn.Module):
    """
    Takes the concatenation [d_i, z_i] and
    outputs gene proportions ρ_i via softmax, so ρ_ig > 0 and
    Σ_g ρ_ig = 1.
    """
    def __init__(self, d_dim, z_dim, g_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_dim + z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.rho_logits = nn.Linear(hidden_dim, g_dim)

    def forward(self, d, z):
        h = self.net(torch.cat([d, z], dim=-1))
        rho = torch.softmax(self.rho_logits(h), dim=-1)  # sums to 1 across G genes
        return rho


class GASTON_scVI(nn.Module):
    """
    Combined GASTON-VI + scVI model.

    Latents:
        d_i ~ N(0,1)             
        z_i ~ N(0,I)                
        l_i ~ LogNormal(l_mu, l_sigma^2) 

    Decoder:
        ρ_i = f(d_i, z_i)
        a_ig ~ NB(l_i * ρ_ig, B_g)   via GammaPoisson
    """
    def __init__(self, g_dim, s_dim, d_dim=1, z_dim=10,
                 hidden_dim_spatial=20,
                 n_layers_bio=1, n_hidden_bio=128, n_layers_lib = 1, n_hidden_lib = 128,
                 dropout_rate=0.1,
                 hidden_dim_decoder=20,
                 log_lib_mean=0.0, log_lib_var=1.0,
                 beta=0.01):
        super().__init__()
        self.g_dim = g_dim
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.beta = beta

        # log-library prior parameters (empirical from data)
        self.register_buffer("log_lib_mean", torch.tensor(log_lib_mean, dtype=torch.float32))
        self.register_buffer("log_lib_var", torch.tensor(log_lib_var, dtype=torch.float32))

        # neural nets
        self.spatial_encoder = SpatialEncoder(s_dim, d_dim, hidden_dim_spatial)
        self.bio_encoder = BiologicalEncoder(g_dim, z_dim, n_layers_bio,
                                             n_hidden_bio, dropout_rate)
        self.lib_encoder = LibrarySizeEncoder(g_dim, n_layers_lib,
                                              n_hidden_lib, dropout_rate)
        self.decoder = CombinedDecoder(d_dim, z_dim, g_dim, hidden_dim_decoder)

    def model(self, x, s, beta=1e-20):
        pyro.module("decoder", self.decoder)

        theta_g = pyro.param("theta_g", x.new_ones((x.shape[1],)), constraint=dist.constraints.positive)

        with pyro.plate("data", x.shape[0]):
            # --- prior on d_i: N(0, 1) ---
            d_prior_mu = x.new_zeros((x.shape[0], self.d_dim))
            d_prior_sigma = x.new_ones((x.shape[0], self.d_dim))
            with pyro.poutine.scale(scale=self.beta):
                d = pyro.sample("d", dist.Normal(d_prior_mu, d_prior_sigma).to_event(1))

            # --- prior on z_i: N(0, I) ---
            z_prior_mu = x.new_zeros((x.shape[0], self.z_dim))
            z_prior_sigma = x.new_ones((x.shape[0], self.z_dim))
            z = pyro.sample("z", dist.Normal(z_prior_mu, z_prior_sigma).to_event(1))

            # --- prior on l_i: LogNormal(l_mu, l_sigma^2) ---
            l_prior_mu = self.log_lib_mean.expand(x.shape[0], 1)  # same for all cells
            l_prior_sigma = self.log_lib_var.sqrt().expand(x.shape[0], 1) # same for all cells
            log_l = pyro.sample("log_l", dist.Normal(l_prior_mu, l_prior_sigma).to_event(1))
            l = torch.exp(log_l)  # (N, 1)

            # --- decode ---
            rho = self.decoder(d, z)         # (N, G), sums to 1
            mu = l * rho                     # (N, G) mean counts

            # --- NB likelihood via GammaPoisson ---
            theta = theta_g.unsqueeze(0).expand_as(mu)  # (N, G)
            rate = theta / (mu + 1e-8)
            pyro.sample("obs", dist.GammaPoisson(concentration=theta, rate=rate).to_event(1), obs=x)

    def guide(self, x, s, beta=1e-20):
        pyro.module("spatial_encoder", self.spatial_encoder)
        pyro.module("bio_encoder", self.bio_encoder)
        pyro.module("lib_encoder", self.lib_encoder)

        x_log1p = torch.log1p(x)
        
        with pyro.plate("data", x.shape[0]):
            with pyro.poutine.scale(scale=beta):
                d_mu, d_sigma = self.spatial_encoder(s)
                d = pyro.sample("d", dist.Normal(d_mu, d_sigma).to_event(1))
            
            # --- q_φ(z_i | a_i) ---
            z_mu, z_sigma = self.bio_encoder(x_log1p)
            z = pyro.sample("z", dist.Normal(z_mu, z_sigma).to_event(1))

            # --- q_τ(l_i | a_i) ---
            l_mu, l_sigma = self.lib_encoder(x_log1p)
            pyro.sample("log_l", dist.Normal(l_mu, l_sigma).to_event(1))

    
    def encode_d_mu(self, s):
        """Return posterior mean of isodepth d_i given spatial coords."""
        d_mu, _ = self.spatial_encoder(s)
        return d_mu

    def encode_d_sigma(self, s):
        """Return posterior std of isodepth d_i."""
        _, d_sigma = self.spatial_encoder(s)
        return d_sigma

    def encode_z_mu(self, x):
        """Return posterior mean of biological latent z_i."""
        z_mu, _ = self.bio_encoder(torch.log1p(x))
        return z_mu

# ============================================================
# 5. EVAL UTILS
# ============================================================
def nb_log_prob(y, mu, theta, eps=1e-8):
    mu = torch.clamp(mu, min=eps)
    return (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y * (torch.log(mu) - torch.log(theta + mu))
    )


def nb_loss(model, X, S, batch_size, eps=1e-8, shuffle=False):
    """Average per-element NB NLL (evaluation metric)."""
    theta_g = pyro.get_param_store()["theta_g"].detach().to(X.device)

    total = 0.0
    denom = 0

    for x, s in iterate_minibatches(X, S, batch_size, shuffle=shuffle):
        x_log1p = torch.log1p(x)
        d_mu, _ = model.spatial_encoder(s)
        z_mu, _ = model.bio_encoder(x_log1p)
        l_mu, _ = model.lib_encoder(x_log1p)
        l = torch.exp(l_mu)  # (N, 1)

        rho = model.decoder(d_mu, z_mu)  # (N, G)
        mu = l * rho                      # (N, G)

        theta = theta_g.unsqueeze(0).expand_as(mu)
        logp = nb_log_prob(x, mu, theta, eps=eps)

        total += (-logp).sum().item()
        denom += x.numel()

    return total / max(1, denom)


def plotting(model, X, S_encoded, S, sigma, freq, rep, corr=None):
    print("Plotting isodepth plots", flush=True)
    model.eval()
    with torch.no_grad():
        mu_q = model.encode_d_mu(S_encoded).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))

    sc1 = ax.scatter(S[:, 0], S[:, 1], c=mu_q.squeeze(), cmap="viridis", s=5)
    CS = ax.tricontour(
        S[:, 0], S[:, 1],
        mu_q.squeeze(),
        levels=10,
        colors="black",
        linewidths=0.8,
        zorder=10
    )
    ax.clabel(CS, CS.levels, inline=True, fontsize=8)

    if corr is None:
        ax.set_title(f"Learned isodepth (σ = {sigma}, β = {beta})")
    else:
        ax.set_title(f"Learned isodepth (σ = {sigma}, β = {beta}) Corr={corr:.4f}")
    pad = 0.02
    xmin, xmax = S[:, 0].min(), S[:, 0].max()
    ymin, ymax = S[:, 1].min(), S[:, 1].max()
    ax.set_xlim(xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin))
    ax.set_ylim(ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin))
    plt.colorbar(sc1, ax=ax)

    plt.tight_layout()
    fpath = f"{ISODEPTH_PLOT_DIR}/results_scale_{sigma}_beta_{beta}_rep_{rep+1}.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    npy_path = f"{ISODEPTH_NPY_DIR}/isodepth_scale_{sigma}_beta_{beta}_rep_{rep+1}.npy"
    np.save(npy_path, mu_q.squeeze())
    print(f"Saved isodepth npy: {npy_path}", flush=True)


def plot_bio_latent(model, X, S, sigma, beta, rep, batch_size=4096):
    """Plot each dimension of the biological latent z_i in spatial coordinates."""
    print("Plotting bio latent dimensions", flush=True)
    model.eval()

    # encode in minibatches
    z_parts = []
    N = X.shape[0]
    with torch.no_grad():
        for start in range(0, N, batch_size):
            x_batch = X[start:start + batch_size]
            z_mu_batch = model.encode_z_mu(x_batch)
            z_parts.append(z_mu_batch.cpu().numpy())
    z_mu = np.concatenate(z_parts, axis=0)  # (N, z_dim)

    for dim in range(z_mu.shape[1]):
        fig, ax = plt.subplots(figsize=(6, 5))
        sc1 = ax.scatter(S[:, 0], S[:, 1], c=z_mu[:, dim], cmap="viridis", s=0.5)
        ax.set_title(f"Bio latent z dim {dim} (σ={sigma}, β={beta})")
        pad = 0.02
        xmin, xmax = S[:, 0].min(), S[:, 0].max()
        ymin, ymax = S[:, 1].min(), S[:, 1].max()
        ax.set_xlim(xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin))
        ax.set_ylim(ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin))
        plt.colorbar(sc1, ax=ax)
        plt.tight_layout()
        fpath = f"{BIO_LATENT_PLOT_DIR}/z_dim{dim}_scale_{sigma}_beta_{beta}_rep_{rep+1}.png"
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {z_mu.shape[1]} bio latent plots to {BIO_LATENT_PLOT_DIR}", flush=True)


def plot_loss_curve(loss_history, sigma, beta, data, rep=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_history, 'b-', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    label = f"GASTON-scVI {data}"
    if rep:
        ax.set_title(f"{label} Loss (σ={sigma}, β={beta}), rep={rep}")
    else:
        ax.set_title(f"{label} Loss (σ={sigma}, β={beta})")
    ax.grid(True, alpha=0.3)

    if rep:
        loss_path = f"{LOSS_PLOT_DIR}/{data}_loss_sigma{sigma}_beta{beta}_rep{rep}.png"
    else:
        loss_path = f"{LOSS_PLOT_DIR}/{data}_loss_sigma{sigma}_beta{beta}.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    return loss_path


def plot_variance_curve(variance_history, sigma, beta, data, rep=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(variance_history, 'b-', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean Sigma^2", fontsize=12)
    if rep:
        ax.set_title(f"VAE {data} Variance (No PE, β={beta}), rep={rep}")
    else:
        ax.set_title(f"VAE {data} Variance (No PE, β={beta})")
    ax.grid(True, alpha=0.3)

    if rep:
        var_path = f"{VARIANCE_PLOT_DIR}/{data}_variance_sigma{sigma}_beta{beta}_rep{rep}.png"
    else:
        var_path = f"{VARIANCE_PLOT_DIR}/{data}_variance_sigma{sigma}_beta{beta}.png"
    plt.savefig(var_path, dpi=150, bbox_inches="tight")
    plt.close()
    return var_path

# ============================================================
# 6. POSITIONAL ENCODING
# ============================================================
class LogFrequencyPositionalEncoding(nn.Module):
    def __init__(self, in_dim, num_frequencies, sigma):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.sigma = sigma
        j = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("freq_bands", 2 * math.pi * sigma ** (j / num_frequencies))
        print(f"CONFIRMATION: POSITIONAL ENCODING WITH SIGMA={sigma}, NUM_FREQS={num_frequencies}")

    def forward(self, s):
        fb = self.freq_bands.to(s.device)
        s_exp = s.unsqueeze(-2) * fb.view(1, -1, 1)
        sin = torch.sin(s_exp)
        cos = torch.cos(s_exp)
        encoded = torch.cat([sin, cos], dim=-1)
        encoded_flat = encoded.flatten(-2)
        return torch.cat([s, encoded_flat], dim=-1)

# ============================================================
# PREPARE DATA
# ============================================================
print("Preparing data tensors...", flush=True)

X_cpu = torch.tensor(A_raw.toarray(), dtype=torch.float32)

S_cpu = torch.tensor(S_np, dtype=torch.float32)

X_full = X_cpu.to(device)
S_full = S_cpu.to(device)

N = X_full.shape[0]

train_gene = X_full
train_spatial = S_full

eps = 1e-8
train_gene_mean = train_gene.mean(dim=0, keepdim=True)
train_gene_std = train_gene.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
train_spatial_mean = train_spatial.mean(dim=0, keepdim=True)
train_spatial_std = train_spatial.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)

train_spatial_norm = (train_spatial - train_spatial_mean) / train_spatial_std

encoder = LogFrequencyPositionalEncoding(in_dim=2, num_frequencies=num_freqs, sigma=sigma)
train_spatial_norm = encoder(train_spatial_norm)

print(f"train_spatial_norm.shape: {train_spatial_norm.shape}", flush=True)
s_dim = train_spatial_norm.shape[1]

# --- library-size prior stats (empirical from data) ---
lib_sizes = train_gene.sum(dim=1)      # sum all gene counts to get the total UMI count (library size)              
log_lib = torch.log(lib_sizes + 1e-8)  # log library size
log_lib_mean = log_lib.mean().item()   # empirical mean
log_lib_var = log_lib.var().item()    # empirical variance
print(f"Log-library size: mean={log_lib_mean:.3f}, var={log_lib_var:.3f}", flush=True)

# ============================================================
# TRAINING LOOP
# ============================================================
print("=== BEGIN VAE TRAINING ===", flush=True)
all_corr = []
all_train_loss = []
all_recon_train_loss = []
all_train_variance = []
BATCH_SIZE = args.batch_size

for rep in range(REPEATS):
    print(f"\nRepeat {rep+1}/{REPEATS}", flush=True)
    seed = 42
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    vae = GASTON_scVI(
        g_dim=G, s_dim=s_dim, d_dim=1, z_dim=z_bio_dim,
        hidden_dim_spatial=20,
        n_layers_bio=1, n_hidden_bio=128, n_layers_lib=1, n_hidden_lib=128,
        dropout_rate=0.1,
        hidden_dim_decoder=20,
        log_lib_mean=log_lib_mean, log_lib_var=log_lib_var,
        beta=beta,
    ).to(device)

    model_with_beta = lambda x, s: vae.model(x, s, beta=beta)
    guide_with_beta = lambda x, s: vae.guide(x, s, beta=beta)

    svi = SVI(model_with_beta, guide_with_beta, Adam({"lr": 1e-3}), loss=Trace_ELBO())

    train_loss_per_run = []
    recon_train_loss_per_run = []
    train_variance_per_run = []

    for epoch in range(EPOCHS):
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS}", flush=True)
        vae.train()
        epoch_loss = 0.0
        num = 0
        for gene, spatial in iterate_minibatches(train_gene, train_spatial_norm, batch_size=BATCH_SIZE, shuffle=True):
            batch_loss = svi.step(gene, spatial)
            epoch_loss += batch_loss
            num += gene.shape[0]
        avg_epoch_train_loss = epoch_loss / max(1, num)
        train_loss_per_run.append(avg_epoch_train_loss)

        vae.eval()
        with torch.no_grad():
            recon_train = nb_loss(vae, train_gene, train_spatial_norm, BATCH_SIZE)
        recon_train_loss_per_run.append(recon_train)

    plotting(vae, X_full, train_spatial_norm, S_np, sigma, num_freqs, rep)
    plot_bio_latent(vae, X_full, S_np, sigma, beta, rep, batch_size=BATCH_SIZE)

    all_recon_train_loss.append(recon_train_loss_per_run)
    all_train_loss.append(train_loss_per_run)

    print("=== plotting loss curves ===", flush=True)
    plot_loss_curve(recon_train_loss_per_run, sigma, beta, "train", rep + 1)
    plot_loss_curve(train_loss_per_run, sigma, beta, "ELBO", rep + 1)


all_train_loss = np.array(all_train_loss)
avg_train_loss = all_train_loss.mean(axis=0)

all_recon_train_loss = np.array(all_recon_train_loss)
avg_recon_train_loss = all_recon_train_loss.mean(axis=0)
std_recon_train_loss = all_recon_train_loss.std(axis=0)

plot_loss_curve(avg_recon_train_loss, sigma, beta, "train")

print("=== ALL VAE DONE ===", flush=True)

stats = []
stats.append({
    "scale": sigma,
    "num_freqs": num_freqs,
    "norm": norm,
    "beta": beta,
    "avg_recon_train_loss": avg_recon_train_loss[-1],
    "std_recon_train_loss": std_recon_train_loss[-1],
    "end_train_ELBO_loss": avg_train_loss[0]
})


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------
df = pd.DataFrame(stats)
csv_path = f"{RESULTS_DIR}/results_scale_{sigma}_freq_{num_freqs}_beta_{beta}.csv"
df.to_csv(csv_path, index=False)
print("Saved:", csv_path)