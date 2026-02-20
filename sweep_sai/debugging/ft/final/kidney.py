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
from gaston import neural_net
from gaston.neural_net import GASTON
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import StandardScaler

# ============================================================
# 0. PARSE SWEEP ARGS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, required=True)
parser.add_argument("--num_freqs", type=int, required=True)
parser.add_argument("--noise_level", type=float, required=True)
parser.add_argument("--variant", type=str, required=True)  # A–F
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--repeats", type=int, default=15)
parser.add_argument("--norm", type=str, required=True)
parser.add_argument("--beta", type=float, required=True)
args = parser.parse_args()

sigma = args.sigma
num_freqs = args.num_freqs
noise_level = float(args.noise_level)
variant = args.variant.upper()
EPOCHS = args.epochs
REPEATS = args.repeats
norm = args.norm
beta = args.beta

print(f"Running variant={variant}, sigma={sigma}, num_freqs={num_freqs}, beta={beta}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}", flush=True)

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
BASE_DIR = "/scratch/uchitra1/projects/GASTON-VAE/sweep_sai/debugging/ft"
RESULTS_DIR = f"{BASE_DIR}/results/kidney"
ISODEPTH_PLOT_DIR = f"{RESULTS_DIR}/isodepth_plots"
ISODEPTH_NPY_DIR = f"{RESULTS_DIR}/isodepth_npy"
LOSS_PLOT_DIR = f"{RESULTS_DIR}/loss_plots"
VARIANCE_PLOT_DIR = f"{RESULTS_DIR}/variance_plots"
GRID_PLOT_DIR = f"{RESULTS_DIR}/grid/isodepth_plots"
GRID_NPY_DIR = f"{RESULTS_DIR}/grid/isodepth_npy"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ISODEPTH_PLOT_DIR, exist_ok=True)
os.makedirs(ISODEPTH_NPY_DIR, exist_ok=True)
os.makedirs(LOSS_PLOT_DIR, exist_ok=True)
os.makedirs(VARIANCE_PLOT_DIR, exist_ok=True)
os.makedirs(GRID_PLOT_DIR, exist_ok=True)
os.makedirs(GRID_NPY_DIR, exist_ok=True)

# ============================================================
# 2. LOAD DATA (S, D, A)
# ============================================================
ROOT = "/scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium"
print("Loading data...", flush=True)
S_np = np.load(f"{ROOT}/coords_mat.npy")  # (N, 2)
A_raw = np.load(f"{ROOT}/analytics_pearson.npy").astype(np.float32)

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
class Encoder_q_z_given_x(nn.Module):
    def __init__(self, g_dim, z_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, s=None):
        h = self.net(x)
        return self.mu(h), torch.exp(self.log_sigma(h))


class Encoder_q_z_given_xs(nn.Module):
    def __init__(self, g_dim, s_dim, z_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim + s_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, s):
        h = self.net(torch.cat([x, s], dim=1))
        return self.mu(h), torch.exp(self.log_sigma(h))


class Encoder_q_z_given_s(nn.Module):
    def __init__(self, s_dim, z_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, s):
        h = self.net(s)
        return self.mu(h), torch.exp(self.log_sigma(h))


class Decoder(nn.Module):
    def __init__(self, z_dim, g_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, g_dim)
        self.sigma = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        h = self.net(z)
        return self.mu(h), torch.exp(self.sigma(h))


class SpatialPrior(nn.Module):
    def __init__(self, s_dim, z_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, s):
        h = self.net(s)
        return self.mu(h), torch.exp(self.sigma(h))


class VAE(nn.Module):
    def __init__(self, variant, g_dim, s_dim, z_dim=1, hidden_dim=32):
        super().__init__()
        self.variant = variant
        if variant in ["A", "D"]:
            self.encoder = Encoder_q_z_given_xs(g_dim, s_dim, z_dim, hidden_dim)
        elif variant in ["B", "E"]:
            self.encoder = Encoder_q_z_given_x(g_dim, z_dim, hidden_dim)
        else:  # C, F
            self.encoder = Encoder_q_z_given_s(s_dim, z_dim, hidden_dim)
        self.learned_prior = True
        self.prior_net = SpatialPrior(s_dim, z_dim, hidden_dim)
        print("USING LEARNED PRIOR", flush=True)

        self.decoder = Decoder(z_dim, g_dim, hidden_dim)
        self.z_dim = z_dim

    def model(self, x, s, beta=1e-20):
        pyro.module("decoder", self.decoder)

        with pyro.plate("data", x.shape[0]):
            if self.learned_prior:
                mu_p, sigma_p = self.prior_net(s)
            with pyro.poutine.scale(scale=beta):
                z = pyro.sample("z", dist.Normal(mu_p, sigma_p).to_event(1))
            x_loc, x_scale = self.decoder(z)
            pyro.sample("obs", dist.Normal(x_loc, x_scale).to_event(1), obs=x)

    def guide(self, x, s, beta=1e-20):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            with pyro.poutine.scale(scale=beta):
                mu_q, sigma_q = self.encoder(x, s)
                pyro.sample("z", dist.Normal(mu_q, sigma_q).to_event(1))

    def encode_mu(self, x, s):
        mu_q, _ = self.encoder(x, s)
        return mu_q

    def encode_sigma(self, x, s):
        _, sigma_q = self.encoder(x, s)
        return sigma_q

# ============================================================
# 5. EVAL UTILS
# ============================================================
def val_corr(model, val_gene_norm, val_spatial_norm, d_true):
    mu_vals = []
    with torch.no_grad():
        mu = model.encode_mu(val_gene_norm, val_spatial_norm).cpu().numpy()
        mu_vals.append(mu)
    mu_val = np.concatenate(mu_vals).squeeze()
    corr, _ = spearmanr(d_true, mu_val)
    return abs(corr)


def mse_loss(model, X, S, batch_size):
    criterion = nn.MSELoss(reduction="sum")
    N = X.shape[0]
    idx = torch.arange(N)

    total_se = 0.0
    total_entries = 0

    for start in range(0, N, batch_size):
        b = idx[start:start + batch_size]
        gene = X[b]
        spatial = S[b]

        mu_z, sigma_z = model.encoder(gene, spatial)
        x_mu, _ = model.decoder(mu_z)

        total_se += criterion(torch.log1p(x_mu), torch.log1p(gene)).item()
        total_entries += gene.numel()

    avg_mse = total_se / max(1, total_entries)
    return avg_mse


def plotting(model, X, S_encoded, S, variant, sigma, freq, rep, corr=None):
    print("Plotting isodepth plots", flush=True)
    model.eval()
    with torch.no_grad():
        mu_q = model.encode_mu(X, S_encoded).detach().cpu().numpy()

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


def plot_loss_curve(loss_history, sigma, beta, data, rep=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_history, 'b-', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    if rep:
        ax.set_title(f"VAE NB {data} Loss (σ={sigma}, β={beta}), rep={rep}")
    else:
        ax.set_title(f"VAE NB {data} Loss (σ={sigma}, β={beta})")
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

G = A_raw.shape[1]

X_cpu = torch.tensor(A_raw, dtype=torch.float32)
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
train_gene = (train_gene - train_gene_mean) / train_gene_std

encoder = LogFrequencyPositionalEncoding(in_dim=2, num_frequencies=num_freqs, sigma=sigma)
train_spatial_norm = encoder(train_spatial_norm)

print(f"train_spatial_norm.shape: {train_spatial_norm.shape}", flush=True)
s_dim = train_spatial_norm.shape[1]

# ============================================================
# TRAINING LOOP
# ============================================================
print("=== BEGIN VAE TRAINING ===", flush=True)
all_corr = []
all_train_loss = []
all_NB_train_loss = []
all_train_variance = []
BATCH_SIZE = S_np.shape[0]

for rep in range(REPEATS):
    print(f"\nRepeat {rep+1}/{REPEATS} for variant {variant}", flush=True)
    seed = 42
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    vae = VAE(variant, g_dim=G, s_dim=s_dim, z_dim=1, hidden_dim=20).to(device)
    model_with_beta = lambda x, s: vae.model(x, s, beta=beta)
    guide_with_beta = lambda x, s: vae.guide(x, s, beta=beta)

    svi = SVI(model_with_beta, guide_with_beta, Adam({"lr": 1e-3}), loss=Trace_ELBO())

    train_loss_per_run = []
    NB_train_loss_per_run = []
    train_variance_per_run = []

    for epoch in range(EPOCHS):
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS}", flush=True)
        vae.train()
        epoch_loss = 0.0
        num = 0
        for gene, spatial in iterate_minibatches(train_gene, train_spatial_norm,
                                                  batch_size=BATCH_SIZE, shuffle=False):
            batch_loss = svi.step(gene, spatial)
            epoch_loss += batch_loss
            num += gene.shape[0]
        avg_epoch_train_loss = epoch_loss / max(1, num)
        train_loss_per_run.append(avg_epoch_train_loss)

        vae.eval()
        NB_train_loss = mse_loss(vae, train_gene, train_spatial_norm, BATCH_SIZE)
        NB_train_loss_per_run.append(NB_train_loss)

    plotting(vae, X_full, train_spatial_norm, S_np, variant, sigma, num_freqs, rep)

    all_NB_train_loss.append(NB_train_loss_per_run)
    all_train_loss.append(train_loss_per_run)

    plot_loss_curve(NB_train_loss_per_run, sigma, beta, "train", rep + 1)
    plot_loss_curve(train_loss_per_run, sigma, beta, "ELBO", rep + 1)


all_train_loss = np.array(all_train_loss)
avg_train_loss = all_train_loss.mean(axis=0)

all_NB_train_loss = np.array(all_NB_train_loss)
avg_NB_train_loss = all_NB_train_loss.mean(axis=0)
std_NB_train_loss = all_NB_train_loss.std(axis=0)

plot_loss_curve(avg_NB_train_loss, sigma, beta, "train")

print("=== ALL VAE DONE ===", flush=True)

stats = []
stats.append({
    "scale": sigma,
    "num_freqs": num_freqs,
    "norm": norm,
    "beta": beta,
    "avg_NB_train_loss": avg_NB_train_loss[-1],
    "std_NB_train_loss": std_NB_train_loss[-1],
    "end_train_ELBO_loss": avg_train_loss[0]
})


print(f"Generating gridpoints for FT", flush=True)
x_min, x_max = train_spatial[:, 0].min(), train_spatial[:, 0].max()
y_min, y_max = train_spatial[:, 1].min(), train_spatial[:, 1].max()
print(f"x_min = {x_min}, x_max = {x_max}, y_min={y_min}, y_max={y_max}")

W = int(np.sqrt(x_max.cpu() - x_min.cpu()))
H = int(np.sqrt(y_max.cpu() - y_min.cpu()))
print(f"num grid points = {W} x {H}", flush=True)


grid_x, grid_y = torch.meshgrid(
    torch.linspace(x_min, x_max, steps=W),
    torch.linspace(y_min, y_max, steps=H),
    indexing="ij"
)

grid_raw = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
grid = grid_raw.to(device)
grid_norm = (grid - train_spatial_mean) / train_spatial_std
grid_norm = encoder(grid_norm)
print(f"Shape of grid coordinates: {grid_norm.shape}", flush=True)

print(f"Evaluating isodepth at gridpoints", flush=True)
vae.eval()
with torch.no_grad():
    mu, logvar = vae.encoder(train_gene, grid_norm)
    d_grid = mu.reshape(W, H)

npy_path = f"{GRID_NPY_DIR}/isodepth_scale_{sigma}_beta_{beta}_rep_{rep+1}.npy"
np.save(npy_path, d_grid.cpu())
print(f"Saved isodepth npy: {npy_path}", flush=True)


fig, ax = plt.subplots(figsize=(6, 5))

grid_x_cpu = grid_x.cpu().numpy()
grid_y_cpu = grid_y.cpu().numpy()
d_grid_cpu = d_grid.cpu().numpy()

sc1 = ax.scatter(grid_x_cpu, grid_y_cpu, c=d_grid_cpu, cmap="viridis", s=5)
CS = ax.contour(
    grid_x_cpu, grid_y_cpu,
    d_grid_cpu,
    levels=10,
    colors="black",
    linewidths=0.8,
    zorder=10
)
ax.clabel(CS, CS.levels, inline=True, fontsize=8)

ax.set_title(f"Learned isodepth (σ = {sigma}, β = {beta})")

pad = 0.02
xmin, xmax = grid_x_cpu.min(), grid_x_cpu.max()
ymin, ymax = grid_y_cpu.min(), grid_y_cpu.max()
ax.set_xlim(xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin))
ax.set_ylim(ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin))
plt.colorbar(sc1, ax=ax)

plt.tight_layout()
fpath = f"{GRID_PLOT_DIR}/results_scale_{sigma}_beta_{beta}_rep_{rep+1}.png"
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------
df = pd.DataFrame(stats)
csv_path = f"{RESULTS_DIR}/results_scale_{sigma}_freq_{num_freqs}_beta_{beta}.csv"
df.to_csv(csv_path, index=False)
print("Saved:", csv_path)