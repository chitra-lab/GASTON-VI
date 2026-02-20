#!/usr/bin/env python3
import sys, os, time
import numpy as np
import torch, torch.nn as nn
import pyro, pyro.distributions as dist
from torch.utils.data import Dataset, random_split, DataLoader
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from scipy.stats import spearmanr
import pandas as pd
import argparse
import math
import matplotlib.pyplot as plt
import wandb
from gaston import neural_net
from gaston.neural_net import GASTON
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import scanpy as sc

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
# 2. LOAD DATA (S, D, A)
# ============================================================
ROOT = "/scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium"
print("Loading data...", flush=True)
S_np = np.load(f"{ROOT}/coords_mat.npy")              # (N, 2)
A_raw = np.load(f"{ROOT}/counts_mat_hvg5000.npy")

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
            "gene": self.X[idx],     # [G]
            "spatial": self.S[idx],  # [s_dim]
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
        train_gene_norm = (x - train_gene_mean) / train_gene_std
        h = self.net(s)
        return self.mu(h), torch.exp(self.log_sigma(h))


class Decoder(nn.Module):
    """
    Gamma-Poisson (Negative Binomial) decoder.

    Outputs:
      - mu: (N, G) mean counts, positive
    """
    def __init__(self, z_dim, g_dim, hidden_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu_raw = nn.Linear(hidden_dim, g_dim)

    def forward(self, z):
        h = self.net(z)
        logits = self.mu_raw(h)
        p = torch.softmax(logits, dim=-1) +  1e-4
        return p


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
        # encoder choice
        if variant in ["A", "D"]:
            self.encoder = Encoder_q_z_given_xs(g_dim, s_dim, z_dim, hidden_dim)
        elif variant in ["B", "E"]:
            self.encoder = Encoder_q_z_given_x(g_dim, z_dim, hidden_dim)
        else:  # C, F
            self.encoder = Encoder_q_z_given_s(s_dim, z_dim, hidden_dim)

        self.decoder = Decoder(z_dim, g_dim, hidden_dim)
        self.z_dim = z_dim

    def model(self, x, s, beta=1e-20):
        pyro.module("decoder", self.decoder)

        # gene-specific dispersion (theta_g > 0)
        theta_g = pyro.param("theta_g", x.new_ones((x.shape[1],)), constraint=dist.constraints.positive)

        with pyro.plate("data", x.shape[0]):
            mu_p = x.new_zeros((x.shape[0], self.z_dim))
            sigma_p = x.new_ones((x.shape[0], self.z_dim))
            with pyro.poutine.scale(scale=beta):
                z = pyro.sample("z", dist.Normal(mu_p, sigma_p).to_event(1))

            # mu learned by decoder
            mu = self.decoder(z)   
            # broadcast theta across cells
            theta = theta_g.unsqueeze(0).expand_as(mu) # (N, G)
            # Gamma-Poisson parameters
            rate = theta / (mu + 1e-8)

            pyro.sample("obs", dist.GammaPoisson(concentration=theta, rate=rate).to_event(1), obs=x)

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
    """
    MSE loss computation
    """
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
        x_mu = model.decoder(mu_z)

        total_se += criterion(torch.log1p(x_mu), torch.log1p(gene)).item()
        total_entries += gene.numel()
        
    avg_mse = total_se / max(1, total_entries)

    return avg_mse


def nb_log_prob(y, mu, theta, eps=1e-8):
    mu = torch.clamp(mu, min=eps)
    #theta = torch.clamp(theta, min=eps)
    return (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y * (torch.log(mu) - torch.log(theta + mu))
    )

def nb_loss(model, X, S, batch_size, eps=1e-8, shuffle=False):
    theta_g = pyro.get_param_store()["theta_g"].detach().to(X.device)  # (G,)

    total = 0.0
    denom = 0

    for x, s in iterate_minibatches(X, S, batch_size, shuffle=shuffle):
        mu_z, _ = model.encoder(x, s)   
        mu = model.decoder(mu_z)        

        theta = theta_g.unsqueeze(0).expand_as(mu)    
        logp = nb_log_prob(x, mu, theta, eps=eps)   

        total += (-logp).sum().item()
        denom += x.numel()

    return total / max(1, denom)


def plotting(model, X, S_encoded, S, variant, sigma, freq, rep, corr=None):
    print("Plotting isodepth plots", flush=True)
    model.eval()
    with torch.no_grad():
        mu_q = model.encode_mu(X, S_encoded).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))

    sc1 = ax.scatter(S[:, 0], S[:, 1], c=mu_q.squeeze(), cmap="viridis", s=5)
    # ---- contour lines ----
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
        ax.set_title(f"Learned isodepth (σ = {sigma}, β = {beta})  Corr={corr:.4f}")
    pad = 0.02
    xmin, xmax = S[:,0].min(), S[:,0].max()
    ymin, ymax = S[:,1].min(), S[:,1].max()
    ax.set_xlim(xmin - pad*(xmax-xmin), xmax + pad*(xmax-xmin))
    ax.set_ylim(ymin - pad*(ymax-ymin), ymax + pad*(ymax-ymin))
    plt.colorbar(sc1, ax=ax)

    plt.tight_layout()
    outdir = "/scratch/uchitra1/projects/GASTON-VAE/sweep/discrete/isodepth_plots"
    os.makedirs(outdir, exist_ok=True)
    fpath = f"{outdir}/results_scale_{sigma}_beta_{beta}_rep_{rep+1}.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# LOSS PLOTTING FUNCTION
# ============================================================
def plot_loss_curve(loss_history, sigma, beta, data, rep=None):
    """Plot loss curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_history, 'b-', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    if rep: 
        ax.set_title(f"VAE NB {data} Loss (σ={sigma}, β={beta}), rep={rep}")
    else:
        ax.set_title(f"VAE NB {data} Loss (σ={sigma}, β={beta})")
    ax.grid(True, alpha=0.3)

    outdir = "/scratch/uchitra1/projects/GASTON-VAE/sweep/discrete/loss_plots"
    os.makedirs(outdir, exist_ok=True)
    if rep:
        loss_path = os.path.join(outdir, f"{data}_loss_sigma{sigma}_beta{beta}_rep{rep}.png")
    else:
        loss_path = os.path.join(outdir, f"{data}_loss_sigma{sigma}_beta{beta}.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    return loss_path


def plot_variance_curve(variance_history, sigma, beta, data, rep=None):
    """Plot variance curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(variance_history, 'b-', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean Sigma^2", fontsize=12)
    if rep: 
        ax.set_title(f"VAE {data} Variance (No PE, β={beta}), rep={rep}")
    else:
        ax.set_title(f"VAE {data} Variance (No PE, β={beta})")
    ax.grid(True, alpha=0.3)

    outdir = "/scratch/uchitra1/projects/GASTON-VAE/sweep/discrete/variance_plots"
    os.makedirs(outdir, exist_ok=True)
    if rep:
        var_path = os.path.join(outdir, f"{data}_variance_sigma{sigma}_beta{beta}_rep{rep}.png")
    else:
        var_path = os.path.join(outdir, f"{data}_variance_sigma{sigma}_beta{beta}.png")
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
        j = torch.arange(1, num_frequencies+1, dtype=torch.float32)
        self.register_buffer("freq_bands", 2 * math.pi * sigma ** (j / num_frequencies))
        print(f"CONFIRMATION: POSITIONAL ENCODING WITH SIGMA={sigma}, NUM_FREQS={num_frequencies}")

    def forward(self, s):
        # s: (N, in_dim)
        fb = self.freq_bands.to(s.device)      # (m,)
        s_exp = s.unsqueeze(-2) * fb.view(1, -1, 1)   # (N, m, in_dim)
        sin = torch.sin(s_exp)
        cos = torch.cos(s_exp)
        encoded = torch.cat([sin, cos], dim=-1)       # (N, m, 2*in_dim)
        encoded_flat = encoded.flatten(-2) # (N, 2*m*in_dim)
        return torch.cat([s, encoded_flat], dim=-1)

# ============================================================
# PREPARE DATA
# ============================================================
print("Preparing data tensors...", flush=True)
X_cpu = torch.tensor(A_raw, dtype=torch.float32)
S_cpu = torch.tensor(S_np, dtype=torch.float32)

X_full = X_cpu.to(device)
S_full = S_cpu.to(device)

N = X_full.shape[0]
train_size = int(0.8 * N)
val_size = N - train_size

all_idx = torch.randperm(N, generator=torch.Generator().manual_seed(42))
train_idx = all_idx[:train_size]
val_idx   = all_idx[train_size:]

# Get training data
train_gene = X_full[train_idx]
train_spatial = S_full[train_idx]

# Get validation data  
val_gene = X_full[val_idx]  
val_spatial = S_full[val_idx]

 # Compute training data statistics
eps = 1e-8
train_gene_mean = train_gene.mean(dim=0, keepdim=True)
train_gene_std  = train_gene.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
train_spatial_mean = train_spatial.mean(dim=0, keepdim=True)
train_spatial_std  = train_spatial.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)

# Transform (normalize) the training spatial data
train_spatial_norm = (train_spatial - train_spatial_mean) / train_spatial_std
    
# Transform (normalize) the validation spatial data
val_spatial_norm = (val_spatial - train_spatial_mean) / train_spatial_std

# PE transformation of spatial coordinates
encoder = LogFrequencyPositionalEncoding(in_dim=2, num_frequencies=num_freqs, sigma=sigma)
train_spatial_norm = encoder(train_spatial_norm)
val_spatial_norm = encoder(val_spatial_norm)

print(f"train_spatial_norm.shape: {train_spatial_norm.shape}", flush=True)
s_dim = train_spatial_norm.shape[1]   # dynamic: 2 * num_freqs * 2

# ============================================================
# TRAINING LOOP
# ============================================================
print("=== BEGIN VAE TRAINING ===", flush=True)
all_corr = []
all_train_loss = []
all_NB_train_loss = []
all_train_variance = []
BATCH_SIZE = 2586

for rep in range(REPEATS):
    print(f"\nRepeat {rep+1}/{REPEATS} for variant {variant}", flush=True)
    seed = 42
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    
    # Model
    vae = VAE(variant, g_dim=G, s_dim=s_dim, z_dim=1, hidden_dim=20).to(device)
    model_with_beta = lambda x, s: vae.model(x, s, beta=beta)
    guide_with_beta = lambda x, s: vae.guide(x, s, beta=beta)
    
    svi = SVI(model_with_beta, guide_with_beta, Adam({"lr": 1e-3}), loss=Trace_ELBO())

    #corr_per_run = []
    train_loss_per_run = []
    NB_train_loss_per_run = []
    NB_val_loss_per_run = []
    train_variance_per_run = []

    for epoch in range(EPOCHS):
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS}", flush=True)
        vae.train()
        epoch_loss = 0.0
        num = 0
        for gene, spatial in iterate_minibatches(train_gene, train_spatial_norm,
                                            batch_size=BATCH_SIZE, shuffle=True,):
            batch_loss = svi.step(gene, spatial)
            epoch_loss += batch_loss
            num += gene.shape[0]
        avg_epoch_train_loss = epoch_loss / max(1, num)
        train_loss_per_run.append(avg_epoch_train_loss)

        vae.eval()
        NB_train_loss = nb_loss(vae, train_gene, train_spatial_norm, BATCH_SIZE)

        #corr_per_run.append(corr)
        NB_train_loss_per_run.append(NB_train_loss)


    N = len(X_full)
    s_encoded_full = torch.zeros((N, s_dim), dtype=torch.float32).to(device)
    s_encoded_full[train_idx] = train_spatial_norm
    s_encoded_full[val_idx] = val_spatial_norm

    #corr = val_corr(vae, val_gene, val_spatial_norm, D_val)
    #all_corr.append(corr)
    
    plotting(vae, X_full, s_encoded_full, S_np, variant, sigma, num_freqs, rep)

    #all_corr.append(corr_per_run)
    
    all_NB_train_loss.append(NB_train_loss_per_run)   # for each epoch for each repeat (2D array) 
    all_train_loss.append(train_loss_per_run)

    plot_loss_curve(NB_train_loss_per_run, sigma, beta, "train", rep+1)
    plot_loss_curve(train_loss_per_run, sigma, beta, "ELBO", rep+1)


all_train_loss = np.array(all_train_loss)
avg_train_loss = all_train_loss.mean(axis=0)

all_NB_train_loss = np.array(all_NB_train_loss)
avg_NB_train_loss = all_NB_train_loss.mean(axis=0)
std_NB_train_loss = all_NB_train_loss.std(axis=0)

plot_loss_curve(avg_NB_train_loss, sigma, beta, "train")

# Average across repeats
#all_corr = np.array(all_corr)
#avg_corr = all_corr.mean(axis=0)
#std_corr = all_corr.std(axis=0)
#best_epoch = np.argmin(avg_val_loss)

#avg_corr = np.mean(all_corr)
#std_corr = np.std(all_corr)

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

# ---------------------------------------------------------------------
# Save + plot
# ---------------------------------------------------------------------
df = pd.DataFrame(stats)
outdir = "/scratch/uchitra1/projects/GASTON-VAE/sweep/discrete"
os.makedirs(outdir, exist_ok=True)
csv_path = f"{outdir}/results_scale_{sigma}_freq_{num_freqs}_beta_{beta}.csv"
df.to_csv(csv_path, index=False)
print("Saved:", csv_path)