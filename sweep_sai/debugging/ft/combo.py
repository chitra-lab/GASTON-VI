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
from gaston import neural_net
from gaston.neural_net import GASTON
from sklearn import preprocessing

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
parser.add_argument("--norm", type=str, required=True)  # z_score or minmax
parser.add_argument("--activation", type=str, required=True)  # relu or softmax
parser.add_argument("--beta", type=float, required=True)
args = parser.parse_args()

sigma = args.sigma
num_freqs = args.num_freqs
noise_level = float(args.noise_level)
variant = args.variant.upper()
EPOCHS = args.epochs
REPEATS = args.repeats
norm = args.norm
activation = args.activation.lower()
beta = args.beta

print(f"Running variant={variant}, sigma={sigma}, num_freqs={num_freqs}, beta={beta}, norm={norm}, activation={activation}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}", flush=True)

# ============================================================
# 0b. GET ACTIVATION FUNCTION
# ============================================================
def get_activation_fn():
    if activation == "relu":
        return nn.ReLU()
    elif activation == "softmax":
        return nn.Softmax(dim=-1)
    elif activation == "softplus":
        return nn.Softplus()
    else:
        raise ValueError(f"Unknown activation: {activation}")

act_fn = get_activation_fn()
print(f"Using activation: {activation}", flush=True)

# ============================================================
# 1. LOAD DATA (S, D, A)
# ============================================================
ROOT = "/scratch/uchitra1/projects/GASTON-VAE/sim_datasets/simulated_hex_dataset"

dataset_name = os.path.basename(os.path.normpath(ROOT))
print(f"Dataset detected: {dataset_name}", flush=True)

print("Loading S and D...", flush=True)
S_np = np.load(f"{ROOT}/S_60.npy")
D_np = np.load(f"{ROOT}/D_60.npy").squeeze()
print("Loaded S.npy and D.npy", flush=True)

A_path = f"{ROOT}/A_60_G6_sigma{noise_level:.1f}.npy"
print(f"Loading A from: {A_path}", flush=True)
A_raw = np.load(A_path)
print(f"NOISE LEVEL = {noise_level} | A_raw shape = {A_raw.shape}", flush=True)

G = A_raw.shape[1]

# ============================================================
# 2. DATASET
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

# ============================================================
# 3. VAE MODELS
# ============================================================
class Encoder_q_z_given_x(nn.Module):
    def __init__(self, g_dim, z_dim, activation_fn, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim, hidden_dim), activation_fn,
            nn.Linear(hidden_dim, hidden_dim), activation_fn
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, s=None):
        h = self.net(x)
        return self.mu(h), torch.exp(self.log_sigma(h))


class Encoder_q_z_given_xs(nn.Module):
    def __init__(self, g_dim, s_dim, z_dim, activation_fn, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim + s_dim, hidden_dim), activation_fn,
            nn.Linear(hidden_dim, hidden_dim), activation_fn
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, s):
        h = self.net(torch.cat([x, s], dim=1))
        return self.mu(h), torch.exp(self.log_sigma(h))


class Encoder_q_z_given_s(nn.Module):
    def __init__(self, s_dim, z_dim, activation_fn, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), activation_fn,
            nn.Linear(hidden_dim, hidden_dim), activation_fn
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, s):
        h = self.net(s)
        return self.mu(h), torch.exp(self.log_sigma(h))


class Decoder(nn.Module):
    def __init__(self, z_dim, g_dim, activation_fn, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), activation_fn,
            nn.Linear(hidden_dim, hidden_dim), activation_fn
        )
        self.mu = nn.Linear(hidden_dim, g_dim)
        self.sigma = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        h = self.net(z)
        return self.mu(h), torch.exp(self.sigma(h))


class SpatialPrior(nn.Module):
    def __init__(self, s_dim, z_dim, activation_fn, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), activation_fn,
            nn.Linear(hidden_dim, hidden_dim), activation_fn
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, s):
        h = self.net(s)
        return self.mu(h), torch.exp(self.sigma(h))


class VAE(nn.Module):
    def __init__(self, variant, g_dim, s_dim, activation_fn, z_dim=1, hidden_dim=32):
        super().__init__()
        self.variant = variant
        if variant in ["A", "D"]:
            self.encoder = Encoder_q_z_given_xs(g_dim, s_dim, z_dim, activation_fn, hidden_dim)
        elif variant in ["B", "E"]:
            self.encoder = Encoder_q_z_given_x(g_dim, z_dim, activation_fn, hidden_dim)
        else:
            self.encoder = Encoder_q_z_given_s(s_dim, z_dim, activation_fn, hidden_dim)

        self.learned_prior = variant in ["D", "E", "F"]
        if self.learned_prior:
            self.prior_net = SpatialPrior(s_dim, z_dim, activation_fn, hidden_dim)

        self.decoder = Decoder(z_dim, g_dim, activation_fn, hidden_dim)
        self.z_dim = z_dim

    def model(self, x, s, beta=1e-20):
        pyro.module("decoder", self.decoder)
        if self.learned_prior:
            pyro.module("prior_net", self.prior_net)
        with pyro.plate("data", x.shape[0]):
            if self.learned_prior:
                mu_p, sigma_p = self.prior_net(s)
            else:
                mu_p = x.new_zeros((x.shape[0], self.z_dim))
                sigma_p = x.new_ones((x.shape[0], self.z_dim))
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

# ============================================================
# 4. EVAL UTILS
# ============================================================
def evaluate(svi, test_loader):
    test_loss = 0.0
    for batch in test_loader:
        gene = batch["gene"].to(device)
        spatial = batch["spatial"].to(device)
        test_loss += svi.evaluate_loss(gene, spatial)
    return test_loss / len(test_loader.dataset)


def val_corr(model, val_loader, D_val):
    mu_vals = []
    with torch.no_grad():
        for batch in val_loader:
            gene = batch["gene"].to(device)
            spatial = batch["spatial"].to(device)
            mu = model.encode_mu(gene, spatial).cpu().numpy()
            mu_vals.append(mu)
    mu_val = np.concatenate(mu_vals).squeeze()
    corr, _ = spearmanr(D_val, mu_val)
    return abs(corr)
    
def train_one_run(vae, svi, train_loader, val_loader, D_val, EPOCHS):
    corr_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(EPOCHS):
        vae.train()
        epoch_loss = 0.0
        for batch in train_loader:
            gene = batch["gene"].to(device)
            spatial = batch["spatial"].to(device)
            epoch_loss += svi.step(gene, spatial)

        vae.eval()
        corr = val_corr(vae, val_loader, D_val)
        vloss = evaluate(svi, val_loader)

        corr_per_epoch.append(corr)
        val_loss_per_epoch.append(vloss)

    return np.array(corr_per_epoch), np.array(val_loss_per_epoch)


# ============================================================
# 5. POSITIONAL ENCODING
# ============================================================
class LogFrequencyPositionalEncoding(nn.Module):
    def __init__(self, in_dim, num_frequencies, sigma):
        super().__init__()
        j = torch.arange(1, num_frequencies+1, dtype=torch.float32)
        self.register_buffer("freq_bands", 2 * math.pi * sigma ** (j / num_frequencies))

    def forward(self, s):
        fb = self.freq_bands.to(s.device)
        s_exp = s.unsqueeze(-2) * fb.view(1, -1, 1)
        sin = torch.sin(s_exp)
        cos = torch.cos(s_exp)
        encoded = torch.cat([sin, cos], dim=-1)
        encoded_flat = encoded.flatten(-2)
        return torch.cat([s, encoded_flat], dim=-1)

# ============================================================
# 6. TRAINING
# ============================================================
X_full = torch.tensor(A_raw, dtype=torch.float32).to(device)
S_full = torch.tensor(S_np, dtype=torch.float32).to(device)

encoder = LogFrequencyPositionalEncoding(2, num_freqs, sigma)
s_encoded = encoder(S_full)
s_dim = s_encoded.shape[1]

BASE_RESULTS_DIR = "/scratch/uchitra1/projects/GASTON-VAE/sweep_sai/debugging/ft/results"
outdir = os.path.join(BASE_RESULTS_DIR, dataset_name)
plot_dir = os.path.join(outdir, "isodepth_plots")
os.makedirs(plot_dir, exist_ok=True)

# ============================================================
# 7. TRAIN + SAVE RESULTS (produces mean_corr/std_corr/best_loss)
# ============================================================
all_corr = []
all_val_loss = []

# split indices ONCE so repeats are comparable
N = len(X_full)
indices = np.arange(N)
rng = np.random.default_rng(42)
rng.shuffle(indices)

train_size = int(0.8 * N)
train_idx = indices[:train_size]
val_idx   = indices[train_size:]

D_val = D_np[val_idx]

# build raw train/val tensors
train_gene = X_full[train_idx]
train_spatial = s_encoded[train_idx]
val_gene = X_full[val_idx]
val_spatial = s_encoded[val_idx]

# normalize using TRAIN ONLY
if norm == "z_score":
    scaler_x = preprocessing.StandardScaler().fit(train_gene.detach().cpu().numpy())
    scaler_s = preprocessing.StandardScaler().fit(train_spatial.detach().cpu().numpy())

    train_gene_norm = torch.tensor(scaler_x.transform(train_gene.detach().cpu().numpy()),
                                   dtype=torch.float32).to(device)
    train_spatial_norm = torch.tensor(scaler_s.transform(train_spatial.detach().cpu().numpy()),
                                      dtype=torch.float32).to(device)

    val_gene_norm = torch.tensor(scaler_x.transform(val_gene.detach().cpu().numpy()),
                                 dtype=torch.float32).to(device)
    val_spatial_norm = torch.tensor(scaler_s.transform(val_spatial.detach().cpu().numpy()),
                                    dtype=torch.float32).to(device)
else:
    A = train_gene.detach().cpu().numpy()
    A_min = A.min(axis=0)
    A_max = A.max(axis=0)
    train_gene_norm = torch.tensor((A - A_min) / (A_max - A_min + 1e-12),
                                   dtype=torch.float32).to(device)

    S = train_spatial.detach().cpu().numpy()
    S_min = S.min(axis=0)
    S_max = S.max(axis=0)
    train_spatial_norm = torch.tensor((S - S_min) / (S_max - S_min + 1e-12),
                                      dtype=torch.float32).to(device)

    A_val = val_gene.detach().cpu().numpy()
    val_gene_norm = torch.tensor((A_val - A_min) / (A_max - A_min + 1e-12),
                                 dtype=torch.float32).to(device)

    S_val = val_spatial.detach().cpu().numpy()
    val_spatial_norm = torch.tensor((S_val - S_min) / (S_max - S_min + 1e-12),
                                    dtype=torch.float32).to(device)

# dataloaders (full-batch like your earlier script)
train_dataset = SpatialGeneDataset(train_gene_norm, train_spatial_norm)
val_dataset   = SpatialGeneDataset(val_gene_norm,   val_spatial_norm)

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=len(val_dataset),   shuffle=False)

for rep in range(REPEATS):
    seed = rep + int(noise_level * 1000)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    vae = VAE(variant, g_dim=G, s_dim=s_dim, activation_fn=act_fn, z_dim=1, hidden_dim=20).to(device)

    model_with_beta = lambda x, s: vae.model(x, s, beta=beta)
    guide_with_beta = lambda x, s: vae.guide(x, s, beta=beta)
    svi = SVI(model_with_beta, guide_with_beta, Adam({"lr": 1e-3}), loss=Trace_ELBO())

    corr_curve, loss_curve = train_one_run(vae, svi, train_loader, val_loader, D_val, EPOCHS)
    all_corr.append(corr_curve)
    all_val_loss.append(loss_curve)

all_corr = np.array(all_corr)          # (REPEATS, EPOCHS)
all_val_loss = np.array(all_val_loss)  # (REPEATS, EPOCHS)

avg_corr = all_corr.mean(axis=0)
std_corr = all_corr.std(axis=0)
avg_loss = all_val_loss.mean(axis=0)

best_loss = float(np.min(avg_loss))

# final epoch correlation (this matches your earlier pipeline)
final_mean_corr = float(avg_corr[-1])
final_std_corr  = float(std_corr[-1])

stats = [{
    "scale": float(sigma),
    "num_freqs": int(num_freqs),
    "norm": norm,
    "activation": activation,
    "beta": float(beta),
    "mean_corr": final_mean_corr,
    "std_corr": final_std_corr,
    "best_loss": best_loss,
}]

df_out = pd.DataFrame(stats)

csv_path = f"{outdir}/results_scale_{sigma}_freq_{num_freqs}_beta_{beta}_norm_{norm}_act_{activation}.csv"
df_out.to_csv(csv_path, index=False)
print("Saved:", csv_path, flush=True)
print(df_out.to_string(index=False), flush=True)
