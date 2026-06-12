#!/usr/bin/env python3
"""
seq_train_additive_batch.py
Variant of seq_train_additive.py that optionally conditions the scVI
encoders on a batch annotation.

New argument:
    --batch_file PATH
        Optional .npy of integer batch indices, shape (N,).  When provided,
        the BiologicalEncoder and LibrarySizeEncoder are built with
        n_cat_list=[n_batches] and receive the batch index at forward time.
        When omitted, behaviour is identical to seq_train_additive.py.
"""

import os, sys, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy import sparse
from scvi.nn import Encoder as scviEncoder


def set_seed(seed):
    """Seed python, numpy, and torch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 0.  ARGS
# ============================================================
parser = argparse.ArgumentParser(description="GASTON-VI sequential training (batch-aware)")

# -- data --
parser.add_argument("--coords_file", type=str, required=True)
parser.add_argument("--counts_file", type=str, required=True)
parser.add_argument("--batch_file",  type=str, default=None,
    help="Optional .npy of integer batch indices, shape (N,). "
         "If provigded, scVI encoders are conditioned on batch.")

# -- model --
parser.add_argument("--z_dim", type=int, default=0)
parser.add_argument("--n_isodepth", type=int, default=1)
parser.add_argument("--sigma", type=float, required=True)
parser.add_argument("--num_freqs", type=int, required=True)
parser.add_argument("--beta", type=float, required=True)
parser.add_argument("--hidden_dim_spatial", type=int, default=20)
parser.add_argument("--hidden_dim_decoder", type=int, default=20)

# -- training --
parser.add_argument("--warmup_epochs", type=int, default=50)
parser.add_argument("--epochs",     type=int,   default=175)
parser.add_argument("--repeats",    type=int,   default=1)
parser.add_argument("--batch_size", type=int,   required=True)
parser.add_argument("--val_frac",   type=float, default=0.3)

# -- output --
parser.add_argument("--results_dir", type=str, required=True)

# -- decoder type --
parser.add_argument("--decoder_type", type=str, default="combined",
    choices=["combined", "additive", "linear_hz"])
parser.add_argument("--gene_names_file", type=str, default=None)

# -- disentanglement --
parser.add_argument("--di_method", type=str, default="none",
    choices=["none", "simvi_mi", "simvi_mmd",
             "simvi_mi_kl", "simvi_mmd_kl"])
parser.add_argument("--di_weight", type=float, default=0.01)

# -- batch isodepth alignment --
parser.add_argument("--batch_align_weight", type=float, default=0.0,
    help="Weight on an RBF-MMD penalty that aligns the marginal distribution "
         "of the learned isodepth d across batches (encourages each batch's "
         "d to be on the same scale). 0 disables it. Only active when a "
         "batch_file with >=2 batches is provided. Gradients from this term "
         "flow to the spatial encoders only.")

# -- optimisation stability --
parser.add_argument("--grad_clip_norm", type=float, default=10.0,
    help="Max global L2 norm for gradient clipping, applied to the manual "
         "batch-MMD optimiser step on the spatial encoders (the Pyro SVI "
         "phases and the di step are left unclipped). Guards against NaN "
         "blow-ups when a large --batch_align_weight explodes the gradients. "
         "0 disables clipping.")

# -- misc --
parser.add_argument("--norm",        type=str,   default="z_score")
parser.add_argument("--noise_level", type=float, default=1.0)

args = parser.parse_args()

z_bio_dim          = args.z_dim
decoder_type       = args.decoder_type
n_isodepth         = args.n_isodepth
sigma              = args.sigma
num_freqs          = args.num_freqs
beta               = args.beta
hidden_dim_spatial = args.hidden_dim_spatial
hidden_dim_decoder = args.hidden_dim_decoder
WARMUP_EPOCHS      = args.warmup_epochs
EPOCHS         = args.epochs
REPEATS        = args.repeats
BATCH_SIZE     = args.batch_size
VAL_FRAC       = args.val_frac
GRAD_CLIP_NORM = args.grad_clip_norm
di_method      = args.di_method
di_weight      = args.di_weight
batch_align_weight = args.batch_align_weight

print(f"z_dim={z_bio_dim}, n_isodepth={n_isodepth}, sigma={sigma}, "
      f"num_freqs={num_freqs}, beta={beta}, "
      f"hidden_dim_spatial={hidden_dim_spatial}, "
      f"hidden_dim_decoder={hidden_dim_decoder}", flush=True)
print(f"warmup_epochs={WARMUP_EPOCHS}, total_epochs={EPOCHS}", flush=True)
print(f"di_method={di_method}, di_weight={di_weight}", flush=True)
print(f"batch_align_weight={batch_align_weight}", flush=True)
print(f"decoder_type={decoder_type}", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

# ============================================================
# 1.  OUTPUT DIRECTORIES
# ============================================================
RESULTS_DIR     = args.results_dir
di_tag          = f"di_{di_method}_w{di_weight}" if di_method != "none" else "di_none"
warmup_tag      = f"warmup{WARMUP_EPOCHS}"
epoch_tag       = f"ep{EPOCHS}"
dec_tag         = f"dec_{decoder_type}"
batch_tag       = "batch" if args.batch_file is not None else "nobatch"
hidden_tag      = f"hspat{hidden_dim_spatial}_hdec{hidden_dim_decoder}"
LOSS_PLOT_DIR   = f"{RESULTS_DIR}/{dec_tag}/loss_plots"
LATENT_DIR      = f"{RESULTS_DIR}/{dec_tag}/latents/zdim{z_bio_dim}_isodepth{n_isodepth}_{di_tag}_{warmup_tag}_{batch_tag}_{hidden_tag}"
LATENT_PLOT_DIR = f"{RESULTS_DIR}/{dec_tag}/latent_plots/zdim{z_bio_dim}_isodepth{n_isodepth}_{di_tag}_{warmup_tag}_{batch_tag}_{hidden_tag}"
MODEL_DIR       = f"{RESULTS_DIR}/{dec_tag}/models"

for d in [RESULTS_DIR, LOSS_PLOT_DIR, LATENT_DIR, LATENT_PLOT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 2.  LOAD DATA
# ============================================================
train_start = time.time()  # time from data loading through end of training
print(f"Loading coords from:  {args.coords_file}", flush=True)
print(f"Loading counts from:  {args.counts_file}", flush=True)

S_np  = np.load(args.coords_file)                                  # (N, 2)

print("=" * 80)
print("DEBUGGING COUNTS FILE")
print("cwd:", os.getcwd())
print("counts_file arg:", args.counts_file)
print("absolute path:", os.path.abspath(args.counts_file))
print("exists:", os.path.exists(args.counts_file))

if os.path.exists(args.counts_file):
    print("size on disk (MB):",
          os.path.getsize(args.counts_file) / 1024**2)

    arr = np.load(args.counts_file, mmap_mode="r")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)

print("=" * 80)

A_raw = sparse.csr_matrix(np.load(args.counts_file).astype(np.float32))
print(f"S shape            = {S_np.shape}", flush=True)
print(f"Count matrix shape = {A_raw.shape}", flush=True)
G = A_raw.shape[1]

# -- batch annotation (optional) --
if args.batch_file is not None:
    print(f"Loading batch from:   {args.batch_file}", flush=True)
    batch_np = np.load(args.batch_file).astype(np.int64)
    assert batch_np.shape[0] == A_raw.shape[0] and (
        batch_np.ndim == 1 or (batch_np.ndim == 2 and batch_np.shape[1] == 1)
    ), (f"batch shape {batch_np.shape} != (N,) or (N,1) with N={A_raw.shape[0]}")
    n_batches = int(batch_np.max()) + 1
    print(f"Loaded batch vector: N={batch_np.shape[0]}, n_batches={n_batches}",
          flush=True)
else:
    batch_np  = None
    n_batches = 0
    print("No batch vector provided (encoders will not be batch-conditioned)",
          flush=True)

# Load gene names
if args.gene_names_file is not None:
    if args.gene_names_file.endswith(".npy"):
        gene_names = [str(g) for g in np.load(args.gene_names_file, allow_pickle=True)]
    else:
        try:
            df = pd.read_csv(args.gene_names_file)
            gene_names = (df["gene"] if "gene" in df.columns else df.iloc[:, 0]).tolist()
        except Exception:
            with open(args.gene_names_file) as f:
                gene_names = [line.strip() for line in f if line.strip()]
    assert len(gene_names) == G, (
        f"Gene name count ({len(gene_names)}) != count matrix columns ({G})")
    print(f"Loaded {len(gene_names)} gene names", flush=True)
else:
    gene_names = [f"gene_{i}" for i in range(G)]
    print("No gene names provided, using gene_0, gene_1, ...", flush=True)

# ============================================================
# 3.  HELPERS
# ============================================================
def iterate_minibatches(X, S, batch_size, B=None, shuffle=True):
    """Yields (x, s, b) tuples; b is None if no batch vector is provided."""
    N   = X.shape[0]
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    for start in range(0, N, batch_size):
        bi = idx[start:start + batch_size]
        yield X[bi], S[bi], (B[bi] if B is not None else None)


def nb_log_prob(y, mu, theta, eps=1e-8):
    mu = torch.clamp(mu, min=eps)
    return (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y     * (torch.log(mu)    - torch.log(theta + mu))
    )


def nb_loss(model, X, S, batch_size, B=None, eps=1e-8):
    """Average per-element NB NLL (evaluation / reconstruction metric)."""
    theta_g = pyro.get_param_store()["theta_g"].detach().to(X.device)
    total, denom = 0.0, 0
    for x, s, b in iterate_minibatches(X, S, batch_size, B=B, shuffle=False):
        x_log1p = torch.log1p(x)
        d_mu    = model.encode_d_mu(s, b)
        z_mu    = (x.new_zeros(x.shape[0], 0) if model.z_dim == 0
                   else model.bio_encoder(x_log1p, b)[0])
        l_mu, _ = model.lib_encoder(x_log1p, b)
        l       = torch.exp(l_mu)
        rho     = model.decoder(d_mu, z_mu, b)
        mu      = l * rho
        theta   = theta_g.unsqueeze(0).expand_as(mu)
        logp    = nb_log_prob(x, mu, theta, eps=eps)
        total  += (-logp).sum().item()
        denom  += x.numel()
    return total / max(1, denom)


@torch.no_grad()
def extract_latents(model, X, S, batch_size, B=None):
    buckets = {k: [] for k in (
        ["d_mu", "d_sigma"] +
        ([f"d_{k}_mu" for k in range(model.n_isodepth)] +
         [f"d_{k}_sigma" for k in range(model.n_isodepth)]
         if model.n_isodepth > 1 else []) +
        ["z_mu", "z_sigma", "l_mu", "l_sigma"]
    )}

    model.eval()
    for x, s, b in iterate_minibatches(X, S, batch_size, B=B, shuffle=False):
        x_log1p = torch.log1p(x)

        if model.n_isodepth == 1:
            d_mu, d_sigma = model.spatial_encoder(s, b)
            buckets["d_mu"].append(d_mu.cpu().numpy())
            buckets["d_sigma"].append(d_sigma.cpu().numpy())
        else:
            dk_mus, dk_sigmas = [], []
            for k, enc in enumerate(model.spatial_encoders):
                d_mu_k, d_sigma_k = enc(s, b)
                buckets[f"d_{k}_mu"].append(d_mu_k.cpu().numpy())
                buckets[f"d_{k}_sigma"].append(d_sigma_k.cpu().numpy())
                dk_mus.append(d_mu_k)
                dk_sigmas.append(d_sigma_k)
            buckets["d_mu"].append(torch.cat(dk_mus,    dim=-1).cpu().numpy())
            buckets["d_sigma"].append(torch.cat(dk_sigmas, dim=-1).cpu().numpy())

        if model.z_dim > 0:
            z_mu, z_sigma = model.bio_encoder(x_log1p, b)
            buckets["z_mu"].append(z_mu.cpu().numpy())
            buckets["z_sigma"].append(z_sigma.cpu().numpy())
        else:
            buckets["z_mu"].append(np.zeros((x.shape[0], 0), dtype=np.float32))
            buckets["z_sigma"].append(np.zeros((x.shape[0], 0), dtype=np.float32))

        l_mu, l_sigma = model.lib_encoder(x_log1p, b)
        buckets["l_mu"].append(l_mu.cpu().numpy())
        buckets["l_sigma"].append(l_sigma.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in buckets.items()}


@torch.no_grad()
def print_per_batch_isodepth_scale(model, S, B, batch_size, tag=""):
    """Print mean/std/min/max/range of the learned isodepth d_mu per batch,
    per isodepth axis. Lets you eyeball whether the batches share a common
    scale (the cross-batch MMD alignment is meant to make these match).
    """
    if B is None or model.n_batches < 1:
        return
    model.eval()
    b_flat = B.view(-1)
    dmu_parts = []
    for start in range(0, S.shape[0], batch_size):
        s = S[start:start + batch_size]
        b = B[start:start + batch_size]
        dmu_parts.append(model.encode_d_mu(s, b))
    d_mu = torch.cat(dmu_parts, dim=0)            # (N, n_isodepth)

    header = "  [isodepth scale" + (f" — {tag}" if tag else "") + "] per-batch d stats:"
    print(header, flush=True)
    for bb in range(model.n_batches):
        mask = (b_flat == bb)
        if mask.sum().item() == 0:
            continue
        d_b = d_mu[mask]
        for k in range(model.n_isodepth):
            col = d_b[:, k]
            print(f"    batch {bb} d_{k}: mean={col.mean().item():+.4f} "
                  f"std={col.std().item():.4f} "
                  f"min={col.min().item():+.4f} max={col.max().item():+.4f} "
                  f"range={(col.max() - col.min()).item():.4f}", flush=True)

# ============================================================
# 3b. DISENTANGLEMENT REGULARISATION
# ============================================================
def _closed_form_mi(z_mu, z_sigma, d_mu, d_sigma):
    z = z_mu + z_sigma * torch.randn_like(z_mu)
    d = d_mu + d_sigma * torch.randn_like(d_mu)
    zd = torch.cat([z, d], dim=-1)
    B  = zd.shape[0]
    zd_c   = zd - zd.mean(0, keepdim=True)
    cov_zd = (zd_c.T @ zd_c) / max(B - 1, 1) + 1e-6 * torch.eye(
        zd.shape[1], device=zd.device)
    dz = z.shape[1]
    cov_z = cov_zd[:dz, :dz]
    cov_d = cov_zd[dz:, dz:]
    logdet_z  = torch.linalg.slogdet(cov_z)[1]
    logdet_d  = torch.linalg.slogdet(cov_d)[1]
    logdet_zd = torch.linalg.slogdet(cov_zd)[1]
    mi = 0.5 * (logdet_z + logdet_d - logdet_zd)
    return torch.clamp(mi, min=0.0)


def _kernel_mmd(z, d, kernel_bandwidth=1.0):
    zd_joint = torch.cat([z, d], dim=-1)
    idx = torch.randperm(d.shape[0], device=d.device)
    zd_prod = torch.cat([z, d[idx]], dim=-1)

    def _gaussian_kernel(x, y, bw):
        dist_sq = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
        return torch.exp(-dist_sq / (2.0 * bw ** 2)).mean()

    bw = kernel_bandwidth
    mmd_sq = (
        _gaussian_kernel(zd_joint, zd_joint, bw)
        + _gaussian_kernel(zd_prod,  zd_prod,  bw)
        - 2.0 * _gaussian_kernel(zd_joint, zd_prod, bw)
    )
    return torch.clamp(mmd_sq, min=0.0)


def _get_d_sigma(model, spatial_batch, batch_idx=None):
    if model.n_isodepth == 1:
        _, d_sigma = model.spatial_encoder(spatial_batch, batch_idx)
    else:
        d_sigma = torch.cat(
            [enc(spatial_batch, batch_idx)[1] for enc in model.spatial_encoders],
            dim=-1)
    return d_sigma


def _kl_to_standard_normal(mu, sigma):
    return 0.5 * (mu.pow(2) + sigma.pow(2) - 2 * sigma.log() - 1).sum(dim=-1)


def compute_disentanglement_reg(method, model, gene_batch, spatial_batch,
                                batch_idx=None):
    if method == "none" or model.z_dim == 0:
        return torch.tensor(0.0, device=gene_batch.device)

    x_log1p = torch.log1p(gene_batch)
    z_mu, z_sigma = model.bio_encoder(x_log1p, batch_idx)
    # only regularize z
    d_mu          = model.encode_d_mu(spatial_batch, batch_idx).detach()
    d_sigma       = _get_d_sigma(model, spatial_batch, batch_idx).detach()

    if method == "simvi_mi":
        return _closed_form_mi(z_mu, z_sigma, d_mu, d_sigma)

    elif method == "simvi_mmd":
        z = z_mu + z_sigma * torch.randn_like(z_mu)
        d = d_mu + d_sigma * torch.randn_like(d_mu)
        return _kernel_mmd(z, d)

    elif method == "simvi_mi_kl":
        mi = _closed_form_mi(z_mu, z_sigma, d_mu, d_sigma)
        kl = _kl_to_standard_normal(z_mu, z_sigma).mean()
        return mi + kl

    elif method == "simvi_mmd_kl":
        z = z_mu + z_sigma * torch.randn_like(z_mu)
        d = d_mu + d_sigma * torch.randn_like(d_mu)
        mmd = _kernel_mmd(z, d)
        kl  = _kl_to_standard_normal(z_mu, z_sigma).mean()
        return mmd + kl

    raise ValueError(f"Unknown di_method: {method}")


# ============================================================
# 3c. CROSS-BATCH ISODEPTH ALIGNMENT (MMD)
# ============================================================
def _two_sample_mmd(a, b, bandwidths=(0.5, 1.0, 2.0, 4.0)):
    """Multi-scale RBF MMD^2 between two sample sets a (n, D) and b (m, D).

    Uses a sum of Gaussian kernels at several fixed bandwidths (robust to the
    unknown scale of d). Returns a non-negative scalar; 0 when the two
    empirical distributions coincide.
    """
    def _rbf_mean(x, y):
        d2 = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
        k = 0.0
        for bw in bandwidths:
            k = k + torch.exp(-d2 / (2.0 * bw ** 2))
        return k.mean()

    mmd_sq = _rbf_mean(a, a) + _rbf_mean(b, b) - 2.0 * _rbf_mean(a, b)
    return torch.clamp(mmd_sq, min=0.0)


def _spatial_encoder_params(model):
    """All parameters of the (per-batch) spatial encoders only."""
    params = []
    if model.n_isodepth == 1:
        params += list(model.spatial_encoder.parameters())
    else:
        for enc in model.spatial_encoders:
            params += list(enc.parameters())
    return params


def compute_batch_d_mmd(model, spatial_batch, batch_idx, min_per_batch=2):
    """Mean pairwise RBF-MMD between the isodepth distributions of the batches
    present in this minibatch.

    Each spot's d is produced by its own batch's spatial encoder
    (model.encode_d_mu routes per batch), so minimising this term pulls every
    batch's marginal distribution of d onto a common scale. Gradients flow into
    all spatial encoders; the caller restricts the optimiser to those params so
    nothing else is touched.
    """
    if model.n_batches < 2 or batch_idx is None:
        return spatial_batch.new_tensor(0.0)

    b_flat = batch_idx.view(-1)
    d_mu   = model.encode_d_mu(spatial_batch, batch_idx)   # (B, n_isodepth)

    groups = []
    for b in range(model.n_batches):
        mask = (b_flat == b)
        if mask.sum().item() >= min_per_batch:
            groups.append(d_mu[mask])

    if len(groups) < 2:
        return spatial_batch.new_tensor(0.0)

    total, n_pairs = spatial_batch.new_tensor(0.0), 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            total = total + _two_sample_mmd(groups[i], groups[j])
            n_pairs += 1
    return total / max(1, n_pairs)


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
    def __init__(self, s_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu        = nn.Linear(hidden_dim, 1)
        self.log_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, s, batch=None):
        if torch.isnan(s).any():
            print("NaN in input s", flush=True)
        if torch.isinf(s).any():
            print("Inf in input s", flush=True)

        h = self.net(s)

        for name, p in self.named_parameters():
            if torch.isnan(p).any():
                print("NaN parameter:", name, flush=True)
            if torch.isinf(p).any():
                print("Inf parameter:", name, flush=True)

        if torch.isnan(h).any():
            print("NaN after hidden layer", flush=True)

        mu = self.mu(h)
        if torch.isnan(mu).any():
            print("NaN after mu layer", flush=True)

        log_sigma = self.log_sigma(h)
        if torch.isnan(log_sigma).any():
            print("NaN after log_sigma layer", flush=True)

        # return mu, torch.exp(log_sigma)
        return mu, F.softplus(log_sigma) + 1e-4


class BatchedSpatialEncoder(nn.Module):
    """One independent SpatialEncoder per batch, shared output space.

    Each spot is routed through the encoder for its batch label. All
    encoders emit into a single isodepth axis, so the downstream decoder
    sees one continuous d regardless of which batch produced it.
    """
    def __init__(self, s_dim, hidden_dim, n_batches):
        super().__init__()
        self.n_batches = n_batches
        self.encoders = nn.ModuleList([
            SpatialEncoder(s_dim, hidden_dim) for _ in range(n_batches)
        ])

    def forward(self, s, batch=None):
        if batch is None:
            return self.encoders[0](s)
        b_flat = batch.view(-1)
        mu    = s.new_empty(s.shape[0], 1)
        sigma = s.new_empty(s.shape[0], 1)
        for b in range(self.n_batches):
            mask = (b_flat == b)
            if mask.any():
                m_b, s_b = self.encoders[b](s[mask])
                mu[mask]    = m_b
                sigma[mask] = s_b
        return mu, sigma


class BiologicalEncoder(nn.Module):
    """scVI encoder: gene expression (log1p) → biological latent z_i.

    When n_batches>0 the encoder is built with n_cat_list=[n_batches] so it
    takes a batch index at forward time.  When n_batches=0 the encoder is
    unconditional and batch indices passed to forward() are ignored.
    """
    def __init__(self, g_dim, z_dim, n_layers=2, n_hidden=128,
                 dropout_rate=0.1, n_batches=0):
        super().__init__()
        self.n_batches = n_batches
        n_cat_list = [n_batches] if n_batches > 0 else None
        self.encoder = scviEncoder(
            n_input=g_dim, n_output=z_dim,
            n_cat_list=n_cat_list,
            n_layers=n_layers, n_hidden=n_hidden,
            dropout_rate=dropout_rate, distribution="normal",
        )

    def forward(self, x, batch=None):
        if self.n_batches > 0 and batch is not None:
            q_m, q_v, _ = self.encoder(x, batch)
        else:
            q_m, q_v, _ = self.encoder(x)
        return q_m, torch.sqrt(q_v)


class LibrarySizeEncoder(nn.Module):
    """scVI encoder: gene expression (log1p) → log library size l_i.

    Optionally batch-conditioned when n_batches>0.
    """
    def __init__(self, g_dim, n_layers=2, n_hidden=128,
                 dropout_rate=0.1, n_batches=0):
        super().__init__()
        self.n_batches = n_batches
        n_cat_list = [n_batches] if n_batches > 0 else None
        self.encoder = scviEncoder(
            n_input=g_dim, n_output=1,
            n_cat_list=n_cat_list,
            n_layers=n_layers, n_hidden=n_hidden,
            dropout_rate=dropout_rate, distribution="normal",
        )

    def forward(self, x, batch=None):
        if self.n_batches > 0 and batch is not None:
            q_m, q_v, _ = self.encoder(x, batch)
        else:
            q_m, q_v, _ = self.encoder(x)
        return q_m, torch.sqrt(q_v)


class CombinedDecoder(nn.Module):
    def __init__(self, d_dim, z_dim, g_dim, hidden_dim=128, n_batches=0):
        super().__init__()
        self.z_dim = z_dim
        self.n_batches = n_batches
        in_dim = d_dim + z_dim + n_batches
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.rho_logits = nn.Linear(hidden_dim, g_dim)

    def forward(self, d, z, batch=None):
        parts = [d]
        if self.z_dim > 0:
            if z.shape[-1] == 0:
                z = d.new_zeros(d.shape[0], self.z_dim)
            parts.append(z)
        if self.n_batches > 0:
            if batch is None:
                b_oh = d.new_zeros(d.shape[0], self.n_batches)
            else:
                b_oh = F.one_hot(batch.view(-1).long(),
                                 num_classes=self.n_batches).to(d.dtype)
            parts.append(b_oh)
        inp = torch.cat(parts, dim=-1)
        return torch.softmax(self.rho_logits(self.net(inp)), dim=-1)


class AdditiveDecoder(nn.Module):
    """ρ = softmax( f(d) + h([z, onehot(batch)]) ).

    Batch is concatenated only to h's (biological) input; f(d) stays pure
    spatial.  h_logits / h_proportions take an optional batch argument.
    """
    def __init__(self, d_dim, z_dim, g_dim, hidden_dim=128, n_batches=0):
        super().__init__()
        self.z_dim = z_dim
        self.n_batches = n_batches

        self.f_net = nn.Sequential(
            nn.Linear(d_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.f_out = nn.Linear(hidden_dim, g_dim)

        if z_dim > 0:
            self.h_net = nn.Sequential(
                nn.Linear(z_dim + n_batches, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            )
            self.h_out = nn.Linear(hidden_dim, g_dim)

    def _z_input(self, z, batch):
        if self.n_batches == 0:
            return z
        if batch is None:
            b_oh = z.new_zeros(z.shape[0], self.n_batches)
        else:
            b_oh = F.one_hot(batch.view(-1).long(),
                             num_classes=self.n_batches).to(z.dtype)
        return torch.cat([z, b_oh], dim=-1)

    def forward(self, d, z, batch=None):
        logits = self.f_out(self.f_net(d))
        if self.z_dim > 0 and z.shape[-1] > 0:
            z_in = self._z_input(z, batch)
            logits = logits + self.h_out(self.h_net(z_in))
        return torch.softmax(logits, dim=-1)

    def f_logits(self, d):
        return self.f_out(self.f_net(d))

    def h_logits(self, z, batch=None):
        z_in = self._z_input(z, batch)
        return self.h_out(self.h_net(z_in))

    def f_proportions(self, d):
        return torch.softmax(self.f_logits(d), dim=-1)


class LinearBiologicalDecoder(nn.Module):
    """ρ = softmax( g(d) + W·[z, onehot(batch)] ).

    Batch is concatenated only to the linear h branch's input; g(d) stays
    pure spatial.  linear_z_importance inspects W[:, :z_dim] (the z columns).
    """
    def __init__(self, d_dim, z_dim, g_dim, hidden_dim=128, n_batches=0):
        super().__init__()
        self.z_dim = z_dim
        self.n_batches = n_batches

        self.f_net = nn.Sequential(
            nn.Linear(d_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.f_out = nn.Linear(hidden_dim, g_dim)
        self.h_linear = (nn.Linear(z_dim + n_batches, g_dim, bias=False)
                        if z_dim > 0 else None)

    def _z_input(self, z, batch):
        if self.n_batches == 0:
            return z
        if batch is None:
            b_oh = z.new_zeros(z.shape[0], self.n_batches)
        else:
            b_oh = F.one_hot(batch.view(-1).long(),
                             num_classes=self.n_batches).to(z.dtype)
        return torch.cat([z, b_oh], dim=-1)

    def forward(self, d, z, batch=None):
        logits = self.f_out(self.f_net(d))
        if self.h_linear is not None and z.shape[-1] > 0:
            z_in = self._z_input(z, batch)
            logits = logits + self.h_linear(z_in)
        return torch.softmax(logits, dim=-1)

    def f_logits(self, d):
        return self.f_out(self.f_net(d))

    def h_logits(self, z, batch=None):
        z_in = self._z_input(z, batch)
        return self.h_linear(z_in)

    def f_proportions(self, d):
        return torch.softmax(self.f_logits(d), dim=-1)


def linear_z_importance(decoder, gene_names=None, save_path=None):
    # h_linear has shape (G, z_dim + n_batches); slice off the batch columns
    W_full = decoder.h_linear.weight.detach().cpu().numpy()
    W = W_full[:, :decoder.z_dim]
    importance = np.linalg.norm(W, axis=0)
    ranked = np.argsort(importance)[::-1]
    print("\n=== Linear h(z) latent importance (||W[:,k]||_2) ===")
    for rank, k in enumerate(ranked):
        print(f"  rank {rank+1:2d}  z_{k}  importance={importance[k]:.4f}")
        if gene_names is not None:
            top5 = np.argsort(np.abs(W[:, k]))[::-1][:5]
            print(f"    top genes: {[gene_names[g] for g in top5]}")
    if save_path:
        fig, ax = plt.subplots(figsize=(max(6, len(importance) * 0.6 + 2), 4))
        ax.bar(range(len(importance)), importance, edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([f"z_{k}" for k in range(len(importance))],
                           rotation=45, ha="right")
        ax.set_ylabel("||W[:,k]||_2"); ax.set_title("Linear h(z) latent importance")
        ax.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return importance

# ============================================================
# 6.  FULL MODEL
# ============================================================
class GASTON_VI(nn.Module):
    def __init__(self, g_dim, s_dim,
                 n_isodepth=1, z_dim=0,
                 hidden_dim_spatial=20,
                 n_layers_bio=2, n_hidden_bio=128,
                 n_layers_lib=2, n_hidden_lib=128,
                 dropout_rate=0.1,
                 hidden_dim_decoder=20,
                 log_lib_mean=0.0, log_lib_var=1.0,
                 beta=0.01,
                 decoder_type="combined",
                 n_batches=0):
        super().__init__()
        self.g_dim      = g_dim
        self.n_isodepth = n_isodepth
        self.z_dim      = z_dim
        self.beta       = beta
        self.n_batches  = n_batches

        self.register_buffer("log_lib_mean",
                             torch.tensor(log_lib_mean, dtype=torch.float32))
        self.register_buffer("log_lib_var",
                             torch.tensor(log_lib_var,  dtype=torch.float32))

        def _make_spatial_encoder():
            if n_batches > 0:
                return BatchedSpatialEncoder(s_dim, hidden_dim_spatial, n_batches)
            return SpatialEncoder(s_dim, hidden_dim_spatial)

        if n_isodepth == 1:
            self.spatial_encoder = _make_spatial_encoder()
        else:
            self.spatial_encoders = nn.ModuleList([
                _make_spatial_encoder() for _ in range(n_isodepth)
            ])
            print(f"n_isodepth={n_isodepth}: {n_isodepth} independent spatial encoders",
                  flush=True)
        if n_batches > 0:
            print(f"Per-batch spatial encoders: {n_batches} encoders per isodepth axis",
                  flush=True)

        if z_dim > 0:
            self.bio_encoder = BiologicalEncoder(g_dim, z_dim, n_layers_bio,
                                                 n_hidden_bio, dropout_rate,
                                                 n_batches=n_batches)
        else:
            self.bio_encoder = None
            print("z_dim=0: no BiologicalEncoder (spatial-only baseline)", flush=True)

        self.lib_encoder = LibrarySizeEncoder(g_dim, n_layers_lib,
                                              n_hidden_lib, dropout_rate,
                                              n_batches=n_batches)

        if decoder_type == "additive":
            self.decoder = AdditiveDecoder(n_isodepth, z_dim, g_dim,
                                           hidden_dim_decoder, n_batches=n_batches)
        elif decoder_type == "linear_hz":
            self.decoder = LinearBiologicalDecoder(n_isodepth, z_dim, g_dim,
                                                    hidden_dim_decoder, n_batches=n_batches)
        else:
            self.decoder = CombinedDecoder(n_isodepth, z_dim, g_dim,
                                           hidden_dim_decoder, n_batches=n_batches)
        print(f"Decoder: {decoder_type}", flush=True)
        if n_batches > 0:
            print(f"scVI encoders batch-conditioned: n_batches={n_batches}", flush=True)

    def encode_d_mu(self, s, b=None):
        if self.n_isodepth == 1:
            return self.spatial_encoder(s, b)[0]
        return torch.cat([enc(s, b)[0] for enc in self.spatial_encoders], dim=-1)

    def model(self, x, s, b=None, beta=None, phase1=False):
        if beta is None:
            beta = self.beta
        pyro.module("decoder",      self.decoder)
        pyro.module("lib_encoder",  self.lib_encoder)
        if self.z_dim > 0 and not phase1:
            pyro.module("bio_encoder", self.bio_encoder)

        theta_g = pyro.param("theta_g", x.new_ones(x.shape[1]),
                             constraint=dist.constraints.positive)

        with pyro.plate("data", x.shape[0]):
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
                d = torch.cat(d_parts, dim=-1)

            if self.z_dim > 0 and not phase1:
                z = pyro.sample("z",
                    dist.Normal(x.new_zeros(x.shape[0], self.z_dim),
                                x.new_ones( x.shape[0], self.z_dim)).to_event(1))
            else:
                z = x.new_zeros(x.shape[0], 0)

            if self.n_batches > 0:
                b_flat        = b.view(-1).long()
                l_prior_mu    = self.log_lib_mean[b_flat].view(-1, 1)
                l_prior_sigma = self.log_lib_var[b_flat].sqrt().view(-1, 1)
            else:
                l_prior_mu    = self.log_lib_mean.expand(x.shape[0], 1)
                l_prior_sigma = self.log_lib_var.sqrt().expand(x.shape[0], 1)
            log_l = pyro.sample("log_l",
                dist.Normal(l_prior_mu, l_prior_sigma).to_event(1))
            l = torch.exp(log_l)

            rho   = self.decoder(d, z, b)
            mu    = l * rho
            theta = theta_g.unsqueeze(0).expand_as(mu)
            rate  = theta / (mu + 1e-8)
            pyro.sample("obs",
                dist.GammaPoisson(concentration=theta, rate=rate).to_event(1),
                obs=x)

    def guide(self, x, s, b=None, beta=None, phase1=False):
        if beta is None:
            beta = self.beta
        pyro.module("lib_encoder", self.lib_encoder)
        if self.z_dim > 0 and not phase1:
            pyro.module("bio_encoder", self.bio_encoder)

        x_log1p = torch.log1p(x)

        with pyro.plate("data", x.shape[0]):
            if self.n_isodepth == 1:
                pyro.module("spatial_encoder", self.spatial_encoder)
                with pyro.poutine.scale(scale=beta):
                    d_mu, d_sigma = self.spatial_encoder(s, b)
                    pyro.sample("d", dist.Normal(d_mu, d_sigma).to_event(1))
            else:
                pyro.module("spatial_encoders", self.spatial_encoders)
                with pyro.poutine.scale(scale=beta):
                    for k, enc in enumerate(self.spatial_encoders):
                        d_mu_k, d_sigma_k = enc(s, b)
                        pyro.sample(f"d_{k}",
                                    dist.Normal(d_mu_k, d_sigma_k).to_event(1))

            if self.z_dim > 0 and not phase1:
                z_mu, z_sigma = self.bio_encoder(x_log1p, b)
                pyro.sample("z", dist.Normal(z_mu, z_sigma).to_event(1))

            l_mu, l_sigma = self.lib_encoder(x_log1p, b)
            pyro.sample("log_l", dist.Normal(l_mu, l_sigma).to_event(1))

# ============================================================
# 7.  PLOTTING HELPERS
# ============================================================
def plot_isodepth(S_xy, d_mu, save_path, title="Learned Isodepth",
                  edge_thresh_mult=3.0):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(S_xy[:, 0], S_xy[:, 1],
                    c=d_mu, cmap="viridis",
                    s=0.6, linewidths=0, rasterized=True)

    triang = mtri.Triangulation(S_xy[:, 0], S_xy[:, 1])
    tri_pts = S_xy[triang.triangles]
    edges = np.stack([
        tri_pts[:, 1] - tri_pts[:, 0],
        tri_pts[:, 2] - tri_pts[:, 1],
        tri_pts[:, 0] - tri_pts[:, 2],
    ], axis=1)
    max_edge = np.linalg.norm(edges, axis=2).max(axis=1)
    triang.set_mask(max_edge > edge_thresh_mult * np.median(max_edge))

    ax.tricontour(triang,
                  np.asarray(d_mu).squeeze(),
                  levels=10, colors="black",
                  linewidths=0.8, zorder=10)
    plt.colorbar(sc, ax=ax, label="isodepth")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_isodepth_per_batch(S_xy, d_mu, batch, save_dir, filename_prefix,
                             title_prefix="Isodepth"):
    """Plot isodepth separately for each batch.

    Subsets spots by batch label and calls plot_isodepth on each subset
    independently so contours are drawn within a single sample's tissue.
    """
    batch = np.asarray(batch).reshape(-1)
    for b in np.unique(batch):
        mask = (batch == b)
        if mask.sum() < 3:
            continue
        plot_isodepth(
            S_xy[mask], np.asarray(d_mu)[mask],
            save_path=f"{save_dir}/{filename_prefix}_batch{int(b)}.png",
            title=f"{title_prefix} — batch {int(b)}",
        )


def plot_all_isodepths(S_xy, d_mu_full, save_path, title_prefix="Isodepth"):
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
                        s=0.6, linewidths=0, rasterized=True)
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


def plot_z_latents_per_batch(S_xy, z_mu, batch, save_dir, filename_prefix,
                              title_prefix="z latents"):
    """Plot all z dimensions in one multi-panel figure per batch."""
    z_dim = z_mu.shape[1]
    if z_dim == 0:
        return
    batch = np.asarray(batch).reshape(-1)
    for b in np.unique(batch):
        mask = (batch == b)
        if mask.sum() < 3:
            continue
        S_b = S_xy[mask]
        z_b = z_mu[mask]

        ncols = min(4, z_dim)
        nrows = math.ceil(z_dim / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(4 * ncols, 3.5 * nrows),
                                  squeeze=False)
        axes_flat = axes.flatten()
        for i in range(z_dim):
            ax = axes_flat[i]
            sc = ax.scatter(S_b[:, 0], S_b[:, 1],
                            c=z_b[:, i], cmap="viridis",
                            s=1.0, linewidths=0, rasterized=True)
            plt.colorbar(sc, ax=ax)
            ax.set_title(f"z_{i}", fontsize=9)
            ax.set_aspect("equal"); ax.axis("off")
        for j in range(z_dim, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{title_prefix} — batch {int(b)}", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{filename_prefix}_batch{int(b)}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()


def plot_loss_curve(train_history, val_history, save_path, title="Loss",
                    warmup_epochs=0):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_history, linewidth=2, color="steelblue", label="train recon loss")
    if val_history is not None:
        ax.plot(val_history, linewidth=2, color="tomato",
                linestyle="--", label="val recon loss")
    if warmup_epochs > 0:
        ax.axvline(x=warmup_epochs, color="gray", linestyle=":", linewidth=1.5,
                   label=f"phase 2 start (epoch {warmup_epochs})")
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
B_full = (torch.tensor(batch_np, dtype=torch.long).to(device)
          if batch_np is not None else None)

N = X_full.shape[0]
set_seed(42)  # seed python/numpy/torch for the train/val split

if VAL_FRAC > 0:
    if B_full is not None:
        b_flat = B_full.view(-1)
        train_parts, val_parts = [], []
        # loop over each batch label
        for lb in torch.unique(b_flat).tolist():
            idx_b   = torch.where(b_flat == lb)[0]  # get indices belonging to this batch 
            idx_b   = idx_b[torch.randperm(len(idx_b), device=idx_b.device)]  # shuffle batch indices
            n_val_b = int(len(idx_b) * VAL_FRAC)  # validation size for this batch
            val_parts.append(idx_b[:n_val_b])
            train_parts.append(idx_b[n_val_b:])
        # merge all batches together
        val_idx   = torch.cat(val_parts)
        train_idx = torch.cat(train_parts)
        val_idx   = val_idx[torch.randperm(len(val_idx),     device=val_idx.device)]
        train_idx = train_idx[torch.randperm(len(train_idx), device=train_idx.device)]
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)} "
              f"({VAL_FRAC*100:.0f}%, stratified by batch)", flush=True)
        for lb in torch.unique(b_flat).tolist():
            n_t = (B_full[train_idx].view(-1) == int(lb)).sum().item()
            n_v = (B_full[val_idx].view(-1)   == int(lb)).sum().item()
            print(f"  batch {int(lb)}: train={n_t}, val={n_v}", flush=True)
    else:
        perm      = torch.randperm(N)
        n_val     = max(1, int(N * VAL_FRAC))
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)} ({VAL_FRAC*100:.0f}%)",
              flush=True)
    val_gene    = X_full[val_idx]
    val_spatial = S_full[val_idx]
    val_batch   = B_full[val_idx] if B_full is not None else None
else:
    train_idx = torch.randperm(N)
    val_idx   = None
    print("No validation split (val_frac=0)", flush=True)
    val_gene = val_spatial = val_batch = None

train_gene    = X_full[train_idx]
train_spatial = S_full[train_idx]
train_batch   = B_full[train_idx] if B_full is not None else None

eps = 1e-8

if B_full is not None:
    s_dim_raw = train_spatial.shape[1]
    train_spatial_mean = torch.zeros(n_batches, s_dim_raw,
                                     dtype=train_spatial.dtype,
                                     device=train_spatial.device)
    train_spatial_std  = torch.ones (n_batches, s_dim_raw,
                                     dtype=train_spatial.dtype,
                                     device=train_spatial.device)
    tb_flat = train_batch.view(-1)
    for b in range(n_batches):
        # compute batch-specific training mean/std
        mask = (tb_flat == b)
        if mask.sum().item() > 1:
            train_spatial_mean[b] = train_spatial[mask].mean(dim=0)
            train_spatial_std [b] = train_spatial[mask].std(
                dim=0, unbiased=False).clamp_min(eps)
        else:
            train_spatial_mean[b] = train_spatial.mean(dim=0)
            train_spatial_std [b] = train_spatial.std(dim=0, unbiased=False).clamp_min(eps)
            print(f"  batch {b}: <2 train spots, using global spatial stats",
                  flush=True)
    print("Per-batch spatial normalization:", flush=True)
    for b in range(n_batches):
        m = train_spatial_mean[b].cpu().tolist()
        s = train_spatial_std [b].cpu().tolist()
        m_str = "[" + ", ".join(f"{v:.2f}" for v in m) + "]"
        s_str = "[" + ", ".join(f"{v:.2f}" for v in s) + "]"
        print(f"  batch {b}: mean={m_str}, std={s_str}", flush=True)

    tb_idx = train_batch.view(-1).long()
    sb_idx = B_full.view(-1).long()
    # normalize spatial coordinates
    train_spatial_norm = (train_spatial - train_spatial_mean[tb_idx]) \
                         / train_spatial_std[tb_idx]
    S_full_norm        = (S_full        - train_spatial_mean[sb_idx]) \
                         / train_spatial_std[sb_idx]
else:
    train_spatial_mean = train_spatial.mean(dim=0, keepdim=True)
    train_spatial_std  = train_spatial.std(dim=0, unbiased=False,
                                            keepdim=True).clamp_min(eps)
    train_spatial_norm = (train_spatial - train_spatial_mean) / train_spatial_std
    S_full_norm        = (S_full        - train_spatial_mean) / train_spatial_std

# PE encoding
pe_encoder = LogFrequencyPositionalEncoding(in_dim=2,num_frequencies=num_freqs,sigma=sigma)
train_spatial_norm = pe_encoder(train_spatial_norm)
S_full_norm        = pe_encoder(S_full_norm)

if val_idx is not None:
    if B_full is not None:
        vb_idx = val_batch.view(-1).long()
        val_spatial_norm = pe_encoder((val_spatial - train_spatial_mean[vb_idx]) / train_spatial_std[vb_idx])
    else:
        val_spatial_norm = pe_encoder((val_spatial - train_spatial_mean)/ train_spatial_std)
else:
    val_spatial_norm = None

s_dim = train_spatial_norm.shape[1]
print(f"Encoded spatial dim: {s_dim}", flush=True)

lib_sizes = train_gene.sum(dim=1)
log_lib   = torch.log(lib_sizes + eps)

if B_full is not None:
    tb_flat     = train_batch.view(-1)
    global_mean = log_lib.mean().item()
    global_var  = log_lib.var().item() if log_lib.numel() > 1 else 1.0
    log_lib_mean = []
    log_lib_var  = []
    for b in range(n_batches):
        mask = (tb_flat == b)
        if mask.sum().item() > 1:
            # Compute batch-specific prior
            ll_b = log_lib[mask]
            log_lib_mean.append(ll_b.mean().item())
            log_lib_var.append(ll_b.var().item())
        else:
            log_lib_mean.append(global_mean)
            log_lib_var.append(global_var)
            print(f"  batch {b}: <2 train spots, falling back to global log-lib prior",
                  flush=True)
    print("Per-batch log-library priors:", flush=True)
    for b in range(n_batches):
        print(f"  batch {b}: mean={log_lib_mean[b]:.3f}, var={log_lib_var[b]:.3f}",
              flush=True)
else:
    log_lib_mean = log_lib.mean().item()
    log_lib_var  = log_lib.var().item()
    print(f"Log-library size: mean={log_lib_mean:.3f}, var={log_lib_var:.3f}",
          flush=True)

# ============================================================
# 9.  TRAINING LOOP  (two-phase sequential)
# ============================================================
print(f"=== BEGIN TRAINING (z_dim={z_bio_dim}, n_isodepth={n_isodepth}, "
      f"di_method={di_method}, n_batches={n_batches}) ===", flush=True)
print(f"Phase 1: epochs 0–{WARMUP_EPOCHS-1}  (spatial-only, bio_encoder frozen)", flush=True)
print(f"Phase 2: epochs {WARMUP_EPOCHS}–{EPOCHS-1}  (joint, di_method={di_method})", flush=True)

all_recon_train_loss = []
all_recon_val_loss   = []
all_elbo_loss        = []
all_di_reg_loss      = []
all_batch_mmd_loss   = []

for rep in range(REPEATS):
    seed = 42 + rep
    print(f"\nRepeat {rep+1}/{REPEATS}  (seed={seed})", flush=True)
    rep_start = time.time()
    set_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    vae = GASTON_VI(
        g_dim=G, s_dim=s_dim,
        n_isodepth=n_isodepth, z_dim=z_bio_dim,
        hidden_dim_spatial=hidden_dim_spatial,
        n_layers_bio=2, n_hidden_bio=128,
        n_layers_lib=2, n_hidden_lib=128,
        dropout_rate=0.1,
        hidden_dim_decoder=hidden_dim_decoder,
        log_lib_mean=log_lib_mean, log_lib_var=log_lib_var,
        beta=beta,
        decoder_type=decoder_type,
        n_batches=n_batches,
    ).to(device)

    svi_phase1 = SVI(
        lambda x, s, b: vae.model(x, s, b, phase1=True),
        lambda x, s, b: vae.guide(x, s, b, phase1=True),
        Adam({"lr": 1e-3}), loss=Trace_ELBO()
    )
    svi_phase2 = SVI(
        lambda x, s, b: vae.model(x, s, b),
        lambda x, s, b: vae.guide(x, s, b),
        Adam({"lr": 1e-3}), loss=Trace_ELBO()
    )

    di_optimizer = None
    if di_method != "none" and z_bio_dim > 0 and di_weight != 0.0:
        di_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # Cross-batch isodepth alignment: optimiser restricted to spatial encoders
    batch_mmd_optimizer = None
    spatial_params = None
    if batch_align_weight != 0.0 and n_batches >= 2:
        spatial_params = _spatial_encoder_params(vae)
        batch_mmd_optimizer = torch.optim.Adam(spatial_params, lr=1e-3)
        print(f"Cross-batch isodepth MMD alignment ON "
              f"(weight={batch_align_weight}, spatial encoders only)", flush=True)

    recon_train_history = []
    recon_val_history   = []
    elbo_history        = []
    di_reg_history      = []
    batch_mmd_history   = []

    for epoch in range(EPOCHS):
        phase1 = (epoch < WARMUP_EPOCHS)

        if epoch == 0:
            print(f"  [Phase 1] Epoch {epoch}/{EPOCHS} — spatial-only warmup", flush=True)
        elif epoch == WARMUP_EPOCHS:
            print(f"  [Phase 2] Epoch {epoch}/{EPOCHS} — joint training begins", flush=True)
        elif epoch % 25 == 0:
            phase_label = "Phase 1" if phase1 else "Phase 2"
            print(f"  [{phase_label}] Epoch {epoch}/{EPOCHS}", flush=True)

        vae.train()
        epoch_loss, num = 0.0, 0
        epoch_di_reg = 0.0
        epoch_batch_mmd = 0.0

        for gene, spatial, batch in iterate_minibatches(
                train_gene, train_spatial_norm, batch_size=BATCH_SIZE,
                B=train_batch, shuffle=True):
            if phase1:
                epoch_loss += svi_phase1.step(gene, spatial, batch)
            else:
                epoch_loss += svi_phase2.step(gene, spatial, batch)

                if di_optimizer is not None:
                    reg = compute_disentanglement_reg(
                        di_method, vae, gene, spatial, batch_idx=batch)
                    di_optimizer.zero_grad()
                    (di_weight * reg).backward()
                    di_optimizer.step()
                    epoch_di_reg += reg.item() * gene.shape[0]

            # Cross-batch isodepth alignment — separate step, spatial encoders
            # only. Runs in BOTH phases since the isodepth scale is set during
            # the phase-1 spatial warmup.
            if batch_mmd_optimizer is not None:
                mmd = compute_batch_d_mmd(vae, spatial, batch)
                batch_mmd_optimizer.zero_grad()
                (batch_align_weight * mmd).backward()
                mmd_grad_norm = torch.norm(torch.stack([
                    p.grad.detach().norm() for p in spatial_params
                    if p.grad is not None]))
                if num == 0 or not torch.isfinite(mmd_grad_norm):
                    print(f"  [mmd grad norm | epoch {epoch}] "
                          f"{mmd_grad_norm.item():.4e}", flush=True)
                if GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        spatial_params, GRAD_CLIP_NORM)
                batch_mmd_optimizer.step()
                epoch_batch_mmd += mmd.item() * gene.shape[0]

            num += gene.shape[0]

        elbo_history.append(epoch_loss / max(1, num))
        di_reg_history.append(epoch_di_reg / max(1, num))
        batch_mmd_history.append(epoch_batch_mmd / max(1, num))

        vae.eval()
        with torch.no_grad():
            recon_train = nb_loss(vae, train_gene, train_spatial_norm,
                                  BATCH_SIZE, B=train_batch)
            recon_train_history.append(recon_train)
            if val_idx is not None:
                recon_val = nb_loss(vae, val_gene, val_spatial_norm,
                                    BATCH_SIZE, B=val_batch)
                recon_val_history.append(recon_val)

        if epoch % 25 == 0:
            msg = f"  recon_train={recon_train:.5f}"
            if val_idx is not None:
                msg += f"  recon_val={recon_val:.5f}"
            if not phase1 and di_method != "none" and z_bio_dim > 0:
                msg += f"  di_reg={di_reg_history[-1]:.5f}"
            if batch_mmd_optimizer is not None:
                msg += f"  batch_mmd={batch_mmd_history[-1]:.5f}"
            print(msg, flush=True)
            if B_full is not None and n_batches >= 2:
                print_per_batch_isodepth_scale(
                    vae, S_full_norm, B_full, BATCH_SIZE, tag=f"epoch {epoch}")

    train_elapsed = time.time() - train_start
    print(f"Repeat {rep+1}/{REPEATS} training time: {train_elapsed:.2f} s "
          f"({train_elapsed/60:.2f} min)", flush=True)

    all_recon_train_loss.append(recon_train_history)
    all_elbo_loss.append(elbo_history)
    all_di_reg_loss.append(di_reg_history)
    all_batch_mmd_loss.append(batch_mmd_history)
    if val_idx is not None:
        all_recon_val_loss.append(recon_val_history)

    plot_loss_curve(
        recon_train_history,
        recon_val_history if val_idx is not None else None,
        save_path=f"{LOSS_PLOT_DIR}/recon_zdim{z_bio_dim}_isodepth{n_isodepth}_{dec_tag}_{di_tag}_{warmup_tag}_{epoch_tag}_{batch_tag}_{hidden_tag}_rep{rep+1}.png",
        title=f"Recon Loss — z_dim={z_bio_dim}, n_isodepth={n_isodepth}, {dec_tag}, {di_tag}, {warmup_tag}, {epoch_tag}, {batch_tag}, {hidden_tag}, rep={rep+1}",
        warmup_epochs=WARMUP_EPOCHS,
    )

    model_path = f"{MODEL_DIR}/model_zdim{z_bio_dim}_isodepth{n_isodepth}_{dec_tag}_{di_tag}_{warmup_tag}_{batch_tag}_{hidden_tag}_rep{rep+1}.pt"
    save_dict = {
        "state_dict":  vae.state_dict(),
        "pyro_params": {k: v.detach().cpu() for k, v in
                        pyro.get_param_store().items()},
        "model_kwargs": {
            "g_dim": G, "s_dim": s_dim,
            "n_isodepth": n_isodepth, "z_dim": z_bio_dim,
            "hidden_dim_spatial": hidden_dim_spatial,
            "n_layers_bio": 2, "n_hidden_bio": 128,
            "n_layers_lib": 2, "n_hidden_lib": 128,
            "dropout_rate": 0.1, "hidden_dim_decoder": hidden_dim_decoder,
            "log_lib_mean": log_lib_mean, "log_lib_var": log_lib_var,
            "beta": beta,
            "decoder_type": decoder_type,
            "n_batches": n_batches,
        },
        "sigma": sigma, "num_freqs": num_freqs,
        "di_method": di_method, "di_weight": di_weight,
        "batch_align_weight": batch_align_weight,
        "warmup_epochs": WARMUP_EPOCHS,
        "train_spatial_mean": train_spatial_mean.cpu().numpy(),
        "train_spatial_std":  train_spatial_std.cpu().numpy(),
        "train_idx": train_idx.cpu().numpy(),
        "val_idx":   val_idx.cpu().numpy() if val_idx is not None else None,
        "seed": seed,
        "gene_names": gene_names,
        "batch_file": args.batch_file,
        "n_batches":  n_batches,
    }
    torch.save(save_dict, model_path)
    print(f"Saved model: {model_path}", flush=True)

    latents = extract_latents(vae, X_full, S_full_norm, BATCH_SIZE, B=B_full)

    if batch_np is not None and n_batches >= 2:
        print("=== Final per-batch isodepth scale (full data) ===", flush=True)
        bflat = np.asarray(batch_np).reshape(-1)
        dmu   = latents["d_mu"]
        for bb in range(n_batches):
            m = (bflat == bb)
            if m.sum() == 0:
                continue
            for k in range(dmu.shape[1]):
                col = dmu[m, k]
                print(f"  batch {bb} d_{k}: mean={col.mean():+.4f} std={col.std():.4f} "
                      f"min={col.min():+.4f} max={col.max():+.4f} "
                      f"range={col.max() - col.min():.4f}", flush=True)

    rep_latent_dir = f"{LATENT_DIR}/rep{rep+1}"
    os.makedirs(rep_latent_dir, exist_ok=True)
    for key, arr in latents.items():
        npy_path = f"{rep_latent_dir}/{key}.npy"
        np.save(npy_path, arr)
        print(f"Saved latent {key} {arr.shape} → {npy_path}", flush=True)

    rep_plot_dir = f"{LATENT_PLOT_DIR}/rep{rep+1}"
    os.makedirs(rep_plot_dir, exist_ok=True)

    if decoder_type == "linear_hz" and z_bio_dim > 0:
        linear_z_importance(
            vae.decoder,
            gene_names=gene_names,
            save_path=f"{rep_plot_dir}/linear_hz_importance_{epoch_tag}_rep{rep+1}.png",
        )

    if n_isodepth == 1:
        plot_isodepth(
            S_np, latents["d_mu"][:, 0],
            save_path=f"{rep_plot_dir}/isodepth_spatial_{epoch_tag}.png",
            title=f"Learned Isodepth — {epoch_tag}",
        )
        if batch_np is not None:
            plot_isodepth_per_batch(
                S_np, latents["d_mu"][:, 0], batch_np,
                save_dir=rep_plot_dir,
                filename_prefix=(
                    f"isodepth_spatial_zdim{z_bio_dim}_{warmup_tag}_{epoch_tag}"
                    f"_{dec_tag}_{di_tag}_rep{rep+1}"),
                title_prefix=(
                    f"Learned Isodepth — z_dim={z_bio_dim}, "
                    f"{warmup_tag}, {epoch_tag}, {dec_tag}, {di_tag}, rep={rep+1}"),
            )
    else:
        plot_all_isodepths(
            S_np, latents["d_mu"],
            save_path=f"{rep_plot_dir}/isodepths_spatial_{epoch_tag}.png",
            title_prefix=f"Isodepths (n={n_isodepth}) — z_dim={z_bio_dim}, {warmup_tag}, {epoch_tag}, {batch_tag}, rep={rep+1}",
        )
        for k in range(n_isodepth):
            plot_isodepth(
                S_np, latents[f"d_{k}_mu"][:, 0],
                save_path=f"{rep_plot_dir}/isodepth_d{k}_spatial_{epoch_tag}.png",
                title=f"Isodepth d_{k} — z_dim={z_bio_dim}, {warmup_tag}, {epoch_tag}, {batch_tag}, rep={rep+1}",
            )
            if batch_np is not None:
                plot_isodepth_per_batch(
                    S_np, latents[f"d_{k}_mu"][:, 0], batch_np,
                    save_dir=rep_plot_dir,
                    filename_prefix=(
                        f"isodepth_d{k}_spatial_zdim{z_bio_dim}_{warmup_tag}_{epoch_tag}"
                        f"_{dec_tag}_{di_tag}_rep{rep+1}"),
                    title_prefix=(
                        f"Isodepth d_{k} — z_dim={z_bio_dim}, "
                        f"{warmup_tag}, {epoch_tag}, {dec_tag}, {di_tag}, rep={rep+1}"),
                )

    if z_bio_dim > 0:
        plot_z_latents(
            S_np, latents["z_mu"], latents["d_mu"][:, 0],
            save_path=f"{rep_plot_dir}/z_latents_spatial_{epoch_tag}.png",
            title_prefix=f"z latents — z_dim={z_bio_dim}, {warmup_tag}, {epoch_tag}, {batch_tag}, rep={rep+1}",
        )
        if batch_np is not None:
            plot_z_latents_per_batch(
                S_np, latents["z_mu"], batch_np,
                save_dir=rep_plot_dir,
                filename_prefix=(
                    f"z_latents_spatial_zdim{z_bio_dim}_{warmup_tag}_{epoch_tag}"
                    f"_{dec_tag}_{di_tag}_rep{rep+1}"),
                title_prefix=(
                    f"z latents — z_dim={z_bio_dim}, "
                    f"{warmup_tag}, {epoch_tag}, {dec_tag}, {di_tag}, rep={rep+1}"),
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
ax.axvline(x=WARMUP_EPOCHS, color="gray", linestyle=":", linewidth=1.5,
           label=f"phase 2 start (epoch {WARMUP_EPOCHS})")
ax.set_xlabel("Epoch"); ax.set_ylabel("NB NLL (recon loss)")
ax.set_title(f"Avg Recon Loss — z_dim={z_bio_dim}, n_isodepth={n_isodepth}, {dec_tag}, {di_tag}, {warmup_tag}, {epoch_tag}, {batch_tag}, {hidden_tag}")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"{LOSS_PLOT_DIR}/recon_zdim{z_bio_dim}_isodepth{n_isodepth}_{dec_tag}_{di_tag}_{warmup_tag}_{epoch_tag}_{batch_tag}_{hidden_tag}_avg.png",
    dpi=150)
plt.close()

row = {
    "z_dim":                  z_bio_dim,
    "n_isodepth":             n_isodepth,
    "decoder_type":           decoder_type,
    "di_method":              di_method,
    "di_weight":              di_weight,
    "batch_align_weight":     batch_align_weight,
    "warmup_epochs":          WARMUP_EPOCHS,
    "sigma":                  sigma,
    "num_freqs":              num_freqs,
    "beta":                   beta,
    "hidden_dim_spatial":     hidden_dim_spatial,
    "hidden_dim_decoder":     hidden_dim_decoder,
    "epochs":                 EPOCHS,
    "repeats":                REPEATS,
    "val_frac":               VAL_FRAC,
    "n_batches":              n_batches,
    "batch_file":             args.batch_file if args.batch_file is not None else "",
    "final_recon_train_loss": float(avg_recon_train[-1]),
    "std_recon_train_loss":   float(std_recon_train[-1]),
    "final_elbo_loss":        float(avg_elbo[-1]),
    "final_di_reg":           float(np.array(all_di_reg_loss).mean(axis=0)[-1])
                              if all_di_reg_loss and all_di_reg_loss[0] else 0.0,
    "final_batch_mmd":        float(np.array(all_batch_mmd_loss).mean(axis=0)[-1])
                              if all_batch_mmd_loss and all_batch_mmd_loss[0] else 0.0,
}
if avg_recon_val is not None:
    row["final_recon_val_loss"] = float(avg_recon_val[-1])
    row["std_recon_val_loss"]   = float(std_recon_val[-1])
    row["train_val_gap"]        = float(avg_recon_val[-1] - avg_recon_train[-1])

csv_path = f"{RESULTS_DIR}/summary_zdim{z_bio_dim}_isodepth{n_isodepth}_{dec_tag}_{di_tag}_{warmup_tag}_{batch_tag}_{hidden_tag}.csv"
pd.DataFrame([row]).to_csv(csv_path, index=False)
print(f"Saved summary: {csv_path}", flush=True)
print("=== DONE ===", flush=True)
