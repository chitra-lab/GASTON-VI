#!/usr/bin/env python3

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import argparse

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from dataset import SpatialDataset
from positional_encoding import LogFrequencyPositionalEncoding
from vae import VAE


# ============================================================
# Metrics
# ============================================================

def reconstruction_mse(vae, loader, device):
    vae.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["gene"].to(device)
            s = batch["spatial"].to(device)
            mu, _ = vae.encoder(x, s)
            loc, _ = vae.decoder(mu)
            mse = ((loc - x) ** 2).mean()
            total += mse.item() * x.size(0)
            count += x.size(0)
    return total / count


def reconstruction_corr(vae, loader, device):
    vae.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["gene"].to(device)
            s = batch["spatial"].to(device)
            mu, _ = vae.encoder(x, s)
            loc, _ = vae.decoder(mu)
            preds.append(loc.cpu())
            trues.append(x.cpu())
    if len(preds) == 0:
        return float("nan")
    Y = torch.cat(preds).numpy()
    X = torch.cat(trues).numpy()
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = (Xc * Yc).sum(0)
    den = np.sqrt((Xc**2).sum(0) * (Yc**2).sum(0)) + 1e-8
    return float(np.nanmean(num / den))


def evaluate_elbo(svi, loader, device):
    tot = 0.0
    cnt = 0
    for batch in loader:
        x = batch["gene"].to(device)
        s = batch["spatial"].to(device)
        loss = svi.evaluate_loss(x, s)
        tot += loss
        cnt += x.size(0)
    return tot / cnt


# ============================================================
# Main
# ============================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pyro.set_rng_seed(args.seed)

    # Build hyperparameter tag
    combo_tag = (
        f"var_{args.variant}"
        f"_noise{args.data_noise}"
        f"_sigma{args.sigma}"
        f"_freq{args.num_freqs}"
    )

    base = os.environ.get("VAE_OUTDIR", "sim_models")
    combo_dir = os.path.join(base, combo_tag)
    os.makedirs(combo_dir, exist_ok=True)

    seed_dir = os.path.join(combo_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)

    print("Saving to:", seed_dir)

    # ====================================================
    # Load dataset
    # ====================================================
    S_path = [f for f in os.listdir(args.data_dir) if f.startswith("S_")][0]
    S_np = np.load(os.path.join(args.data_dir, S_path))

    sigma_tag = f"sigma{args.data_noise}"
    expr_file = [f for f in os.listdir(args.data_dir)
                 if f.startswith("A_") and sigma_tag in f][0]
    X_np = np.load(os.path.join(args.data_dir, expr_file))

    from sklearn.preprocessing import StandardScaler
    scaler_x = StandardScaler().fit(X_np)
    scaler_s = StandardScaler().fit(S_np)

    X_norm = scaler_x.transform(X_np)
    S_norm = scaler_s.transform(S_np)

    X = torch.tensor(X_norm, dtype=torch.float32)
    C = torch.tensor(S_norm, dtype=torch.float32)

    pe = LogFrequencyPositionalEncoding(
        num_frequencies=args.num_freqs,
        sigma=args.sigma
    )
    S = pe(C)

    # Dataset + splits
    dataset = SpatialDataset(X, S)
    N = len(dataset)
    train_N = int(0.8 * N)
    val_N = N - train_N

    train_set, val_set = random_split(
        dataset, [train_N, val_N],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False)

    train_big = DataLoader(train_set, batch_size=4096, shuffle=False)
    val_big   = DataLoader(val_set,   batch_size=4096, shuffle=False)

    # ====================================================
    # Build VAE + SVI
    # ====================================================
    vae = VAE(
        variant=args.variant,
        g_dim=X.shape[1],
        s_dim=S.shape[1],
        z_dim=1,
        hidden=20
    ).to(device)

    pyro.clear_param_store()
    svi = SVI(
        model=vae.model,
        guide=vae.guide,
        optim=Adam({"lr": args.lr}),
        loss=Trace_ELBO()
    )

    best_val = float("inf")
    best_path = os.path.join(seed_dir, "best_vae.pt")

    train_elbos = []
    val_elbos = []
    train_mses = []
    val_mses = []
    train_corrs = []
    val_corrs = []

    # ====================================================
    # Train loop
    # ====================================================
    for epoch in range(args.epochs):
        vae.train()
        total_loss = 0.0
        total_count = 0

        for batch in train_loader:
            x = batch["gene"].to(device)
            s = batch["spatial"].to(device)
            loss = svi.step(x, s)
            total_loss += loss
            total_count += x.size(0)

        train_elbo = total_loss / total_count
        val_elbo = evaluate_elbo(svi, val_loader, device)

        train_mse = reconstruction_mse(vae, train_big, device)
        val_mse   = reconstruction_mse(vae, val_big, device)
        train_corr = reconstruction_corr(vae, train_big, device)
        val_corr   = reconstruction_corr(vae, val_big, device)

        train_elbos.append(train_elbo)
        val_elbos.append(val_elbo)
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        train_corrs.append(train_corr)
        val_corrs.append(val_corr)

        if val_elbo < best_val:
            best_val = val_elbo
            torch.save(vae.state_dict(), best_path)

        if epoch % 10 == 0:
            print(f"[{epoch}] val_elbo={val_elbo:.4f}")

    print("BEST validation ELBO =", best_val)

    np.savetxt(os.path.join(seed_dir, "train_elbo.txt"), train_elbos)
    np.savetxt(os.path.join(seed_dir, "val_elbo.txt"), val_elbos)
    np.savetxt(os.path.join(seed_dir, "train_mse.txt"), train_mses)
    np.savetxt(os.path.join(seed_dir, "val_mse.txt"), val_mses)
    np.savetxt(os.path.join(seed_dir, "train_corr.txt"), train_corrs)
    np.savetxt(os.path.join(seed_dir, "val_corr.txt"), val_corrs)

    np.save(os.path.join(seed_dir, "scaler_x_mean.npy"), scaler_x.mean_)
    np.save(os.path.join(seed_dir, "scaler_x_std.npy"),  scaler_x.scale_)
    np.save(os.path.join(seed_dir, "scaler_s_mean.npy"), scaler_s.mean_)
    np.save(os.path.join(seed_dir, "scaler_s_std.npy"),  scaler_s.scale_)

    # Compute isodepth
    vae.load_state_dict(torch.load(best_path, map_location=device))
    vae.eval()

    full_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    mu_list = []
    with torch.no_grad():
        for batch in full_loader:
            x = batch["gene"].to(device)
            s = batch["spatial"].to(device)
            mu = vae.encode_mu(x, s)
            mu_list.append(mu.cpu())

    mu_all = torch.cat(mu_list).numpy().squeeze()
    np.save(os.path.join(seed_dir, "isodepth.npy"), mu_all)

    print("DONE:", seed_dir)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", required=True)
    p.add_argument("--data_noise", type=float, required=True)

    p.add_argument("--variant", required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--num_freqs", type=int, required=True)

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, required=True)

    args = p.parse_args()
    main(args)
