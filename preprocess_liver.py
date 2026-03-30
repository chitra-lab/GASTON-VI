#!/usr/bin/env python3
"""
preprocess_liver.py
Preprocess mouse liver MERFISH data for GASTON-VI.

Selects the top K highly variable genes (HVGs) by dispersion (variance /
mean) from the raw count matrix and writes standardised outputs to a
user-specified directory.

Usage:
    python preprocess_liver.py \
        --input_dir  /scratch/uchitra1/datasets/spatial-RNA/merfish/mouse_liver_merfish \
        --output_dir /scratch/uchitra1/projects/GASTON-VAE/preprocessed/liver \
        --n_hvgs     5000
"""

import os
import argparse
import numpy as np
from scipy import sparse

# ---- args ----
parser = argparse.ArgumentParser(description="Preprocess mouse liver MERFISH data")
parser.add_argument("--input_dir", type=str,
    default="/scratch/uchitra1/datasets/spatial-RNA/merfish/mouse_liver_merfish",
    help="Directory containing coords_mat.npy and counts_mat.npy (raw dense counts)")
parser.add_argument("--output_dir", type=str, required=True,
    help="Directory to write coords_mat.npy and counts_mat_hvg.npz")
parser.add_argument("--n_hvgs", type=int, default=5000,
    help="Number of top HVGs to retain (by variance/mean dispersion). "
         "Set to -1 to keep all genes.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---- load ----
coords_path = os.path.join(args.input_dir, "coords_mat.npy")
counts_path = os.path.join(args.input_dir, "counts_mat.npy")

print(f"Loading coords from: {coords_path}", flush=True)
S_np = np.load(coords_path)
print(f"  coords shape: {S_np.shape}", flush=True)

print(f"Loading counts from: {counts_path}", flush=True)
counts_raw = np.load(counts_path).astype(np.float32)
print(f"  counts shape: {counts_raw.shape}", flush=True)

# ---- HVG selection ----
if args.n_hvgs > 0 and args.n_hvgs < counts_raw.shape[1]:
    gene_means  = counts_raw.mean(axis=0)
    gene_vars   = counts_raw.var(axis=0)
    dispersions = gene_vars / (gene_means + 1e-8)
    hvg_idx     = np.argsort(dispersions)[-args.n_hvgs:]
    counts_hvg  = counts_raw[:, hvg_idx]
    print(f"Selected top {args.n_hvgs} HVGs → shape {counts_hvg.shape}", flush=True)
else:
    counts_hvg = counts_raw
    print(f"Keeping all {counts_raw.shape[1]} genes", flush=True)

A_out = sparse.csr_matrix(counts_hvg)

# ---- write ----
out_coords = os.path.join(args.output_dir, "coords_mat.npy")
out_counts = os.path.join(args.output_dir, "counts_mat_hvg.npz")

np.save(out_coords, S_np)
sparse.save_npz(out_counts, A_out)

print(f"\nWrote: {out_coords}")
print(f"Wrote: {out_counts}")
print("\nPaths to pass to gaston_vi_model.py:")
print(f"  --coords_file {out_coords}")
print(f"  --counts_file {out_counts}")
