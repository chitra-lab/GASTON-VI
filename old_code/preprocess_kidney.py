#!/usr/bin/env python3
"""
preprocess_kidney.py
Preprocess mouse kidney VisiumHD data for GASTON-VI.

The kidney data is already HVG-selected.  This script verifies the files,
optionally re-selects the top K HVGs from the existing count matrix, and
writes standardised outputs (coords_mat.npy, counts_mat_hvg.npz) to a
user-specified output directory.

Usage:
    python preprocess_kidney.py \
        --input_dir  /scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium \
        --output_dir /scratch/uchitra1/projects/GASTON-VAE/preprocessed/kidney \
        --n_hvgs     5000
"""

import os
import argparse
import numpy as np
from scipy import sparse

# ---- args ----
parser = argparse.ArgumentParser(description="Preprocess mouse kidney VisiumHD data")
parser.add_argument("--input_dir", type=str,
    default="/scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium",
    help="Directory containing coords_mat.npy and counts_mat_hvg5000.npz")
parser.add_argument("--output_dir", type=str, required=True,
    help="Directory to write coords_mat.npy and counts_mat_hvg.npz")
parser.add_argument("--n_hvgs", type=int, default=5000,
    help="Number of HVGs to retain (reselects from existing matrix by dispersion). "
         "Set to -1 to keep all genes in the input matrix.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---- load ----
coords_path = os.path.join(args.input_dir, "coords_mat.npy")
counts_path = os.path.join(args.input_dir, "counts_mat_hvg5000.npz")

print(f"Loading coords from: {coords_path}", flush=True)
S_np = np.load(coords_path)
print(f"  coords shape: {S_np.shape}", flush=True)

print(f"Loading counts from: {counts_path}", flush=True)
A_raw = sparse.load_npz(counts_path)
print(f"  counts shape: {A_raw.shape}", flush=True)

# ---- (optional) re-select HVGs ----
if args.n_hvgs > 0 and args.n_hvgs < A_raw.shape[1]:
    counts_dense = A_raw.toarray().astype(np.float32)
    gene_means   = counts_dense.mean(axis=0)
    gene_vars    = counts_dense.var(axis=0)
    dispersions  = gene_vars / (gene_means + 1e-8)
    hvg_idx      = np.argsort(dispersions)[-args.n_hvgs:]
    A_out        = sparse.csr_matrix(counts_dense[:, hvg_idx])
    print(f"Re-selected top {args.n_hvgs} HVGs by dispersion → shape {A_out.shape}", flush=True)
else:
    A_out = A_raw
    print(f"Keeping all {A_raw.shape[1]} genes (n_hvgs={args.n_hvgs})", flush=True)

# ---- write ----
out_coords = os.path.join(args.output_dir, "coords_mat.npy")
out_counts = os.path.join(args.output_dir, "counts_mat_hvg.npz")

np.save(out_coords, S_np)
sparse.save_npz(out_counts, A_out.astype(np.float32))

print(f"\nWrote: {out_coords}")
print(f"Wrote: {out_counts}")
print("\nPaths to pass to gaston_vi_model.py:")
print(f"  --coords_file {out_coords}")
print(f"  --counts_file {out_counts}")
