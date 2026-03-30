#!/usr/bin/env python3
"""
preprocess_tonsil.py
Preprocess human tonsil VisiumHD data for GASTON-VI.

Loads data from either a pre-saved .h5ad file (preferred) or raw 10x HDF5 +
spatial parquet files.  Selects the top K HVGs by dispersion and writes
standardised outputs.

Usage:
    python preprocess_tonsil.py \
        --input_dir  /scratch/uchitra1/users/sdharma5_code/gaston/binned_outputs/square_016um \
        --output_dir /scratch/uchitra1/projects/GASTON-VAE/preprocessed/tonsil \
        --n_hvgs     5000
"""

import os
import argparse
import numpy as np
from scipy import sparse
import scanpy as sc
import pandas as pd

# ---- args ----
parser = argparse.ArgumentParser(description="Preprocess human tonsil VisiumHD data")
parser.add_argument("--input_dir", type=str,
    default="/scratch/uchitra1/users/sdharma5_code/gaston/binned_outputs/square_016um",
    help="Directory containing either adata_with_leiden.h5ad OR "
         "filtered_feature_bc_matrix.h5 + spatial/tissue_positions.parquet")
parser.add_argument("--output_dir", type=str, required=True,
    help="Directory to write coords_mat.npy and counts_mat_hvg.npz")
parser.add_argument("--n_hvgs", type=int, default=5000,
    help="Number of top HVGs to retain. Set to -1 to keep all genes.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---- load ----
h5ad_path = os.path.join(args.input_dir, "adata_with_leiden.h5ad")
h5_path   = os.path.join(args.input_dir, "filtered_feature_bc_matrix.h5")

if os.path.exists(h5ad_path):
    print(f"Loading AnnData from: {h5ad_path}", flush=True)
    adata = sc.read_h5ad(h5ad_path)
elif os.path.exists(h5_path):
    print(f"Loading from 10x HDF5: {h5_path}", flush=True)
    adata = sc.read_10x_h5(h5_path)
    parquet_path = os.path.join(args.input_dir, "spatial", "tissue_positions.parquet")
    print(f"Loading spatial positions from: {parquet_path}", flush=True)
    spatial_df = (
        pd.read_parquet(parquet_path)
        .set_index("barcode")
        .loc[adata.obs_names]
    )
    adata.obsm["spatial"] = spatial_df[["pxl_col_in_fullres",
                                         "pxl_row_in_fullres"]].values
else:
    raise FileNotFoundError(
        f"Neither '{h5ad_path}' nor '{h5_path}' found in {args.input_dir}"
    )

print(f"AnnData shape: {adata.shape}", flush=True)

# ---- extract spatial coordinates ----
S_np = adata.obsm["spatial"].astype(np.float32)
print(f"Spatial coords shape: {S_np.shape}", flush=True)

# ---- extract counts ----
counts_raw = adata.X
if hasattr(counts_raw, "toarray"):
    counts_raw = counts_raw.toarray()
counts_raw = np.round(counts_raw).astype(np.float32)
print(f"Raw counts shape: {counts_raw.shape}", flush=True)

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
