#!/usr/bin/env python3
"""
pe_heuristic.py -- data-driven choice of the GASTON-VI positional-encoding band set.

Takes exactly what the training script
(healthy_diseased/seq_train_additive_batch_d_condf_at_model_heuristic.py) takes:

    --coords_file  (N,2) spatial coordinates (any physical unit, e.g. um)
    --counts_file  (N,G) raw counts
    --batch_file   optional (N,) integer batch labels
    --condition_file is accepted but ignored (it does not affect the PE)

and returns the PE hyper-parameters that script consumes:

    --sigma, --num_freqs           (the log-frequency ladder)
    --extra_wavelengths            (bands the ladder cannot reach)

Everything is expressed in the training script's NORMALISED coordinate units:
coordinates are z-scored per batch, so 1 unit == 1 s.d. of that batch's
coordinates and the band with wavelength lambda is sin/cos(2*pi*s/lambda).

Heuristic (see README.md for the rationale):

  1. GEOMETRY, per batch (in normalised units)
       delta_nn  = median nearest-neighbour spacing            -> Nyquist floor
       D         = robust tissue diameter (2 x 99th-pct radius) -> largest scale
  2. EXPRESSION SCALE, per batch
       log-normalise counts, PCA, grid the PC scores, FFT autocorrelation,
       radially average -> correlogram rho_k(r) for each PC k.
       Pool them: C(r) = sum_k C_k(r) is the trace of the expression field's
       spatial covariance function (independent per-cell noise only inflates
       C(0), so for r >= h it is noise-free and each PC enters with its
       SPATIALLY STRUCTURED variance as weight; pure-noise PCs with
       rho_1 < --min_spatial_rho are dropped).  ell_c = correlation length of the
       pooled correlogram (rho = C(r)/C(h) falls to 1/e).  r_0 (zero crossing)
       and r_min (first trough ~ half period of a periodic pattern) are recorded
       for diagnostics.  --scale_mode min_top instead takes the min ell_c over
       the --n_keep_pcs PCs with the largest C_k(h) (chases the finest coherent
       structure; noisier).
  3. WAVELENGTH RANGE
       lambda_min = max( c_min * ell_c,  2*delta_nn,  sqrt(n_min_cells / density) )
       lambda_max = c_max * D
       Multi-batch: lambda_min = min over batches, lambda_max = max over batches
       (the PE is shared, only the spatial encoders are batch-specific).
  4. BAND GRID -> (sigma, num_freqs, extras)
       bands lambda = r^k, r = 2^(1/bands_per_octave), k_min..k_max covering
       [lambda_min, lambda_max]. The ladder lambda_j = sigma^(-j/N) can only
       produce lambda = r^1..r^N (all > 1), so
           sigma = r^(-k_max), num_freqs = k_max,
           extra_wavelengths = { r^k : k_min <= k <= 0 }   (all <= 1).

Outputs a JSON with every intermediate quantity plus ready-to-paste CLI / slurm
strings.  Nothing here touches the GPU; runtime is dominated by reading the
counts matrix once.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def robust_diameter(S, pct=99.0):
    """2 x pct-percentile distance from the centroid (robust to outlier cells)."""
    c = S.mean(axis=0)
    r = np.linalg.norm(S - c, axis=1)
    return 2.0 * float(np.percentile(r, pct))


def nn_spacing(S, n_sub=100_000, seed=0):
    """Median nearest-neighbour distance (on a subsample for speed; NN distances
    are estimated on the FULL tree so subsampling does not bias them)."""
    tree = cKDTree(S)
    rng = np.random.default_rng(seed)
    idx = rng.choice(S.shape[0], min(n_sub, S.shape[0]), replace=False)
    d, _ = tree.query(S[idx], k=2)
    return float(np.median(d[:, 1]))


def lognorm_pcs(X_path, rows, n_pcs, n_fit, chunk, seed):
    """Library-size normalise + log1p + gene z-score, PCA (fit on <= n_fit random
    rows, project all `rows`). Streams the counts matrix in row chunks.

    Returns scores (len(rows), n_pcs), explained-variance ratio, gene stats."""
    from sklearn.decomposition import PCA

    X = np.load(X_path, mmap_mode="r")
    rows = np.asarray(rows)
    n = rows.shape[0]
    rng = np.random.default_rng(seed)

    def _load(r):
        r = np.sort(r)
        return np.asarray(X[r], dtype=np.float32)

    # pass 1 (sequential): library sizes for all rows + the raw rows of a random
    # fit subset, collected on the fly so the file is never randomly accessed
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[rng.choice(n, min(n_fit, n), replace=False)] = True
    lib = np.zeros(n, dtype=np.float64)
    F = np.empty((int(fit_mask.sum()), X.shape[1]), dtype=np.float32)   # preallocated, no copies
    f = 0
    for s in range(0, n, chunk):
        blk = _load(rows[s:s + chunk])
        lib[s:s + chunk] = blk.sum(axis=1)
        m = fit_mask[s:s + chunk]
        F[f:f + int(m.sum())] = blk[m]
        f += int(m.sum())
        del blk
    target = float(np.median(lib[lib > 0]))

    def _norm(block, lib_block):
        sf = np.where(lib_block > 0, target / np.maximum(lib_block, 1e-12), 0.0)
        return np.log1p(block * sf[:, None].astype(np.float32))

    # in-place normalisation / standardisation of the fit matrix
    sf = np.where(lib[fit_mask] > 0, target / np.maximum(lib[fit_mask], 1e-12), 0.0).astype(np.float32)
    F *= sf[:, None]
    np.log1p(F, out=F)
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd < 1e-8] = 1.0
    F -= mu
    F /= sd
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=seed)
    pca.fit(F)
    W = pca.components_.T.astype(np.float32)          # (G, k)
    del F

    scores = np.zeros((n, n_pcs), dtype=np.float32)
    for s in range(0, n, chunk):                      # pass 2 (sequential): project
        blk = _norm(_load(rows[s:s + chunk]), lib[s:s + chunk])
        scores[s:s + chunk] = ((blk - mu) / sd) @ W
    return scores, pca.explained_variance_ratio_, target


def radial_correlogram(S, v, h, max_grid=2048):
    """Radially averaged spatial autocorrelation of a per-cell scalar field v(S).

    Cells are binned onto a square grid of pitch h (mean of v per occupied
    pixel); the masked autocovariance is computed with zero-padded FFTs
    (Marcotte's method):  C(l) = sum_x F(x)F(x+l) / sum_x M(x)M(x+l),
    with F = mean-centred field * mask.  Returns (lags, C) with lags in the
    units of S, C[0] = variance at lag 0.
    """
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean()
    lo = S.min(axis=0)
    ij = np.floor((S - lo) / h).astype(np.int64)
    nx, ny = ij.max(axis=0) + 1
    if max(nx, ny) > max_grid:
        raise ValueError(f"grid {nx}x{ny} exceeds max_grid={max_grid}; increase h")

    flat = ij[:, 0] * ny + ij[:, 1]
    cnt = np.bincount(flat, minlength=nx * ny).astype(np.float64)
    sm = np.bincount(flat, weights=v, minlength=nx * ny)
    M = (cnt > 0).astype(np.float64).reshape(nx, ny)
    F = np.zeros(nx * ny)
    F[cnt > 0] = sm[cnt > 0] / cnt[cnt > 0]
    F = F.reshape(nx, ny)
    F -= F[M > 0].mean()
    F *= M

    px, py = 2 * nx, 2 * ny
    fF = np.fft.rfft2(F, s=(px, py))
    fM = np.fft.rfft2(M, s=(px, py))
    num = np.fft.irfft2(fF * np.conj(fF), s=(px, py))
    den = np.fft.irfft2(fM * np.conj(fM), s=(px, py))

    # lag grid (wrap negative lags)
    lx = np.fft.fftfreq(px, d=1.0 / px)
    ly = np.fft.fftfreq(py, d=1.0 / py)
    LX, LY = np.meshgrid(lx, ly, indexing="ij")
    R = np.sqrt(LX ** 2 + LY ** 2) * h
    rbin = np.floor(R / h + 0.5).astype(np.int64)
    ok = den > 1.0                                  # >= 2 overlapping pixel pairs
    ok &= (rbin <= min(nx, ny) // 2)               # ignore poorly-supported long lags
    nb = int(rbin[ok].max()) + 1
    num_r = np.bincount(rbin[ok], weights=num[ok], minlength=nb)
    den_r = np.bincount(rbin[ok], weights=den[ok], minlength=nb)
    C = np.where(den_r > 0, num_r / np.maximum(den_r, 1e-12), np.nan)
    lags = np.arange(nb) * h
    return lags, C


def correlogram_scales(lags, C):
    """Correlation length (1/e), zero crossing and first trough of rho(r) = C(r)/C(h).

    C(0) is inflated by the nugget (per-pixel noise), so rho is normalised at the
    FIRST NON-ZERO lag; rho_1 = C(h)/C(0) is returned as the spatial-signal
    strength (Moran-like: 1 = perfectly smooth field, 0 = pure noise).
    """
    C = np.asarray(C, dtype=np.float64)
    if C.shape[0] < 3 or not np.isfinite(C[0]) or C[0] <= 0:
        return dict(rho1=np.nan, cov1=np.nan, ell_c=np.nan, r0=np.nan, r_min=np.nan)
    rho1 = float(C[1] / C[0])
    cov1 = float(C[1])            # spatially structured variance (noise-free: different cells)
    rho = C / C[1] if C[1] > 0 else C / C[0]
    ell_c = r0 = r_min = np.nan
    for i in range(1, len(rho)):
        if not np.isfinite(rho[i]):
            break
        if np.isnan(ell_c) and rho[i] < math.exp(-1.0):
            # linear interpolation between i-1 and i
            a, b = rho[i - 1], rho[i]
            t = (a - math.exp(-1.0)) / (a - b) if a != b else 0.0
            ell_c = float(lags[i - 1] + t * (lags[i] - lags[i - 1]))
        if np.isnan(r0) and rho[i] < 0:
            a, b = rho[i - 1], rho[i]
            t = a / (a - b) if a != b else 0.0
            r0 = float(lags[i - 1] + t * (lags[i] - lags[i - 1]))
        if (not np.isnan(r0)) and np.isnan(r_min) and i + 1 < len(rho) \
                and np.isfinite(rho[i + 1]) and rho[i] <= rho[i - 1] and rho[i] < rho[i + 1]:
            r_min = float(lags[i])
            break
    return dict(rho1=rho1, cov1=cov1, ell_c=ell_c, r0=r0, r_min=r_min)


def band_grid(lambda_min, lambda_max, bands_per_octave):
    """All lambda = r^k covering [lambda_min, lambda_max] -> (sigma, num_freqs, extras)."""
    r = 2.0 ** (1.0 / bands_per_octave)
    k_min = int(math.floor(math.log(lambda_min) / math.log(r) + 1e-9))
    k_max = int(math.ceil(math.log(lambda_max) / math.log(r) - 1e-9))
    k_max = max(k_max, 1)
    sigma = r ** (-k_max)
    num_freqs = k_max
    extras = [r ** k for k in range(k_min, 1)]          # k <= 0  -> lambda <= 1
    bands = [r ** k for k in range(k_min, k_max + 1)]
    return dict(ratio=r, k_min=k_min, k_max=k_max, sigma=sigma, num_freqs=num_freqs,
                extra_wavelengths=extras, all_wavelengths=bands, n_bands=len(bands))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords_file", required=True)
    ap.add_argument("--counts_file", required=True)
    ap.add_argument("--batch_file", default=None)
    ap.add_argument("--condition_file", default=None, help="accepted for CLI symmetry; ignored")
    ap.add_argument("--out_json", required=True)
    # knobs of the heuristic (defaults are what the README describes)
    ap.add_argument("--c_min", type=float, default=1.0,
                    help="lambda_min = c_min * ell_c (finest band relative to the expression correlation length)")
    ap.add_argument("--c_max", type=float, default=1.0,
                    help="lambda_max = c_max * tissue diameter")
    ap.add_argument("--bands_per_octave", type=float, default=3.0)
    ap.add_argument("--n_pcs", type=int, default=10)
    ap.add_argument("--scale_mode", choices=["pooled", "min_top"], default="pooled",
                    help="pooled: ell_c of the variance-pooled correlogram over all spatial PCs (default); "
                         "min_top: min ell_c over the n_keep_pcs PCs with the most spatially structured variance")
    ap.add_argument("--n_keep_pcs", type=int, default=3, help="only for --scale_mode min_top")
    ap.add_argument("--min_spatial_rho", type=float, default=0.05,
                    help="PCs whose first-lag autocorrelation is below this are treated as non-spatial")
    ap.add_argument("--n_min_cells", type=float, default=50.0,
                    help="floor: at least this many cells per lambda_min^2 area")
    ap.add_argument("--grid_pitch_nn", type=float, default=2.0,
                    help="correlogram grid pitch = this x median NN spacing")
    ap.add_argument("--n_fit", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    S_all = np.load(args.coords_file).astype(np.float64)
    N = S_all.shape[0]
    if args.batch_file:
        B = np.load(args.batch_file).astype(np.int64).reshape(-1)
        assert B.shape[0] == N
    else:
        B = np.zeros(N, dtype=np.int64)
    batches = [int(b) for b in np.unique(B)]
    log(f"N={N} cells, {len(batches)} batch(es): {batches}")

    per_batch = {}
    for b in batches:
        rows = np.where(B == b)[0]
        S = S_all[rows]
        mean, std = S.mean(axis=0), S.std(axis=0)
        std = np.maximum(std, 1e-8)
        Sn = (S - mean) / std                       # == training-script normalisation
        um_per_unit = float(std.mean())             # for reporting only

        dnn_um = nn_spacing(S, seed=args.seed)
        dnn = dnn_um / um_per_unit if np.allclose(std[0], std[1], rtol=0.05) else nn_spacing(Sn, seed=args.seed)
        D = robust_diameter(Sn)
        density = rows.shape[0] / (math.pi * (D / 2.0) ** 2)   # cells per unit^2
        log(f"[batch {b}] n={rows.shape[0]}  std_um={std.round(1).tolist()}  "
            f"nn={dnn_um:.1f} um ({dnn:.4f} u)  diameter={D:.3f} u  density={density:.0f}/u^2")

        # expression PCs
        scores, evr, target = lognorm_pcs(args.counts_file, rows, args.n_pcs,
                                          args.n_fit, args.chunk, args.seed)
        h = args.grid_pitch_nn * dnn
        # cap the grid
        while (S.max(axis=0) - S.min(axis=0)).max() / std.min() / h > 1800:
            h *= 1.5
        pcs, curves = [], []
        for k in range(args.n_pcs):
            lags, C = radial_correlogram(Sn, scores[:, k], h)
            curves.append(C)
            sc = correlogram_scales(lags, C)
            sc.update(pc=k, evr=float(evr[k]))
            sc["rho_curve"] = [float(x) if np.isfinite(x) else None for x in (C / C[1] if C[1] > 0 else C / C[0])[:200]]
            pcs.append(sc)
            log(f"    PC{k}: evr={evr[k]:.3f}  rho1={sc['rho1']:.3f}  spat_var={sc['cov1']:.3f}  ell_c={sc['ell_c']:.4f} u "
                f"({sc['ell_c'] * um_per_unit:.0f} um)  r0={sc['r0']:.4f}  r_min={sc['r_min']:.4f}")
        spatial = [p for p in pcs if np.isfinite(p["rho1"]) and p["rho1"] >= args.min_spatial_rho]
        if not spatial:
            log("    WARNING: no PC passes the spatial-signal threshold; using the single most spatial PC")
            spatial = sorted([p for p in pcs if np.isfinite(p["cov1"])], key=lambda p: -p["cov1"])[:1]
        spatial.sort(key=lambda p: -p["cov1"])          # most spatially structured variance first
        L = min(len(curves[p["pc"]]) for p in spatial)
        C_pool = np.sum([np.nan_to_num(curves[p["pc"]][:L]) for p in spatial], axis=0)
        pooled = correlogram_scales(lags[:L], C_pool)
        log(f"    pooled correlogram over PCs {[p['pc'] for p in spatial]}: ell_c={pooled['ell_c']:.4f} u "
            f"({pooled['ell_c'] * um_per_unit:.0f} um)  r0={pooled['r0']:.4f}  r_min={pooled['r_min']:.4f} "
            f"(r_min ~ half period: {2 * pooled['r_min'] * um_per_unit:.0f} um period)")
        if args.scale_mode == "pooled":
            keep = spatial
            ell_c = float(pooled["ell_c"])
        else:
            keep = [p for p in spatial if np.isfinite(p["ell_c"])][:args.n_keep_pcs]
            ell_c = float(min(p["ell_c"] for p in keep))
        if not np.isfinite(ell_c):
            raise RuntimeError("could not estimate a correlation length (correlogram never fell below 1/e)")
        lam_floor_nyq = 2.0 * dnn
        lam_floor_cells = math.sqrt(args.n_min_cells / density)
        lam_min_b = max(args.c_min * ell_c, lam_floor_nyq, lam_floor_cells)
        lam_max_b = args.c_max * D
        log(f"    scale_mode={args.scale_mode}, PCs {[p['pc'] for p in keep]} -> ell_c={ell_c:.4f} u ({ell_c * um_per_unit:.0f} um); "
            f"floors: nyquist={lam_floor_nyq:.4f}, cells={lam_floor_cells:.4f} -> "
            f"lambda_min={lam_min_b:.4f} u, lambda_max={lam_max_b:.3f} u")
        per_batch[b] = dict(
            n=int(rows.shape[0]), coord_mean=mean.tolist(), coord_std=std.tolist(),
            um_per_unit=um_per_unit, nn_um=dnn_um, nn_unit=dnn, diameter=D, density=density,
            grid_pitch=h, ell_c=ell_c, scale_mode=args.scale_mode, kept_pcs=[p["pc"] for p in keep],
            pooled=dict(ell_c=pooled["ell_c"], r0=pooled["r0"], r_min=pooled["r_min"],
                        rho_curve=[float(x) if np.isfinite(x) else None for x in (C_pool / C_pool[1])[:200]]),
            lambda_floor_nyquist=lam_floor_nyq, lambda_floor_cells=lam_floor_cells,
            lambda_min=lam_min_b, lambda_max=lam_max_b, lag_step=h,
            pcs=[{k: v for k, v in p.items()} for p in pcs],
        )

    lam_min = min(v["lambda_min"] for v in per_batch.values())
    lam_max = max(v["lambda_max"] for v in per_batch.values())
    grid = band_grid(lam_min, lam_max, args.bands_per_octave)
    extras_str = ",".join(f"{x:.6g}" for x in grid["extra_wavelengths"])
    # per-batch band sets (each batch's own [lambda_min, lambda_max]) for the
    # per-batch-PE training variant (seq_train_perbatch_pe.py --pe_per_batch_json)
    per_batch_rec = {}
    for b, v in per_batch.items():
        g = band_grid(v["lambda_min"], v["lambda_max"], args.bands_per_octave)
        per_batch_rec[str(b)] = dict(sigma=g["sigma"], num_freqs=g["num_freqs"],
                                     extra_wavelengths=g["extra_wavelengths"], n_bands=g["n_bands"],
                                     lambda_min=v["lambda_min"], lambda_max=v["lambda_max"])
    out = dict(
        inputs=dict(coords_file=args.coords_file, counts_file=args.counts_file, batch_file=args.batch_file),
        knobs=dict(c_min=args.c_min, c_max=args.c_max, bands_per_octave=args.bands_per_octave,
                   n_pcs=args.n_pcs, scale_mode=args.scale_mode, n_keep_pcs=args.n_keep_pcs,
                   min_spatial_rho=args.min_spatial_rho,
                   n_min_cells=args.n_min_cells, grid_pitch_nn=args.grid_pitch_nn),
        per_batch=per_batch,
        lambda_min=lam_min, lambda_max=lam_max,
        recommendation=dict(sigma=grid["sigma"], num_freqs=grid["num_freqs"],
                            extra_wavelengths=grid["extra_wavelengths"],
                            extra_wavelengths_str=extras_str,
                            ratio=grid["ratio"], n_bands=grid["n_bands"],
                            all_wavelengths=grid["all_wavelengths"],
                            ladder_range=[grid["ratio"], 1.0 / grid["sigma"]]),
        per_batch_recommendation=per_batch_rec,
        cli=(f"--sigma {grid['sigma']:.6g} --num_freqs {grid['num_freqs']}"
             + (f" --extra_wavelengths \"{extras_str}\"" if extras_str else "")),
        slurm_env=(f"SIGMA={grid['sigma']:.6g} NUM_FREQS={grid['num_freqs']} "
                   f"EXTRA_WAVELENGTHS=\"{extras_str}\" EXTRA_WAVELENGTHS_GEOMSPACE=\"\""),
        elapsed_s=time.time() - t0,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    log("")
    log(f"lambda range (union over batches): [{lam_min:.4f}, {lam_max:.3f}] normalised units")
    log(f"band grid: ratio {grid['ratio']:.4f} (={args.bands_per_octave:g}/octave), "
        f"{grid['n_bands']} bands: {[round(x, 4) for x in grid['all_wavelengths']]}")
    log(f"RECOMMENDATION: sigma={grid['sigma']:.6g}  num_freqs={grid['num_freqs']}  "
        f"extra_wavelengths=[{extras_str}]")
    log(f"  ladder covers lambda in [{grid['ratio']:.4f}, {1 / grid['sigma']:.4f}]; "
        f"{len(grid['extra_wavelengths'])} extras cover [{lam_min:.4f}, 1]")
    log(f"  CLI: {out['cli']}")
    log(f"  slurm: {out['slurm_env']}")
    log(f"wrote {args.out_json}  ({out['elapsed_s']:.0f} s)")


if __name__ == "__main__":
    main()
