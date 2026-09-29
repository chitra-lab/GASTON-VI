"""--pe_auto: derive the PE band set from the data via pe_heuristic.

Moved verbatim from train.py (then seq_train_unified.py) (2026-09-29 modularisation, step 2).

NOTE for other modalities: _pe_auto_one calls pe_heuristic.lognorm_pcs, which
normalises counts as log1p-CP10K.  That is wrong for intensity data (proteomics)
and is the seam to change -- pe/bands.py is modality-agnostic and should not need
edits.
"""
import os

import numpy as np


def _pe_auto_fingerprint(a):
    """Everything the derived bands depend on. The cache is keyed on this, so a
    regenerated counts file or a changed knob invalidates it instead of silently
    returning stale bands (the e135 counts file changed size mid-session, which a
    path-only cache would have missed)."""
    def st(pth):
        if not pth or not os.path.exists(pth):
            return None
        s_ = os.stat(pth)
        return dict(path=os.path.abspath(pth), size=s_.st_size, mtime=int(s_.st_mtime))
    return dict(counts=st(a.counts_file), coords=st(a.coords_file), batch=st(a.batch_file),
                mode=a.pe_auto, c_min=a.pe_auto_c_min, c_max=a.pe_auto_c_max,
                bands_per_octave=a.pe_auto_bands_per_octave, n_pcs=a.pe_auto_n_pcs,
                min_spatial_rho=a.pe_auto_min_spatial_rho,
                n_min_cells=a.pe_auto_n_min_cells,
                grid_pitch_nn=a.pe_auto_grid_pitch_nn)


def _pe_auto_one(PH, S_sec, rows, counts_file, a):
    """pe_heuristic's pipeline for one section -> (sigma, num_freqs, extras)."""
    import math as _m
    std = np.maximum(S_sec.std(axis=0), 1e-8)
    z = (S_sec - S_sec.mean(axis=0)) / std
    dnn_um = PH.nn_spacing(S_sec, seed=0)
    dnn = (dnn_um / float(std.mean()) if np.allclose(std[0], std[1], rtol=0.05)
           else PH.nn_spacing(z, seed=0))
    h = a.pe_auto_grid_pitch_nn * dnn
    while (S_sec.max(axis=0) - S_sec.min(axis=0)).max() / std.min() / h > 1800:
        h *= 1.5
    D = PH.robust_diameter(z)
    density = len(rows) / (_m.pi * (D / 2.0) ** 2)
    scores, _evr, _t = PH.lognorm_pcs(counts_file, rows, a.pe_auto_n_pcs,
                                      100_000, 20_000, 0)
    lags, curves, stats = None, [], []
    for k in range(a.pe_auto_n_pcs):
        lg, Ck = PH.radial_correlogram(z, scores[:, k], h)
        lags = lg if lags is None else lags
        curves.append(Ck)
        sc = PH.correlogram_scales(lg, Ck); sc["pc"] = k
        stats.append(sc)
    keep = [p for p in stats if np.isfinite(p["rho1"])
            and p["rho1"] >= a.pe_auto_min_spatial_rho]
    if not keep:
        keep = sorted([p for p in stats if np.isfinite(p["cov1"])],
                      key=lambda p: -p["cov1"])[:1]
    L = min(len(curves[p["pc"]]) for p in keep)
    pooled = PH.correlogram_scales(
        lags[:L], np.sum([np.nan_to_num(curves[p["pc"]][:L]) for p in keep], axis=0))
    ell_c = float(pooled["ell_c"])
    lam_min = max(a.pe_auto_c_min * ell_c, 2.0 * dnn,
                  _m.sqrt(a.pe_auto_n_min_cells / density))
    g = PH.band_grid(lam_min, a.pe_auto_c_max * D, a.pe_auto_bands_per_octave)
    return (dict(sigma=float(g["sigma"]), num_freqs=int(g["num_freqs"]),
                 extra_wavelengths=[float(x) for x in g["extra_wavelengths"]]),
            dict(ell_c=round(ell_c, 5), lam_min=round(lam_min, 5),
                 n_kept_pcs=len(keep), n_bands=int(g["n_bands"]),
                 binds=("ell_c" if a.pe_auto_c_min * ell_c
                        >= max(2.0 * dnn, _m.sqrt(a.pe_auto_n_min_cells / density))
                        else "floor")))


def resolve_pe_auto(a, S_np, batch_np, n_batches):
    """-> (sigma, num_freqs, extras, per_batch_specs_or_None, provenance dict)."""
    import json as _json
    key = _pe_auto_fingerprint(a)
    if a.pe_cache and os.path.exists(a.pe_cache):
        try:
            c = _json.load(open(a.pe_cache))
            if c.get("key") == key:
                print(f"--pe_auto: cache HIT {a.pe_cache}", flush=True)
                r = c["result"]
                return (r.get("sigma"), r.get("num_freqs"), r.get("extras", []),
                        ({int(k): v for k, v in r["per_batch"].items()}
                         if r.get("per_batch") else None), c.get("prov", {}))
            print("--pe_auto: cache STALE (inputs changed) -> recomputing", flush=True)
        except Exception as e:
            print(f"--pe_auto: cache unreadable ({e}) -> recomputing", flush=True)

    # The heuristic is part of this package (it is the method, not a script),
    # so it imports directly -- no sys.path walk-up needed.
    from . import heuristic as PH

    multi = n_batches and n_batches > 1
    prov, per_batch, sigma_o, nf_o, extras_o = {}, None, None, None, []
    if multi and a.pe_auto == "single":
        # requested: auto only for single-section data; plain ladder for multi-batch
        sigma_o, nf_o, extras_o = 0.18, 16, []
        prov["mode"] = "single -> multi-batch data, using the plain sigma=0.18/N=16 ladder"
        print("--pe_auto=single with multi-batch data: using sigma=0.18 num_freqs=16 "
              "(no extras)", flush=True)
    elif multi:
        per_batch = {}
        bf = batch_np.reshape(-1)
        for b in range(n_batches):
            rows = np.where(bf == b)[0]
            spec, info = _pe_auto_one(PH, S_np[rows].astype(np.float64), rows,
                                      a.counts_file, a)
            per_batch[b] = spec
            prov[f"b{b}"] = info
            print(f"--pe_auto: batch {b} ell_c={info['ell_c']} lam_min={info['lam_min']} "
                  f"({info['binds']}) -> {info['n_bands']} bands", flush=True)
        prov["mode"] = "both -> per-batch"
    else:
        rows = np.arange(S_np.shape[0])
        spec, info = _pe_auto_one(PH, S_np.astype(np.float64), rows, a.counts_file, a)
        sigma_o, nf_o, extras_o = spec["sigma"], spec["num_freqs"], spec["extra_wavelengths"]
        prov["b0"] = info; prov["mode"] = "single-section"
        print(f"--pe_auto: ell_c={info['ell_c']} lam_min={info['lam_min']} "
              f"({info['binds']}) -> {info['n_bands']} bands", flush=True)

    if a.pe_cache:
        _json.dump(dict(key=key, prov=prov,
                        result=dict(sigma=sigma_o, num_freqs=nf_o, extras=extras_o,
                                    per_batch={str(k): v for k, v in per_batch.items()}
                                    if per_batch else None)),
                   open(a.pe_cache, "w"), indent=1)
        print(f"--pe_auto: cached -> {a.pe_cache}", flush=True)
    return sigma_o, nf_o, extras_o, per_batch, prov
