#!/usr/bin/env python3
"""Fast structural check of train.py + the gaston_vi package.

    python smoke_test.py

Runs in ~1 min on CPU with no data files. It does NOT check numerical
correctness -- that is pe_heuristic/regression.py, which retrains 4 configs on
the GPU (~8 min) and requires bit-identical d_mu. This script exists to catch
import-time and NameError-class breakage in seconds instead of 4 minutes into a
GPU run, which is how a missing `import math` in viz/latents.py was found after
it had killed four jobs.

Run it after ANY refactor of the package, before submitting the GPU array.
Also useful when adapting the model to another modality: swap the likelihood
and this tells you immediately whether the wiring still holds together.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

rng = np.random.default_rng(0)
N = 200
S_xy = rng.normal(size=(N, 2))
batch = rng.integers(0, 3, size=N)
d_mu = S_xy[:, 0] + 0.1 * rng.normal(size=N)
z_mu = rng.normal(size=(N, 5))
fails = []
skips = []


def run(name, fn, *a, **k):
    # NB: catch BaseException, not Exception -- importing train.py raises
    # SystemExit from argparse, which would otherwise kill this script
    # silently (and a shell pipe can mask the non-zero exit code).
    try:
        fn(*a, **k)
        print(f"  PASS  {name}")
    except KeyboardInterrupt:
        raise
    except BaseException as ex:
        print(f"  FAIL  {name}: {type(ex).__name__}: {ex}")
        fails.append(name)


# NOTE: train.py is an EXECUTABLE, not a module -- its top level parses args and
# runs the pipeline, and nothing in the repo imports it (36 downstream scripts
# consume the saved d_mu.npy / z_mu.npy instead). So this file tests the package
# only. `iterate_minibatches` / `nb_loss` / `extract_latents` live in train.py
# and are deliberately not covered here; the end-to-end check below reproduces
# the same forward pass from package pieces.


def _nb_nll(m, X, S, B, batch_size=64):
    """Mean per-element NB NLL -- the package-only equivalent of train.nb_loss."""
    import pyro
    from gaston_vi.model import nb_log_prob
    theta_g = pyro.get_param_store()["theta_g"].detach()
    total, denom = 0.0, 0
    for i in range(0, X.shape[0], batch_size):
        x = X[i:i + batch_size]
        s = S[i:i + batch_size]
        b = None if B is None else B[i:i + batch_size]
        x_log1p = torch.log1p(x)
        d_mu = m.encode_d_mu(s, b)
        z_mu = (x.new_zeros(x.shape[0], 0) if m.z_dim == 0
                else m.bio_encoder(x_log1p, b)[0])
        l_mu, _ = m.lib_encoder(x_log1p, b)
        mu = torch.exp(l_mu) * m.decoder(d_mu, z_mu, b)
        logp = nb_log_prob(x, mu, theta_g.unsqueeze(0).expand_as(mu))
        total += (-logp).sum().item()
        denom += x.numel()
    return total / max(1, denom)


def _model(z_dim=4, n_batches=3, g_dim=40):
    """Fresh GASTON_VI with theta_g populated by one SVI step."""
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam
    from gaston_vi.model import GASTON_VI
    pyro.clear_param_store()
    X = torch.poisson(torch.full((N, g_dim), 3.0))
    S = torch.as_tensor(S_xy, dtype=torch.float32)
    B = torch.as_tensor(batch).view(-1, 1) if n_batches else None
    kw = (dict(log_lib_mean=np.zeros(n_batches), log_lib_var=np.ones(n_batches))
          if n_batches else {})
    m = GASTON_VI(g_dim=g_dim, s_dim=2, z_dim=z_dim, n_batches=n_batches, **kw)
    SVI(m.model, m.guide, Adam({"lr": 1e-3}), loss=Trace_ELBO()).step(X, S, B)
    return m, X, S, B


# ------------------------------------------------------------------ pe
print("=== gaston_vi.pe ===")
from gaston_vi.pe import (LogFrequencyPositionalEncoding,            # noqa: E402
                          BatchedPositionalEncoding)


def pe_shared():
    pe = LogFrequencyPositionalEncoding(in_dim=2, num_frequencies=16, sigma=0.18)
    assert pe(torch.as_tensor(S_xy, dtype=torch.float32)).shape == (N, pe.out_dim)


def pe_extras():
    pe = LogFrequencyPositionalEncoding(in_dim=2, num_frequencies=16, sigma=0.18,
                                        extra_wavelengths=[0.5, 0.25, 0.125])
    assert pe(torch.as_tensor(S_xy, dtype=torch.float32)).shape == (N, pe.out_dim)


def pe_batched_equiv():
    """Identical specs per batch MUST reproduce the shared PE exactly.

    This invariant is what makes joint runs comparable to single-section ones.
    """
    spec = dict(sigma=0.18, num_freqs=16, extra_wavelengths=[])
    b = BatchedPositionalEncoding(in_dim=2, specs={i: dict(spec) for i in range(3)})
    sh = LogFrequencyPositionalEncoding(in_dim=2, num_frequencies=16, sigma=0.18)
    x = torch.as_tensor(S_xy, dtype=torch.float32)
    assert torch.equal(b(x, torch.as_tensor(batch)), sh(x)), "batched != shared"


def pe_batched_ragged():
    b = BatchedPositionalEncoding(in_dim=2, specs={
        0: dict(sigma=0.18, num_freqs=16, extra_wavelengths=[]),
        1: dict(sigma=0.18, num_freqs=16, extra_wavelengths=[0.5]),
        2: dict(sigma=0.20, num_freqs=8, extra_wavelengths=[0.3, 0.1])})
    out = b(torch.as_tensor(S_xy, dtype=torch.float32), torch.as_tensor(batch))
    assert out.shape == (N, b.out_dim)


run("LogFrequencyPositionalEncoding", pe_shared)
run("LogFrequencyPositionalEncoding +extras", pe_extras)
run("BatchedPositionalEncoding == shared", pe_batched_equiv)
run("BatchedPositionalEncoding ragged specs", pe_batched_ragged)

# ------------------------------------------------------------- pe.auto
print("=== gaston_vi.pe.auto (import + fingerprint; no data on disk) ===")


def auto_fingerprint():
    import argparse
    import json
    from gaston_vi.pe.auto import _pe_auto_fingerprint
    a = argparse.Namespace(
        coords_file="/nonexistent_coords.npy", batch_file=None,
        counts_file="/nonexistent.npy", pe_auto="single", pe_cache=None,
        pe_auto_c_min=1.0, pe_auto_c_max=1.0, pe_auto_bands_per_octave=3,
        pe_auto_n_pcs=10, pe_auto_min_spatial_rho=0.05,
        pe_auto_n_min_cells=50, pe_auto_grid_pitch_nn=2.0)
    fp = _pe_auto_fingerprint(a)
    assert isinstance(fp, dict), type(fp)
    json.dumps(fp)                    # the cache key must round-trip as JSON
    assert set(fp) >= {"counts", "coords", "batch", "mode", "c_min", "c_max"}


def heuristic_is_in_package():
    """The band heuristic is part of the package -- no sys.path walk-up.

    It moved from pe_heuristic/pe_heuristic.py to gaston_vi/pe/heuristic.py on
    2026-09-29, which is what let auto.py drop its walk-up hack. It must stay
    importable BOTH as a module (auto.py does `from . import heuristic`) and as
    a CLI (`python -m gaston_vi.pe.heuristic`), so it may not gain any
    package-relative imports of its own.
    """
    from gaston_vi.pe import heuristic as PH
    for fn in ("band_grid", "lognorm_pcs", "main"):
        assert hasattr(PH, fn), f"heuristic.{fn} missing"
    g = PH.band_grid(0.07, 4.5, bands_per_octave=3)
    assert set(g) >= {"sigma", "num_freqs", "extra_wavelengths", "n_bands"}, g.keys()
    assert g["n_bands"] > 0


run("import + fingerprint", auto_fingerprint)
run("heuristic is in the package", heuristic_is_in_package)

# --------------------------------------------------------------- model
print("=== gaston_vi.model ===")


def model_build_step_and_nll():
    """Encoders + decoder + model/guide + likelihood + minibatching together."""
    from gaston_vi.model import linear_z_importance, nb_log_prob
    m, X, S, B = _model()
    nll = _nb_nll(m, X, S, B)
    assert np.isfinite(nll), nll
    assert m.encode_d_mu(S, B).shape == (N, 1)
    linear_z_importance(m.decoder)
    lp = nb_log_prob(X[:4], torch.full((4, 40), 3.0), torch.full((4, 40), 1.0))
    assert torch.isfinite(lp).all()


def model_no_batch():
    """n_batches=0: unconditional encoders, no batch one-hot."""
    m, X, S, _ = _model(n_batches=0)
    assert np.isfinite(_nb_nll(m, X, S, None))


def model_z_dim_zero():
    """z_dim=0: spatial-only baseline, no BiologicalEncoder."""
    m, X, S, B = _model(z_dim=0)
    assert m.bio_encoder is None
    assert np.isfinite(_nb_nll(m, X, S, B))


def model_rejects_dead_decoder():
    from gaston_vi.model import GASTON_VI
    try:
        GASTON_VI(g_dim=10, s_dim=2, z_dim=2, decoder_type="combined")
    except ValueError:
        return
    raise AssertionError("decoder_type='combined' should be rejected")


run("GASTON_VI build + SVI step + nb_loss", model_build_step_and_nll)
run("GASTON_VI n_batches=0 path", model_no_batch)
run("GASTON_VI z_dim=0 path", model_z_dim_zero)
run("dead decoder_type rejected", model_rejects_dead_decoder)

# -------------------------------------------------------- model.losses
print("=== gaston_vi.model.losses ===")


def losses_all_di_methods():
    from gaston_vi.model.losses import compute_disentanglement_reg
    m, X, S, B = _model()
    for meth in ("none", "simvi_mi", "simvi_mmd", "simvi_mi_kl", "simvi_mmd_kl"):
        v = compute_disentanglement_reg(meth, m, X[:64], S[:64], B[:64])
        assert torch.isfinite(v).all(), (meth, v)
        assert v.item() >= 0.0, (meth, v)
    try:
        compute_disentanglement_reg("bogus", m, X[:64], S[:64], B[:64])
    except ValueError:
        return
    raise AssertionError("unknown di_method should raise")


def losses_batch_align():
    from gaston_vi.model.losses import compute_batch_d_mmd, _spatial_encoder_params
    m, X, S, B = _model()
    v = compute_batch_d_mmd(m, S, B)
    assert torch.isfinite(v) and v.item() >= 0.0, v
    # single-batch model must short-circuit to exactly 0
    m1, X1, S1, _ = _model(n_batches=0)
    assert compute_batch_d_mmd(m1, S1, None).item() == 0.0
    # condition map must suppress cross-condition pairs
    v_same = compute_batch_d_mmd(m, S, B, batch_to_condition=[0, 0, 0])
    v_split = compute_batch_d_mmd(m, S, B, batch_to_condition=[0, 1, 2])
    assert v_split.item() == 0.0, f"all-distinct conditions should give 0, got {v_split}"
    assert torch.isfinite(v_same)
    assert len(_spatial_encoder_params(m)) > 0


run("compute_disentanglement_reg, all methods", losses_all_di_methods)
run("compute_batch_d_mmd + condition gating", losses_batch_align)

# --------------------------------------------------------------- viz
print("=== gaston_vi.viz ===")
from gaston_vi.viz import (plot_isodepth, plot_isodepth_per_batch,    # noqa: E402
                           plot_all_isodepths, plot_z_latents,
                           plot_z_latents_per_batch, plot_loss_curve)

with tempfile.TemporaryDirectory() as td:
    run("plot_isodepth", plot_isodepth, S_xy, d_mu, f"{td}/iso.png")
    run("plot_isodepth_per_batch", plot_isodepth_per_batch, S_xy, d_mu, batch, td, "pb")
    run("plot_all_isodepths", plot_all_isodepths, S_xy, d_mu.reshape(-1, 1), f"{td}/all.png")
    run("plot_z_latents", plot_z_latents, S_xy, z_mu, d_mu, f"{td}/z.png")
    run("plot_z_latents_per_batch", plot_z_latents_per_batch, S_xy, z_mu, batch, td, "zpb")
    run("plot_loss_curve", plot_loss_curve, list(np.linspace(5, 1, 50)),
        list(np.linspace(5.2, 1.3, 50)), f"{td}/loss.png")

print()
print("SMOKE FAIL: " + ", ".join(fails) if fails
      else f"SMOKE: ALL PASS ({len(skips)} skipped)" if skips
      else "SMOKE: ALL PASS")
sys.exit(1 if fails else 0)
