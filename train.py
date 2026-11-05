#!/usr/bin/env python3
"""GASTON-VI training entry point.

Fits an isodepth VAE to spatial expression: per-batch spatial encoders map
coordinates to a 1-D isodepth d, and the decoder factorises expression as

    log rho = f(d, condition) + h([z, batch])

with a negative-binomial observation model. Training is two-phase: d alone for
--warmup_epochs, then the biological latent z is switched on.

Handles single-section AND joint multi-section data through one path -- pass
--batch_file to get per-batch spatial encoders, omit it for a single section.

Model code lives in the `gaston_vi` package next to this file:
    gaston_vi/pe/        positional-encoding bands + the --pe_auto resolver
    gaston_vi/model/     encoders, the linear_hz decoder, the pyro model/guide,
                         and the NB likelihood
    gaston_vi/viz/       isodepth / latent / loss plots
This script keeps argument parsing, data loading, the training loop, and result
aggregation. Adapting the model to another modality (e.g. spatial proteomics)
means editing the package, not this file -- see the "ADAPTATION SEAM" notes in
gaston_vi/model/likelihood.py, gaston_vi/model/gaston_vi.py and
gaston_vi/pe/auto.py.

POSITIONAL ENCODING. Coordinates are z-scored per batch, so wavelengths are in
units of one coordinate s.d. The sigma ladder lambda_j = sigma^(-j/N) can only
produce lambda > 1 u, which is far coarser than the ~0.08-0.17 u zonation scale
on every dataset tested; sub-1u bands therefore have to come from
--extra_wavelengths or --pe_auto. Three ways to set the bands:

    --sigma / --num_freqs [--extra_wavelengths]
        Manual. With no extras this reproduces the original at_model.py
        bit-for-bit (enforced by pe_heuristic/regression.py, case eq5).
    --pe_per_batch_json PATH
        JSON {"<batch>": {"sigma": .., "num_freqs": .., "extra_wavelengths":
        [..]}, ...}. Each batch gets its own PE built from its own band set,
        zero-padded to a common width so the per-batch spatial encoders see a
        fixed s_dim. Cells are routed through their batch's PE exactly as they
        are routed through their batch's spatial encoder.
    --pe_auto {none,single,both} [--pe_cache PATH]
        Derive the bands from the data via pe_heuristic.py. `single` applies it
        to single-section data only and falls back to the plain sigma=0.18/N=16
        ladder when --batch_file is given; `both` also derives per-batch bands
        for joint data. --pe_cache memoises the result, keyed on a fingerprint
        of the counts/coords/batch files plus every heuristic knob.

    Bands are ordered by DESCENDING frequency (ladder in native order, extras
    appended on the fine side) and stored in the checkpoint as `pe_freq_bands`
    (+ `pe_version`, `pe_band_order`); near-duplicates merge at 1e-4 relative.
    Band ORDER is load-bearing -- it changes the seeded init and therefore the
    whole trajectory. The older "v1" script sorted ascending and did NOT
    reproduce at_model.py.

CONDITIONS. --condition_file is an optional .npy of integer condition labels,
shape (N,), with each batch nested entirely within one condition. When given,
(1) the cross-batch isodepth MMD alignment (--batch_align_weight) is restricted
to batch pairs sharing a condition, and (2) the condition one-hot is
concatenated to d in the f branch, giving a separate zonation curve per
condition. The condition is never passed as its own per-spot tensor; the
decoder stores the batch -> condition map and derives the one-hot from the
batch index it already receives.

    NOTE: because the MMD only aligns d WITHIN a condition, the per-condition f
    curves live on uncalibrated d axes -- f at d=0.5 in one condition is not the
    same anatomical depth as d=0.5 in another. Compare curves per condition; do
    not overlay them at "matched d" without a cross-condition anchor.

Only the `linear_hz` decoder and 1-D isodepth are supported; the additive and
combined decoders were removed 2026-09-28.
"""

import os, sys, random, time
import json
import numpy as np
import torch
import torch.nn.functional as F
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy import sparse

# The gaston_vi package sits next to this script. Python already puts the
# script's own directory on sys.path when it is run directly; this insert makes
# the package importable when it is not (invoked via a symlink, or imported).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    """Seed python, numpy, and torch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================torch
# 0.  ARGS
# ============================================================
parser = argparse.ArgumentParser(description="GASTON-VI sequential training (batch-aware)")

# -- data --
parser.add_argument("--coords_file", type=str, required=True)
parser.add_argument("--counts_file", type=str, required=True)
parser.add_argument("--batch_file",  type=str, default=None,
    help="Optional .npy of integer batch indices, shape (N,). "
         "If provigded, scVI encoders are conditioned on batch.")
parser.add_argument("--condition_file", type=str, default=None,
    help="Optional .npy of integer condition labels, shape (N,) (e.g. "
         "0=healthy, 1=diseased). Each batch must lie entirely within one "
         "condition. When provided, the cross-batch isodepth MMD alignment "
         "(--batch_align_weight) is restricted to batch pairs within the same "
         "condition. When omitted, MMD aligns across all batches.")

# -- model --
parser.add_argument("--z_dim", type=int, default=0)
parser.add_argument("--n_isodepth", type=int, default=1)
parser.add_argument("--sigma", type=float, default=None, help="required unless --pe_per_batch_json")
parser.add_argument("--num_freqs", type=int, default=None, help="required unless --pe_per_batch_json")
parser.add_argument("--pe_per_batch_json", type=str, default=None,
    help="per-batch PE spec JSON (see module docstring); requires --batch_file")
parser.add_argument("--pe_auto", choices=["none", "single", "both"], default="none",
    help="derive the PE band set from the data with pe_heuristic instead of "
         "passing --sigma/--num_freqs/--extra_wavelengths by hand.  "
         "'none' (default) = explicit flags, current behaviour.  "
         "'single' = auto ONLY for single-section data; multi-batch data falls "
         "back to the plain sigma=0.18 / num_freqs=16 ladder.  "
         "'both' = auto everywhere, deriving one PE PER BATCH when a batch_file "
         "is given.  Mutually exclusive with --sigma/--num_freqs.")
parser.add_argument("--pe_cache", type=str, default=None,
    help="JSON cache for --pe_auto.  Stores a FINGERPRINT of every input the "
         "result depends on (counts/coords/batch path+size+mtime and the "
         "heuristic knobs); the cache is reused only on an exact match, so "
         "regenerating the counts file or sweeping a knob recomputes rather "
         "than silently reusing stale bands.")
parser.add_argument("--pe_auto_c_min", type=float, default=1.0)
parser.add_argument("--pe_auto_c_max", type=float, default=1.0)
parser.add_argument("--pe_auto_bands_per_octave", type=float, default=3.0)
parser.add_argument("--pe_auto_n_pcs", type=int, default=10)
parser.add_argument("--pe_auto_min_spatial_rho", type=float, default=0.05)
parser.add_argument("--pe_auto_n_min_cells", type=float, default=50.0)
parser.add_argument("--pe_auto_grid_pitch_nn", type=float, default=2.0)
parser.add_argument("--extra_wavelengths", type=str, default="",
    help="Optional extra positional-encoding bands given directly as "
         "WAVELENGTHS in normalised coordinate units, comma- or space-"
         "separated, e.g. \"0.15,0.2,0.3\". These are appended to the "
         "log-frequency ladder built from --sigma/--num_freqs. Needed because "
         "a single sigma can only produce wavelengths on one side of "
         "lambda=1 (sigma<1 -> all lambda>1, sigma>1 -> all lambda<1), so a "
         "band straddling 1 is unreachable with sigma alone. Empty = none.")
parser.add_argument("--extra_wavelengths_geomspace", type=str, default="",
    help="Convenience alternative to --extra_wavelengths: 'lo,hi,n' builds n "
         "log-spaced wavelengths from lo to hi inclusive, e.g. "
         "\"0.15,2.12,26\". Combined with --extra_wavelengths if both given. "
         "Empty = none.")
parser.add_argument("--beta", type=float, required=True)
parser.add_argument("--hidden_dim_spatial", type=int, default=None,
                    help="width of each spatial encoder. Default depends on the "
                         "data: 64 when --batch_file is given (multi-batch, one "
                         "encoder per section), 20 for a single section. Every "
                         "joint result in this project used 64, so do not leave "
                         "this at the single-section width for joint runs.")
parser.add_argument("--hidden_dim_decoder", type=int, default=20)

# -- training --
parser.add_argument("--warmup_epochs", type=int, default=50)
parser.add_argument("--epochs",     type=int,   default=175)
parser.add_argument("--repeats",    type=int,   default=1)
parser.add_argument("--rep_offset", type=int,   default=0,
    help="start the repeat loop at this index instead of 0: rep r uses seed 42+r and is "
         "saved as rep{r+1}. --rep_offset 1 --repeats 1 runs ONLY seed 43 into rep2, "
         "leaving an existing rep1 (seed 42) untouched.")
parser.add_argument("--batch_size", type=int,   required=True)
parser.add_argument("--val_frac",   type=float, default=0.3)

# -- output --
parser.add_argument("--results_dir", type=str, required=True)

# -- decoder type --
parser.add_argument("--decoder_type", type=str, default="linear_hz",
    choices=["linear_hz"],
    help="only linear_hz remains; the combined and additive decoders were "
         "removed 2026-09-28 (unused in 323/323 runs)")
parser.add_argument("--gene_names_file", type=str, default=None)
parser.add_argument("--likelihood", type=str, default="nb", choices=["nb", "gaussian"],
    help="'gaussian' is the CODEX/proteomics branch for data that is already "
         "arcsinh + per-protein min-max scaled to [0,1] (not raw counts); see "
         "gaston_vi/model/gaston_vi.py's ADAPTATION SEAM note and "
         "gaston_vi/preprocess/codex.py for the preprocessing it expects.")

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
         "batch_file with >=2 batches is provided. If a condition_file is also "
         "given, the penalty is restricted to batch pairs within the same "
         "condition (never across conditions). Gradients from this term "
         "flow to the spatial encoders only.")
parser.add_argument("--d_tail_weight", type=float, default=0.0,
    help="Weight on a squared-hinge penalty, d_tail_weight * relu(|d| - "
         "d_tail_cutoff)^2 (mean over the minibatch), discouraging the "
         "learned isodepth from developing long outlier tails. 0 disables it. "
         "Added to the same spatial-encoder-only optimiser step as "
         "--batch_align_weight; also active (optimiser created) when "
         "batch_align_weight==0 but this is nonzero, as long as >=2 batches.")
parser.add_argument("--d_tail_cutoff", type=float, default=0.0,
    help="Cutoff kappa for --d_tail_weight's penalty, AND (independently) "
         "the value d is clamped to (clamp(-kappa, kappa)) before being fed "
         "to the decoder in the generative model -- a safety net against the "
         "decoder extrapolating for outlier d. 0 disables both the penalty "
         "term (regardless of --d_tail_weight) and the clamp.")
parser.add_argument("--project_d", action="store_true", default=False,
    help="Once per epoch (after the minibatch loop), rescale each batch's "
         "SpatialEncoder final linear layer so its d_mu has mean=0, std=1 "
         "over that batch's full training set -- an explicit per-epoch "
         "re-standardization on top of whatever the encoder itself learned. "
         "Requires a batch_file.")

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

def _parse_wavelength_list(spec):
    """'0.15,0.2 0.3' -> [0.15, 0.2, 0.3]. Empty/None -> []."""
    if spec is None:
        return []
    spec = spec.strip().strip("[]")
    if not spec:
        return []
    out = []
    for tok in (t for t in spec.replace(",", " ").split() if t):
        v = float(tok)
        if v <= 0:
            raise ValueError(f"--extra_wavelengths: wavelengths must be > 0, got {v}")
        out.append(v)
    return out


def _parse_geomspace(spec):
    """'lo,hi,n' -> n log-spaced wavelengths. Empty/None -> []."""
    if spec is None or not spec.strip():
        return []
    toks = [t for t in spec.replace(",", " ").split() if t]
    if len(toks) != 3:
        raise ValueError("--extra_wavelengths_geomspace expects 'lo,hi,n', "
                         f"got {spec!r}")
    lo, hi, n = float(toks[0]), float(toks[1]), int(round(float(toks[2])))
    if lo <= 0 or hi <= 0 or n < 1:
        raise ValueError("--extra_wavelengths_geomspace needs lo>0, hi>0, n>=1")
    return np.geomspace(lo, hi, n).tolist()


args = parser.parse_args()

z_bio_dim          = args.z_dim
decoder_type       = args.decoder_type
n_isodepth         = args.n_isodepth
sigma              = args.sigma
num_freqs          = args.num_freqs
extra_wavelengths  = sorted({round(v, 6) for v in
                             (_parse_wavelength_list(args.extra_wavelengths)
                              + _parse_geomspace(args.extra_wavelengths_geomspace))})
if args.pe_auto != "none":
    if args.sigma is not None or args.num_freqs is not None:
        raise ValueError("--pe_auto is mutually exclusive with --sigma/--num_freqs; "
                         "pass one or the other, never both")
    if args.pe_per_batch_json is not None:
        raise ValueError("--pe_auto and --pe_per_batch_json both set the bands; "
                         "use --pe_auto both to derive per-batch specs from data")
    if args.extra_wavelengths or args.extra_wavelengths_geomspace:
        raise ValueError("--pe_auto is mutually exclusive with --extra_wavelengths*")

pe_per_batch = None
if args.pe_per_batch_json is not None:
    import json as _json
    with open(args.pe_per_batch_json) as _f:
        _raw = _json.load(_f)
    _raw = _raw.get("per_batch_recommendation", _raw)   # accept the full pe_heuristic JSON too
    pe_per_batch = {int(k): dict(sigma=float(v["sigma"]), num_freqs=int(v["num_freqs"]),
                                 extra_wavelengths=[float(x) for x in v.get("extra_wavelengths", [])])
                    for k, v in _raw.items()}
    if args.batch_file is None:
        raise ValueError("--pe_per_batch_json requires --batch_file")
    # placeholders for the shared-PE bookkeeping (recorded but unused)
    if sigma is None:     sigma = float("nan")
    if num_freqs is None: num_freqs = 0
elif args.pe_auto != "none":
    # --pe_auto supplies sigma/num_freqs/extras later, once the data is loaded
    # (resolve_pe_auto needs the coords/counts/batch arrays). Placeholders until then.
    if sigma is None:     sigma = float("nan")
    if num_freqs is None: num_freqs = 0
elif sigma is None or num_freqs is None:
    raise ValueError("--sigma and --num_freqs are required unless --pe_per_batch_json "
                     "or --pe_auto is given")
beta               = args.beta
# The spatial-encoder width depends on the data, so resolve it here rather than
# in the slurm wrapper -- running train.py directly on multi-batch data used to
# silently get the single-section width of 20, which is NOT the configuration any
# joint result in this project was produced with. Passing the flag explicitly
# still overrides, so this changes nothing for the existing runners (both always
# pass it).
if args.hidden_dim_spatial is None:
    args.hidden_dim_spatial = 64 if args.batch_file is not None else 20
    print(f"--hidden_dim_spatial not given -> {args.hidden_dim_spatial} "
          f"({'multi-batch' if args.batch_file is not None else 'single-section'}"
          f" default)", flush=True)
hidden_dim_spatial = args.hidden_dim_spatial
hidden_dim_decoder = args.hidden_dim_decoder
WARMUP_EPOCHS      = args.warmup_epochs
EPOCHS         = args.epochs
REPEATS        = args.repeats
REP_OFFSET     = args.rep_offset
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
if pe_per_batch is not None:
    print(f"PER-BATCH positional encoding from {args.pe_per_batch_json}:", flush=True)
    for _b in sorted(pe_per_batch):
        _v = pe_per_batch[_b]
        print(f"  batch {_b}: sigma={_v['sigma']:.6g} num_freqs={_v['num_freqs']} "
              f"extras(n={len(_v['extra_wavelengths'])})={_v['extra_wavelengths']}", flush=True)
elif extra_wavelengths:
    print(f"extra_wavelengths (n={len(extra_wavelengths)}): {extra_wavelengths}",
          flush=True)
elif args.pe_auto != "none":
    # sigma/num_freqs are still placeholders here; the real values are printed by
    # resolve_pe_auto once the data is loaded.
    print(f"extra_wavelengths: deferred to --pe_auto {args.pe_auto}", flush=True)
else:
    print("extra_wavelengths: none (pure sigma ladder, "
          f"lambda in [{sigma ** (-1.0 / num_freqs):.4f}, {1.0 / sigma:.4f}])",
          flush=True)
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
    # scVI's Encoder indexes cat covariates as cat.size(1), so each batch
    # covariate must be 2-D (N, 1) rather than 1-D (N,).
    batch_np = batch_np.reshape(-1, 1)
    print(f"Loaded batch vector: N={batch_np.shape[0]}, n_batches={n_batches}",
          flush=True)
else:
    batch_np  = None
    n_batches = 0
    print("No batch vector provided (encoders will not be batch-conditioned)",
          flush=True)

# -- condition annotation (optional) --
# Used ONLY to restrict the cross-batch isodepth MMD alignment to batches that
# share a condition (e.g. healthy vs diseased). Not fed to encoders/decoder.
# Expects a per-spot vector of length N (parallel to batch_file); each batch is
# assumed nested within a single condition, which is asserted below.
if args.condition_file is not None:
    print(f"Loading condition from: {args.condition_file}", flush=True)
    condition_np = np.load(args.condition_file).astype(np.int64).reshape(-1)
    assert condition_np.shape[0] == A_raw.shape[0], (
        f"condition shape {condition_np.shape} != (N,) with N={A_raw.shape[0]}")
    if batch_np is None:
        raise ValueError(
            "--condition_file requires --batch_file: the condition only "
            "restricts the cross-batch isodepth MMD, which needs batch labels.")
    n_conditions = int(condition_np.max()) + 1
    # Build the batch -> condition map by reading off the condition of the spots
    # in each batch; each batch must resolve to exactly one condition.
    batch_flat_load = batch_np.reshape(-1)
    batch_to_condition = np.full(n_batches, -1, dtype=np.int64)
    for b in range(n_batches):
        cb = condition_np[batch_flat_load == b]
        if cb.size == 0:
            continue
        uniq = np.unique(cb)
        assert uniq.size == 1, (
            f"batch {b} spans multiple conditions {uniq.tolist()}; each batch "
            f"must be nested within a single condition")
        batch_to_condition[b] = int(uniq[0])
    print(f"Loaded condition vector: N={condition_np.shape[0]}, "
          f"n_conditions={n_conditions}", flush=True)
    print("Batch -> condition map:", flush=True)
    for b in range(n_batches):
        print(f"  batch {b} -> condition {int(batch_to_condition[b])}", flush=True)
    print("Cross-batch isodepth MMD will be restricted to within-condition "
          "batch pairs.", flush=True)
else:
    condition_np       = None
    n_conditions       = 0
    batch_to_condition = None
    print("No condition vector provided (cross-batch MMD, if any, aligns across "
          "all batches)", flush=True)

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
# The NB log-density itself lives in the package (it is the adaptation seam);
# the loops that apply it are harness and stay here, next to extract_latents.
from gaston_vi.model import nb_log_prob, gaussian_log_prob            # noqa: E402


def iterate_minibatches(X, S, batch_size, B=None, shuffle=True):
    """Yields (x, s, b) tuples; b is None if no batch vector is provided.

    The shuffle draws from the global torch RNG seeded by set_seed(), so the
    iteration order is part of the seeded trajectory -- do not switch it to
    numpy or a local Generator without re-running pe_heuristic/regression.py.
    """
    N   = X.shape[0]
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    for start in range(0, N, batch_size):
        bi = idx[start:start + batch_size]
        yield X[bi], S[bi], (B[bi] if B is not None else None)


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


def gaussian_loss(model, X, S, batch_size, B=None, eps=1e-8):
    """Gaussian twin of nb_loss (average per-element NLL), for
    likelihood="gaussian". Mirrors GASTON_VI.model()'s obs site: mu = rho
    (no library-size scaling -- see the note there)."""
    sigma_g = pyro.get_param_store()["sigma_g"].detach().to(X.device)
    total, denom = 0.0, 0
    for x, s, b in iterate_minibatches(X, S, batch_size, B=B, shuffle=False):
        d_mu  = model.encode_d_mu(s, b)
        z_mu  = (x.new_zeros(x.shape[0], 0) if model.z_dim == 0
                 else model.bio_encoder(torch.log1p(x), b)[0])
        mu    = model.decoder(d_mu, z_mu, b)
        sigma = sigma_g.unsqueeze(0).expand_as(mu)
        logp  = gaussian_log_prob(x, mu, sigma, eps=eps)
        total += (-logp).sum().item()
        denom += x.numel()
    return total / max(1, denom)


recon_loss_fn = nb_loss if args.likelihood == "nb" else gaussian_loss


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
# 3b. REGULARISATION TERMS  (gaston_vi.model.losses)
# ============================================================
# Disentanglement (--di_method) and cross-batch isodepth alignment
# (--batch_align_weight). Both consume the global torch RNG, so they are
# part of the seeded trajectory.
from gaston_vi.model.losses import (compute_disentanglement_reg,       # noqa: E402
                                    compute_batch_d_mmd,
                                    _spatial_encoder_params,
                                    project_spatial_encoders_epoch)

# ============================================================
# 4.  POSITIONAL ENCODING  (gaston_vi.pe)
# ============================================================
# The band ladder and the --pe_auto resolver live in the package so this
# trainer and the scoring/diagnostic scripts share one definition.
from gaston_vi.pe import (LogFrequencyPositionalEncoding,        # noqa: E402
                          BatchedPositionalEncoding)
from gaston_vi.pe.auto import resolve_pe_auto                    # noqa: E402

# ============================================================
# 5.  MODEL  (gaston_vi.model)
# ============================================================
# Encoders, the linear_hz decoder and the GASTON_VI pyro model/guide.
# gaston_vi/model/likelihood.py and the dist.GammaPoisson site inside
# GASTON_VI.model() are the spatial-proteomics adaptation seam.
from gaston_vi.model import GASTON_VI, linear_z_importance            # noqa: E402

# ============================================================
# 7.  PLOTTING HELPERS  (gaston_vi.viz)
# ============================================================
from gaston_vi.viz import (plot_isodepth, plot_isodepth_per_batch,      # noqa: E402
                           plot_all_isodepths, plot_z_latents,
                           plot_z_latents_per_batch, plot_loss_curve)
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

# ---------------- COORDINATE NORMALISATION ----------------
# This block defines the PROJECT'S UNIT SYSTEM. Coordinates are z-scored, so
# every PE wavelength downstream is measured in units of one coordinate
# standard deviation ("u"), not microns -- which is why the sigma ladder's
# floor of 1.1131 u is comparable across platforms at all, and why the
# zonation scale comes out at 0.08-0.17 u everywhere.
#
# Two things here are deliberate and easy to "fix" by mistake:
#   * The mean/std are computed on the TRAIN SPLIT ONLY and then applied to the
#     full array (S_full_norm). Using full-data statistics would leak the
#     validation split into the transform.
#   * With batches, the scaling is PER BATCH. Each section is standardised by
#     its own s.d., so 1 u is a slightly different physical distance in each
#     batch. Per-batch spatial encoders absorb that, but it means a per-batch
#     PE wavelength is only comparable across sections up to each section's
#     own scale -- do not read a shared lambda as a shared physical length.
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

# --pe_auto: derive the bands from the data now that coords/counts/batch are loaded.
# Done here (not at parse time) so it sees the same arrays the model trains on.
pe_auto_prov = None
if args.pe_auto != "none":
    _sig, _nf, _ex, _pb, pe_auto_prov = resolve_pe_auto(args, S_np, batch_np, n_batches)
    if _pb is not None:
        pe_per_batch = _pb
    else:
        sigma, num_freqs, extra_wavelengths = _sig, _nf, _ex
    print(f"--pe_auto={args.pe_auto} resolved: "
          + (f"per-batch specs for {sorted(pe_per_batch)}" if _pb is not None
             else f"sigma={sigma} num_freqs={num_freqs} n_extra={len(extra_wavelengths)}"),
          flush=True)

# PE encoding (shared, or one PE per batch when --pe_per_batch_json / --pe_auto both)
if pe_per_batch is not None:
    if set(pe_per_batch) != set(range(n_batches)):
        raise ValueError(f"--pe_per_batch_json batches {sorted(pe_per_batch)} != data batches 0..{n_batches - 1}")
    pe_encoder = BatchedPositionalEncoding(in_dim=2, specs=pe_per_batch).to(device)
    _pe = lambda s_norm, b: pe_encoder(s_norm, b)
else:
    pe_encoder = LogFrequencyPositionalEncoding(
        in_dim=2, num_frequencies=num_freqs, sigma=sigma,
        extra_wavelengths=extra_wavelengths,
    )
    _pe = lambda s_norm, b: pe_encoder(s_norm)
train_spatial_norm = _pe(train_spatial_norm, train_batch)
S_full_norm        = _pe(S_full_norm, B_full)

if val_idx is not None:
    if B_full is not None:
        vb_idx = val_batch.view(-1).long()
        val_spatial_norm = _pe((val_spatial - train_spatial_mean[vb_idx]) / train_spatial_std[vb_idx], val_batch)
    else:
        val_spatial_norm = _pe((val_spatial - train_spatial_mean)/ train_spatial_std, None)
else:
    val_spatial_norm = None

s_dim = train_spatial_norm.shape[1]
print(f"Encoded spatial dim: {s_dim}", flush=True)

# ---------------- LIBRARY-SIZE PRIOR ----------------
# Empirical N(mean, var) prior on log library size, fitted per batch (batches
# differ in sequencing depth) and passed to the model as the log_lib_mean /
# log_lib_var buffers. Also computed on the train split only.
#
# ADAPTATION SEAM (spatial proteomics): total counts per cell is a
# count-data notion. Intensity data has no library size, so an adaptation
# either drops this latent entirely or replaces it with whatever per-cell
# scale factor is appropriate. This is the fourth count assumption, alongside
# gaston_vi/model/likelihood.py, the GammaPoisson site in
# gaston_vi/model/gaston_vi.py, and lognorm_pcs in gaston_vi/pe/auto.py.
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

for rep in range(REP_OFFSET, REP_OFFSET + REPEATS):
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
        n_conditions=n_conditions,
        batch_to_condition=batch_to_condition,
        likelihood=args.likelihood,
        d_tail_cutoff=args.d_tail_cutoff,
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

    # Cross-batch isodepth alignment: optimiser restricted to spatial encoders.
    # Also created for --d_tail_weight alone, since the tail penalty needs the
    # same spatial-encoder-only step even when batch_align_weight==0.
    batch_mmd_optimizer = None
    spatial_params = None
    if (batch_align_weight != 0.0 or args.d_tail_weight != 0.0) and n_batches >= 2:
        spatial_params = _spatial_encoder_params(vae)
        batch_mmd_optimizer = torch.optim.Adam(spatial_params, lr=1e-3)
        scope = ("within-condition batch pairs"
                 if batch_to_condition is not None else "all batch pairs")
        print(f"Cross-batch isodepth MMD alignment "
              f"{'ON' if batch_align_weight != 0.0 else 'off'} "
              f"(weight={batch_align_weight}, spatial encoders only, {scope}); "
              f"d_tail penalty {'ON' if args.d_tail_weight != 0.0 else 'off'} "
              f"(weight={args.d_tail_weight}, cutoff={args.d_tail_cutoff})",
              flush=True)

    # Precompute per-batch training spatial tensors for --project_d
    S_train_by_batch = None
    if args.project_d and B_full is not None:
        _tb_flat = train_batch.view(-1)
        S_train_by_batch = [train_spatial_norm[(_tb_flat == b)]
                            for b in range(n_batches)]
        print(f"[project_d] ON, S_train_by_batch sizes: "
              f"{[s.shape[0] for s in S_train_by_batch]}", flush=True)

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
                align_loss = spatial.new_tensor(0.0)

                if batch_align_weight != 0.0:
                    mmd = compute_batch_d_mmd(
                        vae, spatial, batch,
                        batch_to_condition=batch_to_condition)
                    align_loss = align_loss + batch_align_weight * mmd
                    epoch_batch_mmd += mmd.item() * gene.shape[0]

                # Tail penalty: d_tail_weight * relu(|d| - d_tail_cutoff)^2,
                # mean over batch.
                if args.d_tail_weight > 0.0 and args.d_tail_cutoff > 0.0:
                    d_mu_batch = vae.encode_d_mu(spatial, batch)
                    tail = F.relu(d_mu_batch.abs() - args.d_tail_cutoff)
                    align_loss = align_loss + args.d_tail_weight * (tail ** 2).mean()

                batch_mmd_optimizer.zero_grad()
                align_loss.backward()
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

            num += gene.shape[0]

        elbo_history.append(epoch_loss / max(1, num))
        di_reg_history.append(epoch_di_reg / max(1, num))
        batch_mmd_history.append(epoch_batch_mmd / max(1, num))

        # Per-epoch projection (--project_d): standardize each batch's isodepth
        if args.project_d and S_train_by_batch is not None:
            project_spatial_encoders_epoch(vae, S_train_by_batch)

        vae.eval()
        with torch.no_grad():
            recon_train = recon_loss_fn(vae, train_gene, train_spatial_norm,
                                  BATCH_SIZE, B=train_batch)
            recon_train_history.append(recon_train)
            if val_idx is not None:
                recon_val = recon_loss_fn(vae, val_gene, val_spatial_norm,
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
            "n_conditions": n_conditions,
            "batch_to_condition": (batch_to_condition.copy()
                                   if batch_to_condition is not None else None),
            "likelihood": args.likelihood,
        },
        "sigma": sigma, "num_freqs": num_freqs,
        "extra_wavelengths": list(extra_wavelengths),
        "pe_per_batch": pe_per_batch,          # None => shared PE (original behaviour)
        "pe_version": 2, "pe_band_order": "desc_freq",
        "pe_freq_bands": (pe_encoder.freq_bands.cpu().numpy() if hasattr(pe_encoder, "freq_bands")
                          else [pe.freq_bands.cpu().numpy() for pe in pe_encoder.pes]),
        "pe_n_bands":    pe_encoder.n_bands,
        "pe_lambda_min": pe_encoder.lambda_min,
        "pe_lambda_max": pe_encoder.lambda_max,
        "di_method": di_method, "di_weight": di_weight,
        "batch_align_weight": batch_align_weight,
        "d_tail_weight": args.d_tail_weight, "d_tail_cutoff": args.d_tail_cutoff,
        "project_d": args.project_d,
        "warmup_epochs": WARMUP_EPOCHS,
        "train_spatial_mean": train_spatial_mean.cpu().numpy(),
        "train_spatial_std":  train_spatial_std.cpu().numpy(),
        "train_idx": train_idx.cpu().numpy(),
        "val_idx":   val_idx.cpu().numpy() if val_idx is not None else None,
        "seed": seed,
        "gene_names": gene_names,
        "batch_file": args.batch_file,
        "n_batches":  n_batches,
        "condition_file":     args.condition_file,
        "n_conditions":       n_conditions,
        "batch_to_condition": (batch_to_condition.copy()
                               if batch_to_condition is not None else None),
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
    "d_tail_weight":          args.d_tail_weight,
    "d_tail_cutoff":          args.d_tail_cutoff,
    "project_d":              args.project_d,
    "warmup_epochs":          WARMUP_EPOCHS,
    "sigma":                  sigma,
    "num_freqs":              num_freqs,
    "extra_wavelengths":      ";".join(f"{v:g}" for v in extra_wavelengths),
    "pe_source":              args.pe_auto if args.pe_auto != "none" else (
                                  "per_batch_json" if args.pe_per_batch_json else "manual"),
    "pe_auto_prov":           json.dumps(pe_auto_prov) if pe_auto_prov else "",
    "pe_per_batch":           ("" if pe_per_batch is None else " | ".join(
                                  f"b{b}:sigma={v['sigma']:g},N={v['num_freqs']},extras={';'.join(f'{x:g}' for x in v['extra_wavelengths'])}"
                                  for b, v in sorted(pe_per_batch.items()))),
    "pe_version":             2,
    "pe_band_order":          "desc_freq",
    "pe_n_bands":             pe_encoder.n_bands,
    "pe_lambda_min":          pe_encoder.lambda_min,
    "pe_lambda_max":          pe_encoder.lambda_max,
    "beta":                   beta,
    "hidden_dim_spatial":     hidden_dim_spatial,
    "hidden_dim_decoder":     hidden_dim_decoder,
    "epochs":                 EPOCHS,
    "repeats":                REPEATS,
    "val_frac":               VAL_FRAC,
    "n_batches":              n_batches,
    "batch_file":             args.batch_file if args.batch_file is not None else "",
    "n_conditions":           n_conditions,
    "condition_file":         args.condition_file if args.condition_file is not None else "",
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
