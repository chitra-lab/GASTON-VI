# GASTON-VI

Spatial VAE with a 1-D isodepth latent: expression is modelled as
`softmax( f(d) + W·[z, batch] )`, so `f(d)` is an interpretable zonation curve
and `z` carries the rest of the biological variation.

## Layout

| path | what |
|---|---|
| `train.py` | the entry point — single-section **and** joint (multi-batch) models |
| `gaston_vi/model/` | likelihood, encoders, decoders, the `GASTON_VI` module, the disentanglement + cross-batch-MMD losses |
| `gaston_vi/pe/` | positional-encoding modules, the band heuristic, the `--pe_auto` resolver and its cache |
| `gaston_vi/preprocess/` | modality-specific preprocessing (currently: CODEX/spatial-proteomics) |
| `gaston_vi/viz/` | isodepth, latent and loss plots |
| `smoke_test.py` | exercises every package entry point on synthetic arrays in ~10 s |
| `train.slurm` | slurm wrapper for one `train.py` run; edit the parameter block at the top and `sbatch` |

`train.py` puts its own directory on `sys.path`, so it runs from any working
directory as long as `gaston_vi/` sits beside it.

## Running

```bash
python train.py --coords_file coords.npy --counts_file counts.npy \
                --gene_names_file gene_names.npy \
                --z_dim 4 --n_isodepth 1 --sigma 0.18 --num_freqs 16 \
                --epochs 175 --warmup_epochs 100 --repeats 2 \
                --batch_size 2586 --val_frac 0.3 \
                --di_method simvi_mmd --di_weight 0.01 \
                --decoder_type linear_hz --results_dir out/
```

Passing `--batch_file` switches the model to joint (multi-batch) mode; batch-ness
is inferred from that flag alone. `--hidden_dim_spatial` defaults to 64 with
`--batch_file` and 20 without, so it normally does not need to be passed.

The slurm runners resolve the code location from `CODE_DIR`, which defaults to
this checkout; override it to run a different one:

```bash
CODE_DIR=/path/to/GASTON-VI sbatch train.slurm
```

## Positional-encoding bands

The sigma ladder `λ_j = σ^(−j/N)` spans `[σ^(−1/N), 1/σ]` in units of one
coordinate standard deviation, so the original `σ=0.18, N=16` gives
`λ ∈ [1.11, 5.56] u` and **cannot reach below 1 u**. Structure finer than that
(lobules, in liver) needs extra bands, which `--extra_wavelengths` supplies
explicitly and `--pe_auto` derives from the data.

`--pe_auto {none,single,both}`:

- `none` — use `--sigma`/`--num_freqs`/`--extra_wavelengths` as given.
- `single` — derive bands from the data for a single section. On multi-batch
  data this falls back to the plain `σ=0.18, N=16` ladder.
- `both` — derive bands per batch.

`--pe_cache <json>` memoises the derived bands against a fingerprint of the
input files and every knob, so a regenerated counts file invalidates the entry.
`pe_source` and `pe_auto_prov` are recorded in the run's summary CSV.

## CODEX / spatial-proteomics modality (`--likelihood gaussian`)

Protein intensities are not sequencing counts, so adapting the model meant
changing four places (paper Appendix S2.6); all four are done, cross-referenced
in the code via an "ADAPTATION SEAM" comment at each site:

1. `gaussian_log_prob` in `gaston_vi/model/likelihood.py` — the evaluation-side
   twin of `nb_log_prob`.
2. `GASTON_VI.model()`'s `dist.Normal(mu, sigma_g)` obs site (vs. the NB
   branch's `dist.GammaPoisson`), selected by `likelihood="gaussian"`; `mu` is
   the decoder output directly (`LinearBiologicalDecoder(apply_softmax=False)`
   — CODEX intensities are not compositional the way counts are, so forcing a
   sum-to-1 softmax over the markers distorts every channel's curve). (1) and
   (2) describe the same distribution from two directions; change one without
   the other and the model and its reported NLL silently disagree.
3. `lognorm_pcs(..., skip_norm=True)` in `gaston_vi/pe/auto.py` /
   `gaston_vi/pe/heuristic.py` — CP10K/log1p normalisation is meaningless for
   already-scaled intensities, so `--pe_auto` z-scores the raw values instead.
   `skip_norm` is set automatically from `--likelihood`.
4. the library-size prior in `train.py` is still fit the same way for both
   likelihoods (keeps the guide/checkpoint layout identical across runs), but
   on the gaussian branch it is sampled and then unused — the observation's
   mean is the decoder output alone, with no library-size scaling.

Two more CODEX-specific pieces, both off by default (`0` / unset):

- `--d_tail_weight`/`--d_tail_cutoff` — a squared-hinge penalty,
  `d_tail_weight * relu(|d| - d_tail_cutoff)^2`, discouraging outlier isodepth
  tails during multi-batch training; `d_tail_cutoff` also clamps `d` before
  the decoder sees it, independent of the penalty.
- `--project_d` — once per epoch, rescales each batch's spatial-encoder output
  so `d_mu` has mean 0 / std 1 over that batch's training set exactly (on top
  of whatever `--batch_align_weight`'s soft MMD alignment already did that
  epoch). Requires `--batch_file`.

`gaston_vi/preprocess/codex.py` implements the input-side preprocessing this
branch expects: arcsinh-transform (cofactor 1), per-protein 1st/99th-percentile
clip, per-protein min-max scaling to `[0, 1]` — run once per anatomical
region/condition (percentiles and the scaling range are region-specific, not
global).
