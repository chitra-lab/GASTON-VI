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
| `gaston_vi/viz/` | isodepth, latent and loss plots |
| `smoke_test.py` | exercises every package entry point on synthetic arrays in ~10 s |
| `train.slurm` | slurm wrapper for one `train.py` run; edit the parameter block at the top and `sbatch` |
| `old_code/GASTON_VI.py` | the previous single-script implementation, kept as the reference |
| `old_code/GASTON_VI.slurm` | its runner (note: it invokes an absolute path in the original project tree, not the file beside it) |

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

## Relationship to `old_code/`

`train.py` replaces the eight or nine near-duplicate training scripts that
preceded it; `old_code/GASTON_VI.py` is the last single-file generation.

## Note on the NB / count-data assumption

Adapting the model to non-count data (e.g. spatial proteomics) means changing
four places, which are cross-referenced in the code:

1. `nb_log_prob` in `gaston_vi/model/likelihood.py` — the evaluation side.
2. the `dist.GammaPoisson` sample site inside `GASTON_VI.model()` and its
   `theta_g` parameter — the generative side. (1) and (2) describe the same
   distribution from two directions; change one without the other and the model
   and its reported NLL silently disagree.
3. `lognorm_pcs` in `gaston_vi/pe/auto.py` — hardcoded log1p-CP10K.
4. the library-size prior in `train.py` — an empirical normal on log total
   counts per cell, fitted per batch. Intensity data has no library size.
