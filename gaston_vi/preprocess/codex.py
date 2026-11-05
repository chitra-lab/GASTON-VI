"""CODEX / spatial-proteomics preprocessing (paper Appendix S2.6).

Protein intensities are not sequencing counts, so they are preprocessed
differently from the NB branch's count data before being passed to
`train.py --likelihood gaussian`:

  1. arcsinh-transform every value (cofactor 1).
  2. Per-protein, clip the transformed values at the 1st/99th percentile.
  3. Per-protein, rescale the clipped values to [0, 1].

Steps 2-3 are computed separately for each anatomical region (condition),
using all cells from all batches in that region -- call `preprocess_codex`
once per region with only that region's cells. There is no library-size
concept for intensity data, so `train.py` omits the library-size encoder's
scaling for this branch (see `gaston_vi/model/gaston_vi.py`'s likelihood
note) and the decoder outputs the Gaussian mean directly, without softmax.
"""
import numpy as np

ARCSINH_COFACTOR = 1.0


def preprocess_codex(raw, clip_percentile=1.0, cofactor=ARCSINH_COFACTOR):
    """Arcsinh-transform, per-protein percentile-clip, per-protein min-max
    scale to [0, 1].

    Parameters
    ----------
    raw : (n_cells, n_proteins) array
        Raw intensities for ONE anatomical region (pool all batches/donors
        for that region, but no other region -- percentiles and the min-max
        range are computed per region, not globally).
    clip_percentile : float, default 1.0
        Clip each protein's arcsinh-transformed values at
        [clip_percentile, 100 - clip_percentile] before scaling. 0 disables
        clipping (use the true min/max).
    cofactor : float, default 1.0
        Divisor inside the arcsinh transform: arcsinh(raw / cofactor).

    Returns
    -------
    scaled : (n_cells, n_proteins) float32 array in [0, 1].
    stats : dict with 'lo', 'hi' (n_proteins,), the per-protein clip bounds
        used for scaling -- keep these if the same bounds must be applied to
        held-out cells from the same region.
    """
    raw = np.asarray(raw, dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0)
    transformed = np.arcsinh(raw / cofactor)

    if clip_percentile > 0:
        lo = np.percentile(transformed, clip_percentile, axis=0)
        hi = np.percentile(transformed, 100 - clip_percentile, axis=0)
    else:
        lo = transformed.min(axis=0)
        hi = transformed.max(axis=0)

    clipped = np.clip(transformed, lo[None, :], hi[None, :])
    denom = np.where(hi > lo, hi - lo, 1.0)
    scaled = ((clipped - lo[None, :]) / denom[None, :]).astype(np.float32)
    return scaled, {"lo": lo, "hi": hi}


def apply_codex_scaling(raw, stats, cofactor=ARCSINH_COFACTOR):
    """Apply previously-fit per-protein clip bounds (from `preprocess_codex`'s
    `stats`) to new cells from the same region, e.g. a held-out split."""
    raw = np.asarray(raw, dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0)
    transformed = np.arcsinh(raw / cofactor)
    lo, hi = stats["lo"], stats["hi"]
    clipped = np.clip(transformed, lo[None, :], hi[None, :])
    denom = np.where(hi > lo, hi - lo, 1.0)
    return ((clipped - lo[None, :]) / denom[None, :]).astype(np.float32)
