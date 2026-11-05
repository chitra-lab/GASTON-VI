"""Differential gradient score DG_m (paper Appendix S3.2, Eqs. S9-S10).

DG_m captures how much the *shape* of a feature m's (gene/protein) spatial
gradient h_m(d; c) differs across a group of conditions c (e.g. the four
anatomical regions within one tissue group), independent of any difference
in absolute expression level or dynamic range.
"""
import numpy as np


def differential_gradient_score(h, d_grid=None, d_range=None):
    """Compute DG_m for every feature m from its per-condition spatial
    functions h_m(d; c).

    For each feature m and condition c, h_m(d; c) is z-scored to mean 0 and
    standard deviation 1 across d, removing differences in absolute level and
    dynamic range so that only differences in gradient *shape* remain:

        hhat_m(d; c) = (h_m(d; c) - mu_{m,c}) / sigma_{m,c}          (S9)

    DG_m is then the variance of the z-scored curves across the conditions c,
    computed pointwise at each d and averaged over d:

        DG_m = mean_d  Var_c[ hhat_m(d; c) ]                          (S10)

    A high DG_m (close to 1) means the feature's spatial gradient differs
    substantially in shape across conditions (a true differential gradient).
    A low DG_m (close to 0) means the gradient shape is essentially shared
    across all conditions in the group.

    Parameters
    ----------
    h : (n_conditions, n_d, n_features) array
        Spatial functions h_m(d; c), evaluated on a shared d grid, one slice
        per condition c in the group being compared (e.g. the four
        conditions of one tissue group).
    d_grid : (n_d,) array, optional
        The d values h was evaluated on. Required only together with
        `d_range`.
    d_range : (lo, hi), optional
        Restrict the score to a d-window, e.g. (-1.75, 1.75) for a
        "well-supported isodepth" window. Default: use every column of `h`
        (the full grid it was evaluated on).

    Returns
    -------
    DG : (n_features,) array, each entry in [0, 1].
    """
    h = np.asarray(h, dtype=float)
    if d_range is not None:
        if d_grid is None:
            raise ValueError("d_range requires d_grid")
        mask = (d_grid >= d_range[0]) & (d_grid <= d_range[1])
        h = h[:, mask, :]

    mu = h.mean(axis=1, keepdims=True)
    sigma = h.std(axis=1, keepdims=True)
    h_z = (h - mu) / (sigma + 1e-8)           # Eq. S9
    var_over_c = h_z.var(axis=0, ddof=0)      # Var_c[hhat_m(d; c)] at each d
    return var_over_c.mean(axis=0)            # Eq. S10
