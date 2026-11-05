"""Negative-binomial observation model.

=== ADAPTATION SEAM (spatial proteomics) ===
The count assumption is encoded in THREE places, which must change together:

  1. `nb_log_prob` here -- used by the reconstruction / val-NLL metric.
  2. the generative `pyro.sample("obs", dist.GammaPoisson(...))` site inside
     `GASTON_VI.model()` (gaston_vi/model/gaston_vi.py), plus the `theta_g`
     pyro.param it depends on.
  3. the log1p-CP10K normalisation in `lognorm_pcs` (gaston_vi/pe/auto.py).

(1) and (2) describe the SAME distribution from two sides -- change one without
the other and the model and its reported NLL silently disagree, with no error.
A likelihood without a per-feature dispersion also has to drop `theta_g`, which
the caller (`nb_loss` in train.py) reads out of the pyro param store by name.

GammaPoisson(concentration=theta, rate=theta/mu) is NB with mean mu and
dispersion theta, which is why the generative side and nb_log_prob look
different while describing the same thing.
"""
import math

import torch


def nb_log_prob(y, mu, theta, eps=1e-8):
    mu = torch.clamp(mu, min=eps)
    return (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y     * (torch.log(mu)    - torch.log(theta + mu))
    )


# --- codex_2 (2026-09-29): Gaussian counterpart for CODEX/proteomics data ---
# `y` here is arcsinh+min-max-scaled protein intensity in [0,1], not a raw
# count, so the NB/GammaPoisson site above does not apply (see gaston_vi.py's
# ADAPTATION SEAM docstring). This is the evaluation-side twin of the
# `dist.Normal(mu, sigma_g)` obs site added to GASTON_VI.model() for
# likelihood="gaussian"; keep the two in sync the same way nb_log_prob and
# the GammaPoisson site must stay in sync.
_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob(y, mu, sigma, eps=1e-8):
    sigma = torch.clamp(sigma, min=eps)
    return -0.5 * (_LOG_2PI + 2.0 * torch.log(sigma) + ((y - mu) / sigma) ** 2)
