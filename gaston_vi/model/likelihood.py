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
