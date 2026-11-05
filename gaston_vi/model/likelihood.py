"""Observation log-densities, used for the reconstruction and validation NLL.

Each one mirrors the `obs` site in `GASTON_VI.model()`. Change one without the
other and the model and its reported NLL disagree, with no error.

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


# Gaussian counterpart for continuous data such as protein intensities.
# `y` is a scaled intensity in [0, 1], not a raw count. Mirrors the
# `dist.Normal(mu, sigma_g)` site in GASTON_VI.model().
_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob(y, mu, sigma, eps=1e-8):
    sigma = torch.clamp(sigma, min=eps)
    return -0.5 * (_LOG_2PI + 2.0 * torch.log(sigma) + ((y - mu) / sigma) ** 2)
