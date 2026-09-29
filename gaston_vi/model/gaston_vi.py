"""The GASTON-VI model: pyro model/guide over (isodepth d, latent z, library l).

log rho = f(d, condition) + h(z, batch), observed through a negative binomial.
Two-phase training (see train.py): phase 1 fits d alone, then z is switched on.

=== ADAPTATION SEAM (spatial proteomics) ===
The generative likelihood is the `pyro.sample("obs", dist.GammaPoisson(...))`
line in `model()` below, together with the `theta_g` pyro.param above it.
Changing it must be done in step with `nb_log_prob` in likelihood.py, which is
the evaluation-side copy of the same distribution. See that module's docstring.

Only 1-D isodepth is used in practice, but the n_isodepth>1 branches are kept
because the checkpoint/latent layout depends on them.
"""
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

from .encoders import SpatialEncoder, BatchedSpatialEncoder, BiologicalEncoder, LibrarySizeEncoder
from .decoders import LinearBiologicalDecoder


class GASTON_VI(nn.Module):
    def __init__(self, g_dim, s_dim,
                 n_isodepth=1, z_dim=0,
                 hidden_dim_spatial=20,
                 n_layers_bio=2, n_hidden_bio=128,
                 n_layers_lib=2, n_hidden_lib=128,
                 dropout_rate=0.1,
                 hidden_dim_decoder=20,
                 log_lib_mean=0.0, log_lib_var=1.0,
                 beta=0.01,
                 decoder_type="linear_hz",
                 n_batches=0,
                 n_conditions=0, batch_to_condition=None):
        super().__init__()
        self.g_dim        = g_dim
        self.n_isodepth   = n_isodepth
        self.z_dim        = z_dim
        self.beta         = beta
        self.n_batches    = n_batches
        self.n_conditions = n_conditions

        self.register_buffer("log_lib_mean",
                             torch.tensor(log_lib_mean, dtype=torch.float32))
        self.register_buffer("log_lib_var",
                             torch.tensor(log_lib_var,  dtype=torch.float32))

        def _make_spatial_encoder():
            if n_batches > 0:
                return BatchedSpatialEncoder(s_dim, hidden_dim_spatial, n_batches)
            return SpatialEncoder(s_dim, hidden_dim_spatial)

        if n_isodepth == 1:
            self.spatial_encoder = _make_spatial_encoder()
        else:
            self.spatial_encoders = nn.ModuleList([
                _make_spatial_encoder() for _ in range(n_isodepth)
            ])
            print(f"n_isodepth={n_isodepth}: {n_isodepth} independent spatial encoders",
                  flush=True)
        if n_batches > 0:
            print(f"Per-batch spatial encoders: {n_batches} encoders per isodepth axis",
                  flush=True)

        if z_dim > 0:
            self.bio_encoder = BiologicalEncoder(g_dim, z_dim, n_layers_bio,
                                                 n_hidden_bio, dropout_rate,
                                                 n_batches=n_batches)
        else:
            self.bio_encoder = None
            print("z_dim=0: no BiologicalEncoder (spatial-only baseline)", flush=True)

        self.lib_encoder = LibrarySizeEncoder(g_dim, n_layers_lib,
                                              n_hidden_lib, dropout_rate,
                                              n_batches=n_batches)

        # linear_hz is the only decoder in use (323/323 runs); the additive and
        # combined variants were removed 2026-09-28.
        if decoder_type != "linear_hz":
            raise ValueError(f"decoder_type={decoder_type!r} is no longer supported; "
                             "only 'linear_hz' remains")
        self.decoder = LinearBiologicalDecoder(
            n_isodepth, z_dim, g_dim, hidden_dim_decoder,
            n_batches=n_batches,
            n_conditions=n_conditions,
            batch_to_condition=batch_to_condition)
        print(f"Decoder: {decoder_type}", flush=True)
        if n_batches > 0:
            print(f"scVI encoders batch-conditioned: n_batches={n_batches}", flush=True)
        if n_conditions > 0:
            print(f"f(d) conditioned on condition: n_conditions={n_conditions} "
                  f"(separate zonation curve per condition)", flush=True)

    def encode_d_mu(self, s, b=None):
        if self.n_isodepth == 1:
            return self.spatial_encoder(s, b)[0]
        return torch.cat([enc(s, b)[0] for enc in self.spatial_encoders], dim=-1)

    def model(self, x, s, b=None, beta=None, phase1=False):
        if beta is None:
            beta = self.beta
        pyro.module("decoder",      self.decoder)
        pyro.module("lib_encoder",  self.lib_encoder)
        if self.z_dim > 0 and not phase1:
            pyro.module("bio_encoder", self.bio_encoder)

        theta_g = pyro.param("theta_g", x.new_ones(x.shape[1]),
                             constraint=dist.constraints.positive)

        with pyro.plate("data", x.shape[0]):
            if self.n_isodepth == 1:
                with pyro.poutine.scale(scale=beta):
                    d = pyro.sample("d",
                        dist.Normal(x.new_zeros(x.shape[0], 1),
                                    x.new_ones( x.shape[0], 1)).to_event(1))
            else:
                d_parts = []
                with pyro.poutine.scale(scale=beta):
                    for k in range(self.n_isodepth):
                        d_k = pyro.sample(f"d_{k}",
                            dist.Normal(x.new_zeros(x.shape[0], 1),
                                        x.new_ones( x.shape[0], 1)).to_event(1))
                        d_parts.append(d_k)
                d = torch.cat(d_parts, dim=-1)

            if self.z_dim > 0 and not phase1:
                z = pyro.sample("z",
                    dist.Normal(x.new_zeros(x.shape[0], self.z_dim),
                                x.new_ones( x.shape[0], self.z_dim)).to_event(1))
            else:
                z = x.new_zeros(x.shape[0], 0)

            if self.n_batches > 0:
                b_flat        = b.view(-1).long()
                l_prior_mu    = self.log_lib_mean[b_flat].view(-1, 1)
                l_prior_sigma = self.log_lib_var[b_flat].sqrt().view(-1, 1)
            else:
                l_prior_mu    = self.log_lib_mean.expand(x.shape[0], 1)
                l_prior_sigma = self.log_lib_var.sqrt().expand(x.shape[0], 1)
            log_l = pyro.sample("log_l",
                dist.Normal(l_prior_mu, l_prior_sigma).to_event(1))
            l = torch.exp(log_l)

            rho   = self.decoder(d, z, b)
            mu    = l * rho
            theta = theta_g.unsqueeze(0).expand_as(mu)
            rate  = theta / (mu + 1e-8)
            pyro.sample("obs",
                dist.GammaPoisson(concentration=theta, rate=rate).to_event(1),
                obs=x)

    def guide(self, x, s, b=None, beta=None, phase1=False):
        if beta is None:
            beta = self.beta
        pyro.module("lib_encoder", self.lib_encoder)
        if self.z_dim > 0 and not phase1:
            pyro.module("bio_encoder", self.bio_encoder)

        x_log1p = torch.log1p(x)

        with pyro.plate("data", x.shape[0]):
            if self.n_isodepth == 1:
                pyro.module("spatial_encoder", self.spatial_encoder)
                with pyro.poutine.scale(scale=beta):
                    d_mu, d_sigma = self.spatial_encoder(s, b)
                    pyro.sample("d", dist.Normal(d_mu, d_sigma).to_event(1))
            else:
                pyro.module("spatial_encoders", self.spatial_encoders)
                with pyro.poutine.scale(scale=beta):
                    for k, enc in enumerate(self.spatial_encoders):
                        d_mu_k, d_sigma_k = enc(s, b)
                        pyro.sample(f"d_{k}",
                                    dist.Normal(d_mu_k, d_sigma_k).to_event(1))

            if self.z_dim > 0 and not phase1:
                z_mu, z_sigma = self.bio_encoder(x_log1p, b)
                pyro.sample("z", dist.Normal(z_mu, z_sigma).to_event(1))

            l_mu, l_sigma = self.lib_encoder(x_log1p, b)
            pyro.sample("log_l", dist.Normal(l_mu, l_sigma).to_event(1))
