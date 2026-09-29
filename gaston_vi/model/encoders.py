"""Encoders: coordinates -> isodepth d, and expression -> z / library size.

`SpatialEncoder` is the isodepth encoder (coords -> one d axis).
`BatchedSpatialEncoder` gives each batch its own independent SpatialEncoder
emitting into a SHARED d axis -- that is what makes cross-batch isodepth
comparable, and why the cross-batch MMD alignment term exists.

The bio/library encoders wrap scvi.nn.Encoder, so they carry scVI's own
initialisation. Nothing here assumes counts (both take log1p expression), so
spatial proteomics needs no change in this file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.nn import Encoder as scviEncoder


class SpatialEncoder(nn.Module):
    def __init__(self, s_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu        = nn.Linear(hidden_dim, 1)
        self.log_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, s, batch=None):
        if torch.isnan(s).any():
            print("NaN in input s", flush=True)
        if torch.isinf(s).any():
            print("Inf in input s", flush=True)

        h = self.net(s)

        for name, p in self.named_parameters():
            if torch.isnan(p).any():
                print("NaN parameter:", name, flush=True)
            if torch.isinf(p).any():
                print("Inf parameter:", name, flush=True)

        if torch.isnan(h).any():
            print("NaN after hidden layer", flush=True)

        mu = self.mu(h)
        if torch.isnan(mu).any():
            print("NaN after mu layer", flush=True)

        log_sigma = self.log_sigma(h)
        if torch.isnan(log_sigma).any():
            print("NaN after log_sigma layer", flush=True)

        # return mu, torch.exp(log_sigma)
        return mu, F.softplus(log_sigma) + 1e-4


class BatchedSpatialEncoder(nn.Module):
    """One independent SpatialEncoder per batch, shared output space.

    Each spot is routed through the encoder for its batch label. All
    encoders emit into a single isodepth axis, so the downstream decoder
    sees one continuous d regardless of which batch produced it.
    """
    def __init__(self, s_dim, hidden_dim, n_batches):
        super().__init__()
        self.n_batches = n_batches
        self.encoders = nn.ModuleList([
            SpatialEncoder(s_dim, hidden_dim) for _ in range(n_batches)
        ])

    def forward(self, s, batch=None):
        if batch is None:
            return self.encoders[0](s)
        b_flat = batch.view(-1)
        mu    = s.new_empty(s.shape[0], 1)
        sigma = s.new_empty(s.shape[0], 1)
        for b in range(self.n_batches):
            mask = (b_flat == b)
            if mask.any():
                m_b, s_b = self.encoders[b](s[mask])
                mu[mask]    = m_b
                sigma[mask] = s_b
        return mu, sigma


class BiologicalEncoder(nn.Module):
    """scVI encoder: gene expression (log1p) → biological latent z_i.

    When n_batches>0 the encoder is built with n_cat_list=[n_batches] so it
    takes a batch index at forward time.  When n_batches=0 the encoder is
    unconditional and batch indices passed to forward() are ignored.
    """
    def __init__(self, g_dim, z_dim, n_layers=2, n_hidden=128,
                 dropout_rate=0.1, n_batches=0):
        super().__init__()
        self.n_batches = n_batches
        n_cat_list = [n_batches] if n_batches > 0 else None
        self.encoder = scviEncoder(
            n_input=g_dim, n_output=z_dim,
            n_cat_list=n_cat_list,
            n_layers=n_layers, n_hidden=n_hidden,
            dropout_rate=dropout_rate, distribution="normal",
        )

    def forward(self, x, batch=None):
        if self.n_batches > 0 and batch is not None:
            q_m, q_v, _ = self.encoder(x, batch)
        else:
            q_m, q_v, _ = self.encoder(x)
        return q_m, torch.sqrt(q_v)


class LibrarySizeEncoder(nn.Module):
    """scVI encoder: gene expression (log1p) → log library size l_i.

    Optionally batch-conditioned when n_batches>0.
    """
    def __init__(self, g_dim, n_layers=2, n_hidden=128,
                 dropout_rate=0.1, n_batches=0):
        super().__init__()
        self.n_batches = n_batches
        n_cat_list = [n_batches] if n_batches > 0 else None
        self.encoder = scviEncoder(
            n_input=g_dim, n_output=1,
            n_cat_list=n_cat_list,
            n_layers=n_layers, n_hidden=n_hidden,
            dropout_rate=dropout_rate, distribution="normal",
        )

    def forward(self, x, batch=None):
        if self.n_batches > 0 and batch is not None:
            q_m, q_v, _ = self.encoder(x, batch)
        else:
            q_m, q_v, _ = self.encoder(x)
        return q_m, torch.sqrt(q_v)
