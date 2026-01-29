import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

# ------------------- Encoders -------------------
class Encoder_XS(nn.Module):
    """q(z | x, s)"""
    def __init__(self, g_dim, s_dim, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim + s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.log_sigma = nn.Linear(hidden, z_dim)

    def forward(self, x, s):
        h = self.net(torch.cat([x, s], dim=1))
        mu = self.mu(h)
        # stable, strictly positive, avoids 0
        sigma = F.softplus(self.log_sigma(h)) + 1e-4
        return mu, sigma


class Encoder_X(nn.Module):
    """q(z | x)"""
    def __init__(self, g_dim, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.log_sigma = nn.Linear(hidden, z_dim)

    def forward(self, x, s=None):
        h = self.net(x)
        mu = self.mu(h)
        sigma = F.softplus(self.log_sigma(h)) + 1e-4
        return mu, sigma


class Encoder_S(nn.Module):
    """q(z | s)"""
    def __init__(self, s_dim, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.log_sigma = nn.Linear(hidden, z_dim)

    def forward(self, x, s):
        h = self.net(s)
        mu = self.mu(h)
        sigma = F.softplus(self.log_sigma(h)) + 1e-4
        return mu, sigma


# ------------------- Decoder -------------------
class Decoder(nn.Module):
    def __init__(self, z_dim, g_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, g_dim)
        self.log_sigma = nn.Linear(hidden, 1)

    def forward(self, z):
        h = self.net(z)
        loc = self.mu(h)
        # exp is fine here (never exactly 0), but we can use softplus for symmetry
        scale = F.softplus(self.log_sigma(h)) + 1e-4
        return loc, scale


# ------------------- Spatial prior -------------------
class SpatialPrior(nn.Module):
    """p(z | s)"""
    def __init__(self, s_dim, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.log_sigma = nn.Linear(hidden, z_dim)

    def forward(self, s):
        h = self.net(s)
        mu = self.mu(h)
        sigma = F.softplus(self.log_sigma(h)) + 1e-4
        return mu, sigma


# ------------------- Full VAE -------------------
class VAE(nn.Module):
    """
    Variants:
      A: q(z|x,s) fixed prior
      B: q(z|x)   fixed prior
      C: q(z|s)   fixed prior
      D: q(z|x,s) learned prior
      E: q(z|x)   learned prior
      F: q(z|s)   learned prior
    """

    def __init__(self, variant, g_dim, s_dim, z_dim=1, hidden=64):
        super().__init__()

        if variant in ["A", "D"]:
            self.encoder = Encoder_XS(g_dim, s_dim, z_dim, hidden)
        elif variant in ["B", "E"]:
            self.encoder = Encoder_X(g_dim, z_dim, hidden)
        else:  # C, F
            self.encoder = Encoder_S(s_dim, z_dim, hidden)

        self.decoder = Decoder(z_dim, g_dim, hidden)

        self.learned_prior = variant in ["D", "E", "F"]
        if self.learned_prior:
            self.prior_net = SpatialPrior(s_dim, z_dim, hidden)

        self.z_dim = z_dim

    # ------------------- MODEL -------------------
    def model(self, x, s):
        pyro.module("decoder", self.decoder)
        if self.learned_prior:
            pyro.module("prior_net", self.prior_net)

        with pyro.plate("plate", x.size(0)):
            if self.learned_prior:
                mu_p, sigma_p = self.prior_net(s)
            else:
                mu_p = x.new_zeros((x.size(0), self.z_dim))
                sigma_p = x.new_ones((x.size(0), self.z_dim))

            z = pyro.sample("z", dist.Normal(mu_p, sigma_p).to_event(1))
            loc, scale = self.decoder(z)
            pyro.sample("obs", dist.Normal(loc, scale).to_event(1), obs=x)

    # ------------------- GUIDE -------------------
    def guide(self, x, s):
        pyro.module("encoder", self.encoder)
        with pyro.plate("plate", x.size(0)):
            mu_q, sigma_q = self.encoder(x, s)
            pyro.sample("z", dist.Normal(mu_q, sigma_q).to_event(1))

    @torch.no_grad()
    def encode_mu(self, x, s):
        mu, _ = self.encoder(x, s)
        return mu
