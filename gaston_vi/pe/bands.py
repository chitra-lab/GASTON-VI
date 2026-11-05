"""Positional encoding: coordinates -> sin/cos feature bands.

Pure torch, with no dependency on the band heuristic: runs that pass --sigma/
--num_freqs/--extra_wavelengths explicitly never touch pe.auto.

Band order matters. Bands are kept in descending frequency, and reordering them
changes the feature columns, the seeded initialisation and so the whole run.
"""
import math

import torch
import torch.nn as nn


class LogFrequencyPositionalEncoding(nn.Module):
    def __init__(self, in_dim, num_frequencies, sigma, extra_wavelengths=None):
        super().__init__()
        self.in_dim          = in_dim
        self.num_frequencies = num_frequencies
        self.sigma           = sigma

        # Log-frequency ladder: omega_j = 2*pi*sigma**(j/N), equivalently
        # wavelength lambda_j = sigma**(-j/N) for j = 1..N, so
        # lambda_max = 1/sigma and lambda_min = sigma**(-1/N).
        j = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        bands = 2 * math.pi * sigma ** (j / num_frequencies)

        # Extra bands are supplied directly as WAVELENGTHS, in the same units
        # as the normalised coordinates passed to forward(). omega = 2*pi/lambda.
        self.extra_wavelengths = [float(l) for l in (extra_wavelengths or [])]
        if any(l <= 0 for l in self.extra_wavelengths):
            raise ValueError("extra_wavelengths must all be > 0")
        if self.extra_wavelengths:
            extra = torch.tensor([2 * math.pi / l for l in self.extra_wavelengths],
                                 dtype=torch.float32)
            bands = torch.cat([bands, extra])

        # One list, ladder + extras, sorted by descending frequency (ascending
        # wavelength), then near-duplicates dropped. With no extras this is the
        # ladder's native order sigma**(j/N), j=1..N. The 1e-4 relative de-dup
        # tolerance merges a rounded endpoint such as 1.1131 with the ladder's
        # finest band 0.18**(-1/16) = 1.1131289.
        bands, _ = torch.sort(bands, descending=True)
        keep = torch.ones_like(bands, dtype=torch.bool)
        if bands.numel() > 1:
            keep[1:] = (bands[:-1] - bands[1:]).abs() > 1e-4 * bands[1:].abs()
        bands = bands[keep]
        self.pe_version = 2
        self.pe_band_order = "desc_freq"

        self.register_buffer("freq_bands", bands)
        self.n_bands = int(bands.numel())
        self.out_dim = in_dim * (1 + 2 * self.n_bands)
        lam = 2 * math.pi / bands
        self.lambda_min = float(lam.min())
        self.lambda_max = float(lam.max())
        print(f"PE: sigma={sigma}, num_freqs={num_frequencies}, "
              f"n_extra={len(self.extra_wavelengths)}, n_bands={self.n_bands}, "
              f"lambda=[{self.lambda_min:.4f}, {self.lambda_max:.4f}], "
              f"out_dim={self.out_dim}", flush=True)

    def forward(self, s):
        fb    = self.freq_bands.to(s.device)
        s_exp = s.unsqueeze(-2) * fb.view(1, -1, 1)
        sin   = torch.sin(s_exp)
        cos   = torch.cos(s_exp)
        return torch.cat([s, torch.cat([sin, cos], dim=-1).flatten(-2)], dim=-1)


class BatchedPositionalEncoding(nn.Module):
    """One LogFrequencyPositionalEncoding per batch, outputs zero-padded to the
    widest one. forward(s, batch) routes each cell through its batch's PE; the
    unused trailing columns of a narrower PE are zero (a constant input, which
    that batch's own SpatialEncoder simply ignores)."""
    def __init__(self, in_dim, specs):
        super().__init__()
        self.n_batches = max(specs) + 1
        self.pes = nn.ModuleList()
        for b in range(self.n_batches):
            if b not in specs:
                raise ValueError(f"--pe_per_batch_json has no entry for batch {b}")
            v = specs[b]
            print(f"[batch {b}] ", end="", flush=True)
            self.pes.append(LogFrequencyPositionalEncoding(
                in_dim, v["num_freqs"], v["sigma"], v["extra_wavelengths"]))
        self.out_dim = max(pe.out_dim for pe in self.pes)
        self.n_bands = max(pe.n_bands for pe in self.pes)
        self.lambda_min = min(pe.lambda_min for pe in self.pes)
        self.lambda_max = max(pe.lambda_max for pe in self.pes)
        print(f"PE (per-batch): out_dim={self.out_dim} (padded), bands per batch="
              f"{[pe.n_bands for pe in self.pes]}, lambda over batches="
              f"[{self.lambda_min:.4f}, {self.lambda_max:.4f}]", flush=True)

    def forward(self, s, batch):
        b_flat = batch.view(-1)
        out = s.new_zeros(s.shape[0], self.out_dim)
        for b, pe in enumerate(self.pes):
            mask = (b_flat == b)
            if mask.any():
                enc = pe(s[mask])
                out[mask, :enc.shape[1]] = enc
        return out
