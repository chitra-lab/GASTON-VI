import torch
import torch.nn as nn
import math

class LogFrequencyPositionalEncoding(nn.Module):
    """
    NeRF-style log frequency encoding:
      Input:  coords (N, 2)
      Output:
        if num_frequencies == 0:
            (N, 2)   # just [x, y]
        else:
            (N, 2 + 4 * num_frequencies)
            [x, y,
             sin(f_j x), cos(f_j x),
             sin(f_j y), cos(f_j y)]
    """

    def __init__(self, num_frequencies=16, sigma=10.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.sigma = sigma

        if num_frequencies > 0:
            j = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
            freq = sigma ** (j / num_frequencies)
            self.register_buffer("freq_bands", 2 * math.pi * freq)
        else:
            # dummy buffer so .to(device) etc. don't complain; won't be used
            self.register_buffer("freq_bands", torch.zeros(1))

    def forward(self, coords):
        """
        coords: (N, 2)
        """
        # No positional encoding: just return normalized coords
        if self.num_frequencies == 0:
            return coords

        x = coords[:, 0:1]  # (N,1)
        y = coords[:, 1:2]  # (N,1)

        fb = self.freq_bands[None, :, None]  # (1, M, 1)

        x_exp = x[:, None, :] * fb  # (N, M, 1)
        y_exp = y[:, None, :] * fb

        x_sin = torch.sin(x_exp)
        x_cos = torch.cos(x_exp)
        y_sin = torch.sin(y_exp)
        y_cos = torch.cos(y_exp)

        pe = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)  # (N, M, 4)
        pe = pe.reshape(coords.size(0), -1)                   # (N, 4M)

        return torch.cat([coords, pe], dim=1)
