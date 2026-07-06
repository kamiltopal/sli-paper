"""Forecaster architectures (protocol §3.2, frozen v1 subset).

Channel convention: input (B, L, C), C=3, target is channel 0.
Output: (B, H) forecast of the target channel.

Operationalization of the H3 axis:
  - channel-independent models (DLinear, PatchTST) receive ONLY the
    target channel — they cannot mix cross-channel information.
  - channel-mixing models (iTransformer) receive all channels and
    attend across them.

TimesNet is a planned extension (protocol §3.6 stage 4).
"""
from __future__ import annotations

import torch
import torch.nn as nn

CHANNEL_INDEPENDENT = {"dlinear", "patchtst"}
CHANNEL_MIXING = {"itransformer"}


# --------------------------------------------------------------------------
class MovingAvg(nn.Module):
    def __init__(self, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x):  # (B, L)
        front = x[:, :1].repeat(1, (self.kernel - 1) // 2)
        back = x[:, -1:].repeat(1, self.kernel // 2)
        return self.avg(torch.cat([front, x, back], dim=1).unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    """Zeng et al. 2023, univariate form: trend/seasonal decomposition +
    one linear map each. Channel-independent: uses target channel only."""

    def __init__(self, lookback: int, horizon: int, **_):
        super().__init__()
        self.decomp = MovingAvg(25)
        self.lin_trend = nn.Linear(lookback, horizon)
        self.lin_season = nn.Linear(lookback, horizon)

    def forward(self, x):            # (B, L, C)
        x = x[:, :, 0]               # target channel only
        trend = self.decomp(x)
        season = x - trend
        return self.lin_trend(trend) + self.lin_season(season)


# --------------------------------------------------------------------------
class PatchTST(nn.Module):
    """Nie et al. 2023, lite: patch the target channel, transformer
    encoder over patch tokens, flatten -> linear head."""

    def __init__(self, lookback: int, horizon: int, patch_len: int = 16,
                 stride: int = 8, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, **_):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n_patches = (lookback - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, x):            # (B, L, C)
        x = x[:, :, 0]               # target channel only
        patches = x.unfold(1, self.patch_len, self.stride)   # (B, N, P)
        z = self.embed(patches) + self.pos
        z = self.encoder(z)
        return self.head(z.flatten(1))


# --------------------------------------------------------------------------
class ITransformer(nn.Module):
    """Liu et al. 2024, lite: each channel's whole lookback is one token;
    attention runs ACROSS channels (the mixing mechanism)."""

    def __init__(self, lookback: int, horizon: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, **_):
        super().__init__()
        self.embed = nn.Linear(lookback, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):            # (B, L, C)
        z = self.embed(x.permute(0, 2, 1))   # (B, C, d)  channel tokens
        z = self.encoder(z)                  # cross-channel attention
        return self.head(z[:, 0])            # target channel token


# --------------------------------------------------------------------------
MODELS = {"dlinear": DLinear, "patchtst": PatchTST, "itransformer": ITransformer}


def build(name: str, lookback: int, horizon: int) -> nn.Module:
    return MODELS[name.lower()](lookback=lookback, horizon=horizon)
