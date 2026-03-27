from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class ConvNeXtBlock(nn.Module):
    """1D ConvNeXt block adapted from https://github.com/facebookresearch/ConvNeXt."""

    def __init__(self, dim: int, intermediate_dim: int, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        return residual + x


class CausalConv1d(nn.Module):
    """Causal Conv1d — no future context, left-padding only."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalConvNeXtBlock(nn.Module):
    """ConvNeXt block with causal depthwise conv for the decoder."""

    def __init__(self, dim: int, intermediate_dim: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        return residual + x


def safe_log(x: torch.Tensor, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val))
