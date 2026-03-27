from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import Spectrogram


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator — lightweight version per SupertonicTTS paper."""

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(period=p) for p in periods])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    """Single-period sub-discriminator. Lightweight channels: [16, 64, 256, 512, 512, 1]."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, lrelu_slope: float = 0.1):
        super().__init__()
        self.period = period
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(1,   16,  (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(Conv2d(16,  64,  (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(Conv2d(64,  256, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(Conv2d(512, 512, (kernel_size, 1), (1,      1), padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = weight_norm(Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x.unsqueeze(1)
        b, c, t = x.shape
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - (t % self.period)), "reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)

        fmap = []
        for l in self.convs:
            x = F.leaky_relu(l(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiResolutionDiscriminator(nn.Module):
    """Multi-Resolution Discriminator per SupertonicTTS Table 7."""

    def __init__(self, fft_sizes: Tuple[int, ...] = (512, 1024, 2048)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorR(fft_size=f) for f in fft_sizes])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    """Single-resolution sub-discriminator. Architecture from SupertonicTTS Table 7."""

    def __init__(self, fft_size: int, lrelu_slope: float = 0.1):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.spec_fn = Spectrogram(
            n_fft=fft_size, win_length=fft_size, hop_length=fft_size // 4, power=None
        )
        # 6 Conv2D layers per Table 7: channels 1→16→16→16→16→16→1
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(1,  16, (5, 5), (1, 1), padding=(2, 2))),
            weight_norm(Conv2d(16, 16, (5, 5), (2, 1), padding=(2, 2))),
            weight_norm(Conv2d(16, 16, (5, 5), (2, 1), padding=(2, 2))),
            weight_norm(Conv2d(16, 16, (5, 5), (2, 1), padding=(2, 2))),
            weight_norm(Conv2d(16, 16, (5, 5), (1, 1), padding=(2, 2))),
        ])
        self.conv_post = weight_norm(Conv2d(16, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        # Log-scaled linear spectrogram
        spec = self.spec_fn(x)
        spec = torch.view_as_real(spec)                    # (B, F, T, 2)
        mag = spec.pow(2).sum(-1).sqrt()                   # (B, F, T)
        return safe_log_spec(mag).unsqueeze(1)             # (B, 1, F, T)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.spectrogram(x)
        fmap = []
        for l in self.convs:
            x = F.leaky_relu(l(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


def safe_log_spec(x: torch.Tensor, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val))
