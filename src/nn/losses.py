from typing import List, Tuple

import torch
import torchaudio
from torch import nn

from .layers import safe_log


class MultiResolutionMelLoss(nn.Module):
    """Multi-resolution mel spectrogram L1 reconstruction loss.

    FFT sizes [1024, 2048, 4096], mel bands [64, 128, 128], hop = FFT/4.
    Matches SupertonicTTS autoencoder training setup.
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        configs = [
            (1024,  64),
            (2048, 128),
            (4096, 128),
        ]
        self.mel_specs = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=n_fft // 4,
                n_mels=n_mels,
                f_min=0.0,
                f_max=sample_rate / 2,
                center=True,
                power=1,
            )
            for n_fft, n_mels in configs
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=y.device, dtype=y.dtype)
        for mel_spec in self.mel_specs:
            mel_spec = mel_spec.to(y.device)
            loss += torch.nn.functional.l1_loss(
                safe_log(mel_spec(y_hat)),
                safe_log(mel_spec(y)),
            )
        return loss


class GeneratorLoss(nn.Module):
    """Least-squares GAN generator loss."""

    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l
        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """Least-squares GAN discriminator loss."""

    def forward(
        self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses, g_losses = [], []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean((0 - dg) ** 2)
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)
        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """L1 feature matching loss between discriminator intermediate layers."""

    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss
