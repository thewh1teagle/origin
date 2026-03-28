"""
SupertonicTTS Speech Autoencoder

LatentEncoder:  mel+linear spectrogram (1253-dim) → 24-dim latents @ 86 Hz
LatentDecoder:  24-dim latents → waveform @ 44.1 kHz  (used at TTS inference)

Architecture from arXiv:2503.23108 and confirmed via onnx_re/README.md.
"""

from typing import List

import torch
import torchaudio
from torch import nn

from .layers import CausalConv1d, CausalConvNeXtBlock, ConvNeXtBlock, safe_log


class SpecProcessor(nn.Module):
    """Extracts concatenated [linear_stft, mel] features → (B, 1253, T).

    idim = n_fft//2+1 + n_mels = 1025 + 228 = 1253.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
    ):
        super().__init__()
        self.linear_spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, win_length=n_fft, hop_length=hop_length, power=1, center=True
        )
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2,
            center=True,
            power=1,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T_audio)
        linear = safe_log(self.linear_spec(wav))   # (B, 1025, T)
        mel = safe_log(self.mel_spec(wav))          # (B, 228,  T)
        return torch.cat([linear, mel], dim=1)      # (B, 1253, T)


class LatentEncoder(nn.Module):
    """Encodes spectrogram features to 24-dim latents.

    Only used during training (autoencoder phase + TTL latent extraction).
    NOT used at TTS inference time.
    """

    def __init__(
        self,
        idim: int = 1253,
        hdim: int = 512,
        odim: int = 24,
        num_layers: int = 10,
        intermediate_dim: int = 2048,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(idim, hdim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm_in = nn.BatchNorm1d(hdim)
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(hdim, intermediate_dim, kernel_size=kernel_size)
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(hdim, odim)
        self.norm_out = nn.LayerNorm(odim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1253, T)
        x = self.norm_in(self.conv_in(x))           # (B, 512, T)
        x = self.blocks(x)                           # (B, 512, T)
        x = self.proj_out(x.transpose(1, 2))         # (B, T, 24)
        x = self.norm_out(x)
        return x.transpose(1, 2)                     # (B, 24, T)


class LatentDecoder(nn.Module):
    """Decodes 24-dim latents to waveform. Used at TTS inference.

    All convolutions are causal (no future context) to support streaming.
    Dilation schedule: [1, 2, 4, 1, 2, 4, 1, 1, 1, 1].
    """

    DILATIONS = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]

    def __init__(
        self,
        idim: int = 24,
        hdim: int = 512,
        num_layers: int = 10,
        intermediate_dim: int = 2048,
        kernel_size: int = 7,
        head_hdim: int = 2048,
        head_odim: int = 512,
        head_ksz: int = 3,
    ):
        super().__init__()
        assert num_layers == len(self.DILATIONS)

        self.conv_in = CausalConv1d(idim, hdim, kernel_size=kernel_size)
        self.norm_in = nn.BatchNorm1d(hdim)
        self.blocks = nn.Sequential(*[
            CausalConvNeXtBlock(hdim, intermediate_dim, kernel_size=kernel_size, dilation=d)
            for d in self.DILATIONS
        ])
        self.norm_mid = nn.BatchNorm1d(hdim)

        # Head: causal conv → linear → PReLU → linear → flatten to waveform
        self.head_conv = CausalConv1d(hdim, head_hdim, kernel_size=head_ksz)
        self.head_linear = nn.Linear(head_hdim, head_odim)
        self.head_act = nn.PReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, 24, T_latent)
        x = self.norm_in(self.conv_in(z))            # (B, 512, T)
        x = self.blocks(x)                           # (B, 512, T)
        x = self.norm_mid(x)
        x = self.head_conv(x)                        # (B, 2048, T)
        x = self.head_act(self.head_linear(x.transpose(1, 2))).transpose(1, 2)  # (B, 512, T)
        # Flatten: (B, 512, T) → (B, T*512) → (B, 1, T*512)
        B, C, T = x.shape
        return x.permute(0, 2, 1).reshape(B, 1, T * C)  # (B, 1, T_audio)


class SpeechAutoencoder(nn.Module):
    """Full autoencoder: wav → latents → wav. Used for training only."""

    def __init__(self):
        super().__init__()
        self.spec = SpecProcessor()
        self.encoder = LatentEncoder()
        self.decoder = LatentDecoder()

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.spec(wav))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(wav))
