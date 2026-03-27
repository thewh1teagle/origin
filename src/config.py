from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AutoencoderConfig:
    # Audio
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 228

    # Encoder
    enc_idim: int = 1253          # linear(1025) + mel(228)
    enc_hdim: int = 512
    enc_odim: int = 24
    enc_num_layers: int = 10
    enc_intermediate_dim: int = 2048
    enc_kernel_size: int = 7

    # Decoder
    dec_idim: int = 24
    dec_hdim: int = 512
    dec_num_layers: int = 10
    dec_intermediate_dim: int = 2048
    dec_kernel_size: int = 7
    dec_head_hdim: int = 2048
    dec_head_odim: int = 512      # × hop_length = waveform samples per frame

    # Discriminators
    mpd_periods: Tuple[int, ...] = (2, 3, 5, 7, 11)
    mrd_fft_sizes: Tuple[int, ...] = (512, 1024, 2048)

    # Losses
    lambda_recon: float = 45.0
    lambda_adv: float = 1.0
    lambda_fm: float = 0.1

    # Training
    learning_rate: float = 2e-4
    adam_betas: Tuple[float, float] = (0.8, 0.99)
    adam_eps: float = 1e-9
    lr_decay: float = 0.999
    batch_size: int = 28
    max_steps: int = 1_500_000
    segment_length: int = 44100   # 1 second of audio per sample

    # Logging / checkpointing
    log_every: int = 100
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
