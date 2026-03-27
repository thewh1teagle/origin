"""
Autoencoder GAN training loop.

Loss weighting (SupertonicTTS §3.1):
    L_total = 45 × L_recon  +  1 × L_adv  +  0.1 × L_fm

Generator and discriminators are updated in alternating steps:
  1. Update discriminators with real + generated audio.
  2. Update generator (encoder+decoder) with adv + fm + recon losses.

Usage:
    uv run python -m src.train --data_dir /path/to/wav_files
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .autoencoder import SpeechAutoencoder
from .config import AutoencoderConfig
from .discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from .losses import (
    DiscriminatorLoss,
    FeatureMatchingLoss,
    GeneratorLoss,
    MultiResolutionMelLoss,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """Loads mono 44.1 kHz wav files, returns fixed-length segments."""

    def __init__(self, data_dir: str, segment_length: int, sample_rate: int = 44100):
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.files = sorted(Path(data_dir).rglob("*.wav"))
        if not self.files:
            raise ValueError(f"No .wav files found under {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav, sr = torchaudio.load(self.files[idx])
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        # Mix to mono
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)  # (T,)

        # Random crop or pad to segment_length
        T = wav.shape[0]
        if T >= self.segment_length:
            start = torch.randint(0, T - self.segment_length + 1, (1,)).item()
            wav = wav[start : start + self.segment_length]
        else:
            wav = F.pad(wav, (0, self.segment_length - T))
        return wav  # (segment_length,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: AutoencoderConfig, data_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Models
    ae = SpeechAutoencoder().to(device)
    mpd = MultiPeriodDiscriminator(periods=cfg.mpd_periods).to(device)
    mrd = MultiResolutionDiscriminator(fft_sizes=cfg.mrd_fft_sizes).to(device)

    # Losses
    loss_recon = MultiResolutionMelLoss(sample_rate=cfg.sample_rate).to(device)
    loss_gen = GeneratorLoss()
    loss_disc = DiscriminatorLoss()
    loss_fm = FeatureMatchingLoss()

    # Optimizers — generator and discriminators updated separately
    opt_g = torch.optim.AdamW(
        ae.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
    )
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
    )
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=cfg.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=cfg.lr_decay)

    # Data
    dataset = AudioDataset(data_dir, cfg.segment_length, cfg.sample_rate)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    step = 0

    while step < cfg.max_steps:
        for batch in loader:
            if step >= cfg.max_steps:
                break

            y = batch.to(device)          # (B, T)
            y_hat = ae(y).squeeze(1)      # (B, T') — may be slightly longer due to frame rounding
            # Trim to input length (T' >= T always due to ceil framing in encoder)
            y_hat = y_hat[..., : y.shape[-1]]

            # ----------------------------------------------------------------
            # Step 1: Discriminator update
            # ----------------------------------------------------------------
            opt_d.zero_grad()

            _, _, fmap_mpd_r, fmap_mpd_g = mpd(y, y_hat.detach())
            _, _, fmap_mrd_r, fmap_mrd_g = mrd(y, y_hat.detach())

            y_d_r_mpd, y_d_g_mpd, _, _ = mpd(y, y_hat.detach())
            y_d_r_mrd, y_d_g_mrd, _, _ = mrd(y, y_hat.detach())

            d_loss_mpd, _, _ = loss_disc(y_d_r_mpd, y_d_g_mpd)
            d_loss_mrd, _, _ = loss_disc(y_d_r_mrd, y_d_g_mrd)
            d_loss = d_loss_mpd + d_loss_mrd

            d_loss.backward()
            opt_d.step()

            # ----------------------------------------------------------------
            # Step 2: Generator update
            # ----------------------------------------------------------------
            opt_g.zero_grad()

            y_d_r_mpd, y_d_g_mpd, fmap_mpd_r, fmap_mpd_g = mpd(y, y_hat)
            y_d_r_mrd, y_d_g_mrd, fmap_mrd_r, fmap_mrd_g = mrd(y, y_hat)

            adv_mpd, _ = loss_gen(y_d_g_mpd)
            adv_mrd, _ = loss_gen(y_d_g_mrd)
            fm_mpd = loss_fm(fmap_mpd_r, fmap_mpd_g)
            fm_mrd = loss_fm(fmap_mrd_r, fmap_mrd_g)
            recon = loss_recon(y_hat, y)

            g_loss = (
                cfg.lambda_recon * recon
                + cfg.lambda_adv * (adv_mpd + adv_mrd)
                + cfg.lambda_fm * (fm_mpd + fm_mrd)
            )
            g_loss.backward()
            opt_g.step()

            # ----------------------------------------------------------------
            # Logging / checkpointing
            # ----------------------------------------------------------------
            step += 1

            if step % cfg.log_every == 0:
                print(
                    f"step={step:>7d}  "
                    f"g={g_loss.item():.4f}  "
                    f"d={d_loss.item():.4f}  "
                    f"recon={recon.item():.4f}  "
                    f"adv={adv_mpd.item() + adv_mrd.item():.4f}  "
                    f"fm={fm_mpd.item() + fm_mrd.item():.4f}"
                )

            if step % cfg.save_every == 0:
                ckpt = {
                    "step": step,
                    "ae": ae.state_dict(),
                    "mpd": mpd.state_dict(),
                    "mrd": mrd.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                }
                path = os.path.join(cfg.checkpoint_dir, f"ae_{step:07d}.pt")
                torch.save(ckpt, path)
                print(f"Saved checkpoint: {path}")

            if step % 1000 == 0:
                sched_g.step()
                sched_d.step()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the SupertonicTTS speech autoencoder")
    parser.add_argument("--data_dir", required=True, help="Directory of .wav files")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    args = parser.parse_args()

    cfg = AutoencoderConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir

    train(cfg, args.data_dir)


if __name__ == "__main__":
    main()
