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

import torch
from torch.utils.data import DataLoader

from . import constants as C
from .config import get_args
from .nn.autoencoder import SpeechAutoencoder
from .nn.data import AudioDataset
from .nn.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from .nn.losses import (
    DiscriminatorLoss,
    FeatureMatchingLoss,
    GeneratorLoss,
    MultiResolutionMelLoss,
)


def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    ae = SpeechAutoencoder().to(device)
    mpd = MultiPeriodDiscriminator(periods=C.MPD_PERIODS).to(device)
    mrd = MultiResolutionDiscriminator(fft_sizes=C.MRD_FFT_SIZES).to(device)

    loss_recon = MultiResolutionMelLoss(sample_rate=C.SAMPLE_RATE).to(device)
    loss_gen = GeneratorLoss()
    loss_disc = DiscriminatorLoss()
    loss_fm = FeatureMatchingLoss()

    opt_g = torch.optim.AdamW(ae.parameters(), lr=C.LEARNING_RATE, betas=C.ADAM_BETAS, eps=C.ADAM_EPS)
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=C.LEARNING_RATE, betas=C.ADAM_BETAS, eps=C.ADAM_EPS,
    )
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=C.LR_DECAY)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=C.LR_DECAY)

    dataset = AudioDataset(args.data_dir, args.segment_length, C.SAMPLE_RATE)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step = 0

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            y = batch.to(device)
            y_hat = ae(y).squeeze(1)
            y_hat = y_hat[..., : y.shape[-1]]

            opt_d.zero_grad()
            y_d_r_mpd, y_d_g_mpd, _, _ = mpd(y, y_hat.detach())
            y_d_r_mrd, y_d_g_mrd, _, _ = mrd(y, y_hat.detach())
            d_loss_mpd, _, _ = loss_disc(y_d_r_mpd, y_d_g_mpd)
            d_loss_mrd, _, _ = loss_disc(y_d_r_mrd, y_d_g_mrd)
            d_loss = d_loss_mpd + d_loss_mrd
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            y_d_r_mpd, y_d_g_mpd, fmap_mpd_r, fmap_mpd_g = mpd(y, y_hat)
            y_d_r_mrd, y_d_g_mrd, fmap_mrd_r, fmap_mrd_g = mrd(y, y_hat)
            adv_mpd, _ = loss_gen(y_d_g_mpd)
            adv_mrd, _ = loss_gen(y_d_g_mrd)
            fm_mpd = loss_fm(fmap_mpd_r, fmap_mpd_g)
            fm_mrd = loss_fm(fmap_mrd_r, fmap_mrd_g)
            recon = loss_recon(y_hat, y)
            g_loss = (
                C.LAMBDA_RECON * recon
                + C.LAMBDA_ADV * (adv_mpd + adv_mrd)
                + C.LAMBDA_FM * (fm_mpd + fm_mrd)
            )
            g_loss.backward()
            opt_g.step()

            step += 1

            if step % args.log_every == 0:
                print(
                    f"step={step:>7d}  "
                    f"g={g_loss.item():.4f}  "
                    f"d={d_loss.item():.4f}  "
                    f"recon={recon.item():.4f}  "
                    f"adv={adv_mpd.item() + adv_mrd.item():.4f}  "
                    f"fm={fm_mpd.item() + fm_mrd.item():.4f}"
                )

            if step % args.save_every == 0:
                ckpt = {
                    "step": step,
                    "ae": ae.state_dict(),
                    "mpd": mpd.state_dict(),
                    "mrd": mrd.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                }
                path = os.path.join(args.checkpoint_dir, f"ae_{step:07d}.pt")
                torch.save(ckpt, path)
                print(f"Saved checkpoint: {path}")

            if step % 1000 == 0:
                sched_g.step()
                sched_d.step()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
