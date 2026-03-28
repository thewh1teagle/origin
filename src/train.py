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
from tqdm import tqdm

from . import constants as C
from .batch import disc_step, gen_step, save_checkpoint
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

    ae  = SpeechAutoencoder().to(device)
    mpd = MultiPeriodDiscriminator(periods=C.MPD_PERIODS).to(device)
    mrd = MultiResolutionDiscriminator(fft_sizes=C.MRD_FFT_SIZES).to(device)

    loss_recon = MultiResolutionMelLoss(sample_rate=C.SAMPLE_RATE).to(device)
    loss_gen   = GeneratorLoss()
    loss_disc  = DiscriminatorLoss()
    loss_fm    = FeatureMatchingLoss()

    opt_g = torch.optim.AdamW(ae.parameters(), lr=C.LEARNING_RATE, betas=C.ADAM_BETAS, eps=C.ADAM_EPS)
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=C.LEARNING_RATE, betas=C.ADAM_BETAS, eps=C.ADAM_EPS,
    )
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=C.LR_DECAY)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=C.LR_DECAY)

    dataset = AudioDataset(args.data_dir, args.segment_length, C.SAMPLE_RATE)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, pin_memory=True, drop_last=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step, epoch = 0, 0
    pbar = tqdm(total=args.max_steps, desc="training", dynamic_ncols=True)

    while step < args.max_steps:
        for y in loader:
            if step >= args.max_steps:
                break

            y     = y.to(device)
            y_hat = ae(y).squeeze(1)[..., :y.shape[-1]]

            d_loss        = disc_step(y, y_hat, mpd, mrd, loss_disc, opt_d)
            g_loss, recon = gen_step(y, y_hat, ae, mpd, mrd, loss_gen, loss_fm, loss_recon, opt_g)

            step += 1
            pbar.update(1)
            pbar.set_postfix(
                epoch=epoch,
                g=f"{g_loss.item():.4f}",
                d=f"{d_loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                lr=f"{opt_g.param_groups[0]['lr']:.2e}",
            )

            if step % args.save_every == 0:
                save_checkpoint(args, step, ae, mpd, mrd, opt_g, opt_d)

            if step % 1000 == 0:
                sched_g.step()
                sched_d.step()

        epoch += 1

    pbar.close()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
