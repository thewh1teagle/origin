import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from . import constants as C


def disc_step(y: torch.Tensor, y_hat: torch.Tensor, mpd: nn.Module, mrd: nn.Module, loss_disc: nn.Module, opt_d: Optimizer) -> torch.Tensor:
    opt_d.zero_grad()
    y_d_r_mpd, y_d_g_mpd, _, _ = mpd(y, y_hat.detach())
    y_d_r_mrd, y_d_g_mrd, _, _ = mrd(y, y_hat.detach())
    d_loss, _, _ = loss_disc(y_d_r_mpd + y_d_r_mrd, y_d_g_mpd + y_d_g_mrd)
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(mrd.parameters()), max_norm=1.0)
    opt_d.step()
    return d_loss


def gen_step(y: torch.Tensor, y_hat: torch.Tensor, mpd: nn.Module, mrd: nn.Module, loss_gen: nn.Module, loss_fm: nn.Module, loss_recon: nn.Module, opt_g: Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
    opt_g.zero_grad()
    _, y_d_g_mpd, fmap_r_mpd, fmap_g_mpd = mpd(y, y_hat)
    _, y_d_g_mrd, fmap_r_mrd, fmap_g_mrd = mrd(y, y_hat)
    adv   = loss_gen(y_d_g_mpd + y_d_g_mrd)[0]
    fm    = loss_fm(fmap_r_mpd + fmap_r_mrd, fmap_g_mpd + fmap_g_mrd)
    recon = loss_recon(y_hat, y)
    g_loss = C.LAMBDA_RECON * recon + C.LAMBDA_ADV * adv + C.LAMBDA_FM * fm
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
    opt_g.step()
    return g_loss, recon


def save_checkpoint(args: argparse.Namespace, step: int, ae: nn.Module, mpd: nn.Module, mrd: nn.Module, opt_g: Optimizer, opt_d: Optimizer) -> None:
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
    for p in sorted(Path(args.checkpoint_dir).glob("ae_*.pt"))[:-args.keep_ckpts]:
        p.unlink()
