"""
Decode latents (.pt tensor) back to a wav file.

Usage:
    uv run python -m src.decode <checkpoint> <input.pt> <output.wav>
"""

import sys
import torch
import torchaudio
from .nn.autoencoder import SpeechAutoencoder
from . import constants as C


def main():
    if len(sys.argv) != 4:
        print("Usage: python -m src.decode <checkpoint> <input.pt> <output.wav>")
        sys.exit(1)

    ckpt_path, in_pt, out_wav = sys.argv[1], sys.argv[2], sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = SpeechAutoencoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ae.load_state_dict(ckpt["ae"])
    ae.eval()

    z = torch.load(in_pt, map_location=device)  # (1, 24, T_latent)

    with torch.no_grad():
        wav = ae.decode(z)  # (1, 1, T_audio)

    wav = wav.squeeze(0)  # (1, T_audio)
    torchaudio.save(out_wav, wav.cpu(), C.SAMPLE_RATE)
    print(f"Saved audio {tuple(wav.shape)} → {out_wav}")


if __name__ == "__main__":
    main()
