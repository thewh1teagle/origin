"""
Encode a wav file to latents (.pt tensor).

Usage:
    uv run python -m src.encode <checkpoint> <input.wav> <output.pt>
"""

import sys
import torch
import torchaudio
from .nn.autoencoder import SpeechAutoencoder
from . import constants as C


def main():
    if len(sys.argv) != 4:
        print("Usage: python -m src.encode <checkpoint> <input.wav> <output.pt>")
        sys.exit(1)

    ckpt_path, in_wav, out_pt = sys.argv[1], sys.argv[2], sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = SpeechAutoencoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ae.load_state_dict(ckpt["ae"])
    ae.eval()

    wav, sr = torchaudio.load(in_wav)
    if sr != C.SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, C.SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0).unsqueeze(0).to(device)  # (1, T)

    with torch.no_grad():
        z = ae.encode(wav)  # (1, 24, T_latent)

    torch.save(z.cpu(), out_pt)
    print(f"Saved latents {tuple(z.shape)} → {out_pt}")


if __name__ == "__main__":
    main()
