# origin

Speech autoencoder that encodes audio into a compact latent representation and reconstructs high-quality waveform. Based on the SupertonicTTS speech autoencoder (arXiv:2503.23108).

## Training

**1. Download and prepare the dataset** (LJSpeech-enhanced, ~14 GB):

```console
./scripts/train_prepare.sh
```

**2. Train from scratch:**

```console
./scripts/train_scratch.sh
```

Checkpoints are saved to `checkpoints/` every 5000 steps (keeps last 3).

**3. Test reconstruction** after training:

```console
./scripts/reconstruct.sh checkpoints/ae_0005000.pt input.wav output.wav
```

## Requirements

Python 3.12+, PyTorch, torchaudio. Install deps: `uv sync`.
