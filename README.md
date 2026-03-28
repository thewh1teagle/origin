# origin

Speech autoencoder that encodes audio into a compact latent representation and reconstructs high-quality waveform. Based on the SupertonicTTS speech autoencoder (arXiv:2503.23108).

## Training

```bash
uv run python -m src.train --data_dir /path/to/wav_files
```

Input: mono `.wav` files resampled to 44.1 kHz.

## Requirements

Python 3.12+, PyTorch, torchaudio. Install deps: `uv sync`.
