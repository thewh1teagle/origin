# Architecture

Speech autoencoder based on SupertonicTTS (arXiv:2503.23108), Module 1.

## Overview

```
wav → SpecProcessor → LatentEncoder → 24-dim latents → LatentDecoder → wav
```

Only the **LatentDecoder** is used at TTS inference. The encoder is trained to provide latents for the decoder during autoencoder training, and later to extract latents for TTL module training.

---

## SpecProcessor

Extracts concatenated log-magnitude spectral features from waveform.

- Linear STFT: `n_fft=2048` → 1025 bins
- Mel STFT: `n_mels=228`
- Output: `(B, 1253, T)` — log-scaled, concat of both

---

## LatentEncoder

| Layer | Details |
|-------|---------|
| Conv1d | 1253 → 512, k=7 |
| BatchNorm1d | |
| 10× ConvNeXtBlock | hidden=512, intermediate=2048, k=7 |
| Linear | 512 → 24 |
| LayerNorm | |

Output: `(B, 24, T)` latents at ~86 Hz.

**ConvNeXtBlock**: depthwise conv (k=7) → LayerNorm → Linear → GELU → Linear + residual.

---

## LatentDecoder

All convolutions are **causal** (left-padding only) to support streaming inference.

| Layer | Details |
|-------|---------|
| CausalConv1d | 24 → 512, k=7 |
| BatchNorm1d | |
| 10× CausalConvNeXtBlock | hidden=512, intermediate=2048, k=7 |
| BatchNorm1d | |
| CausalConv1d (head) | 512 → 2048, k=3 |
| PReLU + Linear | 2048 → 512 |
| Flatten | (B, 512, T) → (B, 1, T×512) |

Dilation schedule: `[1, 2, 4, 1, 2, 4, 1, 1, 1, 1]`.

The final flatten (WaveNeXt-style) maps 512 frame-level channels × T frames directly to T×512 waveform samples — no upsampling network needed.

Output: `(B, 1, T_audio)` at 44.1 kHz.

---

## Discriminators

### Multi-Period Discriminator (MPD)

5 sub-discriminators, periods `[2, 3, 5, 7, 11]`. Each reshapes waveform into 2D `(T/p, p)` and runs 5 Conv2D layers with channels `[16, 64, 256, 512, 512, 1]`. Weight-normalized.

### Multi-Resolution Discriminator (MRD)

3 sub-discriminators, FFT sizes `[512, 1024, 2048]`. Each computes log-magnitude STFT and runs 6 Conv2D layers `(1→16→16→16→16→16→1)`. Weight-normalized.

---

## Training

### Loss

```
L_G = 45 × L_recon + 1 × L_adv + 0.1 × L_fm
```

| Loss | Description |
|------|-------------|
| `L_recon` | Multi-resolution mel L1 — FFT sizes [1024, 2048, 4096], mel bands [64, 128, 128], hop=FFT/4 |
| `L_adv` | Least-squares GAN: `E[(1 - D(G(x)))²]` |
| `L_fm` | L1 feature matching between discriminator intermediate layers |

### Optimizer

- AdamW, `lr=2e-4`, betas `(0.8, 0.99)`, eps `1e-9`
- ExponentialLR, `gamma=0.999`, stepped every 1000 steps
- Alternating updates: discriminator first, then generator

### Data

- Segment length: 0.19s (`8379` samples at 44.1 kHz)
- Batch size: 28
- Max steps: 1.5M

---

## Audio Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 44,100 Hz |
| FFT size | 2,048 |
| Hop size | 512 |
| Mel bands | 228 |
| Latent dim | 24 |
| Latent rate | ~86 Hz |
