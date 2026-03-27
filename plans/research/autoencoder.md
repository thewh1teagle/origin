# SupertonicTTS — Speech Autoencoder: Deep Dive

> Goal: train the autoencoder from scratch using open source code before tackling the full TTS system.
> The autoencoder is a **standalone neural vocoder** with a compact latent space in the middle.
> At TTS inference, only the **latent decoder** is used.

---

## Role in the Pipeline

```
[Training]   mel_spec → LatentEncoder → z (24-dim) → LatentDecoder → waveform
[TTS Infer]  z (24-dim, from TTL module) → LatentDecoder → waveform
```

The latent encoder is only used during:
- Training of the autoencoder itself
- Fast latent encoding during TTL module training (to avoid storing precomputed latents)

---

## Audio Spec

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 44,100 Hz | All data resampled to this |
| FFT size | 2,048 | 46.43 ms |
| Hop size | 512 | 11.61 ms |
| Window | Hann | Size = FFT size |
| Mel bands | 228 | Input to latent encoder |
| Mel range | not specified | Likely 0–22050 Hz or standard librosa defaults |
| Latent dim C | 24 | Much lower than 228 mel channels |
| Latent rate | ~86 Hz | Same as mel frame rate (44100/512) |

---

## Architecture

### Latent Encoder (Vocos-based)

```
Input: mel spectrogram  [B, 228, T]
  │
  ├─ Conv1d(228→512, k=7, padding=3)
  ├─ BatchNorm1d(512)
  ├─ 10× ConvNeXt Block (hidden=512, intermediate=2048, k=7)
  ├─ Linear(512→24)         ← projects to latent space
  └─ LayerNorm(24)

Output: latents  [B, 24, T]
```

**ConvNeXt Block** (standard 1D version):
```
Input x [B, C, T]
  ├─ DepthwiseConv1d(C, C, k=7, groups=C, padding=3)   # spatial mixing
  ├─ LayerNorm or transpose + LayerNorm
  ├─ Linear(C → 4C)   # channel mixing (intermediate = 4C)
  ├─ GELU
  ├─ Linear(4C → C)
  └─ residual add
```
> Note: Vocos uses LayerNorm, not BatchNorm, inside ConvNeXt blocks. The BatchNorm in the encoder is at the very start (after the initial Conv1d), not inside the ConvNeXt blocks.

### Latent Decoder (Vocos-based, causal + dilated)

```
Input: latents  [B, 24, T]
  │
  ├─ CausalConv1d(24→512, k=7)
  ├─ BatchNorm1d(512)
  ├─ 10× DilatedCausalConvNeXt Block (hidden=512, intermediate=2048, k=7)
  │     dilation rates: [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
  ├─ BatchNorm1d(512)
  ├─ CausalConv1d(512→2048, k=3)
  ├─ Linear(2048→512)    ← frame-level (512 values per frame)
  └─ Flatten: [B, 512, T] → [B, 1, T*hop?]   ← final waveform

Output: waveform  [B, 1, T_audio]
```

**Flatten detail:** The decoder outputs 512 values per frame at ~86 Hz. Flattening: 512 × (hop_size=512) / 512 = 512 samples per frame → 512 × T samples total = original audio length. Actually the 512 output channels after the linear serve as sub-samples within each frame (similar to Vocos/WaveNeXt's "flattening" approach that avoids upsampling artifacts).

**CausalConv1d:** right-padding = 0, left-padding = (k-1)*dilation. All decoder conv layers are causal.

**DilatedCausalConvNeXt Block:**
```
Input x [B, C, T]
  ├─ CausalDepthwiseConv1d(C, C, k=7, dilation=d)
  ├─ LayerNorm (or GroupNorm)
  ├─ Linear(C → 4C)
  ├─ GELU
  ├─ Linear(4C → C)
  └─ residual add
```

---

## Discriminators

### Multi-Period Discriminator (MPD) — lightweight version

Periods: **[2, 3, 5, 7, 11]** — same as original HiFi-GAN.
Each sub-discriminator reshapes audio into 2D and applies Conv2D layers.

Architecture per period p:
```
Reshape: [B, 1, T] → [B, 1, T/p, p]  (pad T to multiple of p)
Conv2d(1,    16,  (5,1), (3,1)) + LeakyReLU
Conv2d(16,   64,  (5,1), (3,1)) + LeakyReLU
Conv2d(64,   256, (5,1), (3,1)) + LeakyReLU
Conv2d(256,  512, (5,1), (3,1)) + LeakyReLU
Conv2d(512,  512, (5,1), (1,1)) + LeakyReLU
Conv2d(512,  1,   (3,1), (1,1))           ← output logits
```
> "Lightweight" = channels [16, 64, 256, 512, 512, 1] vs original HiFi-GAN [32, 128, 512, 1024, 1024, 1]

### Multi-Resolution Discriminator (MRD)

FFT sizes: **[512, 1024, 2048]**.
For each FFT size: log-scaled linear spectrogram → 6× Conv2D.

Architecture per FFT size (from Table 7):
```
Input: log linear spectrogram [B, 1, n_frames, n_fft//2+1]
Conv2d(1,  16, (5,5), (1,1)) + LeakyReLU
Conv2d(16, 16, (5,5), (2,1)) + LeakyReLU
Conv2d(16, 16, (5,5), (2,1)) + LeakyReLU
Conv2d(16, 16, (5,5), (2,1)) + LeakyReLU
Conv2d(16, 16, (5,5), (1,1)) + LeakyReLU
Conv2d(16,  1, (3,3), (1,1))              ← output logits
```
- Window size = FFT size (Hann), hop = FFT/4

---

## Loss Functions

### Generator Loss
```
L_G = 45 * L_recon + 1 * L_adv + 0.1 * L_fm
```

### Reconstruction Loss L_recon
Multi-resolution spectral L1 over **mel spectrograms**:
- FFT sizes: [1024, 2048, 4096]
- Mel bands: [64, 128, 128]
- Hop = FFT/4, Window = Hann
- L1 between log-mel of real vs generated

### Adversarial Loss L_adv (Least-Squares GAN)
```
Generator:     E[(D(G(x)) - 1)²]
Discriminator: E[(D(G(x)) + 1)² + (D(x) - 1)²]
```
Applied to both MPD and MRD outputs, summed.

### Feature Matching Loss L_fm
```
L_fm = (1/L) * Σ_{l=1}^{L} ||φ_l(G(x)) - φ_l(x)||_1
```
L1 distance between intermediate discriminator layer outputs (real vs generated), averaged over all layers L.

---

## Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 2×10⁻⁴ |
| Batch size | 28 |
| Iterations | 1,500,000 |
| GPUs | 4× RTX 4090 |
| Audio segment | 0.19s random crop (~8192 samples at 44.1kHz) |
| λ_recon | 45 |
| λ_adv | 1 |
| λ_fm | 0.1 |

**Training data:**
- 11,167 hours total
- ~14,000 speakers
- Mix of public datasets + internal data
- Public datasets listed in Appendix F (not shown in paper excerpt)

### Likely Public Datasets
Based on the TTL training set (LJSpeech, VCTK, Hi-Fi TTS, LibriTTS = 945h), the autoencoder likely adds:
- LibriSpeech (960h)
- Common Voice
- VoxPopuli
- AISHELL / multilingual datasets
- Possibly GigaSpeech, MLS

---

## Key Implementation Notes

### Vocos vs Standard HiFi-GAN/BigVGAN
- Vocos **does not use transposed convolutions or upsampling** — instead uses frame-level output + flattening
- This avoids aliasing artifacts from upsampling
- The "Fourier head" in original Vocos (produces STFT coefficients) is **replaced** in SupertonicTTS with two linear layers + PReLU
- The decoder in SupertonicTTS is inspired by WaveNeXt: higher dimensionality (512 channels) + nonlinearity

### Causal Convolutions
All decoder conv layers are **causal** (no right-padding). This enables:
- Streaming inference
- Low-latency TTS

### Latent Space Properties
- Continuous (not quantized)
- 24-dim (much lower than 80/100 mel channels used by most vocoders)
- Normalized with channel-wise mean/variance before TTL training
- The encoder produces latents at same temporal resolution as mel (~86 Hz)

---

## Training Strategy (Phase 1)

The paper trains the autoencoder completely **independently** before the TTL module.
Suggested training phases for our implementation:

### Phase 1A: Quick validation (~50k steps)
- Small subset of data (e.g., LJSpeech only, 24h)
- Verify reconstruction quality is reasonable
- Check latent space is well-behaved (not collapsed, not exploding)
- Metrics: mel reconstruction loss, listen to samples

### Phase 1B: Full training (~1.5M steps)
- Full dataset
- Log: reconstruction loss, adversarial losses, feature matching loss
- Checkpoint every 50k steps
- Evaluate: NISQA, V/UV F1

---

## Open Questions / Decisions Needed

- [ ] **Exact mel range** (fmin, fmax) — likely 0–8000 Hz or 0–Nyquist; check Vocos config for 44.1kHz
- [ ] **ConvNeXt normalization inside blocks** — LayerNorm (like Vocos) or GroupNorm?
- [ ] **Weight normalization** in discriminators? (used in HiFi-GAN, may or may not be in this lightweight version)
- [ ] **Segment length** — 0.19s = 8,378 samples at 44.1kHz → round to 8192 or 8448?
- [ ] **EMA weights?** — paper doesn't mention it but many vocoders use it
- [ ] **LR scheduler?** — paper doesn't specify one for autoencoder (TTL uses halving every 300k)
- [ ] **Dataset for Phase 1** — start with LibriTTS + LJSpeech (~1000h) or go bigger?
- [ ] **Normalization of mel input** — raw log-mel or normalized?

---

## Relation to Existing Open Source Code

See `open_source_refs.md` for full analysis. Summary:

**Recommended build strategy:** Fork **Vocos** (`hubertsiuzdak/vocos`, MIT) as the base, lift discriminators from **BigVGAN v2** (`NVIDIA/BigVGAN`, MIT).

| What | Source |
|------|--------|
| ConvNeXtBlock (encoder) | Vocos `modules.py` — use verbatim |
| Mel feature extractor | Vocos `feature_extractors.py` — adapt for 228-band 44.1kHz |
| Causal decoder blocks | Vocos ConvNeXt + add left-only causal padding |
| Dilation schedule [1,2,4,1,2,4,1,1,1,1] | Implement from WaveNeXt paper (no official repo) |
| MPD discriminator | BigVGAN v2 `discriminators.py` (periods [2,3,5,7,11]) |
| MRD discriminator | BigVGAN v2 `discriminators.py` (FFT [512,1024,2048]) |
| Loss functions | Vocos `loss.py` or BigVGAN `loss.py` |
| Training loop | Vocos PyTorch Lightning or BigVGAN distributed |

Key repos:
- **Vocos:** https://github.com/hubertsiuzdak/vocos
- **BigVGAN v2:** https://github.com/NVIDIA/BigVGAN
- **HiFi-GAN:** https://github.com/jik876/hifi-gan (MPD reference)
- **DAC:** https://github.com/descriptinc/descript-audio-codec (clean 44kHz training)
