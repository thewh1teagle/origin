# SupertonicTTS — Paper Notes

**Paper:** SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System
**ArXiv:** 2503.23108 (v3, 23 Sep 2025)
**Authors:** Hyeongju Kim, Jinhyeok Yang, Yechan Yu, Seunghun Ji, Jacob Morton, Frederik Bous, Joon Byun, Juheon Lee (Supertone, Inc.)
**Demo:** https://supertonictts.github.io/
**PDF:** `paper_2503.23108.pdf`

---

## Overview

SupertonicTTS is a zero-shot TTS system with **44M total parameters** and real-time factors of 0.02 (RTX 4090) / 0.05 (RTX 3090). It achieves competitive performance against much larger models (VALL-E 410M, VoiceBox 371M, CLaM-TTS >1.3B) by:

- Operating on **raw characters** (no G2P, no phoneme-level duration)
- Using **no external aligners or pretrained text encoders**
- A very **low-dimensional latent space** (C=24)
- **Temporal compression** of latents (6× compression)
- **ConvNeXt blocks** throughout for efficiency
- **Context-sharing batch expansion** for stable alignment training

---

## System Architecture (3 modules, trained sequentially)

### Module 1: Speech Autoencoder

Trained first independently. Converts mel spectrograms ↔ continuous latents.
Acts as a neural vocoder with a compact latent space in the middle.
**Latent encoder is NOT used at inference** — only the decoder.

#### Latent Encoder
- Input: 228-channel mel spectrogram (44.1 kHz, FFT=2048, hop=512, 228 mel bands)
- Conv1d → BatchNorm → **10 ConvNeXt blocks** (hidden=512, intermediate=2048) → Linear → LayerNorm
- Output: **24-dimensional latents** at same temporal resolution as mel (86 Hz)
- All conv kernel size: 7

#### Latent Decoder (= vocoder at inference)
- Input: 24-dim latents
- CausalConv1d → BatchNorm → **10 dilated CausalConvNeXt blocks** (hidden=512, intermediate=2048)
  - Dilation rates: [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
  - All depthwise conv: kernel size 7, causal (for streaming)
- BatchNorm → CausalConv1d (k=3) → 2048-dim → Linear → 512 channels (frame-level) → **flatten → waveform**
- Inspired by WaveNeXt; causal layers enable streaming

#### Discriminators (GAN training)
- **Multi-Period Discriminators (MPD):** periods [2, 3, 5, 7, 11]; lightweight — 6 conv layers [16, 64, 256, 512, 512, 1]
- **Multi-Resolution Discriminators (MRD):** FFT sizes [512, 1024, 2048]; 6 Conv2D layers (see Table 7)
  - Window sizes = FFT sizes, Hann window

#### Loss (Autoencoder)
```
L_G = λ_recon * L_recon + λ_adv * L_adv + λ_fm * L_fm
λ_recon=45, λ_adv=1, λ_fm=0.1
```
- **L_recon:** multi-resolution spectral L1; FFT sizes [1024, 2048, 4096], mel bands [64, 128, 128]
- **L_adv:** least-squares GAN loss
- **L_fm:** L1 distance between intermediate discriminator features

#### Training (Autoencoder)
- 1.5M iterations, AdamW, lr=2×10⁻⁴, batch=28
- Randomly crop 0.19s segments for adversarial training
- 4× RTX 4090 GPUs

---

### Module 2: Text-to-Latent (TTL) Module

Flow-matching model. Maps (text, reference speech) → latent representation.
Operates on **temporally compressed latents**: reshape (C=24, T) → (K_c×C=144, T/K_c) where **K_c=6**.
This drops the latent rate from ~86 Hz → ~14 Hz, matching semantic token rates.

#### 2a. Reference Encoder
- Input: temporally compressed latents (144-dim) from a cropped segment of reference speech
- Linear(144→128) → **6 ConvNeXt blocks** (kernel=5, intermediate=512) → **2 cross-attention layers**
- First cross-attention: **50 learnable query vectors** (128-dim) → fixed-size reference representation
- Output: reference key/value vectors (50 × 128)

#### 2b. Text Encoder
- Input: raw character sequence
- **Embedder:** character → 128-dim lookup table
- **6 ConvNeXt blocks** (kernel=5, intermediate=512)
- **4 Self-Attention blocks** (512 filter channels, 4 heads, rotary position embeddings)
- **2 Cross-Attention layers** using reference key/value from Reference Encoder
- First cross-attention uses 50 learnable keys (reused from reference encoder)
- Output: speaker-adaptive text representations

#### 2c. Vector Field (VF) Estimator
- Input: noisy latents z_t (144-dim)
- Linear(144→256)
- **N_m=4 repeated main blocks**, each containing:
  - DilatedConvNeXt (dilation rates 1,2,4,8 across the 4 blocks; kernel=5, intermediate=1024)
  - TimeCondBlock: single linear layer projecting 64-dim time embedding (sinusoidal, same as Grad-TTS)
  - TextCondBlock: cross-attention with text encoder output
  - ConvNeXt (kernel=5, intermediate=1024)
  - RefCondBlock: cross-attention with reference encoder output
  - ConvNeXt (kernel=5, intermediate=1024)
- **4 additional ConvNeXt blocks**
- Linear(256→144)
- Output: predicted vector field (144-dim)

#### Context-Sharing Batch Expansion
Key training trick. For each sample in mini-batch:
1. Encode conditioning variables once: `ĉ_i = g_φ(c_i)`
2. Generate **K_e=4** noisy versions with different noise ε and timestep t
3. Share the same encoded condition across all K_e samples
4. Append (x̃ᵢᵏ, ĉᵢ, tᵢᵏ) to expanded batch B_exp

Effectively: B=64, K_e=4 → effective batch of 256 but only 64% memory overhead (vs 253% for true B=256).
Key insight: same text-speech pair with **varying noise+timestep** is more effective for alignment learning than different text-speech pairs.

#### Loss (TTL)
```
L_TTL = E_{t,(z_1,c),p(z_0)} [ ||m · (v(z_t, z_ref, c, t) − (z_1 − (1−σ_min)z_0))||_1 ]
```
- m = reference mask (applied to prevent info leakage from cropped reference)
- σ_min = 1×10⁻⁸
- p_uncond = 0.05 (classifier-free guidance training; conditioning replaced with learnable params)
- Classifier-Free Guidance (CFG) at inference

#### Training (TTL)
- 700k iterations, AdamW, lr=5×10⁻⁴ (halved every 300k), batch=64, K_e=4
- 4× RTX 4090 GPUs
- Latents normalized with precomputed channel-wise mean/variance
- Reference segments: random crop 0.2s–9s, max half of original speech

---

### Module 3: Duration Predictor (~0.5M params)

Predicts **utterance-level total duration** (not phoneme-level). Simple, lightweight.

#### DP Reference Encoder
- Same architecture as TTL Reference Encoder but smaller:
- Linear(144→64) → **4 ConvNeXt blocks** (kernel=5, intermediate=256) → **2 cross-attention layers**
- First cross-attention: **8 learnable query vectors** → stacked → 64-dim final embedding

#### DP Text Encoder
- Embedder: character → 64-dim
- **6 ConvNeXt blocks** (kernel=5, intermediate=256)
- **2 Self-Attention blocks** (256 filter channels, 2 heads, rotary PE)
- Prepend a learnable **64-dim utterance token**
- Linear layer: first token output → **64-dim utterance-level text embedding**

#### Duration Estimator
- Concat [ref_embed (64); text_embed (64)] → 128-dim... wait, paper says 164-dim.
  (Likely 100-dim ref from stacked 8×... check: 8 queries × stacking along channel? Actually the DP ref encoder outputs via stacking → 64 per attention block × 2 blocks... the paper says "stacking outputs along the channel dimension, resulting in a 64-dimensional vector" — so ref=64, text=64, but paper says first linear is 164. Possible that it's [64+64+36] or there's a detail in the arch. Use 128 as working assumption, verify during implementation.)
- Linear(164, 164) → PReLU → Linear(164, 1) → scalar duration

#### Training (Duration Predictor)
- 3,000 iterations, AdamW, lr=5×10⁻⁴, batch=128, single RTX 4090
- Loss: L1 between predicted and ground-truth total duration
- Reference: random 5%–95% segment of input speech

---

## Audio Processing Details

| Parameter | Value |
|-----------|-------|
| Sample rate | 44,100 Hz |
| FFT size | 2,048 (46.43 ms) |
| Hop size | 512 (11.61 ms) |
| Window | Hann |
| Mel bands | 228 |
| Latent dim C | 24 |
| Latent rate (encoder) | ~86 Hz |
| Compression factor K_c | 6 |
| Compressed latent dim | 144 (= 6×24) |
| Compressed latent rate | ~14 Hz |

---

## Training Datasets

### Speech Autoencoder
- Public datasets + internal database
- **11,167 hours** total from **~14,000 speakers**

### TTL Module + Duration Predictor
- LJSpeech, VCTK, Hi-Fi TTS, LibriTTS
- **945 hours**, **2,576 English speakers**
- All resampled to 44.1 kHz

---

## Inference

1. **Duration Predictor** → total number of latent frames T
2. Sample z_0 ~ N(0,I) of shape (144, T/K_c)
3. **TTL module** integrates ODE via Euler's method with **NFE=32** steps (optimal trade-off)
4. CFG applied during ODE integration
5. Decompress latents: (144, T/K_c) → (24, T)
6. **Latent Decoder** → waveform at 44.1 kHz

**NFE trade-off** (Table 8, LS-clean):
| NFE | RTF | WER | SIM | NISQA |
|-----|-----|-----|-----|-------|
| 4 | 0.006 | 11.43 | 0.335 | 2.623 |
| 16 | 0.019 | 2.679 | **0.476** | 3.994 |
| 32 | 0.037 | **2.639** | 0.472 | 4.033 |
| 128 | 0.140 | 2.693 | 0.468 | **4.070** |

---

## Results

### Speech Autoencoder (Table 2)
| Model | NISQA (LT-clean) | UTMOSv2 | V/UV F1 | RTF (RTX 4090) |
|-------|-----------------|---------|---------|----------------|
| GT | 4.09 | **3.26** | — | — |
| BigVGAN | 4.11 | 3.16 | **0.9735** | 0.0124 |
| **Ours** | 4.06 | 3.13 | 0.9587 | **0.0006** |

> 20× faster than BigVGAN at inference.

### Zero-Shot TTS (Table 5)
| Model | WER (LS-clean) | CER | Params | RTF |
|-------|---------------|-----|--------|-----|
| VALL-E | 5.9 | — | 410M† | ~0.64 |
| VoiceBox | **1.9** | — | 371M | ~0.62 |
| CLaM-TTS | 5.11 | 2.87 | >1.3B† | 0.42 |
| DiTTo-TTS | 2.56 | 0.89 | 940M | 0.16 |
| **Ours** | 2.64 | **0.83** | **44M** | **0.02** |

### Parameter Breakdown
| Component | Params |
|-----------|--------|
| Duration Predictor | ~0.5M |
| Text-to-Latent | ~18.5M |
| Vocoder (Latent Decoder) | ~25M |
| **Total** | **~44M** |

---

## Key Design Decisions / Gotchas

1. **No G2P, no aligner, no pretrained text encoder** — everything learned end-to-end from characters
2. **Latent encoder not used at inference** — only trained to enable fast latent encoding during TTL training
3. **Context-sharing batch expansion** is NOT the same as increasing batch size — it specifically helps alignment by showing the model the same text-speech pair with varied noise levels
4. **Temporal compression** reshapes by stacking channels, not subsampling — all temporal info preserved, perfect inversion possible
5. **Utterance-level duration** (not phoneme) — more robust to errors in duration estimation
6. **CFG with p_uncond=0.05** — classifier-free guidance applied at inference
7. **Reference masking** during TTL training — mask is applied to reference segment in the flow-matching loss to prevent info leakage
8. **Causal decoder** — designed for streaming/real-time use

---

## Dependencies / Related Architectures

- **ConvNeXt** (Liu et al. 2022) — backbone throughout
- **Vocos** (Siuzdak 2024) — base for autoencoder
- **WaveNeXt** (Okamoto et al. 2023) — inspiration for decoder
- **Flow Matching** (Lipman et al. 2023) — TTL training objective
- **HiFi-GAN** (Kong et al. 2020) — MPD discriminators
- **NANSY++** (Choi et al. 2023) — timbre token block design for reference encoder
- **Grad-TTS** (Popov et al. 2021) — time embedding method
- **BigVGAN** (Lee et al. 2023) — vocoder baseline

---

## Open Questions for Implementation

- [ ] Exact input dimension discrepancy in Duration Estimator (164 vs expected 128) — verify from appendix or code
- [ ] Exact architecture of self-attention blocks (layer norm positions, FFN details)
- [ ] Rotary position embedding implementation details
- [ ] Whether the reference mask m is a binary or soft mask
- [ ] Whether latent normalization stats are computed globally or per-speaker
