# Open Source References for SupertonicTTS Autoencoder

## Summary Table

| Project | GitHub | License | Training Code | Pretrained (44kHz) | Suitability |
|---------|--------|---------|--------------|-------------------|-------------|
| **Vocos** | `hubertsiuzdak/vocos` | MIT | Yes (PTL) | Yes (mel+EnCodec, HF Hub) | **Best base** — ConvNeXt blocks, MPD+MRD, mel extractor |
| **BigVGAN v2** | `NVIDIA/BigVGAN` | MIT | Yes (distributed) | Yes (44kHz on HF) | **Best discriminators** — MPD [2,3,5,7,11] + MRD [512,1024,2048] match exactly |
| **DAC** | `descriptinc/descript-audio-codec` | MIT | Yes (PTL) | Yes (44kHz on HF) | Very good training loop + discriminators at 44kHz |
| **HiFi-GAN** | `jik876/hifi-gan` | MIT | Yes | Yes (LJSpeech, VCTK) | MPD only (no MRD, no ConvNeXt) |
| **EnCodec** | `facebookresearch/encodec` | MIT (code) / CC-BY-NC (weights) | Partial | CC-BY-NC | Encoder-decoder-GAN loop reference |
| **Amphion/FireflyGAN** | `open-mmlab/Amphion` | MIT | Yes | Yes | Unified framework with ConvNeXt vocoder |
| **WaveNeXt** | No official repo | — | No | No | Paper only; implement on top of Vocos |

---

## Vocos — `hubertsiuzdak/vocos`

**Why it's the best starting point:**
- `vocos/modules.py` — `ConvNeXtBlock` (depthwise conv + LayerNorm + GELU FFN); reuse verbatim for encoder
- `vocos/feature_extractors.py` — `MelSpectrogramFeatures`; adapt for 228-band, 44.1kHz, FFT=2048, hop=512
- `vocos/discriminators.py` — `MultiPeriodDiscriminator` + `MultiResolutionDiscriminator` already present
- `vocos/loss.py` — spectral L1 + adversarial + feature matching
- Full PyTorch Lightning training loop

**Gaps to fill:**
- No encoder (Vocos takes mel directly, no bottleneck) → add Conv1d + 10 ConvNeXt + Linear + LayerNorm projecting to 24-dim
- Decoder ConvNeXt blocks are non-causal → replace depthwise conv padding with causal left-padding
- No dilation schedule → add dilation [1,2,4,1,2,4,1,1,1,1] to decoder blocks
- Output head: replace Fourier head with Linear(2048→512) + PReLU + flatten (WaveNeXt-style)
- 44.1kHz config (Vocos ships 22kHz and 24kHz)

---

## BigVGAN v2 — `NVIDIA/BigVGAN`

**Why to use its discriminators:**
- `discriminators.py` has MPD with periods [2,3,5,7,11] matching spec exactly
- MRD with FFT sizes [1024, 2048, 512] (= your [512, 1024, 2048]) matching exactly
- `loss.py` — feature matching + mel recon + adversarial, all present
- Battle-tested distributed training loop
- 44kHz pretrained models exist (can verify against for autoencoder quality)

**Note:** BigVGAN v2 specifically introduced MRD (v1 used multi-scale discriminator), so use v2 branch.

---

## DAC — `descriptinc/descript-audio-codec`

**Why it's worth referencing:**
- Cleanest 44kHz training setup available in open source
- `dac/model/discriminator.py` — MPD + MSTFTD (multi-scale STFT discriminator) — very close to your MRD
- PyTorch Lightning training, well-organized loss balancing
- 44kHz model on HF can be used for sanity checks

---

## Recommended Build Strategy

### Step 1: Fork Vocos
Use Vocos as the base repository. Retains:
- ConvNeXtBlock (encoder, non-causal decoder blocks)
- MPD + MRD discriminators
- Mel feature extractor (adapt params)
- PyTorch Lightning training loop
- Loss functions

### Step 2: Add Latent Encoder
```python
# Insert between mel extractor and ConvNeXt blocks:
nn.Conv1d(228, 512, kernel_size=7, padding=3)  # mel → hidden
nn.BatchNorm1d(512)
# ... 10x ConvNeXtBlock(dim=512, intermediate=2048, kernel=7) ...
nn.Linear(512, 24)    # hidden → latent
nn.LayerNorm(24)
```

### Step 3: Causal Decoder
Replace Vocos' non-causal decoder ConvNeXt blocks:
```python
# Causal depthwise conv: pad (dilation*(k-1), 0) on the left, no right padding
F.pad(x, ((kernel_size-1)*dilation, 0))
nn.Conv1d(dim, dim, kernel_size=7, groups=dim, dilation=d, padding=0)
```
Apply dilation schedule: [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]

### Step 4: Lift BigVGAN v2 Discriminators
Copy `discriminators.py` from BigVGAN v2 for the MPD + MRD combo.
Adjust MRD FFT sizes if Vocos' version differs.

### Step 5: 44.1kHz Config
- Set sample_rate=44100, n_fft=2048, hop_length=512, n_mels=228
- Segment length: 0.19s × 44100 ≈ 8379 samples → use 8192 or match to hop multiple
- Adjust discriminator input sizes accordingly

---

## Pretrained Weights for Comparison / Fine-tuning

| Model | URL | Notes |
|-------|-----|-------|
| BigVGAN 44kHz 128-band | `nvidia/bigvgan_v2_44khz_128band_512x` | Different mel config but same SR |
| Vocos mel 22kHz | `charactr/vocos-mel-22khz` | Half the SR, useful for architecture check |
| DAC 44kHz | `descript/dac-44khz` | Good quality reference |

**For our autoencoder:** We train from scratch (no fine-tuning of existing weights) since our mel config (228 bands, 44.1kHz) doesn't match any existing checkpoint. Can use BigVGAN 44kHz output for perceptual comparison during dev.
