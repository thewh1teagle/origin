# ONNX Reverse Engineering — Supertonic v1

Findings from inspecting `tts.json`, `tts.yml`, voice styles, unicode indexer, and ONNX model graphs.
Source: `https://huggingface.co/Supertone/supertonic/tree/main/onnx`

---

## Scripts

| Script | What it does |
|--------|-------------|
| `download.py` | Fetch config/style files + ONNX models. Use `--no-onnx` to skip large files |
| `inspect_config.py` | Pretty-print `tts.json` and `tts.yml` |
| `inspect_voice_style.py` | Show shape/stats of reference embeddings |
| `inspect_unicode.py` | Analyze the character vocab |
| `inspect_shapes.py` | Print input/output tensor shapes of all 4 ONNX models |

```bash
uv run --with numpy download.py --no-onnx    # small files only
uv run --with numpy download.py              # everything incl. ONNX
uv run --with onnx inspect_shapes.py
uv run --with numpy inspect_voice_style.py
uv run --with pyyaml inspect_config.py
```

---

## ONNX Model Shapes (ground truth)

| Model | Size | Inputs | Outputs |
|-------|------|--------|---------|
| `duration_predictor.onnx` | 1.5 MB | `text_ids [B, T_text]`, `style_dp [B, 8, 16]`, `text_mask [B, 1, T_text]` | `duration [B]` scalar seconds |
| `text_encoder.onnx` | 27 MB | `text_ids [B, T_text]`, `style_ttl [B, 50, 256]`, `text_mask [B, 1, T_text]` | `text_emb [B, 256, T_text]` |
| `vector_estimator.onnx` | 132 MB | `noisy_latent [B, 144, T_lat]`, `text_emb [B, 256, T_text]`, `style_ttl [B, 50, 256]`, `latent_mask`, `text_mask`, `current_step [B]`, `total_step [B]` | `denoised_latent [B, 144, T_lat]` |
| `vocoder.onnx` | 99 MB | `latent [B, 144, T_lat]` | `wav_tts [B, T_audio]` |

---

## Key Findings

### Encoder input: idim=1253 explained

**1253 = linear STFT bins + mel bins = 1025 + 228**

The encoder does NOT take mel spectrogram alone. It concatenates:
- Linear (magnitude) spectrogram: n_fft//2+1 = 1025 bins
- Mel spectrogram: 228 bands

This gives the encoder both fine-grained frequency resolution and perceptually-weighted features.

### Vocoder ONNX internals (from node inspection)

The vocoder ONNX does 3 things internally:
1. **Denormalize latents**: divide by `normalizer.scale=0.25` (i.e. multiply by 4)
2. **Temporal decompression**: reshape `[B, 144, T/6]` → `[B, 24, T]` (Reshape + Transpose)
3. **Run the causal decoder**: ConvNeXt blocks → head → flatten to waveform

So the vocoder ONNX takes **compressed 144-dim latents** as input (not 24-dim). The 24-dim latent space exists at the PyTorch training level, but in the deployed ONNX the decompression is baked in.

### Decoder head architecture (confirmed from tts.json)

Config: `head: {idim: 512, hdim: 2048, odim: 512, ksz: 3}`

Structure: `CausalConv1d(512→2048, ksz=3)` → `BatchNorm(2048)` → `PReLU` → `Linear(2048→512)` → flatten

- `idim=512` matches decoder `hdim` (input from ConvNeXt blocks)
- `hdim=2048` is the expanded intermediate
- `odim=512` is the final frame-level channel count (flattened to waveform)
- No extra same-dim linear — the expansion/projection is done by conv + one linear only

### fmin/fmax for mel spectrogram

`tts.yml` contains `filter_bank_path` pointing to an internal Supertone file not on HuggingFace.
However, given:
- Sample rate: 44100 Hz
- n_mels: 228 (unusually high)
- Standard librosa defaults: fmin=0, fmax=sr/2=22050

Most likely `fmin=0, fmax=22050 (Nyquist)`. The high mel count (228) only makes sense if covering the full frequency range — otherwise you'd use fewer bands. **Use fmin=0, fmax=22050 as working assumption.**

---

## Paper vs Reality — Complete Gap Table

### Autoencoder

| Parameter | Paper | Actual (tts.json) | Notes |
|-----------|-------|--------|-------|
| Encoder input | mel (228) | **concat(linear, mel) = 1253** | Confirmed from config `encoder.idim=1253` |
| fmin/fmax | not stated | unknown, likely 0/22050 | `filter_bank_path` is internal |
| Encoder dilation | not stated | all 1s | Non-dilated encoder |
| Decoder dilation | [1,2,4,1,2,4,1,1,1,1] | [1,2,4,1,2,4,1,1,1,1] ✓ | |
| Decoder head | CausalConv→BN→Linear→PReLU→Linear | **CausalConv→BN→PReLU→Linear** | No same-dim linear; confirmed from head config |
| Latent normalizer scale | not stated | 0.25 (÷ in vocoder) | TTL output is divided by 0.25 |
| Vocoder input | 24-dim latent | **144-dim compressed latent** | Decompression baked into vocoder ONNX |

### TTL Module

| Parameter | Paper | Actual | Notes |
|-----------|-------|--------|-------|
| char_emb_dim | 128 | **256** | 2× wider than paper |
| TTL hidden dim | 128 | **256** | All TTL ConvNeXt/attn dims |
| VF hidden dim | 256 | **512** | VF estimator 2× wider |
| K_e batch expand | 4 | **6** | Different expansion factor |
| sig_min | 1e-8 | **0** | |
| uncond prob | p_uncond=0.05 | prob_both_uncond=0.04, prob_text_uncond=0.01 | Split into two probabilities |
| rotary_base | not stated | 10000 | Standard RoPE base |
| rotary_scale | not stated | 10 | |
| text_emb output | not stated | 256-dim per token | Confirmed from ONNX |

### Duration Predictor

| Parameter | Paper | Actual | Notes |
|-----------|-------|--------|-------|
| style_value_dim | not stated | 16 | DP style tokens are 16-dim |
| predictor hdim | "164" (paper inconsistency) | **128** | Resolved: 128 is correct |
| style output | not stated | 8×16=128 concat with text 64 → 128 hdim | |

---

## Voice Style Embeddings

Pre-computed offline from reference audio, stored as JSON:

| Field | Shape | Content |
|-------|-------|---------|
| `style_ttl` | [1, 50, 256] | 50 style tokens × 256-dim for TTL cross-attention |
| `style_dp` | [1, 8, 16] | 8 style tokens × 16-dim for duration predictor |

Source audio: `F1.wav` at 44100 Hz. Extracted 2025-11-18.

---

## Character Vocabulary

- **Size: 81 tokens** (indices 0–80)
- **Lookup table**: `unicode_indexer.json` is a flat list of 65536 entries; `list[codepoint]` → token_id, -1 if OOV
- **Characters**: space, punctuation, digits 0-9, A-Z, a-z
- **Only 2 non-ASCII**: `£` (U+00A3) → 79, combining accent `́` (U+0301) → 80
- **Language tags** prepended/appended: `<en>text</en>` — these characters (`<`, `>`, `/`) must be in vocab

---

## All Questions Resolved ✓

### fmin/fmax — CONFIRMED: fmin=0, fmax=22050
Verified via Vocos source (`gemelo-ai/vocos`, cloned to `plans/research/vocos`).
`MelSpectrogramFeatures` uses `torchaudio.transforms.MelSpectrogram` with no explicit `f_min`/`f_max` — torchaudio defaults are `f_min=0.0`, `f_max=sample_rate/2`. At 44100 Hz → **fmax=22050**.

### Mel normalization — CONFIRMED: log(clip(mel, min=1e-5))
Vocos uses `safe_log(x, clip_val=1e-7)` = `torch.log(torch.clip(x, min=1e-7))`.
SupertonicTTS config has `eps=1e-5` — same pattern, slightly different clip value.
```python
log_mel = torch.log(torch.clamp(mel, min=1e-5))  # no z-score, just log with epsilon floor
```
`norm_mean=0.0, norm_std=1.0` in config are identity values — no z-score normalization applied.
