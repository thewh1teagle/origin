# LibriLight Training Plan

## Phase 1: Initial Validation (LJSpeech, 10h)

- Train on LJSpeech-enhanced as-is (already running)
- At ~5k–20k steps, run `reconstruct.sh` on a held-out clip
- Goal: confirm architecture works, losses converge, audio is intelligible
- **Do not invest in infrastructure until this passes**

---

## Phase 2: Full Training (LibriLight, ~11k hours)

### Dataset

- **Source:** Meta's LibriLight — `https://dl.fbaipublicfiles.com/librilight/data/`
- **Subsets:**
  - `small` — 577 hours
  - `medium` — 5,000 hours
  - `large` — 51,000 hours
- **Target:** `small` + `medium` ≈ 5,600 hours (or `large` for full 11k+)
- **Format:** `.flac`, 16 kHz, ~7,000 speakers
- HuggingFace mirror: `facebook/librilight`

### Resampling

- On-the-fly (16 kHz → 44.1 kHz) — cheap enough, GPU is the bottleneck
- No pre-resampling needed

### Hardware

| Setup | Est. time for 1.5M steps |
|-------|--------------------------|
| 1× RTX 4090 | ~13 days |
| 2× RTX 4090 | ~6–7 days |
| 4× RTX 4090 (paper) | ~3–4 days |

- Use `accelerate` for multi-GPU — same pattern as renikud project

### Code Changes Required

1. **`src/nn/data.py`** — glob `*.flac` in addition to `*.wav`
2. **`src/train.py`** — wrap with `accelerate` (model, optimizer, dataloader)
3. **`scripts/train_scratch.sh`** — replace `uv run python` with `accelerate launch`
4. **`src/train.py`** — bump `num_workers` back to 4–8 (safe with accelerate)

---

## Execution Order

1. ✅ LJSpeech training running
2. [ ] Check reconstruction at ~20k steps — run `reconstruct.sh`
3. [ ] Add `.flac` support to data loader
4. [ ] Add `accelerate` multi-GPU support
5. [ ] Download LibriLight `small` + `medium`
6. [ ] Launch full training with `accelerate launch`
