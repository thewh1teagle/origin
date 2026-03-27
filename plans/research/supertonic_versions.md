# Supertonic v1 vs v2 — Differences & Gaps

## Version overview

| | v1 | v2 |
|--|----|----|
| HuggingFace | `Supertone/supertonic` | `Supertone/supertonic-2` |
| Languages | English only | en, ko, es, pt, fr |
| Params (stated) | 44M (paper) | 66M (HF model card) |
| Paper | arXiv:2503.23108 | No separate paper yet |
| Release | 2025 | Jan 2026 |

## Parameter count discrepancy — 44M vs 66M

The paper claims 44M total. The deployed v2 model card says 66M. Gap = ~22M extra params.

Likely explanations:
- **Multilingual embedding table** — v2 supports 5 languages via `<lang>` tokens + unicode indexer. A larger character vocab adds embedding params.
- **Expanded text encoder** — may have grown to handle more scripts (Korean, Spanish etc.)
- **The paper describes v1 (English only)** — v2 may have deliberately scaled up certain modules for multilingual quality

**Implication for us:** We're building English-only first, so 44M is the right target. Don't worry about the 66M figure.

## ONNX model structure (from SDK code)

The deployed model splits into 4 separate ONNX files:

| File | Module |
|------|--------|
| `duration_predictor.onnx` | Duration predictor |
| `text_encoder.onnx` | Text encoder + reference encoder (pre-computed, cached per voice) |
| `vector_estimator.onnx` | VF estimator (the flow-matching ODE step, called N times) |
| `vocoder.onnx` | Latent decoder (autoencoder decoder) |

This split is smart for inference: `text_encoder` runs once per utterance, `vector_estimator` runs N times (one per ODE step).

## Voice styles

Voices are stored as JSON files with pre-computed reference embeddings:
- `style_ttl` — reference embedding for the TTL module
- `style_dp` — reference embedding for the duration predictor

Shape is `[1, dim1, dim2]`. This means at inference time the reference speech is encoded once offline and saved — no need to run the reference encoder live.

## What to inspect next

Download the ONNX models and config to fill paper gaps:
```bash
cd plans/research/supertonic
git lfs install
git clone https://huggingface.co/Supertone/supertonic assets
```

Key files to inspect:
- `assets/tts.json` — exact config (sample rate, latent dim, chunk sizes, mel params)
- `assets/unicode_indexer.json` — character vocab
- ONNX graphs — inspect with Netron or `onnx` Python library for exact shapes
