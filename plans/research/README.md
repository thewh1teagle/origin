# Research Notes — SupertonicTTS

Notes and references for reproducing and building on the SupertonicTTS paper (arXiv:2503.23108).

## Files

| File | What's in it |
|------|-------------|
| `paper_2503.23108.pdf` | Original paper |
| `paper_notes.md` | Full system summary — all 3 modules, training setup, results |
| `autoencoder.md` | Deep dive on the speech autoencoder — exact architecture, losses, hyperparams |
| `autoencoder_feasibility.md` | Honest assessment of what's missing from the paper and training effort |
| `open_source_refs.md` | Open source repos to build from (Vocos, BigVGAN v2, DAC, HiFi-GAN) |
| `supertonic_versions.md` | v1 vs v2 differences, param count discrepancy, HF model info |
| `supertonic/` | Cloned SDK repo (inference-only, multi-language examples) |

## Where to start

1. Read `paper_notes.md` for the big picture
2. Read `autoencoder.md` for implementation details on the first thing to build
3. Check `autoencoder_feasibility.md` for known gaps
4. Use `open_source_refs.md` to pick the right base repos

## Key open questions

- ~~Exact mel config (fmin/fmax)~~ — **resolved: fmin=0, fmax=22050** (Vocos torchaudio defaults)
- ~~Mel normalization~~ — **resolved: `log(clamp(mel, min=1e-5))`**, no z-score
- 44M (paper) vs 66M (deployed v2) parameter gap — see `supertonic_versions.md` (expected: multilingual expansion)
- TTL dims are 2× wider than paper states — see `onnx_re/README.md`

**No blockers. Ready to build the autoencoder.**
