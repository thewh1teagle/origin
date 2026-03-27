# origin

Python code for an audio autoencoder project.

## What is here

- `src/`: core model, losses, discriminators, config, and training code
- `plans/research/`: reference notes and experiments
- `pyproject.toml`: basic project metadata

## Run training

```bash
uv run python -m src.train --data_dir /path/to/wav_files
```

Expected dataset:

- `.wav` files under the given directory
- audio is loaded as mono and resampled to `44.1 kHz`

## Notes

- Python `3.12+`
- current config defaults live in `src/config.py`
- `README.md` intentionally stays minimal until inference/eval flows are finalized
