#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run python -m src.train \
    --data_dir "$REPO_ROOT/dataset" \
    --checkpoint_dir "$REPO_ROOT/checkpoints" \
    --batch_size 28 \
    --max_steps 1500000 \
    --save_every 5000
