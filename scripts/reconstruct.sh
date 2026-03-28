#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <checkpoint> <input.wav> <output.wav>"
    exit 1
fi

CKPT="$1"
INPUT="$2"
OUTPUT="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp /tmp/latents_XXXXXX.pt)"
trap 'rm -f "$TMP"' EXIT

cd "$REPO_ROOT"
uv run python -m src.encode "$CKPT" "$INPUT" "$TMP"
uv run python -m src.decode "$CKPT" "$TMP" "$OUTPUT"
