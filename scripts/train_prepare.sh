#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$REPO_ROOT/dataset"
URL="https://huggingface.co/datasets/thewh1teagle/ljspeech-enhanced/resolve/main/ljspeech-enhanced-v1.7z?download=true"
ARCHIVE="$DATASET_DIR/ljspeech-enhanced-v1.7z"

mkdir -p "$DATASET_DIR"

echo "==> Downloading LJSpeech-enhanced (~14 GB)..."
wget -c "$URL" -O "$ARCHIVE"

echo "==> Extracting..."
7z x "$ARCHIVE" -o"$DATASET_DIR" -y

echo "==> Dataset ready at $DATASET_DIR"
