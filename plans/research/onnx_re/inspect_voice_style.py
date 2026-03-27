"""
Inspect voice style JSON files — reveals shape of reference embeddings.
Usage: uv run inspect_voice_style.py
"""

import json
import os
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def inspect_style(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"Not found: {path} — run download.py first")
        return
    with open(path) as f:
        style = json.load(f)

    print(f"\n--- {filename} ---")
    for key, val in style.items():
        dims = val.get("dims", "?")
        data = np.array(val["data"], dtype=np.float32)
        print(f"  {key}:")
        print(f"    dims:  {dims}")
        print(f"    total elements: {data.size}")
        print(f"    min={data.min():.4f}  max={data.max():.4f}  mean={data.mean():.4f}  std={data.std():.4f}")


def main():
    for name in ["voice_styles_F1.json", "voice_styles_M1.json"]:
        inspect_style(name)


if __name__ == "__main__":
    main()
