"""
Download Supertonic v1 ONNX models and config files from HuggingFace.
Usage: uv run download.py [--no-onnx]
"""

import argparse
import os
import urllib.request

REPO = "Supertone/supertonic"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"

SMALL_FILES = [
    "onnx/tts.json",
    "onnx/tts.yml",
    "onnx/unicode_indexer.json",
    "voice_styles/F1.json",
    "voice_styles/M1.json",
]

ONNX_FILES = [
    "onnx/duration_predictor.onnx",  # 1.5 MB
    "onnx/text_encoder.onnx",        # 27.3 MB
    "onnx/vocoder.onnx",             # unknown
    "onnx/vector_estimator.onnx",    # 132 MB — largest, download last
]


def download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  already exists: {dest}")
        return
    print(f"  downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"  saved {dest} ({size/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-onnx", action="store_true", help="Skip large ONNX files")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "models")

    print("=== Downloading small files ===")
    for path in SMALL_FILES:
        url = f"{BASE_URL}/{path}"
        dest = os.path.join(out_dir, path.replace("/", "_"))
        download(url, dest)

    if not args.no_onnx:
        print("\n=== Downloading ONNX models ===")
        for path in ONNX_FILES:
            url = f"{BASE_URL}/{path}"
            dest = os.path.join(out_dir, os.path.basename(path))
            download(url, dest)
    else:
        print("\nSkipping ONNX files (--no-onnx)")

    print("\nDone.")


if __name__ == "__main__":
    main()
