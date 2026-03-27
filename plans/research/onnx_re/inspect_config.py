"""
Parse and pretty-print tts.json and tts.yml from downloaded models.
Usage: uv run inspect_config.py
"""

import json
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_json(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        print(f"Not found: {path} — run download.py first")
        return None
    with open(path) as f:
        return json.load(f)


def load_yml(name):
    try:
        import yaml
    except ImportError:
        print("Install pyyaml: uv pip install pyyaml")
        sys.exit(1)
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        print(f"Not found: {path} — run download.py first")
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("tts.json")
    print("=" * 60)
    cfg = load_json("onnx_tts.json")
    if cfg:
        print(json.dumps(cfg, indent=2))

    print("\n" + "=" * 60)
    print("tts.yml")
    print("=" * 60)
    yml = load_yml("onnx_tts.yml")
    if yml:
        import pprint
        pprint.pprint(yml)


if __name__ == "__main__":
    main()
