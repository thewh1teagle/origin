"""
Inspect unicode_indexer.json — reveals vocab size and character mapping.
Usage: uv run inspect_unicode.py
"""

import json
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def main():
    path = os.path.join(MODELS_DIR, "onnx_unicode_indexer.json")
    if not os.path.exists(path):
        print(f"Not found: {path} — run download.py first")
        return

    with open(path) as f:
        indexer = json.load(f)

    # indexer maps unicode codepoint (as string int) to token id
    print(f"Vocab size: {len(indexer)} entries")

    # show range of codepoints
    codepoints = [int(k) for k in indexer.keys()]
    print(f"Codepoint range: {min(codepoints)} – {max(codepoints)}")

    # show some examples
    print(f"\nSample entries (codepoint → index):")
    for cp, idx in sorted(indexer.items(), key=lambda x: int(x[0]))[:30]:
        char = chr(int(cp))
        printable = repr(char) if not char.isprintable() else char
        print(f"  U+{int(cp):04X} {printable!s:6s} → {idx}")

    # check for special tokens
    print(f"\nChecking for special/high codepoints:")
    high = [(int(k), v) for k, v in indexer.items() if int(k) > 127]
    print(f"  Non-ASCII entries: {len(high)}")
    if high:
        for cp, idx in sorted(high)[:10]:
            print(f"  U+{cp:04X} {chr(cp)!r} → {idx}")


if __name__ == "__main__":
    main()
