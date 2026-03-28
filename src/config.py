import argparse

from . import constants as C


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SupertonicTTS speech autoencoder")
    parser.add_argument("--data_dir", required=True, help="Directory of .wav files")
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--max_steps", type=int, default=1_500_000)
    parser.add_argument("--segment_length", type=int, default=int(0.19 * C.SAMPLE_RATE),
                        help="Audio samples per training segment (default: 0.19 s per paper)")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--keep_ckpts", type=int, default=3)
    return parser.parse_args()
