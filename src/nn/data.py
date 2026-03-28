from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Loads mono 44.1 kHz wav files, returns fixed-length segments."""

    def __init__(self, data_dir: str, segment_length: int, sample_rate: int = 44100):
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.files = sorted(Path(data_dir).rglob("*.wav"))
        if not self.files:
            raise ValueError(f"No .wav files found under {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav, sr = torchaudio.load(self.files[idx])
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)  # (T,)

        T = wav.shape[0]
        if T >= self.segment_length:
            start = torch.randint(0, T - self.segment_length + 1, (1,)).item()
            wav = wav[start : start + self.segment_length]
        else:
            wav = F.pad(wav, (0, self.segment_length - T))
        return wav  # (segment_length,)
