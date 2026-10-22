"""Reading and writing mono 16-bit PCM WAV files with the standard library.

The DCASE 2020 Task 2 clips are 16 kHz, mono, 16-bit PCM, so the stdlib ``wave``
module is enough and saves a native dependency.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import BinaryIO

import numpy as np

_INT16_SCALE = 32768.0  # same scaling as soundfile/librosa, so values land in [-1, 1)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Return ``(samples, sample_rate)``, samples as float32 in [-1, 1).

    Multi-channel files are reduced to their first channel, the way the baseline
    used channel 0 of the original 8-channel MIMII recordings.
    """
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {8 * wf.getsampwidth()}-bit")
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, n_channels)[:, 0]
    return samples.astype(np.float32) / _INT16_SCALE, sample_rate


def write_wav(path: str | Path | BinaryIO, samples: np.ndarray, sample_rate: int) -> None:
    """Write float samples in [-1, 1] as mono 16-bit PCM (clipping anything outside).

    ``path`` may also be an open binary file, such as a ``BytesIO`` for the dashboard.
    """
    pcm = np.clip(np.round(samples * _INT16_SCALE), -32768, 32767).astype("<i2")
    with wave.open(str(path) if isinstance(path, (str, Path)) else path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
