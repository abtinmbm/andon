"""Log-mel spectrogram features, written in plain NumPy.

This matches the DCASE 2020 Task 2 baseline front end, which used
``librosa.feature.melspectrogram`` with its defaults (periodic Hann window,
centered frames with zero padding, Slaney mel scale and normalization) and then
``10 * log10(mel + eps)``. It is written out by hand instead of calling librosa
because the same steps get ported to C for the Pico in Stage 2, and the tests
cross-check this file against librosa.

Output is time-major, shape ``(n_frames, n_mels)``, because that is the order
frames arrive in when streaming.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# Same epsilon as the baseline (sys.float_info.epsilon), so log values match.
EPS = sys.float_info.epsilon


@dataclass(frozen=True)
class LogMelConfig:
    """Front-end settings. Defaults are the DCASE 2020 baseline values."""

    sample_rate: int = 16_000
    n_fft: int = 1024  # 64 ms frames at 16 kHz
    hop_length: int = 512  # a new frame every 32 ms
    n_mels: int = 128
    power: float = 2.0
    fmin: float = 0.0
    fmax: float | None = None  # None means sample_rate / 2


# --- Slaney mel scale: linear below 1 kHz, logarithmic above. ---
_F_SP = 200.0 / 3  # Hz per mel in the linear part
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP  # = 15 mel
_LOG_STEP = np.log(6.4) / 27.0


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    mel = hz / _F_SP
    log_part = hz >= _MIN_LOG_HZ
    return np.where(
        log_part,
        _MIN_LOG_MEL + np.log(np.maximum(hz, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOG_STEP,
        mel,
    )


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    hz = mel * _F_SP
    log_part = mel >= _MIN_LOG_MEL
    return np.where(log_part, _MIN_LOG_HZ * np.exp(_LOG_STEP * (mel - _MIN_LOG_MEL)), hz)


@lru_cache(maxsize=8)
def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Triangular mel filters, shape ``(n_mels, n_fft // 2 + 1)``, Slaney-normalized.

    Each filter's area is normalized (2 / bandwidth) so that wide high-frequency
    bands don't dominate just because they cover more FFT bins.
    """
    fft_freqs = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    mel_edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    widths = np.diff(mel_edges)
    ramps = mel_edges[:, None] - fft_freqs[None, :]

    weights = np.zeros((n_mels, fft_freqs.size), dtype=np.float64)
    for i in range(n_mels):
        rising = -ramps[i] / widths[i]
        falling = ramps[i + 2] / widths[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(rising, falling))

    weights *= (2.0 / (mel_edges[2 : n_mels + 2] - mel_edges[:n_mels]))[:, None]
    weights = weights.astype(np.float32)
    weights.setflags(write=False)  # cached and shared, so make it read-only
    return weights


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window (the STFT convention, as used by librosa/scipy)."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def power_spectrogram(y: np.ndarray, cfg: LogMelConfig) -> np.ndarray:
    """|STFT|^power with centered, zero-padded frames. Shape ``(n_frames, n_fft // 2 + 1)``."""
    if y.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {y.shape}")
    pad = cfg.n_fft // 2
    padded = np.pad(y.astype(np.float32, copy=False), (pad, pad), mode="constant")
    if padded.size < cfg.n_fft:
        raise ValueError("audio is too short for one frame")

    n_frames = 1 + (padded.size - cfg.n_fft) // cfg.hop_length
    frames = np.lib.stride_tricks.sliding_window_view(padded, cfg.n_fft)[:: cfg.hop_length]
    frames = frames[:n_frames] * hann_window(cfg.n_fft)
    return np.abs(np.fft.rfft(frames, axis=1)) ** cfg.power


def logmel(y: np.ndarray, cfg: LogMelConfig | None = None) -> np.ndarray:
    """Log-mel spectrogram in dB, shape ``(n_frames, n_mels)``, dtype float32."""
    cfg = cfg or LogMelConfig()
    fmax = cfg.fmax if cfg.fmax is not None else cfg.sample_rate / 2
    fb = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, fmax)
    mel = power_spectrogram(y, cfg) @ fb.T
    # 20 / power * log10 gives dB for both power (power=2) and magnitude (power=1) spectra.
    return ((20.0 / cfg.power) * np.log10(mel + EPS)).astype(np.float32)
