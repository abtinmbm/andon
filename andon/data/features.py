"""Log-mel features for whole splits, cached on disk, and the baseline's frame stacking.

The baseline turned every clip into a ``(n_windows, frames * n_mels)`` array up front.
For one machine type that is about 3 GB of float32, so instead we keep only the
log-mel frames (5x smaller) and stack windows when a batch is built. The stacked
vectors are identical to the baseline's: block ``t`` of a vector is frame ``t`` of
its window.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from andon.data.audio import read_wav
from andon.data.dataset import Clip
from andon.features.logmel import LogMelConfig, logmel


@dataclass(frozen=True)
class SplitFeatures:
    """Log-mel frames of many clips, concatenated in clip order.

    Clip ``i`` owns ``frames[offsets[i]:offsets[i + 1]]``.
    """

    frames: np.ndarray  # (total_frames, n_mels), float32
    offsets: np.ndarray  # (n_clips + 1,), int64

    @property
    def n_clips(self) -> int:
        return self.offsets.size - 1

    def clip(self, i: int) -> np.ndarray:
        return self.frames[self.offsets[i] : self.offsets[i + 1]]


def clip_logmel(path: Path, cfg: LogMelConfig) -> np.ndarray:
    """Log-mel frames of one clip, shape ``(n_frames, n_mels)``."""
    samples, sr = read_wav(path)
    # The mel filterbank is built for cfg.sample_rate; a clip at another rate would be
    # silently mis-binned, so refuse it.
    if sr != cfg.sample_rate:
        raise ValueError(f"{path}: sample rate {sr} Hz, expected {cfg.sample_rate} Hz")
    return logmel(samples, cfg)


def extract(clips: Sequence[Clip], cfg: LogMelConfig) -> SplitFeatures:
    per_clip = [clip_logmel(c.path, cfg) for c in clips]
    lengths = np.array([f.shape[0] for f in per_clip], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    frames = np.concatenate(per_clip) if per_clip else np.zeros((0, cfg.n_mels), np.float32)
    return SplitFeatures(frames=frames, offsets=offsets)


def cache_key(clips: Sequence[Clip], cfg: LogMelConfig) -> str:
    """Changes whenever the clip list or any front-end setting changes."""
    payload = json.dumps(
        {"cfg": asdict(cfg), "clips": [f"{c.split}/{c.path.name}" for c in clips]},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def load_or_extract(
    clips: Sequence[Clip], cfg: LogMelConfig, cache_dir: Path, name: str
) -> SplitFeatures:
    """Features for ``clips``, from ``cache_dir/<name>_<key>.npz`` if present, else extracted."""
    path = Path(cache_dir) / f"{name}_{cache_key(clips, cfg)}.npz"
    if path.exists():
        with np.load(path) as data:
            return SplitFeatures(frames=data["frames"], offsets=data["offsets"])
    feats = extract(clips, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-process temp name: seeds trained in parallel may extract the same split at once.
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, frames=feats.frames, offsets=feats.offsets)
    try:
        tmp.replace(path)  # atomic: an interrupted run never leaves a half-written cache
    except PermissionError:
        # Windows: a parallel run wrote the same cache first and has it open. Its copy is identical.
        tmp.unlink(missing_ok=True)
    return feats


def clip_means(feats: SplitFeatures) -> np.ndarray:
    """One vector per clip: the mean of its log-mel frames, shape ``(n_clips, n_mels)``.

    Accumulated in float64; a clip has about 300 frames of values in the tens of dB.
    """
    lengths = np.diff(feats.offsets)
    if (lengths < 1).any():
        raise ValueError("a clip has no frames")
    sums = np.add.reduceat(feats.frames.astype(np.float64), feats.offsets[:-1], axis=0)
    return sums / lengths[:, None]


def window_starts(offsets: np.ndarray, n_context: int) -> np.ndarray:
    """First-frame index of every ``n_context``-frame window that stays inside one clip.

    A clip of ``n`` frames gives ``n - n_context + 1`` windows, as in the baseline.
    """
    starts = [
        np.arange(lo, hi - n_context + 1, dtype=np.int64)
        for lo, hi in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)


def stack_frames(frames: np.ndarray, starts: np.ndarray, n_context: int) -> np.ndarray:
    """Windows of ``n_context`` frames flattened to ``(len(starts), n_context * n_mels)``."""
    idx = starts[:, None] + np.arange(n_context)
    return frames[idx].reshape(len(starts), -1)


def gather_windows(frames: torch.Tensor, starts: torch.Tensor, n_context: int) -> torch.Tensor:
    """Torch twin of :func:`stack_frames`, used to build batches on the training device."""
    idx = starts[:, None] + torch.arange(n_context, device=starts.device)
    return frames[idx].reshape(len(starts), -1)
