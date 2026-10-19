"""A small CNN that tells the machine IDs of one machine type apart, from normal clips only.

This is the idea behind the top DCASE 2020 Task 2 systems: a model trained to recognise
which machine (ID) made a sound learns what each machine normally sounds like. On an
anomalous clip it becomes unsure of the correct ID, so the anomaly score is
``-log p(true ID)``, averaged over windows of the clip.

Layout (sized to train on a laptop CPU; about 98k weights for 128 mel bands)::

    input (1, n_mels, frames), normalised per mel band with training statistics
    Conv3x3 stride 2 -> BN -> ReLU -> MaxPool2
    Conv3x3 -> BN -> ReLU -> MaxPool2      (x2, widening)
    Conv3x3 -> BN -> ReLU
    global average pool -> Linear(n_ids)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv(n_in: int, n_out: int, stride: int = 1) -> list[nn.Module]:
    return [
        nn.Conv2d(n_in, n_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(n_out),
        nn.ReLU(),
    ]


class IdCNN(nn.Module):
    """Classifies a ``(batch, frames, n_mels)`` log-mel patch into one of ``n_classes`` IDs."""

    def __init__(self, n_classes: int, n_mels: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("need at least two conv stages")
        layers = _conv(1, channels[0], stride=2)
        for n_in, n_out in zip(channels[:-1], channels[1:], strict=True):
            layers += [nn.MaxPool2d(2), *_conv(n_in, n_out)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], n_classes)
        # Per-band statistics of the training frames, saved with the weights so a
        # checkpoint scores new audio exactly as it was trained (and a C port can read them).
        self.register_buffer("band_mean", torch.zeros(n_mels))
        self.register_buffer("band_std", torch.ones(n_mels))

    def set_normalization(self, frames: torch.Tensor) -> None:
        """Fit the per-band mean and std on ``(n_frames, n_mels)`` training frames."""
        self.band_mean.copy_(frames.mean(dim=0))
        self.band_std.copy_(frames.std(dim=0).clamp_min(1e-3))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.band_mean) / self.band_std

    def classify(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Logits for already-normalised patches ``(batch, frames, n_mels)``."""
        # Mel bands on the height axis, time on the width axis.
        h = self.features(x_norm.transpose(1, 2).unsqueeze(1))
        return self.head(h.mean(dim=(2, 3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.normalize(x))


def id_scores(logits: torch.Tensor, true_ids: torch.Tensor) -> torch.Tensor:
    """Anomaly score per patch: ``-log p(true ID)``; higher means less like that machine."""
    return -F.log_softmax(logits, dim=1).gather(1, true_ids[:, None]).squeeze(1)


def mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Blend each patch with another from the batch; returns ``x_mixed, y_a, y_b, lam``.

    The loss is then ``lam * CE(y_a) + (1 - lam) * CE(y_b)``, with ``lam ~ Beta(alpha,
    alpha)``. With ``alpha <= 0`` the batch is returned unchanged (``lam = 1``).
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(len(x), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def mask_spectrogram(
    x: torch.Tensor,
    n_freq_masks: int,
    max_freq_width: int,
    n_time_masks: int,
    max_time_width: int,
) -> torch.Tensor:
    """SpecAugment-style masking of normalised ``(batch, frames, n_mels)`` patches.

    Each patch gets its own random mel bands and time spans set to 0, which is the
    training mean after normalisation. Widths are drawn uniformly from 0 to the maximum.
    """
    batch, n_frames, n_mels = x.shape
    band_hit = _random_spans(batch, n_mels, n_freq_masks, max_freq_width, x.device)
    time_hit = _random_spans(batch, n_frames, n_time_masks, max_time_width, x.device)
    keep = ~(band_hit[:, None, :] | time_hit[:, :, None])
    return x * keep


def _random_spans(
    batch: int, length: int, n_spans: int, max_width: int, device: torch.device
) -> torch.Tensor:
    """``(batch, length)`` boolean, True inside ``n_spans`` random spans per row."""
    hit = torch.zeros(batch, length, dtype=torch.bool, device=device)
    pos = torch.arange(length, device=device)
    for _ in range(n_spans):
        width = torch.randint(0, max_width + 1, (batch, 1), device=device)
        start = (torch.rand(batch, 1, device=device) * (length - width + 1)).long()
        hit |= (pos >= start) & (pos < start + width)
    return hit
