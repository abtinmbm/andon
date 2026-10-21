"""Size and compute estimates against the microcontroller budget (PLAN.md, Phase 4).

Budget, for a Raspberry Pi Pico 2 (RP2350, 150 MHz Cortex-M33, 520 KB SRAM):

- at most 256 KB of model data in int8
- under 20 ms of compute per new audio frame; a frame arrives every 32 ms (hop 512 at 16 kHz)

Compute is estimated from multiply-accumulate (MAC) counts, converted to time with an
assumed 0.5 MAC per cycle. That is a deliberately modest figure for int8 kernels on the
M33's DSP extension, which can do two 16-bit MACs per instruction but also spends cycles
on loads, im2col and requantization. It is an estimate until Stage 2 measures it.

A model that scores a window every ``frames_per_decision`` frames has its window's cost
spread over those frames (``ms_per_frame``). The window cost in one go is reported too
(``ms_per_decision``); on the device it can run on the second core while the first keeps
up with the audio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

CLOCK_HZ = 150e6
MACS_PER_CYCLE = 0.5  # assumption, see the module docstring
BUDGET_BYTES = 256 * 1024
BUDGET_MS_PER_FRAME = 20.0


def module_macs(model: nn.Module, example: torch.Tensor) -> int:
    """MACs of one forward pass of ``example`` (batch 1) through the conv and dense layers.

    Batch norm, ReLU and pooling are not counted: batch norm is folded away on the device,
    and the rest are a compare or an add per value, small next to the MACs.
    """
    if len(example) != 1:
        raise ValueError("pass a batch of one")
    total = 0

    def hook(m: nn.Module, inputs: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            total += out.numel() * (m.in_channels // m.groups) * kh * kw
        elif isinstance(m, nn.Linear):
            total += out.numel() * m.in_features

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, nn.Conv2d | nn.Linear)
    ]
    try:
        with torch.no_grad():
            model.eval()(example)
    finally:
        for h in handles:
            h.remove()
    return total


def front_end_macs(n_fft: int, n_mels: int) -> int:
    """Rough MACs to turn one frame of audio into a log-mel frame.

    A real FFT of ``n_fft`` points done as a complex FFT of half the size: 4 real MACs per
    radix-2 butterfly. Plus the window, the power spectrum, and the mel filterbank, where
    each FFT bin falls in at most two triangular filters. Logs are counted as one MAC per band.
    """
    half = n_fft // 2
    fft = 4 * (half // 2) * int(math.log2(half)) + 2 * n_fft  # + the split into real bins
    bins = half + 1
    return n_fft + fft + 2 * bins + 2 * bins + n_mels


def ms_for(macs: float) -> float:
    return macs / MACS_PER_CYCLE / CLOCK_HZ * 1000


@dataclass(frozen=True)
class Budget:
    params: int  # weights and biases after batch norm folding
    int8_bytes: int  # what the device stores for the quantized model
    # One score: a window for the CNN, a frame for the AE, a clip for the GMM.
    macs_per_decision: int
    frames_per_decision: int
    front_end_macs_per_frame: int
    per_frame_macs: int = 0  # model work done on every frame regardless (the GMM's running mean)

    @property
    def ms_per_decision(self) -> float:
        return ms_for(self.macs_per_decision)

    @property
    def ms_per_frame(self) -> float:
        per_frame = (
            self.front_end_macs_per_frame
            + self.per_frame_macs
            + self.macs_per_decision / self.frames_per_decision
        )
        return ms_for(per_frame)

    @property
    def fits(self) -> bool:
        return self.int8_bytes <= BUDGET_BYTES and self.ms_per_frame < BUDGET_MS_PER_FRAME

    def notes(self) -> str:
        """The fields as ``key=value`` pairs for the notes column of the results log."""
        return (
            f"params={self.params} int8_bytes={self.int8_bytes} "
            f"macs_per_decision={self.macs_per_decision} "
            f"frames_per_decision={self.frames_per_decision} "
            f"est_ms_per_frame={self.ms_per_frame:.2f} "
            f"est_ms_per_decision={self.ms_per_decision:.2f}"
        )


def folded_params(model: nn.Module) -> int:
    """Weights and biases of the conv and dense layers once batch norm is folded in.

    Folding gives every layer followed by batch norm a bias, even if it had none.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.numel() + m.out_channels
        elif isinstance(m, nn.Linear):
            total += m.weight.numel() + m.out_features
    return total


def gmm_macs(n_components: int, dim: int) -> int:
    """MACs for one GMM score: per component, ``x - mu``, a triangular matrix-vector
    product, and the squared norm."""
    return n_components * (dim + dim * (dim + 1) // 2 + dim)
