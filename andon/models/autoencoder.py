"""The DCASE 2020 Task 2 baseline autoencoder, ported from Keras to PyTorch.

Original (``keras_model.py``)::

    Input(640)
    4 x [Dense(128), BatchNormalization, ReLU]
    Dense(8), BatchNormalization, ReLU
    4 x [Dense(128), BatchNormalization, ReLU]
    Dense(640)

PyTorch's defaults differ from Keras's in ways that could move the result, so the
Keras ones are reproduced: Glorot-uniform weights with zero biases, and batch norm
with momentum 0.99 (``momentum=0.01`` in PyTorch's convention) and epsilon 1e-3.
"""

from __future__ import annotations

import torch
from torch import nn

# Keras BatchNormalization defaults, in PyTorch terms.
KERAS_BN_MOMENTUM = 0.01  # PyTorch weights the new batch; Keras's 0.99 weights the old average
KERAS_BN_EPS = 1e-3


def _block(n_in: int, n_out: int, bn_momentum: float, bn_eps: float) -> list[nn.Module]:
    return [
        nn.Linear(n_in, n_out),
        nn.BatchNorm1d(n_out, eps=bn_eps, momentum=bn_momentum),
        nn.ReLU(),
    ]


class DenseAutoencoder(nn.Module):
    """Reconstructs stacked log-mel frames; the reconstruction error is the anomaly score."""

    def __init__(
        self,
        input_dim: int = 640,
        hidden: int = 128,
        n_hidden: int = 4,
        bottleneck: int = 8,
        bn_momentum: float = KERAS_BN_MOMENTUM,
        bn_eps: float = KERAS_BN_EPS,
    ) -> None:
        super().__init__()
        enc_sizes = [input_dim] + [hidden] * n_hidden
        layers: list[nn.Module] = []
        for n_in, n_out in zip(enc_sizes[:-1], enc_sizes[1:], strict=True):
            layers += _block(n_in, n_out, bn_momentum, bn_eps)
        layers += _block(hidden, bottleneck, bn_momentum, bn_eps)
        dec_sizes = [bottleneck] + [hidden] * n_hidden
        for n_in, n_out in zip(dec_sizes[:-1], dec_sizes[1:], strict=True):
            layers += _block(n_in, n_out, bn_momentum, bn_eps)
        layers.append(nn.Linear(hidden, input_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Keras Dense defaults: Glorot-uniform kernel, zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def frame_errors(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Mean squared reconstruction error of each input vector, shape ``(n,)``."""
    return ((x - model(x)) ** 2).mean(dim=1)
