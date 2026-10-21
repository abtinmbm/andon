"""Post-training int8 quantization, with an integer reference that the C port must match.

The scheme is the usual one for Cortex-M inference (TFLite, CMSIS-NN):

- Batch norm is folded into the conv or dense layer before it.
- Weights are int8, symmetric, one scale per output channel.
- Activations are int8 with one scale and zero point per tensor, ``real = scale * (q - zero)``,
  calibrated from the min and max seen on normal training data. A ReLU output's range
  starts at 0, so its zero point is -128 and the ReLU is just the clamp to int8.
- Biases are int32 in the accumulator's scale. Accumulators are int32, and each output is
  rescaled by a fixed-point multiplier: ``q = ((acc * mult + 2**(shift-1)) >> shift) + zero``,
  round half up, computed in int64, then clamped to [-128, 127].

Everything up to the last layer's int32 accumulators is integer arithmetic, so the C
code can reproduce it bit for bit. The last layer's outputs and the anomaly score are
float (the RP2350's Cortex-M33 has a single-precision FPU), and are compared with a
tolerance.

The per-ID GMM is quantized differently: its precision Cholesky factors are stored as
int8 with one scale per column, and the score is computed in float from them. Its
compute is small, so only its size needs int8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.nn import functional as F

from andon.models.autoencoder import DenseAutoencoder
from andon.models.idcnn import IdCNN

INT8_MIN, INT8_MAX = -128, 127


@dataclass(frozen=True)
class QParams:
    """Quantization of one activation tensor: ``real = scale * (q - zero_point)``."""

    scale: float
    zero_point: int


def activation_qparams(lo: float, hi: float) -> QParams:
    """Asymmetric int8 parameters covering ``[lo, hi]``, which is widened to include 0.

    Including 0 makes real zero exactly representable, so zero padding and ReLU are exact.
    """
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    scale = (hi - lo) / (INT8_MAX - INT8_MIN)
    if scale == 0.0:
        scale = 1.0  # an all-zero tensor; any scale represents it
    zero = int(np.clip(round(INT8_MIN - lo / scale), INT8_MIN, INT8_MAX))
    return QParams(scale, zero)


def quantize(x: np.ndarray, qp: QParams) -> np.ndarray:
    q = np.round(np.asarray(x, dtype=np.float64) / qp.scale) + qp.zero_point
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def dequantize(q: np.ndarray, qp: QParams) -> np.ndarray:
    return qp.scale * (q.astype(np.float64) - qp.zero_point)


def quantize_weights(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 per output channel (axis 0); returns ``(q, scale)``, ``w ~ scale * q``."""
    flat = np.abs(w.reshape(len(w), -1)).max(axis=1)
    scale = np.where(flat > 0, flat / INT8_MAX, 1.0)
    shape = (-1,) + (1,) * (w.ndim - 1)
    q = np.clip(np.round(w / scale.reshape(shape)), -INT8_MAX, INT8_MAX).astype(np.int8)
    return q, scale


def quantize_multiplier(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Write each positive real ``m`` as ``mult * 2**-shift`` with ``mult`` in [2**30, 2**31).

    ``mult`` fits an int32 and keeps 31 bits of precision; ``shift`` is at least 1 so the
    rounding term ``2**(shift-1)`` exists.
    """
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    if (m <= 0).any():
        raise ValueError("multipliers must be positive")
    exp = np.floor(np.log2(m)).astype(np.int64)
    shift = 30 - exp
    mult = np.round(m * np.exp2(shift.astype(np.float64))).astype(np.int64)
    carry = mult >= 2**31  # rounding pushed it up to 2**31
    mult[carry] //= 2
    shift[carry] -= 1
    if (shift < 1).any() or (shift > 62).any():
        raise ValueError(f"multiplier out of range: {m}")
    return mult, shift


def requantize(acc: np.ndarray, mult: np.ndarray, shift: np.ndarray, zero_point: int) -> np.ndarray:
    """int32 accumulators to int8; ``mult`` and ``shift`` broadcast against ``acc``.

    ``>>`` on a negative int64 is an arithmetic shift (floor), as with GCC on Arm.
    """
    acc = acc.astype(np.int64)
    rounding = np.left_shift(np.int64(1), shift - 1)
    out = np.right_shift(acc * mult + rounding, shift) + zero_point
    return np.clip(out, INT8_MIN, INT8_MAX).astype(np.int8)


def conv2d_acc(x: np.ndarray, w: np.ndarray, stride: int, padding: int) -> np.ndarray:
    """Integer 2-D convolution: ``x`` (n, c, h, w) with the zero point already subtracted.

    Runs through torch in float64 for speed. Every product and partial sum is an integer
    far below 2**53, so float64 holds them exactly and the result is the integer sum.
    Padding is 0, which is real zero once the zero point is subtracted.
    """
    out = F.conv2d(
        torch.from_numpy(x.astype(np.float64)),
        torch.from_numpy(w.astype(np.float64)),
        stride=stride,
        padding=padding,
    )
    return out.numpy().round().astype(np.int64)


def linear_acc(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Integer dense layer ``x @ w.T`` for ``x`` (n, in) with the zero point subtracted.

    Done in float64 for BLAS speed; exact for the same reason as :func:`conv2d_acc`.
    """
    out = x.astype(np.float64) @ w.T.astype(np.float64)
    return out.round().astype(np.int64)


def maxpool2(q: np.ndarray) -> np.ndarray:
    """2x2 max pool with floor on odd sizes, as ``nn.MaxPool2d(2)``; exact on int8."""
    n, c, h, w = q.shape
    q = q[:, :, : h // 2 * 2, : w // 2 * 2]
    return q.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def fold_bn(
    weight: torch.Tensor, bias: torch.Tensor | None, bn: nn.BatchNorm1d | nn.BatchNorm2d
) -> tuple[np.ndarray, np.ndarray]:
    """Weights and bias of a conv or dense layer with its eval-mode batch norm folded in."""
    gamma = bn.weight.detach().double().numpy()
    beta = bn.bias.detach().double().numpy()
    mean = bn.running_mean.double().numpy()
    var = bn.running_var.double().numpy()
    k = gamma / np.sqrt(var + bn.eps)
    w = weight.detach().double().numpy()
    b = np.zeros(len(w)) if bias is None else bias.detach().double().numpy()
    return w * k.reshape((-1,) + (1,) * (w.ndim - 1)), (b - mean) * k + beta


@dataclass
class QLayer:
    """A conv or dense layer with batch norm and ReLU folded in, int8 in and int8 out."""

    weight: np.ndarray  # int8, (out, in) or (out, in, kh, kw)
    bias: np.ndarray  # int32, in the accumulator's scale
    mult: np.ndarray  # int32 (as int64), per output channel
    shift: np.ndarray  # per output channel
    out: QParams
    stride: int = 1  # conv only
    pool_before: bool = False  # conv only: a 2x2 max pool on this layer's input

    def nbytes(self) -> int:
        # int8 weights, int32 biases, an int32 multiplier and an int8 shift per channel
        return self.weight.size + 4 * self.bias.size + 5 * self.mult.size + 2 * 4


def _requant_layer(
    w_float: np.ndarray, b_float: np.ndarray, in_qp: QParams, out_qp: QParams, **kw
) -> QLayer:
    wq, ws = quantize_weights(w_float)
    acc_scale = in_qp.scale * ws
    bias = np.round(b_float / acc_scale).astype(np.int64)
    mult, shift = quantize_multiplier(acc_scale / out_qp.scale)
    return QLayer(wq, bias.astype(np.int32), mult, shift, out_qp, **kw)


def _per_channel(v: np.ndarray, ndim: int) -> np.ndarray:
    return v.reshape((1, -1) + (1,) * (ndim - 2))


@dataclass
class QuantIdCNN:
    """Integer version of :class:`IdCNN`; :meth:`logits` takes the same float patches."""

    band_mean: np.ndarray  # float32 (n_mels,)
    band_std: np.ndarray
    input_q: QParams
    convs: list[QLayer]
    head_weight: np.ndarray  # int8 (n_classes, channels)
    head_bias: np.ndarray  # int32
    head_scale: np.ndarray  # float, logits = head_scale * accumulator
    pool_size: int  # positions averaged by the global pool, fixed by the patch shape

    def quantize_input(self, patches: np.ndarray) -> np.ndarray:
        """Float log-mel patches ``(n, frames, n_mels)`` to int8 input ``(n, 1, n_mels, frames)``.

        Normalised per band in float32, as the device will do before quantizing.
        """
        x = (np.asarray(patches, dtype=np.float32) - self.band_mean) / self.band_std
        return quantize(x.transpose(0, 2, 1)[:, None], self.input_q)

    def head_acc(self, q_in: np.ndarray, trace: list[np.ndarray] | None = None) -> np.ndarray:
        """int8 input to the head's int32 accumulators ``(n, n_classes)``; integer only."""
        q, zero = q_in, self.input_q.zero_point
        for layer in self.convs:
            if layer.pool_before:
                q = maxpool2(q)
            acc = conv2d_acc(q.astype(np.int32) - zero, layer.weight, layer.stride, 1)
            acc += _per_channel(layer.bias.astype(np.int64), 4)
            mult, shift = _per_channel(layer.mult, 4), _per_channel(layer.shift, 4)
            q = requantize(acc, mult, shift, layer.out.zero_point)
            zero = layer.out.zero_point
            if trace is not None:
                trace.append(q)
        positions = q.shape[2] * q.shape[3]
        if positions != self.pool_size:
            raise ValueError(f"patch gives {positions} pooled positions, not {self.pool_size}")
        pooled = (q.astype(np.int64) - zero).sum(axis=(2, 3))  # the 1/HW is in head_scale
        return linear_acc(pooled, self.head_weight) + self.head_bias

    def logits(self, patches: np.ndarray) -> np.ndarray:
        return self.head_acc(self.quantize_input(patches)) * self.head_scale

    def nbytes(self) -> int:
        return (
            4 * (self.band_mean.size + self.band_std.size)
            + 2 * 4  # input scale and zero point
            + sum(layer.nbytes() for layer in self.convs)
            + self.head_weight.size
            + 4 * self.head_bias.size
            + 4 * self.head_scale.size
        )


def _range(t: torch.Tensor) -> tuple[float, float]:
    return float(t.min()), float(t.max())


@torch.no_grad()
def quantize_idcnn(model: IdCNN, calib: torch.Tensor) -> QuantIdCNN:
    """Quantize a trained :class:`IdCNN`, calibrating activations on ``calib`` patches.

    ``calib`` is ``(n, frames, n_mels)`` raw log-mel patches from normal training clips.
    The ranges are taken from the float model, layer by layer.
    """
    model = model.cpu().eval()
    mods = list(model.features)
    x = model.normalize(calib.float()).transpose(1, 2).unsqueeze(1)
    in_qp = activation_qparams(*_range(x))
    input_q, convs, pool_next = in_qp, [], False
    for i, m in enumerate(mods):
        if isinstance(m, nn.MaxPool2d):
            x, pool_next = m(x), True
            continue
        if not isinstance(m, nn.Conv2d):
            continue
        bn, relu = mods[i + 1], mods[i + 2]
        if not (isinstance(bn, nn.BatchNorm2d) and isinstance(relu, nn.ReLU)):
            raise ValueError("expected Conv2d -> BatchNorm2d -> ReLU")
        x = relu(bn(m(x)))
        out_qp = activation_qparams(*_range(x))
        w, b = fold_bn(m.weight, m.bias, bn)
        stride = int(m.stride[0])
        convs.append(_requant_layer(w, b, in_qp, out_qp, stride=stride, pool_before=pool_next))
        in_qp, pool_next = out_qp, False
    pool_size = int(x.shape[2] * x.shape[3])
    wq, ws = quantize_weights(model.head.weight.detach().double().numpy())
    head_scale = in_qp.scale * ws / pool_size
    head_bias = np.round(model.head.bias.detach().double().numpy() / head_scale)
    return QuantIdCNN(
        band_mean=model.band_mean.numpy().astype(np.float32),
        band_std=model.band_std.numpy().astype(np.float32),
        input_q=input_q,
        convs=convs,
        head_weight=wq,
        head_bias=head_bias.astype(np.int32),
        head_scale=head_scale,
        pool_size=pool_size,
    )


@dataclass
class QuantAutoencoder:
    """Integer version of :class:`DenseAutoencoder`; the output layer's result is float."""

    input_q: QParams
    hidden: list[QLayer]
    out_weight: np.ndarray  # int8 (input_dim, hidden)
    out_bias: np.ndarray  # int32
    out_scale: np.ndarray  # float, reconstruction = out_scale * accumulator

    def out_acc(self, x: np.ndarray) -> np.ndarray:
        """Float ``(n, input_dim)`` inputs to the output layer's int32 accumulators."""
        q, zero = quantize(x, self.input_q), self.input_q.zero_point
        for layer in self.hidden:
            acc = linear_acc(q.astype(np.int64) - zero, layer.weight) + layer.bias
            q = requantize(acc, layer.mult[None], layer.shift[None], layer.out.zero_point)
            zero = layer.out.zero_point
        return linear_acc(q.astype(np.int64) - zero, self.out_weight) + self.out_bias

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.out_acc(x) * self.out_scale

    def errors(self, x: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error per input vector, as :func:`frame_errors`."""
        return ((np.asarray(x, dtype=np.float64) - self.reconstruct(x)) ** 2).mean(axis=1)

    def nbytes(self) -> int:
        return (
            2 * 4
            + sum(layer.nbytes() for layer in self.hidden)
            + self.out_weight.size
            + 4 * self.out_bias.size
            + 4 * self.out_scale.size
        )


@torch.no_grad()
def quantize_autoencoder(model: DenseAutoencoder, calib: torch.Tensor) -> QuantAutoencoder:
    """Quantize a trained autoencoder, calibrating on ``calib`` stacked-frame vectors."""
    model = model.cpu().eval()
    mods = list(model.net)
    x = calib.float()
    in_qp = activation_qparams(*_range(x))
    input_q, hidden = in_qp, []
    linears = [i for i, m in enumerate(mods) if isinstance(m, nn.Linear)]
    for i in linears[:-1]:
        lin, bn, relu = mods[i], mods[i + 1], mods[i + 2]
        if not (isinstance(bn, nn.BatchNorm1d) and isinstance(relu, nn.ReLU)):
            raise ValueError("expected Linear -> BatchNorm1d -> ReLU")
        x = relu(bn(lin(x)))
        out_qp = activation_qparams(*_range(x))
        w, b = fold_bn(lin.weight, lin.bias, bn)
        hidden.append(_requant_layer(w, b, in_qp, out_qp))
        in_qp = out_qp
    last = mods[linears[-1]]
    wq, ws = quantize_weights(last.weight.detach().double().numpy())
    out_scale = in_qp.scale * ws
    out_bias = np.round(last.bias.detach().double().numpy() / out_scale)
    return QuantAutoencoder(input_q, hidden, wq, out_bias.astype(np.int32), out_scale)


@dataclass
class QuantGMM:
    """A full-covariance GMM with int8 precision Cholesky factors; scored in float.

    ``-log p(x)`` with ``log N_k(x) = log det L_k - |(x - mu_k) L_k|^2 / 2 - D log(2 pi) / 2``,
    where ``L_k`` is the upper-triangular Cholesky factor of component ``k``'s precision.
    Only the upper triangle is stored. Means, log weights and log determinants stay float32.
    """

    means: np.ndarray  # float32 (K, D)
    chol_q: np.ndarray  # int8 (K, D, D), upper triangular
    chol_scale: np.ndarray  # float32 (K, D), one scale per column
    log_weights: np.ndarray  # float32 (K,)
    log_det: np.ndarray  # float32 (K,), from the dequantized factors

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        chol = self.chol_q.astype(np.float64) * self.chol_scale[:, None, :]
        d = x.shape[1]
        log_n = np.stack(
            [
                self.log_det[k] - 0.5 * (((x - self.means[k]) @ chol[k]) ** 2).sum(axis=1)
                for k in range(len(self.means))
            ],
            axis=1,
        )
        log_n += self.log_weights - 0.5 * d * np.log(2 * np.pi)
        top = log_n.max(axis=1, keepdims=True)
        return -(top[:, 0] + np.log(np.exp(log_n - top).sum(axis=1)))

    def nbytes(self) -> int:
        k, d = self.means.shape
        triangle = d * (d + 1) // 2
        # int8 triangle; float32 mean, column scales, log weight and log determinant
        return k * (triangle + 4 * (d + d + 1 + 1))


def quantize_gmm(gm: GaussianMixture) -> QuantGMM:
    """int8 precision factors of a fitted full-covariance ``GaussianMixture``."""
    if gm.covariance_type != "full":
        raise ValueError("only full-covariance GMMs are supported")
    chol = gm.precisions_cholesky_  # (K, D, D), upper triangular
    # quantize_weights scales axis 0, so put each column first.
    cols = chol.transpose(0, 2, 1).reshape(-1, chol.shape[1])
    q, scale = quantize_weights(cols)
    k, d = chol.shape[:2]
    chol_q = q.reshape(k, d, d).transpose(0, 2, 1)
    chol_scale = scale.reshape(k, d)
    deq = chol_q.astype(np.float64) * chol_scale[:, None, :]
    log_det = np.log(np.abs(np.diagonal(deq, axis1=1, axis2=2))).sum(axis=1)
    return QuantGMM(
        means=gm.means_.astype(np.float32),
        chol_q=np.ascontiguousarray(chol_q),
        chol_scale=chol_scale.astype(np.float32),
        log_weights=np.log(gm.weights_).astype(np.float32),
        log_det=log_det.astype(np.float32),
    )
