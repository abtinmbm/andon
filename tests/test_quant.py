"""Phase 4: int8 quantization and the integer reference the C port will be checked against."""

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.mixture import GaussianMixture
from torch import nn

from andon import quant, train, train_idcnn
from andon.experiment import load_config
from andon.models.autoencoder import DenseAutoencoder
from andon.models.idcnn import IdCNN, id_scores

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _randomize_bn(model: nn.Module, seed: int) -> None:
    """Give every batch norm non-trivial running statistics and affine parameters."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.5)
            m.running_var.copy_(torch.rand(n, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(n, generator=g) * 0.2)


def tiny_cnn(n_classes: int = 3, n_mels: int = 16) -> IdCNN:
    torch.manual_seed(0)
    model = IdCNN(n_classes, n_mels, (4, 8, 8))
    _randomize_bn(model, 1)
    model.band_mean.copy_(torch.linspace(-60, -20, n_mels))
    model.band_std.copy_(torch.linspace(3, 8, n_mels))
    return model.eval()


def log_mel_like(n: int, frames: int, n_mels: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.linspace(-60, -20, n_mels) + torch.randn(n, frames, n_mels, generator=g) * 6


# --- building blocks --------------------------------------------------------------------


def test_activation_qparams_cover_the_range_and_zero():
    qp = quant.activation_qparams(-1.0, 3.0)
    assert quant.quantize(np.array([0.0]), qp)[0] == qp.zero_point  # real zero is exact
    x = np.linspace(-1, 3, 101)
    assert np.abs(quant.dequantize(quant.quantize(x, qp), qp) - x).max() <= qp.scale / 2 + 1e-12
    relu = quant.activation_qparams(0.0, 5.0)
    assert relu.zero_point == quant.INT8_MIN  # so the clamp to int8 is the ReLU


def test_quantize_weights_is_symmetric_per_channel():
    w = np.array([[0.5, -1.0], [0.01, 0.02], [0.0, 0.0]])
    q, scale = quant.quantize_weights(w)
    assert q.dtype == np.int8
    assert q[0].tolist() == [64, -127] and q[1].tolist() == [64, 127]
    assert q[2].tolist() == [0, 0] and scale[2] == 1.0  # an all-zero channel stays usable


def test_quantize_multiplier_keeps_31_bits():
    m = np.exp(np.random.default_rng(0).uniform(np.log(1e-6), np.log(10), 1000))
    mult, shift = quant.quantize_multiplier(m)
    assert ((mult >= 2**30) & (mult < 2**31)).all()
    assert (shift >= 1).all()
    assert (np.abs(mult * np.exp2(-shift.astype(float)) - m) / m).max() < 2**-30
    with pytest.raises(ValueError):
        quant.quantize_multiplier(np.array([0.0]))


def test_requantize_rounds_half_up_and_clamps():
    mult, shift = quant.quantize_multiplier(np.array([0.75]))  # exactly representable
    acc = np.array([2, -2, 3, -3, 10, 1000, -1000])
    out = quant.requantize(acc, mult, shift, zero_point=0)
    # 1.5 -> 2, -1.5 -> -1, 2.25 -> 2, -2.25 -> -2, 7.5 -> 8, then the int8 clamp
    assert out.tolist() == [2, -1, 2, -2, 8, 127, -128]
    assert quant.requantize(np.array([0]), mult, shift, zero_point=-128)[0] == -128


def test_conv2d_acc_matches_a_plain_integer_loop():
    rng = np.random.default_rng(0)
    x = rng.integers(-255, 256, size=(1, 2, 5, 6))
    w = rng.integers(-127, 128, size=(3, 2, 3, 3)).astype(np.int8)
    for stride in (1, 2):
        got = quant.conv2d_acc(x, w, stride=stride, padding=1)
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        h_out, w_out = got.shape[2:]
        want = np.zeros_like(got)
        for o in range(3):
            for i in range(h_out):
                for j in range(w_out):
                    patch = xp[0, :, i * stride : i * stride + 3, j * stride : j * stride + 3]
                    want[0, o, i, j] = int((patch.astype(np.int64) * w[o]).sum())
        assert got.dtype == np.int64
        np.testing.assert_array_equal(got, want)


def test_maxpool2_matches_torch():
    q = np.random.default_rng(0).integers(-128, 128, size=(2, 3, 5, 7)).astype(np.int8)
    want = nn.MaxPool2d(2)(torch.from_numpy(q.astype(np.float32))).numpy()
    np.testing.assert_array_equal(quant.maxpool2(q), want.astype(np.int8))


def test_fold_bn_matches_the_float_layers():
    torch.manual_seed(0)
    conv, bn = nn.Conv2d(2, 3, 3, padding=1, bias=False), nn.BatchNorm2d(3)
    _randomize_bn(bn, 0)
    bn.eval()
    x = torch.randn(4, 2, 6, 5)
    w, b = quant.fold_bn(conv.weight, conv.bias, bn)
    folded = nn.functional.conv2d(x.double(), torch.from_numpy(w), torch.from_numpy(b), padding=1)
    torch.testing.assert_close(folded.float(), bn(conv(x)), atol=1e-5, rtol=1e-5)


# --- whole models -----------------------------------------------------------------------


def test_int8_cnn_tracks_the_float_cnn():
    model = tiny_cnn()
    calib = log_mel_like(64, 16, 16, seed=0)
    qmodel = quant.quantize_idcnn(model, calib)
    x = log_mel_like(32, 16, 16, seed=1)
    with torch.no_grad():
        want = model(x).numpy()
    got = qmodel.logits(x.numpy())
    # measured 1.3 %; one zero point off by 20 gives 6 %
    assert np.abs(got - want).max() < 0.03 * np.abs(want).max()
    ids = torch.zeros(len(x), dtype=torch.long)
    score_f = id_scores(torch.from_numpy(want), ids).numpy()
    score_q = id_scores(torch.from_numpy(got), ids).numpy()
    assert np.corrcoef(score_f, score_q)[0, 1] > 0.99


def test_int8_cnn_is_integer_until_the_head_and_deterministic():
    qmodel = quant.quantize_idcnn(tiny_cnn(), log_mel_like(64, 16, 16, seed=0))
    q_in = qmodel.quantize_input(log_mel_like(4, 16, 16, seed=2).numpy())
    assert q_in.dtype == np.int8 and q_in.shape == (4, 1, 16, 16)
    trace: list[np.ndarray] = []
    acc = qmodel.head_acc(q_in, trace)
    assert acc.dtype == np.int64 and acc.shape == (4, 3)
    assert [t.dtype for t in trace] == [np.int8] * 3
    assert np.abs(acc).max() < 2**31  # fits the int32 accumulator on the device
    np.testing.assert_array_equal(acc, qmodel.head_acc(q_in))
    for layer in qmodel.convs:
        assert layer.weight.dtype == np.int8 and layer.bias.dtype == np.int32
    assert [layer.pool_before for layer in qmodel.convs] == [False, True, True]
    assert [layer.stride for layer in qmodel.convs] == [2, 1, 1]
    with pytest.raises(ValueError, match="pooled positions"):
        qmodel.head_acc(qmodel.quantize_input(log_mel_like(1, 32, 16, seed=3).numpy()))


def test_int8_autoencoder_tracks_the_float_errors():
    torch.manual_seed(0)
    model = DenseAutoencoder(input_dim=40, hidden=16, n_hidden=2, bottleneck=4)
    _randomize_bn(model, 0)
    model.eval()
    calib = torch.randn(256, 40) * 8 - 40
    qmodel = quant.quantize_autoencoder(model, calib)
    x = torch.randn(64, 40) * 8 - 40
    with torch.no_grad():
        want = ((x - model(x)) ** 2).mean(dim=1).numpy()
    got = qmodel.errors(x.numpy())
    assert np.corrcoef(want, got)[0, 1] > 0.99
    assert np.abs(got - want).max() < 0.05 * want.max()
    assert len(qmodel.hidden) == 5  # 2 encoder blocks, the bottleneck, 2 decoder blocks


def test_int8_gmm_tracks_the_float_gmm():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-40, 3, (300, 8)), rng.normal(-30, 2, (300, 8))]
    gm = GaussianMixture(2, covariance_type="full", reg_covar=0.01, random_state=0).fit(x)
    qgmm = quant.quantize_gmm(gm)
    assert qgmm.chol_q.dtype == np.int8
    assert np.allclose(np.tril(qgmm.chol_q.astype(int), -1), 0)  # only the upper triangle
    test = np.r_[rng.normal(-40, 3, (50, 8)), rng.normal(-35, 6, (50, 8))]
    want = -gm.score_samples(test)
    got = qgmm.score(test)
    assert np.corrcoef(want, got)[0, 1] > 0.999
    assert qgmm.nbytes() == 2 * (8 * 9 // 2 + 4 * (8 + 8 + 2))


def test_phase3_cnn_fits_256_kb_and_the_phase1_autoencoder_does_not():
    cnn_cfg = load_config(CONFIGS / "phase3_idcnn_mixup.yaml")
    cnn = train_idcnn.build_model(cnn_cfg, n_classes=7).eval()
    q_cnn = quant.quantize_idcnn(cnn, log_mel_like(8, 64, 128, seed=0))
    ae_cfg = load_config(CONFIGS / "phase1_baseline.yaml")
    ae = train.build_model(ae_cfg, input_dim=640).eval()
    q_ae = quant.quantize_autoencoder(ae, torch.randn(32, 640))
    assert q_cnn.nbytes() < 256 * 1024 < q_ae.nbytes()
    n_int8 = sum(layer.weight.size for layer in q_cnn.convs) + q_cnn.head_weight.size
    assert n_int8 == 96_912 + 128 * 7


def test_linear_acc_is_exact_integer_arithmetic():
    rng = np.random.default_rng(0)
    x = rng.integers(-255, 256, size=(7, 640))
    w = rng.integers(-127, 128, size=(128, 640)).astype(np.int8)
    got = quant.linear_acc(x, w)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, x.astype(np.int64) @ w.T.astype(np.int64))
