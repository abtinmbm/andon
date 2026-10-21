"""Phase 4: MAC counts and the time estimate against the microcontroller budget."""

from pathlib import Path

import pytest
import torch

from andon import budget, train, train_idcnn
from andon.experiment import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_cnn_macs_match_the_hand_count():
    cfg = load_config(CONFIGS / "phase3_idcnn_mixup.yaml")
    model = train_idcnn.build_model(cfg, n_classes=7)
    # 128 mels x 64 frames: stem 64x32 out, then 32x16, 16x8 and 8x4 after each pool
    want = 64 * 32 * 16 * 9 + 32 * 16 * 32 * 16 * 9 + 16 * 8 * 64 * 32 * 9 + 8 * 4 * 128 * 64 * 9
    want += 128 * 7
    assert budget.module_macs(model, torch.zeros(1, 64, 128)) == want == 7_373_696
    assert budget.folded_params(model) == 96_912 + (16 + 32 + 64 + 128) + 128 * 7 + 7


def test_autoencoder_macs_are_its_dense_weights():
    cfg = load_config(CONFIGS / "phase1_baseline.yaml")
    model = train.build_model(cfg, input_dim=640)
    want = 640 * 128 + 3 * 128 * 128 + 128 * 8 + 8 * 128 + 3 * 128 * 128 + 128 * 640
    assert budget.module_macs(model, torch.zeros(1, 640)) == want == 264_192
    with pytest.raises(ValueError, match="batch of one"):
        budget.module_macs(model, torch.zeros(2, 640))


def test_front_end_and_time_estimate():
    fe = budget.front_end_macs(n_fft=1024, n_mels=128)
    assert fe == 1024 + (4 * 256 * 9 + 2048) + 4 * 513 + 128
    # 0.5 MAC per cycle at 150 MHz: 75,000 MACs take 1 ms
    assert budget.ms_for(75_000) == pytest.approx(1.0)
    b = budget.Budget(
        params=10,
        int8_bytes=1000,
        macs_per_decision=75_000 * 32,
        frames_per_decision=32,
        front_end_macs_per_frame=75_000,
    )
    assert b.ms_per_decision == pytest.approx(32.0)
    assert b.ms_per_frame == pytest.approx(2.0)
    assert b.fits
    assert "est_ms_per_frame=2.00" in b.notes()
    too_big = budget.Budget(10, budget.BUDGET_BYTES + 1, 1, 1, 1)
    too_slow = budget.Budget(10, 1000, 75_000 * 20, 1, 0)
    assert not too_big.fits and not too_slow.fits


def test_gmm_macs():
    assert budget.gmm_macs(4, 128) == 4 * (128 + 128 * 129 // 2 + 128)
