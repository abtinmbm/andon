import numpy as np
import pytest

from andon.features.logmel import (
    LogMelConfig,
    hann_window,
    hz_to_mel,
    logmel,
    mel_filterbank,
    mel_to_hz,
)

SR = 16_000


def sine(freq_hz: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_shape_matches_centered_framing():
    y = sine(1000.0, seconds=10.0)
    feats = logmel(y)
    # Centered frames: 1 + len // hop. 10 s at 16 kHz -> 313 frames, as in the baseline.
    assert feats.shape == (1 + y.size // 512, 128)
    assert feats.shape[0] == 313
    assert feats.dtype == np.float32


@pytest.mark.parametrize("freq", [250.0, 1000.0, 3000.0, 6000.0])
def test_sine_peaks_in_the_right_mel_band(freq):
    feats = logmel(sine(freq))
    peak_band = int(np.argmax(feats[10:-10].mean(axis=0)))  # skip padded edge frames

    # The band whose triangle is centered closest to the tone.
    cfg = LogMelConfig()
    centers = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2), cfg.n_mels + 2))[1:-1]
    expected = int(np.argmin(np.abs(centers - freq)))
    assert abs(peak_band - expected) <= 1


def test_louder_signal_is_about_6_db_higher_per_doubling():
    quiet = logmel(sine(1000.0, amp=0.25))[10:-10].max()
    loud = logmel(sine(1000.0, amp=0.5))[10:-10].max()
    assert loud - quiet == pytest.approx(20 * np.log10(2), abs=0.05)


def test_silence_is_finite():
    feats = logmel(np.zeros(SR, dtype=np.float32))
    assert np.isfinite(feats).all()


def test_mel_scale_round_trip_and_knee():
    hz = np.array([0.0, 500.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz)
    assert hz_to_mel(1000.0) == pytest.approx(15.0)  # Slaney: linear part ends at 15 mel


def test_filterbank_shape_and_non_negative():
    fb = mel_filterbank(SR, 1024, 128, 0.0, SR / 2)
    assert fb.shape == (128, 513)
    assert (fb >= 0).all()
    assert (fb.sum(axis=1) > 0).all()  # no empty filters at these settings


def test_hann_is_periodic():
    w = hann_window(8)
    assert w[0] == 0.0
    assert w[4] == pytest.approx(1.0)
    assert w[-1] != 0.0  # periodic, not symmetric


def test_rejects_stereo():
    with pytest.raises(ValueError):
        logmel(np.zeros((2, SR), dtype=np.float32))


def test_matches_librosa_baseline_front_end():
    """Same numbers as the DCASE 2020 baseline's librosa call, within float32 noise."""
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(SR * 2) + sine(440.0, seconds=2.0)).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024, hop_length=512, n_mels=128, power=2.0
    )
    expected = (20.0 / 2.0 * np.log10(mel + np.finfo(float).eps)).T

    ours = logmel(y)
    assert ours.shape == expected.shape
    assert np.max(np.abs(ours - expected)) < 1e-2  # dB
