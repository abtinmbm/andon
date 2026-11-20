"""Phase H5: the C header export of the per-fan GMM and its test vectors."""

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_h3 import RECORDING, manifest

from andon import export_c, export_gmm, h3
from andon.data.features import SplitFeatures
from andon.experiment import Config, load_config
from andon.features.logmel import logmel

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("gmm")
    ex, mixtures = export_gmm.synthetic_export(seed=0)
    export_gmm.write(ex, mixtures, out)  # writes, then verifies the read-back
    return ex, mixtures, out


def test_bank_scores_as_scikit_learn_does(exported):
    ex, mixtures, _ = exported
    rng = np.random.default_rng(1)
    for m, fan in enumerate(ex.bank.fans):
        x = mixtures[fan].means_[0] + rng.normal(0, 3, (5, ex.bank.mean.shape[-1]))
        np.testing.assert_allclose(ex.bank.score(m, x), -mixtures[fan].score_samples(x), 1e-10)


def test_packing_round_trips_and_is_column_by_column():
    chol = np.array([[[[2.0, 1.0, 4.0], [0.0, 3.0, 5.0], [0.0, 0.0, 6.0]]]])
    bank = export_gmm.GmmBank(["a"], np.zeros((1, 1, 3)), chol, np.zeros((1, 1)))
    packed = bank.packed_chol()
    np.testing.assert_array_equal(packed, [[[2.0, 1.0, 3.0, 4.0, 5.0, 6.0]]])
    back = export_gmm.GmmBank.from_packed(["a"], bank.mean, packed, bank.log_const)
    np.testing.assert_array_equal(back.chol, chol)


def test_synthetic_export_round_trips_through_the_headers(exported):
    ex, _, out = exported
    text = (out / export_gmm.GMM_HEADER).read_text(encoding="utf-8")
    back = export_gmm.bank_from_header(text)
    assert back.fans == ["id_00", "id_01", "id_02"]
    np.testing.assert_array_equal(back.mean, ex.bank.mean.astype(np.float32))
    np.testing.assert_array_equal(back.chol, ex.bank.chol.astype(np.float32))
    _, d = export_c.read_header(text)
    assert (d["ANDON_GMM_N_FANS"], d["ANDON_GMM_N_COMPONENTS"], d["ANDON_GMM_N_DIM"]) == (
        3,
        4,
        128,
    )
    assert (d["ANDON_GMM_WINDOW_FRAMES"], d["ANDON_GMM_WINDOW_HOP"]) == (64, 32)
    tv, vd = export_c.read_header((out / export_gmm.VECTORS_HEADER).read_text(encoding="utf-8"))
    assert (vd["ANDON_GV_FRAMES"], vd["ANDON_GV_N_WINDOWS"]) == (128, 3)
    assert vd["ANDON_GV_AUDIO_LEN"] == 127 * 512
    scores = tv["andon_gv_score"].reshape(3, 3, 3)
    # Each fan's audio is far more likely under its own GMM than under another's: the
    # vectors hold both small and very large scores.
    for f in range(3):
        own = scores[f, :, f]
        others = np.delete(scores[f], f, axis=1)
        assert own.max() * 100 < others.min()


def test_both_exporters_write_the_same_front_end(exported, tmp_path):
    ex, _, out = exported
    export_c.write(export_c.synthetic_export(seed=0, n=2), tmp_path, n=2)
    assert (out / export_c.FRONTEND_HEADER).read_bytes() == (
        tmp_path / export_c.FRONTEND_HEADER
    ).read_bytes()


@pytest.mark.parametrize("array", ["andon_gmm_log_const", "andon_gmm_chol"])
def test_verify_catches_a_changed_parameter(exported, tmp_path, array):
    ex, _, out = exported
    for name in (export_c.FRONTEND_HEADER, export_gmm.GMM_HEADER, export_gmm.VECTORS_HEADER):
        shutil.copy(out / name, tmp_path / name)
    path = tmp_path / export_gmm.GMM_HEADER
    text = path.read_text(encoding="utf-8")
    arrays, _ = export_c.read_header(text)
    values = arrays[array].copy()
    # Every fan-0 component, so the change reaches whichever component dominates a window.
    n = values.size // ex.bank.mean.shape[0]
    values[:n] *= 1.01
    start = text.index(f"static const float {array}[")
    end = text.index("};\n", start) + 3
    text = text[:start] + export_c.c_array("float", array, values) + text[end:]
    path.write_text(text, encoding="utf-8")
    with pytest.raises(AssertionError):
        export_gmm.verify(tmp_path, ex.logmel_cfg)


def test_the_c_window_needs_width_twice_the_hop(exported):
    ex, _, _ = exported
    with pytest.raises(ValueError, match="2 \\* hop"):
        export_gmm.gmm_header(replace(ex, width=60))


def _h3_cfg() -> Config:
    raw = load_config(CONFIGS / "h3_gmm.yaml").raw
    raw = dict(raw, data=dict(raw["data"], limit_clips=4), final=False)
    return Config("h3_gmm", raw)


def test_the_export_fits_the_model_h3_scored():
    """Same clips, same windows, same detector: fit_gmms on the training clips alone gives
    the scores h3.score gave on the fold."""
    n_mels, n_frames = 6, 80
    cfg = _h3_cfg()
    fold = h3.Fold(("d1",), "d2")
    clips = h3.fold_clips(cfg, RECORDING, manifest(), fold, "dev_test")
    rng = np.random.default_rng(0)
    level = {"id_00": 0.0, "id_01": 5.0}
    frames = np.concatenate(
        [level[c.clip.machine_id] + rng.normal(0, 1, (n_frames, n_mels)) for c in clips]
    ).astype(np.float32)
    feats = SplitFeatures(frames, np.arange(len(clips) + 1, dtype=np.int64) * n_frames)
    scored = h3.score(cfg, 0, clips, feats, torch.device("cpu"))

    train = [i for i, c in enumerate(clips) if c.role == "train"]
    assert train == list(range(len(train)))  # training clips come first in a fold
    mixtures = export_gmm.fit_gmms(
        cfg.raw["model"],
        0,
        h3._subset(feats, train),
        [clips[i].clip.machine_id for i in train],
        64,
        32,
    )
    starts, owner = h3.windows(feats, 64, 32)
    x = h3.window_means(feats.frames, starts, 64)
    test_w = np.array([clips[o].role != "train" for o in owner])
    fan_of = np.array([clips[o].clip.machine_id for o in owner])[test_w]
    want = np.empty(test_w.sum())
    for fan, gm in mixtures.items():
        want[fan_of == fan] = -gm.score_samples(x[test_w][fan_of == fan])
    np.testing.assert_array_equal(scored.scores, want)


def test_training_clips_are_dev_only_and_match_the_eval_fold():
    rows = manifest()
    cfg = _h3_cfg()
    clips = h3.training_clips(cfg, RECORDING, rows, ("d1", "d2"))
    assert {c.session for c in clips} == {"d1", "d2"}
    assert {c.role for c in clips} == {"train"}
    fold_train = [
        c
        for c in h3.fold_clips(
            Config("t", dict(cfg.raw, final=True)),
            RECORDING,
            rows,
            h3.folds(RECORDING, "eval")[0],
            "eval",
        )
        if c.role == "train"
    ]
    assert clips == fold_train
    with pytest.raises(ValueError, match="dev sessions"):
        h3.training_clips(cfg, RECORDING, rows, ("d1", "d3"))


def test_real_export_needs_a_full_covariance_gmm_config():
    with pytest.raises(ValueError, match="per-fan GMM"):
        export_gmm.real_export(load_config(CONFIGS / "h3_level.yaml"), 0)


# --- H6: the board's detector settings and its stream ------------------------------------


def test_detector_settings_come_from_the_configs(exported):
    _, _, out = exported
    d = export_gmm.detector_from_header((out / export_gmm.GMM_HEADER).read_text(encoding="utf-8"))
    # h2_recording.yaml: gain shift 4, 5 s settle; h3_gmm.yaml: 60 clips of 10 s set the
    # limits, at 2 and 3 sigma, light 2/3/3.
    assert (d.gain_shift, d.holdoff_samples) == (4, 5 * 16000)
    # 600 s streamed: (600 * 16000 - 1024) // 512 + 1 = 18749 frames; (18749 - 64) // 32 + 1.
    assert d.calib_windows == 584
    assert (d.warn_sigma, d.control_sigma) == (2.0, 3.0)
    assert (d.light.warn_on, d.light.warn_off, d.light.red_hold) == (2, 3, 3)


def test_the_board_cannot_take_log_scale_limits():
    raw = load_config(CONFIGS / "h3_gmm.yaml").raw
    raw = dict(raw, limits=dict(raw["limits"], log_scale=True))
    with pytest.raises(ValueError, match="log_scale"):
        export_gmm.DetectorSettings.from_config(Config("h3_gmm", raw), 64, 32)


def test_light_vectors_reach_every_state_and_follow_andon_light(exported):
    ex, _, _ = exported
    lv = export_gmm.light_vectors(ex.detector)
    assert set(lv.light.tolist()) == {0, 1, 2}
    assert lv.light[-1] == 2 and lv.n_calib == export_gmm.LIGHT_CALIB
    calib = lv.score[: lv.n_calib].astype(np.float64)
    assert lv.limits[0] == calib.mean() and lv.limits[1] == calib.std(ddof=1)


def test_stream_frames_are_the_centred_frames_one_later(exported):
    """The board's frame t is samples t * hop to t * hop + n_fft - 1, with no padding: the
    middle frame of the centred log-mel of exactly those samples, and centred frame t + 1."""
    ex, _, _ = exported
    cfg = ex.logmel_cfg
    audio = ex.audio[0].astype(np.float32) / 32768
    centred = logmel(audio, cfg)
    for t in (0, 1, 50, centred.shape[0] - 3):
        chunk = audio[t * cfg.hop_length : t * cfg.hop_length + cfg.n_fft]
        # Same samples, but NumPy's batched matrix product rounds a 3-frame and a 128-frame
        # spectrogram differently in the last float32 bit (seen: 1.9e-07 relative).
        np.testing.assert_allclose(logmel(chunk, cfg)[1], centred[t + 1], rtol=1e-6)


def test_stream_scores_are_the_scores_of_the_shifted_windows(exported):
    ex, mixtures, _ = exported
    v = export_gmm.reference_vectors(ex, mixtures)
    inner = v.logmel[0, 1:-1]
    assert v.stream_score.shape[1] == (inner.shape[0] - ex.width) // ex.hop + 1
    x = inner[ex.hop : ex.hop + ex.width].mean(axis=0, dtype=np.float64)[None]
    np.testing.assert_allclose(v.stream_score[0, 1, 0], -mixtures["id_00"].score_samples(x)[0])
