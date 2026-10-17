"""Phase 1 pipeline: training data selection, features, model, training, scoring, logging."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from andon.data.audio import write_wav
from andon.data.dataset import dev_test_clips, train_clips
from andon.data.features import (
    SplitFeatures,
    cache_key,
    clip_logmel,
    extract,
    gather_windows,
    load_or_extract,
    stack_frames,
    window_starts,
)
from andon.evaluate import clip_scores, per_id_metrics, summary_table
from andon.experiment import append_result, is_logged, load_config, read_results, set_seed
from andon.features.logmel import LogMelConfig
from andon.models.autoencoder import DenseAutoencoder
from andon.train import holdout_split, run, train_autoencoder

SR = 16_000
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase1_baseline.yaml"


def noise_wav(path: Path, seed: int, n: int = 8000, gain: float = 0.1) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, gain * rng.standard_normal(n).astype(np.float32), SR)


def make_tree(root: Path, machine_type: str = "fan", n_train: int = 4) -> None:
    """Half-second clips: normals are quiet noise, anomalies are much louder."""
    base = root / "dev_data" / machine_type
    for k, machine_id in enumerate(("id_00", "id_02")):
        for i in range(n_train):
            noise_wav(base / "train" / f"normal_{machine_id}_{i:08d}.wav", 100 * k + i)
        for i in range(3):
            noise_wav(base / "test" / f"normal_{machine_id}_{i:08d}.wav", 1000 + 100 * k + i)
            noise_wav(base / "test" / f"anomaly_{machine_id}_{i:08d}.wav", 2000 + i, gain=0.8)


def tiny_config(tmp_path: Path, epochs: int = 3) -> Path:
    raw = yaml.safe_load(CONFIG.read_text())
    raw["data"] = {"root": str(tmp_path / "data"), "machine_types": ["fan"]}
    raw["train"]["epochs"] = epochs
    raw["train"]["batch_size"] = 8
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


# --- training data selection -------------------------------------------------------------


def test_train_clips_reads_only_train_normals(tmp_path):
    make_tree(tmp_path)
    clips = train_clips(tmp_path, "fan")
    assert len(clips) == 8
    assert all(c.split == "train" and not c.is_anomaly for c in clips)
    assert [c.path.name for c in clips] == sorted(c.path.name for c in clips)
    test = dev_test_clips(tmp_path, "fan")
    assert len(test) == 12 and sum(c.is_anomaly for c in test) == 6


def test_train_clips_refuses_anomaly_in_train_folder(tmp_path):
    make_tree(tmp_path)
    noise_wav(tmp_path / "dev_data" / "fan" / "train" / "anomaly_id_00_00000000.wav", 0)
    with pytest.raises(ValueError, match="anomalous clip"):
        train_clips(tmp_path, "fan")


def test_split_helpers_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown machine type"):
        train_clips(tmp_path, "drill")
    with pytest.raises(FileNotFoundError):
        train_clips(tmp_path, "fan")


# --- features ----------------------------------------------------------------------------


def test_clip_logmel_rejects_wrong_sample_rate(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, np.zeros(8000, np.float32), 8000)
    with pytest.raises(ValueError, match="sample rate 8000"):
        clip_logmel(path, LogMelConfig())


def baseline_vectors(log_mel_time_major: np.ndarray, frames: int) -> np.ndarray:
    """The baseline's own stacking loop (common.py), on its (n_mels, n_frames) layout."""
    log_mel = log_mel_time_major.T
    n_mels = log_mel.shape[0]
    size = log_mel.shape[1] - frames + 1
    vectors = np.zeros((size, n_mels * frames))
    for t in range(frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel[:, t : t + size].T
    return vectors


def test_window_stacking_matches_the_baseline_loop():
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal((n, 4)).astype(np.float32) for n in (7, 5, 9)]
    feats = SplitFeatures(np.concatenate(clips), np.r_[0, np.cumsum([7, 5, 9])])
    starts = window_starts(feats.offsets, 5)
    assert len(starts) == 3 + 1 + 5  # no window crosses a clip boundary
    expected = np.concatenate([baseline_vectors(c, 5) for c in clips])
    np.testing.assert_array_equal(stack_frames(feats.frames, starts, 5), expected)
    torch_version = gather_windows(torch.from_numpy(feats.frames), torch.from_numpy(starts), 5)
    np.testing.assert_array_equal(torch_version.numpy(), expected)


def test_feature_cache_is_reused_and_keyed_on_clips_and_config(tmp_path):
    make_tree(tmp_path)
    clips = train_clips(tmp_path, "fan")
    cfg = LogMelConfig()
    first = load_or_extract(clips, cfg, tmp_path / "cache", "fan_train")
    assert first.n_clips == 8 and first.frames.shape[1] == 128
    np.testing.assert_array_equal(first.clip(0), clip_logmel(clips[0].path, cfg))
    for c in clips:  # a second call must come from the cache, not the audio
        c.path.unlink()
    again = load_or_extract(clips, cfg, tmp_path / "cache", "fan_train")
    np.testing.assert_array_equal(again.frames, first.frames)
    assert cache_key(clips, cfg) != cache_key(clips[:-1], cfg)
    assert cache_key(clips, cfg) != cache_key(clips, LogMelConfig(n_mels=64))


# --- model -------------------------------------------------------------------------------


def test_autoencoder_matches_keras_architecture():
    model = DenseAutoencoder()
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == (
        [(640, 128)] + [(128, 128)] * 3 + [(128, 8), (8, 128)] + [(128, 128)] * 3 + [(128, 640)]
    )
    # Keras model.summary() of the baseline: 267,928 trainable parameters.
    assert sum(p.numel() for p in model.parameters()) == 267_928
    assert all(torch.all(m.bias == 0) for m in linears)
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert len(bns) == 9 and all(b.momentum == 0.01 and b.eps == 1e-3 for b in bns)
    model.eval()
    assert model(torch.zeros(3, 640)).shape == (3, 640)


def test_holdout_is_the_unshuffled_tail():
    train, val = holdout_split(np.arange(100), 0.1)
    np.testing.assert_array_equal(train, np.arange(90))
    np.testing.assert_array_equal(val, np.arange(90, 100))


# --- training and scoring ----------------------------------------------------------------


def synthetic_feats(seed: int = 0, n_clips: int = 6, n_frames: int = 20) -> SplitFeatures:
    rng = np.random.default_rng(seed)
    pattern = rng.standard_normal(128).astype(np.float32)
    frames = pattern + 0.1 * rng.standard_normal((n_clips * n_frames, 128)).astype(np.float32)
    return SplitFeatures(frames, np.arange(n_clips + 1) * n_frames)


def test_training_reduces_loss_and_is_reproducible(tmp_path):
    cfg = load_config(tiny_config(tmp_path, epochs=15))
    cpu = torch.device("cpu")
    set_seed(0)
    model, history = train_autoencoder(synthetic_feats(), cfg, cpu, log=lambda _: None)
    assert history[-1]["train_loss"] < 0.5 * history[0]["train_loss"]
    assert not model.training
    set_seed(0)
    _, again = train_autoencoder(synthetic_feats(), cfg, cpu, log=lambda _: None)
    assert again == history


def test_clip_scores_are_mean_window_errors():
    torch.manual_seed(0)
    model = DenseAutoencoder().eval()
    feats = synthetic_feats(n_clips=3, n_frames=9)
    scores = clip_scores(model, feats, 5, torch.device("cpu"))
    with torch.no_grad():
        for i in range(3):
            x = torch.from_numpy(stack_frames(feats.clip(i), np.arange(5), 5))
            expected = ((x - model(x)) ** 2).mean(dim=1).mean().item()
            assert scores[i] == pytest.approx(expected, rel=1e-5)


def test_per_id_metrics_are_computed_per_machine_id(tmp_path):
    make_tree(tmp_path)
    clips = dev_test_clips(tmp_path, "fan")
    # id_00 perfectly separated, id_02 perfectly inverted.
    scores = np.array([(c.is_anomaly == (c.machine_id == "id_00")) * 1.0 for c in clips])
    result = per_id_metrics(clips, scores)
    assert result["id_00"] == (1.0, pytest.approx(1.0))
    assert result["id_02"][0] == 0.0


# --- results log and the end-to-end run ----------------------------------------------------


def test_results_rows_and_summary(tmp_path):
    cfg = load_config(tiny_config(tmp_path))
    log = tmp_path / "results.csv"
    for seed, auc in enumerate((0.65, 0.66, 0.67)):
        append_result(cfg, "fan", "dev_test", seed, auc, 0.52345678, "x", log=log)
    rows = read_results(log)
    assert len(rows) == 3 and rows[0]["pauc"] == "0.5235" and rows[2]["auc"] == "0.6700"
    assert is_logged(cfg, "fan", 1, "dev_test", log=log)
    assert not is_logged(cfg, "fan", 3, "dev_test", log=log)
    table = summary_table(cfg, log=log)
    assert "66.00±1.00" in table and "yes" in table  # published fan AUC is 65.83
    with pytest.raises(ValueError, match="split"):
        append_result(cfg, "fan", "test", 0, 0.5, 0.5, log=log)


def test_run_end_to_end_and_resume(tmp_path, monkeypatch):
    make_tree(tmp_path / "data")
    monkeypatch.chdir(tmp_path)  # checkpoints/, results/ and data/cache/ land in tmp
    cfg = load_config(tiny_config(tmp_path))
    run(cfg, seed=0, machine_types=["fan"], device=torch.device("cpu"))
    rows = read_results(Path("results/results_log.csv"))
    assert len(rows) == 1
    row = rows[0]
    assert (row["config"], row["machine_type"], row["split"], row["seed"]) == (
        "tiny",
        "fan",
        "dev_test",
        "0",
    )
    assert "id_00=" in row["notes"] and "id_02=" in row["notes"]
    assert cfg.checkpoint_path("fan", 0).exists()
    run(cfg, seed=0, machine_types=["fan"], device=torch.device("cpu"))  # already logged
    assert len(read_results(Path("results/results_log.csv"))) == 1


def test_extract_offsets_match_clip_lengths(tmp_path):
    make_tree(tmp_path)
    clips = train_clips(tmp_path, "fan")[:3]
    feats = extract(clips, LogMelConfig())
    assert feats.n_clips == 3
    assert feats.offsets[-1] == feats.frames.shape[0]


def test_results_append_waits_for_the_lock(tmp_path):
    from andon import experiment

    cfg = load_config(tiny_config(tmp_path))
    log = tmp_path / "results.csv"
    log.with_suffix(".lock").touch()  # another run is mid-write
    with pytest.raises(TimeoutError), experiment._file_lock(log.with_suffix(".lock"), 0.2):
        pass
    log.with_suffix(".lock").unlink()
    append_result(cfg, "fan", "dev_test", 0, 0.5, 0.5, log=log)
    assert not log.with_suffix(".lock").exists()
    assert len(read_results(log)) == 1
