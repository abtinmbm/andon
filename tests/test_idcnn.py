"""Phase 3: the machine-ID CNN, its augmentations, scoring, and the score-only entry point."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

from andon import score, train, train_idcnn
from andon.data.audio import write_wav
from andon.data.features import SplitFeatures, concat
from andon.experiment import Config, load_config, read_results, set_seed
from andon.models.idcnn import IdCNN, id_scores, mask_spectrogram, mixup

SR = 16_000
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def tiny_cfg(root: Path, **train_overrides) -> Config:
    raw = yaml.safe_load((CONFIGS / "phase3_idcnn.yaml").read_text())
    raw["data"] = {"root": str(root), "machine_types": ["fan"]}
    raw["features"]["n_mels"] = 16
    raw["model"].update(channels=[4, 8], patch_frames=8, score_hop=4)
    raw["train"].update({"epochs": 2, "batch_size": 8, **train_overrides})
    return Config(name="tiny_idcnn", raw=raw)


# --- model ------------------------------------------------------------------------------


def test_model_shapes_and_size_of_the_phase3_config():
    cfg = load_config(CONFIGS / "phase3_idcnn.yaml")
    model = train_idcnn.build_model(cfg, n_classes=7)
    logits = model(torch.randn(5, cfg.raw["model"]["patch_frames"], 128))
    assert logits.shape == (5, 7)
    n_weights = sum(p.numel() for p in model.parameters())
    assert 90_000 < n_weights < 110_000  # the size that was benchmarked for CPU training


def test_normalization_is_fitted_on_training_frames_and_saved():
    model = IdCNN(n_classes=2, n_mels=3, channels=(4, 8))
    frames = torch.tensor([[1.0, 10.0, -5.0], [3.0, 30.0, -5.0]])
    model.set_normalization(frames)
    assert torch.allclose(model.band_mean, torch.tensor([2.0, 20.0, -5.0]))
    assert model.band_std[2] > 0  # a constant band must not divide by zero
    assert {"band_mean", "band_std"} <= set(model.state_dict())
    z = model.normalize(frames[None])
    assert torch.allclose(z[0, :, :2].mean(dim=0), torch.zeros(2), atol=1e-6)


def test_id_score_is_minus_log_probability_of_the_true_id():
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
    s = id_scores(logits, torch.tensor([0, 1]))
    expected = -F.log_softmax(logits, dim=1)[[0, 1], [0, 1]]
    assert torch.allclose(s, expected)
    assert s[1] > s[0]  # unsure about the true ID -> more anomalous


# --- augmentation ------------------------------------------------------------------------


def test_mixup_off_returns_the_batch_unchanged():
    x, y = torch.randn(4, 8, 3), torch.tensor([0, 1, 0, 1])
    xm, ya, yb, lam = mixup(x, y, 0.0)
    assert xm is x and lam == 1.0 and torch.equal(ya, y) and torch.equal(yb, y)


def test_mixup_blends_each_patch_with_one_partner():
    torch.manual_seed(0)
    x, y = torch.randn(6, 8, 3), torch.arange(6)
    xm, ya, yb, lam = mixup(x, y, 0.2)
    assert 0.0 <= lam <= 1.0
    assert torch.equal(ya, y) and sorted(yb.tolist()) == list(range(6))
    assert torch.allclose(xm, lam * x + (1 - lam) * x[yb])  # labels are 0..5, so yb is perm


def test_masks_zero_whole_bands_and_time_spans_per_patch():
    torch.manual_seed(1)
    x = torch.ones(16, 20, 12)
    out = mask_spectrogram(x, n_freq_masks=1, max_freq_width=3, n_time_masks=1, max_time_width=4)
    zero = out == 0
    for b in range(16):
        bands = zero[b].all(dim=0)  # a masked band is zero at every frame
        frames = zero[b].all(dim=1)
        assert bands.sum() <= 3 and frames.sum() <= 4
        # every zero is explained by a masked band or a masked frame
        assert torch.equal(zero[b], bands[None, :] | frames[:, None])
    assert len({tuple(zero[b].all(dim=0).tolist()) for b in range(16)}) > 1  # masks differ
    assert torch.equal(mask_spectrogram(x, 0, 0, 0, 0), x)


# --- patches and scoring -----------------------------------------------------------------


def test_patch_starts_cover_the_clip_to_its_last_frame():
    assert patch_list(313, 64, 32) == [0, 32, 64, 96, 128, 160, 192, 224, 249]
    assert patch_list(128, 64, 32) == [0, 32, 64]
    assert patch_list(64, 64, 32) == [0]
    with pytest.raises(ValueError):
        train_idcnn.patch_starts(63, 64, 32)


def patch_list(n, w, h):
    return train_idcnn.patch_starts(n, w, h).tolist()


def test_random_patches_stay_inside_their_clip():
    lengths = [10, 12, 9]
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    owner = np.repeat(np.arange(3), lengths).astype(np.float32)
    frames = torch.from_numpy(np.stack([owner, np.arange(owner.size)], axis=1))
    rng = np.random.default_rng(0)
    idx = np.array([0, 1, 2, 2, 1])
    patches = train_idcnn.random_patches(frames, offsets, idx, 8, rng)
    assert patches.shape == (5, 8, 2)
    for p, i in zip(patches, idx, strict=True):
        assert (p[:, 0] == i).all()  # all frames come from clip i
        assert (p[1:, 1] - p[:-1, 1] == 1).all()  # consecutive frames


def test_clip_score_is_the_mean_patch_score():
    torch.manual_seed(0)
    model = IdCNN(n_classes=2, n_mels=4, channels=(4, 8)).eval()
    lengths = [12, 9]
    feats = SplitFeatures(
        frames=np.random.default_rng(0).standard_normal((21, 4)).astype(np.float32),
        offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64),
    )
    true_ids = np.array([1, 0])
    got = train_idcnn.clip_scores(
        model, feats, true_ids, width=8, hop=4, device=torch.device("cpu")
    )
    for i in range(2):
        clip = torch.from_numpy(feats.clip(i))
        starts = train_idcnn.patch_starts(len(clip), 8, 4)
        x = torch.stack([clip[s : s + 8] for s in starts])
        with torch.no_grad():
            want = id_scores(model(x), torch.full((len(x),), int(true_ids[i]))).mean()
        assert got[i] == pytest.approx(float(want), rel=1e-5)


def test_scoring_refuses_an_id_the_classifier_never_saw(tmp_path):
    from andon.data.dataset import Clip

    cfg = tiny_cfg(tmp_path)
    clip = Clip(tmp_path / "x.wav", "fan", "test", "id_09", False)
    model = train_idcnn.build_model(cfg, 2)
    feats = SplitFeatures(np.zeros((8, 16), np.float32), np.array([0, 8]))
    with pytest.raises(ValueError, match="id_09"):
        train_idcnn.score_clips(cfg, model, ["id_00", "id_01"], [clip], feats, torch.device("cpu"))


def test_concat_keeps_clip_order_and_offsets():
    a = SplitFeatures(np.arange(6, dtype=np.float32).reshape(3, 2), np.array([0, 1, 3]))
    b = SplitFeatures(np.full((2, 2), 9, np.float32), np.array([0, 2]))
    c = concat([a, b])
    assert c.n_clips == 3
    assert c.offsets.tolist() == [0, 1, 3, 5]
    assert (c.clip(2) == 9).all() and np.array_equal(c.clip(1), a.clip(1))


# --- training ----------------------------------------------------------------------------


def two_machine_features(n_clips: int = 24, n_frames: int = 20, n_mels: int = 16):
    """Machine 0 is loud in the low bands, machine 1 in the high bands."""
    rng = np.random.default_rng(0)
    clips, labels = [], []
    for i in range(n_clips):
        k = i % 2
        f = rng.standard_normal((n_frames, n_mels)).astype(np.float32)
        f[:, : n_mels // 2] += 3.0 if k == 0 else 0.0
        f[:, n_mels // 2 :] += 3.0 if k == 1 else 0.0
        clips.append(f)
        labels.append(k)
    offsets = np.r_[0, np.cumsum([n_frames] * n_clips)].astype(np.int64)
    return SplitFeatures(np.concatenate(clips), offsets), np.array(labels)


@pytest.mark.parametrize(
    "aug",
    [
        {},
        {"mixup_alpha": 0.2},
        {"spec_mask": {"freq_masks": 1, "freq_width": 2, "time_masks": 1, "time_width": 2}},
    ],
    ids=["plain", "mixup", "masks"],
)
def test_training_learns_two_ids_and_is_reproducible(tmp_path, aug):
    cfg = tiny_cfg(tmp_path, epochs=15, learning_rate=0.01, **aug)
    feats, labels = two_machine_features()
    cpu = torch.device("cpu")
    runs = []
    for _ in range(2):
        set_seed(0)
        model, history = train_idcnn.train_idcnn(feats, labels, 2, cfg, cpu, log=lambda _: None)
        runs.append((model, history))
    assert runs[0][1] == runs[1][1]  # same seed, same run
    model = runs[0][0]
    scores = train_idcnn.clip_scores(model, feats, labels, 8, 4, cpu)
    wrong = train_idcnn.clip_scores(model, feats, 1 - labels, 8, 4, cpu)
    assert (wrong > scores).mean() > 0.9  # the wrong ID is far less likely than the right one


# --- end to end: train, log dev test, score eval from checkpoints ---------------------------


def tone_wav(path: Path, freq: float, seed: int, gain: float = 0.3) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(8000) / SR  # half a second, about 16 frames
    y = gain * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(t.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, y.astype(np.float32), SR)


def make_dataset(root: Path) -> None:
    """Dev IDs 00 and 02, eval IDs 01 and 03, each machine a different tone."""
    dev, ev = root / "dev_data" / "fan", root / "eval_data" / "fan"
    gt = ["fan"]
    for k, mid in enumerate(("id_00", "id_02", "id_01", "id_03")):
        base, freq = (dev if k < 2 else ev), 300.0 * (k + 1)
        for i in range(4):
            tone_wav(base / "train" / f"normal_{mid}_{i:08d}.wav", freq, 10 * k + i)
        for i in range(2):
            if k < 2:
                tone_wav(dev / "test" / f"normal_{mid}_{i:08d}.wav", freq, 100 + 10 * k + i)
                tone_wav(dev / "test" / f"anomaly_{mid}_{i:08d}.wav", 3000.0, 200 + i)
            else:
                for j, (f, y) in enumerate(((freq, 0), (3000.0, 1))):
                    name = f"{mid}_{2 * i + j:08d}.wav"
                    tone_wav(ev / "test" / name, f, 300 + 10 * k + 2 * i + j)
                    gt.append(f"{name},orig_{name},{y}")
    (root / "eval_data" / "eval_data_list.csv").write_text("\n".join(gt) + "\n")


def write_cfg(cfg: Config, path: Path) -> Config:
    path.write_text(yaml.safe_dump(cfg.raw))
    return load_config(path)


def test_run_trains_logs_dev_test_then_scores_eval_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # checkpoints/, results/ and data/cache/ land in tmp
    make_dataset(tmp_path / "data")
    cfg = write_cfg(tiny_cfg(tmp_path / "data"), tmp_path / "tiny_idcnn.yaml")
    cpu = torch.device("cpu")
    log = tmp_path / "results" / "results_log.csv"

    train_idcnn.run(cfg, 0, ["fan"], cpu, log=log)
    ckpt = torch.load(cfg.checkpoint_path("fan", 0), weights_only=True)
    assert ckpt["classes"] == ["id_00", "id_01", "id_02", "id_03"]  # dev and eval IDs
    rows = read_results(log)
    assert [(r["split"], r["model"]) for r in rows] == [("dev_test", "id_cnn")]
    assert "id_00=" in rows[0]["notes"] and "id_01=" not in rows[0]["notes"]

    train_idcnn.run(cfg, 0, ["fan"], cpu, log=log)  # logged: nothing new
    assert len(read_results(log)) == 1

    with pytest.raises(FileNotFoundError, match="seed1"):  # seed 1 was never trained
        score.run(cfg, [0, 1], ["fan"], "eval", cpu, log=log)
    assert len(read_results(log)) == 1  # nothing was scored

    score.run(cfg, [0], ["fan"], "eval", cpu, log=log)
    score.run(cfg, [0], ["fan"], "eval", cpu, log=log)  # second call skips: eval runs once
    rows = read_results(log)
    assert [r["split"] for r in rows] == ["dev_test", "eval"]
    assert "id_01=" in rows[1]["notes"] and "id_03=" in rows[1]["notes"]
    assert "id_00=" not in rows[1]["notes"]

    # Scoring dev test again from the checkpoint gives the logged dev result.
    log2 = tmp_path / "log2.csv"
    score.run(cfg, [0], ["fan"], "dev_test", cpu, log=log2)
    assert read_results(log2)[0]["auc"] == rows[0]["auc"]


def test_autoencoder_on_additional_train_is_only_checkpointed_then_scored_on_eval(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    make_dataset(tmp_path / "data")
    raw = yaml.safe_load((CONFIGS / "phase3_baseline_eval.yaml").read_text())
    raw["data"]["root"] = str(tmp_path / "data")
    raw["data"]["machine_types"] = ["fan"]
    raw["train"].update(epochs=2, batch_size=8)
    cfg = write_cfg(Config("tiny_ae_eval", raw), tmp_path / "tiny_ae_eval.yaml")
    cpu = torch.device("cpu")
    log = tmp_path / "results" / "results_log.csv"

    train.run(cfg, 0, ["fan"], cpu)
    assert cfg.checkpoint_path("fan", 0).exists()
    assert read_results(log) == []  # trained on eval IDs: no dev test row

    score.run(cfg, [0], ["fan"], "eval", cpu, log=log)
    rows = read_results(log)
    assert [(r["split"], r["model"]) for r in rows] == [("eval", "dense_autoencoder")]


def test_phase3_configs_share_the_phase1_front_end_and_baseline():
    p1 = yaml.safe_load((CONFIGS / "phase1_baseline.yaml").read_text())
    cnn = yaml.safe_load((CONFIGS / "phase3_idcnn.yaml").read_text())
    ae = yaml.safe_load((CONFIGS / "phase3_baseline_eval.yaml").read_text())
    front_end = {k: v for k, v in p1["features"].items() if k != "frames"}
    assert cnn["features"] == front_end  # the dev train cache is reused
    assert ae["features"] == p1["features"] and ae["model"] == p1["model"]
    assert ae["train"] == p1["train"]
    assert ae["data"]["train_set"] == "additional"
    assert cnn["train"]["mixup_alpha"] == 0.0 and cnn["train"]["spec_mask"] is None


def test_mixup_config_changes_one_thing():
    base = yaml.safe_load((CONFIGS / "phase3_idcnn.yaml").read_text())
    mix = yaml.safe_load((CONFIGS / "phase3_idcnn_mixup.yaml").read_text())
    assert mix["train"].pop("mixup_alpha") > 0
    base["train"].pop("mixup_alpha")
    assert mix == base


def test_mask_config_adds_masking_to_mixup_and_changes_nothing_else():
    mix = yaml.safe_load((CONFIGS / "phase3_idcnn_mixup.yaml").read_text())
    mask = yaml.safe_load((CONFIGS / "phase3_idcnn_mixup_mask.yaml").read_text())
    assert mix["train"].pop("spec_mask") is None
    assert set(mask["train"].pop("spec_mask")) == {
        "freq_masks",
        "freq_width",
        "time_masks",
        "time_width",
    }
    assert mask == mix
