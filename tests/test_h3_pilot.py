import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from andon import h3_pilot
from andon.data.audio import write_wav
from andon.data.features import SplitFeatures
from andon.experiment import Config
from andon.record import MANIFEST_FIELDS

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "h3_pilot_s1.yaml"
FANS = ("id_00", "id_01")
FAULTS = ("imbalance", "obstruction")


def make_session(root: Path, n_normal: int = 8, n_fault: int = 3) -> None:
    """Two synthetic fans: each a tone of its own pitch in noise; a fault adds a second tone."""
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    rows = []
    (root / "s1").mkdir(parents=True)
    for k, fan in enumerate(FANS):
        counts = {"normal": 0, "anomaly": 0}
        for fault, n in [("none", n_normal)] + [(f, n_fault) for f in FAULTS]:
            condition = "normal" if fault == "none" else "anomaly"
            for i in range(n):
                x = 3000 * np.sin(2 * np.pi * (500 + 700 * k) * t) + rng.normal(0, 300, t.size)
                if fault != "none":
                    x += 6000 * np.sin(2 * np.pi * (2500 + 900 * FAULTS.index(fault)) * t)
                name = f"s1/{condition}_{fan}_{counts[condition]:08d}.wav"
                counts[condition] += 1
                write_wav(root / name, x.astype(np.int16), 16000)
                row = dict.fromkeys(MANIFEST_FIELDS, "")
                row.update(file=name, session="s1", fan_id=fan, condition=condition, fault=fault)
                row.update(take=f"s1-{fan}-{fault}", clip_in_take=str(i))
                rows.append(row)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def tiny_cfg(root: Path) -> Config:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["data"].update(root=str(root), train_clips=5)
    raw["features"]["n_mels"] = 16
    raw["model"].update(channels=[4, 8], patch_frames=8, score_hop=4)
    raw["train"].update(epochs=150, batch_size=8, learning_rate=0.01, mixup_alpha=0.0)
    raw["gmm"]["n_components"] = 1
    return Config("tiny_pilot", raw)


def test_pilot_config_copies_the_stage_1_settings():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cnn = yaml.safe_load((CONFIG.parent / "phase3_idcnn_mixup.yaml").read_text(encoding="utf-8"))
    gmm = yaml.safe_load((CONFIG.parent / "phase2_gmm_id.yaml").read_text(encoding="utf-8"))
    assert raw["features"] == cnn["features"]
    assert raw["model"] == cnn["model"] and raw["train"] == cnn["train"]
    assert raw["gmm"] == gmm["model"]


def test_split_trains_on_the_first_normal_clips_of_each_fan_only(tmp_path):
    make_session(tmp_path)
    clips = h3_pilot.read_session(tmp_path, "s1")
    train, test = h3_pilot.split_normals(clips, 5)
    assert sorted(train + test) == list(range(len(clips)))
    assert all(clips[i].fault == "none" and clips[i].order < 5 for i in train)
    assert len(train) == 5 * len(FANS)
    held_back = [clips[i] for i in test if clips[i].fault == "none"]
    assert sorted(c.order for c in held_back) == [5, 5, 6, 6, 7, 7]
    with pytest.raises(ValueError, match="need more than"):
        h3_pilot.split_normals(clips, 8)
    with pytest.raises(ValueError, match="no clips"):
        h3_pilot.read_session(tmp_path, "s2")


def test_subset_keeps_the_chosen_clips_in_order():
    feats = SplitFeatures(np.arange(12, dtype=np.float32).reshape(6, 2), np.array([0, 1, 3, 6]))
    sub = h3_pilot.subset(feats, [2, 0])
    assert sub.offsets.tolist() == [0, 3, 4]
    assert sub.frames[:, 0].tolist() == [6, 8, 10, 0]


def test_auc_pct_is_100_when_every_positive_scores_higher():
    assert h3_pilot.auc_pct(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 100.0
    assert h3_pilot.auc_pct(np.array([1.0]), np.array([1.0])) == 50.0


def test_pilot_end_to_end_finds_loud_synthetic_faults(tmp_path):
    make_session(tmp_path)
    cfg = tiny_cfg(tmp_path)
    clips = h3_pilot.read_session(cfg.data_root, "s1")
    feats = h3_pilot.load_or_extract(
        [c.clip for c in clips], cfg.logmel, tmp_path / "cache", "realfan_s1"
    )
    run = h3_pilot.run_seed(cfg, clips, feats, 0)
    for fan in FANS:
        assert run[(fan, "gmm all")] == 100.0  # a strong extra tone is far from normal
        assert run[(fan, "id_acc")] == 100.0  # the two pitches are easy to tell apart
        assert run[(fan, "outside none")] < run[(fan, "outside imbalance")]
    text = h3_pilot.report(cfg, clips, {0: run})
    assert "5 normal clips train, 3 normal clips and 3 clips of each fault" in text
    assert "| GMM per fan, obstruction | 100.00 ± nan | 100.00 ± nan |" in text


def test_control_clips_are_other_session_normals_at_the_same_gain_and_never_faults(tmp_path):
    make_session(tmp_path)
    with (tmp_path / "manifest.csv").open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        for name, fault, gain in [("a", "none", "4"), ("b", "none", "0"), ("c", "imbalance", "4")]:
            row = dict.fromkeys(MANIFEST_FIELDS, "")
            row.update(file=f"pilot/{name}.wav", session="pilot", fan_id="id_01")
            row.update(condition="normal", fault=fault, gain_shift=gain, clip_in_take="0")
            writer.writerow(row)
    control = h3_pilot.read_control(tmp_path, {"session": "pilot", "gain_shift": 4})
    assert [c.clip.path.name for c in control] == ["a.wav"]
    assert control[0].fault == h3_pilot.CONTROL and not control[0].clip.is_anomaly
    assert not h3_pilot.is_fault(h3_pilot.CONTROL) and h3_pilot.is_fault("imbalance")
    # A control clip is scored, never trained on.
    clips = h3_pilot.read_session(tmp_path, "s1") + control
    train, test = h3_pilot.split_normals(clips, 5)
    assert len(clips) - 1 in test and len(clips) - 1 not in train
