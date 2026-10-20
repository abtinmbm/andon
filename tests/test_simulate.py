from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from andon import simulate
from andon.data.dataset import MACHINE_TYPES, Clip
from andon.data.features import SplitFeatures
from andon.experiment import Config, append_result, load_config
from andon.light import LightSettings, Limits
from andon.models.idcnn import IdCNN

LIN = Limits(center=0.0, sigma=1.0, warn=2.0, control=3.0, log_scale=False)
HOLD2 = LightSettings(warn_on=2, warn_off=3, red_hold=2)


def clip(name: str, anomaly: bool = False, mid: str = "id_00") -> Clip:
    return Clip(Path(name), "fan", "test", mid, anomaly)


class Loudness(nn.Module):
    """Stub classifier: the louder the patch, the less sure it is of ID 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(1, 2))
        return torch.stack([torch.zeros_like(m), 4.0 * m - 4.0], dim=1)


def test_phase5_config_parses():
    cfg = load_config("configs/phase5_andon.yaml")
    st = simulate.sim_settings(cfg, simulate.source_config(cfg))
    assert (st.width, st.hop, st.warmup_clips, st.fault_clips) == (64, 32, 2, 3)
    assert st.frame_s == pytest.approx(0.032)
    assert st.light == LightSettings(warn_on=2, warn_off=3, red_hold=3)
    assert (st.warn_sigma, st.control_sigma, st.log_scale) == (2.0, 3.0, False)


def test_fault_episodes_group_anomalies_in_name_order_and_drop_leftovers():
    clips = [clip(f"anomaly_{k}.wav", True) for k in (4, 1, 3, 0, 2)] + [clip("normal_0.wav")]
    faults = simulate.fault_episodes(clips, list(range(6)), fault_clips=2)
    assert [[clips[i].path.name for i in f] for f in faults] == [
        ["anomaly_0.wav", "anomaly_1.wav"],
        ["anomaly_2.wav", "anomaly_3.wav"],
    ]
    with pytest.raises(ValueError):
        simulate.fault_episodes(clips, list(range(6)), fault_clips=6)


def test_calibration_split_alternates_normals_in_name_order():
    clips = [clip("normal_03.wav"), clip("anomaly_00.wav", True), clip("normal_01.wav")]
    clips += [clip("normal_02.wav"), clip("normal_00.wav")]
    calib, sim = simulate.calibration_split(clips, list(range(5)))
    assert [clips[i].path.name for i in calib] == ["normal_00.wav", "normal_02.wav"]
    assert [clips[i].path.name for i in sim] == ["normal_01.wav", "normal_03.wav"]


def test_calibration_split_needs_four_normals():
    clips = [clip(f"normal_{k}.wav") for k in range(3)]
    with pytest.raises(ValueError):
        simulate.calibration_split(clips, [0, 1, 2])


def test_stream_frames_and_window_ends():
    offsets = np.array([0, 3, 5, 9])
    assert simulate.stream_frames(offsets, [2, 0]).tolist() == [5, 6, 7, 8, 0, 1, 2]
    assert simulate.window_ends(10, 4, 3).tolist() == [4, 7, 10]
    assert simulate.window_ends(9, 4, 3).tolist() == [4, 7]  # no window past the end
    with pytest.raises(ValueError):
        simulate.window_ends(3, 4, 1)


def test_score_streams_matches_scoring_each_window_directly():
    torch.manual_seed(0)
    model = IdCNN(3, 8, (4, 8)).eval()
    frames = torch.randn(40, 8)
    streams = [np.arange(0, 20), np.r_[np.arange(30, 40), np.arange(0, 12)]]
    got = simulate.score_streams(model, frames, streams, true_id=1, width=8, hop=4)
    for s, scores in zip(streams, got, strict=True):
        ends = simulate.window_ends(len(s), 8, 4)
        x = torch.stack([frames[torch.from_numpy(s[e - 8 : e])] for e in ends])
        want = -torch.log_softmax(model(x), dim=1)[:, 1]
        np.testing.assert_allclose(scores, want.detach().numpy(), rtol=1e-5, atol=1e-6)


def test_count_false_alarms_counts_each_red_once_and_resets():
    scores = [0, 4, 4, 4, 4, 0, 4, 0, 4, 4]
    # Reds at windows 2, 4 and 9, reset after each; the lone 4 at window 6 is not enough.
    assert simulate.count_false_alarms(np.array(scores), LIN, HOLD2) == 3


def test_frames_to_red_measures_from_onset_to_window_end():
    scores = np.array([0, 0, 4, 4, 4])
    ends = np.array([10, 20, 30, 40, 50])
    assert simulate.frames_to_red(scores, ends, onset=25, limits=LIN, settings=HOLD2) == 15


def test_frames_to_red_ignores_a_red_before_the_fault():
    scores = np.array([4, 4, 0, 0, 4, 4])
    ends = np.array([10, 20, 30, 40, 50, 60])
    # Red at end 20 is before onset 35: reset, then the fault turns it red at end 60.
    assert simulate.frames_to_red(scores, ends, onset=35, limits=LIN, settings=HOLD2) == 25


def test_frames_to_red_returns_none_when_missed():
    scores, ends = np.zeros(5), np.arange(1, 6) * 10
    assert simulate.frames_to_red(scores, ends, onset=0, limits=LIN, settings=HOLD2) is None


def synthetic_split(n_normal: int, n_anom: int, length: int = 40):
    """Quiet normal clips and loud anomalous ones, as ``(clips, feats)``."""
    rng = np.random.default_rng(0)
    clips = [clip(f"normal_{k:02d}.wav") for k in range(n_normal)]
    clips += [clip(f"anomaly_{k:02d}.wav", True) for k in range(n_anom)]
    per = [
        rng.normal(3.0 if c.is_anomaly else 0.0, 0.3, size=(length, 4)).astype(np.float32)
        for c in clips
    ]
    offsets = np.arange(len(clips) + 1, dtype=np.int64) * length
    return clips, SplitFeatures(frames=np.concatenate(per), offsets=offsets)


ST = simulate.SimSettings(
    width=8, hop=4, frame_s=0.032, warmup_clips=2, fault_clips=1, warn_sigma=2.0,
    control_sigma=3.0, log_scale=False, light=LightSettings(warn_on=2, warn_off=3, red_hold=3),
)  # fmt: skip


def test_simulate_id_catches_loud_faults_without_false_alarms():
    clips, feats = synthetic_split(n_normal=10, n_anom=3)
    frames = torch.from_numpy(feats.frames)
    r, limits, sim = simulate.simulate_id(Loudness(), frames, feats, clips, list(range(13)), 0, ST)
    assert len(sim) == 5 and all(not clips[i].is_anomaly for i in sim)
    assert r.false_alarms == 0
    assert r.normal_s == pytest.approx(5 * 40 * 0.032)
    assert r.detected == 3
    # Red needs 3 windows above the limit; the window ending 4 frames into the fault
    # already counts, so the earliest red ends 4 + 2 * hop = 12 frames after the onset.
    assert all(d == pytest.approx(12 * 0.032) for d in r.delays_s)
    assert limits.control > limits.warn > limits.center


def test_simulate_id_keeps_calibration_clips_out_of_every_judged_stream(monkeypatch):
    clips, feats = synthetic_split(n_normal=10, n_anom=3)
    calib, _ = simulate.calibration_split(clips, list(range(13)))
    # Calibration clips get one constant level, so the fitted centre shows what it was fitted on.
    marked = SplitFeatures(frames=feats.frames.copy(), offsets=feats.offsets)
    for i in calib:
        marked.frames[i * 40 : (i + 1) * 40] = 0.5
    seen = []

    def spy(model, frames, streams, *args):
        seen.extend(streams)
        return real(model, frames, streams, *args)

    real = simulate.score_streams
    monkeypatch.setattr(simulate, "score_streams", spy)
    frames = torch.from_numpy(marked.frames)
    _, limits, sim = simulate.simulate_id(Loudness(), frames, marked, clips, list(range(13)), 0, ST)

    calib_frames = set(simulate.stream_frames(marked.offsets, calib).tolist())
    assert seen[0].tolist() == sorted(calib_frames)  # the limits' stream is calibration only
    assert limits.center == pytest.approx(np.log1p(np.exp(4 * 0.5 - 4)))  # Loudness at 0.5
    assert limits.sigma == pytest.approx(0.0, abs=1e-6)
    assert not set(sim) & set(calib)
    for s in seen[1:]:  # false-alarm stream, then warmup + fault streams
        assert not calib_frames & set(s.tolist())


def test_simulate_id_plays_a_fault_as_consecutive_anomaly_clips():
    clips, feats = synthetic_split(n_normal=10, n_anom=5)
    # Only the second clip of each two-clip fault is loud: the red comes one clip later.
    quiet = SplitFeatures(frames=feats.frames.copy(), offsets=feats.offsets)
    for k in (10, 12, 14):  # anomaly_00, _02, _04
        quiet.frames[k * 40 : (k + 1) * 40] = 0.0
    st = simulate.SimSettings(**{**ST.__dict__, "fault_clips": 2})
    frames = torch.from_numpy(quiet.frames)
    r, _, _ = simulate.simulate_id(Loudness(), frames, quiet, clips, list(range(15)), 0, st)
    assert len(r.delays_s) == 2  # anomaly_04 is left over
    assert all(d == pytest.approx((40 + 12) * 0.032) for d in r.delays_s)


def test_simulate_id_reports_misses_as_none():
    clips, feats = synthetic_split(n_normal=10, n_anom=2)
    quiet = SplitFeatures(frames=feats.frames.copy(), offsets=feats.offsets)
    quiet.frames[10 * 40 :] = 0.0  # the "anomalies" now sound like normal running
    frames = torch.from_numpy(quiet.frames)
    r, _, _ = simulate.simulate_id(Loudness(), frames, quiet, clips, list(range(12)), 0, ST)
    assert r.delays_s == [None, None] and r.detected == 0


def test_type_notes_round_trip_through_parse_notes():
    results = [
        simulate.IdResult("id_00", 1, 1800.0, [2.0, None, 4.0]),
        simulate.IdResult("id_02", 0, 1800.0, [3.0]),
    ]
    notes = simulate.type_notes(results)
    n = simulate.parse_notes(notes)
    assert n["fa"] == "1" and n["normal_h"] == "1.0000" and n["fa_per_h"] == "1.00"
    assert n["detected"] == "3/4" and n["delay_mean_s"] == "3.00" and n["delay_median_s"] == "3.00"
    assert "id_00:fa=1,det=2/3" in notes and "id_00:fa" not in n


def log_rows(log: Path, seeds: list[int], split: str = "dev_test") -> None:
    cfg = Config("phase5_test", {"phase": 5, "model": {"type": "id_cnn"}})
    for t in MACHINE_TYPES:
        for s in seeds:
            fa = 1.0 + s  # seed 0: 1/h, seed 1: 2/h
            notes = (
                f"fa=1 normal_h=0.5000 fa_per_h={fa:.2f} detected={8 + s}/10 "
                f"delay_mean_s={3.0 + s:.2f} delay_median_s=3.00 id_00:fa=1,det=8/10"
            )
            append_result(cfg, t, split, s, 0.9, 0.8, notes, log=log)


def test_summary_table_means_and_spreads_over_seeds(tmp_path):
    log = tmp_path / "log.csv"
    log_rows(log, [0, 1])
    table = simulate.summary_table("phase5_test", "dev_test", log=log)
    assert "seeds 0, 1" in table
    # fa/h 1 and 2, caught 80% and 90%, delay 3 and 4 s: sample std 0.71, 7.07, 0.71.
    assert "| fan | 1.50 ± 0.71 | 85.00 ± 7.07 | 3.50 ± 0.71 | 0.50 | 10 |" in table
    assert "| **Mean of six types** | 1.50 ± 0.71 | 85.00 ± 7.07 | 3.50 ± 0.71 | | |" in table
    assert "no delay to average" not in table


def test_summaries_skip_splits_a_config_was_never_run_on(tmp_path):
    log = tmp_path / "log.csv"
    log_rows(log, [0, 1])
    log_rows(log, [0, 1], split="eval")
    tables = simulate.summaries(["phase5_test", "phase5_other"], ["dev_test", "eval"], log=log)
    assert [t.split(" on ")[1].split(",")[0] for t in tables] == ["dev_test", "eval"]
    with pytest.raises(ValueError, match="no rows"):
        simulate.summaries(["phase5_other"], ["dev_test"], log=log)


def test_summary_table_explains_nan_delays(tmp_path):
    log = tmp_path / "log.csv"
    cfg = Config("phase5_test", {"phase": 5, "model": {"type": "id_cnn"}})
    for t in MACHINE_TYPES:
        for s in (0, 1):
            d = "nan" if t == "valve" else "3.00"
            notes = f"fa=0 normal_h=0.5000 fa_per_h=0.00 detected=0/10 delay_mean_s={d}"
            append_result(cfg, t, "dev_test", s, 0.9, 0.8, notes, log=log)
    table = simulate.summary_table("phase5_test", "dev_test", log=log)
    assert "| valve | 0.00 ± 0.00 | 0.00 ± 0.00 | nan ± nan |" in table
    assert "no delay to average" in table


def test_summary_table_refuses_partial_runs(tmp_path):
    log = tmp_path / "log.csv"
    log_rows(log, [0, 1])
    cfg = Config("phase5_test", {"phase": 5, "model": {"type": "id_cnn"}})
    append_result(cfg, "fan", "dev_test", 2, 0.9, 0.8, "fa=0 normal_h=1 fa_per_h=0", log=log)
    with pytest.raises(ValueError, match="same seeds"):
        simulate.summary_table("phase5_test", "dev_test", log=log)
    with pytest.raises(ValueError, match="same seeds"):
        simulate.summary_table("phase5_test", "eval", log=log)
