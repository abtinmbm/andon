"""The H2 dataset checker and the tone check, on tiny synthetic recordings."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from andon import realfan, record
from andon.data.audio import write_wav

RATE = 16000


@pytest.fixture
def cfg(tmp_path):
    return {
        "data": {"root": str(tmp_path / "realfan")},
        "audio": {"sample_rate": RATE, "clip_seconds": 1, "gain_shift": 0},
        "recording": {"settle_seconds": 0, "speed_pct": 100},
        "sessions": {"dev": ["s1", "s2"], "eval": ["s3"], "other": ["pilot"]},
        "fans": {"id_00": "Sunon", "id_01": "ANVISION (A)"},
        "faults": {"imbalance": "tape", "rattle": "loose screw"},
    }


def _noise(seed: int, level: float = 1000.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, level, RATE).astype(np.int16)


noise = _noise  # replaced in one test to make a louder fault


def take(cfg, session, fan, fault, n, gaps=0, seed=0):
    t = record.Take(session, fan, fault, n)
    clips = [noise(seed + i) for i in range(n)]
    record.save_take(t, clips, cfg, "abc", f"{session}-{fan}-{fault}-{seed}", "2026-11-08")
    logs = Path(cfg["data"]["root"]) / session / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{session}-{fan}-{fault}-{seed}.txt").write_text(
        f"# andon.record x firmware abc clips={n} frames={n * 63} gaps={gaps} "
        f"frames_missing={gaps} corrupt=0 discarded_seconds={0.5 * gaps}\nack ...\n",
        encoding="utf-8",
    )


def test_a_clean_dataset_passes_with_counts_per_cell(cfg):
    take(cfg, "s1", "id_00", "none", 3)
    take(cfg, "s1", "id_00", "imbalance", 2, seed=10)
    take(cfg, "s3", "id_01", "none", 1, seed=20)
    text, problems = realfan.check(cfg, listen=1)
    assert problems == []
    assert "| s1 | dev | id_00 | none | 3 | 0.1 |" in text
    assert "| s1 | dev | id_00 | imbalance | 2 |" in text
    assert "| s3 | eval | id_01 | none | 1 |" in text
    assert "| s1 | 2 | 315 | 0 | 0 | 0 | 0.00 |" in text  # 5 clips x 63 frames, no breaks
    assert text.count("- [ ] `") == 3  # one clip to listen to per row
    assert "Problems: 0" in text


def test_stream_breaks_in_a_real_session_are_problems_but_not_in_the_pilot(cfg):
    take(cfg, "pilot", "id_00", "none", 1, gaps=2)
    assert realfan.check(cfg)[1] == []
    take(cfg, "s2", "id_00", "none", 1, gaps=1, seed=5)
    problems = realfan.check(cfg)[1]
    assert len(problems) == 1 and "s2: stream broke 1 times" in problems[0]


def test_clipping_wrong_length_strays_and_missing_files_are_reported(cfg, tmp_path):
    take(cfg, "s1", "id_00", "none", 3)
    root = tmp_path / "realfan"
    loud = noise(0)
    loud[100] = 32767
    write_wav(root / "s1" / "normal_id_00_00000000.wav", loud / 32768.0, RATE)
    write_wav(root / "s1" / "normal_id_00_00000001.wav", np.zeros(RATE - 1), RATE)
    (root / "s1" / "normal_id_00_00000002.wav").unlink()
    shutil.copy(
        root / "s1" / "normal_id_00_00000001.wav", root / "s1" / "normal_id_00_00000099.wav"
    )
    problems = realfan.check(cfg)[1]
    joined = "\n".join(problems)
    assert "normal_id_00_00000000.wav: 1 clipped samples" in joined
    assert "normal_id_00_00000001.wav: 16000 Hz, 1 ch, 15999 samples" in joined
    assert "normal_id_00_00000002.wav: missing" in joined
    assert "normal_id_00_00000099.wav: on disk but not in the manifest" in joined
    assert len(problems) == 4


def test_labels_that_contradict_each_other_are_reported(cfg, tmp_path):
    take(cfg, "s1", "id_00", "none", 1)
    manifest = tmp_path / "realfan" / "manifest.csv"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace(",normal,none,", ",normal,imbalance,"), encoding="utf-8")
    problems = realfan.check(cfg)[1]
    assert any("condition normal with fault imbalance" in p for p in problems)


def test_fault_takes_are_compared_band_by_band_with_normal(cfg, monkeypatch):
    take(cfg, "s1", "id_00", "none", 4)
    # A fault twice as loud as normal in amplitude is +6 dB in every band.
    monkeypatch.setattr(sys.modules[__name__], "noise", lambda seed: 2 * _noise(seed))
    take(cfg, "s1", "id_00", "imbalance", 4, seed=10)
    text = realfan.check(cfg)[0]
    row = next(line for line in text.splitlines() if line.startswith("| s1 | id_00 | imbalance |"))
    cells = row.strip("| ").split(" | ")
    diffs = [float(cell) for cell in cells[4:-1]]
    assert cells[-1].count(" to ") == 1  # the per-clip range of the biggest band
    assert len(diffs) == len(realfan.BAND_EDGES_HZ)
    assert all(abs(d - 6.02) < 1.0 for d in diffs)


def test_band_powers_are_referred_to_gain_zero():
    x = _noise(3)
    b0 = realfan.band_powers(x, RATE)
    b2 = realfan.band_powers((x.astype(np.int32) * 4).astype(np.int16), RATE, gain_shift=2)
    np.testing.assert_allclose(b2, b0, rtol=1e-9)


def test_a_retired_fault_is_accepted_in_the_pilot_only(cfg):
    take(cfg, "pilot", "id_00", "rattle", 1)
    take(cfg, "s1", "id_00", "rattle", 1, seed=5)
    cfg["retired_faults"] = {"rattle": cfg["faults"].pop("rattle")}
    problems = realfan.check(cfg)[1]
    assert len(problems) == 1
    assert "s1/anomaly_id_00_00000000.wav: unknown fault rattle" in problems[0]


def test_tone_check_measures_the_frequency():
    t = np.arange(10 * RATE) / RATE
    tone = (8000 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.int16)
    freq, prominence = realfan.dominant_frequency(tone, RATE)
    assert abs(freq - 1000.0) < 0.05
    assert prominence > 60
    # The same tone played back at the wrong rate (a wrong clock divider) is caught.
    freq_wrong, _ = realfan.dominant_frequency(tone, int(RATE * 1.02))
    assert abs(freq_wrong - 1020.0) < 0.1


def test_tone_report_passes_and_fails(tmp_path):
    t = np.arange(5 * RATE) / RATE
    good, bad = tmp_path / "good.wav", tmp_path / "bad.wav"
    write_wav(good, 0.25 * np.sin(2 * np.pi * 1000.0 * t), RATE)
    write_wav(bad, 0.25 * np.sin(2 * np.pi * 1010.0 * t), RATE)
    assert realfan.tone_report(good, 1000.0)[1]
    assert not realfan.tone_report(bad, 1000.0)[1]
