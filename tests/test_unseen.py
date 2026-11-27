"""H7's unseen fault: the d4 scorer on tiny synthetic audio, and its once-only guards."""

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from andon import unseen
from andon.data.audio import write_wav
from andon.experiment import Config, load_config, read_results
from andon.record import MANIFEST_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def _sessions(root: Path, fault_kind: str, extra_fan: bool = False) -> None:
    """d1 and d2: normal running of two fans. d4: id_00's fault take, then its control take.

    ``shape`` adds a low second tone (the spectrum changes shape); ``loud`` plays normal
    running 2x louder, noise and all (only the level changes, as a remount might).
    """
    rng = np.random.default_rng(3)
    t = np.arange(16000) / 16000
    rows = []
    plan = [("d1", fan, "none", 8, 1.0) for fan in ("id_00", "id_01")]
    plan += [("d2", fan, "none", 8, 1.0) for fan in ("id_00", "id_01")]
    plan += [("d4", "id_00", "imbalance_large", 6, 2.0 if fault_kind == "loud" else 1.0)]
    plan += [("d4", "id_00", "none", 6, 1.0)]
    if extra_fan:
        plan += [("d4", "id_01", "none", 2, 1.0)]
    counts: dict = {}
    for k, (session, fan, fault, n, gain) in enumerate(plan):
        (root / session).mkdir(parents=True, exist_ok=True)
        cond = "normal" if fault == "none" else "anomaly"
        take = f"{session}-{fan}-{fault}-T{k}"
        f0 = 500 if fan == "id_00" else 1300
        for i in range(n):
            x = gain * (0.05 * np.sin(2 * np.pi * f0 * t) + rng.normal(0, 0.003, t.size))
            if fault != "none" and fault_kind == "shape":
                x += 0.05 * np.sin(2 * np.pi * 150 * t)
            key = (session, cond)
            counts[key] = counts.get(key, 0) + 1
            name = f"{session}/{cond}_{fan}_{counts[key]:08d}.wav"
            write_wav(root / name, x, 16000)
            row = dict.fromkeys(MANIFEST_FIELDS, "")
            row.update(file=name, session=session, fan_id=fan, condition=cond)
            row.update(fault=fault, take=take, clip_in_take=str(i))
            row.update(recorded_utc=f"2026-11-2{k % 10}T00:00:{i:02d}Z")
            rows.append(row)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


RECORDING = {
    "sessions": {"dev": ["d1", "d2"], "eval": ["d3"], "unseen": ["d4"]},
    "unseen_faults": {"imbalance_large": {"sessions": ["d4"]}},
}


def _setup(tmp_path: Path, fault_kind: str = "shape", recording=None, extra_fan=False):
    root = tmp_path / "realfan"
    _sessions(root, fault_kind, extra_fan)
    rec = root / "recording.yaml"
    rec.write_text(yaml.safe_dump(recording or RECORDING), encoding="utf-8")
    return root, rec


def tiny(name: str, root: Path, rec: Path, final: bool = True) -> Config:
    raw = load_config(ROOT / "configs" / f"{name}.yaml").raw
    raw["final"] = final
    raw["data"].update(root=str(root), recording_config=str(rec))
    raw["features"]["n_mels"] = 16
    raw["window"] = {"frames": 8, "hop": 4}
    raw["model"]["n_components"] = 1
    return Config(name, raw)


def _run(tmp_path, name, fault_kind="shape", seeds=(0, 1, 2)):
    root, rec = _setup(tmp_path, fault_kind)
    log = tmp_path / "log.csv"
    rows = unseen.run(tiny(name, root, rec), list(seeds), log=log, cache_dir=tmp_path / "c")
    return rows, log


@pytest.mark.parametrize("name", unseen.CONFIGS)
def test_a_fault_that_changes_the_shape_is_found_raw_and_at_matched_volume(tmp_path, name):
    rows, log = _run(tmp_path, name)
    assert [r["seed"] for r in rows] == ["0", "1", "2"]
    assert all(r["split"] == "eval" and r["machine_type"] == "id_00" for r in rows)
    for r in rows:
        n = dict(p.split("=", 1) for p in r["notes"].split())
        assert float(r["auc"]) == 1.0
        assert float(n["det_at_fa"]) == 1.0
        assert float(n["fa"]) <= 0.01
        assert (n["n_fault"], n["n_control"], n["session"]) == ("6", "6", "d4")
    assert len(read_results(log)) == 3


def test_a_louder_fan_is_found_raw_but_not_at_matched_volume(tmp_path):
    raw, _ = _run(tmp_path / "raw", "h7_unseen", "loud", seeds=(0,))
    matched, _ = _run(tmp_path / "lm", "h7_unseen_levelmatched", "loud", seeds=(0,))
    assert float(raw[0]["auc"]) == 1.0
    assert float(matched[0]["auc"]) < 0.8  # only the level differed, and it was taken out


def test_it_scores_once_and_only_a_final_config(tmp_path):
    root, rec = _setup(tmp_path)
    log = tmp_path / "log.csv"
    with pytest.raises(ValueError, match="final"):
        unseen.run(tiny("h7_unseen", root, rec, final=False), [0], log=log)
    assert not log.exists()
    unseen.run(tiny("h7_unseen", root, rec), [0], log=log, cache_dir=tmp_path / "c")
    with pytest.raises(ValueError, match="already logged"):
        unseen.run(tiny("h7_unseen", root, rec), [1], log=log, cache_dir=tmp_path / "c")
    assert len(read_results(log)) == 1


def test_the_unseen_session_must_be_unseen_and_hold_one_fan(tmp_path):
    as_dev = {**RECORDING, "sessions": {"dev": ["d1", "d2", "d4"], "eval": [], "unseen": ["d4"]}}
    root, rec = _setup(tmp_path / "a", recording=as_dev)
    with pytest.raises(ValueError, match="dev or eval"):
        unseen.run(tiny("h7_unseen", root, rec), [0], log=tmp_path / "a.csv")
    root, rec = _setup(tmp_path / "b", extra_fan=True)
    with pytest.raises(ValueError, match="id_00 only"):
        unseen.run(tiny("h7_unseen", root, rec), [0], log=tmp_path / "b.csv")
    assert not (tmp_path / "a.csv").exists() and not (tmp_path / "b.csv").exists()


def test_detection_at_allows_at_most_the_rate_of_normals_above():
    normal = np.arange(200.0)  # 1 % of 200 is 2: the threshold is the third highest, 197
    fault = np.array([196.0, 197.0, 198.0, 250.0])
    det, fa, thr = unseen.detection_at(fault, normal, 0.01)
    assert (det, fa, thr) == (0.5, 0.01, 197.0)
    assert unseen.detection_at(fault, normal[:50], 0.01)[2] == 49.0  # under 1 window: the max
    with pytest.raises(ValueError, match="finite"):
        unseen.detection_at(np.array([np.nan]), normal, 0.01)


def test_the_table_comes_from_the_log(tmp_path):
    _, log = _run(tmp_path, "h7_unseen")
    text = unseen.table(["h7_unseen"], log=log)
    assert "| h7_unseen | id_00 | imbalance_large | 0, 1, 2 | 100.00 ± 0.00 |" in text


def test_the_configs_are_h3_gmm_unchanged_and_final():
    gmm = load_config(ROOT / "configs" / "h3_gmm.yaml").raw
    for name in unseen.CONFIGS:
        raw = load_config(ROOT / "configs" / f"{name}.yaml").raw
        assert raw["final"] is True and raw["phase"] == "H7"
        assert raw["features"] == gmm["features"] and raw["window"] == gmm["window"]
        model = dict(raw["model"])
        assert model.pop("level_matched", False) == (name == "h7_unseen_levelmatched")
        assert model == gmm["model"]
        assert {k: raw["data"][k] for k in ("root", "recording_config")} == {
            k: gmm["data"][k] for k in ("root", "recording_config")
        }
        assert raw["unseen"] == {
            "session": "d4",
            "fan": "id_00",
            "fault": "imbalance_large",
            "false_alarm_rate": 0.01,
        }
