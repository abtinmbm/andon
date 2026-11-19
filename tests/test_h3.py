"""H3: the split rules on a synthetic manifest, and the scorers on tiny synthetic audio."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from andon import h3
from andon.data.audio import write_wav
from andon.data.dataset import Clip
from andon.data.features import SplitFeatures
from andon.experiment import Config, load_config
from andon.light import LightSettings, fit_limits
from andon.record import MANIFEST_FIELDS

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [f"h3_{n}" for n in ("level", "gmm", "gmm_centred", "idcnn_mixup")]
CPU = torch.device("cpu")

RECORDING = {
    "sessions": {"dev": ["d1", "d2"], "eval": ["d3"], "other": ["s1"]},
    "retired_faults": {"imbalance": {}},
    "skip_first_clips": {"underspeed": 3},
    "excluded_clips": {"d2/normal_id_00_00000001.wav": "a knock"},
    "excluded_takes": {"d2-id_01-none-T9": "wrong fan label"},
    "out_of_order_takes": {"d1-id_00-underspeed-T5": "remounted"},
}


def _take(session, fan, fault, time, n, start=0):
    cond = "normal" if fault == "none" else "anomaly"
    take = f"{session}-{fan}-{fault}-{time}"
    return [
        {
            "file": f"{session}/{cond}_{fan}_{start + i:08d}.wav",
            "session": session,
            "fan_id": fan,
            "condition": cond,
            "fault": fault,
            "take": take,
            "clip_in_take": str(i),
            "recorded_utc": f"{time}-{i:03d}",
        }
        for i in range(n)
    ]


def manifest() -> list[dict]:
    rows = []
    for s in ("d1", "d2", "d3", "s1"):
        for fan in ("id_00", "id_01"):
            # The control take is listed first: the main take is found by time, not position.
            rows += _take(s, fan, "none", "T8", 3, start=100)
            rows += _take(s, fan, "none", "T1", 10)
            rows += _take(s, fan, "underspeed", "T5", 6)
            rows += _take(s, fan, "obstruction", "T6", 4, start=50)
            rows += _take(s, fan, "imbalance", "T7", 2, start=60)
    rows += _take("d2", "id_01", "none", "T9", 3, start=200)  # an excluded third normal take
    return rows


def cfg(final: bool = False) -> Config:
    raw = {"phase": "H3", "data": {"root": "data/realfan", "limit_clips": 4}}
    if final:
        raw["final"] = True
    return Config("h3_test", raw)


def by_role(clips, role, session=None, fan="id_00", fault=None):
    return [
        c
        for c in clips
        if c.role == role
        and c.clip.machine_id == fan
        and (session is None or c.session == session)
        and (fault is None or c.fault == fault)
    ]


def test_dev_folds_test_each_dev_session_with_a_model_trained_on_the_other():
    folds = h3.folds(RECORDING, "dev_test")
    assert [(f.train, f.test) for f in folds] == [(("d2",), "d1"), (("d1",), "d2")]
    assert [f.name for f in folds] == ["d2>d1", "d1>d2"]
    assert [(f.train, f.test) for f in h3.folds(RECORDING, "eval")] == [(("d1", "d2"), "d3")]


def test_roles_follow_the_protocol():
    fold = h3.Fold(("d1",), "d2")
    clips = h3.fold_clips(cfg(), RECORDING, manifest(), fold, "dev_test")
    train = by_role(clips, "train")
    assert {c.session for c in train} == {"d1"}
    assert {c.fault for c in train} == {"none"}
    assert len(train) == 13  # main and control takes both train
    limits = by_role(clips, "limits", "d2")
    assert [c.order for c in limits] == [0, 2, 3]  # clip 1 is excluded
    assert {c.take for c in limits} == {"d2-id_00-none-T1"}
    assert [c.order for c in by_role(clips, "normal", "d2")] == list(range(4, 10))
    assert {c.take for c in by_role(clips, "control", "d2")} == {"d2-id_00-none-T8"}
    under = by_role(clips, "fault", "d2", fault="underspeed")
    assert [c.order for c in under] == [3, 4, 5]  # settling clips dropped
    assert not any(c.fault == "imbalance" for c in clips)  # retired
    assert {c.session for c in clips} == {"d1", "d2"}  # never s1 or d3


def test_no_clip_has_two_roles_and_no_fault_trains():
    for fold in h3.folds(RECORDING, "dev_test"):
        clips = h3.fold_clips(cfg(), RECORDING, manifest(), fold, "dev_test")
        files = [c.clip.path for c in clips]
        assert len(files) == len(set(files))
        assert all(not c.clip.is_anomaly for c in clips if c.role == "train")
        assert all(c.clip.is_anomaly == (c.role == "fault") for c in clips)


def test_an_excluded_take_does_not_count_as_a_normal_take():
    fold = h3.Fold(("d1",), "d2")
    clips = h3.fold_clips(cfg(), RECORDING, manifest(), fold, "dev_test")
    assert not any(c.take == "d2-id_01-none-T9" for c in clips)
    assert by_role(clips, "control", "d2", fan="id_01")


def test_out_of_order_takes_are_flagged_and_only_they_are():
    fold = h3.Fold(("d2",), "d1")
    clips = h3.fold_clips(cfg(), RECORDING, manifest(), fold, "dev_test")
    flagged = {(c.clip.machine_id, c.fault) for c in clips if c.flagged}
    assert flagged == {("id_00", "underspeed")}


def test_eval_needs_a_final_config_and_dev_never_reads_eval():
    rows = manifest()
    fold = h3.folds(RECORDING, "eval")[0]
    with pytest.raises(ValueError, match="final"):
        h3.fold_clips(cfg(), RECORDING, rows, fold, "eval")
    clips = h3.fold_clips(cfg(final=True), RECORDING, rows, fold, "eval")
    assert {c.session for c in clips if c.role != "train"} == {"d3"}
    assert {c.session for c in clips if c.role == "train"} == {"d1", "d2"}
    with pytest.raises(ValueError, match="not a dev session"):
        h3.fold_clips(cfg(final=True), RECORDING, rows, h3.Fold(("d1",), "d3"), "dev_test")
    with pytest.raises(ValueError, match="must be dev sessions"):
        h3.fold_clips(cfg(), RECORDING, rows, h3.Fold(("d3",), "d1"), "dev_test")
    with pytest.raises(ValueError, match="not an eval session"):
        h3.fold_clips(cfg(final=True), RECORDING, rows, h3.Fold(("d1",), "d2"), "eval")


def test_a_fault_clip_listed_as_normal_is_refused_for_training():
    rows = manifest()
    bad = next(r for r in rows if r["session"] == "d1" and r["fault"] == "none")
    bad["condition"] = "anomaly"
    with pytest.raises(ValueError, match="fault clip in a training set"):
        h3.fold_clips(cfg(), RECORDING, rows, h3.Fold(("d1",), "d2"), "dev_test")


def test_a_session_without_a_control_take_is_refused():
    rows = [r for r in manifest() if not r["take"].startswith("d2-id_00-none-T8")]
    with pytest.raises(ValueError, match="1 usable normal takes"):
        h3.fold_clips(cfg(), RECORDING, rows, h3.Fold(("d1",), "d2"), "dev_test")


def test_the_four_configs_share_split_features_window_and_light():
    shared = ("phase", "data", "features", "window", "limits", "light", "stream")
    raws = [load_config(ROOT / "configs" / f"{n}.yaml").raw for n in CONFIGS]
    for raw in raws[1:]:
        assert {k: raw[k] for k in shared} == {k: raws[0][k] for k in shared}
    assert raws[0]["phase"] == "H3"
    final = {n for n, r in zip(CONFIGS, raws, strict=True) if r.get("final") is True}
    assert final == {"h3_level", "h3_gmm"}  # the keep rule's choice and its reference
    cnn = raws[3]["model"]
    assert (cnn["patch_frames"], cnn["score_hop"]) == (64, 32) == tuple(raws[0]["window"].values())


def test_the_configs_copy_the_stage_1_settings():
    def raw(name):
        return load_config(ROOT / "configs" / f"{name}.yaml").raw

    for name in ("h3_gmm", "h3_gmm_centred"):
        m = dict(raw(name)["model"])
        m.pop("type")
        stage1 = dict(raw("phase2_gmm_id")["model"])
        stage1.pop("type")
        assert m == stage1
    s1 = raw("phase3_idcnn_mixup")
    h = raw("h3_idcnn_mixup")
    assert h["model"] == s1["model"] and h["train"] == s1["train"]
    assert h["features"] == s1["features"]
    p5 = raw("phase5_andon")
    assert h["limits"] == p5["limits"] and h["light"] == p5["light"]
    assert h["stream"]["warmup_clips"] == p5["stream"]["warmup_clips"]


def test_the_keep_rule_is_in_every_config_comment():
    texts = [(ROOT / "configs" / f"{n}.yaml").read_text(encoding="utf-8") for n in CONFIGS]
    rule = [t.split("phase: H3")[0].splitlines()[1:] for t in texts]  # after each first line
    assert all(r == rule[0] for r in rule) and "worst-cell AUC" in "\n".join(rule[0])


def test_the_recording_config_flags_only_real_takes():
    rec = yaml.safe_load((ROOT / "configs" / "h2_recording.yaml").read_text(encoding="utf-8"))
    flagged = rec["out_of_order_takes"]
    assert set(flagged) == {
        "d1-id_00-underspeed-20261117T021646Z",
        "d1-id_01-underspeed-20261117T020518Z",
    }
    assert all(t.split("-")[0] in rec["sessions"]["dev"] for t in flagged)  # never eval


# --- Scoring, on tiny synthetic audio ------------------------------------------------------


def test_window_means_and_levels_match_a_direct_computation():
    rng = np.random.default_rng(0)
    frames = rng.normal(-40, 5, (20, 3)).astype(np.float32)
    starts = np.array([0, 4, 12])
    means = h3.window_means(frames, starts, 8)
    levels = h3.window_levels(frames, starts, 8)
    for k, s in enumerate(starts):
        w = frames[s : s + 8].astype(np.float64)
        assert np.allclose(means[k], w.mean(axis=0))
        assert np.isclose(levels[k], 10 * np.log10((10 ** (w / 10)).mean()))


def _fan_clip(fan, role, take, fault="none", order=0):
    clip = Clip(Path(f"{take}_{order}.wav"), "realfan", "test", fan, fault != "none")
    return h3.FanClip(clip, "d1", take, fault, order, role)


def test_centres_use_the_own_take_in_training_and_the_limit_clips_at_test():
    clips = [
        _fan_clip("id_00", "train", "a"),
        _fan_clip("id_00", "train", "b"),
        _fan_clip("id_00", "limits", "m"),
        _fan_clip("id_00", "normal", "m", order=1),
        _fan_clip("id_00", "control", "c"),
        _fan_clip("id_00", "fault", "f", "obstruction"),
    ]
    values = [1.0, 2.0, 5.0, 7.0, 9.0, 11.0]
    feats = SplitFeatures(
        frames=np.repeat(np.array(values, dtype=np.float32), 4)[:, None],
        offsets=np.arange(0, 25, 4, dtype=np.int64),
    )
    assert h3.centres(clips, feats)[:, 0].tolist() == [1.0, 2.0, 5.0, 5.0, 5.0, 5.0]


def _row(fan, fault, seed, auc, flagged="0", fold="d1>d2"):
    notes = f"fold={fold} fault={fault} flagged={flagged}"
    return {"machine_type": fan, "seed": str(seed), "auc": str(auc), "notes": notes}


def test_worst_cell_averages_folds_then_seeds_and_skips_flagged_takes():
    rows = [
        _row("id_00", "underspeed", 0, 0.9),
        _row("id_00", "underspeed", 0, 0.1, flagged="1", fold="d2>d1"),  # not counted
        _row("id_00", "obstruction", 0, 0.8),
        _row("id_00", "obstruction", 0, 0.6, fold="d2>d1"),
        _row("id_00", "obstruction", 1, 0.5),
        _row("id_00", "obstruction", 1, 0.7, fold="d2>d1"),
        _row("id_00", "control", 0, 0.0),  # a check, never a cell
    ]
    cells = h3.cell_values(rows)
    assert cells[("id_00", "underspeed")] == {"0": 0.9}
    assert cells[("id_00", "obstruction")] == pytest.approx({"0": 0.7, "1": 0.6})
    value, cell = h3.worst_cell(rows)
    assert cell == ("id_00", "obstruction") and value == pytest.approx(0.65)


TONE = {"id_00": 500, "id_01": 1300}


def _make_sessions(root: Path) -> None:
    """Two fans, two dev sessions and one eval session, each in protocol order.

    Normal: a tone per fan in noise. Obstruction adds a loud second tone; underspeed plays
    the tone lower and quieter. The control take is a slightly louder normal (a remount).
    """
    rng = np.random.default_rng(1)
    t = np.arange(16000) / 16000
    rows = []
    for s, session in enumerate(("d1", "d2", "d3")):
        (root / session).mkdir(parents=True)
        for fan, f0 in TONE.items():
            counts = {"normal": 0, "anomaly": 0}
            plan = [("none", 8, 1.0), ("underspeed", 5, 1.0), ("obstruction", 3, 1.0)]
            plan.append(("none", 3, 1.1))  # the control take, recorded last
            for k, (fault, n, gain) in enumerate(plan):
                take = f"{session}-{fan}-{fault}-T{k}"
                cond = "normal" if fault == "none" else "anomaly"
                for i in range(n):
                    pitch, amp = (0.85 * f0, 0.4) if fault == "underspeed" else (f0, 1.0)
                    x = gain * amp * 3000 * np.sin(2 * np.pi * pitch * t)
                    x = x + rng.normal(0, 200, t.size)
                    if fault == "obstruction":
                        x += 5000 * np.sin(2 * np.pi * 3100 * t)
                    name = f"{session}/{cond}_{fan}_{counts[cond]:08d}.wav"
                    counts[cond] += 1
                    write_wav(root / name, x.astype(np.int16), 16000)
                    row = dict.fromkeys(MANIFEST_FIELDS, "")
                    row.update(file=name, session=session, fan_id=fan, condition=cond)
                    row.update(fault=fault, take=take, clip_in_take=str(i))
                    row.update(recorded_utc=f"2026-11-1{s}T0{k}:00:{i:02d}Z")
                    rows.append(row)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("realfan")
    _make_sessions(root)
    rec = {
        "sessions": {"dev": ["d1", "d2"], "eval": ["d3"]},
        "skip_first_clips": {"underspeed": 2},
        "out_of_order_takes": {"d1-id_01-underspeed-T1": "test"},
    }
    rec_path = root / "recording.yaml"
    rec_path.write_text(yaml.safe_dump(rec), encoding="utf-8")
    return root, rec_path


def tiny(name: str, root: Path, rec_path: Path) -> Config:
    raw = load_config(ROOT / "configs" / f"{name}.yaml").raw
    raw.pop("final", None)  # a tiny test variant is never the chosen model
    raw["data"].update(root=str(root), recording_config=str(rec_path), limit_clips=3)
    raw["features"]["n_mels"] = 16
    raw["window"] = {"frames": 8, "hop": 4}
    if raw["model"]["type"] == "id_cnn":
        raw["model"].update(channels=[4, 8], patch_frames=8, score_hop=4)
        raw["train"].update(epochs=40, batch_size=8, learning_rate=0.01, mixup_alpha=0.0)
    elif "n_components" in raw["model"]:
        raw["model"]["n_components"] = 1
    return Config(name, raw)


def _run(name, synthetic, tmp_path, seeds=(0,)):
    root, rec = synthetic
    cfg = tiny(name, root, rec)
    log = tmp_path / "log.csv"
    dirs = dict(cache_dir=tmp_path / "cache", scores_dir=tmp_path / "scores")
    rows = h3.run(cfg, list(seeds), "dev_test", CPU, log=log, checkpoint_dir=tmp_path, **dirs)
    return cfg, log, rows, dirs


@pytest.mark.parametrize("name", CONFIGS)
def test_every_candidate_logs_a_row_per_fold_fan_and_fault_and_saves_window_scores(
    name, synthetic, tmp_path
):
    cfg, log, rows, dirs = _run(name, synthetic, tmp_path)
    assert len(rows) == 2 * 2 * 3  # folds x fans x (two faults + the control check)
    assert {r["phase"] for r in rows} == {"H3"} and {r["split"] for r in rows} == {"dev_test"}
    notes = [h3.parse_notes(r["notes"]) for r in rows]
    assert {(n["fold"], n["fault"]) for n in notes} == {
        (f, k) for f in ("d2>d1", "d1>d2") for k in ("obstruction", "underspeed", "control")
    }
    flagged = {
        (r["machine_type"], n["fold"])
        for r, n in zip(rows, notes, strict=True)
        if n.get("flagged") == "1"
    }
    assert flagged == {("id_01", "d2>d1")}
    fault_rows = [n for n in notes if n["fault"] != "control"]
    assert {(n["n_fault"], n["n_normal"], n["n_control"]) for n in fault_rows} == {("3", "5", "3")}
    saved = np.load(dirs["scores_dir"] / f"{name}_dev_test_d1-to-d2_seed0.npz")
    assert "train" not in set(saved["role"][saved["owner"]])  # training clips are not scored
    assert np.isfinite(saved["scores"]).all()
    if name == "h3_idcnn_mixup":
        assert "outside_normal" in notes[-1] and "train_acc" in notes[-1]
        assert (tmp_path / f"{name}_dev_test_d1-to-d2_seed0.pt").exists()
    # A second run of the same seed logs nothing new.
    assert h3.run(cfg, [0], "dev_test", CPU, log=log, checkpoint_dir=tmp_path, **dirs) == []


def test_the_level_baseline_finds_the_synthetic_faults(synthetic, tmp_path):
    _, _, rows, _ = _run("h3_level", synthetic, tmp_path)
    for r in rows:
        if h3.parse_notes(r["notes"])["fault"] != "control":
            assert float(r["auc"]) > 0.9, r


def test_the_level_baseline_takes_one_seed_and_eval_needs_final(synthetic, tmp_path):
    root, rec = synthetic
    with pytest.raises(ValueError, match="no seed"):
        h3.run(tiny("h3_level", root, rec), [0, 1], "dev_test", CPU, log=tmp_path / "l.csv")
    with pytest.raises(ValueError, match="final"):
        h3.run(tiny("h3_gmm", root, rec), [0], "eval", CPU, log=tmp_path / "l.csv")
    assert not (tmp_path / "l.csv").exists()


def test_tables_come_from_the_log_and_name_the_worst_cell(synthetic, tmp_path):
    _, log, _, _ = _run("h3_gmm", synthetic, tmp_path, seeds=(0, 1))
    text = h3.tables(["h3_level", "h3_gmm"], "dev_test", log=log)
    assert "`h3_gmm`" in text and "`h3_level`" not in text  # no rows, no line
    value, (fan, fault) = h3.worst_cell(h3.read_results(log))
    assert f"{100 * value:.2f} ({fan} {fault})" in text
    assert "| yes |" in text  # the flagged take is shown
    with pytest.raises(ValueError, match="no eval rows"):
        h3.tables(["h3_gmm"], "eval", log=log)


# --- The andon light -------------------------------------------------------------------------


def test_seconds_to_red_counts_from_the_fault_and_resets_an_early_red():
    limits = fit_limits(np.array([0.0, 1.0, 0.0, 1.0]))  # control limit about 2.2
    settings = LightSettings(warn_on=1, warn_off=1, red_hold=2)
    ends = np.arange(1.0, 11.0)  # a window ends every second
    scores = np.array([9, 9, 0, 0, 0, 0, 9, 9, 9, 9], dtype=float)
    # Red at the 2nd window (t=2) is before the fault at t=5: reset. Red again at t=8.
    assert h3.seconds_to_red(scores, ends, 5.0, limits, settings) == 3.0
    assert h3.seconds_to_red(np.zeros(10), ends, 5.0, limits, settings) is None


def test_the_light_runs_on_saved_scores_and_logs_one_row_per_fold_and_fan(synthetic, tmp_path):
    cfg, log, _, dirs = _run("h3_level", synthetic, tmp_path)
    rows = h3.light(cfg, [0], "dev_test", log=log, **dirs)
    assert len(rows) == 2 * 2
    notes = [h3.parse_notes(r["notes"]) for r in rows]
    assert {n["fault"] for n in notes} == {"light"}
    for n in notes:
        assert n["delay_obstruction"] != "missed" and n["delay_underspeed"] != "missed"
        # 5 same-mount clips of 1 s, 3 control clips; each clip is 32 frames of 32 ms.
        assert (float(n["normal_s"]), float(n["control_s"])) == (5.12, 3.07)
    flagged = {
        (r["machine_type"], n["fold"])
        for r, n in zip(rows, notes, strict=True)
        if n["flagged_underspeed"] == "1"
    }
    assert flagged == {("id_01", "d2>d1")}
    # The light rows don't count as scoring rows, and vice versa.
    assert h3.light(cfg, [0], "dev_test", log=log, **dirs) == []
    assert h3.run(cfg, [0], "dev_test", CPU, log=log, checkpoint_dir=tmp_path, **dirs) == []
    text = h3.tables(["h3_level"], "dev_test", log=log)
    # 2 folds x 2 fans x 2 faults, less the flagged take.
    assert "## The andon light" in text and "7 of 7" in text
    assert "| `h3_level` | d1>d2 | id_00 |" in text


def test_the_light_needs_the_saved_scores_and_eval_needs_final(synthetic, tmp_path):
    root, rec = synthetic
    cfg = tiny("h3_level", root, rec)
    dirs = dict(cache_dir=tmp_path / "cache", scores_dir=tmp_path / "none")
    with pytest.raises(FileNotFoundError, match="run"):
        h3.light(cfg, [0], "dev_test", log=tmp_path / "l.csv", **dirs)
    with pytest.raises(ValueError, match="final"):
        h3.light(cfg, [0], "eval", log=tmp_path / "l.csv", **dirs)


# --- Known values ----------------------------------------------------------------------------


def test_fold_rows_gives_known_aucs_and_keeps_limit_clips_out_of_the_negatives():
    clips = [_fan_clip("id_00", "limits", "m", order=0)]  # scored very high: must not count
    clips += [_fan_clip("id_00", "normal", "m", order=k) for k in range(1, 5)]
    clips += [_fan_clip("id_00", "control", "c", order=k) for k in range(2)]
    clips += [_fan_clip("id_00", "fault", "f", "obstruction", order=k) for k in range(2)]
    score = np.array([100.0, 1, 2, 3, 4, 5, 5, 5, 5])
    fault, check = h3.fold_rows(clips, score)
    assert fault["fault"] == "obstruction" and check["fault"] == "control"
    assert fault["vs_same"] == 1.0 and fault["vs_control"] == 0.5
    # Against all 6 negatives: 8 wins over the normals and 4 ties with the control take.
    assert fault["auc"] == pytest.approx((8 + 0.5 * 4) / 12)
    assert (fault["n_fault"], fault["n_normal"], fault["n_control"]) == (2, 4, 2)
    assert check["auc"] == 1.0 and check["drift"] == 1.0  # later normals [3, 4] > [1, 2]


def _light_case(raise_normal: bool = False):
    cfg = load_config(ROOT / "configs" / "h3_level.yaml")
    cfg.raw["window"] = {"frames": 8, "hop": 4}  # 7 windows per 32-frame clip
    plan = [("limits", "none")] * 3 + [("normal", "none")] * 6 + [("control", "none")] * 2
    plan += [("fault", "obstruction")] * 2 + [("fault", "underspeed")] * 2
    n = len(plan)
    rng = np.random.default_rng(0)
    scores = []
    for i, (role, _fault) in enumerate(plan):
        if role == "limits":
            s = rng.normal(0, 1, 7)
        elif role == "fault":
            s = np.full(7, 50.0)
        else:
            s = np.zeros(7)
        if raise_normal and role == "normal" and i == 3:  # the first normal clip, not a warm-up
            s = np.full(7, 50.0)
        scores.append(s)
    order = [0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 0, 1, 0, 1]
    rows = h3.light_rows(
        cfg,
        np.array(["id_00"] * n),
        np.array([r for r, _ in plan]),
        np.array([f for _, f in plan]),
        np.array(order),
        np.zeros(n, dtype=bool),
        np.full(n, 32),
        np.repeat(np.arange(n), 7),
        np.concatenate(scores),
    )
    return rows[0]


def test_light_limits_come_from_the_limit_clips_only():
    base = _light_case()
    raised = _light_case(raise_normal=True)
    # 7 high windows: red at the 3rd, reset, red again at the 6th.
    assert base["fa_normal"] == 0 and raised["fa_normal"] == 2
    # The limits didn't move: the faults turn red at the same time and the control take is
    # unchanged. Red needs 3 windows over the limit: the 3rd ends at 16 frames x 32 ms.
    for key in ("delay_obstruction", "delay_underspeed", "fa_control"):
        assert raised[key] == base[key]
    assert base["delay_obstruction"] == f"{16 * 0.032:.2f}"
    assert base["normal_s"] == round(6 * 32 * 0.032, 2)


def _light_note(fan, fa_n, fa_c, d_obs, d_und, flag_und="0"):
    notes = (
        f"fold=d1>d2 fault=light fa_normal={fa_n} normal_s=1800 fa_control={fa_c} "
        f"control_s=180 delay_obstruction={d_obs} flagged_obstruction=0 "
        f"delay_underspeed={d_und} flagged_underspeed={flag_und}"
    )
    return {"machine_type": fan, "seed": "0", "notes": notes}


def test_the_light_summary_is_total_reds_over_total_time_and_skips_flagged_takes():
    rows = [
        _light_note("id_00", 1, 1, "4.10", "missed"),
        _light_note("id_01", 0, 0, "2.00", "1.00", flag_und="1"),  # flagged: not counted
    ]
    text = "\n".join(h3._light_lines({"cfg": rows}))
    # 1 red in 3600 s; 1 red in 360 s; caught 2 of the 3 unflagged takes, mean (4.10 + 2.00) / 2.
    assert "| `cfg` | 1.00 | 10.00 | 2 of 3 | 3.05 |" in text
    assert "missed in 1 of 1" in text and "(flagged)" in text
