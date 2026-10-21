"""Phase 4: the quantize entry point, its float cross-check, the eval guard and the summary."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from test_idcnn import CONFIGS, make_dataset, tiny_cfg, write_cfg

from andon import budget, classic, quant, quantize, score, train, train_idcnn
from andon.data.dataset import MACHINE_TYPES
from andon.data.features import load_or_extract
from andon.experiment import Config, append_result, load_config, read_results

CPU = torch.device("cpu")


def int8_cfg(tmp_path: Path, name: str, source: Path, model_type: str, **extra) -> Config:
    raw = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    raw["data"] = {"root": str(tmp_path / "data"), "machine_types": ["fan"]}
    raw["model"] = {"type": model_type, "from_config": str(source)}
    raw.pop("final", None)  # a test opts in to eval explicitly
    raw.update(extra)
    return write_cfg(Config(f"tiny_{name}", raw), tmp_path / f"tiny_{name}.yaml")


def test_cnn_int8_run_logs_once_checks_the_float_model_and_guards_eval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_dataset(tmp_path / "data")
    src = write_cfg(tiny_cfg(tmp_path / "data"), tmp_path / "tiny_idcnn.yaml")
    log = tmp_path / "results" / "results_log.csv"
    train_idcnn.run(src, 0, ["fan"], CPU, log=log)  # the float model and its dev row
    cfg = int8_cfg(tmp_path, "phase4_idcnn_int8", tmp_path / "tiny_idcnn.yaml", "id_cnn_int8")

    quantize.run(cfg, [0], ["fan"], "dev_test", log=log)
    quantize.run(cfg, [0], ["fan"], "dev_test", log=log)  # logged: nothing new
    rows = read_results(log)
    assert [(r["config"], r["model"]) for r in rows] == [
        ("tiny_idcnn", "id_cnn"),
        ("tiny_phase4_idcnn_int8", "id_cnn_int8"),
    ]
    notes = quantize.parse_notes(rows[1]["notes"])
    assert notes["frames_per_decision"] == "4"  # the source's score_hop
    assert int(notes["int8_bytes"]) > 0 and float(notes["est_ms_per_frame"]) > 0
    assert "id_00" in notes and "id_02" in notes  # per-ID AUCs, as every row has
    assert float(notes["input_sat_anomaly_pct"]) > float(notes["input_sat_normal_pct"])

    # The float model must reproduce its logged row before anything int8 is scored.
    bad = tmp_path / "bad.csv"
    float_row = rows[0]
    auc = float(float_row["auc"]) - 0.01
    append_result(src, "fan", "dev_test", 0, auc, float(float_row["pauc"]), log=bad)
    with pytest.raises(ValueError, match="float model gives"):
        quantize.run(cfg, [0], ["fan"], "dev_test", log=bad)
    assert len(read_results(bad)) == 1

    # Eval only for a config marked final, and only once.
    score.run(src, [0], ["fan"], "eval", CPU, log=log)  # the float model's eval row
    with pytest.raises(ValueError, match="final"):
        quantize.run(cfg, [0], ["fan"], "eval", log=log)
    final = int8_cfg(
        tmp_path, "phase4_idcnn_int8", tmp_path / "tiny_idcnn.yaml", "id_cnn_int8", final=True
    )
    quantize.run(final, [0], ["fan"], "eval", log=log)
    quantize.run(final, [0], ["fan"], "eval", log=log)
    assert [r["split"] for r in read_results(log)].count("eval") == 2  # float and int8, once each


def test_int8_cnn_matches_float_on_normal_clips_and_saturates_on_loud_faults(tmp_path, monkeypatch):
    """The int8 input range comes from normal audio. The synthetic faults are a loud tone
    in other bands (about 36 normalised units against a calibrated top of about 6), so
    their input saturates and their score drops; normal clips score as in float."""
    monkeypatch.chdir(tmp_path)
    make_dataset(tmp_path / "data")
    src = write_cfg(tiny_cfg(tmp_path / "data", epochs=15, learning_rate=0.01), tmp_path / "s.yaml")
    train_idcnn.run(src, 0, ["fan"], CPU, log=tmp_path / "log.csv")
    model, ckpt = train_idcnn.load_checkpoint(src.checkpoint_path("fan", 0), src, CPU)
    train_feats, _, classes = train_idcnn.training_data(src, "fan", Path("data/cache"))
    clips, cache_name = score.split_clips(src, "fan", "dev_test")
    feats = load_or_extract(clips, src.logmel, Path("data/cache"), cache_name)
    rng = np.random.default_rng(0)
    idx = np.arange(train_feats.n_clips)
    calib = train_idcnn.random_patches(
        torch.from_numpy(train_feats.frames), train_feats.offsets, idx, 8, rng
    )
    qmodel = quant.quantize_idcnn(model, calib)
    true_ids = np.array([classes.index(c.machine_id) for c in clips])

    float_scores = train_idcnn.score_clips(src, model, classes, clips, feats, CPU)
    int8_scores, sat = quantize.cnn_clip_scores(qmodel, feats, true_ids, 8, 4)
    anomaly = np.array([c.is_anomaly for c in clips])
    np.testing.assert_allclose(int8_scores[~anomaly], float_scores[~anomaly], atol=0.05)
    assert sat[~anomaly].max() < 0.01 < sat[anomaly].min()
    assert (int8_scores[anomaly] < float_scores[anomaly] - 1).all()


def test_autoencoder_and_gmm_int8_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_dataset(tmp_path / "data")
    log = tmp_path / "results" / "results_log.csv"

    raw = yaml.safe_load((CONFIGS / "phase1_baseline.yaml").read_text())
    raw["data"] = {"root": str(tmp_path / "data"), "machine_types": ["fan"]}
    raw["train"].update(epochs=2, batch_size=8)
    ae = write_cfg(Config("tiny_ae", raw), tmp_path / "tiny_ae.yaml")
    train.run(ae, 0, ["fan"], CPU)  # logs its dev row to results/results_log.csv
    cfg = int8_cfg(
        tmp_path, "phase4_baseline_int8", tmp_path / "tiny_ae.yaml", "dense_autoencoder_int8"
    )
    quantize.run(cfg, [0], ["fan"], "dev_test", log=log)

    raw = yaml.safe_load((CONFIGS / "phase2_gmm_id.yaml").read_text())
    raw["data"] = {"root": str(tmp_path / "data"), "machine_types": ["fan"]}
    raw["model"]["n_components"] = 1  # 4 training clips per ID
    gmm = write_cfg(Config("tiny_gmm", raw), tmp_path / "tiny_gmm.yaml")
    classic.run(gmm, [0], ["fan"], log=log)
    cfg = int8_cfg(tmp_path, "phase4_gmm_id_int8", tmp_path / "tiny_gmm.yaml", "gmm_int8")
    quantize.run(cfg, [0], ["fan"], "dev_test", log=log)

    rows = {r["config"]: quantize.parse_notes(r["notes"]) for r in read_results(log)}
    ae_notes = rows["tiny_phase4_baseline_int8"]
    assert ae_notes["frames_per_decision"] == "1"
    assert int(ae_notes["int8_bytes"]) > budget.BUDGET_BYTES  # the real autoencoder's size
    gmm_notes = rows["tiny_phase4_gmm_id_int8"]
    assert gmm_notes["ids"] == "2" and int(gmm_notes["frames_per_decision"]) > 1


def test_phase4_configs_point_at_the_logged_float_models():
    for name, source, model, final in [
        ("phase4_idcnn_int8", "phase3_idcnn_mixup", "id_cnn_int8", True),
        ("phase4_baseline_int8", "phase1_baseline", "dense_autoencoder_int8", False),
        ("phase4_gmm_id_int8", "phase2_gmm_id", "gmm_int8", False),
    ]:
        cfg = load_config(CONFIGS / f"{name}.yaml")
        assert quantize.source_config(cfg).name == source
        assert cfg.model == model and cfg.phase == 4
        # Only the model chosen on dev test may be scored on eval.
        assert (cfg.raw.get("final") is True) == final
        assert cfg.machine_types == list(MACHINE_TYPES)


def _log_pair(log: Path, src: Config, cfg: Config, shift: float) -> None:
    """Six types x seeds 0-2 for a float config, and its int8 twin ``shift`` lower."""
    notes = (
        "params=100 int8_bytes=2048 macs_per_decision=750000 frames_per_decision=32 "
        "est_ms_per_frame=0.50 est_ms_per_decision=10.00"
    )
    for k, t in enumerate(MACHINE_TYPES):
        for s in range(3):
            auc = 0.8 + 0.01 * k + 0.001 * s
            append_result(src, t, "dev_test", s, auc, auc - 0.1, log=log)
            append_result(cfg, t, "dev_test", s, auc - shift, auc - 0.1, notes, log=log)


def test_budget_table_pairs_each_int8_model_with_its_float_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    src = write_cfg(
        Config("float_model", {"phase": 3, "model": {"type": "id_cnn"}}),
        tmp_path / "configs" / "float_model.yaml",
    )
    raw = {"phase": 4, "model": {"type": "id_cnn_int8", "from_config": "configs/float_model.yaml"}}
    cfg = write_cfg(Config("int8_model", raw), tmp_path / "configs" / "int8_model.yaml")
    log = tmp_path / "log.csv"
    _log_pair(log, src, cfg, shift=0.002)

    table = quantize.budget_table(["int8_model"], log=log)
    row = next(line for line in table.splitlines() if line.startswith("| `int8_model`"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    # params, float32 KB, int8 KB, MACs, frames, ms/frame, ms/score, fits
    assert cells[2:10] == ["100", "0.4", "2.0", "750,000", "32", "0.50", "10.00", "yes"]
    mean = 100 * np.mean([0.8 + 0.01 * k + 0.001 for k in range(6)])  # seed 1 is the middle
    assert cells[10] == f"{mean:.2f} +/- 0.10"
    assert cells[11] == f"{mean - 0.2:.2f} +/- 0.10"
    assert cells[12] == "-0.20 +/- 0.00"

    full = quantize.summary(["int8_model"], ["dev_test"], log=log)
    assert "| AUC (%) | float_model | int8_model |" in full
    with pytest.raises(ValueError, match="no eval rows"):
        quantize.summary(["int8_model"], ["dev_test", "eval"], log=log)


def test_budget_table_refuses_partial_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    src = write_cfg(
        Config("float_model", {"phase": 3, "model": {"type": "id_cnn"}}),
        tmp_path / "configs" / "float_model.yaml",
    )
    raw = {"phase": 4, "model": {"type": "id_cnn_int8", "from_config": "configs/float_model.yaml"}}
    cfg = write_cfg(Config("int8_model", raw), tmp_path / "configs" / "int8_model.yaml")
    log = tmp_path / "log.csv"
    _log_pair(log, src, cfg, shift=0.0)
    append_result(src, "fan", "dev_test", 3, 0.9, 0.8, log=log)  # a seed the int8 model lacks
    with pytest.raises(ValueError, match="same seeds"):
        quantize.budget_table(["int8_model"], log=log)
