import hashlib
from pathlib import Path

import pytest
import torch
from test_idcnn import make_dataset, tiny_cfg, write_cfg

from andon import reproduce, score, train_idcnn
from andon.experiment import RESULTS_COLUMNS, load_config, read_results

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CPU = torch.device("cpu")


# --- released checkpoints --------------------------------------------------------------


def test_release_covers_every_type_and_seed_of_the_final_config():
    cfg = load_config(CONFIGS / "phase3_idcnn_mixup.yaml")
    wanted = {cfg.checkpoint_path(t, s).name for t in cfg.machine_types for s in (0, 1, 2)}
    assert set(reproduce.CHECKPOINT_SHA256) == wanted
    assert all(len(h) == 64 for h in reproduce.CHECKPOINT_SHA256.values())
    assert reproduce.DEFAULT_CONFIG.name == "phase3_idcnn_mixup.yaml"


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A fake release folder (served by file:// URL) holding one checkpoint."""
    monkeypatch.chdir(tmp_path)
    cfg = tiny_cfg(tmp_path / "data")
    name = cfg.checkpoint_path("fan", 0).name
    served = tmp_path / "release"
    served.mkdir()
    (served / name).write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setattr(reproduce, "CHECKPOINT_SHA256", {name: digest})
    return cfg, served.as_uri()


def test_missing_checkpoint_is_downloaded_and_checked(release):
    cfg, url = release
    reproduce.fetch_checkpoints(cfg, ["fan"], [0], base_url=url)
    assert cfg.checkpoint_path("fan", 0).read_bytes() == b"weights"
    reproduce.fetch_checkpoints(cfg, ["fan"], [0], base_url=url)  # present and right: no-op


def test_a_different_local_checkpoint_is_never_overwritten(release):
    cfg, url = release
    path = cfg.checkpoint_path("fan", 0)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"my own retrained weights")
    with pytest.raises(RuntimeError, match="differs from the released checkpoint"):
        reproduce.fetch_checkpoints(cfg, ["fan"], [0], base_url=url)
    assert path.read_bytes() == b"my own retrained weights"


def test_a_corrupt_download_is_rejected(release, monkeypatch):
    cfg, url = release
    name = cfg.checkpoint_path("fan", 0).name
    monkeypatch.setattr(reproduce, "CHECKPOINT_SHA256", {name: "0" * 64})
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        reproduce.fetch_checkpoints(cfg, ["fan"], [0], base_url=url)
    assert not cfg.checkpoint_path("fan", 0).exists()


def test_a_checkpoint_outside_the_release_needs_train(release):
    cfg, url = release
    with pytest.raises(ValueError, match="--train"):
        reproduce.fetch_checkpoints(cfg, ["fan"], [1], base_url=url)


# --- end to end on a tiny synthetic dataset --------------------------------------------


@pytest.fixture
def scored(tmp_path, monkeypatch):
    """A tiny model trained, checkpointed and scored once on eval into the results log."""
    monkeypatch.chdir(tmp_path)
    make_dataset(tmp_path / "data")
    cfg = write_cfg(tiny_cfg(tmp_path / "data"), tmp_path / "tiny_idcnn.yaml")
    log = tmp_path / "results" / "results_log.csv"
    train_idcnn.train_and_save(cfg, "fan", 0, CPU)
    score.run(cfg, [0], ["fan"], "eval", CPU, log=log)
    # Downloads are covered above; here the data and checkpoints are already local.
    monkeypatch.setattr(reproduce, "fetch_checkpoints", lambda *a, **k: None)
    monkeypatch.setattr(reproduce, "fetch_eval_data", lambda *a, **k: None)
    monkeypatch.setattr(reproduce, "fetch_training_data", lambda *a, **k: None)
    return cfg, log, tmp_path / "data" / "reproduced_eval.csv"


def test_rescoring_matches_the_log_and_appends_nothing(scored, capsys):
    cfg, log, out = scored
    before = log.read_text()
    assert reproduce.run(cfg, [0], ["fan"], CPU, reproduced=out, log=log)
    printed = capsys.readouterr().out
    assert "re-scored into" in printed and "from results_log.csv" not in printed
    assert log.read_text() == before  # the eval run is not repeated in the results log
    assert read_results(out)[0]["auc"] == read_results(log)[0]["auc"]

    # A second run starts its own log afresh instead of skipping the rows it wrote.
    assert reproduce.run(cfg, [0], ["fan"], CPU, reproduced=out, log=log)
    assert len(read_results(out)) == 1


def test_a_result_that_differs_from_the_log_fails(scored):
    cfg, log, out = scored
    rows = read_results(log)
    rows[0]["auc"] = f"{float(rows[0]['auc']) - 0.01:.4f}"
    text = ",".join(RESULTS_COLUMNS) + "\n"
    text += "".join(",".join(f'"{r[c]}"' for c in RESULTS_COLUMNS) + "\n" for r in rows)
    log.write_text(text)
    assert not reproduce.run(cfg, [0], ["fan"], CPU, reproduced=out, log=log)
    lines, worst = reproduce.compare(cfg, ["fan"], [0], out, log)
    assert "DIFFERS" in lines[0] and worst == pytest.approx(0.01, abs=1e-4)


def test_a_run_never_scored_on_eval_is_refused_before_anything_runs(scored, monkeypatch):
    cfg, log, out = scored
    calls = []
    for name in ("fetch_checkpoints", "fetch_eval_data", "fetch_training_data", "train_missing"):
        monkeypatch.setattr(reproduce, name, lambda *a, n=name, **k: calls.append(n))
    monkeypatch.setattr(score, "run", lambda *a, **k: calls.append("score"))
    with pytest.raises(ValueError, match="only re-checks a logged eval run"):
        reproduce.run(cfg, [0, 1], ["fan"], CPU, retrain=True, reproduced=out, log=log)
    assert calls == [] and not out.exists()


def test_only_the_id_cnn_can_be_rechecked(scored, tmp_path):
    _, log, out = scored
    cfg = load_config(CONFIGS / "phase1_baseline.yaml")
    with pytest.raises(ValueError, match="machine-ID CNN only"):
        reproduce.run(cfg, [0], ["fan"], CPU, reproduced=out, log=log)


def test_the_results_log_is_never_the_rescored_log(scored):
    cfg, log, _ = scored
    with pytest.raises(ValueError, match="must not go to the results log"):
        reproduce.run(cfg, [0], ["fan"], CPU, reproduced=log, log=log)
    assert len(read_results(log)) == 1


def test_train_retrains_only_missing_checkpoints_and_reproduces(scored):
    cfg, log, out = scored
    ckpt = cfg.checkpoint_path("fan", 0)
    ckpt.unlink()
    assert reproduce.run(cfg, [0], ["fan"], CPU, retrain=True, reproduced=out, log=log)
    assert ckpt.exists()  # retrained with the same seed, it scores the same on this data
    mtime = ckpt.stat().st_mtime_ns
    reproduce.train_missing(cfg, ["fan"], [0], CPU)
    assert ckpt.stat().st_mtime_ns == mtime


def test_headline_averages_types_within_each_seed(tmp_path):
    cfg = tiny_cfg(tmp_path)
    log = tmp_path / "log.csv"
    rows = [("fan", 0, 0.90), ("pump", 0, 0.80), ("fan", 1, 0.94), ("pump", 1, 0.86)]
    text = ",".join(RESULTS_COLUMNS) + "\n"
    for t, s, auc in rows:
        text += f"x,y,3,{cfg.name},id_cnn,{t},eval,{s},{auc},0.5,\n"
    log.write_text(text)
    mean, sd = reproduce.headline(cfg, log, ["fan", "pump"], [0, 1])
    assert mean == pytest.approx(87.5)  # seed means 85 and 90
    assert sd == pytest.approx(3.5355, abs=1e-4)
    mean, _ = reproduce.headline(cfg, log, ["fan", "pump"], [1])  # only the seeds asked for
    assert mean == pytest.approx(90.0)
    with pytest.raises(ValueError, match="no eval row"):
        reproduce.headline(cfg, log, ["fan", "pump"], [0, 2])
