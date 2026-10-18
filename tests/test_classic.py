"""Phase 2: clip-mean features, the three classic detectors, and their runner."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from test_pipeline import make_tree, noise_wav

from andon.classic import run
from andon.data.features import SplitFeatures, clip_means
from andon.experiment import load_config, read_results
from andon.models.classic import build_detector, fit_and_score

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("phase2_*.yaml"))
MODEL_CFGS = {
    "mahalanobis": {"type": "mahalanobis", "fit_per": "machine_type"},
    "gmm": {
        "type": "gmm",
        "fit_per": "machine_type",
        "n_components": 2,
        "covariance_type": "full",
        "reg_covar": 0.01,
    },
    "isolation_forest": {
        "type": "isolation_forest",
        "fit_per": "machine_type",
        "n_estimators": 50,
        "max_samples": "auto",
    },
}


def blob(seed: int, n: int = 200, dim: int = 8, center: float = 0.0) -> np.ndarray:
    return center + np.random.default_rng(seed).standard_normal((n, dim))


# --- features ----------------------------------------------------------------------------


def test_clip_means_is_the_mean_of_each_clips_frames():
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal((n, 4)).astype(np.float32) for n in (7, 1, 9)]
    feats = SplitFeatures(np.concatenate(clips), np.r_[0, np.cumsum([7, 1, 9])])
    means = clip_means(feats)
    assert means.shape == (3, 4) and means.dtype == np.float64
    for i, c in enumerate(clips):
        np.testing.assert_allclose(means[i], c.astype(np.float64).mean(axis=0))


def test_clip_means_rejects_an_empty_clip():
    feats = SplitFeatures(np.zeros((3, 4), np.float32), np.array([0, 3, 3]))
    with pytest.raises(ValueError, match="no frames"):
        clip_means(feats)


# --- detectors ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", MODEL_CFGS)
def test_outliers_score_higher_than_normal_points(kind):
    detector = build_detector(MODEL_CFGS[kind], seed=0).fit(blob(0))
    normal = detector.score(blob(1, n=50))
    outliers = detector.score(blob(2, n=50, center=4.0))
    assert outliers.min() > np.median(normal)
    assert normal.shape == (50,)


@pytest.mark.parametrize("kind", MODEL_CFGS)
def test_same_seed_gives_the_same_scores(kind):
    def scores(seed):
        return build_detector(MODEL_CFGS[kind], seed).fit(blob(0)).score(blob(3, n=20))

    np.testing.assert_array_equal(scores(0), scores(0))


def test_isolation_forest_depends_on_the_seed():
    def scores(seed):
        return build_detector(MODEL_CFGS["isolation_forest"], seed).fit(blob(0)).score(blob(3))

    assert not np.array_equal(scores(0), scores(1))


def test_mahalanobis_matches_the_closed_form_on_a_known_gaussian():
    # Many samples from N(0, diag(1, 4)): shrinkage is negligible, so the squared distance
    # of (2, 2) is close to 2^2/1 + 2^2/4 = 5.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50_000, 2)) * np.array([1.0, 2.0])
    d = build_detector(MODEL_CFGS["mahalanobis"], 0).fit(x).score(np.array([[2.0, 2.0]]))
    assert d[0] == pytest.approx(5.0, rel=0.05)


def test_unknown_model_or_fit_per_fails_loudly():
    with pytest.raises(ValueError, match="unknown classic model"):
        build_detector({"type": "svm"}, 0)
    bad = {**MODEL_CFGS["mahalanobis"], "fit_per": "everything"}
    with pytest.raises(ValueError, match="fit_per"):
        fit_and_score(bad, 0, blob(0), ["id_00"] * 200, blob(1, n=5), ["id_00"] * 5)


def test_fit_per_machine_id_scores_each_clip_with_its_own_ids_detector():
    x_train = np.vstack([blob(0, center=0.0), blob(1, center=6.0)])
    train_ids = ["id_00"] * 200 + ["id_02"] * 200
    point = np.zeros((2, 8))  # at id_00's centre, far from id_02's
    per_id = {**MODEL_CFGS["mahalanobis"], "fit_per": "machine_id"}
    s = fit_and_score(per_id, 0, x_train, train_ids, point, ["id_00", "id_02"])
    assert s[0] < 20 < s[1]
    pooled = fit_and_score(MODEL_CFGS["mahalanobis"], 0, x_train, train_ids, point, ["id_00"] * 2)
    assert pooled[0] == pooled[1]  # pooled ignores the ID


def test_fit_per_machine_id_refuses_a_test_id_with_no_training_clips():
    per_id = {**MODEL_CFGS["mahalanobis"], "fit_per": "machine_id"}
    with pytest.raises(ValueError, match="id_04"):
        fit_and_score(per_id, 0, blob(0), ["id_00"] * 200, blob(1, n=2), ["id_00", "id_04"])


# --- configs and the runner --------------------------------------------------------------


def test_there_are_six_phase2_configs_sharing_phase1_features():
    phase1 = yaml.safe_load((CONFIGS[0].parent / "phase1_baseline.yaml").read_text())
    expected = {k: v for k, v in phase1["features"].items() if k != "frames"}
    assert len(CONFIGS) == 6
    for path in CONFIGS:
        cfg = load_config(path)
        assert cfg.phase == 2 and cfg.raw["features"] == expected  # so the cache is reused
        build_detector(cfg.raw["model"], 0)  # every config builds


def tiny_config(tmp_path: Path, source: Path) -> Path:
    raw = yaml.safe_load(source.read_text())
    raw["data"] = {"root": str(tmp_path / "data"), "machine_types": ["fan"]}
    path = tmp_path / source.name
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("source", CONFIGS, ids=lambda p: p.stem)
def test_run_logs_one_row_per_seed_and_skips_logged_rows(tmp_path, source):
    make_tree(tmp_path / "data", n_train=8)
    cfg = load_config(tiny_config(tmp_path, source))
    log = tmp_path / "results.csv"
    run(cfg, [0, 1], ["fan"], cache_dir=tmp_path / "cache", log=log)
    rows = read_results(log)
    assert [(r["config"], r["model"], r["seed"], r["split"]) for r in rows] == [
        (source.stem, cfg.model, "0", "dev_test"),
        (source.stem, cfg.model, "1", "dev_test"),
    ]
    assert all("id_00=" in r["notes"] and "id_02=" in r["notes"] for r in rows)
    assert all(f"fit_per={cfg.raw['model']['fit_per']}" in r["notes"] for r in rows)
    # Anomalies in the synthetic tree are 8x louder: any sane detector separates them.
    assert all(float(r["auc"]) == 1.0 for r in rows)
    run(cfg, [0, 1, 2], ["fan"], cache_dir=tmp_path / "cache", log=log)
    assert [r["seed"] for r in read_results(log)] == ["0", "1", "2"]  # only seed 2 was new


def test_run_fits_on_train_clip_means_and_only_scores_the_test_clips(tmp_path, monkeypatch):
    from andon import classic
    from andon.data.dataset import dev_test_clips, train_clips
    from andon.data.features import load_or_extract

    make_tree(tmp_path / "data", n_train=8)
    cfg = load_config(tiny_config(tmp_path, CONFIGS[0]))
    train = train_clips(cfg.data_root, "fan")
    test = dev_test_clips(cfg.data_root, "fan")
    seen = {}

    def spy(model_cfg, seed, x_train, train_ids, x_test, test_ids):
        seen.update(x_train=x_train, train_ids=train_ids, x_test=x_test, test_ids=test_ids)
        return np.arange(len(x_test), dtype=float)

    monkeypatch.setattr(classic, "fit_and_score", spy)
    classic.run(cfg, [0], ["fan"], cache_dir=tmp_path / "cache", log=tmp_path / "results.csv")
    expected_train = clip_means(load_or_extract(train, cfg.logmel, tmp_path / "cache", "x"))
    np.testing.assert_array_equal(seen["x_train"], expected_train)
    assert seen["train_ids"] == [c.machine_id for c in train]
    assert len(seen["x_test"]) == len(test) and seen["test_ids"] == [c.machine_id for c in test]


def test_run_refuses_an_anomalous_clip_in_the_training_folder(tmp_path):
    make_tree(tmp_path / "data", n_train=8)
    noise_wav(tmp_path / "data" / "dev_data" / "fan" / "train" / "anomaly_id_00_00000099.wav", 0)
    cfg = load_config(tiny_config(tmp_path, CONFIGS[0]))
    with pytest.raises(ValueError, match="anomalous clip"):
        run(cfg, [0], ["fan"], cache_dir=tmp_path / "cache", log=tmp_path / "results.csv")


# --- comparison table --------------------------------------------------------------------


def test_comparison_table_means_stds_bold_best_and_the_mean_row(tmp_path):
    from andon.data.dataset import MACHINE_TYPES
    from andon.evaluate import comparison_table
    from andon.experiment import append_result

    log = tmp_path / "results.csv"
    a = load_config(tiny_config(tmp_path, CONFIGS[0]))
    b = load_config(tiny_config(tmp_path, CONFIGS[1]))
    for seed in range(3):
        for i, t in enumerate(MACHINE_TYPES):
            append_result(a, t, "dev_test", seed, 0.60 + 0.01 * seed, 0.5, log=log)
            if not (t == "valve" and seed == 2):  # b is missing one run
                append_result(b, t, "dev_test", seed, 0.70 + 0.02 * i, 0.5, log=log)
    append_result(a, "fan", "eval", 0, 0.99, 0.99, log=log)  # eval rows never count

    table = comparison_table([a.name, b.name], "auc", log=log)
    lines = table.splitlines()
    assert lines[0] == f"| AUC (%) | {a.name} | {b.name} |"
    fan = next(line for line in lines if line.startswith("| fan |"))
    assert fan == "| fan | 61.00 +/- 1.00 | **74.00 +/- 0.00** |"
    mean = next(line for line in lines if line.startswith("| mean |"))
    assert mean == "| mean | **61.00 +/- 1.00** | n/a |"  # b lacks valve seed 2
    assert f"{a.name}: 3; {b.name}: 3" in table

    append_result(a, "fan", "dev_test", 0, 0.5, 0.5, log=log)
    with pytest.raises(ValueError, match="duplicate"):
        comparison_table([a.name], "auc", log=log)


def test_evaluate_compare_cli_writes_both_tables(tmp_path, monkeypatch, capsys):
    from andon.evaluate import main
    from andon.experiment import append_result

    monkeypatch.chdir(tmp_path)
    cfg = load_config(tiny_config(tmp_path, CONFIGS[0]))
    append_result(cfg, "fan", "dev_test", 0, 0.6, 0.55, log=Path("results/results_log.csv"))
    main(["--compare", cfg.name, "--out", "results/cmp.md"])
    printed = capsys.readouterr().out
    assert "| AUC (%) |" in printed and "| PAUC (%) |" in printed
    written = Path("results/cmp.md").read_text(encoding="utf-8")
    assert written.startswith("<!-- Generated by `uv run python -m andon.evaluate --compare")
    assert "| fan | **60.00 +/- nan** |" in written
