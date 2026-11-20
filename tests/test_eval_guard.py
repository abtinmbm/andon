"""Eval is scored only for configs marked ``final: true``, by every command that can score it."""

from pathlib import Path

import pytest
import torch
from test_idcnn import tiny_cfg

from andon import classic, h3, h3_pilot, quantize, score, simulate
from andon.experiment import Config, load_config, read_results, require_final

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CPU = torch.device("cpu")

# Every config that has eval rows in the results log, and nothing else.
FINAL = {
    "phase2_gmm_id",
    "phase3_baseline_eval",
    "phase3_idcnn_mixup",
    "phase4_idcnn_int8",
    "phase5_andon",
    "h3_gmm",
    "h3_level",
    "h6_jidoka",
}
# Chosen on dev and marked final, but not yet scored on eval. Each moves to FINAL in the
# commit that logs its one eval run, so a final config can never stay unscored unnoticed.
PENDING: set[str] = set()


def test_only_the_configs_scored_on_eval_are_final():
    final = {p.stem for p in CONFIGS.glob("*.yaml") if load_config(p).raw.get("final") is True}
    assert final == FINAL | PENDING
    rows = read_results(ROOT / "results" / "results_log.csv")
    logged = {r["config"] for r in rows if r["split"] == "eval"}
    assert logged == FINAL


def test_require_final_refuses_only_eval_of_a_config_not_marked_final():
    cfg = Config("variant", {"phase": 3})
    require_final(cfg, "dev_test")
    require_final(Config("chosen", {"phase": 3, "final": True}), "eval")
    with pytest.raises(ValueError, match="final"):
        require_final(cfg, "eval")
    with pytest.raises(ValueError, match="final"):
        require_final(Config("typo", {"phase": 3, "final": "yes"}), "eval")  # only true counts


def test_score_refuses_eval_for_a_dev_only_config(tmp_path):
    log = tmp_path / "log.csv"
    with pytest.raises(ValueError, match="final"):
        score.run(tiny_cfg(tmp_path / "data"), [0], ["fan"], "eval", CPU, log=log)
    assert not log.exists()


def test_simulate_refuses_eval_for_a_dev_only_variant(tmp_path):
    cfg = load_config(CONFIGS / "phase5_andon_log.yaml")
    log = tmp_path / "log.csv"
    with pytest.raises(ValueError, match="final"):
        simulate.run(cfg, [0], ["fan"], "eval", CPU, log=log)
    assert not log.exists()


def test_classic_refuses_eval_for_a_dev_only_config(tmp_path):
    cfg = load_config(CONFIGS / "phase2_gmm_type.yaml")
    log = tmp_path / "log.csv"
    with pytest.raises(ValueError, match="final"):
        classic.run(cfg, [0], ["fan"], cache_dir=tmp_path / "cache", log=log, split="eval")
    assert not log.exists()


def test_quantize_refuses_eval_for_a_config_not_marked_final(tmp_path):
    cfg = load_config(CONFIGS / "phase4_gmm_id_int8.yaml")
    with pytest.raises(ValueError, match="final"):
        quantize.run(cfg, [0], ["fan"], "eval", log=tmp_path / "log.csv")


def test_h3_refuses_the_real_fan_eval_session_for_a_config_not_marked_final():
    recording = {"sessions": {"dev": ["d1", "d2"], "eval": ["d3"]}}
    for name in ("h3_gmm_centred", "h3_idcnn_mixup"):  # the H3 configs not chosen
        cfg = load_config(CONFIGS / f"{name}.yaml")
        fold = h3.folds(recording, "eval")[0]
        with pytest.raises(ValueError, match="final"):
            h3.fold_clips(cfg, recording, [], fold, "eval")


def test_the_h3_pilot_never_reads_the_real_fan_eval_session(tmp_path):
    (tmp_path / "manifest.csv").write_text("file,session\n", encoding="utf-8")
    recording = {"sessions": {"dev": ["d1", "d2"], "eval": ["d3"]}}
    with pytest.raises(ValueError, match="eval"):
        h3_pilot.read_session(tmp_path, "d3", recording)
    with pytest.raises(ValueError, match="no clips"):  # a dev session gets past the guard
        h3_pilot.read_session(tmp_path, "d1", recording)


def test_each_final_h3_config_was_scored_on_eval_exactly_once():
    rows = read_results(ROOT / "results" / "results_log.csv")
    seeds = {"h3_gmm": {"0", "1", "2"}, "h3_level": {"0"}}
    for name, want in seeds.items():
        ev = [r for r in rows if r["config"] == name and r["split"] == "eval"]
        notes = [h3.parse_notes(r["notes"]) for r in ev]
        assert {n["fold"] for n in notes} == {"d1+d2>d3"}
        assert {r["seed"] for r in ev} == want
        # Per seed: 3 fans x (2 faults + the control check) scoring rows, plus 3 light rows.
        keys = [(r["seed"], r["machine_type"], n["fault"]) for r, n in zip(ev, notes, strict=True)]
        assert len(keys) == len(set(keys)) == 12 * len(want)
