"""Eval is scored only for configs marked ``final: true``, by every command that can score it."""

from pathlib import Path

import pytest
import torch
from test_idcnn import tiny_cfg

from andon import classic, quantize, score, simulate
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
}


def test_only_the_configs_scored_on_eval_are_final():
    final = {p.stem for p in CONFIGS.glob("*.yaml") if load_config(p).raw.get("final") is True}
    assert final == FINAL
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
