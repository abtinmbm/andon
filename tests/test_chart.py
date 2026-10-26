import csv

import pytest

from andon.data.dataset import MACHINE_TYPES
from andon.evaluate import type_stats

COLUMNS = ["timestamp", "git_commit", "phase", "config", "model", "machine_type", "split",
           "seed", "auc", "pauc", "notes"]  # fmt: skip


def write_log(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for config, machine_type, seed, auc in rows:
            w.writerow({"config": config, "machine_type": machine_type, "split": "eval",
                        "seed": seed, "auc": auc, "pauc": auc})  # fmt: skip


def full_rows(config, auc_by_seed):
    return [(config, t, s, auc) for t in MACHINE_TYPES for s, auc in auc_by_seed.items()]


def test_type_stats_match_the_table_convention(tmp_path):
    log = tmp_path / "log.csv"
    # One type differs, so the mean of six is checked per seed, not per type.
    rows = full_rows("m", {0: 0.80, 1: 0.90})
    rows = [(c, t, s, 0.50 if t == "valve" and s == 1 else a) for c, t, s, a in rows]
    write_log(log, rows)
    stats = type_stats("m", "eval", log=log)
    assert stats["fan"] == pytest.approx((85.0, 7.0710678))
    assert stats["valve"] == pytest.approx((65.0, 21.2132034))
    per_seed = [80.0, (5 * 90 + 50) / 6]
    assert stats["mean"][0] == pytest.approx(sum(per_seed) / 2)


def test_type_stats_needs_every_type_for_every_seed(tmp_path):
    log = tmp_path / "log.csv"
    write_log(log, [r for r in full_rows("m", {0: 0.8, 1: 0.9}) if r[1:3] != ("pump", 1)])
    with pytest.raises(ValueError, match="same seeds"):
        type_stats("m", "eval", log=log)
    with pytest.raises(ValueError, match="same seeds"):
        type_stats("missing", "eval", log=log)


def test_draw_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")  # the dashboard group: `uv sync --all-groups`
    from PIL import Image

    from andon import chart

    log = tmp_path / "log.csv"
    write_log(log, full_rows("a", {0: 0.7, 1: 0.75}) + full_rows("b", {0: 0.9, 1: 0.92}))
    out = tmp_path / "sub" / "auc.png"
    chart.draw(out, models={"a": ("A", "#2a78d6"), "b": ("B", "#eb6834")}, log=log)
    with Image.open(out) as img:
        assert img.format == "PNG" and img.width > 500
