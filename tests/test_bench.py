from pathlib import Path

import pytest

from andon.bench import BUDGET_MS_PER_FRAME, bench_table, main, parse_log, phase4_estimate

ROOT = Path(__file__).resolve().parents[1]


def block(run: int, total: float, patch: float = 800.0, verdict: str = "PASS") -> str:
    e2e = "end to end: 3 patches, int8 input diffs 0 of 24576, worst score err 3.00e-07"
    return f"""== andon bench, run {run}, clk_sys 150000000 Hz
log-mel: 128 frames, worst |C - Python| 1.00e-05 dB (tol 1e-03), 1.500 ms/frame avg, 1.700 ms max
cnn: 4 patches, head sum mismatches 0, worst score err 2.00e-07 (tol 1e-05)
{e2e} (tol 1e-03), {patch:.2f} ms/patch avg, 801.00 ms max
real time: {total:.3f} ms of compute per 32 ms of audio (fits)
static RAM: 1000 bytes
self-test: {verdict}
"""


def test_parse_reads_every_field():
    (run,) = parse_log("# comment\n" + block(2, 26.5))
    assert run.run == 2 and run.clk_hz == 150_000_000
    assert run.logmel_ms_per_frame == 1.5 and run.logmel_worst_db == 1e-5
    assert run.head_mismatches == 0 and (run.input_diffs, run.input_values) == (0, 24576)
    assert run.worst_score_err == 3e-7 and run.cnn_ms_per_patch == 800.0
    assert run.ms_per_frame == 26.5 and run.frame_ms == 32 and run.ram_bytes == 1000
    assert run.passed
    assert run.cnn_ms_per_frame == pytest.approx(25.0)


def test_parse_drops_a_block_cut_off_by_the_capture():
    text = block(1, 26.5) + block(2, 26.5).split("static RAM")[0]
    assert [r.run for r in parse_log(text)] == [1]


def test_parse_reports_a_failed_self_test():
    (run,) = parse_log(block(1, 26.5, verdict="FAIL"))
    assert not run.passed


def test_table_flags_over_budget_and_compares_with_the_estimate():
    runs = parse_log(block(1, 30.0, patch=900.0) + block(2, 28.0, patch=700.0))
    text = bench_table(runs, est_ms_per_frame=3.0, est_ms_per_score=100.0)
    assert "| mean of 2 |" in text and "| 800.00 |" in text
    assert (
        f"29.00 ms of compute per 32 ms frame on one core, **over** the {BUDGET_MS_PER_FRAME:.0f}"
        in text
    )
    assert "measured 800.00 ms per score, 8.0 times the estimate" in text


def test_table_under_budget():
    text = bench_table(parse_log(block(1, 10.0)), est_ms_per_frame=3.0, est_ms_per_score=100.0)
    assert ", under the 20 ms budget" in text


def test_table_needs_at_least_one_run():
    with pytest.raises(ValueError):
        bench_table([], 3.0, 100.0)


def test_phase4_estimate_reads_the_named_row(tmp_path):
    table = tmp_path / "t.md"
    table.write_text(
        "| Model | Est. ms per frame | Est. ms per score |\n| --- | ---: | ---: |\n"
        "| `other` | 9.99 | 1.00 |\n| `phase4_idcnn_int8` | 3.27 | 98.32 |\n",
        encoding="utf-8",
    )
    assert phase4_estimate(table) == (3.27, 98.32)
    with pytest.raises(ValueError):
        phase4_estimate(table, "missing")


def test_committed_bench_log_parses_and_table_is_current(tmp_path):
    """The committed table must be what the generator makes from the committed log."""
    runs = parse_log((ROOT / "results" / "h5_bench_pico2.txt").read_text(encoding="utf-8"))
    assert [r.run for r in runs] == [5, 6, 7] and all(r.passed for r in runs)
    out = tmp_path / "bench.md"
    main(
        [
            str(ROOT / "results" / "h5_bench_pico2.txt"),
            "--out",
            str(out),
            "--estimate",
            str(ROOT / "results" / "phase4_mcu.md"),
        ]
    )
    committed = (ROOT / "results" / "h5_bench_pico2.md").read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8").split("-->", 1)[1] == committed.split("-->", 1)[1]
