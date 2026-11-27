"""Phase H6: the live view of a jidoka trial, read from a capture as it grows."""

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from test_jidoka import FAULT, Board, run_trial  # noqa: E402

from andon import live  # noqa: E402
from andon.light import Light  # noqa: E402


def test_a_finished_trial_reads_as_the_board_reported_it():
    v = live.read_capture(run_trial(Board()), FAULT)
    assert v.fan == "id_01" and v.stage == "stopped" and v.calib_windows == 3
    assert v.light is Light.RED and v.trip_reason == "score"
    assert v.limits is not None and v.limits.control == 13.0
    assert v.calibrating[:3] == [True] * 3 and not any(v.calibrating[3:])
    assert v.times_s[0] == 0.0 and v.switch_s < v.red_s < v.stop_s


def test_every_prefix_of_a_capture_reads_and_draws():
    """The page rereads a file that is still being written: any cut must work."""
    text = run_trial(Board())
    fig = plt.figure(figsize=(10, 4))
    stages = []
    for cut in range(0, len(text), max(1, len(text) // 40)):
        v = live.read_capture(text[:cut], FAULT)
        live.draw(fig, v)
        stages.append(v.stage)
    plt.close(fig)
    order = ["waiting", "calibrating", "watching", "switched", "stopped"]
    seen = [s for i, s in enumerate(stages) if i == 0 or s != stages[i - 1]]
    assert seen == [s for s in order if s in seen]  # never goes backwards
    assert {"calibrating", "watching", "switched"} <= set(seen)


def test_a_half_written_last_line_is_left_for_the_next_read():
    text = run_trial(Board())
    i = text.index("\ntrip ") + 1
    v = live.read_capture(text[: i + 10], FAULT)  # "trip t_us=" cut short
    assert v.trip_reason == "" and v.light is not Light.RED


def test_a_limits_line_after_the_stop_keeps_the_page_stopped():
    # Every status prints the limits again, the closing one too.
    text = run_trial(Board()) + (
        "limits t_us=99 fan=id_01 n=3 center=10.0000 sigma=1.0000 warn=12.0000 control=13.0000\n"
    )
    v = live.read_capture(text, FAULT)
    assert v.stage == "stopped" and v.limits.control == 13.0
