import numpy as np
import pytest

pytest.importorskip("matplotlib")  # the dashboard group: `uv sync --all-groups`

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from andon import demo  # noqa: E402
from andon.experiment import load_config  # noqa: E402
from andon.light import Light, LightSettings, Limits  # noqa: E402

G, Y, R = Light.GREEN, Light.YELLOW, Light.RED


def trace() -> demo.DemoTrace:
    # 10 quiet windows, then a fault from 10 s that sends the score far above the limits.
    scores = np.r_[np.full(10, 0.01), np.full(5, 5.0)]
    return demo.DemoTrace(
        title="test",
        times_s=np.arange(1, 16, dtype=float),
        scores=scores,
        limits=Limits(center=0.01, sigma=0.01, warn=0.03, control=0.04, log_scale=False),
        settings=LightSettings(warn_on=2, warn_off=3, red_hold=3),
        fault_start_s=10.0,
        fault_end_s=15.0,
    )


def test_demo_config_points_at_the_simulator_config():
    d = load_config("configs/phase5_demo.yaml").raw["demo"]
    sim = load_config(d["from_config"])
    assert sim.name == "phase5_andon"
    assert d["split"] == "dev_test"  # the demo never uses the evaluation set
    assert int(d["normal_clips"]) > 0 and float(d["gif_fps"]) > 0


def test_states_run_the_light_from_green():
    assert demo.states(trace()) == [G] * 10 + [G, Y, R, R, R]


def texts(fig) -> list[str]:
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def test_draw_shows_the_light_and_the_delay_to_red():
    t = trace()
    lights = demo.states(t)
    fig = plt.figure()
    demo.draw(fig, t, 0, lights)  # before the first window
    assert demo.LIGHT_TEXT[G] in texts(fig)
    demo.draw(fig, t, 12, lights)
    assert demo.LIGHT_TEXT[Y] in texts(fig)
    demo.draw(fig, t, 15, lights)
    assert demo.LIGHT_TEXT[R] in texts(fig)
    assert "red 3.0 s later" in texts(fig)  # red at 13 s, fault from 10 s
    plt.close(fig)


def test_draw_times_the_current_red_not_an_earlier_one():
    # As on the dashboard after a reset: an early red, then green, then red again at 14 s.
    t = trace()
    lights = [R, R] + [G] * 11 + [R, R]
    fig = plt.figure()
    demo.draw(fig, t, 15, lights)
    assert "red 4.0 s later" in texts(fig)
    plt.close(fig)


def test_render_gif_writes_one_frame_per_window_plus_a_hold(tmp_path):
    path = tmp_path / "demo.gif"
    n = demo.render_gif(trace(), path, fps=5, hold_s=1.0)
    assert n == 15 + 5
    with Image.open(path) as im:
        assert im.format == "GIF" and im.is_animated
