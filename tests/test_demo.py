import numpy as np
import pytest

pytest.importorskip("matplotlib")  # the dashboard group: `uv sync --all-groups`

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from andon import demo  # noqa: E402
from andon.data.audio import read_wav, write_wav  # noqa: E402
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


def test_windows_ready_counts_scores_available_by_a_time():
    t = trace()  # windows end at 1, 2, ..., 15 s
    assert [demo.windows_ready(t, x) for x in (0.0, 0.99, 1.0, 7.5, 15.0, 99.0)] == [
        0,
        0,
        1,
        7,
        15,
        15,
    ]


def test_replay_clock_follows_the_wall_clock_through_pause_and_speed_changes():
    c = demo.ReplayClock()
    assert not c.playing and c.position(50.0) == 0.0
    c.start(100.0, speed=4)
    assert c.playing and c.position(102.0) == 8.0
    c.stop(102.0)
    assert c.position(500.0) == 8.0  # paused: the stream stays put
    c.start(600.0, speed=4)
    c.start(601.0, speed=2)  # speed change: keeps the position reached at the old speed
    assert c.position(601.0) == 12.0 and c.position(603.0) == 16.0


def test_stream_audio_lines_up_with_the_frame_timeline(tmp_path):
    rng = np.random.default_rng(0)
    long, short = rng.uniform(-0.5, 0.5, 1000), rng.uniform(-0.5, 0.5, 450)
    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    write_wav(paths[0], long, 16000)
    write_wav(paths[1], short, 16000)
    t = demo.DemoTrace(**{**trace().__dict__, "clip_paths": tuple(paths), "clip_frames": (8, 5),
                          "frame_samples": 100})  # fmt: skip
    audio = demo.stream_audio(t)
    assert len(audio) == (8 + 5) * 100  # one frame of timeline = one hop of audio
    np.testing.assert_array_equal(audio[:800], read_wav(paths[0])[0][:800])  # trimmed
    np.testing.assert_array_equal(audio[800:1250], read_wav(paths[1])[0])
    assert not audio[1250:].any()  # padded with silence


def test_wav_bytes_is_a_playable_wav(tmp_path):
    samples = np.linspace(-0.5, 0.5, 320).astype(np.float32)
    path = tmp_path / "x.wav"
    path.write_bytes(demo.wav_bytes(samples, 16000))
    back, sr = read_wav(path)
    assert sr == 16000
    np.testing.assert_allclose(back, samples, atol=1 / 32768)
