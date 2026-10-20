"""Live andon dashboard: replays the demo stream window by window and runs the light as it goes.

    uv run --group dashboard streamlit run dashboard/app.py

Needs the dev data and the phase3_idcnn_mixup checkpoints locally. The stream, limits
and drawing are those of ``andon.demo``, so this page and docs/andon_demo.gif match.
Unlike the GIF, the light here is live: after a red, "Reset light" clears it as an
operator would, and the replay carries on with the light judged from that point.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from andon import demo  # noqa: E402
from andon.experiment import load_config  # noqa: E402
from andon.light import AndonLight, Light  # noqa: E402

CONFIG = Path("configs/phase5_demo.yaml")


@st.cache_resource(show_spinner="Scoring the demo stream...")
def load_trace() -> demo.DemoTrace:
    cfg = load_config(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return demo.build_trace(cfg, int(cfg.raw["demo"]["seed"]), device)


def restart(trace: demo.DemoTrace) -> None:
    st.session_state.n = 0
    st.session_state.lights = []
    st.session_state.light = AndonLight(trace.limits, trace.settings)


def step(trace: demo.DemoTrace) -> None:
    """Feed the next window to the live light."""
    n = st.session_state.n
    st.session_state.lights.append(st.session_state.light.update(float(trace.scores[n])))
    st.session_state.n = n + 1


st.set_page_config(page_title="Andon", layout="wide")
st.title("Andon: machine-sound anomaly light")
trace = load_trace()
if "n" not in st.session_state:
    restart(trace)
    st.session_state.playing = False

window_s = float(trace.times_s[1] - trace.times_s[0])
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
if c1.button("Play / pause", use_container_width=True):
    st.session_state.playing = not st.session_state.playing
if c2.button("Reset light", use_container_width=True):
    st.session_state.light.reset()
if c3.button("Restart", use_container_width=True):
    restart(trace)
speed = c4.slider("Replay speed (x real time)", 1, 16, 4)

st.caption(
    f"{trace.title}. About two minutes of normal running, then a fault from "
    f"{trace.fault_start_s:.0f} s. A new score every {window_s:.2f} s. Yellow needs "
    f"{trace.settings.warn_on} windows in a row above the warning limit; red needs "
    f"{trace.settings.red_hold} above the control limit and stays on until reset."
)
chart = st.empty()
fig = plt.figure(figsize=(10, 4), dpi=100)


def show() -> None:
    n = st.session_state.n
    # The live light, not a replay from green, so a reset shows up immediately.
    lights = st.session_state.lights[:-1] + [st.session_state.light.state] if n else []
    demo.draw(fig, trace, n, lights)
    chart.pyplot(fig, clear_figure=False)


show()
while st.session_state.playing and st.session_state.n < len(trace.scores):
    step(trace)
    show()
    time.sleep(window_s / speed)
if st.session_state.n >= len(trace.scores):
    st.session_state.playing = False
    if st.session_state.light.state is Light.RED:
        st.info("End of the stream. The light stays red until someone resets it.")
