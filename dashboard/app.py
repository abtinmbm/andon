"""Live andon dashboard: replays the demo stream window by window and runs the light as it goes.

    uv run --group dashboard streamlit run dashboard/app.py

Needs the dev data and the phase3_idcnn_mixup checkpoints locally. The stream, limits
and drawing are those of ``andon.demo``, so this page and docs/andon_demo.gif match.
Unlike the GIF, the light here is live: after a red, "Reset light" clears it as an
operator would, and the replay carries on with the light judged from that point.

The page also plays the recording being scored, at the replay speed (the browser keeps
the pitch). The chart and the sound both follow one wall-clock :class:`ReplayClock`, so
a slow redraw skips ahead instead of falling behind the sound.
"""

from __future__ import annotations

import base64
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


@st.cache_resource(show_spinner="Loading the audio...")
def load_audio_b64() -> str:
    trace = load_trace()
    return base64.b64encode(demo.wav_bytes(demo.stream_audio(trace), trace.sample_rate)).decode()


def player_html(audio_b64: str, clock: demo.ReplayClock) -> str:
    """An audio player that starts where the clock says the stream is *when it can play*.

    Loading takes a moment; the start position is worked out from the wall clock at that
    point, not from when the page was sent, so the sound lands where the chart is.
    """
    return f"""<div id="msg" style="font: 13px sans-serif; color: #555">Sound: loading...</div>
<audio id="a" preload="auto" src="data:audio/wav;base64,{audio_b64}"></audio>
<script>
const a = document.getElementById("a"), msg = document.getElementById("msg");
const POS = {clock.pos_s}, SPEED = {clock.speed}, STARTED_MS = {clock.started_at * 1000};
const target = () => POS + (Date.now() - STARTED_MS) / 1000 * SPEED;
function sync() {{
  a.playbackRate = SPEED;
  a.preservesPitch = true;
  a.currentTime = Math.min(target(), a.duration);
}}
function go() {{
  sync();
  a.play()
    .then(() => {{ msg.textContent = "Sound on, " + SPEED + "x"; }})
    .catch(() => {{
      msg.innerHTML = '<button id="b">Click to turn the sound on</button>';
      document.getElementById("b").onclick = go;
    }});
}}
// Starting playback after the seek takes a little time too: correct once it is running.
a.addEventListener("playing", () => {{ if (Math.abs(a.currentTime - target()) > 0.25) sync(); }},
                   {{ once: true }});
a.addEventListener("loadedmetadata", go, {{ once: true }});
</script>"""


def restart(trace: demo.DemoTrace) -> None:
    st.session_state.n = 0
    st.session_state.lights = []
    st.session_state.light = AndonLight(trace.limits, trace.settings)
    st.session_state.clock = demo.ReplayClock()


def step(trace: demo.DemoTrace) -> None:
    """Feed the next window to the live light."""
    n = st.session_state.n
    st.session_state.lights.append(st.session_state.light.update(float(trace.scores[n])))
    st.session_state.n = n + 1


st.set_page_config(page_title="Andon", layout="wide")
st.title("Andon: machine-sound anomaly light")
trace = load_trace()
if "clock" not in st.session_state:
    restart(trace)
clock: demo.ReplayClock = st.session_state.clock
now = time.time()

window_s = float(trace.times_s[1] - trace.times_s[0])
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 2, 1])
speed = c4.slider("Replay speed (x real time)", 1, 16, 4)
sound_on = c5.toggle("Sound", value=True)
if c1.button("Play / pause", use_container_width=True):
    if clock.playing:
        clock.stop(now)
    else:
        if st.session_state.n >= len(trace.scores):
            restart(trace)  # play again from the start
            clock = st.session_state.clock
        clock.start(now, speed)
if c2.button("Reset light", use_container_width=True):
    st.session_state.light.reset()
if c3.button("Restart", use_container_width=True):
    was_playing = clock.playing
    restart(trace)
    clock = st.session_state.clock
    if was_playing:
        clock.start(now, speed)
if clock.playing and clock.speed != speed:
    clock.start(now, speed)  # keeps the position, changes the rate from here

st.caption(
    f"{trace.title}. About two minutes of normal running, then a fault from "
    f"{trace.fault_start_s:.0f} s. A new score every {window_s:.2f} s. Yellow needs "
    f"{trace.settings.warn_on} windows in a row above the warning limit; red needs "
    f"{trace.settings.red_hold} above the control limit and stays on until reset. "
    "Sound plays the recording being scored, sped up with the replay; above 4x some "
    "browsers (Firefox) mute it, Chrome and Edge play up to 16x."
)
sound = st.empty()
chart = st.empty()
fig = plt.figure(figsize=(10, 4), dpi=100)


def show() -> None:
    n = st.session_state.n
    # The live light, not a replay from green, so a reset shows up immediately.
    lights = st.session_state.lights[:-1] + [st.session_state.light.state] if n else []
    demo.draw(fig, trace, n, lights)
    chart.pyplot(fig, clear_figure=False)


show()
if clock.playing and sound_on:
    with sound:
        st.iframe(player_html(load_audio_b64(), clock), height=40)  # our own HTML only
while clock.playing:
    pos = clock.position(time.time())
    ready = min(demo.windows_ready(trace, pos), len(trace.scores))
    if st.session_state.n < ready:
        while st.session_state.n < ready:  # several at once if drawing fell behind
            step(trace)
        show()
    if st.session_state.n >= len(trace.scores):
        clock.stop(time.time())
        break
    next_s = float(trace.times_s[st.session_state.n])
    time.sleep(max(0.01, (next_s - pos) / clock.speed))
if st.session_state.n >= len(trace.scores) and st.session_state.light.state is Light.RED:
    st.info("End of the stream. The light stays red until someone resets it.")
