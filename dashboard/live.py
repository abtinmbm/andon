"""Live andon dashboard for the board (Phase H6): follows a jidoka trial as it runs.

    uv run --group dashboard streamlit run dashboard/live.py

Only one program can hold the serial port, so this page does not open it: it rereads the
capture file ``andon.jidoka run`` writes line by line (``results/h6_jidoka_<fan>.txt``) once
a second, and draws the board's own scores, limits and light. It works on a finished
capture too.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import streamlit as st  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from andon import jidoka, live  # noqa: E402
from andon.experiment import load_config  # noqa: E402

CONFIG = Path("configs/h6_jidoka.yaml")
CAPTURES = Path("results")
REFRESH_S = 1.0

st.set_page_config(page_title="Andon live", layout="wide")
st.title("Andon: live from the board")
settings = jidoka.TrialSettings.from_config(load_config(CONFIG))
captures = sorted(CAPTURES.glob("h6_jidoka_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
if not captures:
    st.info(
        "No capture yet. Start a trial (andon.jidoka run ... --out results/h6_jidoka_<fan>.txt)"
        " and this page follows it."
    )
    st.stop()
c1, c2 = st.columns([3, 1])
path = c1.selectbox("Capture (newest first)", captures, format_func=lambda p: p.name)
follow = c2.toggle("Follow live", value=True)

view = live.read_capture(path.read_text(encoding="utf-8"), settings.fault_pct)
st.caption(
    f"The board scores a window every 1.02 s. After a hold-off at each start it fits its "
    f"limits on the first {view.calib_windows} windows of normal running (H3's first 10 "
    f"minutes), then watches. The trial switches the fan from {settings.normal_pct} % to "
    f"{settings.fault_pct} % (underspeed) by a serial command; the board must stop it."
)
fig = plt.figure(figsize=(10, 4), dpi=100)
live.draw(fig, view)
st.pyplot(fig, clear_figure=True)
if view.stage == "stopped":
    st.success(
        f"Stopped by the board ({view.trip_reason}): red {view.red_s - view.switch_s:.2f} s and "
        f"fan off {view.stop_s - view.switch_s:.2f} s after the switch. Only the button "
        "restarts it."
    )
if follow and view.stage != "stopped":
    time.sleep(REFRESH_S)
    st.rerun()
