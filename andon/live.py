"""The live view of a jidoka trial (Phase H6): the board's scores, limits and light, read from
the capture ``andon.jidoka run`` writes line by line, so a page can follow a trial as it runs
(``dashboard/live.py``) or redraw a finished one. Everything shown comes from the board's
own lines; nothing is recomputed on the PC.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from andon import jidoka
from andon.demo import LIGHT_COLOURS, LIGHT_TEXT, plt_circle
from andon.light import Light, Limits

_LIGHTS = {s.value: s for s in Light}
# The status reply's detector line (main.c say_detector): calibration windows so far / needed.
_DETECTOR = re.compile(r"^detector watch=\S+ .* calib=(\d+)/(\d+)")


@dataclass
class LiveView:
    fan: str = ""
    stage: str = "waiting"  # waiting, calibrating, watching, switched, stopped
    t0_us: int = -1  # the first score's board time; every time below is from it
    times_s: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    calibrating: list[bool] = field(default_factory=list)  # the score fitted the limits
    light: Light = Light.GREEN
    limits: Limits | None = None
    switch_s: float = math.nan
    red_s: float = math.nan
    stop_s: float = math.nan
    trip_reason: str = ""
    calib_windows: int = -1  # how many the board calibrates on, from its status reply

    def seconds(self, t_us: int) -> float:
        return (t_us - self.t0_us) / 1e6 if self.t0_us >= 0 else 0.0


def read_capture(text: str, fault_pct: int) -> LiveView:
    """What the capture so far shows; a partial line at the end (still being written) is
    left for the next read."""
    v = LiveView()
    lines = text.split("\n")
    if lines and not text.endswith("\n"):
        lines = lines[:-1]
    for line in lines:
        if m := jidoka._ACK.match(line):
            cmd = m[1]
            if cmd.startswith("watch ") and m[2] == "ok":
                v.fan = cmd.split()[1]
            elif cmd == f"fan speed {fault_pct}" and v.stage == "watching" and m[2] == "ok":
                v.switch_s, v.stage = v.seconds(int(m[3])), "switched"
            v.light = _LIGHTS.get(m[8], v.light)
        elif m := jidoka._SCORE.match(line):
            if v.t0_us < 0:
                v.t0_us = int(m[1])
            v.times_s.append(v.seconds(int(m[1])))
            v.scores.append(float(m[5]))
            v.calibrating.append(m[4] == "calibrating")
            v.light = _LIGHTS.get(m[6], v.light)
            if v.stage == "waiting":
                v.stage = "calibrating"
        elif m := jidoka._LIMITS.match(line):
            v.limits = Limits(float(m[4]), float(m[5]), float(m[6]), float(m[7]), False)
            v.stage = "watching"
        elif m := jidoka._TRIP.match(line):
            v.red_s, v.trip_reason, v.light = v.seconds(int(m[1])), m[2], Light.RED
        elif (m := jidoka._FAN_UPDATE.match(line)) and int(m[4]) == 0 and int(m[3]) == 0:
            if not math.isnan(v.red_s) and math.isnan(v.stop_s):
                v.stop_s, v.stage = v.seconds(int(m[2])), "stopped"
        elif m := jidoka._LIGHT.match(line):
            v.light = _LIGHTS.get(m[2], v.light)
        elif m := _DETECTOR.match(line):
            v.calib_windows = int(m[2])
    return v


_STAGE_TEXT = {
    "waiting": "waiting for the board",
    "calibrating": "calibrating: fitting the limits on normal running",
    "watching": "watching normal running",
    "switched": "switched to underspeed",
    "stopped": "stopped by the board",
}


def draw(fig, v: LiveView) -> None:
    """The light, and the scores against the limits, onto ``fig``."""
    fig.clear()
    ax_light = fig.add_axes((0.0, 0.08, 0.27, 0.86))
    ax = fig.add_axes((0.37, 0.17, 0.61, 0.74))
    ax_light.set_xlim(-1, 1)
    ax_light.set_ylim(-1.5, 1.2)
    ax_light.set_aspect("equal")
    ax_light.axis("off")
    for y, s in zip((0.75, 0.0, -0.75), (Light.RED, Light.YELLOW, Light.GREEN), strict=True):
        ax_light.add_patch(
            plt_circle((0, y), 0.32, LIGHT_COLOURS[s] if s is v.light else "#d9d9d9")
        )
    ax_light.text(0, -1.3, LIGHT_TEXT[v.light], ha="center", va="center", fontsize=9,
                  weight="bold", color=LIGHT_COLOURS[v.light])  # fmt: skip

    t, s = np.asarray(v.times_s), np.asarray(v.scores)
    cal = np.asarray(v.calibrating, dtype=bool)
    if cal.any():
        ax.plot(t[cal], s[cal], color="#9a9a9a", lw=1.0, label="calibration windows")
    if (~cal).any():
        ax.plot(t[~cal], s[~cal], color="#264653", lw=1.2, label="watched windows")
    if v.limits is not None:
        ax.axhline(v.limits.warn, color=LIGHT_COLOURS[Light.YELLOW], ls="--", lw=1.2,
                   label="warning limit (mean + 2σ)")  # fmt: skip
        ax.axhline(v.limits.control, color=LIGHT_COLOURS[Light.RED], ls="--", lw=1.2,
                   label="control limit (mean + 3σ)")  # fmt: skip
    marks = [(v.switch_s, "switched to underspeed", "#a01c1c", 0.97)]
    if not math.isnan(v.red_s) and not math.isnan(v.switch_s):
        marks.append((v.red_s, f"red {v.red_s - v.switch_s:.1f} s later", "#d62828", 0.86))
    for x, label, colour, y in marks:
        if not math.isnan(x):
            ax.axvline(x, color=colour, lw=1.0, ls=":")
            ax.text(x, y, " " + label, transform=ax.get_xaxis_transform(), fontsize=8,
                    va="top", color=colour)  # fmt: skip
    if len(s):
        ax.plot(t[-1:], s[-1:], "o", color=LIGHT_COLOURS[v.light], ms=6)
        ax.set_xlim(0, max(float(t[-1]) * 1.05, 10.0))
    n_cal = int(cal.sum())
    stage = _STAGE_TEXT[v.stage]
    if v.stage == "calibrating":
        stage += f" ({n_cal} of {v.calib_windows} windows)"
    if v.stage == "stopped":
        stage += f": fan off {v.stop_s - v.switch_s:.2f} s after the switch"
    ax.set_title(f"{v.fan or 'no fan yet'}: {stage}", fontsize=9, loc="left")
    ax.set_xlabel("board time (s)")
    ax.set_ylabel("anomaly score (negative log-likelihood)")
    if len(s):
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
