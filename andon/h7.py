"""Phase H7: live measurements on the board. The false-stop hour, fault series on one
calibration, and the table and log rows from their captures.

With ``andon.uf2`` on the Pico and the fan on its mount, first the hour of normal running::

    uv run python -m andon.h7 soak --config configs/h7_soak.yaml --port COM6 `
        --fan id_00 --out results/h7_soak_id_00.txt

It tells the board to watch the fan and waits for its limits (the hold-off, then 10 minutes
of calibration), unless the board already holds limits for that fan. Then it counts every
stop in ``minutes`` of watched running. After a stop it waits for the button: the board
restarts the fan by itself, and only watched time counts towards the hour.

Then a fault series, on the same limits::

    uv run python -m andon.h7 series --config configs/h7_underspeed.yaml --port COM6 `
        --fan id_00 --out results/h7_underspeed_id_00.txt

Each trial starts the fan normal, runs it watched for ``normal_seconds`` (a stop there is a
false stop), starts the fault, and waits for the board to stop the fan. How the fault starts
is set by the config's ``start``:

- ``switch``: a serial command switches the duty (underspeed). Timed from its arrival.
- ``fit_each``: the fault is fitted by hand with the fan off, in every trial (the card). The
  tool asks for it, then sends ``fan on``. Timed from that command, so the 5 s hold-off and
  the kick-start are inside the time.
- ``fit_once``: fitted once before the first trial and left in place (the tape: re-taping a
  blade changes the fault). Each trial is a restart with the fault on; there is no normal
  stretch, since a restart of the normal fan is what every other trial's start already is.

After each stop the tool asks for the button that clears red. Every line goes into the
capture file as it arrives (``dashboard/live.py`` can follow it). Then::

    uv run python -m andon.h7 report results/h7_*_id_00.txt --out results/h7_hardware.md --log

writes the tables and, with ``--log``, one results-log row per capture (once; commit first).
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import yaml

from andon.experiment import (
    RESULTS_LOG,
    Config,
    append_result,
    git_commit,
    is_logged,
    load_config,
    require_final,
)
from andon.jidoka import (
    _ACK,
    _BUILD,
    _FAN_UPDATE,
    _LIGHT,
    _LIMITS,
    _REC,
    _SCORE,
    _TRIP,
    Capture,
    Trial,
    TrialError,
    _expand,
    _read,
    _send,
    _shut_down,
    _wait_for,
)

_DETECTOR = re.compile(
    r"^detector watch=(\S+) phase=(\w+) light=(\w+) trip=(\w+) windows=(\d+) "
    r"calib=(-?\d+)/(\d+) restarts=(\d+)"
)
_HEADER = re.compile(r"^# andon\.h7 (soak|series), .* config (\S+), ")
_TRIAL_MARK = re.compile(r"^# pc_s=\S+ trial (\d+) of (\d+)")
_FAULT_MARK = re.compile(r"^# pc_s=\S+ fault start")
_NO_STOP_MARK = re.compile(r"^# pc_s=\S+ no stop within")
_GREEN = re.compile(r"^light t_us=\d+ light=green ")
STARTS = ("switch", "fit_each", "fit_once")
MAX_GAP_S = 5.0  # windows come every ~1 s; a longer gap is a stop or a restart

# Asks the person at the bench to do something; returns a check that turns true when done.
Ask = Callable[[str], Callable[[], bool]]


def console_ask(text: str) -> Callable[[], bool]:
    """Wait for Enter on another thread, so the board's lines keep being read meanwhile
    (an unread USB buffer makes the board drop its lines)."""
    done = threading.Event()

    def wait() -> None:
        input(f"\n>>> {text}, then press Enter ")
        done.set()

    threading.Thread(target=wait, daemon=True).start()
    return done.is_set


@dataclass(frozen=True)
class Settings:
    normal_pct: int
    settle_s: float
    calib_timeout_s: float
    wait_s: float  # for the person at the bench (button, fitting a fault)
    soak_s: float = 0.0
    max_wall_s: float = 0.0
    fault: str = ""
    start: str = ""
    fault_pct: int = -1  # the duty a switch sets; the normal duty for a fitted fault
    trials: int = 0
    normal_s: float = 0.0
    stop_timeout_s: float = 0.0

    @classmethod
    def from_config(cls, cfg: Config) -> Settings:
        rec = yaml.safe_load(Path(cfg.raw["data"]["recording_config"]).read_text(encoding="utf-8"))
        normal = int(rec["recording"]["speed_pct"])
        common = dict(
            normal_pct=normal,
            settle_s=float(rec["recording"]["settle_seconds"]),
            calib_timeout_s=float(cfg.raw["calib_timeout_seconds"]),
            wait_s=float(cfg.raw["wait_minutes"]) * 60,
        )
        if "soak" in cfg.raw:
            s = cfg.raw["soak"]
            return cls(
                **common,
                soak_s=float(s["minutes"]) * 60,
                max_wall_s=float(s["max_wall_minutes"]) * 60,
            )
        s = cfg.raw["series"]
        fault, start = str(s["fault"]), str(s["start"])
        if start not in STARTS:
            raise ValueError(f"start must be one of {STARTS}, got {start}")
        known = set(rec["faults"]) | set(rec.get("unseen_faults") or {})
        if fault not in known:
            raise ValueError(f"{fault} is not a fault of {cfg.raw['data']['recording_config']}")
        speeds = rec["recording"].get("fault_speed_pct", {})
        if (start == "switch") != (fault in speeds):
            raise ValueError(f"only a speed fault starts by a switch ({fault}, {start})")
        return cls(
            **common,
            fault=fault,
            start=start,
            fault_pct=int(speeds.get(fault, normal)),
            trials=int(s["trials"]),
            normal_s=float(s["normal_seconds"]),
            stop_timeout_s=float(s["stop_timeout_seconds"]),
        )


# --- running on the board -----------------------------------------------------------------


def _status(port, cap: Capture) -> dict:
    """``status``, and the detector, limits and recorder lines main.c prints after its ack."""
    out: dict = {"ack": _send(port, cap, "status")}
    deadline = cap.clock() + 2.0
    while cap.clock() < deadline:
        line = _read(port, cap)
        if not line:
            continue
        if m := _DETECTOR.match(line):
            out["detector"] = m
        elif m := _LIMITS.match(line):
            out["limits"] = m
        elif _REC.match(line):
            break
    return out


def _wait_until(port, cap: Capture, done: Callable[[], bool], timeout_s: float, what: str):
    deadline = cap.clock() + timeout_s
    while not done():
        if cap.clock() > deadline:
            raise TrialError(f"gave up waiting for {what}")
        _read(port, cap)


def _ask(port, cap: Capture, ask: Ask, text: str, s: Settings) -> None:
    cap.note(f"asked: {text}")
    _wait_until(port, cap, ask(text), s.wait_s, f"'{text}'")


def _clear_red(port, cap: Capture, s: Settings) -> None:
    """Wait for the button. The light line comes from the board, so it is the proof."""
    cap.note("asked: press the button to clear red")
    print("\n>>> press the button on the board to clear red", flush=True)
    if _wait_for(port, cap, _GREEN, s.wait_s) is None:
        raise TrialError("the light was not cleared")


def _ensure_limits(port, cap: Capture, fan: str, s: Settings, recalibrate: bool) -> None:
    """Keep the board's limits if they are this fan's; otherwise watch it and calibrate."""
    st = _status(port, cap)
    if st["ack"][8] == "red":
        raise TrialError("the light is red: press the button to reset it, then rerun")
    det, lim = st.get("detector"), st.get("limits")
    if not recalibrate and det and det[1] == fan and lim and lim[2] == fan:
        cap.note(f"keeping the board's limits for {fan} (control {lim[7]})")
        return
    _send(port, cap, "fan off")
    if _send(port, cap, f"watch {fan}")[2] != "ok":
        raise TrialError(f"the board has no model for {fan}")
    _send(port, cap, f"fan speed {s.normal_pct}")
    _send(port, cap, "fan on")
    cap.note("waiting for the limits (hold-off, then calibration)")
    m = _wait_for(port, cap, _LIMITS, s.calib_timeout_s)
    if m is None or m.re is _TRIP:
        raise TrialError("no limits: " + ("a stop came first" if m else "timed out"))


def _finish(port, cap: Capture, s: Settings) -> bool:
    """However it ended: the fan off (see ``_shut_down``), then the closing stream counts.
    True if the board confirmed the fan is off."""
    off = _shut_down(port, cap, s.normal_pct)
    try:
        _status(port, cap)
    except TrialError as e:
        cap.note(f"cleanup: {e}")
    return off


def _header(cap: Capture, kind: str, cfg: Config, fan: str, s: Settings) -> None:
    what = (
        f"{s.soak_s / 60:g} min watched"
        if kind == "soak"
        else (f"{s.trials} x {s.fault} ({s.start}), normal {s.normal_s:g} s")
    )
    cap.add(
        f"# andon.h7 {kind}, {datetime.now(UTC).isoformat(timespec='seconds')}, fan {fan}, "
        f"{what}, config {cfg.name}, tools {git_commit()}"
    )


class WatchClock:
    """Seconds of watched running, from the board's own window times. The time between two
    watched windows in a row counts, unless a stop or a gap (a restart) lies between."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.prev_us: int | None = None

    def feed(self, line: str) -> None:
        if m := _SCORE.match(line):
            t = int(m[1])
            if m[4] != "watching":
                self.prev_us = None
                return
            if self.prev_us is not None and 0 < t - self.prev_us <= MAX_GAP_S * 1e6:
                self.seconds += (t - self.prev_us) / 1e6
            self.prev_us = t
        elif _TRIP.match(line):
            self.prev_us = None


def run_soak(
    cfg: Config,
    port,
    fan: str,
    out: TextIO,
    recalibrate: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """The false-stop measurement on an open serial port; returns the captured lines."""
    s = Settings.from_config(cfg)
    cap = Capture(out, clock)
    _header(cap, "soak", cfg, fan, s)
    try:
        _ensure_limits(port, cap, fan, s, recalibrate)
        _send(port, cap, f"fan speed {s.normal_pct}")
        _send(port, cap, "fan on")
        cap.note(f"counting stops over {s.soak_s / 60:g} min of watched running")
        watched = WatchClock()
        next_note = 300.0
        deadline = cap.clock() + s.max_wall_s
        while watched.seconds < s.soak_s:
            if cap.clock() > deadline:
                cap.note(f"stopped at the {s.max_wall_s / 60:g} min limit, short of the time")
                break
            line = _read(port, cap)
            if not line:
                continue
            watched.feed(line)
            if m := _TRIP.match(line):
                cap.note(
                    f"stop ({m[2]}) after {watched.seconds / 60:.1f} watched min: press the "
                    "button; the board restarts the fan and the count goes on"
                )
                print("\n>>> press the button on the board to clear red", flush=True)
            if watched.seconds >= next_note:
                cap.note(f"{watched.seconds / 60:.1f} of {s.soak_s / 60:g} watched min")
                next_note += 300.0
    finally:
        _finish(port, cap, s)
    return cap.lines


def run_series(
    cfg: Config,
    port,
    fan: str,
    out: TextIO,
    ask: Ask = console_ask,
    recalibrate: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """``trials`` fault trials on one set of limits; returns the captured lines."""
    s = Settings.from_config(cfg)
    cap = Capture(out, clock)
    _header(cap, "series", cfg, fan, s)
    try:
        _ensure_limits(port, cap, fan, s, recalibrate)
        for k in range(1, s.trials + 1):
            cap.note(f"trial {k} of {s.trials}")
            _one_trial(port, cap, ask, s, first=k == 1)
    finally:
        off = _finish(port, cap, s)
    # Only once the board has confirmed the fan is off is anyone asked to reach into it.
    if s.start != "switch" and off:
        cap.note(f"series done: remove the {s.fault} with the fan off")
    return cap.lines


def _one_trial(port, cap: Capture, ask: Ask, s: Settings, first: bool) -> None:
    st = _status(port, cap)  # its detector line opens the trial's restart count
    _send(port, cap, "fan off")
    _send(port, cap, f"fan speed {s.normal_pct}")
    if st["ack"][8] == "red":
        _clear_red(port, cap, s)
    if s.start == "fit_each" and not first:
        _ask(port, cap, ask, f"make sure the {s.fault} is off (the fan is off)", s)
    if s.start == "fit_once":
        if first:
            _ask(port, cap, ask, f"fit the {s.fault} (the fan is off; wait for the blades)", s)
    else:
        _send(port, cap, "fan on")
        if not _normal_stretch(port, cap, s):
            _status(port, cap)
            return
        if s.start == "fit_each":
            _send(port, cap, "fan off")
            _ask(port, cap, ask, f"wait for the blades to stop, then fit the {s.fault}", s)
    cap.note("fault start")
    _send(port, cap, f"fan speed {s.fault_pct}" if s.start == "switch" else "fan on")
    if _wait_for(port, cap, _TRIP, s.stop_timeout_s) is None:
        cap.note(f"no stop within {s.stop_timeout_s:g} s")
    else:
        _wait_for(port, cap, _FAN_UPDATE, 2.0)  # the update that cut the fan
    _status(port, cap)  # closes the trial's restart count


def _normal_stretch(port, cap: Capture, s: Settings) -> bool:
    """The hold-off, then ``normal_s`` watched. False if the board stopped the fan."""
    deadline = cap.clock() + s.settle_s + 30.0
    while cap.clock() < deadline:
        line = _read(port, cap)
        if line and _TRIP.match(line):
            cap.note("false stop in the hold-off")
            return False
        if line and (m := _SCORE.match(line)) and m[4] == "watching":
            break
    else:
        raise TrialError("the board never started watching after the fan started")
    if _wait_for(port, cap, re.compile(r"(?!)"), s.normal_s) is not None:
        cap.note("false stop before the fault")
        return False
    return True


# --- reading captures ---------------------------------------------------------------------


@dataclass
class Limits:
    fan: str = ""
    calib_windows: int = -1
    center: float = math.nan
    control: float = math.nan


def _limits(m: re.Match) -> Limits:
    return Limits(m[2], int(m[3]), float(m[4]), float(m[7]))


@dataclass
class Soak:
    fan: str = ""
    build: str = "unknown"
    limits: Limits = field(default_factory=Limits)
    watched_s: float = 0.0
    windows: int = 0
    above_control: int = 0  # single windows; red needs several in a row
    max_score: float = math.nan
    yellows: int = 0
    stops: list[tuple[str, float]] = field(default_factory=list)  # (reason, watched s)
    restarts: int = -1
    problems: list[str] = field(default_factory=list)

    def stops_of(self, score: bool) -> int:
        return sum((r == "score") == score for r, _ in self.stops)

    @property
    def per_hour(self) -> float:
        return len(self.stops) / (self.watched_s / 3600) if self.watched_s > 0 else math.nan


def _restarts(text: str) -> int:
    counts = [int(m[8]) for line in text.splitlines() if (m := _DETECTOR.match(line))]
    return counts[-1] - counts[0] if len(counts) >= 2 else -1


def parse_soak(text: str, soak_s: float) -> Soak:
    out = Soak()
    watched = WatchClock()
    for line in text.splitlines():
        # Only windows judged against limits count. The window that completes the
        # calibration is printed as watching just before its limits line, so it is not one.
        if out.limits.fan:
            watched.feed(line)
        if m := _BUILD.match(line):
            out.build = m[1]
        elif (m := _ACK.match(line)) and m[1].startswith("watch ") and m[2] == "ok":
            out.limits = Limits()  # a new calibration follows; older limits don't apply
        elif m := _LIMITS.match(line):
            lim = _limits(m)
            if out.limits.fan and (lim.center, lim.control) != (
                out.limits.center,
                out.limits.control,
            ):
                out.problems.append("the limits changed during the soak")
            out.limits = lim
            out.fan = lim.fan
        elif (m := _SCORE.match(line)) and m[4] == "watching" and out.limits.fan:
            score = float(m[5])
            out.windows += 1
            out.above_control += score > out.limits.control
            out.max_score = score if math.isnan(out.max_score) else max(out.max_score, score)
        elif (m := _TRIP.match(line)) and out.limits.fan:
            out.stops.append((m[2], watched.seconds))
        elif (m := _LIGHT.match(line)) and m[2] == "yellow":
            out.yellows += 1
    out.watched_s = watched.seconds
    out.restarts = _restarts(text)
    if not out.limits.fan:
        out.problems.append("the board never had limits")
    if out.watched_s < soak_s:
        out.problems.append(f"watched {out.watched_s / 60:.1f} min of {soak_s / 60:g}")
    if out.restarts > 0:
        out.problems.append(f"the audio stream restarted {out.restarts} time(s)")
    return out


@dataclass
class SeriesTrial(Trial):
    index: int = 0
    false_stop: bool = False  # stopped before the fault started


@dataclass
class Series:
    fault: str
    start: str
    fan: str = ""
    build: str = "unknown"
    limits: Limits = field(default_factory=Limits)
    trials: list[SeriesTrial] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # about the series as a whole

    @property
    def caught(self) -> list[SeriesTrial]:
        return [t for t in self.trials if t.passed]

    @property
    def false_stops(self) -> int:
        return sum(t.false_stop for t in self.trials)


def parse_series(text: str, s: Settings) -> Series:
    """Every trial in a series capture. A trial passes only if the fault started from a
    green light (after a watched normal stretch, unless the fault is fitted once), and the
    board then went red on its scores and cut the fan, with no command or button between."""
    out = Series(s.fault, s.start)
    segments: list[list[str]] = [[]]
    for line in text.splitlines():
        if (m := _TRIAL_MARK.match(line)) and int(m[1]) == len(segments):
            segments.append([])
        segments[-1].append(line)
    for line in segments[0]:
        if m := _BUILD.match(line):
            out.build = m[1]
        elif m := _LIMITS.match(line):
            out.limits = _limits(m)
    out.fan = out.limits.fan
    if not out.fan:
        out.problems.append("the board never had limits")
    for k, seg in enumerate(segments[1:], start=1):
        out.trials.append(_parse_trial(seg, k, s, out))
    if len(out.trials) != s.trials:
        out.problems.append(f"{len(out.trials)} trials of {s.trials}")
    return out


def _parse_trial(lines: list[str], k: int, s: Settings, series: Series) -> SeriesTrial:
    t = SeriesTrial(index=k, fan=series.fan, build=series.build, fault_pct=s.fault_pct)
    t.calib_windows, t.center, t.control = (
        series.limits.calib_windows,
        series.limits.center,
        series.limits.control,
    )
    stage = "before"  # before -> starting (next ack) -> started -> stopped or timed out
    # The board's window count when the fault starts. A fit-once trial prints no score line
    # before its start, so the status line that opens each trial sets it first.
    last_n = n_at_start = 0
    for line in lines:
        if _FAULT_MARK.match(line):
            stage = "starting"
        elif (m := _DETECTOR.match(line)) and stage == "before":
            last_n = int(m[5])
        elif _NO_STOP_MARK.match(line) and stage == "started":
            stage = "timed out"  # the tool's own closing commands follow
        elif m := _ACK.match(line):
            cmd, result = m[1], m[2]
            if stage == "starting":
                t.switch_us, stage, n_at_start = int(m[3]), "started", last_n
                want = f"fan speed {s.fault_pct}" if s.start == "switch" else "fan on"
                if cmd != want or result != "ok" or m[8] != "green":
                    t.problems.append(f'the fault started with "{cmd}" ({result}, {m[8]})')
                if s.start == "switch" and int(m[7]) != s.fault_pct:
                    t.problems.append(f"the switch left the fan at duty {m[7]}")
            elif stage == "started" and cmd.split()[:1] == ["fan"]:
                t.problems.append(f'"{cmd}" was sent before the board stopped the fan')
        elif m := _LIMITS.match(line):
            if (float(m[4]), float(m[7])) != (series.limits.center, series.limits.control):
                t.problems.append("the limits changed: the series must keep one calibration")
        elif m := _SCORE.match(line):
            last_n = int(m[2])
            if stage == "before" and m[4] == "watching":
                t.normal_windows += 1
                v = float(m[5])
                t.max_normal_score = (
                    v if math.isnan(t.max_normal_score) else max(t.max_normal_score, v)
                )
        elif m := _TRIP.match(line):
            if stage == "started" and t.trip_us < 0:
                t.trip_us, t.trip_reason = int(m[1]), m[2]
                t.windows_to_red = int(m[4]) - n_at_start
            elif stage == "before":
                t.false_stop = True
                t.problems.append(f"a stop before the fault (reason {m[2]}): a false stop")
        elif m := _FAN_UPDATE.match(line):
            cut = int(m[4]) == 0 and int(m[3]) == 0 and int(m[2]) >= 0
            if t.trip_us >= 0 and t.stop_us < 0 and cut:
                t.stop_us, stage = int(m[2]), "stopped"
        elif (m := _LIGHT.match(line)) and m[4] == "1" and stage == "started":
            t.problems.append("the button was pressed before the board stopped the fan")
    t.restarts = _restarts("\n".join(lines))
    if t.restarts > 0:
        t.problems.append(f"the audio stream restarted {t.restarts} time(s) in the trial")
    if s.start != "fit_once" and not t.false_stop and t.normal_windows == 0:
        t.problems.append("no normal window was watched before the fault")
    if t.false_stop:
        return t  # its problem is already listed; nothing after it is timed
    if t.switch_us < 0:
        t.problems.append("the fault never started")
    elif t.trip_us < 0:
        t.problems.append(f"the board never went red within {s.stop_timeout_s:g} s")
    elif t.trip_reason != "score":
        t.problems.append(f"red for another reason ({t.trip_reason}), not the scores")
    elif t.stop_us < 0:
        t.problems.append("no fan update cut the fan after red")
    return t


# --- report and log -----------------------------------------------------------------------


def _fmt(values: list[float], digits: int = 3) -> str:
    """mean ± sd (n-1) over the trials, then the range."""
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return (
        f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f} "
        f"({min(values):.{digits}f} to {max(values):.{digits}f})"
    )


START_WORDS = {
    "switch": "serial switch to the fault duty",
    "fit_each": "fitted with the fan off, then `fan on`",
    "fit_once": "fitted once, then `fan on` each trial",
}


def soak_table(soaks: list[Soak]) -> str:
    head = (
        "| Fan | Watched (h) | Watched windows | Stops on scores | Other stops "
        "| Stops per hour | Yellow lights | Windows above control (max score / control) "
        "| Stream restarts | Complete |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    rows = [
        f"| {o.fan} | {o.watched_s / 3600:.3f} | {o.windows} | {o.stops_of(True)} | "
        f"{o.stops_of(False)} | {o.per_hour:.2f} | {o.yellows} | {o.above_control} "
        f"({o.max_score:.4f} / {o.limits.control:.4f}) | {o.restarts} | "
        + ("yes" if not o.problems else "**no**: " + "; ".join(o.problems))
        + " |"
        for o in soaks
    ]
    return head + "\n".join(rows) + "\n"


def series_table(series: list[Series]) -> str:
    head = (
        "| Fault | How it starts | Trials | Caught | False stops before the fault "
        "| Start to fan off (s) | Start to red (s) | Windows to red | Red to fan off (ms) |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    for x in series:
        c = x.caught
        rows.append(
            f"| {x.fault} | {START_WORDS[x.start]} | {len(x.trials)} | {len(c)} | "
            f"{x.false_stops} | {_fmt([t.switch_to_stop_s for t in c])} | "
            f"{_fmt([t.switch_to_red_s for t in c])} | "
            f"{_fmt([float(t.windows_to_red) for t in c], 1)} | "
            f"{_fmt([t.red_to_stop_ms for t in c])} |"
        )
    return head + "\n".join(rows) + "\n"


def trials_table(x: Series) -> str:
    head = (
        "| Trial | Normal windows (max score) | Windows to red | Start to red (s) "
        "| Red to fan off (ms) | Start to fan off (s) | Result |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    rows = []
    for t in x.trials:
        done = t.stop_us >= 0 and not t.false_stop
        rows.append(
            f"| {t.index} | {t.normal_windows} ({t.max_normal_score:.4f}) | "
            + (
                f"{t.windows_to_red} | {t.switch_to_red_s:.3f} | {t.red_to_stop_ms:.3f} | "
                f"{t.switch_to_stop_s:.3f} | "
                if done
                else "- | - | - | - | "
            )
            + ("PASS" if t.passed else "**FAIL**: " + "; ".join(t.problems))
            + " |"
        )
    return head + "\n".join(rows) + "\n"


def _stat(values: list[float], f) -> float:
    return f(values) if len(values) >= (2 if f is statistics.stdev else 1) else math.nan


def soak_notes(o: Soak, path: Path) -> str:
    return (
        f"trial=soak complete={int(not o.problems)} watched_h={o.watched_s / 3600:.4f} "
        f"windows={o.windows} stops_score={o.stops_of(True)} stops_other={o.stops_of(False)} "
        f"stops_per_h={o.per_hour:.2f} yellows={o.yellows} above_control={o.above_control} "
        f"max_score={o.max_score:.4f} control={o.limits.control:.4f} "
        f"calib_windows={o.limits.calib_windows} restarts={o.restarts} build={o.build} "
        f"capture={path.as_posix()}"
    )


def series_notes(x: Series, path: Path) -> str:
    c = x.caught
    stop = [t.switch_to_stop_s for t in c]
    red = [t.switch_to_red_s for t in c]
    return (
        f"trial=series fault={x.fault} start={x.start} trials={len(x.trials)} "
        f"caught={len(c)} false_stops={x.false_stops} "
        f"start_to_stop_mean_s={_stat(stop, statistics.mean):.3f} "
        f"start_to_stop_sd_s={_stat(stop, statistics.stdev):.3f} "
        f"start_to_stop_min_s={_stat(stop, min):.3f} start_to_stop_max_s={_stat(stop, max):.3f} "
        f"start_to_red_mean_s={_stat(red, statistics.mean):.3f} "
        f"red_to_stop_mean_ms={_stat([t.red_to_stop_ms for t in c], statistics.mean):.3f} "
        f"control={x.limits.control:.4f} calib_windows={x.limits.calib_windows} "
        f"build={x.build} capture={path.as_posix()}"
    )


def read_capture(path: Path) -> tuple[Config, str, Soak | Series]:
    text = path.read_text(encoding="utf-8")
    m = _HEADER.match(text)
    if m is None:
        raise ValueError(f"{path} is not an andon.h7 capture")
    cfg = load_config(Path("configs") / f"{m[2]}.yaml")
    s = Settings.from_config(cfg)
    return cfg, m[1], parse_soak(text, s.soak_s) if m[1] == "soak" else parse_series(text, s)


def log_rows(items: list[tuple[Config, Soak | Series, Path]], seed: int, log: Path) -> None:
    """One eval row per capture: AUC does not apply to a live measurement (logged as nan)."""
    keys = [(cfg.name, x.fan) for cfg, x, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"one capture per config and fan: {keys}")
    for cfg, x, _ in items:
        require_final(cfg, "eval")
        if not x.fan:
            raise ValueError(f"{cfg.name}: no fan in the capture")
        if is_logged(cfg, x.fan, seed, "eval", log):
            raise ValueError(f"{cfg.name} {x.fan} seed {seed} is already logged (once)")
    for cfg, x, path in items:
        notes = soak_notes(x, path) if isinstance(x, Soak) else series_notes(x, path)
        append_result(cfg, x.fan, "eval", seed, math.nan, math.nan, notes, log=log)


def report(items: list[tuple[Config, Soak | Series, Path]]) -> str:
    soaks = [x for _, x, _ in items if isinstance(x, Soak)]
    series = [x for _, x, _ in items if isinstance(x, Series)]
    sources = ", ".join(f"`{p.as_posix()}`" for _, _, p in items)
    parts = [
        "# H7: live measurements on the board\n\n"
        f"Generated by `andon.h7 report` from {sources}. The board watches the fan with "
        "`h3_gmm` (seed 0) and fits its limits on the device; every time is from the "
        "board's clock. Rows in `results/results_log.csv` (configs `h7_*`).\n"
    ]
    if soaks:
        parts.append(
            "\n## False stops in normal running\n\nThe fan ran at its normal speed, watched, "
            "with nothing changed. Watched time is the time between watched windows; the "
            "hold-off after a restart does not count.\n\n" + soak_table(soaks)
        )
    if series:
        parts.append(
            "\n## Fault trials\n\nEach trial starts the fault from a green light and waits "
            "for the board to stop the fan by itself. Start to fan off is from the command "
            "that starts the fault to the gate pin reading back low (the power cut; "
            "spin-down is not timed). For a fitted fault that command is `fan on`, so the "
            "5 s hold-off and the kick-start are inside the time. Times are mean ± sd over "
            "the caught trials, then the range.\n\n" + series_table(series)
        )
        for x in series:
            parts.append(f"\n### {x.fault} ({x.fan})\n\n" + trials_table(x))
    return "".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    runs = [sub.add_parser("soak", help="count stops in normal running")]
    runs.append(sub.add_parser("series", help="fault trials on one calibration"))
    for p in runs:
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--port", required=True, help="serial port of the Pico (e.g. COM6)")
        p.add_argument("--fan", required=True, help="the machine ID of the fan on the bench")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--recalibrate", action="store_true", help="fit new limits first")
    rep = sub.add_parser("report", help="the tables (and log rows) from captures")
    rep.add_argument("captures", nargs="+")
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--log", action="store_true", help="append one row per capture (once)")
    for p in [*runs, rep]:
        p.add_argument("--seed", type=int, default=0, help="the exported model's seed")
    args = parser.parse_args(argv)

    if args.cmd == "report":
        items = []
        for path in _expand(args.captures):
            cfg, _, x = read_capture(path)
            if args.seed != int(cfg.raw["seed"]):
                parser.error(f"{cfg.name}: the board's model has seed {cfg.raw['seed']}")
            items.append((cfg, x, path))
        text = report(items)
        args.out.write_text(text, encoding="utf-8")
        print(text, end="")
        if args.log:
            log_rows(items, args.seed, RESULTS_LOG)
            print(f"logged {len(items)} rows to {RESULTS_LOG}")
        return

    cfg = load_config(args.config)
    if args.seed != int(cfg.raw["seed"]):
        parser.error(f"the board's model was exported with seed {cfg.raw['seed']}")
    if args.out.exists():
        parser.error(f"{args.out} exists: a capture is never overwritten, pick a new name")
    import serial  # only needed with a board attached

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The port opens first, so a busy or mistyped port leaves no empty capture behind.
    with (
        serial.Serial(args.port, 115200, timeout=0.2) as port,
        args.out.open("x", encoding="utf-8", newline="\n") as out,
    ):
        if args.cmd == "soak":
            lines = run_soak(cfg, port, args.fan, out, args.recalibrate)
        else:
            lines = run_series(cfg, port, args.fan, out, recalibrate=args.recalibrate)
    print(f"wrote {args.out}")
    s = Settings.from_config(cfg)
    text = "\n".join(lines)
    if args.cmd == "soak":
        print(soak_table([parse_soak(text, s.soak_s)]), end="")
    else:
        x = parse_series(text, s)
        print(series_table([x]) + "\n" + trials_table(x), end="")


if __name__ == "__main__":
    main()
