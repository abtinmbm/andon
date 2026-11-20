"""Phase H6: a jidoka trial on the board, and the table and log rows from its captures.

With ``andon.uf2`` on the Pico and the fan running on the bench, one trial per fan::

    uv run python -m andon.jidoka run --config configs/h6_jidoka.yaml --seed 0 `
        --port COM6 --fan id_00 --out results/h6_jidoka_id_00.txt

It tells the board to watch the fan, starts it at the normal speed and waits for the limits
(the hold-off, then 10 minutes of calibration on the board). The fan then runs watched for
``normal_seconds``. A false stop there ends the trial and is reported. Then the tool sends
``fan speed <underspeed duty>`` and waits for the board to stop the fan by itself. Every
line the board prints goes into the capture file as it arrives, so the dashboard can follow
it live (``dashboard/live.py``). Lines starting with ``#`` are the tool's own notes.

The times come from the board's clock: the switch is the ``t_rx_us`` of the speed
command's ack, red is the ``trip`` line, and the stop is the ``t_pin_us`` of the ``fan
update`` that cut the fan (the gate pin read back low). Then::

    uv run python -m andon.jidoka report --config configs/h6_jidoka.yaml --seed 0 `
        results/h6_jidoka_id_*.txt --out results/h6_jidoka.md --log

writes the table and, with ``--log``, one results-log row per fan (once; commit first).
"""

from __future__ import annotations

import argparse
import math
import re
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

_ACK = re.compile(
    r'^ack cmd="([^"]*)" result=(\w+) t_rx_us=(\d+) t_set_us=(\d+) t_pin_us=(-?\d+) '
    r"pin=(-?\d+) duty=(\d+) light=(\w+)"
)
_SCORE = re.compile(r"^score t_us=(\d+) n=(\d+) fan=(\S+) phase=(\w+) s=(\S+) light=(\w+)")
_LIMITS = re.compile(
    r"^limits t_us=(\d+) fan=(\S+) n=(\d+) center=(\S+) sigma=(\S+) warn=(\S+) control=(\S+)"
)
_TRIP = re.compile(r"^trip t_us=(\d+) reason=(\w+) fan=(\S+) n=(\d+) s=(\S+)")
_FAN_UPDATE = re.compile(
    r"^fan update t_set_us=(\d+) t_pin_us=(-?\d+) pin=(-?\d+) duty=(\d+) light=(\w+) "
    r"kick=(\d) trip=(\w+)"
)
_LIGHT = re.compile(r"^light t_us=(\d+) light=(\w+) trip=(\w+) reset=(\d)")
_BUILD = re.compile(r"^build=(\S+)")
_DETECTOR = re.compile(r"^detector watch=\S+ .* restarts=(\d+) say_dropped=(\d+)")
_REC = re.compile(r"^rec ")
# Matches nothing: waiting on it captures lines until a trip or the timeout.
_NOTHING = re.compile(r"(?!)")


@dataclass(frozen=True)
class TrialSettings:
    normal_pct: int
    fault_pct: int
    normal_s: float
    settle_s: float  # the board's hold-off after each start
    calib_timeout_s: float
    stop_timeout_s: float

    @classmethod
    def from_config(cls, cfg: Config) -> TrialSettings:
        path = Path(cfg.raw["data"]["recording_config"])
        rec = yaml.safe_load(path.read_text(encoding="utf-8"))["recording"]
        t = cfg.raw["trial"]
        return cls(
            normal_pct=int(rec["speed_pct"]),
            fault_pct=int(rec["fault_speed_pct"]["underspeed"]),
            normal_s=float(t["normal_seconds"]),
            settle_s=float(rec["settle_seconds"]),
            calib_timeout_s=float(t["calib_timeout_seconds"]),
            stop_timeout_s=float(t["stop_timeout_seconds"]),
        )


# --- running a trial on the board ---------------------------------------------------------


class Capture:
    """Every line, written and flushed as it comes, so a dashboard can follow the file."""

    def __init__(self, out: TextIO, clock: Callable[[], float], echo: bool = True) -> None:
        self.out, self.clock, self.echo = out, clock, echo
        self.t0 = clock()
        self.lines: list[str] = []

    def add(self, line: str) -> None:
        self.lines.append(line)
        self.out.write(line + "\n")
        self.out.flush()
        if self.echo:
            print(line, flush=True)

    def note(self, text: str) -> None:
        self.add(f"# pc_s={self.clock() - self.t0:.1f} {text}")


class TrialError(RuntimeError):
    """The trial could not go on; the capture so far says why."""


def _read(port, cap: Capture) -> str | None:
    line = port.readline().decode(errors="replace").strip()
    if not line:
        return None
    cap.add(line)
    return line


def _send(port, cap: Capture, cmd: str, timeout_s: float = 2.0) -> re.Match:
    """Write a command and read (and capture) lines until its ack."""
    cap.note(f"sent: {cmd}")
    port.write((cmd + "\n").encode())
    deadline = cap.clock() + timeout_s
    while cap.clock() < deadline:
        line = _read(port, cap)
        if line and (m := _ACK.match(line)) and m[1] == cmd:
            return m
    raise TrialError(f"no ack for {cmd!r} within {timeout_s} s: is andon.uf2 (H6) flashed?")


def _wait_for(port, cap: Capture, pattern: re.Pattern, timeout_s: float) -> re.Match | None:
    """Capture lines until one matches, a trip happens first (returned too) or time is up."""
    deadline = cap.clock() + timeout_s
    while cap.clock() < deadline:
        line = _read(port, cap)
        if not line:
            continue
        if m := pattern.match(line):
            return m
        if m := _TRIP.match(line):
            return m
    return None


def run(
    cfg: Config, port, fan: str, out: TextIO, clock: Callable[[], float] = time.monotonic
) -> list[str]:
    """One trial on an open serial port; returns the captured lines."""
    s = TrialSettings.from_config(cfg)
    cap = Capture(out, clock)
    cap.add(
        f"# andon.jidoka, {datetime.now(UTC).isoformat(timespec='seconds')}, fan {fan}, "
        f"speed {s.normal_pct} -> {s.fault_pct} %, normal {s.normal_s:g} s, "
        f"config {cfg.name}, tools {git_commit()}"
    )
    if _send(port, cap, "status")[8] == "red":
        raise TrialError("the light is red: press the button to reset it, then rerun")
    try:
        _trial(port, cap, s, fan)
    finally:
        # However the trial ended, leave the fan asked off at the normal speed, so the
        # button that clears red does not start it again.
        _send(port, cap, f"fan speed {s.normal_pct}")
        _send(port, cap, "fan off")
        _send(port, cap, "status")
        # The detector and recorder lines follow the status ack; the detector's stream
        # restart count is how a trial shows it had no gap in the audio.
        _wait_for(port, cap, _REC, 2.0)
    return cap.lines


def _trial(port, cap: Capture, s: TrialSettings, fan: str) -> None:
    _send(port, cap, "fan off")
    if _send(port, cap, f"watch {fan}")[2] != "ok":
        raise TrialError(f"the board has no model for {fan}")
    _send(port, cap, f"fan speed {s.normal_pct}")
    _send(port, cap, "fan on")
    cap.note("waiting for the limits (hold-off, then calibration)")
    m = _wait_for(port, cap, _LIMITS, s.calib_timeout_s)
    if m is None or m.re is _TRIP:
        cap.note("no limits: " + ("a trip came first" if m else "timed out"))
        return
    cap.note(f"limits fitted; running watched for {s.normal_s:g} s")
    if _wait_for(port, cap, _NOTHING, s.normal_s) is not None:  # only a trip ends it early
        cap.note("false stop before the switch")
        return
    _send(port, cap, f"fan speed {s.fault_pct}")
    cap.note("switched to underspeed; waiting for the board to stop the fan")
    if _wait_for(port, cap, _NOTHING, s.stop_timeout_s) is None:
        cap.note(f"no stop within {s.stop_timeout_s:g} s")
    else:
        # The fan update that cuts the fan comes right after the trip line.
        _wait_for(port, cap, _FAN_UPDATE, 2.0)


# --- reading a capture --------------------------------------------------------------------


@dataclass
class Trial:
    """What one capture shows, with every problem that makes it fail."""

    fan: str = ""
    build: str = "unknown"
    fault_pct: int = -1
    calib_windows: int = -1
    center: float = math.nan
    control: float = math.nan
    normal_windows: int = 0  # scored after the limits and before the switch
    max_normal_score: float = math.nan
    switch_us: int = -1
    trip_us: int = -1
    trip_reason: str = ""
    windows_to_red: int = -1
    stop_us: int = -1  # the gate pin read back low
    restarts: int = -1  # audio stream restarts during the trial; -1 if not captured
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems

    @staticmethod
    def _span(start_us: int, end_us: int, unit_us: float) -> float:
        # nan, not a negative time, when either end never happened.
        return (end_us - start_us) / unit_us if start_us >= 0 and end_us >= 0 else math.nan

    @property
    def switch_to_red_s(self) -> float:
        return self._span(self.switch_us, self.trip_us, 1e6)

    @property
    def red_to_stop_ms(self) -> float:
        return self._span(self.trip_us, self.stop_us, 1e3)

    @property
    def switch_to_stop_s(self) -> float:
        return self._span(self.switch_us, self.stop_us, 1e6)


def parse_capture(text: str, fault_pct: int) -> Trial:
    """The trial in a capture. It passes only if the board calibrated, ran the normal
    stretch without a stop, and after the switch to ``fault_pct`` cut the fan itself on its
    scores, with no command or button press in between that could have stopped it."""
    t = Trial(fault_pct=fault_pct)
    stage = "setup"  # setup -> calibrating -> watching -> switched -> stopped
    last_n = last_n_at_switch = 0  # the board's window count, so windows to red can be counted
    restarts: list[int] = []  # from each status's detector line: at the start and the end
    for line in text.splitlines():
        if m := _BUILD.match(line):
            t.build = m[1]
        elif m := _ACK.match(line):
            cmd, result = m[1], m[2]
            if cmd.startswith("watch ") and result == "ok":
                t.fan = cmd.split()[1]
            elif cmd == "fan on" and stage == "setup":
                stage = "calibrating" if result == "ok" else stage
            elif cmd == f"fan speed {fault_pct}" and stage == "watching" and result == "ok":
                t.switch_us, stage = int(m[3]), "switched"
                # A switch while the light is already off green, or that did not take the
                # fan to the fault duty, would not time the fault from a clean start.
                if m[8] != "green" or int(m[7]) != fault_pct:
                    t.problems.append(
                        f"the switch left the fan at duty {m[7]} with the light {m[8]}"
                    )
            elif stage == "switched" and cmd.split()[:2] in (["fan", "off"], ["fan", "speed"]):
                t.problems.append(f'"{cmd}" was sent before the board stopped the fan')
        elif m := _LIMITS.match(line):
            if stage == "calibrating":
                t.calib_windows, t.center, t.control = int(m[3]), float(m[4]), float(m[7])
                stage = "watching"
        elif m := _SCORE.match(line):
            last_n = int(m[2])
            if stage == "watching":
                t.normal_windows += 1
                s = float(m[5])
                t.max_normal_score = (
                    s if math.isnan(t.max_normal_score) else max(t.max_normal_score, s)
                )
        elif m := _TRIP.match(line):
            if stage == "switched" and t.trip_us < 0:
                t.trip_us, t.trip_reason = int(m[1]), m[2]
                t.windows_to_red = int(m[4]) - last_n_at_switch
            elif stage in ("calibrating", "watching"):
                t.problems.append(f"a stop before the switch (reason {m[2]}): a false stop")
        elif m := _FAN_UPDATE.match(line):
            # duty 0 with the gate pin read back low: the fan has no power.
            cut = int(m[4]) == 0 and int(m[3]) == 0 and int(m[2]) >= 0
            if t.trip_us >= 0 and t.stop_us < 0 and cut:
                t.stop_us, stage = int(m[2]), "stopped"
        elif (m := _LIGHT.match(line)) and m[4] == "1" and stage == "switched":
            t.problems.append("the button was pressed before the board stopped the fan")
        elif m := _DETECTOR.match(line):
            restarts.append(int(m[1]))
        if stage != "switched":
            last_n_at_switch = last_n
    if len(restarts) >= 2:
        t.restarts = restarts[-1] - restarts[0]
        if t.restarts > 0:
            t.problems.append(f"the audio stream restarted {t.restarts} time(s) in the trial")
    if not t.fan:
        t.problems.append("no fan was watched")
    if t.calib_windows < 0:
        t.problems.append("the board never fitted its limits")
    elif t.normal_windows == 0:
        t.problems.append("no window was watched before the switch")
    if t.switch_us < 0:
        t.problems.append(f"no switch to {fault_pct} %")
    elif t.trip_us < 0:
        t.problems.append("the board never went red after the switch")
    elif t.trip_reason != "score":
        t.problems.append(f"red for another reason ({t.trip_reason}), not the scores")
    elif t.stop_us < 0:
        t.problems.append("no fan update cut the fan after red")
    return t


def jidoka_table(trials: list[Trial]) -> str:
    head = (
        "| Fan | Calibration windows | Control limit | Watched normal windows (max score) "
        "| Windows to red | Switch to red (s) | Red to fan off (ms) | Switch to fan off (s) "
        "| Result |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    rows = []
    for t in trials:
        done = t.stop_us >= 0
        rows.append(
            f"| {t.fan} | {t.calib_windows} | {t.control:.4f} | "
            f"{t.normal_windows} ({t.max_normal_score:.4f}) | "
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


def log_rows(cfg: Config, seed: int, trials: list[Trial], captures: list[Path], log: Path):
    """One eval row per fan: AUC does not apply to a live trial (logged as nan)."""
    require_final(cfg, "eval")
    if len(trials) != len(captures):
        raise ValueError("one capture per trial")
    fans = [t.fan for t in trials]
    if len(set(fans)) != len(fans):
        raise ValueError(f"one trial per fan: {fans}")
    for t in trials:
        if is_logged(cfg, t.fan, seed, "eval", log):
            raise ValueError(f"{cfg.name} {t.fan} seed {seed} is already logged (once per fan)")
    for t, path in zip(trials, captures, strict=True):
        notes = (
            f"trial=jidoka passed={int(t.passed)} switch_to_stop_s={t.switch_to_stop_s:.3f} "
            f"switch_to_red_s={t.switch_to_red_s:.3f} red_to_stop_ms={t.red_to_stop_ms:.3f} "
            f"windows_to_red={t.windows_to_red} normal_windows={t.normal_windows} "
            f"calib_windows={t.calib_windows} restarts={t.restarts} speed={t.fault_pct} "
            f"build={t.build} "
            f"capture={path.as_posix()}"
        )
        append_result(cfg, t.fan, "eval", seed, math.nan, math.nan, notes, log=log)


def _expand(patterns: list[str]) -> list[Path]:
    """PowerShell passes globs through unexpanded, so expand them here."""
    paths: list[Path] = []
    for p in patterns:
        if any(c in p for c in "*?["):
            found = sorted(Path().glob(p))
            if not found:
                raise FileNotFoundError(f"no capture matches {p}")
            paths += found
        else:
            paths.append(Path(p))
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one trial on the board")
    r.add_argument("--port", required=True, help="serial port of the Pico (e.g. COM6)")
    r.add_argument("--fan", required=True, help="the machine ID of the fan on the bench")
    r.add_argument("--out", type=Path, required=True)
    rep = sub.add_parser("report", help="the table (and log rows) from captures")
    rep.add_argument("captures", nargs="+")
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--log", action="store_true", help="append one row per fan (once)")
    for p in (r, rep):
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--seed", type=int, default=0, help="the exported model's seed")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.seed != int(cfg.raw["seed"]):
        parser.error(f"the board's model was exported with seed {cfg.raw['seed']}")
    fault_pct = TrialSettings.from_config(cfg).fault_pct

    if args.cmd == "run":
        import serial  # only needed with a board attached

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with (
            args.out.open("w", encoding="utf-8", newline="\n") as out,
            serial.Serial(args.port, 115200, timeout=0.2) as port,
        ):
            lines = run(cfg, port, args.fan, out)
        trial = parse_capture("\n".join(lines), fault_pct)
        print(f"wrote {args.out}")
        print(jidoka_table([trial]), end="")
    else:
        captures = _expand(args.captures)
        trials = [parse_capture(p.read_text(encoding="utf-8"), fault_pct) for p in captures]
        table = jidoka_table(trials)
        calib = {t.calib_windows for t in trials}
        if len(calib) != 1:
            raise SystemExit(f"the trials calibrated on different window counts: {calib}")
        s = TrialSettings.from_config(cfg)
        sources = ", ".join(f"`{p.as_posix()}`" for p in captures)
        args.out.write_text(
            f"# H6: jidoka on the board\n\nGenerated by `andon.jidoka report` from {sources}. "
            "Times are from the board's clock: the switch is the speed command's arrival, "
            "red the trip, and fan off the gate pin reading back low (the fan's power cut; "
            "spin-down is not timed). Red is stamped at the start of the loop pass that "
            "scored the deciding window, so red to fan off is mostly that window's log-mel "
            "and GMM score; the fan is cut before anything is printed. Each fan calibrated "
            f"on its own first {calib.pop()} windows after a {s.settle_s:g} s hold-off, ran "
            f"{s.normal_s:g} s watched, then was switched from {s.normal_pct} % to "
            f"{s.fault_pct} % (underspeed) by a serial command. One trial per fan; rows in "
            f"`results/results_log.csv` (`{cfg.name}`).\n\n" + table,
            encoding="utf-8",
        )
        print(table, end="")
        if args.log:
            log_rows(cfg, args.seed, trials, captures, RESULTS_LOG)
            print(f"logged {len(trials)} rows to {RESULTS_LOG}")


if __name__ == "__main__":
    main()
