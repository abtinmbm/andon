"""Measure how fast a serial command cuts the fan (Phase H1).

With ``andon.uf2`` on the Pico, this sends ``fan on``, lets the fan spin up, sends ``fan off``,
and repeats. Every command is answered with an ``ack`` line that carries the Pico's own timer:
when the command arrived (``t_rx_us``) and when the MOSFET's gate pin read back at its new
level (``t_pin_us``). On the PC side it times each command from the write to the ack
arriving, which covers USB both ways, so it is an upper bound on command-to-off.

"Off" here means GP16, which drives the MOSFET's gate, reads back low: the fan has no power.
The blades then coast to a stop over a few seconds, which no kill switch can shorten for a
free-spinning fan. The log header names the firmware build the board reported.

Run on the board, then regenerate the table from the committed log::

    uv run python -m andon.fan_latency --port COM6 --trials 20 --out results/h1_fan_latency.txt
    uv run python -m andon.fan_latency results/h1_fan_latency.txt --out results/h1_fan_latency.md

At a part speed each start is a kick-start (full speed for the firmware's FAN_KICK_MS), so a
short ``--spin`` sends ``fan off`` during the kick and a long one after it::

    uv run python -m andon.fan_latency --port COM6 --speed 50 --spin 0.5 --out ...

Record everything the board prints while you press the button (``fan on`` is sent every few
seconds, so a refusal on red shows up too)::

    uv run python -m andon.fan_latency --port COM6 --listen 60 --out results/h1_andon_red.txt

Send commands by hand, e.g. to try a lower speed::

    uv run python -m andon.fan_latency --port COM6 --send "fan speed 50"
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

# PLAN.md, Phase H1 "done when": a serial command turns the fan off within this long.
LIMIT_MS = 100.0
# firmware/src/fan.h, FAN_KICK_MS: a start from stopped at a part speed runs full speed this long.
KICK_MS = 1000

_ACK = re.compile(
    r'^pc_rtt_us=(\d+) ack cmd="([^"]*)" result=(\w+) t_rx_us=(\d+) t_set_us=(\d+) '
    r"t_pin_us=(-?\d+) pin=(-?\d+) duty=(\d+) light=(\w+)"
)
_KICK = re.compile(r" kick=(\d)")
_BUILD = re.compile(r"^build=(\S+)")


@dataclass
class Trial:
    cmd: str
    result: str
    pc_rtt_us: int
    t_rx_us: int
    t_set_us: int
    t_pin_us: int  # -1 when the pin was not read back at the expected level
    pin: int
    duty: int
    light: str
    kick: int = -1  # 1 while kick-starting, 0 if not, -1 for firmware that didn't report it

    @property
    def confirmed(self) -> bool:
        """The gate pin read back at the level the command asked for."""
        want = 0 if self.duty == 0 else 1
        return self.t_pin_us >= 0 and self.pin == want

    @property
    def board_us(self) -> int:
        """Pico time from the command's last character to the pin at its new level."""
        return self.t_pin_us - self.t_rx_us


def parse_log(text: str) -> list[Trial]:
    """Every timed ack in the log; other board lines (status, comments) are skipped."""
    trials = []
    for line in text.splitlines():
        line = line.strip()
        if m := _ACK.match(line):
            k = _KICK.search(line)
            trials.append(
                Trial(
                    cmd=m[2],
                    result=m[3],
                    pc_rtt_us=int(m[1]),
                    t_rx_us=int(m[4]),
                    t_set_us=int(m[5]),
                    t_pin_us=int(m[6]),
                    pin=int(m[7]),
                    duty=int(m[8]),
                    light=m[9],
                    kick=int(k[1]) if k else -1,
                )
            )
    return trials


def _offs_with_their_on(trials: list[Trial]) -> list[tuple[Trial, Trial]]:
    """The trials: each ``fan off`` that directly follows a ``fan on``, paired with it. Any
    other ``fan off`` (stopping the fan before the first trial) is setup, not a trial."""
    return [
        (trials[i - 1], t)
        for i, t in enumerate(trials)
        if t.cmd == "fan off" and i > 0 and trials[i - 1].cmd == "fan on"
    ]


def _was_running(on: Trial) -> bool:
    """The ``fan on`` before the off was accepted on green and powered the fan."""
    return on.result == "ok" and on.duty > 0 and on.light == "green"


def mid_kick(on: Trial, off: Trial) -> bool:
    """The ``fan off`` arrived while the start before it was still kick-starting."""
    return on.kick == 1 and off.t_rx_us - on.t_rx_us < KICK_MS * 1000


def passed(trials: list[Trial]) -> bool:
    """Done-when check: every ``fan off`` follows a start that ran, is confirmed at the pin,
    and is acked within LIMIT_MS on the PC."""
    pairs = _offs_with_their_on(trials)
    return bool(pairs) and all(
        _was_running(on)
        and off.result == "ok"
        and off.confirmed
        and off.pc_rtt_us / 1000 < LIMIT_MS
        for on, off in pairs
    )


def latency_table(trials: list[Trial]) -> str:
    """One row per command, then the pass/fail line for the H1 done-when check."""
    rows = [
        "| Command | Trials | Pin confirmed | Pico rx to pin, median (us) "
        "| Pico rx to pin, max (us) | PC round trip, median (ms) | PC round trip, max (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    pairs = _offs_with_their_on(trials)
    for cmd, ts in (("fan off", [off for _, off in pairs]), ("fan on", [on for on, _ in pairs])):
        if not ts:
            continue
        ok = [t for t in ts if t.confirmed]
        board = [t.board_us for t in ok]
        rtt = [t.pc_rtt_us / 1000 for t in ts]
        board_med = f"{median(board):.0f}" if board else "n/a"
        board_max = f"{max(board)}" if board else "n/a"
        rows.append(
            f"| `{cmd}` | {len(ts)} | {len(ok)} of {len(ts)} | {board_med} | {board_max} "
            f"| {median(rtt):.2f} | {max(rtt):.2f} |"
        )
    worst = max((off.pc_rtt_us / 1000 for _, off in pairs), default=float("nan"))
    running = sum(_was_running(on) for on, _ in pairs)
    kicked = sum(mid_kick(on, off) for on, off in pairs)
    verdict = "PASS" if passed(trials) else "**FAIL**"
    rows += [
        "",
        f"{verdict}: {len(pairs)} `fan off` commands, {running} of them right after a start "
        f"that ran ({kicked} during a kick-start); worst PC round trip {worst:.2f} ms against "
        f"the {LIMIT_MS:.0f} ms limit (PLAN.md, Phase H1). The round trip runs from the PC "
        "writing the command to the Pico's ack arriving, so it includes USB both ways; the "
        "Pico's own time from the command to GP16 (the MOSFET's gate drive) reading low is the "
        "rx-to-pin column. The fan then has no power and its blades coast to a stop.",
    ]
    return "\n".join(rows) + "\n"


def _send(port, cmd: str, log: list[str], timeout_s: float = 1.0) -> Trial:
    """Write one command, wait for its ack and log it with the PC round trip."""
    port.reset_input_buffer()
    t0 = time.perf_counter_ns()
    port.write((cmd + "\n").encode())
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        raw = port.readline()
        t1 = time.perf_counter_ns()
        line = raw.decode(errors="replace").strip()
        if not line:
            continue
        if line.startswith(f'ack cmd="{cmd}"'):
            logged = f"pc_rtt_us={(t1 - t0) // 1000} {line}"
            log.append(logged)
            (trial,) = parse_log(logged)
            return trial
        log.append(line)
    raise TimeoutError(f"no ack for {cmd!r} within {timeout_s} s: is andon.uf2 flashed?")


def _build(log: list[str]) -> str:
    """The firmware build the board reported in its status reply, if any."""
    for line in log:
        if m := _BUILD.match(line):
            return m[1]
    return "unknown"


def _header(port_name: str, details: str, build: str) -> str:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    return f"# andon.fan_latency, {stamp}, port {port_name}, firmware {build}, {details}"


def run(port_name: str, trials: int, spin_s: float, stop_s: float, speed: int = 100) -> str:
    """Alternate fan on / fan off on the board; return the raw log."""
    import serial  # only needed with a board attached

    log: list[str] = []
    with serial.Serial(port_name, 115200, timeout=0.2) as port:
        status = _send(port, "status", log)
        if status.light == "red":
            raise RuntimeError("the light is red: press the button to reset it, then rerun")
        _send(port, "fan off", log)  # start every trial from stopped
        time.sleep(stop_s)
        _send(port, f"fan speed {speed}", log)
        for i in range(trials):
            on = _send(port, "fan on", log)
            time.sleep(spin_s)
            off = _send(port, "fan off", log)
            print(
                f"trial {i + 1}/{trials}: off acked in {off.pc_rtt_us / 1000:.2f} ms "
                f"(Pico rx to pin {off.board_us} us, pin={off.pin}); on {on.result}"
                + (" (kick)" if on.kick == 1 else ""),
                flush=True,
            )
            time.sleep(stop_s)
        _send(port, "fan speed 100", log)
    details = f"speed {speed}%, {trials} trials, spin {spin_s} s, stop {stop_s} s"
    return "\n".join([_header(port_name, details, _build(log)), *log]) + "\n"


def listen(port_name: str, seconds: float, probe_s: float = 5.0) -> str:
    """Record every line the board prints for a while, sending ``fan on`` every probe_s
    seconds so a refusal on red is logged too; return the raw log."""
    import serial  # only needed with a board attached

    log: list[str] = []
    with serial.Serial(port_name, 115200, timeout=0.2) as port:
        _send(port, "status", log)
        _send(port, "fan on", log)
        start = time.monotonic()
        next_probe = start + probe_s
        while (now := time.monotonic()) < start + seconds:
            if now >= next_probe:
                port.write(b"fan on\n")
                log.append(f"pc_s={now - start:.1f} sent: fan on")
                next_probe += probe_s
            line = port.readline().decode(errors="replace").strip()
            if line:
                log.append(f"pc_s={time.monotonic() - start:.1f} {line}")
                print(line, flush=True)
    details = f"listen {seconds} s, fan on every {probe_s} s"
    return "\n".join([_header(port_name, details, _build(log)), *log]) + "\n"


def send(port_name: str, commands: list[str]) -> list[str]:
    """Send commands by hand (e.g. to try a fan speed) and return the board's replies."""
    import serial  # only needed with a board attached

    log: list[str] = []
    with serial.Serial(port_name, 115200, timeout=0.2) as port:
        for cmd in commands:
            _send(port, cmd, log)
    return log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", nargs="?", type=Path, help="existing log to tabulate")
    parser.add_argument("--port", help="serial port of the Pico (e.g. COM6) to run trials")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--speed", type=int, default=100, help="fan speed (%%) for the trials")
    parser.add_argument("--spin", type=float, default=2.0, help="seconds on before each off")
    parser.add_argument("--stop", type=float, default=1.0, help="seconds off before each on")
    parser.add_argument("--listen", type=float, metavar="SECONDS", help="record board output")
    parser.add_argument(
        "--send",
        action="append",
        metavar="CMD",
        help='with --port: send these commands (e.g. "fan speed 50") and print the acks, '
        "instead of running trials",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.out is None and not (args.port and args.send):
        parser.error("--out is required unless sending commands with --send")

    if args.port and args.send:
        for line in send(args.port, args.send):
            print(line)
    elif args.port and args.listen:
        args.out.write_text(listen(args.port, args.listen), encoding="utf-8")
        print(f"wrote {args.out}")
    elif args.port:
        text = run(args.port, args.trials, args.spin, args.stop, args.speed)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
        print(latency_table(parse_log(text)), end="")
    elif args.log:
        table = latency_table(parse_log(args.log.read_text(encoding="utf-8")))
        args.out.write_text(
            f"# H1: serial command to fan off\n\nGenerated by `andon.fan_latency` from "
            f"`{args.log.as_posix()}`.\n\n{table}",
            encoding="utf-8",
        )
        print(table, end="")
    else:
        parser.error("give a log to tabulate or --port to run trials")


if __name__ == "__main__":
    main()
