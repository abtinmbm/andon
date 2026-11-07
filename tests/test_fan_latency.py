import sys
import types
from pathlib import Path

import pytest

from andon import fan_latency
from andon.fan_latency import LIMIT_MS, latency_table, main, mid_kick, parse_log, passed

ROOT = Path(__file__).resolve().parents[1]


def ack(cmd: str, rtt_us: int, rx: int = 1000, pin_after: int = 45, duty: int = 0, **kw) -> str:
    pin = kw.get("pin", 0 if duty == 0 else 1)
    t_pin = kw.get("t_pin", rx + pin_after)
    line = (
        f'pc_rtt_us={rtt_us} ack cmd="{cmd}" result={kw.get("result", "ok")} t_rx_us={rx} '
        f"t_set_us={rx + 2} t_pin_us={t_pin} pin={pin} duty={duty} "
        f"light={kw.get('light', 'green')} speed=100 want_on=0"
    )
    return line + (f" kick={kw['kick']}" if "kick" in kw else "")


def trial(off_rtt: int = 1000, rx: int = 1000, **off_kw) -> list[str]:
    """One accepted start at full speed followed by a stop, as the trials send them."""
    on = ack("fan on", 2000, rx=rx, duty=100)
    return [on, ack("fan off", off_rtt, rx=rx + 2_000_000, **off_kw)]


def test_parse_reads_acks_and_skips_other_lines():
    text = "\n".join(
        [
            "# andon.fan_latency, header",
            "t=5000 ms alive build=abc light=green fan duty=100",
            ack("fan off", 812, rx=2000, pin_after=37),
            ack("fan on", 900, duty=100, kick=1),
            'ack cmd="fan off" result=ok t_rx_us=1',  # board line without a PC time: skipped
        ]
    )
    off, on = parse_log(text)
    assert (off.cmd, off.result, off.pc_rtt_us, off.board_us) == ("fan off", "ok", 812, 37)
    assert off.confirmed and off.light == "green" and off.kick == -1  # old firmware: no kick=
    assert on.duty == 100 and on.pin == 1 and on.confirmed and on.kick == 1


def test_unconfirmed_pin_is_not_confirmed():
    (t,) = parse_log(ack("fan off", 800, t_pin=-1, pin=-1))
    assert not t.confirmed
    (t,) = parse_log(ack("fan off", 800, pin=1))  # read back high after "off"
    assert not t.confirmed


def test_pass_needs_every_off_confirmed_and_under_the_limit():
    limit_us = int(LIMIT_MS * 1000)
    good = trial(900) + trial(limit_us - 1)
    assert passed(parse_log("\n".join(good)))
    assert not passed(parse_log("\n".join(good + trial(limit_us))))
    assert not passed(parse_log("\n".join(good + trial(900, t_pin=-1, pin=-1))))
    assert not passed(parse_log("\n".join(good + trial(900, result="refused_red"))))
    assert not passed(parse_log(ack("fan on", 900, duty=100)))  # no "off" at all


def test_off_after_a_refused_or_powerless_start_fails():
    # The light went red mid-run: "fan on" was refused, so the next off proves nothing.
    refused = [ack("fan on", 900, result="refused_red", light="red"), ack("fan off", 900)]
    assert not passed(parse_log("\n".join(trial() + refused)))
    powerless = [ack("fan on", 900, duty=0), ack("fan off", 900)]  # e.g. speed 0
    assert not passed(parse_log("\n".join(trial() + powerless)))


def test_setup_off_is_not_a_trial():
    setup = [ack("status", 900, duty=0), ack("fan off", 900)]
    trials = parse_log("\n".join(setup + trial()))
    assert passed(trials)
    assert "| `fan off` | 1 | 1 of 1 |" in latency_table(trials)


def pair(on_rx: int, kick: int, off_rx: int):
    return parse_log(
        "\n".join([ack("fan on", 1, rx=on_rx, duty=100, kick=kick), ack("fan off", 1, rx=off_rx)])
    )


def test_mid_kick_uses_board_time():
    assert mid_kick(*pair(0, 1, 999_999))
    assert not mid_kick(*pair(0, 1, 1_000_000))  # the kick is over after FAN_KICK_MS
    assert not mid_kick(*pair(0, 0, 10))  # no kick at all


def test_table_rows_and_verdict():
    text = "\n".join(trial(1000, pin_after=30) + trial(3000, pin_after=50))
    table = latency_table(parse_log(text))
    assert "| `fan off` | 2 | 2 of 2 | 40 | 50 | 2.00 | 3.00 |" in table
    assert "| `fan on` | 2 | 2 of 2 | 45 | 45 | 2.00 | 2.00 |" in table
    last = table.splitlines()[-1]
    assert last.startswith("PASS: 2 `fan off` commands, 2 of them right after a start that ran")
    assert "(0 during a kick-start); worst PC round trip 3.00" in last


def test_table_marks_a_failure():
    table = latency_table(parse_log("\n".join(trial(250_000))))
    assert "**FAIL**" in table


class FakePico:
    """Answers like andon.uf2: one ack per command, an alive line in between, a build line
    before the status ack, and a kick-start when started from stopped at a part speed."""

    def __init__(self, *args, light: str = "green", script: list[bytes] | None = None, **kwargs):
        self.light = light
        self.lines: list[bytes] = []
        self.t_us = 1000
        self.written: list[str] = []
        self.speed = 100
        self.duty = 0
        self.script = list(script or [])  # unsolicited lines, e.g. button presses

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def reset_input_buffer(self):
        self.lines.clear()

    def write(self, data: bytes):
        cmd = data.decode().strip()
        self.written.append(cmd)
        kick = 0
        if cmd.startswith("fan speed "):
            self.speed = int(cmd.split()[-1])
        elif cmd == "fan off":
            self.duty = 0
        elif cmd == "fan on":
            kick = int(self.duty == 0 and self.speed < 100)
            self.duty = 100 if kick else self.speed
        self.t_us += 1000
        self.lines.append(b"t=1 ms alive build=abc1234 light=green fan duty=100\r\n")
        if cmd == "status":
            self.lines.append(b"build=abc1234\r\n")
        self.lines.append(
            (
                f'ack cmd="{cmd}" result=ok t_rx_us={self.t_us} t_set_us={self.t_us + 3} '
                f"t_pin_us={self.t_us + 40} pin={0 if self.duty == 0 else 1} duty={self.duty} "
                f"light={self.light} speed={self.speed} want_on=1 kick={kick}\r\n"
            ).encode()
        )

    def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        return self.script.pop(0) if self.script else b""


def fake_serial(monkeypatch, **kwargs) -> list[FakePico]:
    made: list[FakePico] = []

    def make(*args, **kw):
        made.append(FakePico(*args, **kwargs))
        return made[-1]

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=make))
    monkeypatch.setattr(fan_latency.time, "sleep", lambda s: None)
    return made


def test_run_against_a_fake_board(monkeypatch, tmp_path):
    made = fake_serial(monkeypatch)
    out = tmp_path / "log.txt"
    main(["--port", "COM9", "--trials", "3", "--out", str(out)])
    (pico,) = made
    setup = ["status", "fan off", "fan speed 100"]
    assert pico.written == setup + ["fan on", "fan off"] * 3 + ["fan speed 100"]
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# andon.fan_latency")
    assert "firmware abc1234, speed 100%, 3 trials" in text.splitlines()[0]
    trials = parse_log(text)
    assert passed(trials)
    assert "| `fan off` | 3 | 3 of 3 |" in latency_table(trials)


def test_run_at_part_speed_logs_the_kick(monkeypatch, tmp_path):
    fake_serial(monkeypatch)
    out = tmp_path / "log.txt"
    main(["--port", "COM9", "--trials", "2", "--speed", "50", "--spin", "0.5", "--out", str(out)])
    trials = parse_log(out.read_text(encoding="utf-8"))
    ons = [t for t in trials if t.cmd == "fan on"]
    assert all(t.kick == 1 and t.duty == 100 for t in ons)  # every start from stopped kicks
    assert passed(trials)
    # The fake's clock advances 1 ms per command, so every off lands inside the kick.
    assert "(2 during a kick-start)" in latency_table(trials)


def test_run_refuses_to_start_on_red(monkeypatch, tmp_path):
    fake_serial(monkeypatch, light="red")
    with pytest.raises(RuntimeError, match="light is red"):
        main(["--port", "COM9", "--out", str(tmp_path / "log.txt")])


def test_listen_records_board_lines_and_probes(monkeypatch, tmp_path):
    script = [
        b"t=7000 ms light=red\r\n",
        b"fan update t_set_us=1 t_pin_us=2 pin=0 duty=0 light=red kick=0\r\n",
    ]
    made = fake_serial(monkeypatch, script=script)
    clock = iter(x * 0.01 for x in range(100_000))  # 10 ms per clock read
    monkeypatch.setattr(fan_latency.time, "monotonic", lambda: next(clock))
    out = tmp_path / "listen.txt"
    main(["--port", "COM9", "--listen", "12", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "firmware abc1234, listen 12.0 s" in text.splitlines()[0]
    assert "t=7000 ms light=red" in text and "duty=0 light=red" in text
    probes = [c for c in made[0].written if c == "fan on"]
    assert len(probes) >= 2  # the start, then at least one probe every 5 s
    assert "sent: fan on" in text


def test_table_from_a_log(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("\n".join(trial(1500)) + "\n", encoding="utf-8")
    out = tmp_path / "table.md"
    main([str(log), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# H1: serial command to fan off")
    assert "PASS: 1 `fan off`" in text


def test_needs_a_log_or_a_port(tmp_path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "x.md")])


def test_send_prints_the_acks(monkeypatch, capsys):
    made = fake_serial(monkeypatch)
    main(["--port", "COM9", "--send", "fan speed 50", "--send", "status"])
    assert made[0].written == ["fan speed 50", "status"]
    out = capsys.readouterr().out
    assert 'ack cmd="fan speed 50" result=ok' in out and "build=abc1234" in out


def test_out_is_required_for_trials():
    with pytest.raises(SystemExit):
        main(["--port", "COM9"])


@pytest.mark.parametrize(
    ("name", "offs", "kicked"),
    [("h1_fan_latency", 20, 0), ("h1_fan_latency_50", 10, 0), ("h1_fan_latency_kick", 10, 10)],
)
def test_committed_tables_are_generated_from_the_committed_logs(tmp_path, name, offs, kicked):
    # The H1 evidence: each table in results/ must be exactly what its raw board log
    # produces, and each log must pass (100 %, 50 % after the kick, 50 % during the kick).
    log = ROOT / "results" / f"{name}.txt"
    text = log.read_text(encoding="utf-8")
    assert "firmware bafe954-dirty" in text.splitlines()[0]  # the build that was flashed
    trials = parse_log(text)
    assert passed(trials)
    out = tmp_path / "table.md"
    main([str(log), "--out", str(out)])
    regenerated = out.read_text(encoding="utf-8")
    committed = (ROOT / "results" / f"{name}.md").read_text(encoding="utf-8")
    assert regenerated.replace(log.as_posix(), f"results/{name}.txt") == committed
    assert f"PASS: {offs} `fan off` commands, {offs} of them right after a start" in committed
    assert f"({kicked} during a kick-start)" in committed
