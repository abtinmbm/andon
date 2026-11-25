"""Phase H7: the soak and fault-series tools, against a scripted board that prints what the
firmware prints (firmware/src/main.c), and the reading of their captures."""

import io
import math
import time
from pathlib import Path

import pytest

from andon import h7
from andon.experiment import Config, load_config, read_results

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
FAULT = 85  # h2_recording.yaml: fault_speed_pct.underspeed
CALIB = 3


def cfg(name: str, **over) -> Config:
    raw = load_config(CONFIGS / f"{name}.yaml").raw
    key = "soak" if "soak" in raw else "series"
    return Config(name, dict(raw, **{key: dict(raw[key], **over)}))


class Board:
    """Answers commands the way main.c does and plays one window per second of board time:
    a hold-off after each start, calibration if it has no limits, then watched windows. A
    window scores high at the underspeed duty, with a fitted fault, or at the window counts
    in ``bad_at``; three high in a row trip red and cut the fan. While red, the button is
    pressed by itself after ``button_after`` quiet reads, and a fan still asked on restarts."""

    def __init__(
        self,
        limits_for: str | None = None,
        bad_at: tuple[int, ...] = (),
        hears_faults: bool = True,
        button_after: int = 8,
        yellow_at: tuple[int, ...] = (),
    ) -> None:
        self.t_us, self.n, self.clock_s = 1_000_000, 0, 0.0
        self.out: list[str] = []
        self.watch = limits_for or "none"
        self.have_limits = limits_for is not None
        self.want_on = self.red = self.fitted = False
        self.speed, self.phase, self.holdoff, self.calib_n, self.high = 100, "idle", 0, 0, 0
        self.bad_at, self.hears, self.yellow_at = set(bad_at), hears_faults, set(yellow_at)
        self.button_after, self.red_reads = button_after, 0
        self.sent: list[str] = []

    def clock(self) -> float:
        return self.clock_s

    @property
    def running(self) -> bool:
        return self.want_on and not self.red

    def _start(self) -> None:
        if self.watch != "none":
            self.phase, self.holdoff = "holdoff", 2

    def write(self, data: bytes) -> None:
        cmd = data.decode().strip()
        self.sent.append(cmd)
        result = "ok"
        if cmd == "fan on":
            if self.red:
                result = "refused_red"
            elif not self.want_on:
                self.want_on = True
                self._start()
        elif cmd == "fan off":
            self.want_on, self.phase = False, "idle"
        elif cmd.startswith("fan speed "):
            self.speed = int(cmd.split()[2])
        elif cmd.startswith("watch "):
            self.watch, self.have_limits = cmd.split()[1], False
            self.phase = "idle"
            if self.running:
                self._start()
        elif cmd == "status":
            self.out.append("build=abc1234")
        self.t_us += 500
        duty = self.speed if self.running else 0
        part = 0 < duty < 100  # a part duty toggles the pin, so it is not read back
        t_pin, pin = (-1, -1) if part else (self.t_us + 10, int(duty > 0))
        light = "red" if self.red else "green"
        self.out.append(
            f'ack cmd="{cmd}" result={result} t_rx_us={self.t_us} t_set_us={self.t_us + 3} '
            f"t_pin_us={t_pin} pin={pin} duty={duty} light={light} speed={self.speed} "
            f"want_on={int(self.want_on)} kick=0 watch={self.watch} phase={self.phase}"
        )
        if cmd == "status":
            self.out.append(
                f"detector watch={self.watch} phase={self.phase} light={light} trip=none "
                f"windows={self.n} calib={self.calib_n}/{CALIB} restarts=0 say_dropped=0"
            )
            if self.have_limits:
                self.out.append(self._limits())
            self.out.append("rec streaming=0 gain=4 frames_sent=0 frames_dropped=0 overruns=0")

    def _limits(self) -> str:
        return (
            f"limits t_us={self.t_us} fan={self.watch} n={CALIB} center=10.0000 "
            "sigma=1.0000 warn=12.0000 control=13.0000"
        )

    def _window(self) -> None:
        self.t_us += 1_024_000
        if self.phase == "holdoff":
            self.holdoff -= 1
            if self.holdoff == 0:
                self.phase = "watching" if self.have_limits else "calibrating"
                self.calib_n = 0
            return
        if self.phase == "idle":
            return
        self.n += 1
        if self.phase == "calibrating":
            self.calib_n += 1
            if self.calib_n == CALIB:
                self.have_limits, self.phase = True, "watching"
            # As on the board: the window that completes the calibration is printed after
            # the phase has moved on, so it says watching, and its limits line follows.
            self.out.append(
                f"score t_us={self.t_us} n={self.n} fan={self.watch} phase={self.phase} "
                "s=10.0000 light=green"
            )
            if self.calib_n == CALIB:
                self.out.append(self._limits())
            return
        fault = self.speed == FAULT or self.fitted
        bad = (fault and self.hears) or self.n in self.bad_at
        s = 50.0 if bad else 10.5
        self.out.append(
            f"score t_us={self.t_us} n={self.n} fan={self.watch} phase=watching s={s:.4f} "
            "light=green"
        )
        if self.n in self.yellow_at:
            self.out.append(f"light t_us={self.t_us} light=yellow trip=none reset=0")
            self.out.append(f"light t_us={self.t_us + 1} light=green trip=none reset=0")
        self.high = self.high + 1 if bad else 0
        if self.high >= 3:
            self.red, self.high, self.red_reads, self.phase = True, 0, 0, "idle"
            self.out.append(
                f"trip t_us={self.t_us + 200} reason=score fan={self.watch} n={self.n} s={s}"
            )
            self.out.append(
                f"fan update t_set_us={self.t_us + 250} t_pin_us={self.t_us + 260} pin=0 "
                "duty=0 light=red kick=0 trip=score"
            )

    def _button(self) -> None:
        self.red = False
        self.t_us += 1000
        self.out.append(f"light t_us={self.t_us} light=green trip=none reset=1")
        if self.want_on:  # the fan restarts by itself
            self.out.append(
                f"fan update t_set_us={self.t_us} t_pin_us={self.t_us + 10} pin=1 duty=100 "
                "light=green kick=1 trip=none"
            )
            self._start()

    def readline(self) -> bytes:
        self.clock_s += 0.25
        if not self.out:
            if self.red:
                self.red_reads += 1
                if self.button_after and self.red_reads >= self.button_after:
                    self._button()
            elif self.running and self.clock_s % 1.0 == 0:
                self._window()
        return (self.out.pop(0) + "\n").encode() if self.out else b""


def fitter(board: Board, asked: list[str]) -> h7.Ask:
    """The person at the bench: fits or removes the fault when asked, at once."""

    def ask(text: str):
        asked.append(text)
        if "fit the" in text:
            board.fitted = True
        elif text.startswith("make sure"):
            board.fitted = False
        return lambda: True

    return ask


def soak(board: Board, minutes: float = 0.2, recalibrate: bool = False) -> str:
    out = io.StringIO()
    c = cfg("h7_soak", minutes=minutes, max_wall_minutes=10)
    h7.run_soak(c, board, "id_00", out, recalibrate=recalibrate, clock=board.clock)
    return out.getvalue()


def series(name: str, board: Board, asked: list[str] | None = None, trials: int = 3) -> str:
    out = io.StringIO()
    c = cfg(name, trials=trials, normal_seconds=3)
    h7.run_series(
        c, board, "id_00", out, ask=fitter(board, [] if asked is None else asked), clock=board.clock
    )
    return out.getvalue()


def settings(name: str, **over) -> h7.Settings:
    return h7.Settings.from_config(cfg(name, **over))


def assert_ends_asked_off(board: Board) -> None:
    assert board.sent[-3:] == ["fan speed 100", "fan off", "status"]


def test_settings_come_from_the_configs():
    s = h7.Settings.from_config(load_config(CONFIGS / "h7_soak.yaml"))
    assert s.soak_s == 3600 and s.normal_pct == 100 and s.settle_s == 5
    assert s.calib_timeout_s > 5 + 584 * 1.024  # the hold-off plus the board's calibration
    want = {
        "h7_underspeed": ("underspeed", "switch", FAULT),
        "h7_obstruction": ("obstruction", "fit_each", 100),
        "h7_imbalance_large": ("imbalance_large", "fit_once", 100),
    }
    for name, (fault, start, pct) in want.items():
        s = h7.Settings.from_config(load_config(CONFIGS / f"{name}.yaml"))
        assert (s.fault, s.start, s.fault_pct, s.trials) == (fault, start, pct, 10)


@pytest.mark.parametrize(
    "over",
    [{"start": "switch", "fault": "obstruction"}, {"start": "fit_each", "fault": "underspeed"}],
)
def test_only_a_speed_fault_starts_by_a_switch(over):
    with pytest.raises(ValueError, match="switch"):
        settings("h7_underspeed", **over)


def test_an_unknown_fault_or_start_is_refused():
    with pytest.raises(ValueError, match="not a fault"):
        settings("h7_obstruction", fault="rattle")
    with pytest.raises(ValueError, match="start must be"):
        settings("h7_obstruction", start="sometimes")


def test_watched_time_skips_stops_and_gaps():
    w = h7.WatchClock()
    for t, phase in [(0, "watching"), (1, "watching"), (2, "holdoff"), (3, "watching")]:
        w.feed(f"score t_us={t * 1_000_000} n=1 fan=id_00 phase={phase} s=1.0 light=green")
    w.feed("score t_us=4000000 n=2 fan=id_00 phase=watching s=1.0 light=green")
    w.feed("trip t_us=4100000 reason=score fan=id_00 n=2 s=1.0")
    w.feed("score t_us=5000000 n=3 fan=id_00 phase=watching s=1.0 light=green")
    w.feed("score t_us=12000000 n=4 fan=id_00 phase=watching s=1.0 light=green")  # a gap
    assert w.seconds == pytest.approx(2.0)


def test_a_soak_with_no_stop_is_complete():
    board = Board()
    text = soak(board)
    o = h7.parse_soak(text, 12.0)
    assert not o.problems, o.problems
    assert o.fan == "id_00" and o.build == "abc1234" and o.limits.calib_windows == CALIB
    assert o.watched_s >= 12.0 and o.windows >= 12 and o.stops == [] and o.per_hour == 0
    assert o.max_score == 10.5 and o.above_control == 0 and o.restarts == 0
    assert board.sent.count("watch id_00") == 1
    assert text.startswith("# andon.h7 soak, ") and "config h7_soak, " in text.splitlines()[0]
    assert_ends_asked_off(board)


def test_a_stop_in_the_soak_is_counted_and_the_count_goes_on_after_the_button():
    board = Board(bad_at=(CALIB + 3, CALIB + 4, CALIB + 5), yellow_at=(CALIB + 1,))
    text = soak(board)
    o = h7.parse_soak(text, 12.0)
    assert not o.problems, o.problems
    assert [r for r, _ in o.stops] == ["score"] and o.stops_of(True) == 1
    assert o.stops[0][1] == pytest.approx(o.stops[0][1]) and 0 < o.stops[0][1] < 12
    assert o.watched_s >= 12.0 and o.per_hour == pytest.approx(3600 / o.watched_s)
    assert o.yellows == 1 and o.above_control == 3 and o.max_score == 50.0
    assert "reset=1" in text and "press the button" in text
    # The tool sent nothing between the setup and its closing commands: the board stopped
    # and (after the button) restarted the fan by itself.
    assert board.sent[:-3][-2:] == ["fan speed 100", "fan on"]


def test_a_soak_keeps_the_boards_limits_for_the_same_fan_unless_told():
    board = Board(limits_for="id_00")
    o = h7.parse_soak(soak(board), 12.0)
    assert not o.problems and "watch id_00" not in board.sent
    board = Board(limits_for="id_00")
    soak(board, recalibrate=True)
    assert board.sent.count("watch id_00") == 1
    board = Board(limits_for="id_01")  # another fan's limits are never used
    soak(board)
    assert board.sent.count("watch id_00") == 1


def test_a_short_soak_is_marked_incomplete():
    o = h7.parse_soak(soak(Board(), minutes=0.1), 12.0)
    assert o.problems == [f"watched {o.watched_s / 60:.1f} min of 0.2"]


def test_a_red_light_at_the_start_stops_the_tool():
    board = Board(button_after=0)
    board.red = True
    with pytest.raises(h7.TrialError, match="red"):
        soak(board)


def test_an_underspeed_series_runs_every_trial_on_one_calibration():
    board = Board()
    text = series("h7_underspeed", board)
    x = h7.parse_series(text, settings("h7_underspeed", trials=3))
    assert not x.problems and len(x.trials) == 3 and len(x.caught) == 3, [
        t.problems for t in x.trials
    ]
    assert board.sent.count("watch id_00") == 1 and x.limits.control == 13.0
    for t in x.trials:
        assert t.normal_windows >= 3 and t.max_normal_score == 10.5 and t.windows_to_red == 3
        assert t.switch_us < t.trip_us < t.stop_us and t.restarts == 0
        assert t.red_to_stop_ms == pytest.approx(0.060)
    assert board.sent.count(f"fan speed {FAULT}") == 3
    assert_ends_asked_off(board)


def test_an_obstruction_series_asks_for_the_card_and_starts_with_fan_on():
    board, asked = Board(), []
    text = series("h7_obstruction", board, asked)
    x = h7.parse_series(text, settings("h7_obstruction", trials=3))
    assert len(x.caught) == 3, [t.problems for t in x.trials]
    assert asked == [
        "wait for the blades to stop, then fit the obstruction",
        "make sure the obstruction is off (the fan is off)",
        "wait for the blades to stop, then fit the obstruction",
        "make sure the obstruction is off (the fan is off)",
        "wait for the blades to stop, then fit the obstruction",
    ]
    assert all(t.normal_windows >= 3 for t in x.trials)
    assert f"fan speed {FAULT}" not in board.sent
    assert "series done: remove the obstruction" in text


def test_a_fit_once_series_fits_the_fault_once_and_has_no_normal_stretch():
    board, asked = Board(), []
    text = series("h7_imbalance_large", board, asked)
    x = h7.parse_series(text, settings("h7_imbalance_large", trials=3))
    assert len(x.caught) == 3, [t.problems for t in x.trials]
    assert asked == ["fit the imbalance_large (the fan is off; wait for the blades)"]
    assert all(t.normal_windows == 0 for t in x.trials)


def test_a_false_stop_before_the_fault_fails_that_trial_only():
    board = Board(bad_at=(CALIB + 1, CALIB + 2, CALIB + 3))
    x = h7.parse_series(series("h7_underspeed", board), settings("h7_underspeed", trials=3))
    assert x.false_stops == 1 and len(x.caught) == 2
    t = x.trials[0]
    assert t.false_stop and not t.passed and math.isnan(t.switch_to_stop_s)
    assert t.problems == ["a stop before the fault (reason score): a false stop"]


def test_a_fault_the_board_never_hears_fails():
    board = Board(hears_faults=False)
    s = settings("h7_underspeed", trials=1, stop_timeout_seconds=3)
    out = io.StringIO()
    c = cfg("h7_underspeed", trials=1, normal_seconds=3, stop_timeout_seconds=3)
    h7.run_series(c, board, "id_00", out, ask=fitter(board, []), clock=board.clock)
    x = h7.parse_series(out.getvalue(), s)
    assert x.caught == [] and x.trials[0].problems == ["the board never went red within 3 s"]
    assert math.isnan(x.trials[0].switch_to_stop_s)


def _edit_trial(text: str, trial: int, before: str, insert: str) -> str:
    """Insert a line before the first line starting with ``before`` inside trial ``trial``."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if f" trial {trial} of " in ln)
    i = next(i for i in range(start, len(lines)) if lines[i].startswith(before))
    return "\n".join(lines[:i] + [insert] + lines[i:])


def test_a_command_or_the_button_between_the_start_and_the_stop_fails():
    text = series("h7_underspeed", Board())
    s = settings("h7_underspeed", trials=3)
    cmd = 'ack cmd="fan off" result=ok t_rx_us=1 t_set_us=1 t_pin_us=1 pin=0 duty=0 light=green'
    x = h7.parse_series(_edit_trial(text, 2, "trip ", cmd), s)
    assert x.trials[1].problems == ['"fan off" was sent before the board stopped the fan']
    assert x.trials[0].passed and x.trials[2].passed
    button = "light t_us=1 light=green trip=none reset=1"
    x = h7.parse_series(_edit_trial(text, 3, "trip ", button), s)
    assert x.trials[2].problems == ["the button was pressed before the board stopped the fan"]


def test_new_limits_inside_a_series_fail_the_trial():
    text = series("h7_underspeed", Board())
    other = "limits t_us=1 fan=id_00 n=3 center=10.0000 sigma=2.0000 warn=14.0000 control=16.0"
    x = h7.parse_series(_edit_trial(text, 2, "ack ", other), settings("h7_underspeed"))
    assert "the limits changed: the series must keep one calibration" in x.trials[1].problems
    assert x.problems == ["3 trials of 10"]


def test_a_stream_restart_in_a_trial_fails_it():
    text = series("h7_underspeed", Board(), trials=1)
    lines = text.splitlines()
    i = max(k for k, ln in enumerate(lines) if ln.startswith("detector "))
    lines[i] = lines[i].replace("restarts=0", "restarts=2")
    x = h7.parse_series("\n".join(lines[: i + 1]), settings("h7_underspeed", trials=1))
    assert x.trials[0].problems == ["the audio stream restarted 2 time(s) in the trial"]


def _captures(tmp_path: Path) -> list[Path]:
    paths = []
    for name, text in [
        ("h7_soak", soak(Board())),
        ("h7_underspeed", series("h7_underspeed", Board())),
    ]:
        p = tmp_path / f"{name}_id_00.txt"
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths


def test_report_tables_and_log_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)  # read_capture loads configs/<name>.yaml
    paths = _captures(tmp_path)
    items = [(c, x, p) for p in paths for c, _, x in [h7.read_capture(p)]]
    text = h7.report(items)
    assert "## False stops in normal running" in text and "## Fault trials" in text
    assert "| underspeed | serial switch to the fault duty | 3 | 3 | 0 |" in text
    assert "### underspeed (id_00)" in text
    log = tmp_path / "log.csv"
    h7.log_rows(items, 0, log)
    rows = read_results(log)
    assert [(r["config"], r["machine_type"], r["split"]) for r in rows] == [
        ("h7_soak", "id_00", "eval"),
        ("h7_underspeed", "id_00", "eval"),
    ]
    assert rows[0]["auc"] == "nan" and "trial=soak complete=0 " in rows[0]["notes"]
    assert "fault=underspeed start=switch trials=3 caught=3 false_stops=0 " in rows[1]["notes"]
    with pytest.raises(ValueError, match="already logged"):
        h7.log_rows(items[:1], 0, log)
    with pytest.raises(ValueError, match="one capture per config and fan"):
        h7.log_rows([items[0], items[0]], 0, tmp_path / "other.csv")


def test_rows_need_a_final_config(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    path = _captures(tmp_path)[0]
    c, _, x = h7.read_capture(path)
    draft = Config(c.name, dict(c.raw, final=False))
    with pytest.raises(ValueError, match="final"):
        h7.log_rows([(draft, x, path)], 0, tmp_path / "log.csv")


def test_the_report_command_writes_the_tables(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    paths = _captures(tmp_path)
    out = tmp_path / "h7.md"
    h7.main(["report", *map(str, paths), "--out", str(out)])
    assert out.read_text(encoding="utf-8").startswith("# H7: live measurements on the board")
    with pytest.raises(SystemExit):
        h7.main(["report", str(paths[0]), "--out", str(out), "--seed", "1"])


def test_console_ask_waits_for_enter_on_another_thread(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    done = h7.console_ask("fit the card")
    deadline = time.monotonic() + 2
    while not done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert done()


def test_the_h7_configs_are_final_and_watch_with_the_boards_model():
    for p in CONFIGS.glob("h7_*.yaml"):
        raw = load_config(p).raw
        assert raw["final"] is True and raw["phase"] == "H7" and raw["seed"] == 0
        assert raw["model_config"] == "configs/h3_gmm.yaml"
