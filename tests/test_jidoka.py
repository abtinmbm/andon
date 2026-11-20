"""Phase H6: the jidoka trial tool, against a scripted board that prints what the firmware
prints (firmware/src/main.c), and the reading of its captures."""

import io
import math
from pathlib import Path

import pytest

from andon import jidoka
from andon.experiment import Config, load_config, read_results

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
FAULT = 85  # h2_recording.yaml: fault_speed_pct.underspeed


def cfg(normal_s: float = 2.0) -> Config:
    raw = load_config(CONFIGS / "h6_jidoka.yaml").raw
    raw = dict(raw, trial=dict(raw["trial"], normal_seconds=normal_s))
    return Config("h6_jidoka", raw)


class Board:
    """Answers commands the way main.c does, and plays a stream of score lines: calibration,
    normal running, then high scores after the switch to the underspeed duty."""

    def __init__(
        self, calib: int = 3, false_stop: bool = False, stops: bool = True, restarts: int = 0
    ) -> None:
        self.t_us, self.n = 1_000_000, 0
        self.out: list[str] = []
        self.calib, self.false_stop, self.stops = calib, false_stop, stops
        self.running = self.red = self.under = False
        self.watch = "none"
        self.clock_s = 0.0
        self.restarts, self.restarts_at_end = 0, restarts

    def clock(self) -> float:
        return self.clock_s

    def write(self, data: bytes) -> None:
        cmd = data.decode().strip()
        result, duty = "ok", 0
        if cmd == "fan on":
            result = "refused_red" if self.red else "ok"
            self.running = not self.red
        elif cmd == "fan off":
            self.running = False
        elif cmd.startswith("watch "):
            self.watch = cmd.split()[1]
            result = "ok" if self.watch in ("id_00", "id_01", "id_02") else "unknown_fan"
        elif cmd == f"fan speed {FAULT}":
            self.under = True
        elif cmd == "status":
            self.out.append("build=abc1234")
        duty = (FAULT if self.under else 100) if self.running and not self.red else 0
        self.t_us += 500
        # A part duty toggles the pin, so it is not read back (t_pin_us=-1, pin=-1).
        part = 0 < duty < 100
        t_pin, pin = (-1, -1) if part else (self.t_us + 10, int(duty > 0))
        self.out.append(
            f'ack cmd="{cmd}" result={result} t_rx_us={self.t_us} t_set_us={self.t_us + 3} '
            f"t_pin_us={t_pin} pin={pin} "
            f"duty={duty} light={'red' if self.red else 'green'} speed=100 want_on=1 kick=0 "
            f"watch={self.watch} phase=calibrating"
        )
        if cmd == "status":  # main.c prints the detector and recorder lines after the ack
            self.out.append(
                f"detector watch={self.watch} phase=idle light=green trip=none windows={self.n} "
                f"calib=0/{self.calib} restarts={self.restarts} say_dropped=0"
            )
            self.out.append("rec streaming=0 gain=0 frames_sent=0 frames_dropped=0 overruns=0")
            self.restarts = self.restarts_at_end  # what the closing status will report

    def _window(self) -> None:
        # One window per second of board time.
        self.t_us += 1_024_000
        self.n += 1
        if self.n <= self.calib:
            self.out.append(
                f"score t_us={self.t_us} n={self.n} fan={self.watch} phase=calibrating "
                "s=10.0 light=green"
            )
            if self.n == self.calib:
                self.out.append(
                    f"limits t_us={self.t_us} fan={self.watch} n={self.calib} center=10.0000 "
                    "sigma=1.0000 warn=12.0000 control=13.0000"
                )
            return
        bad = self.under and self.stops or (self.false_stop and self.n == self.calib + 2)
        s = 50.0 if bad else 10.5
        self.out.append(
            f"score t_us={self.t_us} n={self.n} fan={self.watch} phase=watching s={s} light=green"
        )
        self.high = getattr(self, "high", 0) + 1 if bad else 0
        if self.high >= 3 or (self.false_stop and bad):
            self.red = True
            self.out.append(
                f"trip t_us={self.t_us + 200} reason=score fan={self.watch} n={self.n} s={s}"
            )
            self.out.append(
                f"fan update t_set_us={self.t_us + 250} t_pin_us={self.t_us + 260} pin=0 duty=0 "
                "light=red kick=0 trip=score"
            )
            self.running = False

    def readline(self) -> bytes:
        self.clock_s += 0.25
        if not self.out and self.running and not self.red and self.clock_s % 1.0 == 0:
            self._window()
        return (self.out.pop(0) + "\n").encode() if self.out else b""


def run_trial(board: Board, normal_s: float = 2.0) -> str:
    out = io.StringIO()
    jidoka.run(cfg(normal_s), board, "id_01", out, clock=board.clock)
    return out.getvalue()


def test_settings_come_from_the_configs():
    s = jidoka.TrialSettings.from_config(load_config(CONFIGS / "h6_jidoka.yaml"))
    assert (s.normal_pct, s.fault_pct) == (100, FAULT)
    assert s.calib_timeout_s > 5 + 584 * 1.024  # the hold-off plus the board's calibration


def test_a_trial_that_stops_by_itself_passes():
    text = run_trial(Board())
    t = jidoka.parse_capture(text, FAULT)
    assert t.passed, t.problems
    assert t.fan == "id_01" and t.build == "abc1234" and t.calib_windows == 3
    assert t.normal_windows >= 1 and t.max_normal_score == 10.5
    assert t.trip_reason == "score" and t.windows_to_red == 3 and t.restarts == 0
    assert t.switch_us < t.trip_us < t.stop_us
    assert t.red_to_stop_ms == pytest.approx(0.060)
    assert t.switch_to_stop_s == pytest.approx((t.stop_us - t.switch_us) / 1e6)
    # The capture is the tool's notes and every line the board printed, in order.
    assert text.startswith("# andon.jidoka, ")
    assert f'ack cmd="fan speed {FAULT}"' in text and "\ntrip " in text
    assert_ends_asked_off(text)
    assert text.rstrip().splitlines()[-1].startswith("rec ")  # the closing status, whole


def assert_ends_asked_off(text: str) -> None:
    """The last commands leave the fan off at the normal speed, so the button that clears
    red does not start it again."""
    sent = [line.split("sent: ")[1] for line in text.splitlines() if "sent: " in line]
    assert sent[-3:] == ["fan speed 100", "fan off", "status"]


def test_a_false_stop_before_the_switch_fails_and_ends_the_trial():
    text = run_trial(Board(false_stop=True), normal_s=5.0)
    assert f"sent: fan speed {FAULT}" not in text
    assert_ends_asked_off(text)
    t = jidoka.parse_capture(text, FAULT)
    assert not t.passed
    assert any("false stop" in p for p in t.problems)


def test_no_stop_after_the_switch_fails():
    t = jidoka.parse_capture(run_trial(Board(stops=False)), FAULT)
    assert "the board never went red after the switch" in t.problems


def test_a_stop_by_hand_does_not_count():
    text = run_trial(Board())
    i = text.index("\ntrip ")
    manual = 'ack cmd="fan off" result=ok t_rx_us=1 t_set_us=2 t_pin_us=3 pin=0 duty=0 light=green'
    t = jidoka.parse_capture(text[:i] + "\n" + manual + text[i:], FAULT)
    assert any('"fan off" was sent' in p for p in t.problems)
    pressed = "light t_us=5 light=green trip=none reset=1"
    t = jidoka.parse_capture(text[:i] + "\n" + pressed + text[i:], FAULT)
    assert any("button" in p for p in t.problems)


def test_a_switch_off_green_does_not_count():
    text = run_trial(Board())
    ack = next(s for s in text.splitlines() if s.startswith(f'ack cmd="fan speed {FAULT}"'))
    t = jidoka.parse_capture(text.replace(ack, ack.replace("light=green", "light=yellow")), FAULT)
    assert any("light yellow" in p for p in t.problems)


def test_a_stream_restart_in_the_trial_fails():
    t = jidoka.parse_capture(run_trial(Board(restarts=1)), FAULT)
    assert t.restarts == 1
    assert "the audio stream restarted 1 time(s) in the trial" in t.problems


def test_red_for_another_reason_fails():
    text = run_trial(Board()).replace("reason=score", "reason=mic_flat")
    t = jidoka.parse_capture(text, FAULT)
    assert any("mic_flat" in p for p in t.problems)


def test_a_missing_cut_fails():
    lines = run_trial(Board()).splitlines()
    text = "\n".join(line for line in lines if not line.startswith("fan update"))
    t = jidoka.parse_capture(text, FAULT)
    assert "no fan update cut the fan after red" in t.problems


def test_an_empty_capture_lists_what_is_missing():
    t = jidoka.parse_capture("", FAULT)
    assert {"no fan was watched", "the board never fitted its limits"} <= set(t.problems)


def test_a_red_light_at_the_start_stops_the_tool():
    board = Board()
    board.red = True
    with pytest.raises(jidoka.TrialError, match="press the button"):
        run_trial(board)


def test_table_and_log_rows(tmp_path):
    good = jidoka.parse_capture(run_trial(Board()), FAULT)
    bad = jidoka.parse_capture(run_trial(Board(stops=False)), FAULT)
    table = jidoka.jidoka_table([good, bad])
    assert "| id_01 | 3 | 13.0000 |" in table
    assert f"{good.switch_to_stop_s:.3f}" in table and "PASS" in table and "**FAIL**" in table

    log = tmp_path / "log.csv"
    good2 = jidoka.parse_capture(run_trial(Board()).replace("watch id_01", "watch id_02"), FAULT)
    paths = [Path("results/h6_jidoka_id_01.txt"), Path("results/h6_jidoka_id_02.txt")]
    jidoka.log_rows(load_config(CONFIGS / "h6_jidoka.yaml"), 0, [good, good2], paths, log)
    rows = read_results(log)
    assert [r["machine_type"] for r in rows] == ["id_01", "id_02"]
    assert all(r["split"] == "eval" and r["phase"] == "H6" and r["auc"] == "nan" for r in rows)
    assert f"switch_to_stop_s={good.switch_to_stop_s:.3f}" in rows[0]["notes"]
    assert "capture=results/h6_jidoka_id_01.txt" in rows[0]["notes"]
    with pytest.raises(ValueError, match="already logged"):
        jidoka.log_rows(load_config(CONFIGS / "h6_jidoka.yaml"), 0, [good], paths[:1], log)


def test_a_failed_trial_logs_nan_times_and_fans_are_logged_once(tmp_path):
    bad = jidoka.parse_capture(run_trial(Board(stops=False)), FAULT)
    assert math.isnan(bad.switch_to_stop_s) and math.isnan(bad.red_to_stop_ms)
    log = tmp_path / "log.csv"
    final = load_config(CONFIGS / "h6_jidoka.yaml")
    jidoka.log_rows(final, 0, [bad], [Path("x.txt")], log)
    notes = read_results(log)[0]["notes"]
    assert "passed=0 switch_to_stop_s=nan switch_to_red_s=nan red_to_stop_ms=nan" in notes
    with pytest.raises(ValueError, match="one trial per fan"):
        jidoka.log_rows(final, 0, [bad, bad], [Path("a"), Path("b")], tmp_path / "l2.csv")


def test_rows_need_a_final_config(tmp_path):
    t = jidoka.parse_capture(run_trial(Board()), FAULT)
    with pytest.raises(ValueError, match="final"):
        not_final = Config("h6_jidoka", dict(cfg().raw, final=False))
        jidoka.log_rows(not_final, 0, [t], [Path("x.txt")], tmp_path / "log.csv")


def test_globs_are_expanded_for_powershell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for f in ("h6_jidoka_id_00.txt", "h6_jidoka_id_01.txt"):
        (tmp_path / f).write_text("", encoding="utf-8")
    assert [p.name for p in jidoka._expand(["h6_jidoka_id_*.txt"])] == [
        "h6_jidoka_id_00.txt",
        "h6_jidoka_id_01.txt",
    ]
    with pytest.raises(FileNotFoundError):
        jidoka._expand(["nothing_*.txt"])


def test_the_seed_must_be_the_exported_models():
    with pytest.raises(SystemExit):
        jidoka.main(
            [
                "report",
                "--config",
                str(CONFIGS / "h6_jidoka.yaml"),
                "--seed",
                "1",
                "x",
                "--out",
                "y",
            ]
        )


def test_nan_formats_as_the_log_expects():
    assert f"{math.nan:.4f}" == "nan"


def test_committed_captures_pass_and_the_table_is_current(tmp_path, monkeypatch):
    """H6's done-when: every fan stopped by itself; the committed table is what the report
    makes from the committed captures, and the logged rows carry the same times."""
    monkeypatch.chdir(ROOT)
    captures = sorted((ROOT / "results").glob("h6_jidoka_id_*.txt"))
    assert [p.stem[-5:] for p in captures] == ["id_00", "id_01", "id_02"]
    trials = [jidoka.parse_capture(p.read_text(encoding="utf-8"), FAULT) for p in captures]
    assert all(t.passed and t.trip_reason == "score" for t in trials)
    assert {t.build for t in trials} == {"55cd2cf"}
    out = tmp_path / "h6.md"
    jidoka.main(["report", "--config", "configs/h6_jidoka.yaml", "--seed", "0",
                 "results/h6_jidoka_id_*.txt", "--out", str(out)])  # fmt: skip
    assert out.read_text(encoding="utf-8") == (ROOT / "results" / "h6_jidoka.md").read_text(
        encoding="utf-8"
    )
    rows = {r["machine_type"]: r for r in read_results(ROOT / "results" / "results_log.csv")
            if r["config"] == "h6_jidoka"}  # fmt: skip
    assert sorted(rows) == ["id_00", "id_01", "id_02"]
    for t in trials:
        assert f"switch_to_stop_s={t.switch_to_stop_s:.3f} " in rows[t.fan]["notes"]
        assert rows[t.fan]["git_commit"] == t.build
