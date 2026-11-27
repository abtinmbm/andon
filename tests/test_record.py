"""The H2 recorder without a board: frames are built here from the layout documented in
firmware/src/mic_stream.h (independently of andon.record's parser), fed through the stream
parser in awkward pieces, and assembled into clips and WAV files."""

import binascii
import contextlib
import csv
import functools
import struct
from pathlib import Path

import numpy as np
import pytest

from andon import record
from andon.data.audio import read_wav


def frame_bytes(seq: int, samples: np.ndarray, gain: int = 0, flags: int = 0) -> bytes:
    body = struct.pack("<4sIBBH", b"AMIC", seq, gain, flags, 256) + samples.astype("<i2").tobytes()
    return body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))


def ramp(seq: int) -> np.ndarray:
    return ((np.arange(256) + 256 * seq) % 65536 - 32768).astype(np.int16)


def test_frame_size_matches_the_firmware():
    assert record.FRAME_BYTES == 526  # MIC_FRAME_BYTES, checked in firmware/test/test_mic.c


def test_crc_is_the_firmwares():
    # mic_crc16("123456789") == 0x29B1 in test_mic.c
    assert binascii.crc_hqx(b"123456789", 0xFFFF) == 0x29B1


def test_parser_separates_frames_and_text_across_any_split():
    stream = (
        b'ack cmd="rec start 0" result=ok\nrec streaming=1\n'
        + frame_bytes(0, ramp(0))
        + b"t=5000 ms alive build=abc light=green fan duty=100\n"
        + frame_bytes(1, ramp(1))
        + frame_bytes(2, ramp(2))
    )
    for chunk in [1, 3, 7, 100, 525, 4096]:
        p = record.StreamParser()
        items = []
        for i in range(0, len(stream), chunk):
            items += list(p.feed(stream[i : i + chunk]))
        frames = [x for x in items if isinstance(x, record.Frame)]
        text = [x for x in items if isinstance(x, str)]
        assert [f.seq for f in frames] == [0, 1, 2]
        assert all(np.array_equal(f.samples, ramp(f.seq)) for f in frames)
        assert text == [
            'ack cmd="rec start 0" result=ok',
            "rec streaming=1",
            "t=5000 ms alive build=abc light=green fan duty=100",
        ]
        assert p.corrupt == 0


def test_a_corrupt_frame_is_dropped_and_the_next_one_found():
    bad = bytearray(frame_bytes(0, ramp(0)))
    bad[100] ^= 0x01  # one flipped bit in the samples
    p = record.StreamParser()
    items = list(p.feed(bytes(bad) + frame_bytes(1, ramp(1))))
    frames = [x for x in items if isinstance(x, record.Frame)]
    assert [f.seq for f in frames] == [1]
    assert p.corrupt == 1


def test_samples_that_look_like_the_magic_do_not_confuse_the_parser():
    s = np.zeros(256, dtype=np.int16)
    s[10:12] = np.frombuffer(b"AMIC", dtype="<i2")
    p = record.StreamParser()
    frames = [
        x for x in p.feed(frame_bytes(0, s) + frame_bytes(1, s)) if isinstance(x, record.Frame)
    ]
    assert [f.seq for f in frames] == [0, 1]


def frames(seqs, flags=None):
    flags = flags or {}
    return [record.Frame(s, 0, flags.get(s, 0), ramp(s)) for s in seqs]


def test_clips_are_whole_runs_of_frames():
    a = record.ClipAssembler(clip_samples=4 * 256)
    clips = [c for f in frames(range(9)) if (c := a.add(f)) is not None]
    assert len(clips) == 2
    assert np.array_equal(clips[1], np.concatenate([ramp(s) for s in range(4, 8)]))
    assert a.gaps == 0


def test_a_missing_frame_discards_the_clip_in_progress():
    a = record.ClipAssembler(clip_samples=4 * 256)
    # 0 1 | 3 4 5 6 : frame 2 is missing, so 0-1 are thrown away and the clip is 3-6.
    clips = [c for f in frames([0, 1, 3, 4, 5, 6]) if (c := a.add(f)) is not None]
    assert len(clips) == 1
    assert np.array_equal(clips[0], np.concatenate([ramp(s) for s in [3, 4, 5, 6]]))
    assert (a.gaps, a.frames_missing, a.discarded_samples) == (1, 1, 512)


def test_a_board_overrun_restarts_the_clip_without_a_sequence_gap():
    a = record.ClipAssembler(clip_samples=2 * 256)
    out = [
        c
        for f in frames([0, 1, 2, 3], flags={1: record.FLAG_OVERRUN})
        if (c := a.add(f)) is not None
    ]
    assert len(out) == 1
    assert np.array_equal(out[0], np.concatenate([ramp(1), ramp(2)]))
    assert a.discarded_samples == 256


@pytest.fixture
def cfg(tmp_path):
    return {
        "data": {"root": str(tmp_path / "realfan")},
        "audio": {"sample_rate": 16000, "clip_seconds": 10, "gain_shift": 2},
        "recording": {"settle_seconds": 0, "speed_pct": 100},
        "sessions": {"dev": ["s1", "s2"], "eval": ["s3"], "other": ["pilot"]},
        "fans": {"id_00": "Sunon", "id_01": "ANVISION (A)"},
        "faults": {"imbalance": "tape", "rattle": "loose screw"},
    }


def test_save_take_writes_exact_samples_and_manifest_rows(cfg, tmp_path):
    rng = np.random.default_rng(0)
    clips = [rng.integers(-32768, 32768, 160000).astype(np.int16) for _ in range(2)]
    take = record.Take("s1", "id_01", "imbalance", 20)
    paths = record.save_take(take, clips, cfg, "abc1234", "take-1", "2026-11-08T10:00:00+00:00")
    assert [p.name for p in paths] == ["anomaly_id_01_00000000.wav", "anomaly_id_01_00000001.wav"]
    audio, rate = read_wav(paths[1])
    assert rate == 16000
    assert np.array_equal(np.round(audio * 32768).astype(np.int16), clips[1])  # bit-exact
    # A second take continues the numbering instead of overwriting.
    more = record.save_take(take, clips[:1], cfg, "abc1234", "take-2", "2026-11-08T10:05:00+00:00")
    assert more[0].name == "anomaly_id_01_00000002.wav"
    with (tmp_path / "realfan" / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["file"] == "s1/anomaly_id_01_00000000.wav"
    assert rows[0]["fault"] == "imbalance" and rows[0]["condition"] == "anomaly"
    assert rows[0]["gain_shift"] == "2" and rows[0]["fan_model"] == "ANVISION (A)"
    assert rows[2]["take"] == "take-2"


def test_numbering_is_per_condition_and_fan(cfg):
    clip = [np.zeros(160000, dtype=np.int16)]
    record.save_take(record.Take("s1", "id_00", "none", 10), clip, cfg, "b", "t1", "x")
    record.save_take(record.Take("s1", "id_01", "none", 10), clip, cfg, "b", "t2", "x")
    p = record.save_take(record.Take("s1", "id_00", "none", 10), clip, cfg, "b", "t3", "x")
    assert p[0].name == "normal_id_00_00000001.wav"


@pytest.mark.parametrize(
    "take",
    [
        record.Take("s9", "id_00", "none", 10),  # unknown session
        record.Take("s1", "id_07", "none", 10),  # unknown fan
        record.Take("s1", "id_00", "tape", 10),  # unknown fault
    ],
)
def test_bad_labels_are_refused(cfg, take):
    with pytest.raises(ValueError):
        record.check_take(take, cfg)


def test_an_unseen_fault_is_accepted_only_in_its_sessions(cfg):
    cfg["sessions"]["unseen"] = ["u1"]
    cfg["unseen_faults"] = {"wobble": {"how": "tape", "sessions": ["u1"]}}
    record.check_take(record.Take("u1", "id_00", "wobble", 10), cfg)
    with pytest.raises(ValueError, match="unknown fault"):
        record.check_take(record.Take("s1", "id_00", "wobble", 10), cfg)


def test_the_real_config_is_consistent():
    import yaml

    cfg = yaml.safe_load(Path("configs/h2_recording.yaml").read_text(encoding="utf-8"))
    rate, clip_s = cfg["audio"]["sample_rate"], cfg["audio"]["clip_seconds"]
    assert rate == 16000 and (rate * clip_s) % record.FRAME_SAMPLES == 0
    assert 0 <= cfg["audio"]["gain_shift"] <= 8  # MIC_GAIN_SHIFT_MAX
    groups = cfg["sessions"]
    assert not set(groups["dev"]) & set(groups["eval"])
    assert list(cfg["fans"]) == ["id_00", "id_01", "id_02"]
    # rattle dropped in the pilot, imbalance in d1 (DECISIONS 2026-11-16)
    assert list(cfg["faults"]) == ["underspeed", "obstruction"]
    assert set(cfg["retired_faults"]) == {"rattle", "imbalance"}
    sessions = {s for group in groups.values() for s in group}
    for spec in cfg["retired_faults"].values():
        assert set(spec["sessions"]) <= sessions
    # H7's unseen fault is recorded only in the unseen sessions, which no fold trains on.
    assert set(cfg["unseen_faults"]) == {"imbalance_large"}
    for spec in cfg["unseen_faults"].values():
        assert set(spec["sessions"]) <= set(groups["unseen"])
    assert not set(groups["unseen"]) & (set(groups["dev"]) | set(groups["eval"]))
    # Each exclusion names one clip of a known session and says why.
    for name, why in cfg["excluded_clips"].items():
        assert name.split("/")[0] in sessions and name.endswith(".wav") and why
    # The fan only kick-starts and runs below full speed for a speed fault.
    rec = cfg["recording"]
    assert rec["speed_pct"] == 100
    assert set(rec["fault_speed_pct"]) <= set(cfg["faults"])
    assert all(1 <= s < rec["speed_pct"] for s in rec["fault_speed_pct"].values())


class FakeBoard:
    """Answers commands like andon.uf2 and streams frames between rec start and rec stop."""

    def __init__(self, drop: set[int] | None = None, watch: str | None = None):
        self.watch = watch  # None: an H2 build, whose acks have no watch field
        self.sent: list[str] = []
        self.out = bytearray()
        self.streaming = self.want_on = False
        self.seq = 0
        self.drop = drop or set()

    def reset_input_buffer(self):
        self.out.clear()

    def write(self, data: bytes):
        for cmd in data.decode().splitlines():
            self.sent.append(cmd)
            if cmd.startswith("rec start"):
                self.streaming, self.seq = True, 0
            elif cmd == "rec stop":
                self.streaming = False
            elif cmd == "watch none":
                self.watch = "none"
            elif cmd in ("fan on", "fan off"):
                self.want_on = cmd == "fan on"
            build = "build=abc1234\n" if cmd == "status" else ""
            watch = "" if self.watch is None else f" watch={self.watch}"
            on = int(self.want_on)
            self.out += (
                f'{build}ack cmd="{cmd}" result=ok t_rx_us=1 t_set_us=2 t_pin_us=3 pin={on} '
                f"duty={100 * on} light=green speed=100 want_on={on} kick=0{watch}\n"
            ).encode()

    def read(self, n: int) -> bytes:
        if self.streaming and len(self.out) < n:
            for _ in range(8):
                if self.seq not in self.drop:
                    self.out += frame_bytes(self.seq, ramp(self.seq), gain=2)
                self.seq += 1
        data = bytes(self.out[:n])
        del self.out[:n]
        return data


def test_a_take_runs_the_fan_records_whole_clips_and_stops_the_fan(cfg):
    cfg["audio"]["clip_seconds"] = 0.064  # 4 frames per clip keeps the test fast
    board = FakeBoard(drop={5})
    log: list[str] = []
    clips, stats = record.record_take(board, record.Take("s1", "id_00", "none", 0.192), cfg, log)
    assert board.sent[:4] == ["status", "fan speed 100", "fan on", "rec start 2"]
    assert board.sent[-2:] == ["rec stop", "fan off"]  # always ends stopped
    # Frames 0-3 make clip 1; 4 is cut short by the missing 5; 6-9 and 10-13 make clips 2-3.
    assert len(clips) == 3
    assert np.array_equal(clips[1], np.concatenate([ramp(s) for s in range(6, 10)]))
    assert stats["gaps"] == 1 and stats["frames_missing"] == 1 and stats["corrupt"] == 0
    assert any("build=abc1234" in line for line in log)


def test_a_take_refuses_to_start_while_the_board_watches_a_fan(cfg):
    # The H6 detector would stop the fan part-way through the take.
    cfg["audio"]["clip_seconds"] = 0.064
    take = record.Take("s1", "id_00", "none", 0.064)
    board = FakeBoard(watch="id_00")
    with pytest.raises(RuntimeError, match="--unwatch"):
        record.record_take(board, take, cfg, [])
    assert "fan on" not in board.sent
    board = FakeBoard(watch="id_00")
    clips, _ = record.record_take(board, take, cfg, [], unwatch=True)
    assert board.sent[:3] == ["status", "watch none", "fan speed 100"] and len(clips) == 1
    board = FakeBoard(watch="none")  # nothing watched: no extra command
    record.record_take(board, take, cfg, [])
    assert board.sent[:2] == ["status", "fan speed 100"]


def test_an_underspeed_take_runs_the_fan_slower_and_logs_its_speed(cfg, tmp_path):
    cfg["audio"]["clip_seconds"] = 0.064
    cfg["faults"]["underspeed"] = "nothing to change"
    cfg["recording"]["fault_speed_pct"] = {"underspeed": 70}
    take = record.Take("s1", "id_00", "underspeed", 0.064)
    board = FakeBoard()
    clips, _ = record.record_take(board, take, cfg, [])
    assert board.sent[:3] == ["status", "fan speed 70", "fan on"]
    record.save_take(take, clips, cfg, "b", "t1", "x")
    normal = record.Take("s1", "id_00", "none", 0.064)
    assert record.take_speed(normal, cfg) == 100  # other takes keep the normal speed
    record.save_take(normal, clips, cfg, "b", "t2", "x")
    with (tmp_path / "realfan" / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["fault"], r["speed_pct"]) for r in rows] == [("underspeed", "70"), ("none", "100")]


@pytest.mark.parametrize(
    "extra",
    [
        ["--session", "d1", "--condition", "anomaly", "--fault", "underspeed", "--speed", "70"],
        ["--session", "dpilot", "--condition", "normal", "--speed", "70"],
        ["--session", "dpilot", "--condition", "anomaly", "--fault", "underspeed", "--speed", "0"],
    ],
)
def test_trial_speeds_are_refused_outside_the_pilot_sessions(extra, capsys):
    # Refused while parsing, before any serial port is opened.
    with pytest.raises(SystemExit):
        record.main(["--port", "COM99", "--fan", "id_00", "--minutes", "1", *extra])
    assert "--speed" in capsys.readouterr().err


def test_a_trial_speed_in_an_other_session_runs_the_fan_at_that_duty(monkeypatch):
    serial = pytest.importorskip("serial")
    monkeypatch.setattr(serial, "Serial", lambda *a, **k: contextlib.nullcontext("port"))
    runs = []
    monkeypatch.setattr(record, "run_take", lambda port, take, cfg, **k: runs.append((take, cfg)))
    record.main(
        ["--port", "COM99", "--session", "dpilot", "--fan", "id_00", "--condition", "anomaly"]
        + ["--fault", "underspeed", "--minutes", "1", "--speed", "70"]
    )
    ((take, cfg),) = runs
    assert take.session == "dpilot" and record.take_speed(take, cfg) == 70


def test_the_guided_run_asks_to_undo_a_physical_fault_but_not_a_speed_fault(cfg, monkeypatch):
    cfg["faults"] = {"underspeed": "nothing", "obstruction": "card"}
    cfg["recording"].update(fault_speed_pct={"underspeed": 85}, normal_minutes=1, fault_minutes=1)
    prompts, takes = [], []
    monkeypatch.setattr("builtins.input", lambda msg: prompts.append(msg) or "")
    monkeypatch.setattr(record, "run_take", lambda port, take, c: takes.append(take.fault))
    record.guided_fan(None, "s1", "id_00", cfg)
    assert takes == ["none", "underspeed", "obstruction"]
    assert not any("Undo 'underspeed'" in m for m in prompts)
    assert sum("Undo 'obstruction'" in m for m in prompts) == 1


def test_the_tone_check_never_turns_the_fan_on(cfg, tmp_path):
    cfg["audio"]["clip_seconds"] = 0.064
    board = FakeBoard()
    out = tmp_path / "tone.wav"
    record.record_tone(board, out, cfg)
    assert "fan on" not in board.sent
    assert board.sent[-1] == "fan off"
    audio, _ = read_wav(out)
    assert len(audio) == 1024
    assert out.with_suffix(".txt").read_text(encoding="utf-8").startswith("# andon.record tone")


def test_an_error_mid_take_still_stops_the_fan(cfg):
    cfg["audio"]["clip_seconds"] = 0.064
    board = FakeBoard()
    cfg["audio"]["gain_shift"] = 3  # the fake board streams gain 2: a mismatch aborts
    with pytest.raises(RuntimeError, match="frame gain"):
        record.record_take(board, record.Take("s1", "id_00", "none", 0.128), cfg, [])
    assert board.sent[-1] == "fan off"


def test_a_text_line_is_complete_without_waiting_for_more_data():
    p = record.StreamParser()
    assert list(p.feed(b'ack cmd="fan on" result=ok t_rx_us=1\n')) == [
        'ack cmd="fan on" result=ok t_rx_us=1'
    ]
    # A magic split across reads is still found.
    f = frame_bytes(0, ramp(0))
    assert list(p.feed(b"x\nAM")) == ["x"]
    assert [i.seq for i in p.feed(f[2:]) if isinstance(i, record.Frame)] == [0]


class LostAckBoard(FakeBoard):
    """Applies every command but loses the ack of one, as a dropped serial line would."""

    def __init__(self, lost: str):
        super().__init__()
        self.lost = lost

    def write(self, data: bytes):
        before = len(self.out)
        super().write(data)
        if data.decode().strip() == self.lost:
            del self.out[before:]


def test_a_lost_fan_on_ack_still_stops_the_fan(cfg, monkeypatch):
    cfg["audio"]["clip_seconds"] = 0.064
    monkeypatch.setattr(record, "_command", functools.partial(record._command, timeout_s=0.05))
    board = LostAckBoard("fan on")
    with pytest.raises(TimeoutError, match="fan on"):
        record.record_take(board, record.Take("s1", "id_00", "none", 0.128), cfg, [])
    assert board.sent[-2:] == ["rec stop", "fan off"]


class StuckOnBoard(FakeBoard):
    """Its closing fan off ack says the fan is still asked on."""

    def write(self, data: bytes):
        super().write(data)
        if data.decode().endswith("fan off\n"):
            self.out = self.out.replace(b"want_on=0", b"want_on=1")


def test_a_take_whose_fan_off_is_not_confirmed_is_saved_and_then_stops_the_run(cfg):
    cfg["audio"]["clip_seconds"] = 0.064
    take = record.Take("s1", "id_00", "none", 0.064)
    clips, stats = record.record_take(FakeBoard(), take, cfg, [])
    assert stats["fan_off_confirmed"] == 1 and len(clips) == 1
    with pytest.raises(RuntimeError, match="unplug it"):
        record.run_take(StuckOnBoard(), take, cfg)
    assert len(list((Path(cfg["data"]["root"]) / "s1").glob("normal_id_00_*.wav"))) == 1


@pytest.mark.parametrize(("clip_seconds", "seconds"), [(0.064, 0.0), (0.064, 0.01), (0.0, 1.0)])
def test_a_take_that_could_make_no_clip_never_starts_the_fan(cfg, clip_seconds, seconds):
    cfg["audio"]["clip_seconds"] = clip_seconds
    board = FakeBoard()
    with pytest.raises(ValueError):
        record.record_take(board, record.Take("s1", "id_00", "none", seconds), cfg, [])
    assert board.sent == []


def test_noise_with_no_newline_does_not_grow_the_text_buffer():
    p = record.StreamParser()
    for _ in range(64):
        list(p.feed(b"x" * 1024))
    assert len(p.text) <= record.MAX_LINE_BYTES
    assert list(p.feed(b"\nack ok\n")) == ["ack ok"]  # the next whole line still reads


def test_a_recording_is_created_exclusively(cfg):
    take = record.Take("s1", "id_00", "none", 10.0)
    clip = np.zeros(1024, dtype=np.int16)
    (path,) = record.save_take(take, [clip], cfg, "b", "t1", "x")
    real_next = record.next_index
    try:
        record.next_index = lambda *a: 0  # a second recorder that picked the same number
        with pytest.raises(FileExistsError):
            record.save_take(take, [clip], cfg, "b", "t2", "x")
    finally:
        record.next_index = real_next
