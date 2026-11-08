"""The H2 recorder without a board: frames are built here from the layout documented in
firmware/src/mic_stream.h (independently of andon.record's parser), fed through the stream
parser in awkward pieces, and assembled into clips and WAV files."""

import binascii
import csv
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


def test_the_real_config_is_consistent():
    import yaml

    cfg = yaml.safe_load(Path("configs/h2_recording.yaml").read_text(encoding="utf-8"))
    rate, clip_s = cfg["audio"]["sample_rate"], cfg["audio"]["clip_seconds"]
    assert rate == 16000 and (rate * clip_s) % record.FRAME_SAMPLES == 0
    assert 0 <= cfg["audio"]["gain_shift"] <= 8  # MIC_GAIN_SHIFT_MAX
    groups = cfg["sessions"]
    assert not set(groups["dev"]) & set(groups["eval"])
    assert list(cfg["fans"]) == ["id_00", "id_01", "id_02"]
    assert list(cfg["faults"]) == ["imbalance", "obstruction"]  # rattle dropped in the pilot
    assert set(cfg["retired_faults"]) == {"rattle"}


class FakeBoard:
    """Answers commands like andon.uf2 and streams frames between rec start and rec stop."""

    def __init__(self, drop: set[int] | None = None):
        self.sent: list[str] = []
        self.out = bytearray()
        self.streaming = False
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
            build = "build=abc1234\n" if cmd == "status" else ""
            self.out += f'{build}ack cmd="{cmd}" result=ok t_rx_us=1\n'.encode()

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
