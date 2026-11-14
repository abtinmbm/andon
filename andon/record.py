"""Record the real fan through the Pico's microphone into labelled WAV clips (Phase H2).

With ``andon.uf2`` on the board, ``rec start <gain>`` makes it stream binary frames over USB
serial (layout in ``firmware/src/mic_stream.h``): 256 int16 samples each, with a sequence
number and a CRC. The samples are the ones the C front end will see, already converted from
the mic's 24 bits with the fixed gain, so the WAV files hold exactly those samples.

A take turns the fan on, waits for it to settle, records, and turns it off again (so the fan
is always stopped when you touch it). The take is cut into 10 s clips named like the DCASE
files (``normal_id_00_00000000.wav``), and one row per clip goes into ``manifest.csv``. A clip
is kept only if all of its frames arrived in order: a dropped or corrupt frame ends the clip
early, that clip is discarded and counted, and the take's log says so.

One fan, the normal take and then each fault, with prompts in between (the usual way)::

    uv run python -m andon.record --port COM6 --session d1 --fan id_00

A single take, e.g. a short pilot at another gain, or a tone check::

    uv run python -m andon.record --port COM6 --session pilot --fan id_00 --condition normal
        --minutes 2 --gain 2

(one line in PowerShell). ``--config`` defaults to ``configs/h2_recording.yaml``.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import re
import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from andon.data.audio import write_wav

MAGIC = b"AMIC"
FRAME_SAMPLES = 256
HEADER = struct.Struct("<4sIBBH")  # magic, seq, gain shift, flags, sample count
FRAME_BYTES = HEADER.size + 2 * FRAME_SAMPLES + 2
FLAG_OVERRUN = 0x01
FLAG_CLIPPED = 0x02
MANIFEST_FIELDS = [
    "file",
    "session",
    "fan_id",
    "fan_model",
    "condition",
    "fault",
    "speed_pct",
    "gain_shift",
    "firmware_build",
    "take",
    "clip_in_take",
    "recorded_utc",
]
_BUILD = re.compile(r"build=([^\s:]+)")
_CLIP_NAME = re.compile(r"^(normal|anomaly)_(id_\d{2})_(\d{8})\.wav$")


@dataclass(frozen=True)
class Frame:
    seq: int
    gain_shift: int
    flags: int
    samples: np.ndarray  # int16


@dataclass
class StreamParser:
    """Splits the serial byte stream into frames and text lines.

    Text (acks, the alive line) can arrive between frames. A frame is found by its magic and
    accepted only if its length field and CRC check out; anything else is text or noise.
    """

    buf: bytearray = field(default_factory=bytearray)
    text: bytearray = field(default_factory=bytearray)
    corrupt: int = 0  # magic found but the CRC or length was wrong

    def feed(self, data: bytes) -> Iterator[Frame | str]:
        self.buf += data
        while True:
            at = self.buf.find(MAGIC)
            if at < 0:
                # Hold back a tail only if it could be the start of a magic split across
                # reads, so a text line's ending is never kept waiting for more data.
                cut = len(self.buf)
                for k in range(len(MAGIC) - 1, 0, -1):
                    if self.buf.endswith(MAGIC[:k]):
                        cut -= k
                        break
                yield from self._text(bytes(self.buf[:cut]))
                del self.buf[:cut]
                return
            yield from self._text(bytes(self.buf[:at]))
            del self.buf[:at]
            if len(self.buf) < FRAME_BYTES:
                return
            frame = self._frame(bytes(self.buf[:FRAME_BYTES]))
            if frame is None:
                self.corrupt += 1
                yield from self._text(bytes(self.buf[:1]))
                del self.buf[:1]  # look for the next magic after this one
            else:
                del self.buf[:FRAME_BYTES]
                yield frame

    def _text(self, data: bytes) -> Iterator[str]:
        self.text += data
        while (nl := self.text.find(b"\n")) >= 0:
            line = self.text[:nl].decode(errors="replace").strip()
            del self.text[: nl + 1]
            if line:
                yield line

    @staticmethod
    def _frame(raw: bytes) -> Frame | None:
        magic, seq, gain, flags, n = HEADER.unpack_from(raw)
        if magic != MAGIC or n != FRAME_SAMPLES:
            return None
        body = len(raw) - 2
        (crc,) = struct.unpack_from("<H", raw, body)
        # CRC-16/CCITT-FALSE, the same as mic_crc16 in the firmware.
        if binascii.crc_hqx(raw[:body], 0xFFFF) != crc:
            return None
        samples = np.frombuffer(raw, dtype="<i2", count=FRAME_SAMPLES, offset=HEADER.size)
        return Frame(seq, gain, flags, samples.copy())


@dataclass
class ClipAssembler:
    """Collects frames into whole clips; a gap in the sequence restarts the clip."""

    clip_samples: int
    expected_seq: int | None = None
    parts: list[np.ndarray] = field(default_factory=list)
    have: int = 0
    gaps: int = 0  # times the stream broke (missing frames or a board overrun)
    frames_missing: int = 0
    discarded_samples: int = 0  # samples of clips cut short by a gap

    def add(self, frame: Frame) -> np.ndarray | None:
        """Add a frame; return a finished clip (int16) when one is complete."""
        broken = frame.flags & FLAG_OVERRUN or (
            self.expected_seq is not None and frame.seq != self.expected_seq
        )
        if broken:
            self.gaps += 1
            if self.expected_seq is not None and frame.seq > self.expected_seq:
                self.frames_missing += frame.seq - self.expected_seq
            self.discarded_samples += self.have
            self.parts, self.have = [], 0
        self.expected_seq = frame.seq + 1
        self.parts.append(frame.samples)
        self.have += len(frame.samples)
        if self.have < self.clip_samples:
            return None
        # clip_samples is a whole number of frames for 10 s clips, so nothing is left over.
        clip = np.concatenate(self.parts)
        self.parts, self.have = [], 0
        return clip[: self.clip_samples]


def next_index(session_dir: Path, condition: str, fan_id: str) -> int:
    """The next free clip number for this condition and fan in the session folder."""
    used = [
        int(m[3])
        for p in session_dir.glob(f"{condition}_{fan_id}_*.wav")
        if (m := _CLIP_NAME.match(p.name))
    ]
    return max(used, default=-1) + 1


def append_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Take:
    session: str
    fan_id: str
    fault: str  # "none" for normal running
    seconds: float

    @property
    def condition(self) -> str:
        return "normal" if self.fault == "none" else "anomaly"


def check_take(take: Take, cfg: dict) -> None:
    """Refuse a take that would break the dataset's labels."""
    sessions = [s for group in cfg["sessions"].values() for s in group]
    if take.session not in sessions:
        raise ValueError(f"unknown session {take.session!r}; the config lists {sessions}")
    if take.fan_id not in cfg["fans"]:
        raise ValueError(f"unknown fan {take.fan_id!r}; the config lists {list(cfg['fans'])}")
    if take.fault != "none" and take.fault not in cfg["faults"]:
        raise ValueError(f"unknown fault {take.fault!r}; the config lists {list(cfg['faults'])}")


def save_take(
    take: Take,
    clips: list[np.ndarray],
    cfg: dict,
    build: str,
    take_id: str,
    recorded_utc: str,
) -> list[Path]:
    """Write the take's clips as WAV files and their manifest rows."""
    root = Path(cfg["data"]["root"])
    session_dir = root / take.session
    session_dir.mkdir(parents=True, exist_ok=True)
    start = next_index(session_dir, take.condition, take.fan_id)
    rate = cfg["audio"]["sample_rate"]
    paths, rows = [], []
    for i, clip in enumerate(clips):
        path = session_dir / f"{take.condition}_{take.fan_id}_{start + i:08d}.wav"
        if path.exists():
            raise FileExistsError(path)  # never overwrite a recording
        # write_wav scales by 32768, so int16 / 32768 round-trips exactly.
        write_wav(path, clip.astype(np.float32) / 32768.0, rate)
        paths.append(path)
        rows.append(
            {
                "file": path.relative_to(root).as_posix(),
                "session": take.session,
                "fan_id": take.fan_id,
                "fan_model": cfg["fans"][take.fan_id],
                "condition": take.condition,
                "fault": take.fault,
                "speed_pct": cfg["recording"]["speed_pct"],
                "gain_shift": cfg["audio"]["gain_shift"],
                "firmware_build": build,
                "take": take_id,
                "clip_in_take": i,
                "recorded_utc": recorded_utc,
            }
        )
    append_manifest(root / "manifest.csv", rows)
    return paths


def _command(port, cmd: str, log: list[str], parser: StreamParser, timeout_s: float = 2.0):
    """Send a command and wait for its ack, keeping any frames that arrive meanwhile out."""
    port.write((cmd + "\n").encode())
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for item in parser.feed(port.read(4096)):
            if isinstance(item, str):
                log.append(item)
                if item.startswith(f'ack cmd="{cmd}"'):
                    if "result=ok" not in item:
                        raise RuntimeError(f"{cmd!r} was refused: {item}")
                    return item
    raise TimeoutError(f"no ack for {cmd!r}: is andon.uf2 (H2) flashed, and the light green?")


def record_take(
    port, take: Take, cfg: dict, log: list[str], fan: bool = True
) -> tuple[list[np.ndarray], dict]:
    """Run one take on the board; return the clips and the stream statistics. With
    fan=False the fan stays off (the tone check)."""
    rate = cfg["audio"]["sample_rate"]
    clip_samples = int(cfg["audio"]["clip_seconds"] * rate)
    if clip_samples % FRAME_SAMPLES:
        raise ValueError("clip_seconds must be a whole number of 256-sample frames")
    parser = StreamParser()
    port.reset_input_buffer()
    _command(port, "status", log, parser)
    _command(port, f"fan speed {cfg['recording']['speed_pct']}", log, parser)
    if fan:
        _command(port, "fan on", log, parser)
    try:
        if fan:
            time.sleep(cfg["recording"]["settle_seconds"])
        _command(port, f"rec start {cfg['audio']['gain_shift']}", log, parser)
        assembler = ClipAssembler(clip_samples)
        clips: list[np.ndarray] = []
        frames = 0
        need = int(take.seconds * rate) // clip_samples
        while len(clips) < need:
            data = port.read(4096)
            if not data:
                raise TimeoutError("the board stopped streaming")
            for item in parser.feed(data):
                if isinstance(item, str):
                    log.append(item)
                    continue
                frames += 1
                if item.gain_shift != cfg["audio"]["gain_shift"]:
                    raise RuntimeError(f"frame gain {item.gain_shift} != config")
                if (clip := assembler.add(item)) is not None:
                    clips.append(clip)
                    print(f"  clip {len(clips)}/{need}", end="\r", flush=True)
        print()
        _command(port, "rec stop", log, parser)
    finally:
        port.write(b"rec stop\nfan off\n")  # also on Ctrl+C: never leave the fan running
    stats = {
        "frames": frames,
        "gaps": assembler.gaps,
        "frames_missing": assembler.frames_missing,
        "corrupt": parser.corrupt,
        "discarded_seconds": assembler.discarded_samples / rate,
    }
    return clips[:need], stats


def run_take(port, take: Take, cfg: dict) -> None:
    check_take(take, cfg)
    log: list[str] = []
    started = datetime.now(UTC)
    take_id = f"{take.session}-{take.fan_id}-{take.fault}-{started:%Y%m%dT%H%M%SZ}"
    print(f"{take_id}: {take.seconds / 60:g} min ...", flush=True)
    clips, stats = record_take(port, take, cfg, log)
    build = next((m[1] for line in log if (m := _BUILD.search(line))), "unknown")
    paths = save_take(take, clips, cfg, build, take_id, started.isoformat(timespec="seconds"))
    logs = Path(cfg["data"]["root"]) / take.session / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    summary = " ".join(f"{k}={v}" for k, v in stats.items())
    header = f"# andon.record {take_id} firmware {build} clips={len(paths)} {summary}"
    (logs / f"{take_id}.txt").write_text("\n".join([header, *log]) + "\n", encoding="utf-8")
    print(f"  saved {len(paths)} clips; {summary}", flush=True)


def record_tone(port, out: Path, cfg: dict) -> None:
    """One clip with the fan off, outside the dataset, for the tone check."""
    log: list[str] = []
    seconds = cfg["audio"]["clip_seconds"]
    print(f"recording {seconds} s with the fan off: play the tone now", flush=True)
    clips, stats = record_take(port, Take("pilot", "id_00", "none", seconds), cfg, log, fan=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, clips[0].astype(np.float32) / 32768.0, cfg["audio"]["sample_rate"])
    summary = " ".join(f"{k}={v}" for k, v in stats.items())
    header = f"# andon.record tone gain={cfg['audio']['gain_shift']} {summary}"
    out.with_suffix(".txt").write_text("\n".join([header, *log]) + "\n", encoding="utf-8")
    print(f"wrote {out}; {summary}", flush=True)


def guided_fan(port, session: str, fan_id: str, cfg: dict) -> None:
    """The normal take and then each fault for one fan, prompting for each change."""
    rec = cfg["recording"]
    input(f"Wire fan {fan_id} ({cfg['fans'][fan_id]}) in the mount. Press Enter to start.")
    run_take(port, Take(session, fan_id, "none", rec["normal_minutes"] * 60), cfg)
    for fault, how in cfg["faults"].items():
        input(f"The fan is off. Fault '{fault}': {how}. Press Enter when ready.")
        run_take(port, Take(session, fan_id, fault, rec["fault_minutes"] * 60), cfg)
        input(f"The fan is off. Undo '{fault}' (back to normal). Press Enter.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/h2_recording.yaml"))
    parser.add_argument("--port", required=True, help="serial port of the Pico, e.g. COM6")
    parser.add_argument("--session")
    parser.add_argument("--fan", help="machine ID, e.g. id_00")
    parser.add_argument(
        "--tone", type=Path, metavar="WAV", help="record one clip with the fan off to this file"
    )
    parser.add_argument("--condition", choices=["normal", "anomaly"], help="single take")
    parser.add_argument("--fault", default="none", help="with --condition anomaly")
    parser.add_argument("--minutes", type=float, help="single take length")
    parser.add_argument("--gain", type=int, help="pilot only: try another gain shift (0-8)")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.gain is not None:
        # The gain is part of the fixed setup; only the pilot may try others.
        if args.session != "pilot" or not 0 <= args.gain <= 8:
            parser.error("--gain is for the pilot session only, 0 to 8")
        cfg["audio"]["gain_shift"] = args.gain

    import serial  # only needed with a board attached

    if args.tone is None and (args.session is None or args.fan is None):
        parser.error("--session and --fan are required (except with --tone)")
    with serial.Serial(args.port, 115200, timeout=1.0) as port:
        if args.tone is not None:
            record_tone(port, args.tone, cfg)
            return
        if args.condition is None:
            guided_fan(port, args.session, args.fan, cfg)
            return
        if (args.condition == "anomaly") == (args.fault == "none") or args.minutes is None:
            parser.error("a single take needs --minutes, and --fault exactly for anomaly")
        run_take(port, Take(args.session, args.fan, args.fault, args.minutes * 60), cfg)


if __name__ == "__main__":
    main()
