"""Check the real-fan dataset and the mic (Phase H2).

``check`` verifies every clip against the manifest and the recording config (16 kHz, mono,
16-bit, exactly one clip length, names that match their labels, no stray files), adds up
what the stream lost (from the take logs), measures each clip's level, and writes a table
per session, fan and condition. Each fault take is also compared with the normal running of the
same fan in the same session, band by band, so a fault that does not change the sound shows up
as a row of zeros. It exits non-zero if anything is wrong. ``--listen N`` adds
N random clips per row for the listening check (``--config`` defaults to
``configs/h2_recording.yaml``)::

    uv run python -m andon.realfan check --listen 2 --out results/h2_dataset.md

``tone`` finds the strongest frequency in a clip, to check the sample rate with a known tone
(e.g. a phone playing 1 kHz next to the mic)::

    uv run python -m andon.realfan tone data/realfan/pilot/normal_id_00_00000000.wav
        --expect 1000 --out results/h2_mic_tone.md

(one line in PowerShell).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import wave
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml

from andon.record import MANIFEST_FIELDS

_CLIP_NAME = re.compile(r"^(normal|anomaly)_(id_\d{2})_(\d{8})\.wav$")
_LOG_STATS = re.compile(r"(\w+)=([\d.]+)")
FULL_SCALE = 32768.0
# Octave-wide bands for the fault-vs-normal table; the top edge is the Nyquist frequency.
BAND_EDGES_HZ = (0, 125, 250, 500, 1000, 2000, 4000)


def dbfs(x: float) -> float:
    return 20 * np.log10(max(x, 1e-12) / FULL_SCALE)


@dataclass(frozen=True)
class ClipStats:
    peak_dbfs: float
    rms_dbfs: float
    dc: float  # mean sample value, in int16 steps
    clipped: int  # samples at the int16 limits


def read_pcm16(path: Path) -> tuple[np.ndarray, int, int]:
    """(int16 samples, sample rate, channels); raises on anything but 16-bit PCM."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("not 16-bit PCM")
        raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype="<i2"), wf.getframerate(), wf.getnchannels()


def clip_stats(samples: np.ndarray) -> ClipStats:
    x = samples.astype(np.float64)
    return ClipStats(
        peak_dbfs=dbfs(float(np.max(np.abs(x)))),
        rms_dbfs=dbfs(float(np.sqrt(np.mean(x**2)))),
        dc=float(np.mean(x)),
        clipped=int(np.sum((samples == 32767) | (samples == -32768))),
    )


def band_powers(samples: np.ndarray, rate: int, gain_shift: int = 0) -> np.ndarray:
    """Power per band of BAND_EDGES_HZ (up to rate / 2), referred to gain 0.

    Dividing out the recording gain (2**gain_shift in amplitude) lets takes recorded at
    different gains be compared directly.
    """
    x = samples.astype(np.float64) / 2.0**gain_shift
    x -= x.mean()
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), 1 / rate)
    edges = (*BAND_EDGES_HZ, rate / 2 + 1)
    return np.array([power[(freq >= lo) & (freq < hi)].sum() for lo, hi in pairwise(edges)])


def read_take_logs(root: Path) -> list[dict[str, float]]:
    """The stream statistics in each take log's header line."""
    out = []
    for log in sorted(root.glob("*/logs/*.txt")):
        header = log.read_text(encoding="utf-8").splitlines()[0]
        stats = {k: float(v) for k, v in _LOG_STATS.findall(header)}
        stats["session"] = log.parent.parent.name  # type: ignore[assignment]
        out.append(stats)
    return out


def check(cfg: dict, listen: int = 0, seed: int = 0) -> tuple[str, list[str]]:
    """Return the markdown report and the list of problems found (empty = clean)."""
    root = Path(cfg["data"]["root"])
    rate = cfg["audio"]["sample_rate"]
    clip_len = int(cfg["audio"]["clip_seconds"] * rate)
    sessions = {s: group for group, names in cfg["sessions"].items() for s in names}
    problems: list[str] = []
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return "", [f"{manifest} not found"]
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != MANIFEST_FIELDS:
            problems.append(f"manifest columns {reader.fieldnames} != {MANIFEST_FIELDS}")
        rows = list(reader)

    cells: dict[tuple, list[tuple[str, ClipStats]]] = defaultdict(list)
    bands: dict[tuple, list[np.ndarray]] = defaultdict(list)  # (session, fan, fault, take)
    listed = set()
    excluded: dict[str, str] = cfg.get("excluded_clips") or {}
    excluded_stats: dict[str, ClipStats] = {}
    excluded_takes: dict[str, str] = cfg.get("excluded_takes") or {}
    excluded_take_clips: dict[str, int] = defaultdict(int)
    for row in rows:
        name = row["file"]
        if name in listed:
            problems.append(f"{name}: listed twice in the manifest")
        listed.add(name)
        m = _CLIP_NAME.match(Path(name).name)
        if m is None or m[1] != row["condition"] or m[2] != row["fan_id"]:
            problems.append(f"{name}: file name does not match its labels")
        if Path(name).parent.name != row["session"] or row["session"] not in sessions:
            problems.append(f"{name}: session {row['session']!r} is not a configured session")
        if row["fan_id"] not in cfg["fans"]:
            problems.append(f"{name}: unknown fan {row['fan_id']}")
        if (row["condition"] == "normal") != (row["fault"] == "none"):
            problems.append(f"{name}: condition {row['condition']} with fault {row['fault']}")
        # A dropped fault stays readable in the sessions it was recorded in, never elsewhere.
        known = set(cfg["faults"])
        known |= {
            name
            for name, spec in cfg.get("retired_faults", {}).items()
            if row["session"] in spec["sessions"]
        }
        if row["fault"] != "none" and row["fault"] not in known:
            problems.append(f"{name}: unknown fault {row['fault']}")
        if int(row["gain_shift"]) != cfg["audio"]["gain_shift"] and row["session"] != "pilot":
            problems.append(f"{name}: gain {row['gain_shift']} != config")
        # Trial duties belong to the 'other' sessions only; a dev or eval clip must have been
        # recorded at the duty its fault is defined by, or its label means something else.
        if sessions.get(row["session"]) not in (None, "other"):
            rec = cfg["recording"]
            want = rec.get("fault_speed_pct", {}).get(row["fault"], rec["speed_pct"])
            if int(row["speed_pct"]) != int(want):
                problems.append(f"{name}: speed {row['speed_pct']} % != config {want} %")
        path = root / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        try:
            samples, sr, ch = read_pcm16(path)
        except (ValueError, wave.Error, EOFError) as e:
            problems.append(f"{name}: unreadable ({e})")
            continue
        if sr != rate or ch != 1 or len(samples) != clip_len:
            problems.append(f"{name}: {sr} Hz, {ch} ch, {len(samples)} samples")
            continue
        st = clip_stats(samples)
        # An excluded clip is still checked as a file, but its known defect (e.g. a knock that
        # clipped) is not a problem, and it stays out of the tables, the bands and listening.
        if name in excluded:
            excluded_stats[name] = st
            continue
        # A whole take whose labels are wrong (e.g. recorded under another fan's ID) is kept
        # as recorded, never relabelled; it is verified as files and left out of every use.
        if row["take"] in excluded_takes:
            excluded_take_clips[row["take"]] += 1
            continue
        if st.clipped:
            problems.append(f"{name}: {st.clipped} clipped samples")
        cells[(row["session"], row["fan_id"], row["fault"])].append((name, st))
        key = (row["session"], row["fan_id"], row["fault"], row["take"])
        bands[key].append(band_powers(samples, rate, int(row["gain_shift"])))

    on_disk = {p.relative_to(root).as_posix() for p in root.glob("*/*.wav")}
    for stray in sorted(on_disk - listed):
        problems.append(f"{stray}: on disk but not in the manifest")
    for name in sorted(set(excluded) - listed):
        problems.append(f"{name}: excluded in the config but not in the manifest")
    takes = {row["take"] for row in rows}
    for name in sorted(set(excluded_takes) - takes):
        problems.append(f"{name}: take excluded in the config but not in the manifest")

    logs = read_take_logs(root)
    lost = defaultdict(lambda: defaultdict(float))
    for st in logs:
        for k in ("frames", "gaps", "frames_missing", "corrupt", "discarded_seconds"):
            lost[st["session"]][k] += st.get(k, 0.0)
    for session, t in lost.items():
        if session in sessions and sessions[session] != "other" and (t["gaps"] or t["corrupt"]):
            problems.append(
                f"{session}: stream broke {t['gaps']:.0f} times, {t['corrupt']:.0f} corrupt frames"
            )

    order = {s: i for i, s in enumerate(sessions)}
    lines = [
        "| Session | Split | Fan | Fault | Clips | Minutes | Median RMS (dBFS) | Max peak (dBFS) "
        "| DC (LSB, mean) | Clipped samples | Loudest clip |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in sorted(cells, key=lambda k: (order.get(k[0], 99), k[1], k[2] != "none", k[2])):
        session, fan, fault = key
        stats = [s for _, s in cells[key]]
        # Named so an outlier peak (a knock, a bump) can be found and listened to.
        loudest = max(cells[key], key=lambda c: c[1].peak_dbfs)[0]
        lines.append(
            f"| {session} | {sessions.get(session, '?')} | {fan} | {fault} | {len(stats)} "
            f"| {len(stats) * clip_len / rate / 60:.1f} "
            f"| {np.median([s.rms_dbfs for s in stats]):.1f} "
            f"| {max(s.peak_dbfs for s in stats):.1f} "
            f"| {np.mean([s.dc for s in stats]):.1f} | {sum(s.clipped for s in stats)} "
            f"| `{loudest}` |"
        )
    lines += [
        "",
        "Stream, from the take logs (frames received; breaks; frames missing; corrupt; "
        "seconds of partial clips discarded):",
        "",
        "| Session | Takes | Frames | Breaks | Missing | Corrupt | Discarded (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for session in sorted(lost, key=lambda s: order.get(s, 99)):
        t = lost[session]
        n = sum(1 for st in logs if st["session"] == session)
        lines.append(
            f"| {session} | {n} | {t['frames']:.0f} | {t['gaps']:.0f} | {t['frames_missing']:.0f} "
            f"| {t['corrupt']:.0f} | {t['discarded_seconds']:.2f} |"
        )
    lines += fault_vs_normal(bands, rate, order)
    repeat_lines, repeat_problems = repeatability(bands, cfg, sessions)
    lines += repeat_lines
    problems += repeat_problems
    if excluded_stats:
        lines += ["", "Excluded clips (kept on disk, left out of everything above):", ""]
        for name, st in sorted(excluded_stats.items()):
            lines.append(
                f"- `{name}`: {excluded[name]} (peak {st.peak_dbfs:.1f} dBFS, "
                f"{st.clipped} clipped samples)"
            )
    if excluded_take_clips:
        lines += ["", "Excluded takes (kept on disk, left out of everything above):", ""]
        for name, n in sorted(excluded_take_clips.items()):
            lines.append(f"- `{name}`: {excluded_takes[name]} ({n} clips)")
    if listen:
        rng = random.Random(seed)
        lines += ["", f"Listening check ({listen} random clips per row, seed {seed}):", ""]
        for key in sorted(cells, key=lambda k: (order.get(k[0], 99), k[1], k[2])):
            names = sorted(n for n, _ in cells[key])
            for name in rng.sample(names, min(listen, len(names))):
                lines.append(f"- [ ] `{name}` ({key[2]})")
    lines += ["", f"Problems: {len(problems)}"] + [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n", problems


def _band_change(bands: dict[tuple, list[np.ndarray]], session: str, fan: str, fault: str):
    """Mean band level of a session's fault clips minus its normal clips (dB), or None."""
    normal = [
        b for (s, f, x, _), v in bands.items() if (s, f, x) == (session, fan, "none") for b in v
    ]
    fault_b = [
        b for (s, f, x, _), v in bands.items() if (s, f, x) == (session, fan, fault) for b in v
    ]
    if not normal or not fault_b:
        return None
    return 10 * np.log10(np.mean(fault_b, axis=0) / np.mean(normal, axis=0))


def repeatability(
    bands: dict[tuple, list[np.ndarray]], cfg: dict, sessions: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Compare listed (fan, fault) pairs in each dev/eval session with a reference session.

    A fault whose size drifts between sessions would make eval score something other than
    what dev tuned on. The band compared is the reference's biggest change; a session outside
    the tolerance is a problem (DECISIONS 2026-11-16: in dev, stop and decide; in eval, report).
    """
    spec = cfg.get("repeatability")
    if not spec:
        return [], []
    ref, tol = spec["reference_session"], float(spec["tolerance_db"])
    edges = (*BAND_EDGES_HZ, cfg["audio"]["sample_rate"] // 2)
    names = [f"{lo}-{hi} Hz" for lo, hi in pairwise(edges)]
    lines = [
        "",
        f"Repeatability against {ref} (change from normal in the reference's biggest band, "
        f"tolerance ±{tol:g} dB):",
        "",
        "| Fan | Fault | Band | Session | Change (dB) | Difference from reference (dB) |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    problems = []
    for fan, fault in spec["pairs"]:
        base = _band_change(bands, ref, fan, fault)
        if base is None:
            problems.append(f"repeatability: no {fault} take of {fan} in {ref}")
            continue
        k = int(np.argmax(np.abs(base)))
        lines.append(f"| {fan} | {fault} | {names[k]} | {ref} | {base[k]:+.1f} | |")
        for session, group in sessions.items():
            if session == ref or group == "other":
                continue
            d = _band_change(bands, session, fan, fault)
            if d is None:
                continue
            gap = d[k] - base[k]
            lines.append(f"| {fan} | {fault} | {names[k]} | {session} | {d[k]:+.1f} | {gap:+.1f} |")
            if abs(gap) > tol:
                problems.append(
                    f"{session}: {fan} {fault} is {gap:+.1f} dB from {ref} at {names[k]} "
                    f"(tolerance ±{tol:g})"
                )
    return lines, problems


def usable_rows(rows: list[dict], cfg: dict) -> list[dict]:
    """The manifest rows a model may use: no excluded clips or takes, no retired faults, and
    none of the clips at the start of a take that the config says are still settling.

    Every loader of real-fan clips for training or scoring goes through this, so the rules in
    the recording config are applied in one place.
    """
    excluded = cfg.get("excluded_clips") or {}
    excluded_takes = cfg.get("excluded_takes") or {}
    retired = set(cfg.get("retired_faults") or {})
    skip = cfg.get("skip_first_clips") or {}
    return [
        r
        for r in rows
        if r["file"] not in excluded
        and r["take"] not in excluded_takes
        and r["fault"] not in retired
        and int(r["clip_in_take"]) >= int(skip.get(r["fault"], 0))
    ]


def fault_vs_normal(bands: dict[tuple, list[np.ndarray]], rate: int, order: dict) -> list[str]:
    """Rows of mean band level of each fault take minus the same fan's normal clips (dB)."""
    normal: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for (session, fan, fault, _), b in bands.items():
        if fault == "none":
            normal[(session, fan)] += b
    edges = (*BAND_EDGES_HZ, rate // 2)
    names = [f"{lo}-{hi}" for lo, hi in pairwise(edges)]
    lines = [
        "",
        "Fault takes against normal running of the same fan in the same session: mean band "
        "level of the take minus that of the normal clips, in dB, at gain 0. Near 0 in every "
        "band means the fault does not change the sound. The last column is the range over the "
        "take's clips in the band that changed most, to show whether the change is steady or "
        "comes from a few clips.",
        "",
        "| Session | Fan | Fault | Take | "
        + " | ".join(f"{n} Hz" for n in names)
        + " | Per clip, biggest band |",
        "| --- | --- | --- | --- |" + " ---: |" * len(names) + " --- |",
    ]
    rows = sorted(bands, key=lambda k: (order.get(k[0], 99), k[1], k[2], k[3]))
    for session, fan, fault, take in rows:
        if fault == "none" or (session, fan) not in normal:
            continue
        ref = np.mean(normal[(session, fan)], axis=0)
        clips = np.array(bands[(session, fan, fault, take)])
        diff = 10 * np.log10(clips.mean(axis=0) / ref)
        k = int(np.argmax(np.abs(diff)))
        per_clip = 10 * np.log10(clips[:, k] / ref[k])
        lines.append(
            f"| {session} | {fan} | {fault} | {take} | "
            + " | ".join(f"{d:+.1f}" for d in diff)
            + f" | {names[k]} Hz: {per_clip.min():+.1f} to {per_clip.max():+.1f} |"
        )
    return lines


def dominant_frequency(samples: np.ndarray, rate: int) -> tuple[float, float]:
    """(frequency in Hz of the strongest bin, its level above the median bin in dB)."""
    x = samples.astype(np.float64)
    x -= x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = int(np.argmax(spec[1:]) + 1)
    # Parabolic interpolation between the bins around the peak, for sub-bin accuracy.
    a, b, c = np.log(spec[k - 1 : k + 2] + 1e-12)
    offset = 0.5 * (a - c) / (a - 2 * b + c)
    freq = (k + offset) * rate / len(x)
    return float(freq), float(20 * np.log10(spec[k] / np.median(spec[1:])))


def tone_report(path: Path, expect: float) -> tuple[str, bool]:
    samples, rate, ch = read_pcm16(path)
    freq, prominence = dominant_frequency(samples, rate)
    err = 100 * (freq - expect) / expect
    # A 1 kHz phone tone is accurate to well under 0.5 %, and so is the Pico's crystal, so a
    # wrong clock divider or a dropped bit per frame would show up as a much larger error.
    ok = abs(err) < 0.5 and prominence > 20
    st = clip_stats(samples)
    text = "\n".join(
        [
            f"Tone check on `{path.as_posix()}` ({rate} Hz, {ch} ch, {len(samples)} samples)",
            "",
            "| Expected (Hz) | Measured (Hz) | Error (%) | Peak above median bin (dB) "
            "| RMS (dBFS) | Peak (dBFS) | Result |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            f"| {expect:.1f} | {freq:.2f} | {err:+.3f} | {prominence:.1f} | {st.rms_dbfs:.1f} "
            f"| {st.peak_dbfs:.1f} | {'PASS' if ok else 'FAIL'} |",
        ]
    )
    return text + "\n", ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--config", type=Path, default=Path("configs/h2_recording.yaml"))
    c.add_argument("--listen", type=int, default=0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=Path)
    t = sub.add_parser("tone")
    t.add_argument("wav", type=Path)
    t.add_argument("--expect", type=float, default=1000.0)
    t.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "check":
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        text, problems = check(cfg, args.listen, args.seed)
        ok = not problems
    else:
        text, ok = tone_report(args.wav, args.expect)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
