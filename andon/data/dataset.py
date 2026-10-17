"""Find and describe the DCASE 2020 Task 2 development clips on disk.

Layout after download::

    <root>/dev_data/<machine_type>/train/normal_id_00_00000000.wav
    <root>/dev_data/<machine_type>/test/normal_id_00_00000000.wav
    <root>/dev_data/<machine_type>/test/anomaly_id_00_00000000.wav

Machine type comes from the folder; condition and machine ID come from the file name.

Run ``uv run python -m andon.data.dataset`` to print clip counts per machine ID.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MACHINE_TYPES = ("ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve")
SPLITS = ("train", "test")

_FILENAME = re.compile(r"^(?P<cond>normal|anomaly)_(?P<id>id_\d{2})_(?P<idx>\d+)\.wav$")


@dataclass(frozen=True)
class Clip:
    path: Path
    machine_type: str
    split: str  # "train" or "test"
    machine_id: str  # e.g. "id_00"
    is_anomaly: bool


def parse_clip(path: Path, machine_type: str, split: str) -> Clip:
    match = _FILENAME.match(path.name)
    if match is None:
        raise ValueError(f"unexpected file name: {path}")
    clip = Clip(
        path=path,
        machine_type=machine_type,
        split=split,
        machine_id=match["id"],
        is_anomaly=match["cond"] == "anomaly",
    )
    # Experiment rule 1: training data is normal only. Fail loudly if that's ever broken.
    if clip.split == "train" and clip.is_anomaly:
        raise ValueError(f"anomalous clip found in a training folder: {path}")
    return clip


def dev_root(root: Path) -> Path:
    return Path(root) / "dev_data"


def scan(root: Path, machine_types: Iterable[str] | None = None) -> list[Clip]:
    """All clips under ``<root>/dev_data``, sorted by path. Missing machine types are skipped."""
    types = list(machine_types) if machine_types is not None else list(MACHINE_TYPES)
    unknown = set(types) - set(MACHINE_TYPES)
    if unknown:
        raise ValueError(f"unknown machine types: {sorted(unknown)}")

    clips: list[Clip] = []
    for machine_type in types:
        for split in SPLITS:
            folder = dev_root(root) / machine_type / split
            if folder.is_dir():
                clips.extend(parse_clip(p, machine_type, split) for p in folder.glob("*.wav"))
    return sorted(clips, key=lambda c: c.path)


def train_clips(root: Path, machine_type: str) -> list[Clip]:
    """Normal training clips for one machine type, sorted by name. The only source of training data.

    Reads ``train/`` and nothing else, so test-split normals can never leak into training.
    Sorted by name as the baseline's ``sorted(glob(...))`` was; Keras ``validation_split``
    holds out the tail of that order, so the order matters for reproducing it.
    """
    clips = _split_clips(root, machine_type, "train")
    bad = [c.path for c in clips if c.is_anomaly or c.split != "train"]
    if bad:  # parse_clip already refuses these; keep the guard next to the training entry point
        raise ValueError(f"non-normal or non-train clips in training set: {bad[:3]}")
    return clips


def dev_test_clips(root: Path, machine_type: str) -> list[Clip]:
    """Labelled dev test clips (normal and anomalous) for one machine type, sorted by name."""
    return _split_clips(root, machine_type, "test")


def _split_clips(root: Path, machine_type: str, split: str) -> list[Clip]:
    if machine_type not in MACHINE_TYPES:
        raise ValueError(f"unknown machine type: {machine_type}")
    folder = dev_root(root) / machine_type / split
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; download it with andon.data.download")
    clips = [parse_clip(p, machine_type, split) for p in folder.glob("*.wav")]
    if not clips:
        raise FileNotFoundError(f"no .wav files in {folder}")
    return sorted(clips, key=lambda c: c.path.name)


_COLUMNS = ("train", "test_norm", "test_anom")


def count_table(clips: Iterable[Clip]) -> str:
    """Plain-text table: clips per machine type and ID, by split and condition."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for c in clips:
        column = "train" if c.split == "train" else ("test_anom" if c.is_anomaly else "test_norm")
        counts[(c.machine_type, c.machine_id, column)] += 1

    keys = sorted({(t, i) for t, i, _ in counts})
    if not keys:
        return "No clips found."

    header = f"{'machine_type':<12} {'id':<6} {'train':>6} {'test_norm':>9} {'test_anom':>9}"
    lines = [header, "-" * len(header)]
    totals: Counter[str] = Counter()
    for machine_type, machine_id in keys:
        row = [counts[(machine_type, machine_id, col)] for col in _COLUMNS]
        totals.update(dict(zip(_COLUMNS, row, strict=True)))
        lines.append(f"{machine_type:<12} {machine_id:<6} {row[0]:>6} {row[1]:>9} {row[2]:>9}")
    lines.append("-" * len(header))
    lines.append(
        f"{'total':<12} {'':<6} {totals['train']:>6} {totals['test_norm']:>9} "
        f"{totals['test_anom']:>9}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Count DCASE 2020 Task 2 dev clips.")
    parser.add_argument("machines", nargs="*", default=None, help="machine types (default: all)")
    parser.add_argument("--root", type=Path, default=Path("data"), help="data folder")
    args = parser.parse_args(argv)
    print(count_table(scan(args.root, args.machines or None)))


if __name__ == "__main__":
    main()
