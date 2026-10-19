"""Find and describe the DCASE 2020 Task 2 clips on disk.

Layout after download::

    <root>/dev_data/<machine_type>/train/normal_id_00_00000000.wav
    <root>/dev_data/<machine_type>/test/normal_id_00_00000000.wav
    <root>/dev_data/<machine_type>/test/anomaly_id_00_00000000.wav
    <root>/eval_data/<machine_type>/train/normal_id_01_00000000.wav  (additional training set)
    <root>/eval_data/<machine_type>/test/id_01_00000000.wav          (evaluation set)
    <root>/eval_data/eval_data_list.csv                              (evaluation ground truth)

Machine type comes from the folder; condition and machine ID come from the file name,
except for evaluation clips, whose names hide the condition: it comes from the ground
truth file. The evaluation machine IDs are different machines from the development ones.

Run ``uv run python -m andon.data.dataset`` to print clip counts per machine ID.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MACHINE_TYPES = ("ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve")
SPLITS = ("train", "test")
DATASETS = ("dev_data", "eval_data")
GROUND_TRUTH_FILE = "eval_data_list.csv"

_FILENAME = re.compile(r"^(?P<cond>normal|anomaly)_(?P<id>id_\d{2})_(?P<idx>\d+)\.wav$")
_EVAL_TEST_FILENAME = re.compile(r"^(?P<id>id_\d{2})_(?P<idx>\d+)\.wav$")


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


def eval_root(root: Path) -> Path:
    return Path(root) / "eval_data"


def ground_truth_path(root: Path) -> Path:
    return eval_root(root) / GROUND_TRUTH_FILE


def read_ground_truth(path: Path) -> dict[str, dict[str, bool]]:
    """Evaluation labels: ``{machine_type: {file_name: is_anomaly}}``.

    The file (Zenodo record 3951620) has a line holding only a machine type, followed by
    ``id_01_00000000.wav,<original name>,<0 or 1>`` lines for that type.
    """
    labels: dict[str, dict[str, bool]] = {}
    current: dict[str, bool] | None = None
    with Path(path).open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) == 1:
                if row[0] not in MACHINE_TYPES:
                    raise ValueError(f"{path}: unknown machine type line {row[0]!r}")
                current = labels.setdefault(row[0], {})
                continue
            if current is None or len(row) != 3 or row[2] not in ("0", "1"):
                raise ValueError(f"{path}: unexpected line {row}")
            current[row[0]] = row[2] == "1"
    return labels


def scan(
    root: Path, machine_types: Iterable[str] | None = None, dataset: str = "dev_data"
) -> list[Clip]:
    """All clips under ``<root>/<dataset>``, sorted by path. Missing machine types are skipped.

    Evaluation test clips are labelled from the ground truth file, which must be present.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    types = list(machine_types) if machine_types is not None else list(MACHINE_TYPES)
    unknown = set(types) - set(MACHINE_TYPES)
    if unknown:
        raise ValueError(f"unknown machine types: {sorted(unknown)}")

    clips: list[Clip] = []
    for machine_type in types:
        for split in SPLITS:
            folder = Path(root) / dataset / machine_type / split
            if not folder.is_dir():
                continue
            if dataset == "eval_data" and split == "test":
                clips.extend(eval_test_clips(root, machine_type))
            else:
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


def additional_train_clips(root: Path, machine_type: str) -> list[Clip]:
    """Normal clips of the evaluation machine IDs (the additional training set), sorted by name."""
    return _split_clips(root, machine_type, "train", dataset="eval_data")


def all_train_clips(root: Path, machine_type: str) -> list[Clip]:
    """Normal training clips of every machine ID of one type: dev train, then additional train.

    The machine-ID classifier trains on these, so one model scores both the dev test and
    the evaluation clips. Neither set holds a test clip, and ``parse_clip`` refuses any
    anomaly-named file in a train folder.
    """
    dev, extra = train_clips(root, machine_type), additional_train_clips(root, machine_type)
    shared = {c.machine_id for c in dev} & {c.machine_id for c in extra}
    if shared:  # the two sets are different machines; a shared ID would merge two classes
        raise ValueError(f"{machine_type}: IDs in both dev and additional train: {sorted(shared)}")
    return dev + extra


def eval_test_clips(root: Path, machine_type: str) -> list[Clip]:
    """Evaluation clips of one machine type, sorted by name, labelled from the ground truth.

    Every clip must have a label and every label a clip, so a partial download or a
    mismatched ground truth file fails here instead of giving a quietly wrong AUC.
    """
    if machine_type not in MACHINE_TYPES:
        raise ValueError(f"unknown machine type: {machine_type}")
    folder = eval_root(root) / machine_type / "test"
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; download it with andon.data.download")
    gt_path = ground_truth_path(root)
    if not gt_path.exists():
        raise FileNotFoundError(f"{gt_path} not found; download it with andon.data.download")
    labels = read_ground_truth(gt_path).get(machine_type, {})

    paths = sorted(folder.glob("*.wav"), key=lambda p: p.name)
    names = {p.name for p in paths}
    if not paths or names != set(labels):
        raise ValueError(
            f"{machine_type}: eval clips and ground truth disagree "
            f"({len(set(labels) - names)} labelled clips missing, "
            f"{len(names - set(labels))} clips without a label)"
        )
    clips = []
    for p in paths:
        match = _EVAL_TEST_FILENAME.match(p.name)
        if match is None:
            raise ValueError(f"unexpected file name: {p}")
        clip = Clip(p, machine_type, "test", match["id"], is_anomaly=labels[p.name])
        clips.append(clip)
    return clips


def _split_clips(
    root: Path, machine_type: str, split: str, dataset: str = "dev_data"
) -> list[Clip]:
    if machine_type not in MACHINE_TYPES:
        raise ValueError(f"unknown machine type: {machine_type}")
    folder = Path(root) / dataset / machine_type / split
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
    parser = argparse.ArgumentParser(description="Count DCASE 2020 Task 2 clips.")
    parser.add_argument("machines", nargs="*", default=None, help="machine types (default: all)")
    parser.add_argument("--root", type=Path, default=Path("data"), help="data folder")
    parser.add_argument("--dataset", choices=DATASETS, default="dev_data")
    args = parser.parse_args(argv)
    print(count_table(scan(args.root, args.machines or None, args.dataset)))


if __name__ == "__main__":
    main()
