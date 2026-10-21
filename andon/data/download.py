"""Download and unpack the DCASE 2020 Task 2 data from Zenodo.

    uv run python -m andon.data.download fan ToyConveyor
    uv run python -m andon.data.download all --dataset eval_data

Downloads resume if interrupted (re-run the same command), every file is checked
against Zenodo's MD5, and the clip counts are printed at the end. The data lands
in ``data/dev_data/<machine_type>/`` or ``data/eval_data/<machine_type>/`` and stays
out of git. ``eval_data`` holds the additional training set (normal clips of the
evaluation machine IDs), the evaluation clips, and their ground truth.

Data: DCASE 2020 Task 2 (ToyADMOS + MIMII), CC BY-NC-SA 4.0. Development set
https://zenodo.org/records/3678171, additional training set
https://zenodo.org/records/3727685, evaluation set https://zenodo.org/records/3841772,
evaluation ground truth https://zenodo.org/records/3951620.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from andon.data.dataset import (
    DATASETS,
    GROUND_TRUTH_FILE,
    MACHINE_TYPES,
    count_table,
    ground_truth_path,
    scan,
)

ZENODO_RECORD = "3678171"  # development set
EVAL_TRAIN_RECORD = "3727685"  # additional training set
EVAL_TEST_RECORD = "3841772"  # evaluation set
GROUND_TRUTH_RECORD = "3951620"
GROUND_TRUTH_MD5 = "f16aaa185df9906e588864aaf7cb58ea"


@dataclass(frozen=True)
class Archive:
    """One zip on Zenodo and the split folders it unpacks into ``<root>/<dataset>/<type>/``."""

    record: str
    file_name: str
    md5: str
    dataset: str
    splits: tuple[str, ...]


# File name and MD5 for each machine type, from the Zenodo record pages.
DEV_FILES: dict[str, tuple[str, str]] = {
    "fan": ("dev_data_fan.zip", "649bdfc06263ae7a838963f43b6641e6"),
    "pump": ("dev_data_pump.zip", "90e7091ef722b7238a7f1009365779cd"),
    "slider": ("dev_data_slider.zip", "da24a757719f0d94d5aa2d646bbfdc86"),
    "ToyCar": ("dev_data_ToyCar.zip", "4dec75ca8d9f666aa9e4c1894a740501"),
    "ToyConveyor": ("dev_data_ToyConveyor.zip", "03b6aa1bfd09a39d53af0ba39f71fa91"),
    "valve": ("dev_data_valve.zip", "34d2672f55bb041589ef79c10dc89934"),
}
EVAL_TRAIN_FILES: dict[str, tuple[str, str]] = {
    "fan": ("eval_data_train_fan.zip", "567798854130c8019df8b664c095ed1e"),
    "pump": ("eval_data_train_pump.zip", "4e33ae9c0db5cc88437675f5e317caee"),
    "slider": ("eval_data_train_slider.zip", "f84f186f4295fc23754abcdd1b0580a7"),
    "ToyCar": ("eval_data_train_ToyCar.zip", "a5d4ccd498af70b04dd74b644b820006"),
    "ToyConveyor": ("eval_data_train_ToyConveyor.zip", "0e808ac0b761e9fb13d5e0c65caff00d"),
    "valve": ("eval_data_train_valve.zip", "057cdd560caffb8305449541afd28c6d"),
}
EVAL_TEST_FILES: dict[str, tuple[str, str]] = {
    "fan": ("eval_data_test_fan.zip", "1eb9356a768cadfd0f2e59a5c57e578b"),
    "pump": ("eval_data_test_pump.zip", "23a8f8f924be218c69df67fe07360348"),
    "slider": ("eval_data_test_slider.zip", "0193c769073840332ce3aad84b1ccaa2"),
    "ToyCar": ("eval_data_test_ToyCar.zip", "bea2bdd612f616be8f2b8eb087b32c7c"),
    "ToyConveyor": ("eval_data_test_ToyConveyor.zip", "ccfcc9847c7d3404d7ee9d6d9e0b2ba2"),
    "valve": ("eval_data_test_valve.zip", "f69c551d2088691050d20cccca9d631c"),
}
assert set(DEV_FILES) == set(EVAL_TRAIN_FILES) == set(EVAL_TEST_FILES) == set(MACHINE_TYPES)

_CHUNK = 1 << 20  # 1 MiB


def zenodo_url(file_name: str, record: str = ZENODO_RECORD) -> str:
    return f"https://zenodo.org/records/{record}/files/{file_name}?download=1"


def archives(machine_type: str, dataset: str = "dev_data") -> list[Archive]:
    """The zips that make up one machine type of ``dataset``."""
    if dataset == "dev_data":
        name, md5 = DEV_FILES[machine_type]
        return [Archive(ZENODO_RECORD, name, md5, dataset, ("train", "test"))]
    if dataset == "eval_data":
        train_name, train_md5 = EVAL_TRAIN_FILES[machine_type]
        test_name, test_md5 = EVAL_TEST_FILES[machine_type]
        return [
            Archive(EVAL_TRAIN_RECORD, train_name, train_md5, dataset, ("train",)),
            Archive(EVAL_TEST_RECORD, test_name, test_md5, dataset, ("test",)),
        ]
    raise ValueError(f"unknown dataset: {dataset}")


def file_digest(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def md5sum(path: Path) -> str:
    return file_digest(path, "md5")


def _progress(done: int, total: int | None, name: str) -> None:
    if total:
        pct = 100 * done / total
        sys.stderr.write(f"\r  {name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.0f}%)")
    else:
        sys.stderr.write(f"\r  {name}: {done / 1e9:.2f} GB")
    sys.stderr.flush()


def download(
    url: str, dest: Path, expected_hash: str | None = None, algorithm: str = "md5"
) -> Path:
    """Download ``url`` to ``dest``, resuming from ``dest.part`` if it exists.

    Skips the download when ``dest`` already exists with the right hash. ``expected_hash``
    is checked with ``algorithm``: Zenodo publishes MD5s, the checkpoint release SHA-256s.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_hash is None or file_digest(dest, algorithm) == expected_hash):
        print(f"  {dest.name}: already downloaded")
        return dest

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "andon-downloader"})
    if have:
        request.add_header("Range", f"bytes={have}-")

    try:
        _stream(request, part, have, dest.name)
    except urllib.error.HTTPError as err:
        # 416 means the .part file already holds every byte (the last run stopped
        # before renaming it), so go straight to the hash check.
        if err.code != 416 or not have:
            raise

    if expected_hash is not None:
        actual = file_digest(part, algorithm)
        if actual != expected_hash:
            part.unlink()
            raise RuntimeError(
                f"{dest.name}: {algorithm.upper()} mismatch (got {actual}, "
                f"expected {expected_hash}). "
                "The partial file was deleted; run the command again."
            )
    part.replace(dest)
    return dest


def _stream(request: urllib.request.Request, part: Path, have: int, name: str) -> None:
    """Write the response body to ``part``, appending if the server honoured the Range."""
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = have > 0 and response.getcode() == 206
        if not resumed:
            have = 0  # server ignored the Range header (or nothing to resume): start over
        length = response.headers.get("Content-Length")
        total = have + int(length) if length else None
        with part.open("ab" if resumed else "wb") as out:
            done = have
            for block in iter(lambda: response.read(_CHUNK), b""):
                out.write(block)
                done += len(block)
                _progress(done, total, name)
    sys.stderr.write("\n")


def extract(
    zip_path: Path,
    root: Path,
    machine_type: str,
    dataset: str = "dev_data",
    splits: tuple[str, ...] = ("train", "test"),
) -> Path:
    """Unpack one zip's ``splits`` into ``<root>/<dataset>/<machine_type>/``.

    Works whether the zip's top folder is ``<dataset>/<type>/`` or just ``<type>/``. Only
    the listed split folders are replaced, so the additional training zip and the
    evaluation zip can share ``eval_data/<type>/``.
    """
    target = Path(root) / dataset / machine_type
    staging = Path(root) / f"_extract_{machine_type}"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging)

    candidates = [p for p in staging.rglob(machine_type) if p.is_dir()]
    candidates = [p for p in candidates if all((p / s).is_dir() for s in splits)]
    if len(candidates) != 1:
        shutil.rmtree(staging)
        wanted = " and ".join(f"{machine_type}/{s}" for s in splits)
        raise RuntimeError(f"{zip_path.name}: could not find {wanted}")

    target.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if (target / split).exists():
            shutil.rmtree(target / split)
        shutil.move(str(candidates[0] / split), str(target / split))
    shutil.rmtree(staging)
    return target


def is_extracted(
    root: Path,
    machine_type: str,
    dataset: str = "dev_data",
    splits: tuple[str, ...] = ("train", "test"),
) -> bool:
    folder = Path(root) / dataset / machine_type
    return all(any((folder / split).glob("*.wav")) for split in splits)


def fetch(
    machine_type: str,
    root: Path,
    keep_zip: bool = False,
    dataset: str = "dev_data",
    splits: tuple[str, ...] | None = None,
) -> None:
    """Download and unpack one machine type; ``splits`` limits it to the zips holding only those
    folders (``("test",)`` skips the additional training set)."""
    for archive in archives(machine_type, dataset):
        if splits is not None and not set(archive.splits) <= set(splits):
            continue
        label = f"{machine_type} ({archive.file_name})"
        if is_extracted(root, machine_type, archive.dataset, archive.splits):
            print(f"{label}: already extracted, skipping")
            continue
        print(f"{label}: downloading")
        dest = Path(root) / "downloads" / archive.file_name
        zip_path = download(zenodo_url(archive.file_name, archive.record), dest, archive.md5)
        print(f"{label}: extracting")
        extract(zip_path, root, machine_type, archive.dataset, archive.splits)
        if not keep_zip:
            zip_path.unlink()


def fetch_ground_truth(root: Path) -> Path:
    """The evaluation labels, ``<root>/eval_data/eval_data_list.csv`` (MD5-checked)."""
    url = zenodo_url(GROUND_TRUTH_FILE, GROUND_TRUTH_RECORD)
    return download(url, ground_truth_path(root), GROUND_TRUTH_MD5)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download DCASE 2020 Task 2 data.")
    parser.add_argument(
        "machines",
        nargs="*",
        default=["fan", "ToyConveyor"],
        help="machine types to fetch (default: fan ToyConveyor; 'all' for all six)",
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="data folder")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="dev_data",
        help="dev_data, or eval_data (additional training set, evaluation set, ground truth)",
    )
    parser.add_argument("--keep-zip", action="store_true", help="keep zips after extracting")
    args = parser.parse_args(argv)

    machines = list(MACHINE_TYPES) if args.machines == ["all"] else args.machines
    unknown = set(machines) - set(MACHINE_TYPES)
    if unknown:
        parser.error(f"unknown machine types: {sorted(unknown)}; choose from {MACHINE_TYPES}")

    if args.dataset == "eval_data":
        fetch_ground_truth(args.root)
    for machine_type in machines:
        fetch(machine_type, args.root, keep_zip=args.keep_zip, dataset=args.dataset)

    print()
    print(count_table(scan(args.root, machines, args.dataset)))


if __name__ == "__main__":
    main()
