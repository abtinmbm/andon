"""Download and unpack the DCASE 2020 Task 2 development data from Zenodo.

    uv run python -m andon.data.download fan ToyConveyor

Downloads resume if interrupted (re-run the same command), every zip is checked
against Zenodo's MD5, and the clip counts are printed at the end. The data lands
in ``data/dev_data/<machine_type>/`` and stays out of git.

Data: DCASE 2020 Task 2 development dataset (ToyADMOS + MIMII), CC BY-NC-SA 4.0,
https://zenodo.org/records/3678171
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from andon.data.dataset import MACHINE_TYPES, count_table, dev_root, scan

ZENODO_RECORD = "3678171"

# File name and MD5 for each machine type, from the Zenodo record page.
DEV_FILES: dict[str, tuple[str, str]] = {
    "fan": ("dev_data_fan.zip", "649bdfc06263ae7a838963f43b6641e6"),
    "pump": ("dev_data_pump.zip", "90e7091ef722b7238a7f1009365779cd"),
    "slider": ("dev_data_slider.zip", "da24a757719f0d94d5aa2d646bbfdc86"),
    "ToyCar": ("dev_data_ToyCar.zip", "4dec75ca8d9f666aa9e4c1894a740501"),
    "ToyConveyor": ("dev_data_ToyConveyor.zip", "03b6aa1bfd09a39d53af0ba39f71fa91"),
    "valve": ("dev_data_valve.zip", "34d2672f55bb041589ef79c10dc89934"),
}
assert set(DEV_FILES) == set(MACHINE_TYPES)

_CHUNK = 1 << 20  # 1 MiB


def zenodo_url(file_name: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{file_name}?download=1"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(done: int, total: int | None, name: str) -> None:
    if total:
        pct = 100 * done / total
        sys.stderr.write(f"\r  {name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.0f}%)")
    else:
        sys.stderr.write(f"\r  {name}: {done / 1e9:.2f} GB")
    sys.stderr.flush()


def download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    """Download ``url`` to ``dest``, resuming from ``dest.part`` if it exists.

    Skips the download when ``dest`` already exists with the right MD5.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_md5 is None or md5sum(dest) == expected_md5):
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
        # before renaming it), so go straight to the MD5 check.
        if err.code != 416 or not have:
            raise

    if expected_md5 is not None:
        actual = md5sum(part)
        if actual != expected_md5:
            part.unlink()
            raise RuntimeError(
                f"{dest.name}: MD5 mismatch (got {actual}, expected {expected_md5}). "
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


def extract(zip_path: Path, root: Path, machine_type: str) -> Path:
    """Unpack one machine type's zip into ``<root>/dev_data/<machine_type>``.

    Works whether the zip's top folder is ``dev_data/<type>/`` or just ``<type>/``.
    """
    target = dev_root(root) / machine_type
    staging = Path(root) / f"_extract_{machine_type}"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging)

    candidates = [p.parent for p in staging.rglob("train") if p.is_dir()]
    candidates = [p for p in candidates if p.name == machine_type and (p / "test").is_dir()]
    if len(candidates) != 1:
        shutil.rmtree(staging)
        raise RuntimeError(f"{zip_path.name}: could not find {machine_type}/train and /test")

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidates[0]), str(target))
    shutil.rmtree(staging)
    return target


def is_extracted(root: Path, machine_type: str) -> bool:
    folder = dev_root(root) / machine_type
    return all(any((folder / split).glob("*.wav")) for split in ("train", "test"))


def fetch(machine_type: str, root: Path, keep_zip: bool = False) -> None:
    if is_extracted(root, machine_type):
        print(f"{machine_type}: already extracted, skipping")
        return
    file_name, md5 = DEV_FILES[machine_type]
    print(f"{machine_type}: downloading {file_name}")
    zip_path = download(zenodo_url(file_name), Path(root) / "downloads" / file_name, md5)
    print(f"{machine_type}: extracting")
    extract(zip_path, root, machine_type)
    if not keep_zip:
        zip_path.unlink()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download DCASE 2020 Task 2 dev data.")
    parser.add_argument(
        "machines",
        nargs="*",
        default=["fan", "ToyConveyor"],
        help="machine types to fetch (default: fan ToyConveyor; 'all' for all six)",
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="data folder")
    parser.add_argument("--keep-zip", action="store_true", help="keep zips after extracting")
    args = parser.parse_args(argv)

    machines = list(MACHINE_TYPES) if args.machines == ["all"] else args.machines
    unknown = set(machines) - set(MACHINE_TYPES)
    if unknown:
        parser.error(f"unknown machine types: {sorted(unknown)}; choose from {MACHINE_TYPES}")

    for machine_type in machines:
        fetch(machine_type, args.root, keep_zip=args.keep_zip)

    print()
    print(count_table(scan(args.root, machines)))


if __name__ == "__main__":
    main()
