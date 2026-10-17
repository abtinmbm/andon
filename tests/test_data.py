import zipfile
from pathlib import Path

import numpy as np
import pytest

from andon.data import download as dl
from andon.data.audio import read_wav, write_wav
from andon.data.dataset import count_table, parse_clip, scan

SR = 16_000


def make_tree(root: Path, machine_type: str = "fan", ids=("id_00", "id_02")) -> None:
    """A tiny fake dev_data tree: 3 train, 2 normal test, 1 anomaly test per ID."""
    quiet = np.zeros(160, dtype=np.float32)
    for machine_id in ids:
        for split, cond, n in (
            ("train", "normal", 3),
            ("test", "normal", 2),
            ("test", "anomaly", 1),
        ):
            folder = root / "dev_data" / machine_type / split
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                write_wav(folder / f"{cond}_{machine_id}_{i:08d}.wav", quiet, SR)


def test_wav_round_trip(tmp_path):
    y = np.sin(np.linspace(0, 20, SR)).astype(np.float32) * 0.5
    write_wav(tmp_path / "a.wav", y, SR)
    back, sr = read_wav(tmp_path / "a.wav")
    assert sr == SR
    assert back.dtype == np.float32
    assert np.max(np.abs(back - y)) < 1 / 32768 + 1e-7


def test_parse_clip_reads_condition_and_id(tmp_path):
    clip = parse_clip(tmp_path / "anomaly_id_04_00000012.wav", "fan", "test")
    assert clip.machine_id == "id_04"
    assert clip.is_anomaly


def test_anomaly_in_train_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="training"):
        parse_clip(tmp_path / "anomaly_id_00_00000000.wav", "fan", "train")


def test_bad_file_name_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        parse_clip(tmp_path / "something.wav", "fan", "train")


def test_scan_and_count_table(tmp_path):
    make_tree(tmp_path)
    clips = scan(tmp_path, ["fan"])
    assert len(clips) == 2 * (3 + 2 + 1)
    table = count_table(clips)
    assert "fan          id_00       3         2         1" in table
    assert "total" in table and "6         4         2" in table


def test_scan_skips_missing_types_and_rejects_unknown(tmp_path):
    assert scan(tmp_path, ["pump"]) == []
    with pytest.raises(ValueError):
        scan(tmp_path, ["blender"])


@pytest.mark.parametrize("prefix", ["dev_data/", ""])
def test_extract_handles_either_zip_layout(tmp_path, prefix):
    src = tmp_path / "src"
    make_tree(src)
    zip_path = tmp_path / "dev_data_fan.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in (src / "dev_data").rglob("*.wav"):
            arc = f.relative_to(src / "dev_data").as_posix()
            zf.write(f, prefix + arc)

    root = tmp_path / "data"
    dl.extract(zip_path, root, "fan")
    assert dl.is_extracted(root, "fan")
    assert len(scan(root, ["fan"])) == 12
    assert not (root / "_extract_fan").exists()


def test_download_checks_md5_and_restarts_without_range_support(tmp_path):
    payload = b"andon" * 1000
    src = tmp_path / "remote.bin"
    src.write_bytes(payload)
    url = src.resolve().as_uri()  # file:// ignores Range, like a server without resume
    dest = tmp_path / "out" / "file.bin"

    # A stale partial file must not corrupt the result.
    dest.parent.mkdir()
    (dest.parent / "file.bin.part").write_bytes(b"garbage")

    md5 = dl.md5sum(src)
    assert dl.download(url, dest, md5) == dest
    assert dest.read_bytes() == payload
    assert not (dest.parent / "file.bin.part").exists()


def test_download_rejects_bad_md5(tmp_path):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"x" * 100)
    with pytest.raises(RuntimeError, match="MD5"):
        dl.download(src.resolve().as_uri(), tmp_path / "file.bin", "0" * 32)
    assert not (tmp_path / "file.bin.part").exists()


def test_zenodo_urls_and_md5_table():
    assert set(dl.DEV_FILES) == {"ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"}
    assert dl.zenodo_url("dev_data_fan.zip").startswith("https://zenodo.org/records/3678171/")
    assert all(len(md5) == 32 for _, md5 in dl.DEV_FILES.values())
