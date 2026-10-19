import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

from andon.data import download as dl
from andon.data.audio import read_wav, write_wav
from andon.data.dataset import (
    all_train_clips,
    count_table,
    eval_test_clips,
    parse_clip,
    read_ground_truth,
    scan,
)

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


class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, code: int):
        super().__init__(body)
        self.code = code
        self.headers = {"Content-Length": str(len(body))}

    def getcode(self) -> int:
        return self.code


def _range_server(payload: bytes):
    """A stand-in for urlopen that honours Range headers the way Zenodo does."""

    def urlopen(request, timeout=None):
        start = int((request.get_header("Range") or "bytes=0-")[6:-1])
        if start >= len(payload):
            raise urllib.error.HTTPError(request.full_url, 416, "Range Not Satisfiable", {}, None)
        return _FakeResponse(payload[start:], 206 if start else 200)

    return urlopen


@pytest.mark.parametrize("already_have", [2000, 5000], ids=["resume", "part-complete"])
def test_download_resumes_from_partial_file(tmp_path, monkeypatch, already_have):
    payload = b"andon" * 1000
    monkeypatch.setattr(urllib.request, "urlopen", _range_server(payload))
    dest = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(payload[:already_have])

    md5 = hashlib.md5(payload).hexdigest()
    assert dl.download("https://example.invalid/file.bin", dest, md5) == dest
    assert dest.read_bytes() == payload


def test_zenodo_urls_and_md5_table():
    assert set(dl.DEV_FILES) == {"ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"}
    assert dl.zenodo_url("dev_data_fan.zip").startswith("https://zenodo.org/records/3678171/")
    for table in (dl.DEV_FILES, dl.EVAL_TRAIN_FILES, dl.EVAL_TEST_FILES):
        assert all(len(md5) == 32 for _, md5 in table.values())
    train, test = dl.archives("pump", "eval_data")
    assert (train.record, train.splits) == ("3727685", ("train",))
    assert (test.record, test.splits) == ("3841772", ("test",))
    assert test.file_name == "eval_data_test_pump.zip"


# --- evaluation data -------------------------------------------------------------------

EVAL_LABELS = {"id_01_00000000.wav": 0, "id_01_00000001.wav": 1, "id_03_00000000.wav": 1}


def make_eval_tree(root: Path, machine_type: str = "fan", labels=EVAL_LABELS) -> None:
    """A tiny fake eval_data tree: 2 normal train clips per ID, test clips, ground truth."""
    quiet = np.zeros(160, dtype=np.float32)
    base = root / "eval_data" / machine_type
    (base / "train").mkdir(parents=True, exist_ok=True)
    (base / "test").mkdir(parents=True, exist_ok=True)
    for machine_id in ("id_01", "id_03"):
        for i in range(2):
            write_wav(base / "train" / f"normal_{machine_id}_{i:08d}.wav", quiet, SR)
    for name in labels:
        write_wav(base / "test" / name, quiet, SR)
    # The real file's layout: a machine-type line, then name, original name, label.
    lines = ["pump", "id_01_00000000.wav,normal_id_01_00000005.wav,0", machine_type]
    lines += [f"{n},{'anomaly' if y else 'normal'}_{n},{y}" for n, y in labels.items()]
    (root / "eval_data" / "eval_data_list.csv").write_text("\n".join(lines) + "\n")


def test_read_ground_truth_splits_by_machine_type(tmp_path):
    make_eval_tree(tmp_path)
    labels = read_ground_truth(tmp_path / "eval_data" / "eval_data_list.csv")
    assert labels["pump"] == {"id_01_00000000.wav": False}
    assert labels["fan"] == {k: bool(v) for k, v in EVAL_LABELS.items()}


def test_read_ground_truth_rejects_a_label_before_any_machine_type(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text("id_01_00000000.wav,normal_id_01_00000005.wav,0\n")
    with pytest.raises(ValueError, match="unexpected line"):
        read_ground_truth(path)


def test_eval_test_clips_take_labels_from_ground_truth(tmp_path):
    make_eval_tree(tmp_path)
    clips = eval_test_clips(tmp_path, "fan")
    assert [(c.path.name, c.machine_id, c.is_anomaly) for c in clips] == [
        ("id_01_00000000.wav", "id_01", False),
        ("id_01_00000001.wav", "id_01", True),
        ("id_03_00000000.wav", "id_03", True),
    ]
    assert all(c.split == "test" for c in clips)


def test_eval_test_clips_refuse_a_clip_without_a_label(tmp_path):
    make_eval_tree(tmp_path)
    write_wav(tmp_path / "eval_data/fan/test/id_03_00000001.wav", np.zeros(160, np.float32), SR)
    with pytest.raises(ValueError, match="1 clips without a label"):
        eval_test_clips(tmp_path, "fan")


def test_all_train_clips_joins_dev_and_additional_normals(tmp_path):
    make_tree(tmp_path)  # dev IDs 00, 02
    make_eval_tree(tmp_path)  # eval IDs 01, 03
    clips = all_train_clips(tmp_path, "fan")
    assert len(clips) == 2 * 3 + 2 * 2
    assert {c.machine_id for c in clips} == {"id_00", "id_01", "id_02", "id_03"}
    assert all(c.split == "train" and not c.is_anomaly for c in clips)
    assert all(c.path.parent.name == "train" for c in clips)


def test_all_train_clips_refuses_an_id_in_both_sets(tmp_path):
    make_tree(tmp_path, ids=("id_00", "id_01"))
    make_eval_tree(tmp_path)
    with pytest.raises(ValueError, match="id_01"):
        all_train_clips(tmp_path, "fan")


def test_scan_eval_counts_labelled_test_clips(tmp_path):
    make_eval_tree(tmp_path)
    table = count_table(scan(tmp_path, ["fan"], dataset="eval_data"))
    assert "fan          id_01       2         1         1" in table
    assert "fan          id_03       2         0         1" in table


def test_extract_eval_train_then_test_into_one_folder(tmp_path):
    src = tmp_path / "src"
    make_eval_tree(src)
    for split in ("train", "test"):
        with zipfile.ZipFile(tmp_path / f"{split}.zip", "w") as zf:
            for f in (src / "eval_data" / "fan" / split).glob("*.wav"):
                zf.write(f, f"eval_data/fan/{split}/{f.name}")

    root = tmp_path / "data"
    dl.extract(tmp_path / "train.zip", root, "fan", "eval_data", ("train",))
    dl.extract(tmp_path / "test.zip", root, "fan", "eval_data", ("test",))
    assert dl.is_extracted(root, "fan", "eval_data", ("train", "test"))
    assert len(list((root / "eval_data/fan/train").glob("*.wav"))) == 4
    assert len(list((root / "eval_data/fan/test").glob("*.wav"))) == 3
    with pytest.raises(RuntimeError, match="fan/test"):
        dl.extract(tmp_path / "train.zip", root, "fan", "eval_data", ("test",))
