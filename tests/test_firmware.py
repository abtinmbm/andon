"""Build and run the firmware's host tests: the hardware-free C files in firmware/src, compiled
for the PC together with each firmware/test/test_*.c, which exits non-zero on a failed check.

The C code is built against two sets of generated headers: a synthetic model exported here
(runs everywhere, including CI), and the real fan model in firmware/model when it has been
exported locally (its weights come from the training data, so it is not in git)."""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from andon import export_c

FIRMWARE = Path(__file__).resolve().parents[1] / "firmware"
REAL_MODEL = FIRMWARE / "model"
COMPILER = shutil.which("gcc") or shutil.which("cc")
# These call the Pico SDK; everything else in src must build on the PC.
DEVICE_ONLY = {"main.c", "bench.c", "mic.c", "recorder.c"}
HOST_SOURCES = sorted(p for p in (FIRMWARE / "src").glob("*.c") if p.name not in DEVICE_ONLY)
TESTS = sorted((FIRMWARE / "test").glob("test_*.c"))
# The generated headers define every array static, so a file that uses only some of them
# would warn about the rest. -O2 because some warnings (maybe-uninitialized) need the
# optimiser, and the device build is -O3.
FLAGS = [
    "-std=c11",
    "-O2",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-pedantic",
    "-Wno-unused-const-variable",
]
HEADER_SETS = ["synthetic"] + (["real"] if (REAL_MODEL / export_c.VECTORS_HEADER).exists() else [])


# Windows Smart App Control sometimes blocks a freshly built .exe (WinError 4551), and keeps
# blocking that file, while a rebuild under a new name runs. Only that error is retried; a
# failed check is never retried.
SAC_BLOCKED = 4551
SAC_ATTEMPTS = 3


def _build_and_run(cmd: list[str], exe: Path) -> subprocess.CompletedProcess:
    for attempt in range(SAC_ATTEMPTS):
        out = exe.with_name(f"{exe.stem}_{attempt}{exe.suffix}")
        build = subprocess.run([*cmd, "-o", str(out)], capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        try:
            return subprocess.run([str(out)], capture_output=True, text=True, timeout=60)
        except OSError as e:
            if getattr(e, "winerror", None) != SAC_BLOCKED or attempt == SAC_ATTEMPTS - 1:
                raise
            time.sleep(1.0)
    raise AssertionError("unreachable")


@pytest.fixture(scope="module")
def synthetic_headers(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synthetic_model")
    export_c.write(export_c.synthetic_export(seed=0), out)
    return out


def test_host_tests_exist():
    assert TESTS, "no firmware/test/test_*.c found"
    assert HOST_SOURCES, "no hardware-free firmware/src/*.c found"


@pytest.mark.skipif(COMPILER is None, reason="no C compiler")
@pytest.mark.parametrize("headers", HEADER_SETS)
@pytest.mark.parametrize("test_src", TESTS, ids=lambda p: p.stem)
def test_firmware_host(test_src: Path, headers: str, synthetic_headers: Path, tmp_path: Path):
    assert COMPILER is not None  # skipped above otherwise; narrows the type
    model_dir = synthetic_headers if headers == "synthetic" else REAL_MODEL
    exe = tmp_path / (test_src.stem + (".exe" if sys.platform == "win32" else ""))
    sources = [str(test_src), *map(str, HOST_SOURCES)]
    includes = [f"-I{FIRMWARE / 'src'}", f"-I{model_dir}"]
    run = _build_and_run([COMPILER, *FLAGS, *includes, *sources, "-lm"], exe)
    print(run.stdout)  # measured errors and RAM sizes, shown with pytest -s
    assert run.returncode == 0, run.stdout + run.stderr
    assert "all checks passed" in run.stdout
