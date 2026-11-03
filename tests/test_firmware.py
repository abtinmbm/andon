"""Build and run the firmware's host tests: the hardware-free C files in firmware/src, compiled
for the PC together with each firmware/test/test_*.c, which exits non-zero on a failed check."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIRMWARE = Path(__file__).resolve().parents[1] / "firmware"
COMPILER = shutil.which("gcc") or shutil.which("cc")
# main.c is the only file that calls the Pico SDK; everything else must build on the PC.
HOST_SOURCES = sorted(p for p in (FIRMWARE / "src").glob("*.c") if p.name != "main.c")
TESTS = sorted((FIRMWARE / "test").glob("test_*.c"))
FLAGS = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic"]


def test_host_tests_exist():
    assert TESTS, "no firmware/test/test_*.c found"
    assert HOST_SOURCES, "no hardware-free firmware/src/*.c found"


@pytest.mark.skipif(COMPILER is None, reason="no C compiler")
@pytest.mark.parametrize("test_src", TESTS, ids=lambda p: p.stem)
def test_firmware_host(test_src: Path, tmp_path: Path):
    exe = tmp_path / (test_src.stem + (".exe" if sys.platform == "win32" else ""))
    sources = [str(test_src), *map(str, HOST_SOURCES)]
    build = subprocess.run(
        [COMPILER, *FLAGS, f"-I{FIRMWARE / 'src'}", *sources, "-o", str(exe)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "all checks passed" in run.stdout
