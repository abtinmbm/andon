"""Build and run the firmware's host tests: the hardware-free C files in firmware/src, compiled
for the PC together with each firmware/test/test_*.c, which exits non-zero on a failed check.

The C code is built against two sets of generated headers: synthetic models exported here
(runs everywhere, including CI), and the real ones in firmware/model when both have been
exported locally: the int8 CNN trained on the DCASE fans and the per-fan GMM trained on the
desk fans (both come from training data, so they are not in git)."""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from andon import export_c, export_gmm
from andon.experiment import load_config

FIRMWARE = Path(__file__).resolve().parents[1] / "firmware"
REAL_MODEL = FIRMWARE / "model"
CONFIGS = FIRMWARE.parent / "configs"
COMPILER = shutil.which("gcc") or shutil.which("cc")
# These call the Pico SDK; everything else in src must build on the PC.
DEVICE_ONLY = {"main.c", "bench.c", "mic.c", "recorder.c", "say.c"}
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
REAL_HEADERS = [
    export_c.FRONTEND_HEADER,
    export_c.MODEL_HEADER,
    export_c.VECTORS_HEADER,
    export_gmm.GMM_HEADER,
    export_gmm.VECTORS_HEADER,
]
HAVE_REAL = all((REAL_MODEL / h).exists() for h in REAL_HEADERS)
HEADER_SETS = ["synthetic"] + (["real"] if HAVE_REAL else [])


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
    # Writes the same andon_frontend.h again: both exports use the Stage 1 features.
    export_gmm.write(*export_gmm.synthetic_export(seed=0), out)
    return out


def test_host_tests_exist():
    assert TESTS, "no firmware/test/test_*.c found"
    assert HOST_SOURCES, "no hardware-free firmware/src/*.c found"


# Skipped only off CI: on CI a missing compiler must fail, or H4's check would pass by
# running nothing.
@pytest.mark.skipif(COMPILER is None and not os.environ.get("CI"), reason="no C compiler")
@pytest.mark.parametrize("headers", HEADER_SETS)
@pytest.mark.parametrize("test_src", TESTS, ids=lambda p: p.stem)
def test_firmware_host(test_src: Path, headers: str, synthetic_headers: Path, tmp_path: Path):
    assert COMPILER is not None, "CI needs a C compiler for the firmware host tests"
    model_dir = synthetic_headers if headers == "synthetic" else REAL_MODEL
    exe = tmp_path / (test_src.stem + (".exe" if sys.platform == "win32" else ""))
    sources = [str(test_src), *map(str, HOST_SOURCES)]
    includes = [f"-I{FIRMWARE / 'src'}", f"-I{model_dir}"]
    run = _build_and_run([COMPILER, *FLAGS, *includes, *sources, "-lm"], exe)
    print(run.stdout)  # measured errors and RAM sizes, shown with pytest -s
    assert run.returncode == 0, run.stdout + run.stderr
    assert "all checks passed" in run.stdout


def _c_defines(path: Path) -> dict[str, int]:
    """Every integer ``#define NAME value`` in a C header (an optional ``u`` suffix allowed)."""
    text = path.read_text(encoding="utf-8")
    found = re.findall(r"^#define (\w+) \(?(\d+)u?\)?\s*(?://.*)?$", text, re.M)
    return {name: int(value) for name, value in found}


# The device has one front end: C constants for the frame and the mic, and the mel bank in
# andon_frontend.h, which both exporters build from the features of their config. A model
# for the real fans is only usable on the device if its features are those, so a config
# change that breaks this fails here instead of on the board.
FRONT_END_SOURCE = CONFIGS / "phase3_idcnn_mixup.yaml"
_STAGE2 = [
    (p, c)
    for p in sorted(CONFIGS.glob("*.yaml"))
    if (c := load_config(p)).raw.get("final") is True and str(c.phase).startswith("H")
]
# A trial config (H6 on) names the model it runs with model_config instead of holding one.
STAGE2_FINAL = [p for p, c in _STAGE2 if "model_config" not in c.raw]
STAGE2_TRIALS = [c for _, c in _STAGE2 if "model_config" in c.raw]


def test_stage2_final_configs_exist():
    assert {p.stem for p in STAGE2_FINAL} >= {"h3_gmm", "h3_level"}
    assert {c.name for c in STAGE2_TRIALS} >= {"h6_jidoka"}


def test_stage2_trials_run_a_final_model_with_the_device_front_end():
    for c in STAGE2_TRIALS:
        assert (CONFIGS.parent / c.raw["model_config"]).resolve() in STAGE2_FINAL, c.name


# Config.logmel reads only these keys, so any other key (fmax, say) would be ignored
# everywhere, by training and the export alike, instead of failing.
FEATURE_KEYS = {"sample_rate", "n_fft", "hop_length", "n_mels", "power"}


@pytest.mark.parametrize("path", STAGE2_FINAL, ids=lambda p: p.stem)
def test_stage2_final_features_are_the_device_front_end(path: Path):
    for cfg_path in (path, FRONT_END_SOURCE):
        assert set(load_config(cfg_path).raw["features"]) == FEATURE_KEYS, cfg_path.name
    assert load_config(path).logmel == load_config(FRONT_END_SOURCE).logmel


# The flashed headers (firmware/model, exported locally from the trained models) are checked
# too when they exist; CI has only the synthetic ones.
@pytest.mark.parametrize("headers", HEADER_SETS)
def test_front_end_source_matches_c_constants(headers: str, synthetic_headers: Path):
    cfg = load_config(FRONT_END_SOURCE).logmel
    src = FIRMWARE / "src"
    assert cfg.power == 2.0  # logmel.c squares the magnitude
    assert cfg.n_fft == _c_defines(src / "logmel.h")["LOGMEL_N_FFT"]
    # The board's detector frames its stream every LOGMEL_HOP samples.
    assert cfg.hop_length == _c_defines(src / "logmel.h")["LOGMEL_HOP"]
    assert cfg.sample_rate == _c_defines(src / "mic_stream.h")["MIC_SAMPLE_RATE_HZ"]
    model_dir = synthetic_headers if headers == "synthetic" else REAL_MODEL
    fe = _c_defines(model_dir / export_c.FRONTEND_HEADER)
    assert (fe["ANDON_SAMPLE_RATE"], fe["ANDON_N_FFT"]) == (cfg.sample_rate, cfg.n_fft)
    assert (fe["ANDON_HOP_LENGTH"], fe["ANDON_N_MELS"]) == (cfg.hop_length, cfg.n_mels)
    assert _c_defines(model_dir / export_c.MODEL_HEADER)["ANDON_MODEL_N_MELS"] == cfg.n_mels
    gmm = _c_defines(model_dir / export_gmm.GMM_HEADER)
    h3 = load_config(CONFIGS / "h3_gmm.yaml").raw["window"]
    assert gmm["ANDON_GMM_N_DIM"] == cfg.n_mels
    assert (gmm["ANDON_GMM_WINDOW_FRAMES"], gmm["ANDON_GMM_WINDOW_HOP"]) == (
        h3["frames"],
        h3["hop"],
    )
    # Every clip was recorded at this gain, so the board must listen at it too.
    recording = yaml.safe_load((CONFIGS / "h2_recording.yaml").read_text(encoding="utf-8"))
    assert gmm["ANDON_MIC_GAIN_SHIFT"] == recording["audio"]["gain_shift"] == 4
