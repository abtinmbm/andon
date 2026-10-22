"""Phase 4: the C header export of the int8 CNN and its test vectors."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_idcnn import CONFIGS

from andon import export_c
from andon.experiment import load_config


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[export_c.Export, Path]:
    out = tmp_path_factory.mktemp("model")
    ex = export_c.synthetic_export(seed=0, n=2)
    export_c.write(ex, out, n=2)  # writes, then verifies the read-back
    return ex, out


def test_c_arrays_round_trip_exactly():
    floats = np.array([0.1, -3.0, 1e-7, 12345.678, 2.0**-20], dtype=np.float32)
    ints = np.array([-128, 0, 127], dtype=np.int8)
    text = export_c.c_array("float", "f", floats) + export_c.c_array("int8_t", "i", ints)
    assert "-3.0f" in text  # a whole number still reads as a float literal
    arrays, _ = export_c.read_header(text)
    np.testing.assert_array_equal(arrays["f"], floats)
    np.testing.assert_array_equal(arrays["i"], ints)
    assert export_c._c_float(0.047613306606517115, double=True) == "0.047613306606517115"


def test_synthetic_export_round_trips_through_the_headers(exported):
    ex, out = exported
    text = (out / export_c.MODEL_HEADER).read_text(encoding="utf-8")
    back = export_c.model_from_header(text)
    q = ex.qmodel
    assert back.input_q == q.input_q and back.pool_size == q.pool_size == 32
    for a, b in zip(back.convs, q.convs, strict=True):
        np.testing.assert_array_equal(a.weight, b.weight)
        np.testing.assert_array_equal(a.mult, b.mult)
        np.testing.assert_array_equal(a.shift, b.shift)
        assert (a.stride, a.pool_before, a.out.zero_point) == (
            b.stride,
            b.pool_before,
            b.out.zero_point,
        )
    np.testing.assert_array_equal(back.head_weight, q.head_weight)
    # The final model's layer shapes: 128 mels, 64-frame patches, 16-32-64-128 channels.
    assert [c.weight.shape for c in back.convs] == [
        (16, 1, 3, 3),
        (32, 16, 3, 3),
        (64, 32, 3, 3),
        (128, 64, 3, 3),
    ]
    tv, d = export_c.read_header((out / export_c.VECTORS_HEADER).read_text(encoding="utf-8"))
    assert d["ANDON_TV_N"] == 3 and d["ANDON_TV_N_AUDIO"] == 2  # 2 from the audio + stress
    assert d["ANDON_TV_FRAMES"] == 64 + 32
    assert tv["andon_tv_conv3"].size == 3 * 128 * 8 * 4


def test_stress_vector_hits_the_int8_clamps(exported):
    """The last vector saturates the input at both ends and every conv layer at 127."""
    _, out = exported
    tv, d = export_c.read_header((out / export_c.VECTORS_HEADER).read_text(encoding="utf-8"))
    n = d["ANDON_TV_N"]
    stress = tv["andon_tv_input"].reshape(n, -1)[-1]
    assert (stress == -128).any() and (stress == 127).any()
    for i in range(4):
        assert (tv[f"andon_tv_conv{i}"].reshape(n, -1)[-1] == 127).any()


def test_verify_catches_a_changed_weight(exported, tmp_path):
    _, out = exported
    for name in (export_c.MODEL_HEADER, export_c.VECTORS_HEADER):
        shutil.copy(out / name, tmp_path / name)
    path = tmp_path / export_c.MODEL_HEADER
    text = path.read_text(encoding="utf-8")
    start = text.index("andon_conv1_weight")
    body = text.index("\n    ", start) + 5
    first = text[body : text.index(",", body)]
    changed = str(int(first) + 1 if int(first) < 127 else int(first) - 1)
    path.write_text(text[:body] + changed + text[body + len(first) :], encoding="utf-8")
    with pytest.raises(AssertionError):
        export_c.verify(tmp_path, export_c.synthetic_export(seed=0, n=2).logmel_cfg)


@pytest.mark.skipif(not (shutil.which("gcc") or shutil.which("cc")), reason="no C compiler")
def test_headers_compile_as_strict_c(exported, tmp_path):
    _, out = exported
    src = tmp_path / "smoke.c"
    src.write_text(
        '#include "andon_model.h"\n#include "andon_test_vectors.h"\n'
        "int main(void) { return andon_conv0_weight[0] + andon_tv_input[0] > 1000; }\n"
    )
    compiler = shutil.which("gcc") or shutil.which("cc")
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-Wno-unused-const-variable"]
    result = subprocess.run(
        [compiler, *flags, f"-I{out}", "-c", str(src), "-o", str(tmp_path / "smoke.o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_real_export_needs_an_int8_cnn_config():
    with pytest.raises(ValueError, match="int8 machine-ID CNN"):
        export_c.real_export(load_config(CONFIGS / "phase4_gmm_id_int8.yaml"), "fan", 0)
