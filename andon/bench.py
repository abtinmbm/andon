"""Turn the Pico 2 bench's serial log into a results table.

`andon_bench.uf2` prints one block per run over USB serial. The capture is committed as is
(``results/h5_bench_pico2.txt``, and ``results/h5_gmm_bench_pico2.txt`` once the bench also
ran the per-fan GMM), and this module parses it into Markdown tables, so every on-device
figure the docs quote comes from a generated file, like the AUC tables do.

    uv run python -m andon.bench results/h5_bench_pico2.txt --out results/h5_bench_pico2.md
    uv run python -m andon.bench results/h5_gmm_bench_pico2.txt --out results/h5_gmm_bench_pico2.md

With ``--port COM6`` it first records the board into the log file, then tabulates it. A log
whose runs carry the GMM lines gets a second table, for the GMM.
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from andon.experiment import git_commit

# The Phase 4 budget (PLAN.md): under this much compute per 32 ms audio frame.
BUDGET_MS_PER_FRAME = 20.0

_RUN = re.compile(r"^== andon bench, run (\d+), clk_sys (\d+) Hz")
_LOGMEL = re.compile(
    r"^log-mel: (\d+) frames, worst \|C - Python\| (\S+) dB .*?, ([\d.]+) ms/frame avg"
)
_CNN = re.compile(r"^cnn: (\d+) patches, head sum mismatches (\d+), worst score err (\S+)")
_E2E = re.compile(
    r"^end to end: \d+ patches, int8 input diffs (\d+) of (\d+), worst score err (\S+) .*?, "
    r"([\d.]+) ms/patch avg"
)
_REAL = re.compile(r"^real time: ([\d.]+) ms of compute per (\d+) ms of audio")
_GMM_MODEL = re.compile(r"^gmm model: (.+)$")
_GMM_FE = re.compile(
    r"^gmm front end: (\d+) frames of fan audio, worst \|C - Python\| (\S+) dB .*?, "
    r"worst window mean err (\S+) dB"
)
_GMM = re.compile(
    r"^gmm: (\d+) window scores, worst rel err (\S+) from Python's means .*?, (\S+) end to "
    r"end .*?, ([\d.]+) ms/score avg, ([\d.]+) ms max"
)
_GMM_REAL = re.compile(r"^gmm real time: ([\d.]+) ms of compute per (\d+) ms of audio")
_RAM = re.compile(r"^static RAM: (\d+) bytes")
_PASS = re.compile(r"^self-test: (PASS|FAIL)")


@dataclass
class BenchRun:
    run: int
    clk_hz: int
    logmel_ms_per_frame: float
    logmel_worst_db: float
    head_mismatches: int
    input_diffs: int
    input_values: int
    worst_score_err: float
    cnn_ms_per_patch: float
    ms_per_frame: float
    frame_ms: int
    ram_bytes: int
    passed: bool
    # The GMM block (H5, per-fan GMM); None in logs from before the bench ran it.
    gmm_model: str | None = None
    gmm_frames: int | None = None
    gmm_logmel_worst_db: float | None = None
    gmm_mean_worst_db: float | None = None
    gmm_scores: int | None = None
    gmm_score_rel: float | None = None  # from Python's window means
    gmm_e2e_rel: float | None = None  # from the C log-mel and window mean
    gmm_ms_per_score: float | None = None
    gmm_ms_max: float | None = None
    gmm_ms_per_frame: float | None = None  # log-mel + one fan's score every hop frames

    @property
    def cnn_ms_per_frame(self) -> float:
        """The CNN's share of each frame: total compute minus the front end."""
        return self.ms_per_frame - self.logmel_ms_per_frame

    @property
    def has_gmm(self) -> bool:
        return all(getattr(self, f) is not None for f in GMM_FIELDS)

    @property
    def gmm_share_ms_per_frame(self) -> float:
        """The GMM's share of each frame: its total compute minus the front end."""
        assert self.gmm_ms_per_frame is not None
        return self.gmm_ms_per_frame - self.logmel_ms_per_frame


GMM_FIELDS = tuple(f for f in BenchRun.__dataclass_fields__ if f.startswith("gmm_"))
REQUIRED = tuple(f for f in BenchRun.__dataclass_fields__ if f not in GMM_FIELDS)


def parse_log(text: str) -> list[BenchRun]:
    """Parse every complete run block; a block cut off by the capture is dropped."""
    runs: list[BenchRun] = []
    fields: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if m := _RUN.match(line):
            fields = {"run": int(m[1]), "clk_hz": int(m[2])}
        elif not fields:
            continue
        elif m := _LOGMEL.match(line):
            fields |= {"logmel_worst_db": float(m[2]), "logmel_ms_per_frame": float(m[3])}
        elif m := _CNN.match(line):
            fields |= {"head_mismatches": int(m[2])}
        elif m := _E2E.match(line):
            fields |= {
                "input_diffs": int(m[1]),
                "input_values": int(m[2]),
                "worst_score_err": float(m[3]),
                "cnn_ms_per_patch": float(m[4]),
            }
        elif m := _REAL.match(line):
            fields |= {"ms_per_frame": float(m[1]), "frame_ms": int(m[2])}
        elif m := _GMM_MODEL.match(line):
            fields |= {"gmm_model": m[1]}
        elif m := _GMM_FE.match(line):
            fields |= {
                "gmm_frames": int(m[1]),
                "gmm_logmel_worst_db": float(m[2]),
                "gmm_mean_worst_db": float(m[3]),
            }
        elif m := _GMM.match(line):
            fields |= {
                "gmm_scores": int(m[1]),
                "gmm_score_rel": float(m[2]),
                "gmm_e2e_rel": float(m[3]),
                "gmm_ms_per_score": float(m[4]),
                "gmm_ms_max": float(m[5]),
            }
        elif m := _GMM_REAL.match(line):
            fields |= {"gmm_ms_per_frame": float(m[1])}
        elif m := _RAM.match(line):
            fields |= {"ram_bytes": int(m[1])}
        elif m := _PASS.match(line):
            fields |= {"passed": m[1] == "PASS"}
            gmm = [f for f in GMM_FIELDS if f in fields]
            # A run with only part of the GMM block was cut off or garbled: dropped too.
            if all(f in fields for f in REQUIRED) and len(gmm) in (0, len(GMM_FIELDS)):
                runs.append(BenchRun(**fields))
            fields = {}
    return runs


def bench_table(runs: list[BenchRun], est_ms_per_frame: float, est_ms_per_score: float) -> str:
    """Per-run rows, a mean row, and the comparison with the Phase 4 estimate."""
    if not runs:
        raise ValueError("no complete bench runs in the log")
    rows = [
        "| Run | Clock (MHz) | Log-mel ms/frame | CNN ms/patch | CNN ms/frame "
        "| Total ms per 32 ms frame | Worst log-mel err (dB) | Int8 input diffs "
        "| Worst score err | Static RAM (B) | Self-test |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in runs:
        rows.append(
            f"| {r.run} | {r.clk_hz / 1e6:.0f} | {r.logmel_ms_per_frame:.3f} "
            f"| {r.cnn_ms_per_patch:.2f} | {r.cnn_ms_per_frame:.3f} | {r.ms_per_frame:.3f} "
            f"| {r.logmel_worst_db:.2e} "
            f"| {r.input_diffs} of {r.input_values} | {r.worst_score_err:.2e} | {r.ram_bytes} "
            f"| {'PASS' if r.passed else 'FAIL'} |"
        )
    total = mean(r.ms_per_frame for r in runs)
    patch = mean(r.cnn_ms_per_patch for r in runs)
    rows.append(
        f"| mean of {len(runs)} | | {mean(r.logmel_ms_per_frame for r in runs):.3f} | {patch:.2f} "
        f"| {mean(r.cnn_ms_per_frame for r in runs):.3f} | {total:.3f} | | | | | |"
    )
    verdict = "under" if total < BUDGET_MS_PER_FRAME else "**over**"
    summary = [
        "",
        f"Measured: {total:.2f} ms of compute per 32 ms frame on one core, {verdict} the "
        f"{BUDGET_MS_PER_FRAME:.0f} ms budget. Phase 4 estimate (`results/phase4_mcu.md`): "
        f"{est_ms_per_frame:.2f} ms per frame and {est_ms_per_score:.2f} ms per CNN score; "
        f"measured {patch:.2f} ms per score, {patch / est_ms_per_score:.1f} times the estimate.",
    ]
    return "\n".join(rows + summary)


def gmm_table(runs: list[BenchRun]) -> str:
    """Per-run rows and a mean row for the per-fan GMM, timed as the device will run it."""
    if not runs or not all(r.has_gmm for r in runs):
        raise ValueError("no complete GMM blocks in the log")
    models = {r.gmm_model for r in runs}
    if len(models) != 1:
        raise ValueError(f"the runs flashed different GMMs: {sorted(models)}")
    rows = [
        "| Run | Clock (MHz) | Log-mel ms/frame | GMM ms/score | GMM ms/frame "
        "| Total ms per 32 ms frame | Worst log-mel err, fan audio (dB) "
        "| Worst window-mean err (dB) | Worst score err (relative), from Python's means "
        "| Worst score err (relative), end to end | Static RAM (B) | Self-test |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in runs:
        rows.append(
            f"| {r.run} | {r.clk_hz / 1e6:.0f} | {r.logmel_ms_per_frame:.3f} "
            f"| {r.gmm_ms_per_score:.3f} | {r.gmm_share_ms_per_frame:.3f} "
            f"| {r.gmm_ms_per_frame:.3f} | {r.gmm_logmel_worst_db:.2e} "
            f"| {r.gmm_mean_worst_db:.2e} | {r.gmm_score_rel:.2e} | {r.gmm_e2e_rel:.2e} "
            f"| {r.ram_bytes} | {'PASS' if r.passed else 'FAIL'} |"
        )
    total = mean(r.gmm_ms_per_frame for r in runs)
    rows.append(
        f"| mean of {len(runs)} | | {mean(r.logmel_ms_per_frame for r in runs):.3f} "
        f"| {mean(r.gmm_ms_per_score for r in runs):.3f} "
        f"| {mean(r.gmm_share_ms_per_frame for r in runs):.3f} | {total:.3f} "
        "| | | | | | |"
    )
    verdict = "under" if total < BUDGET_MS_PER_FRAME else "**over**"
    first = runs[0]
    failed = [str(r.run) for r in runs if not r.passed]
    # The scores matching Python is half of H5's check: a failed self-test must not read
    # as a clean result just because the timing fits.
    checks = (
        f"**Self-test FAILED in run(s) {', '.join(failed)}**: the C results do not match "
        "Python within the tolerances."
        if failed
        else "Every run's self-test passed: the C scores match Python within the tolerances."
    )
    summary = [
        "",
        f"{checks} Measured: {total:.3f} ms of compute per 32 ms frame on one core (the "
        f"log-mel of every frame, plus one fan's GMM score for each window), {verdict} the "
        f"{BUDGET_MS_PER_FRAME:.0f} ms budget. Each run scores {first.gmm_scores} windows "
        f"({first.gmm_frames} frames of fan audio); slowest score "
        f"{max(r.gmm_ms_max for r in runs):.3f} ms.",
    ]
    return "\n".join(rows + summary)


def phase4_estimate(table: Path, config: str = "phase4_idcnn_int8") -> tuple[float, float]:
    """The estimated (ms per frame, ms per score) for `config` from the generated Phase 4 table."""
    header: list[str] | None = None
    for line in table.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if "Est. ms per frame" in cells:
            header = cells
        elif header and cells and cells[0] == f"`{config}`":
            return float(cells[header.index("Est. ms per frame")]), float(
                cells[header.index("Est. ms per score")]
            )
    raise ValueError(f"no estimate for {config} in {table}")


def capture(
    readline: Callable[[], bytes],
    seconds: float,
    header: list[str],
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Every non-empty line the board prints for ``seconds``, verbatim, after ``header``
    (``#`` comment lines). The bench repeats its block every few seconds, so a block the
    capture cuts at either end is dropped by ``parse_log``."""
    lines = [f"# {h}" for h in header]
    end = clock() + seconds
    while clock() < end:
        line = readline().decode(errors="replace").rstrip("\r\n")
        if line.strip():
            lines.append(line)
            print(line, flush=True)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="raw serial capture of andon_bench.uf2")
    parser.add_argument("--out", type=Path, help="write the Markdown table here")
    parser.add_argument(
        "--estimate", type=Path, default=Path("results/phase4_mcu.md"), help="Phase 4 budget table"
    )
    parser.add_argument("--port", help="record the board (e.g. COM6) into LOG first")
    parser.add_argument("--seconds", type=float, default=25.0, help="with --port")
    args = parser.parse_args(argv)
    est_frame, est_score = phase4_estimate(args.estimate)

    if args.port:
        import serial  # only needed with a board attached

        header = [
            f"andon_bench.uf2 on a Pico 2, USB serial on {args.port}, "
            f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC; working tree {git_commit()}.",
            f"Verbatim capture of {args.seconds:g} s.",
        ]
        with serial.Serial(args.port, 115200, timeout=0.2) as port:
            text = capture(port.readline, args.seconds, header)
        args.log.write_text(text, encoding="utf-8")
        print(f"wrote {args.log}")

    runs = parse_log(args.log.read_text(encoding="utf-8"))
    header = (
        f"<!-- Generated by `uv run python -m andon.bench {args.log.as_posix()} --out "
        f"{args.out.as_posix() if args.out else '...'}`. -->\n\n"
        "## Int8 CNN and log-mel on the Pico 2 (on-device bench)\n\n"
        "Model `phase4_idcnn_int8`, fan, seed 0, built into `andon_bench.uf2`. Errors are "
        "against the Python reference on the exported test vectors. Times from the Pico's "
        "microsecond timer.\n\n"
    )
    text = header + bench_table(runs, est_frame, est_score) + "\n"
    if runs and all(r.has_gmm for r in runs):
        text += (
            "\n## Per-fan GMM on the Pico 2 (on-device bench)\n\n"
            f"Model: {runs[0].gmm_model} (float32, as `andon.export_gmm` writes it, read "
            "from the board's log). Each fan's test audio is streamed through the C "
            "log-mel and the window mean, and every window is scored by all three fans' GMMs. "
            "Errors are against Python (scikit-learn) on the exported test vectors.\n\n"
            + gmm_table(runs)
            + "\n"
        )
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
