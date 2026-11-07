"""Turn the Pico 2 bench's serial log into a results table.

`andon_bench.uf2` prints one block per run over USB serial. The capture is committed as is
(``results/h5_bench_pico2.txt``), and this module parses it into a Markdown table, so every
on-device figure the docs quote comes from a generated file, like the AUC tables do.

    uv run python -m andon.bench results/h5_bench_pico2.txt --out results/h5_bench_pico2.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

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

    @property
    def cnn_ms_per_frame(self) -> float:
        """The CNN's share of each frame: total compute minus the front end."""
        return self.ms_per_frame - self.logmel_ms_per_frame


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
        elif m := _RAM.match(line):
            fields |= {"ram_bytes": int(m[1])}
        elif m := _PASS.match(line):
            fields |= {"passed": m[1] == "PASS"}
            if len(fields) == len(BenchRun.__dataclass_fields__):
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="raw serial capture of andon_bench.uf2")
    parser.add_argument("--out", type=Path, help="write the Markdown table here")
    parser.add_argument(
        "--estimate", type=Path, default=Path("results/phase4_mcu.md"), help="Phase 4 budget table"
    )
    args = parser.parse_args(argv)
    est_frame, est_score = phase4_estimate(args.estimate)

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
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
