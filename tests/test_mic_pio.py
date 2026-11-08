"""Run firmware/src/mic.pio in a small PIO simulator against a model of the ICS-43434, so the
I2S bit alignment is checked before the board: the mic is a peripheral that gives no error
when the clocks are wrong, just wrong numbers.

The simulator covers only the instructions mic.pio uses (set x, in pins, jmp x--, nop, each
with a 2-bit side-set). The mic model follows the datasheet (DS-000069 v1.2): with SEL low it
drives the 24-bit sample MSB first, starting one BCLK after the WS falling edge, changing its
output on falling BCLK edges, and tristates the rest of the frame (modelled as random bits)."""

import random
import re
from pathlib import Path

import pytest

PIO = Path(__file__).resolve().parents[1] / "firmware" / "src" / "mic.pio"
CYCLES_PER_FRAME = 128  # must match I2S_MIC_CYCLES_PER_FRAME in mic.pio


def parse(path: Path) -> tuple[list[tuple[str, list[str], int]], dict[str, int], int, int]:
    """Instructions as (op, args, side), labels, wrap target and wrap source."""
    program, labels = [], {}
    wrap_target = wrap = None
    body = path.read_text().split("% c-sdk")[0]
    for raw in body.splitlines():
        line = raw.split(";")[0].strip()
        if not line or line.startswith((".program", ".side_set")):
            continue
        if line == ".wrap_target":
            wrap_target = len(program)
        elif line == ".wrap":
            wrap = len(program) - 1
        elif line.endswith(":"):
            labels[line.removeprefix("public ").removesuffix(":")] = len(program)
        else:
            m = re.fullmatch(r"(\w+)\s*(.*?)\s+side\s+0b([01]{2})", line)
            assert m, f"unsupported line: {raw!r}"
            args = re.split(r"[\s,]+", m.group(2)) if m.group(2) else []
            program.append((m.group(1), args, int(m.group(3), 2)))
    assert wrap_target is not None and wrap is not None
    return program, labels, wrap_target, wrap


class Mic:
    """ICS-43434 with SEL low, as a function of the BCLK and WS lines."""

    def __init__(self, samples: list[int], rng: random.Random):
        self.samples = samples
        self.rng = rng
        self.sd = 0
        self.ws_prev = 0  # WS is driven low before the program starts
        self.slot = 99  # BCLK cycles since the last WS edge; 99 = not yet in sync
        self.frame = -1
        self.word = 0

    def falling_edge(self, ws: int) -> None:
        if ws != self.ws_prev:
            self.slot = 0
            if ws == 0:  # a new frame: the mic samples at the WS falling edge
                self.frame += 1
                self.word = self.samples[self.frame] & 0xFFFFFF
        else:
            self.slot += 1
        self.ws_prev = ws
        if ws == 0 and 1 <= self.slot <= 24 and self.frame >= 0:
            self.sd = (self.word >> (24 - self.slot)) & 1
        else:
            self.sd = self.rng.getrandbits(1)  # tristated: whatever the line floats to


def run(samples: list[int], seed: int = 0):
    program, labels, wrap_target, wrap = parse(PIO)
    mic = Mic(samples, random.Random(seed))
    pc, x = labels["entry_point"], 0
    isr, isr_bits, words = 0, 0, []
    bclk = ws = 0  # pio_sm_set_pins_with_mask drives both low before the start
    trace = []  # (bclk, ws) per instruction cycle
    # The first word is read before the mic has seen a WS edge (start-up junk; on the board
    # the mic needs 20 ms to wake anyway), so run one word longer and drop it.
    while len(words) < len(samples) + 1:
        op, args, side = program[pc]
        new_bclk, new_ws = side & 1, side >> 1
        if bclk == 1 and new_bclk == 0:
            mic.falling_edge(new_ws)
        bclk, ws = new_bclk, new_ws
        trace.append((bclk, ws))
        nxt = pc + 1
        if op == "set":
            assert args[0] == "x"
            x = int(args[1])
        elif op == "in":
            assert args == ["pins", "1"]
            assert bclk == 1, "data must be read while BCLK is high (after the rising edge)"
            isr = ((isr << 1) | mic.sd) & 0xFFFFFFFF
            isr_bits += 1
            if isr_bits == 32:  # autopush
                words.append(isr)
                isr, isr_bits = 0, 0
        elif op == "jmp":
            assert args[0] == "x--"
            if x != 0:
                nxt = labels[args[1]]
            x = (x - 1) & 0xFFFFFFFF
        else:
            assert op == "nop"
        pc = wrap_target if pc == wrap else nxt
    return words[1:], trace


def to_signed24(v: int) -> int:
    return v - (1 << 24) if v & 0x800000 else v


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_word_is_one_frames_sample(seed: int):
    rng = random.Random(100 + seed)
    samples = [-(1 << 23), (1 << 23) - 1, 0, -1, 1] + [
        rng.randrange(-(1 << 23), 1 << 23) for _ in range(40)
    ]
    words, _ = run(samples, seed)
    # One word per frame, in order; the tristated low byte is ignored.
    assert len(words) == len(samples)
    assert [to_signed24(w >> 8) for w in words] == samples


def test_clock_shape_matches_the_datasheet():
    words, trace = run([0] * 6)
    # Whole frames, from one WS falling edge to the next.
    ws_falls = [i for i in range(1, len(trace)) if trace[i - 1][1] == 1 and trace[i][1] == 0]
    assert len(ws_falls) >= 4
    for start, end in zip(ws_falls, ws_falls[1:], strict=False):
        frame = trace[start:end]
        # The clock divider assumes this many instructions per frame.
        assert len(frame) == CYCLES_PER_FRAME
        before = trace[start - 1 : end - 1]  # the cycle before each one in the frame
        rises = sum(1 for a, b in zip(before, frame, strict=True) if a[0] == 0 and b[0] == 1)
        assert rises == 64  # 64 BCLK cycles per WS frame
        assert sum(1 for b, _ in frame if b) == 64  # BCLK high exactly half the time: 50 % duty
        assert sum(1 for _, w in frame if w) == 64  # WS high for half the frame
    # WS only ever changes while BCLK goes low (on a falling edge).
    for (b0, w0), (b1, w1) in zip(trace, trace[1:], strict=False):
        if w0 != w1:
            assert b0 == 1 and b1 == 0


def test_bclk_is_within_the_datasheet_limits_at_16_khz():
    bclk_hz = 16000 * CYCLES_PER_FRAME / 2
    assert bclk_hz == 1_024_000
    period_ns = 1e9 / bclk_hz
    assert 303 <= period_ns <= 2500  # tSCP, DS-000069 v1.2
    assert 6250 <= 16000 <= 18750  # low-power mode sample-rate range
