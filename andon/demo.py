"""One replayable andon demo: a stretch of normal running, then a fault, scored window by window.

    uv run --group dashboard python -m andon.demo --config configs/phase5_demo.yaml --seed 0 \\
        --gif docs/andon_demo.gif

The stream is built exactly as the simulator builds its streams (same limits, same
calibration clips, same windows), for the one machine ID named in the config: the first
``normal_clips`` simulation normals, then the ID's first fault. The dashboard
(``dashboard/app.py``) replays it live; ``--gif`` renders the same replay to a file.
Both draw with :func:`draw`, so the GIF shows what the dashboard shows. The dashboard
also plays the stream's audio (:func:`stream_audio`), kept in step with the chart by
:class:`ReplayClock`.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from andon import score, simulate, train_idcnn
from andon.data.audio import read_wav, write_wav
from andon.data.features import load_or_extract
from andon.experiment import Config, load_config, set_seed
from andon.light import AndonLight, Light, LightSettings, Limits, fit_limits

LIGHT_COLOURS = {Light.GREEN: "#2e9d4f", Light.YELLOW: "#f2b705", Light.RED: "#d62828"}
LIGHT_TEXT = {
    Light.GREEN: "RUNNING NORMALLY",
    Light.YELLOW: "WARNING: SOUNDS UNUSUAL",
    Light.RED: "STOP: FAULT DETECTED",
}


@dataclass(frozen=True)
class DemoTrace:
    title: str
    times_s: np.ndarray  # when each window's score is ready: the end of the window
    scores: np.ndarray
    limits: Limits
    settings: LightSettings
    fault_start_s: float
    fault_end_s: float
    # The clips the stream is made of, for playing its audio; each clip's frame count sets
    # how much of the timeline it fills.
    clip_paths: tuple[Path, ...] = ()
    clip_frames: tuple[int, ...] = ()
    frame_samples: int = 512  # audio samples per log-mel frame (the hop)
    sample_rate: int = 16000


def states(trace: DemoTrace) -> list[Light]:
    """The light after each window, from green, with no reset."""
    return AndonLight(trace.limits, trace.settings).run(trace.scores)


def build_trace(
    cfg: Config, seed: int, device: torch.device, cache_dir: Path = simulate.CACHE_DIR
) -> DemoTrace:
    d = cfg.raw["demo"]
    sim_cfg = load_config(d["from_config"])
    source = simulate.source_config(sim_cfg)
    st = simulate.sim_settings(sim_cfg, source)
    machine_type, machine_id = str(d["machine_type"]), str(d["machine_id"])

    clips, cache_name = score.split_clips(source, machine_type, str(d["split"]))
    feats = load_or_extract(clips, source.logmel, cache_dir, cache_name)
    model, ckpt = train_idcnn.load_checkpoint(
        source.checkpoint_path(machine_type, seed), source, device
    )
    idx = [i for i, c in enumerate(clips) if c.machine_id == machine_id]
    if not idx:
        raise ValueError(f"no {machine_type} clips with ID {machine_id} in {d['split']}")
    calib, sim = simulate.calibration_split(clips, idx)
    fault = simulate.fault_episodes(clips, idx, st.fault_clips)[0]
    normals = sim[: int(d["normal_clips"])]

    off = feats.offsets
    frames = torch.from_numpy(feats.frames).to(device)
    stream = simulate.stream_frames(off, normals + fault)
    calib_scores, scores = simulate.score_streams(
        model,
        frames,
        [simulate.stream_frames(off, calib), stream],
        ckpt["classes"].index(machine_id),
        st.width,
        st.hop,
    )
    onset = int(sum(off[i + 1] - off[i] for i in normals))
    return DemoTrace(
        title=f"{machine_type} {machine_id}, {d['split']} clips, model seed {seed}",
        times_s=simulate.window_ends(len(stream), st.width, st.hop) * st.frame_s,
        scores=scores,
        limits=fit_limits(calib_scores, st.warn_sigma, st.control_sigma, st.log_scale),
        settings=st.light,
        fault_start_s=onset * st.frame_s,
        fault_end_s=len(stream) * st.frame_s,
        clip_paths=tuple(clips[i].path for i in normals + fault),
        clip_frames=tuple(int(off[i + 1] - off[i]) for i in normals + fault),
        frame_samples=source.logmel.hop_length,
        sample_rate=source.logmel.sample_rate,
    )


def stream_audio(trace: DemoTrace) -> np.ndarray:
    """The stream's audio, with second ``t`` of the audio at second ``t`` of the chart.

    A 10 s clip gives 313 centred frames, 10.016 s of timeline, so each clip is padded
    with silence (or trimmed) to its frames' length; otherwise the sound would drift ahead
    of the chart by 16 ms a clip.
    """
    parts = []
    for path, frames in zip(trace.clip_paths, trace.clip_frames, strict=True):
        samples, sr = read_wav(path)
        if sr != trace.sample_rate:
            raise ValueError(f"{path}: {sr} Hz, the stream is {trace.sample_rate} Hz")
        n = frames * trace.frame_samples
        parts.append(np.pad(samples[:n], (0, max(0, n - len(samples)))))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    write_wav(buf, samples, sample_rate)
    return buf.getvalue()


def windows_ready(trace: DemoTrace, t_s: float) -> int:
    """How many windows have been scored by second ``t_s`` of the stream."""
    return int(np.searchsorted(trace.times_s, t_s, side="right"))


@dataclass
class ReplayClock:
    """Where the replay is in the stream, driven by the wall clock while playing.

    The chart and the audio both follow it, so the chart can't fall behind the sound
    when drawing is slow, and pausing, changing speed or restarting keeps them together.
    """

    pos_s: float = 0.0  # stream position when last started or stopped
    started_at: float | None = None  # wall-clock time of the last start; None when paused
    speed: float = 1.0

    @property
    def playing(self) -> bool:
        return self.started_at is not None

    def position(self, now: float) -> float:
        if self.started_at is None:
            return self.pos_s
        return self.pos_s + (now - self.started_at) * self.speed

    def start(self, now: float, speed: float) -> None:
        self.pos_s, self.started_at, self.speed = self.position(now), now, speed

    def stop(self, now: float) -> None:
        self.pos_s, self.started_at = self.position(now), None


def draw(fig, trace: DemoTrace, n: int, lights: list[Light]) -> None:
    """Draw the light and the score chart after the first ``n`` windows onto ``fig``."""
    fig.clear()
    # Fixed positions, not tight_layout: the label under the light changes width, and a
    # layout that follows it would make the chart jump between frames.
    ax_light = fig.add_axes((0.0, 0.08, 0.27, 0.86))
    ax = fig.add_axes((0.37, 0.17, 0.61, 0.74))
    state = lights[n - 1] if n else Light.GREEN
    t_now = trace.times_s[n - 1] if n else 0.0

    ax_light.set_xlim(-1, 1)
    ax_light.set_ylim(-1.5, 1.2)
    ax_light.set_aspect("equal")
    ax_light.axis("off")
    for y, s in zip((0.75, 0.0, -0.75), (Light.RED, Light.YELLOW, Light.GREEN), strict=True):
        on = s is state
        colour = LIGHT_COLOURS[s] if on else "#d9d9d9"
        ax_light.add_patch(plt_circle((0, y), 0.32, colour))
    ax_light.text(0, -1.3, LIGHT_TEXT[state], ha="center", va="center", fontsize=9, weight="bold",
                  color=LIGHT_COLOURS[state])  # fmt: skip

    x, y = trace.times_s[:n], trace.scores[:n]
    t_end = float(trace.times_s[-1])
    ax.axvspan(trace.fault_start_s, trace.fault_end_s, color="#d62828", alpha=0.08, lw=0)
    ax.text(trace.fault_start_s - 1, 0.97, "fault audio starts →", ha="right", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8, color="#a01c1c")  # fmt: skip
    if state is Light.RED:
        # The start of the current red run: after a reset on the dashboard, an earlier
        # red is history and must not set the label.
        k = n - 1
        while k > 0 and lights[k - 1] is Light.RED:
            k -= 1
        t_red = trace.times_s[k]
        if t_red > trace.fault_start_s:
            ax.text(trace.fault_start_s - 1, 0.86, f"red {t_red - trace.fault_start_s:.1f} s later",
                    ha="right", va="top", transform=ax.get_xaxis_transform(), fontsize=8,
                    color="#a01c1c", weight="bold")  # fmt: skip
    ax.axhline(trace.limits.warn, color=LIGHT_COLOURS[Light.YELLOW], ls="--", lw=1.2,
               label="warning limit (mean + 2σ)")  # fmt: skip
    ax.axhline(trace.limits.control, color=LIGHT_COLOURS[Light.RED], ls="--", lw=1.2,
               label="control limit (mean + 3σ)")  # fmt: skip
    ax.plot(x, y, color="#264653", lw=1.2, label="anomaly score, one window per second")
    if n:
        ax.plot(x[-1:], y[-1:], "o", color=LIGHT_COLOURS[state], ms=6)
    # Log axis: normal running sits near 0.01 and a fault near 10, and both must be readable.
    ax.set_yscale("log")
    lo = max(float(trace.scores.min()), 1e-4) / 2
    hi = max(float(trace.scores.max()), trace.limits.control) * 3
    ax.set_xlim(0, t_end)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"time (s)    now: {t_now:5.1f} s")
    ax.set_ylabel("anomaly score (log scale)")
    ax.set_title(trace.title, fontsize=9, loc="left")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plt_circle(xy: tuple[float, float], r: float, colour: str):
    from matplotlib.patches import Circle

    return Circle(xy, r, color=colour, ec="#555555", lw=1)


def render_gif(trace: DemoTrace, path: Path, fps: float, hold_s: float = 3.0) -> int:
    """Write the replay as a GIF, one frame per window; returns the number of frames."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from PIL import Image

    lights = states(trace)
    fig = plt.figure(figsize=(8, 3.2), dpi=90)
    frames = []
    for n in range(1, len(trace.scores) + 1):
        draw(fig, trace, n, lights)
        fig.canvas.draw()
        rgb = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        frames.append(rgb.convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    plt.close(fig)
    frames += [frames[-1]] * int(hold_s * fps)  # rest on the final state before looping
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return len(frames)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the andon demo replay.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_demo.yaml"))
    parser.add_argument("--seed", type=int, help="model seed (default: the config's)")
    parser.add_argument("--gif", type=Path, help="render the replay to this GIF")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed = int(cfg.raw["demo"]["seed"]) if args.seed is None else args.seed
    set_seed(seed)
    trace = build_trace(cfg, seed, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    lights = states(trace)
    first = {
        s: next((t for t, x in zip(trace.times_s, lights, strict=True) if x is s), None)
        for s in Light
    }
    print(f"{trace.title}: {len(trace.scores)} windows, fault from {trace.fault_start_s:.1f} s")
    for s, t in first.items():
        print(f"  first {s.value}: {'never' if t is None else f'{t:.1f} s'}")
    if args.gif:
        n = render_gif(trace, args.gif, fps=float(cfg.raw["demo"]["gif_fps"]))
        print(f"wrote {args.gif} ({n} frames, {args.gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
