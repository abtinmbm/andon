"""Stream test clips through the model window by window and drive the andon light, like a device.

    uv run python -m andon.simulate --config configs/phase5_andon.yaml --seed 0 1 2 \\
        --split dev_test
    uv run python -m andon.simulate --summary --split dev_test eval \\
        --out results/phase5_simulator.md --config configs/phase5_andon.yaml \\
        configs/phase5_andon_log.yaml configs/phase5_andon_1clip.yaml

A device hears one continuous stream, so the clips of a machine ID are played back to
back and a 64-frame window is scored every ``hop_frames`` frames, straddling the joins.
Per machine ID, half of the normal test clips fit the light's limits and the other half
are the normal running the light is judged on (the protocol is in the config's comment).

Two operating figures come out per machine type and seed, and go in the log row's notes:

- ``fa_per_h``: reds on normal audio per hour of normal audio
- ``delay_mean_s``: seconds from the start of a fault to red, over the faults caught
  (``detected`` says how many faults were caught before they ended)

A DCASE anomaly is one 10 s clip, but a failing machine stays faulty, so a fault is
played as ``fault_clips`` consecutive anomaly clips of the same machine ID.

The row's AUC and pAUC are the Phase 3 clip scores restricted to the simulation normals
and every anomaly clip, so the numbers can be tied back to Phase 3. Audio time is
counted as frames x 32 ms.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from andon import score, train_idcnn
from andon.data.dataset import MACHINE_TYPES, Clip
from andon.data.features import SplitFeatures, load_or_extract
from andon.evaluate import per_id_metrics
from andon.experiment import (
    RESULTS_LOG,
    Config,
    append_result,
    is_logged,
    load_config,
    read_results,
    set_seed,
)
from andon.light import AndonLight, Light, LightSettings, Limits, fit_limits
from andon.models.idcnn import IdCNN, id_scores

CACHE_DIR = Path("data/cache")
_SCORE_BATCH = 512  # windows per forward pass


@dataclass(frozen=True)
class SimSettings:
    width: int  # frames per window
    hop: int  # frames between window starts
    frame_s: float  # seconds per frame
    warmup_clips: int  # normal clips played before each fault
    fault_clips: int  # anomaly clips played back to back as one fault
    warn_sigma: float
    control_sigma: float
    log_scale: bool
    light: LightSettings


def sim_settings(cfg: Config, source: Config) -> SimSettings:
    """Settings from the Phase 5 config, with the window width taken from the scoring model's."""
    lim, st = cfg.raw["limits"], cfg.raw["stream"]
    return SimSettings(
        width=int(source.raw["model"]["patch_frames"]),
        hop=int(st["hop_frames"]),
        frame_s=source.logmel.hop_length / source.logmel.sample_rate,
        warmup_clips=int(st["warmup_clips"]),
        fault_clips=int(st["fault_clips"]),
        warn_sigma=float(lim["warn_sigma"]),
        control_sigma=float(lim["control_sigma"]),
        log_scale=bool(lim["log_scale"]),
        light=LightSettings(**{k: int(v) for k, v in cfg.raw["light"].items()}),
    )


def source_config(cfg: Config) -> Config:
    """The config of the model doing the scoring (its checkpoints, features and patch width)."""
    return load_config(cfg.raw["model"]["from_config"])


def calibration_split(clips: list[Clip], idx: list[int]) -> tuple[list[int], list[int]]:
    """Normal clips among ``idx``, in name order, split alternately: (calibration, simulation)."""
    normals = sorted((i for i in idx if not clips[i].is_anomaly), key=lambda i: clips[i].path.name)
    if len(normals) < 4:
        raise ValueError(f"need at least 4 normal clips to split, got {len(normals)}")
    return normals[0::2], normals[1::2]


def stream_frames(offsets: np.ndarray, clip_idx: list[int]) -> np.ndarray:
    """Frame indices of the clips ``clip_idx`` played back to back."""
    return np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in clip_idx])


def window_ends(n_frames: int, width: int, hop: int) -> np.ndarray:
    """End (exclusive) of each window a device would score: one every ``hop`` frames."""
    if n_frames < width:
        raise ValueError(f"a stream of {n_frames} frames is shorter than {width}")
    return np.arange(0, n_frames - width + 1, hop) + width


@torch.no_grad()
def score_streams(
    model: IdCNN,
    frames: torch.Tensor,
    streams: list[np.ndarray],
    true_id: int,
    width: int,
    hop: int,
) -> list[np.ndarray]:
    """``-log p(true ID)`` of every window of every stream; streams are frame-index arrays."""
    model.eval()
    windows = [
        s[(window_ends(len(s), width, hop) - width)[:, None] + np.arange(width)] for s in streams
    ]
    flat = np.concatenate(windows)
    out = np.empty(len(flat), dtype=np.float64)
    for i in range(0, len(flat), _SCORE_BATCH):
        idx = torch.from_numpy(flat[i : i + _SCORE_BATCH]).to(frames.device)
        ids = torch.full((len(idx),), true_id, dtype=torch.long, device=frames.device)
        out[i : i + len(idx)] = id_scores(model(frames[idx]), ids).cpu().numpy()
    return np.split(out, np.cumsum([len(w) for w in windows])[:-1])


def count_false_alarms(scores: np.ndarray, limits: Limits, settings: LightSettings) -> int:
    """Reds on a normal stream; each is reset at once, as an operator who finds nothing would."""
    light, n = AndonLight(limits, settings), 0
    for s in scores:
        if light.update(float(s)) is Light.RED:
            n += 1
            light.reset()
    return n


def frames_to_red(
    scores: np.ndarray, ends: np.ndarray, onset: int, limits: Limits, settings: LightSettings
) -> int | None:
    """Frames from ``onset`` to the end of the window that turned the light red, or None.

    A red from a window that ends before the fault starts is a false alarm, already
    counted on the false-alarm stream; it is reset and the light carries on.
    """
    light = AndonLight(limits, settings)
    for s, e in zip(scores, ends, strict=True):
        if light.update(float(s)) is Light.RED:
            if e > onset:
                return int(e - onset)
            light.reset()
    return None


@dataclass
class IdResult:
    machine_id: str
    false_alarms: int
    normal_s: float
    delays_s: list[float | None] = field(default_factory=list)  # one per fault; None = missed

    @property
    def detected(self) -> int:
        return sum(d is not None for d in self.delays_s)


def fault_episodes(clips: list[Clip], idx: list[int], fault_clips: int) -> list[list[int]]:
    """Anomaly clips among ``idx``, in name order, grouped into faults of ``fault_clips``.

    The last few clips that don't fill a whole fault are left out.
    """
    anomalies = sorted((i for i in idx if clips[i].is_anomaly), key=lambda i: clips[i].path.name)
    n = len(anomalies) // fault_clips
    if n == 0:
        raise ValueError(f"need at least {fault_clips} anomaly clips, got {len(anomalies)}")
    return [anomalies[k * fault_clips : (k + 1) * fault_clips] for k in range(n)]


def simulate_id(
    model: IdCNN,
    frames: torch.Tensor,
    feats: SplitFeatures,
    clips: list[Clip],
    idx: list[int],
    true_id: int,
    st: SimSettings,
) -> tuple[IdResult, Limits, list[int]]:
    """Fit limits, count false alarms and time every fault for one machine ID.

    Returns the result, the limits and the simulation normal clips.
    """
    calib, sim = calibration_split(clips, idx)
    faults = fault_episodes(clips, idx, st.fault_clips)
    off = feats.offsets
    # Warmup normals rotate through the simulation pool, never the calibration clips.
    warmups = [
        [sim[(k * st.warmup_clips + j) % len(sim)] for j in range(st.warmup_clips)]
        for k in range(len(faults))
    ]
    streams = [stream_frames(off, calib), stream_frames(off, sim)]
    streams += [stream_frames(off, w + f) for w, f in zip(warmups, faults, strict=True)]
    scored = score_streams(model, frames, streams, true_id, st.width, st.hop)

    limits = fit_limits(scored[0], st.warn_sigma, st.control_sigma, st.log_scale)
    result = IdResult(
        machine_id=clips[idx[0]].machine_id,
        false_alarms=count_false_alarms(scored[1], limits, st.light),
        normal_s=len(streams[1]) * st.frame_s,
    )
    for w, s, scores in zip(warmups, streams[2:], scored[2:], strict=True):
        onset = int(sum(off[i + 1] - off[i] for i in w))
        f = frames_to_red(scores, window_ends(len(s), st.width, st.hop), onset, limits, st.light)
        result.delays_s.append(None if f is None else f * st.frame_s)
    return result, limits, sim


def type_notes(results: list[IdResult]) -> str:
    """The operating figures of one machine type, pooled over its IDs, as ``key=value`` notes."""
    n_fa = sum(r.false_alarms for r in results)
    normal_h = sum(r.normal_s for r in results) / 3600
    delays = [d for r in results for d in r.delays_s if d is not None]
    n_anom = sum(len(r.delays_s) for r in results)
    detected = len(delays)
    mean_d = f"{np.mean(delays):.2f}" if delays else "nan"
    median_d = f"{np.median(delays):.2f}" if delays else "nan"
    per_id = " ".join(
        f"{r.machine_id}:fa={r.false_alarms},det={r.detected}/{len(r.delays_s)}" for r in results
    )
    return (
        f"fa={n_fa} normal_h={normal_h:.4f} fa_per_h={n_fa / normal_h:.2f} "
        f"detected={detected}/{n_anom} delay_mean_s={mean_d} delay_median_s={median_d} {per_id}"
    )


def simulate_type(
    source: Config,
    st: SimSettings,
    model: IdCNN,
    classes: list[str],
    clips: list[Clip],
    feats: SplitFeatures,
    device: torch.device,
) -> tuple[float, float, str]:
    """AUC, pAUC (Phase 3 clip scores, simulation clips only) and the notes of one type."""
    frames = torch.from_numpy(feats.frames).to(device)
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clips):
        by_id[c.machine_id].append(i)
    clip_scores = train_idcnn.score_clips(source, model, classes, clips, feats, device)

    results, keep = [], []
    for machine_id, idx in sorted(by_id.items()):
        r, _, sim = simulate_id(model, frames, feats, clips, idx, classes.index(machine_id), st)
        results.append(r)
        keep += sim + [i for i in idx if clips[i].is_anomaly]
    keep.sort()
    per_id = per_id_metrics([clips[i] for i in keep], clip_scores[keep])
    auc = float(np.mean([a for a, _ in per_id.values()]))
    pauc = float(np.mean([p for _, p in per_id.values()]))
    return auc, pauc, type_notes(results)


def run(
    cfg: Config,
    seeds: list[int],
    machine_types: list[str],
    split: str,
    device: torch.device,
    cache_dir: Path = CACHE_DIR,
    log: Path = RESULTS_LOG,
) -> None:
    source = source_config(cfg)
    st = sim_settings(cfg, source)
    todo = {
        t: [s for s in seeds if not is_logged(cfg, t, s, split, log=log)] for t in machine_types
    }
    missing = [source.checkpoint_path(t, s) for t, ss in todo.items() for s in ss]
    missing = [p for p in missing if not p.exists()]
    if missing:  # check all first, so the log never holds part of a run
        raise FileNotFoundError(f"missing checkpoints: {[str(p) for p in missing]}")

    for machine_type in machine_types:
        for s in sorted(set(seeds) - set(todo[machine_type])):
            print(f"[{machine_type} seed {s}] {split} already logged, skipping")
        if not todo[machine_type]:
            continue
        clips, cache_name = score.split_clips(source, machine_type, split)
        feats = load_or_extract(clips, source.logmel, cache_dir, cache_name)
        for seed in todo[machine_type]:
            set_seed(seed)  # nothing here is random; kept so every entry point seeds the same way
            path = source.checkpoint_path(machine_type, seed)
            model, ckpt = train_idcnn.load_checkpoint(path, source, device)
            auc, pauc, notes = simulate_type(
                source, st, model, ckpt["classes"], clips, feats, device
            )
            append_result(cfg, machine_type, split, seed, auc, pauc, notes, log=log)
            print(f"[{machine_type} seed {seed}] {split} AUC {auc:.4f} pAUC {pauc:.4f} {notes}")


def parse_notes(notes: str) -> dict[str, str]:
    """The ``key=value`` pairs of a simulator row's notes (per-ID detail is skipped)."""
    return dict(p.split("=", 1) for p in notes.split() if "=" in p and ":" not in p)


def _mean_sd(values: list[float]) -> str:
    a = np.array(values, dtype=np.float64)
    sd = f"{a.std(ddof=1):.2f}" if a.size > 1 else "nan"
    return f"{a.mean():.2f} ± {sd}"


def summary_table(config: str, split: str, log: Path = RESULTS_LOG) -> str:
    """Markdown table of the operating figures on one split, mean ± sample std over seeds."""
    rows = [r for r in read_results(log) if r["config"] == config and r["split"] == split]
    by_type: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        if r["seed"] in by_type[r["machine_type"]]:
            raise ValueError(
                f"duplicate row: {config} {r['machine_type']} seed {r['seed']} {split}"
            )
        by_type[r["machine_type"]][r["seed"]] = parse_notes(r["notes"])
    seed_sets = [set(by_type[t]) for t in MACHINE_TYPES]
    if not all(seed_sets) or len(set(map(frozenset, seed_sets))) != 1:
        raise ValueError(f"{config} on {split}: every machine type needs the same seeds logged")
    seeds = sorted(seed_sets[0], key=int)

    def figures(n: dict[str, str]) -> tuple[float, float, float]:
        det, n_anom = (int(x) for x in n["detected"].split("/"))
        return float(n["fa_per_h"]), 100 * det / n_anom, float(n["delay_mean_s"])

    lines = [
        f"`{config}` on {split}, seeds {', '.join(seeds)}: mean ± sample std over seeds.",
        "",
        "| Machine type | False alarms per hour | Faults caught (%) | Mean delay to red (s) "
        "| Normal audio (h) | Faults |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    per_seed: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for t in MACHINE_TYPES:
        figs = [figures(by_type[t][s]) for s in seeds]
        for s, f in zip(seeds, figs, strict=True):
            per_seed[s].append(f)
        n0 = by_type[t][seeds[0]]
        cols = [_mean_sd([f[k] for f in figs]) for k in range(3)]
        lines.append(
            f"| {t} | {' | '.join(cols)} | {float(n0['normal_h']):.2f} "
            f"| {n0['detected'].split('/')[1]} |"
        )
    # The average of the six types, taken per seed, then mean ± std over seeds.
    avg = [np.mean(per_seed[s], axis=0) for s in seeds]
    cols = [_mean_sd([a[k] for a in avg]) for k in range(3)]
    lines.append(f"| **Mean of six types** | {' | '.join(cols)} | | |")
    if any("nan" in line for line in lines[4:]):
        lines += [
            "",
            "nan: in at least one seed no fault was caught, so there is no delay to average.",
        ]
    return "\n".join(lines)


def summaries(configs: list[str], splits: list[str], log: Path = RESULTS_LOG) -> list[str]:
    """A table per config and split that has rows; dev-only variants have no eval table."""
    logged = {(r["config"], r["split"]) for r in read_results(log)}
    tables = [summary_table(c, s, log) for c in configs for s in splits if (c, s) in logged]
    if not tables:
        raise ValueError(f"no rows for {configs} on {splits}")
    return tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stream test clips through the andon light.")
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        default=[Path("configs/phase5_andon.yaml")],
        help="one config to run; with --summary, one or more",
    )
    parser.add_argument("--seed", type=int, nargs="+", help="one or more seeds")
    parser.add_argument("--split", nargs="+", choices=sorted(score.SPLITS), required=True)
    parser.add_argument("--machines", nargs="*", help="machine types (default: all in the config)")
    parser.add_argument("--threads", type=int, help="CPU threads (default: PyTorch's choice)")
    parser.add_argument("--summary", action="store_true", help="print the tables, run nothing")
    parser.add_argument("--out", type=Path, help="with --summary: also write the tables here")
    args = parser.parse_args(argv)

    if args.summary:
        text = "\n\n".join(summaries([c.stem for c in args.config], args.split)) + "\n"
        print(text)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        return

    if not args.seed or len(args.split) != 1 or len(args.config) != 1:
        parser.error("a run needs --seed, exactly one --split and exactly one --config")
    cfg = load_config(args.config[0])
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config {cfg.name}, seeds {args.seed}, split {args.split[0]}, device {device}")
    run(cfg, args.seed, args.machines or cfg.machine_types, args.split[0], device)


if __name__ == "__main__":
    main()
