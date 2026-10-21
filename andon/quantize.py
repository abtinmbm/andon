"""Quantize a trained model to int8 and score a test split; one row per machine type and seed.

    uv run python -m andon.quantize --config configs/phase4_idcnn_int8.yaml --seed 0 1 2
    uv run python -m andon.quantize --summary --config phase4_idcnn_int8 phase4_baseline_int8
        phase4_gmm_id_int8 --out results/phase4_mcu.md

An int8 config names its float model (``model.from_config``). Before the int8 model is
scored, the float model is scored on the same clips and must reproduce its logged row
(within 0.0005), so the "before" AUC in the table belongs to exactly the model that was
quantized. Each row's notes carry the size and compute estimate from ``andon.budget``.

Experiment rule: eval is scored once per final model. ``--split eval`` is refused unless
the config says ``final: true``, which is set only after the choice is made on dev test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from andon import budget, quant, train, train_idcnn
from andon.data.dataset import MACHINE_TYPES, Clip, train_clips
from andon.data.features import (
    SplitFeatures,
    clip_means,
    load_or_extract,
    stack_frames,
    window_starts,
)
from andon.evaluate import _cells, _mean_sd, comparison_tables, log_scores, per_id_metrics
from andon.evaluate import clip_scores as ae_clip_scores
from andon.experiment import RESULTS_LOG, Config, is_logged, load_config, read_results, set_seed
from andon.models.classic import GMM, build_detector
from andon.models.idcnn import IdCNN
from andon.score import split_clips

CACHE_DIR = Path("data/cache")
MATCH_TOLERANCE = 0.0005  # logged AUCs have 4 decimals; as andon.reproduce
_BATCH = 256  # CNN patches per integer forward pass
_AE_CHUNK = 32_768  # autoencoder vectors per integer forward pass
CPU = torch.device("cpu")


def source_config(cfg: Config) -> Config:
    """The float model's config (its checkpoints or detector settings, and features)."""
    return load_config(cfg.raw["model"]["from_config"])


# --- scoring ----------------------------------------------------------------------------


def log_softmax(logits: np.ndarray) -> np.ndarray:
    top = logits.max(axis=1, keepdims=True)
    return logits - top - np.log(np.exp(logits - top).sum(axis=1, keepdims=True))


def cnn_clip_scores(
    qmodel: quant.QuantIdCNN, feats: SplitFeatures, true_ids: np.ndarray, width: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Int8 twin of ``train_idcnn.clip_scores``: mean over patches of ``-log p(true ID)``.

    Also returns, per clip, the share of int8 input values at -128 or 127. The input range
    is calibrated on normal audio, so a loud fault can saturate it and lose the very
    energy that sets it apart.
    """
    starts, owner = [], []
    for i in range(feats.n_clips):
        s = train_idcnn.patch_starts(int(feats.offsets[i + 1] - feats.offsets[i]), width, hop)
        starts.append(feats.offsets[i] + s)
        owner.append(np.full(len(s), i))
    starts_all, owner_all = np.concatenate(starts), np.concatenate(owner)
    patch_scores, patch_sat = np.empty(len(starts_all)), np.empty(len(starts_all))
    for i in range(0, len(starts_all), _BATCH):
        s, who = starts_all[i : i + _BATCH], owner_all[i : i + _BATCH]
        q_in = qmodel.quantize_input(feats.frames[s[:, None] + np.arange(width)])
        log_p = log_softmax(qmodel.head_acc(q_in) * qmodel.head_scale)
        patch_scores[i : i + len(s)] = -log_p[np.arange(len(s)), true_ids[who]]
        at_limit = (q_in == quant.INT8_MIN) | (q_in == quant.INT8_MAX)
        patch_sat[i : i + len(s)] = at_limit.reshape(len(s), -1).mean(axis=1)
    counts = np.bincount(owner_all, minlength=feats.n_clips)

    def per_clip(v: np.ndarray) -> np.ndarray:
        return np.bincount(owner_all, weights=v, minlength=feats.n_clips) / counts

    return per_clip(patch_scores), per_clip(patch_sat)


def ae_int8_clip_scores(
    qmodel: quant.QuantAutoencoder, feats: SplitFeatures, n_context: int
) -> np.ndarray:
    """Int8 twin of ``evaluate.clip_scores``: mean window reconstruction error per clip."""
    starts = window_starts(feats.offsets, n_context)
    errors = np.empty(len(starts))
    for i in range(0, len(starts), _AE_CHUNK):
        s = starts[i : i + _AE_CHUNK]
        errors[i : i + len(s)] = qmodel.errors(stack_frames(feats.frames, s, n_context))
    windows_per_clip = np.diff(feats.offsets) - n_context + 1
    first = np.r_[0, np.cumsum(windows_per_clip)[:-1]]
    return np.add.reduceat(errors, first) / windows_per_clip


def check_float_matches_log(
    src: Config,
    machine_type: str,
    seed: int,
    split: str,
    clips: list[Clip],
    scores: np.ndarray,
    log: Path = RESULTS_LOG,
) -> None:
    """Raise unless the float model's scores give its logged AUC and pAUC."""
    rows = [
        r
        for r in read_results(log)
        if r["config"] == src.name
        and r["machine_type"] == machine_type
        and r["seed"] == str(seed)
        and r["split"] == split
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {src.name} {machine_type} seed {seed} {split} row")
    per_id = per_id_metrics(clips, scores)
    auc = float(np.mean([a for a, _ in per_id.values()]))
    pauc = float(np.mean([p for _, p in per_id.values()]))
    logged = float(rows[0]["auc"]), float(rows[0]["pauc"])
    if abs(auc - logged[0]) > MATCH_TOLERANCE or abs(pauc - logged[1]) > MATCH_TOLERANCE:
        raise ValueError(
            f"{src.name} {machine_type} seed {seed}: float model gives AUC {auc:.4f} "
            f"pAUC {pauc:.4f}, but the log has {logged[0]:.4f} {logged[1]:.4f}"
        )


# --- one model type each ----------------------------------------------------------------


def calibrated_cnn(
    model: IdCNN, train_feats: SplitFeatures, width: int, n_calib: int, seed: int
) -> quant.QuantIdCNN:
    """Quantize ``model`` with one random patch from each of ``n_calib`` random training
    clips, drawn with ``seed``. The C export calls this too, so it exports exactly the
    model that was scored."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(train_feats.n_clips, size=n_calib, replace=False)
    frames = torch.from_numpy(train_feats.frames)
    calib = train_idcnn.random_patches(frames, train_feats.offsets, idx, width, rng)
    return quant.quantize_idcnn(model, calib)


def run_cnn(
    cfg: Config,
    src: Config,
    machine_type: str,
    seeds: list[int],
    split: str,
    cache_dir: Path,
    log: Path,
) -> list[tuple[int, np.ndarray, str]]:
    m = src.raw["model"]
    width, hop, n_mels = int(m["patch_frames"]), int(m["score_hop"]), src.logmel.n_mels
    train_feats, _, classes = train_idcnn.training_data(src, machine_type, cache_dir)
    clips, cache_name = split_clips(src, machine_type, split)
    feats = load_or_extract(clips, src.logmel, cache_dir, cache_name)
    index = {mid: k for k, mid in enumerate(classes)}
    true_ids = np.array([index[c.machine_id] for c in clips])
    out = []
    for seed in seeds:
        model, ckpt = train_idcnn.load_checkpoint(src.checkpoint_path(machine_type, seed), src, CPU)
        if ckpt["classes"] != classes:
            raise ValueError(f"checkpoint IDs {ckpt['classes']} differ from the data's {classes}")
        float_scores = train_idcnn.score_clips(src, model, classes, clips, feats, CPU)
        check_float_matches_log(src, machine_type, seed, split, clips, float_scores, log)

        n_calib = min(int(cfg.raw["quantize"]["calib_clips"]), train_feats.n_clips)
        qmodel = calibrated_cnn(model, train_feats, width, n_calib, seed)
        scores, sat = cnn_clip_scores(qmodel, feats, true_ids, width, hop)
        anomaly = np.array([c.is_anomaly for c in clips])

        b = budget.Budget(
            params=budget.folded_params(model),
            int8_bytes=qmodel.nbytes(),
            macs_per_decision=budget.module_macs(model, torch.zeros(1, width, n_mels)),
            frames_per_decision=hop,
            front_end_macs_per_frame=budget.front_end_macs(src.logmel.n_fft, n_mels),
        )
        sat_notes = (
            f"input_sat_normal_pct={100 * sat[~anomaly].mean():.3f} "
            f"input_sat_anomaly_pct={100 * sat[anomaly].mean():.3f}"
        )
        out.append((seed, scores, f"{b.notes()} calib_patches={n_calib} {sat_notes}"))
    return out


def run_autoencoder(
    cfg: Config,
    src: Config,
    machine_type: str,
    seeds: list[int],
    split: str,
    cache_dir: Path,
    log: Path,
) -> list[tuple[int, np.ndarray, str]]:
    n_context, n_mels = src.n_context, src.logmel.n_mels
    train_c = train_clips(src.data_root, machine_type)
    train_feats = load_or_extract(train_c, src.logmel, cache_dir, f"{machine_type}_train")
    clips, cache_name = split_clips(src, machine_type, split)
    feats = load_or_extract(clips, src.logmel, cache_dir, cache_name)
    all_starts = window_starts(train_feats.offsets, n_context)
    out = []
    for seed in seeds:
        model, _ = train.load_checkpoint(src.checkpoint_path(machine_type, seed), src, CPU)
        float_scores = ae_clip_scores(model, feats, n_context, CPU)
        check_float_matches_log(src, machine_type, seed, split, clips, float_scores, log)

        rng = np.random.default_rng(seed)
        n_calib = min(int(cfg.raw["quantize"]["calib_vectors"]), len(all_starts))
        starts = np.sort(rng.choice(all_starts, size=n_calib, replace=False))
        calib = torch.from_numpy(stack_frames(train_feats.frames, starts, n_context))
        qmodel = quant.quantize_autoencoder(model, calib)
        scores = ae_int8_clip_scores(qmodel, feats, n_context)

        b = budget.Budget(
            params=budget.folded_params(model),
            int8_bytes=qmodel.nbytes(),
            macs_per_decision=budget.module_macs(model, torch.zeros(1, n_context * n_mels)),
            frames_per_decision=1,
            front_end_macs_per_frame=budget.front_end_macs(src.logmel.n_fft, n_mels),
        )
        out.append((seed, scores, f"{b.notes()} calib_vectors={n_calib}"))
    return out


def run_gmm(
    cfg: Config,
    src: Config,
    machine_type: str,
    seeds: list[int],
    split: str,
    cache_dir: Path,
    log: Path,
) -> list[tuple[int, np.ndarray, str]]:
    if split != "dev_test":
        raise NotImplementedError("the GMM is refitted per split; only dev_test is wired up")
    model_cfg = src.raw["model"]
    if model_cfg["type"] != "gmm" or model_cfg["fit_per"] != "machine_id":
        raise ValueError("expected a per-ID GMM source config")
    train_c = train_clips(src.data_root, machine_type)
    train_feats = load_or_extract(train_c, src.logmel, cache_dir, f"{machine_type}_train")
    clips, cache_name = split_clips(src, machine_type, split)
    feats = load_or_extract(clips, src.logmel, cache_dir, cache_name)
    x_train, x_test = clip_means(train_feats), clip_means(feats)
    train_ids = np.array([c.machine_id for c in train_c])
    test_ids = np.array([c.machine_id for c in clips])
    n_mels = src.logmel.n_mels
    out = []
    for seed in seeds:
        set_seed(seed)
        float_scores, scores = np.empty(len(clips)), np.empty(len(clips))
        n_bytes = n_params = 0
        for machine_id in sorted(set(test_ids)):  # the same fit order as andon.classic
            detector = build_detector(model_cfg, seed).fit(x_train[train_ids == machine_id])
            if not isinstance(detector, GMM):
                raise TypeError(f"expected a GMM detector, got {type(detector).__name__}")
            qgmm = quant.quantize_gmm(detector.mixture)
            rows = test_ids == machine_id
            float_scores[rows] = detector.score(x_test[rows])
            scores[rows] = qgmm.score(x_test[rows])
            n_bytes += qgmm.nbytes()
            k, d = qgmm.means.shape
            n_params += k * (d + d * (d + 1) // 2 + 2)  # mean, triangle, log weight, log det
        check_float_matches_log(src, machine_type, seed, split, clips, float_scores, log)

        k = int(model_cfg["n_components"])
        b = budget.Budget(
            params=n_params,
            int8_bytes=n_bytes,
            macs_per_decision=budget.gmm_macs(k, n_mels),
            frames_per_decision=int(np.median(np.diff(feats.offsets))),  # one score per clip
            front_end_macs_per_frame=budget.front_end_macs(src.logmel.n_fft, n_mels),
            per_frame_macs=n_mels,  # the running mean of the log-mel frames
        )
        out.append((seed, scores, f"{b.notes()} ids={len(set(test_ids))}"))
    return out


RUNNERS = {
    "id_cnn_int8": run_cnn,
    "dense_autoencoder_int8": run_autoencoder,
    "gmm_int8": run_gmm,
}


def run(
    cfg: Config,
    seeds: list[int],
    machine_types: list[str],
    split: str,
    cache_dir: Path = CACHE_DIR,
    log: Path = RESULTS_LOG,
) -> None:
    if split == "eval" and cfg.raw.get("final") is not True:
        raise ValueError(f"{cfg.name} is not marked final: true; eval is for the chosen model")
    src = source_config(cfg)
    runner = RUNNERS[cfg.model]
    for machine_type in machine_types:
        todo = [s for s in seeds if not is_logged(cfg, machine_type, s, split, log=log)]
        for s in sorted(set(seeds) - set(todo)):
            print(f"[{machine_type} seed {s}] {split} already logged, skipping")
        if not todo:
            continue
        clips, _ = split_clips(src, machine_type, split)
        for seed, scores, notes in runner(cfg, src, machine_type, todo, split, cache_dir, log):
            log_scores(cfg, machine_type, seed, clips, scores, notes, log, split)


# --- summary ----------------------------------------------------------------------------


def parse_notes(notes: str) -> dict[str, str]:
    """The ``key=value`` pairs of a row's notes; per-ID ``id_xx=auc/pauc`` pairs included."""
    return dict(p.split("=", 1) for p in notes.split() if "=" in p)


def _type_means(config: str, metric: str, split: str, log: Path) -> dict[str, float]:
    """``{seed: mean over the six machine types}``; every type needs the same seeds."""
    rows = [r for r in read_results(log) if r["config"] == config and r["split"] == split]
    cells = _cells(rows, metric)
    seed_sets = {frozenset(cells[(config, t)]) for t in MACHINE_TYPES}
    if len(seed_sets) != 1 or not next(iter(seed_sets)):
        raise ValueError(f"{config} {split}: every machine type needs the same seeds logged")
    seeds = sorted(next(iter(seed_sets)), key=int)
    return {s: float(np.mean([cells[(config, t)][s] for t in MACHINE_TYPES])) for s in seeds}


def budget_table(configs: list[str], split: str = "dev_test", log: Path = RESULTS_LOG) -> str:
    """Size, compute and AUC before and after int8, one line per int8 config.

    Size and compute are the largest over the six machine types (the model must fit for
    any of them). AUC is the mean of the six types per seed, then mean +/- sample std over
    seeds; the change is the same, taken on the per-seed differences.
    """
    lines = [
        "| Model (int8 config) | Float model | Params | Float32 KB | Int8 KB | MACs per score "
        "| Score every (frames) | Est. ms per frame | Est. ms per score | Fits budget "
        "| AUC float (%) | AUC int8 (%) | AUC change | pAUC float (%) | pAUC int8 (%) |",
        "| --- | --- |" + " ---: |" * 13,
    ]
    for name in configs:
        cfg = load_config(Path("configs") / f"{name}.yaml")
        src = source_config(cfg)
        rows = [r for r in read_results(log) if r["config"] == name and r["split"] == split]
        if not rows:
            raise ValueError(f"no {split} rows for {name}")
        notes = [parse_notes(r["notes"]) for r in rows]

        def largest(key: str, notes: list[dict[str, str]] = notes) -> float:
            return max(float(n[key]) for n in notes)

        params, n_bytes = int(largest("params")), int(largest("int8_bytes"))
        ms_frame = largest("est_ms_per_frame")
        fits = n_bytes <= budget.BUDGET_BYTES and ms_frame < budget.BUDGET_MS_PER_FRAME
        frames = sorted({int(n["frames_per_decision"]) for n in notes})
        cols = [
            f"`{name}`",
            f"`{src.name}`",
            f"{params:,}",
            f"{4 * params / 1024:.1f}",
            f"{n_bytes / 1024:.1f}",
            f"{int(largest('macs_per_decision')):,}",
            "/".join(map(str, frames)),
            f"{ms_frame:.2f}",
            f"{largest('est_ms_per_decision'):.2f}",
            "yes" if fits else "**no**",
        ]
        for metric in ("auc", "pauc"):
            before = _type_means(src.name, metric, split, log)
            after = _type_means(name, metric, split, log)
            if set(before) != set(after):
                raise ValueError(f"{name} and {src.name} have different seeds on {split}")
            seeds = sorted(after, key=int)
            b_mean, b_sd = _mean_sd([before[s] for s in seeds])
            a_mean, a_sd = _mean_sd([after[s] for s in seeds])
            cols += [f"{b_mean:.2f} +/- {b_sd:.2f}", f"{a_mean:.2f} +/- {a_sd:.2f}"]
            if metric == "auc":
                d_mean, d_sd = _mean_sd([after[s] - before[s] for s in seeds])
                cols.append(f"{d_mean:+.2f} +/- {d_sd:.2f}")
        lines.append("| " + " | ".join(cols) + " |")
    lines += [
        "",
        f"Budget: at most {budget.BUDGET_BYTES // 1024} KB of int8 model data and under "
        f"{budget.BUDGET_MS_PER_FRAME:.0f} ms of compute per 32 ms audio frame. Times are "
        f"estimates: MACs at {budget.MACS_PER_CYCLE} per cycle on a "
        f"{budget.CLOCK_HZ / 1e6:.0f} MHz Cortex-M33, the per-score cost spread over the "
        "frames between scores, plus the log-mel front end. Float32 KB is 4 bytes per "
        "parameter. Size and compute: the largest over the six machine types.",
    ]
    return "\n".join(lines)


SPLIT_TITLES = {
    "dev_test": "Dev test: every model, used to choose",
    "eval": "Eval: the chosen model only, scored once",
}


def summary(configs: list[str], splits: list[str], log: Path = RESULTS_LOG) -> str:
    """A budget table and per-type AUC tables for each split; a config with no rows in a
    split (only the chosen model has eval rows) is left out of that split."""
    logged = {(r["config"], r["split"]) for r in read_results(log)}
    sections = []
    for split in splits:
        here = [c for c in configs if (c, split) in logged]
        if not here:
            raise ValueError(f"no {split} rows for any of {configs}")
        cfgs = [load_config(Path("configs") / f"{c}.yaml") for c in here]
        sources = [source_config(c).name for c in cfgs]
        paired = [name for pair in zip(sources, here, strict=True) for name in pair]
        sections.append(
            f"## {SPLIT_TITLES[split]}\n\n{budget_table(here, split, log)}\n\n"
            + comparison_tables(paired, [split], log)
        )
    return "\n\n".join(sections)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quantize a model to int8 and score it.")
    parser.add_argument("--config", nargs="+", required=True, help="one to run; several to sum")
    parser.add_argument("--seed", type=int, nargs="+", help="one or more seeds")
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["dev_test", "eval"],
        default=["dev_test"],
        help="one to run; with --summary, one or more",
    )
    parser.add_argument("--machines", nargs="*", help="machine types (default: all in the config)")
    parser.add_argument("--threads", type=int, help="CPU threads (default: PyTorch's choice)")
    parser.add_argument("--summary", action="store_true", help="print the tables, run nothing")
    parser.add_argument("--out", type=Path, help="with --summary: also write the tables here")
    args = parser.parse_args(argv)

    if args.summary:
        names = [Path(c).stem for c in args.config]
        command = "uv run python -m andon.quantize --summary --config " + " ".join(names)
        command += " --split " + " ".join(args.split)
        if args.out:
            command += f" --out {args.out.as_posix()}"
        text = f"<!-- Generated by `{command}`. -->\n\n{summary(names, args.split)}\n"
        print(text)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        return

    if not args.seed or len(args.config) != 1 or len(args.split) != 1:
        parser.error("a run needs --seed, exactly one --config and exactly one --split")
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config[0])
    print(f"config {cfg.name}, seeds {args.seed}, split {args.split}")
    run(cfg, args.seed, args.machines or cfg.machine_types, args.split[0])


if __name__ == "__main__":
    main()
