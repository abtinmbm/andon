"""Score dev test clips with a trained autoencoder, and summarise logged runs.

Scoring follows the baseline's ``01_test.py``: a clip's anomaly score is the mean,
over its stacked-frame windows, of the per-window mean squared reconstruction error.
AUC and pAUC are computed per machine ID, then averaged over IDs (unweighted) to
give the machine type's figure.

    uv run python -m andon.evaluate --config configs/phase1_baseline.yaml --summary
    uv run python -m andon.evaluate --compare phase1_baseline phase2_gmm_id --out results/x.md
    uv run python -m andon.evaluate --compare A B --split dev_test eval
    uv run python -m andon.evaluate --beats phase3_idcnn_mixup phase3_baseline_eval --split eval
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from andon import metrics
from andon.data.dataset import MACHINE_TYPES, Clip, dev_test_clips
from andon.data.features import SplitFeatures, gather_windows, load_or_extract, window_starts
from andon.experiment import RESULTS_LOG, Config, append_result, load_config, read_results

CACHE_DIR = Path("data/cache")
_CHUNK = 16_384  # windows per forward pass when scoring

# Published DCASE 2020 Task 2 baseline, development set, as fractions (AUC, pAUC).
PUBLISHED_BASELINE = {
    "ToyCar": (0.7877, 0.6758),
    "ToyConveyor": (0.7253, 0.6043),
    "fan": (0.6583, 0.5245),
    "pump": (0.7289, 0.5999),
    "slider": (0.8476, 0.6653),
    "valve": (0.6628, 0.5098),
}
REPRODUCTION_TOLERANCE = 0.02  # "done when": mean AUC within 2 points of the published value


@torch.no_grad()
def clip_scores(
    model: nn.Module, feats: SplitFeatures, n_context: int, device: torch.device
) -> np.ndarray:
    """One anomaly score per clip: mean window reconstruction error."""
    model.eval()
    starts = window_starts(feats.offsets, n_context)
    frames = torch.from_numpy(feats.frames).to(device)
    errors = np.empty(len(starts), dtype=np.float64)
    for i in range(0, len(starts), _CHUNK):
        s = torch.from_numpy(starts[i : i + _CHUNK]).to(device)
        x = gather_windows(frames, s, n_context)
        errors[i : i + len(s)] = ((x - model(x)) ** 2).mean(dim=1).cpu().numpy()
    windows_per_clip = np.diff(feats.offsets) - n_context + 1
    if (windows_per_clip < 1).any():
        raise ValueError(f"a clip is shorter than {n_context} frames")
    first = np.r_[0, np.cumsum(windows_per_clip)[:-1]]
    return np.add.reduceat(errors, first) / windows_per_clip


def per_id_metrics(clips: list[Clip], scores: np.ndarray) -> dict[str, tuple[float, float]]:
    """``{machine_id: (auc, pauc)}`` with the DCASE pAUC (max FPR 0.1, McClish-standardized)."""
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, clip in enumerate(clips):
        by_id[clip.machine_id].append(i)
    result = {}
    for machine_id, idx in sorted(by_id.items()):
        y = np.array([clips[i].is_anomaly for i in idx])
        s = scores[idx]
        result[machine_id] = (metrics.auc(y, s), metrics.pauc(y, s))
    return result


def evaluate_and_log(
    cfg: Config,
    model: nn.Module,
    machine_type: str,
    seed: int,
    device: torch.device,
    final_val_loss: float | None = None,
    log: Path = RESULTS_LOG,
) -> dict[str, str]:
    """Score one machine type's dev test clips and append its row to the results log."""
    clips = dev_test_clips(cfg.data_root, machine_type)
    feats = load_or_extract(clips, cfg.logmel, CACHE_DIR, f"{machine_type}_test")
    scores = clip_scores(model, feats, cfg.n_context, device)
    extra = f"val_loss={final_val_loss:.4f}" if final_val_loss is not None else ""
    return log_scores(cfg, machine_type, seed, clips, scores, extra, log=log)


def log_scores(
    cfg: Config,
    machine_type: str,
    seed: int,
    clips: list[Clip],
    scores: np.ndarray,
    extra_notes: str = "",
    log: Path = RESULTS_LOG,
    split: str = "dev_test",
) -> dict[str, str]:
    """Turn test clip scores into the machine type's AUC and pAUC and log one ``split`` row."""
    per_id = per_id_metrics(clips, scores)
    auc = float(np.mean([a for a, _ in per_id.values()]))
    pauc = float(np.mean([p for _, p in per_id.values()]))
    notes = " ".join(f"{mid}={a:.4f}/{p:.4f}" for mid, (a, p) in per_id.items())
    if extra_notes:
        notes += f" {extra_notes}"
    row = append_result(cfg, machine_type, split, seed, auc, pauc, notes, log=log)
    print(f"[{machine_type} seed {seed}] {split} AUC {auc:.4f}  pAUC {pauc:.4f}  ({notes})")
    return row


def summary_table(cfg: Config, log: Path = RESULTS_LOG) -> str:
    """Mean +/- sample std over seeds per machine type, next to the published baseline."""
    rows = [r for r in read_results(log) if r["config"] == cfg.name and r["split"] == "dev_test"]
    header = (
        f"{'machine_type':<12} {'seeds':>5} {'AUC mean+/-std':>16} {'pub AUC':>8} {'diff':>6} "
        f"{'pAUC mean+/-std':>16} {'pub pAUC':>8} {'diff':>6}  within 2 pts"
    )
    lines = [f"config {cfg.name}, split dev_test, values in %", header, "-" * len(header)]
    for machine_type in cfg.machine_types:
        mine = [r for r in rows if r["machine_type"] == machine_type]
        if not mine:
            lines.append(f"{machine_type:<12} {0:>5}  (no runs logged)")
            continue
        aucs = np.array([float(r["auc"]) for r in mine]) * 100
        paucs = np.array([float(r["pauc"]) for r in mine]) * 100
        pub_auc, pub_pauc = (100 * v for v in PUBLISHED_BASELINE[machine_type])
        sd = (lambda a: a.std(ddof=1)) if len(mine) > 1 else (lambda a: float("nan"))
        d_auc = aucs.mean() - pub_auc
        ok = "yes" if abs(d_auc) <= 100 * REPRODUCTION_TOLERANCE else "NO"
        lines.append(
            f"{machine_type:<12} {len(mine):>5} {aucs.mean():>8.2f}+/-{sd(aucs):<5.2f} "
            f"{pub_auc:>8.2f} {d_auc:>+6.2f} {paucs.mean():>8.2f}+/-{sd(paucs):<5.2f} "
            f"{pub_pauc:>8.2f} {paucs.mean() - pub_pauc:>+6.2f}  {ok}"
        )
    seeds = sorted({r["seed"] for r in rows}, key=int)
    lines.append(f"seeds: {', '.join(seeds) or 'none'}; std is the sample std (ddof=1) over seeds")
    return "\n".join(lines)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    a = np.array(values) * 100
    sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
    return float(a.mean()), sd


SPLIT_NAMES = {"dev_test": "Dev test", "eval": "Eval (evaluation set test clips)"}


def _cells(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str], dict[str, float]]:
    """``{(config, machine_type): {seed: value}}``; a repeated row is an error."""
    by_cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        seeds = by_cell[(r["config"], r["machine_type"])]
        if r["seed"] in seeds:  # runs skip logged rows, so a repeat means the log was edited
            raise ValueError(
                f"duplicate row: {r['config']} {r['machine_type']} {r['split']} seed {r['seed']}"
            )
        seeds[r["seed"]] = float(r[metric])
    return by_cell


def comparison_table(
    configs: list[str], metric: str = "auc", log: Path = RESULTS_LOG, split: str = "dev_test"
) -> str:
    """Markdown table of one metric on one split, mean +/- std over seeds in %, side by side.

    The "mean" row averages the six machine types within each seed first, then takes
    the mean and std of those per-seed averages, so it needs every type for every seed.
    The best mean in each row is in bold.
    """
    rows = [r for r in read_results(log) if r["split"] == split and r["config"] in configs]
    by_cell = _cells(rows, metric)

    def cell(stats: tuple[float, float] | None, best: float | None) -> str:
        if stats is None:
            return "n/a"
        text = f"{stats[0]:.2f} +/- {stats[1]:.2f}"
        return f"**{text}**" if best is not None and round(stats[0], 2) == round(best, 2) else text

    def table_row(label: str, stats: list[tuple[float, float] | None]) -> str:
        means = [s[0] for s in stats if s is not None]
        best = max(means) if means else None
        return f"| {label} | " + " | ".join(cell(s, best) for s in stats) + " |"

    lines = [
        f"| {metric.upper()} (%) | " + " | ".join(configs) + " |",
        "| --- |" + " ---: |" * len(configs),
    ]
    for machine_type in MACHINE_TYPES:
        stats = [
            _mean_sd(list(by_cell[(c, machine_type)].values()))
            if by_cell[(c, machine_type)]
            else None
            for c in configs
        ]
        lines.append(table_row(machine_type, stats))

    averages: list[tuple[float, float] | None] = []
    for c in configs:
        seed_sets = [set(by_cell[(c, t)]) for t in MACHINE_TYPES]
        seeds = set.intersection(*seed_sets)
        if not seeds or any(s != seeds for s in seed_sets):
            averages.append(None)
            continue
        per_seed = [np.mean([by_cell[(c, t)][s] for t in MACHINE_TYPES]) for s in sorted(seeds)]
        averages.append(_mean_sd(per_seed))
    lines.append(table_row("mean", averages))

    seeds_note = "; ".join(
        f"{c}: {len({r['seed'] for r in rows if r['config'] == c})}" for c in configs
    )
    lines.append("")
    lines.append(
        f"{SPLIT_NAMES[split]}, from results_log.csv. Mean +/- sample std (ddof=1) over seeds "
        f"(seeds per config: {seeds_note})."
    )
    return "\n".join(lines)


def comparison_tables(configs: list[str], splits: list[str], log: Path = RESULTS_LOG) -> str:
    """AUC and pAUC tables for each split.

    With several splits, a config with no rows in a split is left out of that split's
    table (the eval-only autoencoder has no dev rows). A config with no rows in any of
    the splits is an error, so a mistyped name can't vanish silently.
    """
    rows = read_results(log)
    present = {split: {r["config"] for r in rows if r["split"] == split} for split in splits}
    missing = [c for c in configs if not any(c in present[s] for s in splits)]
    if missing:
        raise ValueError(f"no {' or '.join(splits)} rows for: {missing}")
    tables = []
    for split in splits:
        here = [c for c in configs if c in present[split]] if len(splits) > 1 else configs
        if here:
            tables += [comparison_table(here, m, log, split) for m in ("auc", "pauc")]
    return "\n\n".join(tables)


def head_to_head(
    candidate: str, baseline: str, split: str, metric: str = "auc", log: Path = RESULTS_LOG
) -> tuple[str, int]:
    """Per machine type, candidate vs baseline mean +/- std over seeds, and the number of
    types where the candidate's mean is higher.

    Both configs must have the same seeds (at least two), one row each, for every machine
    type: the check fails rather than compare partial runs.
    """
    rows = [
        r for r in read_results(log) if r["split"] == split and r["config"] in (candidate, baseline)
    ]
    by_cell = _cells(rows, metric)
    seeds = set(by_cell[(candidate, MACHINE_TYPES[0])])
    for c in (candidate, baseline):
        for t in MACHINE_TYPES:
            if len(seeds) < 2 or set(by_cell[(c, t)]) != seeds:
                got = sorted(by_cell[(c, t)], key=int)
                raise ValueError(f"{c} {t} {split}: seeds {got}, expected {sorted(seeds, key=int)}")
    seed_list = ", ".join(sorted(seeds, key=int))
    lines = [
        f"{SPLIT_NAMES[split]}, {metric.upper()} %, mean +/- std over seeds {seed_list}",
        f"{'machine_type':<12} {candidate:>24} {baseline:>24} {'diff':>7}",
    ]
    wins = 0
    for t in MACHINE_TYPES:
        cm, cs = _mean_sd(list(by_cell[(candidate, t)].values()))
        bm, bs = _mean_sd(list(by_cell[(baseline, t)].values()))
        wins += cm > bm
        c_text, b_text = f"{cm:.2f} +/- {cs:.2f}", f"{bm:.2f} +/- {bs:.2f}"
        lines.append(f"{t:<12} {c_text:>24} {b_text:>24} {cm - bm:>+7.2f}")
    lines.append(f"{candidate} beats {baseline} on {wins}/{len(MACHINE_TYPES)} machine types")
    return "\n".join(lines), wins


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise and compare logged results.")
    parser.add_argument("--config", type=Path, help="config for --summary")
    parser.add_argument(
        "--summary", action="store_true", help="print mean +/- std per machine type"
    )
    parser.add_argument(
        "--compare", nargs="+", metavar="NAME", help="config names to compare side by side"
    )
    parser.add_argument(
        "--beats", nargs=2, metavar=("CANDIDATE", "BASELINE"), help="per-type head to head"
    )
    parser.add_argument(
        "--split", nargs="+", choices=sorted(SPLIT_NAMES), default=["dev_test"], help="test splits"
    )
    parser.add_argument("--out", type=Path, help="also write the --compare tables to this file")
    args = parser.parse_args(argv)
    if args.summary:
        if args.config is None:
            parser.error("--summary needs --config")
        print(summary_table(load_config(args.config)))
    elif args.compare:
        text = comparison_tables(args.compare, args.split)
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            command = "uv run python -m andon.evaluate --compare " + " ".join(args.compare)
            if args.split != ["dev_test"]:
                command += " --split " + " ".join(args.split)
            header = f"<!-- Generated by `{command} --out {args.out.as_posix()}`. -->\n\n"
            args.out.write_text(header + text + "\n", encoding="utf-8")
    elif args.beats:
        if len(args.split) != 1:
            parser.error("--beats takes one --split")
        print(head_to_head(*args.beats, args.split[0])[0])
    else:
        parser.error("nothing to do: pass --summary, --compare or --beats")


if __name__ == "__main__":
    main()
