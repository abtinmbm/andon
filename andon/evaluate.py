"""Score dev test clips with a trained autoencoder, and summarise logged runs.

Scoring follows the baseline's ``01_test.py``: a clip's anomaly score is the mean,
over its stacked-frame windows, of the per-window mean squared reconstruction error.
AUC and pAUC are computed per machine ID, then averaged over IDs (unweighted) to
give the machine type's figure.

    uv run python -m andon.evaluate --config configs/phase1_baseline.yaml --summary
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from andon import metrics
from andon.data.dataset import Clip, dev_test_clips
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
    per_id = per_id_metrics(clips, clip_scores(model, feats, cfg.n_context, device))
    auc = float(np.mean([a for a, _ in per_id.values()]))
    pauc = float(np.mean([p for _, p in per_id.values()]))
    notes = " ".join(f"{mid}={a:.4f}/{p:.4f}" for mid, (a, p) in per_id.items())
    if final_val_loss is not None:
        notes += f" val_loss={final_val_loss:.4f}"
    row = append_result(cfg, machine_type, "dev_test", seed, auc, pauc, notes, log=log)
    print(f"[{machine_type} seed {seed}] dev test AUC {auc:.4f}  pAUC {pauc:.4f}  ({notes})")
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise dev test results for a config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--summary", action="store_true", help="print mean +/- std per machine type"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if not args.summary:
        parser.error("nothing to do: training scores dev test itself; pass --summary")
    print(summary_table(cfg))


if __name__ == "__main__":
    main()
