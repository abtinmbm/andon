"""Fit a classic detector on normal clips and score a test split, one row per type and seed.

    uv run python -m andon.classic --config configs/phase2_mahalanobis_type.yaml --seed 0 1 2
    uv run python -m andon.classic --config configs/phase2_gmm_id.yaml --seed 0 1 2 --split eval

The detectors see one vector per clip, the mean log-mel spectrum, computed from the same
feature cache as the autoencoder. Several seeds can run in one call so each machine
type's features are loaded once; every seed is still fitted from scratch and logged as
its own row. Rows already in the log are skipped.

With ``--split eval`` the detector is fitted on the additional training set (the normal
clips of the evaluation machine IDs) and scores the evaluation test clips, as the
official baseline does. A detector is refitted in well under a second, so unlike the
neural models it needs no checkpoint; the eval split is still run once per final model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from andon.data.dataset import (
    Clip,
    additional_train_clips,
    dev_test_clips,
    eval_test_clips,
    train_clips,
)
from andon.data.features import clip_means, load_or_extract
from andon.evaluate import log_scores
from andon.experiment import RESULTS_LOG, Config, is_logged, load_config, set_seed
from andon.models.classic import fit_and_score

CACHE_DIR = Path("data/cache")
# split -> (train clips, test clips, train cache suffix, test cache suffix). The cache names
# are the ones the neural models use, so features are extracted once for every model.
SPLITS = {
    "dev_test": (train_clips, dev_test_clips, "train", "test"),
    "eval": (additional_train_clips, eval_test_clips, "additional_train", "eval_test"),
}


def run(
    cfg: Config,
    seeds: list[int],
    machine_types: list[str],
    cache_dir: Path = CACHE_DIR,
    log: Path = RESULTS_LOG,
    split: str = "dev_test",
) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split}")
    train_of, test_of, train_suffix, test_suffix = SPLITS[split]
    for machine_type in machine_types:
        todo = [s for s in seeds if not is_logged(cfg, machine_type, s, split, log=log)]
        for s in sorted(set(seeds) - set(todo)):
            print(f"[{machine_type} seed {s}] {split} already logged, skipping")
        if not todo:
            continue
        train = train_of(cfg.data_root, machine_type)
        test = test_of(cfg.data_root, machine_type)
        x_train = clip_means(
            load_or_extract(train, cfg.logmel, cache_dir, f"{machine_type}_{train_suffix}")
        )
        x_test = clip_means(
            load_or_extract(test, cfg.logmel, cache_dir, f"{machine_type}_{test_suffix}")
        )
        for seed in todo:
            set_seed(seed)
            tic = time.perf_counter()
            scores = fit_and_score(cfg.raw["model"], seed, x_train, _ids(train), x_test, _ids(test))
            secs = time.perf_counter() - tic
            fit_per = cfg.raw["model"]["fit_per"]
            notes = f"fit_per={fit_per} fit_s={secs:.1f}"
            log_scores(cfg, machine_type, seed, test, scores, notes, log, split)


def _ids(clips: list[Clip]) -> list[str]:
    return [c.machine_id for c in clips]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit a classic detector and score a test split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, nargs="+", required=True, help="one or more seeds")
    parser.add_argument("--machines", nargs="*", help="machine types (default: all in the config)")
    parser.add_argument("--split", choices=sorted(SPLITS), default="dev_test")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config {cfg.name}, seeds {args.seed}, split {args.split}")
    run(cfg, args.seed, args.machines or cfg.machine_types, split=args.split)


if __name__ == "__main__":
    main()
