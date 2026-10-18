"""Fit a classic detector on normal clips and score dev test, one row per machine type and seed.

    uv run python -m andon.classic --config configs/phase2_mahalanobis_type.yaml --seed 0 1 2

The detectors see one vector per clip, the mean log-mel spectrum, computed from the same
feature cache as the autoencoder. Several seeds can run in one call so each machine
type's features are loaded once; every seed is still fitted from scratch and logged as
its own row. Rows already in the log are skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from andon.data.dataset import Clip, dev_test_clips, train_clips
from andon.data.features import clip_means, load_or_extract
from andon.evaluate import log_scores
from andon.experiment import RESULTS_LOG, Config, is_logged, load_config, set_seed
from andon.models.classic import fit_and_score

CACHE_DIR = Path("data/cache")


def run(
    cfg: Config,
    seeds: list[int],
    machine_types: list[str],
    cache_dir: Path = CACHE_DIR,
    log: Path = RESULTS_LOG,
) -> None:
    for machine_type in machine_types:
        todo = [s for s in seeds if not is_logged(cfg, machine_type, s, "dev_test", log=log)]
        for s in sorted(set(seeds) - set(todo)):
            print(f"[{machine_type} seed {s}] already logged, skipping")
        if not todo:
            continue
        train = train_clips(cfg.data_root, machine_type)
        test = dev_test_clips(cfg.data_root, machine_type)
        # Same cache names as the autoencoder, so the frames extracted in Phase 1 are reused.
        x_train = clip_means(load_or_extract(train, cfg.logmel, cache_dir, f"{machine_type}_train"))
        x_test = clip_means(load_or_extract(test, cfg.logmel, cache_dir, f"{machine_type}_test"))
        for seed in todo:
            set_seed(seed)
            tic = time.perf_counter()
            scores = fit_and_score(cfg.raw["model"], seed, x_train, _ids(train), x_test, _ids(test))
            secs = time.perf_counter() - tic
            fit_per = cfg.raw["model"]["fit_per"]
            log_scores(
                cfg, machine_type, seed, test, scores, f"fit_per={fit_per} fit_s={secs:.1f}", log
            )


def _ids(clips: list[Clip]) -> list[str]:
    return [c.machine_id for c in clips]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit a classic detector and score dev test.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, nargs="+", required=True, help="one or more seeds")
    parser.add_argument("--machines", nargs="*", help="machine types (default: all in the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(f"config {cfg.name}, seeds {args.seed}")
    run(cfg, args.seed, args.machines or cfg.machine_types)


if __name__ == "__main__":
    main()
