"""Score saved checkpoints on a test split without retraining; one row per machine type and seed.

    uv run python -m andon.score --config configs/phase3_idcnn.yaml --seed 0 1 2 --split eval

Experiment rule: the evaluation set is scored once per final model, after every choice
has been made on dev test. Rows already in the log are skipped, so running the command
again never produces a second eval result for the same model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from andon import evaluate, train, train_idcnn
from andon.data.dataset import Clip, dev_test_clips, eval_test_clips
from andon.data.features import load_or_extract
from andon.evaluate import log_scores
from andon.experiment import RESULTS_LOG, Config, is_logged, load_config

CACHE_DIR = Path("data/cache")
SPLITS = {"dev_test": (dev_test_clips, "test"), "eval": (eval_test_clips, "eval_test")}


def split_clips(cfg: Config, machine_type: str, split: str) -> tuple[list[Clip], str]:
    """The clips of ``split`` and the feature-cache name they are stored under."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split}")
    clips_of, cache_suffix = SPLITS[split]
    return clips_of(cfg.data_root, machine_type), f"{machine_type}_{cache_suffix}"


def checkpoint_scores(
    cfg: Config,
    ckpt_path: Path,
    clips: list[Clip],
    feats,
    device: torch.device,
) -> np.ndarray:
    if cfg.model == "id_cnn":
        model, ckpt = train_idcnn.load_checkpoint(ckpt_path, cfg, device)
        return train_idcnn.score_clips(cfg, model, ckpt["classes"], clips, feats, device)
    if cfg.model == "dense_autoencoder":
        model, _ = train.load_checkpoint(ckpt_path, cfg, device)
        return evaluate.clip_scores(model, feats, cfg.n_context, device)
    raise ValueError(f"no checkpoint scorer for model type {cfg.model}")


def run(
    cfg: Config,
    seeds: list[int],
    machine_types: list[str],
    split: str,
    device: torch.device,
    cache_dir: Path = CACHE_DIR,
    log: Path = RESULTS_LOG,
) -> None:
    todo = {
        t: [s for s in seeds if not is_logged(cfg, t, s, split, log=log)] for t in machine_types
    }
    # Every checkpoint must exist before anything is scored, so a missing one can't leave
    # the log with eval rows for some types or seeds of a model and not others.
    missing = [cfg.checkpoint_path(t, s) for t, ss in todo.items() for s in ss]
    missing = [p for p in missing if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {[str(p) for p in missing]}")

    for machine_type in machine_types:
        for s in sorted(set(seeds) - set(todo[machine_type])):
            print(f"[{machine_type} seed {s}] {split} already logged, skipping")
        if not todo[machine_type]:
            continue
        clips, cache_name = split_clips(cfg, machine_type, split)
        feats = load_or_extract(clips, cfg.logmel, cache_dir, cache_name)
        for seed in todo[machine_type]:
            path = cfg.checkpoint_path(machine_type, seed)
            scores = checkpoint_scores(cfg, path, clips, feats, device)
            log_scores(cfg, machine_type, seed, clips, scores, "scored_from=checkpoint", log, split)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score saved checkpoints on a test split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, nargs="+", required=True, help="one or more seeds")
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--machines", nargs="*", help="machine types (default: all in the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config {cfg.name}, seeds {args.seed}, split {args.split}, device {device}")
    run(cfg, args.seed, args.machines or cfg.machine_types, args.split, device)


if __name__ == "__main__":
    main()
