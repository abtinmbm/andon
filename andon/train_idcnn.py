"""Train the machine-ID CNN on normal clips, one model per machine type, then score dev test.

    uv run python -m andon.train_idcnn --config configs/phase3_idcnn.yaml --seed 0

The classifier learns every machine ID of a type: the dev IDs (dev train) and the
evaluation IDs (additional training set), all normal clips. Each epoch draws one random
patch per training clip. A test clip is scored by sliding a patch window over it and
averaging ``-log p(its own ID)``. Rows already in the log are skipped and finished
checkpoints are reused, so an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from andon.data.dataset import Clip, all_train_clips, dev_test_clips, train_clips
from andon.data.features import SplitFeatures, concat, load_or_extract
from andon.evaluate import log_scores
from andon.experiment import RESULTS_LOG, Config, is_logged, load_config, set_seed
from andon.models.idcnn import IdCNN, id_scores, mask_spectrogram, mixup

CACHE_DIR = Path("data/cache")
_SCORE_BATCH = 512  # patches per forward pass when scoring


def build_model(cfg: Config, n_classes: int) -> IdCNN:
    m = cfg.raw["model"]
    return IdCNN(n_classes, int(cfg.raw["features"]["n_mels"]), tuple(m["channels"]))


def training_data(
    cfg: Config, machine_type: str, cache_dir: Path = CACHE_DIR
) -> tuple[SplitFeatures, list[Clip], list[str]]:
    """Frames and clips of every normal training clip of one type, and the sorted ID list.

    The dev part reuses the cache Phase 1 built (same name and key).
    """
    clips = all_train_clips(cfg.data_root, machine_type)  # dev first; refuses a shared ID
    n_dev = len(train_clips(cfg.data_root, machine_type))
    dev, extra = clips[:n_dev], clips[n_dev:]
    feats = concat(
        [
            load_or_extract(dev, cfg.logmel, cache_dir, f"{machine_type}_train"),
            load_or_extract(extra, cfg.logmel, cache_dir, f"{machine_type}_additional_train"),
        ]
    )
    return feats, clips, sorted({c.machine_id for c in clips})


def random_patches(
    frames: torch.Tensor, offsets: np.ndarray, clip_idx: np.ndarray, width: int, rng
) -> torch.Tensor:
    """One random ``width``-frame patch from each clip in ``clip_idx``: ``(n, width, n_mels)``."""
    lengths = offsets[clip_idx + 1] - offsets[clip_idx]
    if (lengths < width).any():
        raise ValueError(f"a clip is shorter than {width} frames")
    starts = offsets[clip_idx] + rng.integers(0, lengths - width + 1)
    idx = torch.from_numpy(starts)[:, None] + torch.arange(width)
    return frames[idx.to(frames.device)]


def patch_starts(n_frames: int, width: int, hop: int) -> np.ndarray:
    """Window starts covering a clip: every ``hop`` frames, plus one flush with the end."""
    if n_frames < width:
        raise ValueError(f"a clip of {n_frames} frames is shorter than {width}")
    starts = np.arange(0, n_frames - width + 1, hop)
    if starts[-1] != n_frames - width:
        starts = np.r_[starts, n_frames - width]
    return starts


def train_idcnn(
    feats: SplitFeatures,
    labels: np.ndarray,
    n_classes: int,
    cfg: Config,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> tuple[IdCNN, list[dict[str, float]]]:
    """Fit on random patches of the training clips; returns the model (eval mode) and history.

    Call ``set_seed`` first: patch positions, shuffling, mixup and masks all draw from the
    seeded generators.
    """
    t = cfg.raw["train"]
    width = int(cfg.raw["model"]["patch_frames"])
    batch = int(t["batch_size"])
    epochs = int(t["epochs"])
    alpha = float(t["mixup_alpha"])  # 0 = off
    masks = t["spec_mask"] or {}  # null = off

    frames = torch.from_numpy(feats.frames).to(device)
    y_all = torch.from_numpy(labels).long().to(device)
    model = build_model(cfg, n_classes).to(device)
    model.set_normalization(frames)  # training frames only; saved with the weights
    opt = torch.optim.Adam(model.parameters(), lr=float(t["learning_rate"]), fused=True)
    # Cosine decay to 0, so the last epoch (the one we keep) is not a noisy mid-training step.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Patch positions and shuffling come from NumPy, seeded from the run seed via torch.
    rng = np.random.default_rng(torch.initial_seed() % 2**32)

    history: list[dict[str, float]] = []
    n_clips = feats.n_clips
    for epoch in range(1, epochs + 1):
        tic = time.perf_counter()
        model.train()
        order = rng.permutation(n_clips)
        loss_sum, correct = 0.0, 0
        for i in range(0, n_clips, batch):
            idx = order[i : i + batch]
            x = model.normalize(random_patches(frames, feats.offsets, idx, width, rng))
            y = y_all[torch.from_numpy(idx).to(device)]
            if masks:
                x = mask_spectrogram(
                    x,
                    int(masks["freq_masks"]),
                    int(masks["freq_width"]),
                    int(masks["time_masks"]),
                    int(masks["time_width"]),
                )
            x, y_a, y_b, lam = mixup(x, y, alpha)
            logits = model.classify(x)
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)
            correct += int((logits.argmax(1) == y).sum())  # against the unmixed label
        sched.step()
        history.append(
            {"epoch": epoch, "train_loss": loss_sum / n_clips, "train_acc": correct / n_clips}
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            log(
                f"  epoch {epoch:3d}/{epochs}  loss {history[-1]['train_loss']:.4f}  "
                f"acc {history[-1]['train_acc']:.3f}  ({time.perf_counter() - tic:.1f} s)"
            )
    model.eval()
    return model, history


@torch.no_grad()
def clip_scores(
    model: IdCNN,
    feats: SplitFeatures,
    true_ids: np.ndarray,
    width: int,
    hop: int,
    device: torch.device,
) -> np.ndarray:
    """One anomaly score per clip: mean over its patches of ``-log p(true ID)``."""
    model.eval()
    frames = torch.from_numpy(feats.frames).to(device)
    starts, owner = [], []
    for i in range(feats.n_clips):
        s = patch_starts(int(feats.offsets[i + 1] - feats.offsets[i]), width, hop)
        starts.append(feats.offsets[i] + s)
        owner.append(np.full(len(s), i))
    starts_all, owner_all = np.concatenate(starts), np.concatenate(owner)
    patch_scores = np.empty(len(starts_all), dtype=np.float64)
    for i in range(0, len(starts_all), _SCORE_BATCH):
        s = torch.from_numpy(starts_all[i : i + _SCORE_BATCH])
        x = frames[(s[:, None] + torch.arange(width)).to(device)]
        ids = torch.from_numpy(true_ids[owner_all[i : i + _SCORE_BATCH]]).long().to(device)
        patch_scores[i : i + len(s)] = id_scores(model(x), ids).cpu().numpy()
    counts = np.bincount(owner_all, minlength=feats.n_clips)
    return np.bincount(owner_all, weights=patch_scores, minlength=feats.n_clips) / counts


def score_clips(
    cfg: Config,
    model: IdCNN,
    classes: list[str],
    clips: list[Clip],
    feats: SplitFeatures,
    device: torch.device,
) -> np.ndarray:
    unknown = {c.machine_id for c in clips} - set(classes)
    if unknown:
        raise ValueError(f"test IDs the classifier never saw: {sorted(unknown)}")
    index = {mid: k for k, mid in enumerate(classes)}
    true_ids = np.array([index[c.machine_id] for c in clips])
    m = cfg.raw["model"]
    return clip_scores(model, feats, true_ids, int(m["patch_frames"]), int(m["score_hop"]), device)


def save_checkpoint(
    path: Path,
    model: IdCNN,
    cfg: Config,
    machine_type: str,
    seed: int,
    classes: list[str],
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.raw,
            "machine_type": machine_type,
            "seed": seed,
            "classes": classes,
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a killed run never leaves a checkpoint that looks finished


def load_checkpoint(path: Path, cfg: Config, device: torch.device) -> tuple[IdCNN, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(cfg, len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def train_and_save(
    cfg: Config, machine_type: str, seed: int, device: torch.device, cache_dir: Path = CACHE_DIR
) -> tuple[IdCNN, list[str], list[dict[str, float]]]:
    """Train one machine type's classifier and save its checkpoint; logs nothing."""
    feats, clips, classes = training_data(cfg, machine_type, cache_dir)
    labels = np.array([classes.index(c.machine_id) for c in clips])
    print(f"[{machine_type} seed {seed}] {len(clips)} normal train clips, IDs {classes}")
    set_seed(seed)  # per machine type, so each model is the same in any run order
    tic = time.perf_counter()
    model, history = train_idcnn(feats, labels, len(classes), cfg, device)
    mins = (time.perf_counter() - tic) / 60
    print(f"[{machine_type} seed {seed}] trained in {mins:.1f} min")
    path = cfg.checkpoint_path(machine_type, seed)
    save_checkpoint(path, model, cfg, machine_type, seed, classes, history)
    return model, classes, history


def run(
    cfg: Config,
    seed: int,
    machine_types: list[str],
    device: torch.device,
    cache_dir: Path = CACHE_DIR,
    log: Path = RESULTS_LOG,
) -> None:
    for machine_type in machine_types:
        if is_logged(cfg, machine_type, seed, "dev_test", log=log):
            print(f"[{machine_type} seed {seed}] already logged, skipping")
            continue
        ckpt_path = cfg.checkpoint_path(machine_type, seed)
        if ckpt_path.exists():
            print(f"[{machine_type} seed {seed}] reusing {ckpt_path}")
            model, ckpt = load_checkpoint(ckpt_path, cfg, device)
            classes, history = ckpt["classes"], ckpt["history"]
        else:
            model, classes, history = train_and_save(cfg, machine_type, seed, device, cache_dir)
        test = dev_test_clips(cfg.data_root, machine_type)
        feats_test = load_or_extract(test, cfg.logmel, cache_dir, f"{machine_type}_test")
        scores = score_clips(cfg, model, classes, test, feats_test, device)
        final = history[-1]
        notes = f"train_loss={final['train_loss']:.4f} train_acc={final['train_acc']:.3f}"
        log_scores(cfg, machine_type, seed, test, scores, notes, log=log)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the machine-ID CNN and score dev test.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--machines", nargs="*", help="machine types (default: all in the config)")
    parser.add_argument(
        "--threads",
        type=int,
        help="CPU threads (default: PyTorch's choice); lower it to run seeds in parallel",
    )
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config {cfg.name}, seed {args.seed}, device {device}")
    run(cfg, args.seed, args.machines or cfg.machine_types, device)


if __name__ == "__main__":
    main()
