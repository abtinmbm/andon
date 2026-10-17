"""Train the baseline autoencoder on normal clips, one model per machine type, then score dev test.

    uv run python -m andon.train --config configs/phase1_baseline.yaml --seed 0

Each machine type is trained, checkpointed and logged to ``results/results_log.csv`` on its
own. Re-running the same command skips what is already logged and reuses finished
checkpoints, so an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from andon.data.dataset import train_clips
from andon.data.features import SplitFeatures, gather_windows, load_or_extract, window_starts
from andon.evaluate import evaluate_and_log
from andon.experiment import Config, is_logged, load_config, set_seed
from andon.models.autoencoder import DenseAutoencoder

CACHE_DIR = Path("data/cache")


def build_model(cfg: Config, input_dim: int) -> DenseAutoencoder:
    m = cfg.raw["model"]
    return DenseAutoencoder(
        input_dim=input_dim,
        hidden=int(m["hidden"]),
        n_hidden=int(m["n_hidden"]),
        bottleneck=int(m["bottleneck"]),
        bn_momentum=float(m["bn_momentum"]),
        bn_eps=float(m["bn_eps"]),
    )


def holdout_split(starts: np.ndarray, validation_split: float) -> tuple[np.ndarray, np.ndarray]:
    """Keras ``validation_split``: the last fraction of the (unshuffled) data is held out.

    The baseline relied on this, so the model never trained on the tail of the sorted
    file list (mostly the highest machine ID). Reproduced to match its training set.
    """
    split_at = int(len(starts) * (1.0 - validation_split))
    return starts[:split_at], starts[split_at:]


@torch.no_grad()
def mean_loss(
    model: nn.Module, frames: torch.Tensor, starts: torch.Tensor, n_context: int, batch: int
) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(starts), batch):
        x = gather_windows(frames, starts[i : i + batch], n_context)
        total += float(((x - model(x)) ** 2).mean()) * len(x)
    return total / max(len(starts), 1)


def train_autoencoder(
    feats: SplitFeatures,
    cfg: Config,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> tuple[DenseAutoencoder, list[dict[str, float]]]:
    """Fit on stacked windows of ``feats``; returns the model (in eval mode) and epoch losses."""
    t = cfg.raw["train"]
    n_context = cfg.n_context
    batch = int(t["batch_size"])
    starts_np = window_starts(feats.offsets, n_context)
    train_np, val_np = holdout_split(starts_np, float(t["validation_split"]))
    if len(train_np) == 0:
        raise ValueError("no training windows")

    frames = torch.from_numpy(feats.frames).to(device)
    train_starts = torch.from_numpy(train_np).to(device)
    val_starts = torch.from_numpy(val_np).to(device)

    model = build_model(cfg, input_dim=n_context * feats.frames.shape[1]).to(device)
    # Keras 2.1 Adam defaults: betas (0.9, 0.999), epsilon 1e-7 (PyTorch's default is 1e-8).
    # fused=True is the same update in one kernel; the per-tensor loop was 4x slower on CPU.
    opt = torch.optim.Adam(
        model.parameters(), lr=float(t["learning_rate"]), eps=float(t["adam_eps"]), fused=True
    )
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    epochs = int(t["epochs"])
    for epoch in range(1, epochs + 1):
        tic = time.perf_counter()
        model.train()
        order = (
            train_starts[torch.randperm(len(train_starts), device=device)]
            if t["shuffle"]
            else train_starts
        )
        running = 0.0
        for i in range(0, len(order), batch):
            x = gather_windows(frames, order[i : i + batch], n_context)
            loss = loss_fn(model(x), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(x)
        train_loss = running / len(order)
        val_loss = (
            mean_loss(model, frames, val_starts, n_context, batch) if len(val_np) else float("nan")
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            secs = time.perf_counter() - tic
            log(
                f"  epoch {epoch:3d}/{epochs}  loss {train_loss:.4f}  "
                f"val {val_loss:.4f}  ({secs:.1f} s)"
            )
    model.eval()
    return model, history


def save_checkpoint(
    path: Path, model: nn.Module, cfg: Config, machine_type: str, seed: int, history
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.raw,
            "machine_type": machine_type,
            "seed": seed,
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a killed run never leaves a checkpoint that looks finished


def load_checkpoint(path: Path, cfg: Config, device: torch.device) -> tuple[DenseAutoencoder, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    n_mels = int(cfg.raw["features"]["n_mels"])
    model = build_model(cfg, input_dim=cfg.n_context * n_mels).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def run(cfg: Config, seed: int, machine_types: list[str], device: torch.device) -> None:
    for machine_type in machine_types:
        if is_logged(cfg, machine_type, seed, "dev_test"):
            print(f"[{machine_type} seed {seed}] already logged, skipping")
            continue
        ckpt_path = cfg.checkpoint_path(machine_type, seed)
        if ckpt_path.exists():
            print(f"[{machine_type} seed {seed}] reusing {ckpt_path}")
            model, ckpt = load_checkpoint(ckpt_path, cfg, device)
            history = ckpt["history"]
        else:
            clips = train_clips(cfg.data_root, machine_type)
            print(f"[{machine_type} seed {seed}] features for {len(clips)} normal train clips")
            feats = load_or_extract(clips, cfg.logmel, CACHE_DIR, f"{machine_type}_train")
            # Seed per machine type, so each model is the same whatever order they run in.
            set_seed(seed)
            tic = time.perf_counter()
            model, history = train_autoencoder(feats, cfg, device)
            print(
                f"[{machine_type} seed {seed}] trained in "
                f"{(time.perf_counter() - tic) / 60:.1f} min"
            )
            save_checkpoint(ckpt_path, model, cfg, machine_type, seed, history)
        evaluate_and_log(
            cfg, model, machine_type, seed, device, final_val_loss=history[-1]["val_loss"]
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the baseline autoencoder and score dev test."
    )
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
