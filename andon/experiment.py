"""What every run shares: the YAML config, seeding, and the results log."""

from __future__ import annotations

import csv
import os
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from andon.features.logmel import LogMelConfig

RESULTS_LOG = Path("results/results_log.csv")
UNTRACKED_CODE_DIRS = ("andon", "configs")  # untracked files here also make a run "-dirty"
RESULTS_COLUMNS = (
    "timestamp",
    "git_commit",
    "phase",
    "config",
    "model",
    "machine_type",
    "split",
    "seed",
    "auc",
    "pauc",
    "notes",
)


@dataclass(frozen=True)
class Config:
    name: str  # the YAML file's stem; checkpoints and log rows are keyed by it
    raw: dict[str, Any]

    @property
    def phase(self) -> int:
        return int(self.raw["phase"])

    @property
    def model(self) -> str:
        return str(self.raw["model"]["type"])

    @property
    def machine_types(self) -> list[str]:
        return list(self.raw["data"]["machine_types"])

    @property
    def data_root(self) -> Path:
        return Path(self.raw["data"]["root"])

    @property
    def n_context(self) -> int:
        return int(self.raw["features"]["frames"])

    @property
    def logmel(self) -> LogMelConfig:
        f = self.raw["features"]
        return LogMelConfig(
            sample_rate=int(f["sample_rate"]),
            n_fft=int(f["n_fft"]),
            hop_length=int(f["hop_length"]),
            n_mels=int(f["n_mels"]),
            power=float(f["power"]),
        )

    def checkpoint_path(self, machine_type: str, seed: int) -> Path:
        return Path("checkpoints") / f"{self.name}_{machine_type}_seed{seed}.pt"


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(name=path.stem, raw=raw)


def set_seed(seed: int) -> None:
    """Seed every random source a run touches. Call once at the start of each run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str:
    """Short hash of HEAD, with ``-dirty`` if tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            # The log is tracked and grows with every row, so it must not count as a code change.
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                ".",
                f":(exclude){RESULTS_LOG.as_posix()}",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # A new module or config that isn't committed yet is code the hash doesn't contain.
        dirty += subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *UNTRACKED_CODE_DIRS],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def read_results(log: Path = RESULTS_LOG) -> list[dict[str, str]]:
    if not log.exists():
        return []
    with log.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_logged(
    cfg: Config, machine_type: str, seed: int, split: str, log: Path = RESULTS_LOG
) -> bool:
    return any(
        r["config"] == cfg.name
        and r["machine_type"] == machine_type
        and r["seed"] == str(seed)
        and r["split"] == split
        for r in read_results(log)
    )


def append_result(
    cfg: Config,
    machine_type: str,
    split: str,
    seed: int,
    auc: float,
    pauc: float,
    notes: str = "",
    log: Path = RESULTS_LOG,
) -> dict[str, str]:
    """Append one row; AUC and pAUC are stored as fractions with 4 decimals."""
    if split not in ("dev_test", "eval"):
        raise ValueError(f"split must be dev_test or eval, got {split}")
    row = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "phase": str(cfg.phase),
        "config": cfg.name,
        "model": cfg.model,
        "machine_type": machine_type,
        "split": split,
        "seed": str(seed),
        "auc": f"{auc:.4f}",
        "pauc": f"{pauc:.4f}",
        "notes": notes,
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(log.with_suffix(".lock")):
        new_file = not log.exists() or log.stat().st_size == 0
        with log.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    return row


@contextmanager
def _file_lock(path: Path, timeout_s: float = 60.0) -> Iterator[None]:
    """Cross-platform lock by exclusive file creation, so parallel seeds never interleave rows."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{path} is held; delete it if no run is writing") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)
