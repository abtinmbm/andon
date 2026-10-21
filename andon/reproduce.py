"""Reproduce the headline result: the final CNN's AUC on the evaluation set.

    uv sync
    uv run python -m andon.reproduce

Downloads the evaluation test clips and their ground truth from Zenodo (MD5-checked)
and the 18 trained ``phase3_idcnn_mixup`` checkpoints from the project's GitHub release
(SHA-256-checked), scores the evaluation set, and checks every machine type and seed
against its row in ``results/results_log.csv``.

Nothing is appended to the results log: this re-checks a final model that was already
scored once, and is not a new evaluation run. The re-scored rows go to
``data/reproduced_eval.csv`` instead.

Only a config and seeds that already have eval rows in the log can be re-checked, so
this can't be used to score the evaluation set for a new model.

``--train`` retrains the checkpoints that are missing instead of downloading them; ones
already on disk are scored as they are. It needs the full development set and additional
training set (about 15 GB of audio more) and takes many hours on a CPU. CPU training is
seeded but not guaranteed to be bit-identical across machines, so retrained results can
differ from the logged ones by more than the tolerance and then report "NOT reproduced".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from andon import score, train_idcnn
from andon.data import download
from andon.evaluate import comparison_table
from andon.experiment import RESULTS_LOG, Config, is_logged, load_config, read_results

DEFAULT_CONFIG = Path("configs/phase3_idcnn_mixup.yaml")
REPRODUCED_LOG = Path("data/reproduced_eval.csv")
RELEASE_URL = "https://github.com/abtinmbm/andon/releases/download/checkpoints-v1"
# Largest difference from a logged AUC or pAUC (as a fraction) that still counts as a
# match. Scores are deterministic, but another CPU or PyTorch build can round floats
# differently and swap two near-tied clips.
TOLERANCE = 0.0005

# SHA-256 of each released checkpoint, the files that produced the logged eval rows.
CHECKPOINT_SHA256 = {
    "phase3_idcnn_mixup_ToyCar_seed0.pt": (
        "829e58f911024ad479a7887e92a4d200678ad5519684a7b3ec3264604d257d7d"
    ),
    "phase3_idcnn_mixup_ToyCar_seed1.pt": (
        "f50e88ff349379d388a9b2ced79f3a4a145b2d03dacf42c2dcd5456a55f1084c"
    ),
    "phase3_idcnn_mixup_ToyCar_seed2.pt": (
        "94beb1153a6f212b7c02d15f1055af47d02f89d4eec3ae14624cb97d16fd77fc"
    ),
    "phase3_idcnn_mixup_ToyConveyor_seed0.pt": (
        "31442e9b919ae5044d9115a267c1f0feaa7db9a53871a6e70d98892e0d3820f2"
    ),
    "phase3_idcnn_mixup_ToyConveyor_seed1.pt": (
        "10ad89ea2740735c18afba069b87ee533981e4139f9483e23503909d55b4fac2"
    ),
    "phase3_idcnn_mixup_ToyConveyor_seed2.pt": (
        "2c7df2357c0cf0248254c43cfc61772bf5f8f1d773f7b00671f67602727ad6ac"
    ),
    "phase3_idcnn_mixup_fan_seed0.pt": (
        "c1f4f239f2e7c6cc7d3ae13b758c18eed1ccdbfaf0772aacf3d407e0e1a19f41"
    ),
    "phase3_idcnn_mixup_fan_seed1.pt": (
        "c4a07897d74339c57854a14f9e272d772e454bcfaf1ef28dd9b6c2310621a0d1"
    ),
    "phase3_idcnn_mixup_fan_seed2.pt": (
        "c40799e734fad351d963f1259d97e50b15014efa4c200881f09de4c3ca6a0cf7"
    ),
    "phase3_idcnn_mixup_pump_seed0.pt": (
        "331975d7cae556db920293107f33732bea29c76f2a51f254aad5543919a21cb2"
    ),
    "phase3_idcnn_mixup_pump_seed1.pt": (
        "a9403aa5cefd44189d5f9dc837395ea1b94a09156de355b1b2e4de7f6f27d45f"
    ),
    "phase3_idcnn_mixup_pump_seed2.pt": (
        "d56e96035785e635d5892f596a1d9b2943d28a7635a119b9b53f1fa22a480eb0"
    ),
    "phase3_idcnn_mixup_slider_seed0.pt": (
        "84c8b546b7bd1cb42a30120eaada168935745f9b090c75ba39f91e3589acc6b7"
    ),
    "phase3_idcnn_mixup_slider_seed1.pt": (
        "9be4185cab2be20ca519cc55544ba182b5ae78d5da2683d1cd0eaabbd0f1d5aa"
    ),
    "phase3_idcnn_mixup_slider_seed2.pt": (
        "b31f6a8fd2dac44efae8ba39e0448e5e3e20de9ea6e983b5688d75853b5539f4"
    ),
    "phase3_idcnn_mixup_valve_seed0.pt": (
        "740db3f3c00308c0f43b0d8dc4adc63ba133429365dd4a8e64aa0f9c01df88f9"
    ),
    "phase3_idcnn_mixup_valve_seed1.pt": (
        "26d6b75a819ec33206264d45f9f4e3bcb7f680d2f3cf1ac7a58fbfdcdc397c53"
    ),
    "phase3_idcnn_mixup_valve_seed2.pt": (
        "c777fadf15f484cc5207c43ac8f2c840becc72fb9d0423ef4d2fe55b0742b706"
    ),
}


def fetch_eval_data(root: Path, machine_types: list[str]) -> None:
    """The evaluation test clips and labels only; the CNN needs no training data to score."""
    download.fetch_ground_truth(root)
    for machine_type in machine_types:
        download.fetch(machine_type, root, dataset="eval_data", splits=("test",))


def fetch_training_data(root: Path, machine_types: list[str]) -> None:
    for machine_type in machine_types:
        download.fetch(machine_type, root, dataset="dev_data")
        download.fetch(machine_type, root, dataset="eval_data", splits=("train",))


def fetch_checkpoints(
    cfg: Config, machine_types: list[str], seeds: list[int], base_url: str = RELEASE_URL
) -> None:
    """Download each released checkpoint that is missing; refuse to overwrite a different one."""
    for machine_type in machine_types:
        for seed in seeds:
            path = cfg.checkpoint_path(machine_type, seed)
            expected = CHECKPOINT_SHA256.get(path.name)
            if expected is None:
                raise ValueError(f"{path.name} is not in the release; use --train")
            if path.exists():
                actual = download.file_digest(path, "sha256")
                if actual != expected:
                    raise RuntimeError(
                        f"{path} differs from the released checkpoint (SHA-256 {actual}, "
                        f"expected {expected}). Move it away to download the released one, "
                        "or pass --train to score your own."
                    )
                continue
            download.download(f"{base_url}/{path.name}", path, expected, algorithm="sha256")


def train_missing(
    cfg: Config, machine_types: list[str], seeds: list[int], device: torch.device
) -> None:
    for machine_type in machine_types:
        for seed in seeds:
            if cfg.checkpoint_path(machine_type, seed).exists():
                print(f"[{machine_type} seed {seed}] checkpoint exists, not retraining")
                continue
            train_idcnn.train_and_save(cfg, machine_type, seed, device)


def compare(
    cfg: Config,
    machine_types: list[str],
    seeds: list[int],
    reproduced: Path = REPRODUCED_LOG,
    log: Path = RESULTS_LOG,
) -> tuple[list[str], float]:
    """One line per machine type and seed, and the largest AUC or pAUC difference."""

    def by_key(path: Path) -> dict[tuple[str, str], dict[str, str]]:
        return {
            (r["machine_type"], r["seed"]): r
            for r in read_results(path)
            if r["config"] == cfg.name and r["split"] == "eval"
        }

    logged, mine = by_key(log), by_key(reproduced)
    lines, worst = [], 0.0
    for machine_type in machine_types:
        for seed in seeds:
            key = (machine_type, str(seed))
            new, old = mine[key], logged[key]
            diff = max(abs(float(new[m]) - float(old[m])) for m in ("auc", "pauc"))
            worst = max(worst, diff)
            verdict = "match" if diff <= TOLERANCE else "DIFFERS"
            lines.append(
                f"{machine_type:<12} seed {seed}  AUC {new['auc']} (logged {old['auc']})  "
                f"pAUC {new['pauc']} (logged {old['pauc']})  {verdict}"
            )
    return lines, worst


def headline(
    cfg: Config, log: Path, machine_types: list[str], seeds: list[int]
) -> tuple[float, float]:
    """Mean of the machine types within each seed, then mean and sample std over seeds, in %,
    as ``evaluate.comparison_table`` computes its "mean" row. Every type and seed must be there.
    """
    auc = {
        (r["machine_type"], r["seed"]): float(r["auc"])
        for r in read_results(log)
        if r["config"] == cfg.name and r["split"] == "eval"
    }
    missing = [(t, s) for t in machine_types for s in seeds if (t, str(s)) not in auc]
    if missing:
        raise ValueError(f"{log}: no eval row for {missing}")
    per_seed = [np.mean([auc[(t, str(s))] for t in machine_types]) for s in seeds]
    a = 100 * np.array(per_seed)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def run(
    cfg: Config,
    seeds: list[int],
    machine_types: list[str],
    device: torch.device,
    retrain: bool = False,
    reproduced: Path = REPRODUCED_LOG,
    log: Path = RESULTS_LOG,
) -> bool:
    """Fetch, score and compare; True when every type and seed matches the log."""
    # Checked before anything is downloaded, trained or scored: an eval result printed
    # here for a model that was never scored would be an unlogged second look at eval.
    if cfg.model != "id_cnn":
        raise ValueError(f"reproduce re-checks the machine-ID CNN only, not {cfg.model}")
    unlogged = [
        (t, s) for t in machine_types for s in seeds if not is_logged(cfg, t, s, "eval", log=log)
    ]
    if unlogged:
        raise ValueError(
            f"reproduce only re-checks a logged eval run; {cfg.name} has no eval row for {unlogged}"
        )
    if reproduced.resolve() == log.resolve():
        raise ValueError("the re-scored rows must not go to the results log")

    if retrain:
        fetch_training_data(cfg.data_root, machine_types)
        train_missing(cfg, machine_types, seeds, device)
    else:
        fetch_checkpoints(cfg, machine_types, seeds)
    fetch_eval_data(cfg.data_root, machine_types)

    reproduced.unlink(missing_ok=True)  # score.run skips rows already in its log
    score.run(cfg, seeds, machine_types, "eval", device, log=reproduced)

    print()
    table = comparison_table([cfg.name], "auc", reproduced, "eval")
    print(table.replace("from results_log.csv", f"re-scored into {reproduced.as_posix()}"))
    print()
    lines, worst = compare(cfg, machine_types, seeds, reproduced, log)
    print("\n".join(lines))
    print()
    ok = worst <= TOLERANCE
    if len(seeds) > 1:
        mean, sd = headline(cfg, reproduced, machine_types, seeds)
        old_mean, old_sd = headline(cfg, log, machine_types, seeds)
        print(
            f"Eval AUC, mean of {len(machine_types)} machine types, seeds "
            f"{', '.join(map(str, seeds))}: {mean:.2f} +/- {sd:.2f} % "
            f"(logged: {old_mean:.2f} +/- {old_sd:.2f} %)"
        )
    print(
        f"Largest difference from the log: {worst:.4f} (tolerance {TOLERANCE}). "
        + ("Reproduced." if ok else "NOT reproduced.")
    )
    return ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce the headline eval result.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--machines", nargs="*", help="machine types (default: all in the config)")
    parser.add_argument(
        "--train", action="store_true", help="retrain the checkpoints instead of downloading them"
    )
    parser.add_argument("--threads", type=int, help="CPU threads for --train")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config {cfg.name}, seeds {args.seed}, device {device}")
    ok = run(cfg, args.seed, args.machines or cfg.machine_types, device, retrain=args.train)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
