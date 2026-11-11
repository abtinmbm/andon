"""H3 pilot: train on part of one real-fan session's normal audio and score the rest.

    uv run python -m andon.h3_pilot --config configs/h3_pilot_s1.yaml --seed 0 1 2
        --out results/h3_pilot.md

(one line in PowerShell). Run on session s1 before s2 and s3 exist, to find problems with
the dataset while the recording protocol can still change. It is a check of the data, not
an H3 result: the split and the questions it answers are in the config's comment.

Per fan it reports the AUC of each fault against the held-back normal clips, for the
machine-ID CNN and the per-fan GMM, and three checks:

- ``id accuracy``: how often the CNN names the right fan on held-back normal clips. If it
  can't tell the fans apart, ``-log p(true ID)`` means little.
- ``drift``: the AUC of the later half of the held-back normal clips against the earlier
  half. Both are normal, so about 50 % is expected; a high value means the sound changes
  over minutes of running, and a detector could be hearing the time instead of the fault.
- ``outside range``: the share of normalised log-mel values outside the range the training
  clips cover, which is the range the int8 input is calibrated on. A fault far outside it
  would be clipped on the device.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from andon import metrics, train_idcnn
from andon.data.dataset import Clip
from andon.data.features import SplitFeatures, clip_means, concat, load_or_extract
from andon.experiment import Config, git_commit, load_config, set_seed
from andon.models.classic import fit_and_score

CACHE_DIR = Path("data/cache")


@dataclass(frozen=True)
class PilotClip:
    clip: Clip
    fault: str  # "none" for normal clips
    order: int  # position in its take


def read_session(root: Path, session: str) -> list[PilotClip]:
    """The session's clips from the manifest, sorted by fan, fault and position in the take."""
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["session"] == session]
    if not rows:
        raise ValueError(f"no clips of session {session!r} in {root / 'manifest.csv'}")
    clips = [
        PilotClip(
            Clip(root / r["file"], "realfan", "test", r["fan_id"], r["condition"] == "anomaly"),
            r["fault"],
            int(r["clip_in_take"]),
        )
        for r in rows
    ]
    return sorted(clips, key=lambda c: (c.clip.machine_id, c.fault != "none", c.fault, c.order))


CONTROL = "control"  # normal running from another take, scored like a fault


def read_control(root: Path, spec: dict) -> list[PilotClip]:
    """Normal clips of another session recorded at the same gain, labelled ``CONTROL``.

    They come from a different take than the training clips: the fan was taken off its
    mount and put back in between. If they score like faults, a detector trained on one
    take is hearing the take, and the fault AUCs can't be trusted.
    """
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["session"] == spec["session"]
            and r["fault"] == "none"
            and int(r["gain_shift"]) == int(spec["gain_shift"])
        ]
    return [
        PilotClip(Clip(root / r["file"], "realfan", "test", r["fan_id"], False), CONTROL, i)
        for i, r in enumerate(rows)
    ]


def is_fault(name: str) -> bool:
    return name not in ("none", CONTROL)


def split_normals(clips: list[PilotClip], n_train: int) -> tuple[list[int], list[int]]:
    """Indices of (training, test) clips: per fan, the first ``n_train`` normal clips in
    recording order train, and every other clip, normal or faulty, is a test clip."""
    train, test = [], []
    by_fan: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clips):
        if c.fault == "none":
            by_fan[c.clip.machine_id].append(i)
        else:
            test.append(i)
    for fan, idx in by_fan.items():
        idx.sort(key=lambda i: clips[i].order)
        if len(idx) <= n_train:
            raise ValueError(f"{fan}: {len(idx)} normal clips, need more than {n_train}")
        train += idx[:n_train]
        test += idx[n_train:]
    return sorted(train), sorted(test)


def subset(feats: SplitFeatures, idx: list[int]) -> SplitFeatures:
    """The features of the clips ``idx``, in that order."""
    parts = [feats.clip(i) for i in idx]
    offsets = np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int64)
    return SplitFeatures(frames=np.concatenate(parts), offsets=offsets)


def auc_pct(positive: np.ndarray, negative: np.ndarray) -> float:
    y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    return 100 * metrics.auc(y, np.r_[positive, negative])


def outside_range_pct(model, feats: SplitFeatures, lo: float, hi: float) -> float:
    x = model.normalize(torch.from_numpy(feats.frames)).numpy()
    return 100 * float(((x < lo) | (x > hi)).mean())


def run_seed(cfg: Config, clips: list[PilotClip], feats: SplitFeatures, seed: int) -> dict:
    """Every figure of one seed: ``{(fan, key): value}`` plus ``("all", "train_acc")``."""
    train, test = split_normals(clips, int(cfg.raw["data"]["train_clips"]))
    fans = sorted({c.clip.machine_id for c in clips})
    f_train, f_test = subset(feats, train), subset(feats, test)
    ids_train = [clips[i].clip.machine_id for i in train]
    ids_test = [clips[i].clip.machine_id for i in test]
    labels_test = np.array([fans.index(m) for m in ids_test])
    device = torch.device("cpu")

    set_seed(seed)
    labels = np.array([fans.index(m) for m in ids_train])
    model, history = train_idcnn.train_idcnn(f_train, labels, len(fans), cfg, device)
    m = cfg.raw["model"]
    width, hop = int(m["patch_frames"]), int(m["score_hop"])
    cnn = train_idcnn.clip_scores(model, f_test, labels_test, width, hop, device)
    # Which fan the CNN would name for each test clip: the ID with the lowest clip score.
    named = np.argmin(
        [
            train_idcnn.clip_scores(model, f_test, np.full(len(test), k), width, hop, device)
            for k in range(len(fans))
        ],
        axis=0,
    )
    gmm = fit_and_score(
        cfg.raw["gmm"], seed, clip_means(f_train), ids_train, clip_means(f_test), ids_test
    )
    x_train = model.normalize(torch.from_numpy(f_train.frames)).numpy()
    lo, hi = float(x_train.min()), float(x_train.max())

    out: dict = {("all", "train_acc"): 100 * history[-1]["train_acc"]}
    faults = sorted({c.fault for c in clips if is_fault(c.fault)})
    for fan in fans:
        pos = {
            f: [j for j, i in enumerate(test) if ids_test[j] == fan and clips[i].fault == f]
            for f in ["none", CONTROL, *faults]
        }
        normal = pos["none"]  # in recording order, as split_normals sorted them
        half = len(normal) // 2
        out[(fan, "id_acc")] = 100 * float((named[normal] == fans.index(fan)).mean())
        out[(fan, "outside none")] = outside_range_pct(model, subset(f_test, normal), lo, hi)
        for name, scores in (("cnn", cnn), ("gmm", gmm)):
            out[(fan, f"{name} drift")] = auc_pct(scores[normal[half:]], scores[normal[:half]])
            any_fault = [j for f in faults for j in pos[f]]
            out[(fan, f"{name} all")] = auc_pct(scores[any_fault], scores[normal])
            for f in faults:
                out[(fan, f"{name} {f}")] = auc_pct(scores[pos[f]], scores[normal])
            other = pos[CONTROL]
            out[(fan, f"{name} control")] = (
                auc_pct(scores[other], scores[normal]) if other else float("nan")
            )
            # The fair test of a fault: against normal audio that is also from another take.
            out[(fan, f"{name} faults vs control")] = (
                auc_pct(scores[any_fault], scores[other]) if other else float("nan")
            )
        for f in faults:
            out[(fan, f"outside {f}")] = outside_range_pct(model, subset(f_test, pos[f]), lo, hi)
    return out


def _cell(values: list[float]) -> str:
    a = np.array(values)
    if np.isnan(a).any():
        return "n/a"
    sd = f"{a.std(ddof=1):.2f}" if len(a) > 1 else "nan"
    return f"{a.mean():.2f} ± {sd}"


def report(cfg: Config, clips: list[PilotClip], runs: dict[int, dict]) -> str:
    """Markdown tables: mean ± sample std over seeds, per fan."""
    seeds = sorted(runs)
    fans = sorted({c.clip.machine_id for c in clips})
    faults = sorted({c.fault for c in clips if is_fault(c.fault)})
    train, test = split_normals(clips, int(cfg.raw["data"]["train_clips"]))
    n_normal_test = sum(clips[i].fault == "none" for i in test) // len(fans)
    n_fault = sum(clips[i].fault == faults[0] for i in test) // len(fans)

    def row(label: str, key: str) -> str:
        cells = [_cell([runs[s][(fan, key)] for s in seeds]) for fan in fans]
        return f"| {label} | " + " | ".join(cells) + " |"

    head = "| | " + " | ".join(fans) + " |\n| --- |" + " ---: |" * len(fans)
    lines = [
        f"`{cfg.name}`, session {cfg.raw['data']['session']}, seeds "
        f"{', '.join(map(str, seeds))}: mean ± sample std over seeds. A pilot on one session, "
        "not an H3 result.",
        "",
        f"Per fan: {len(train) // len(fans)} normal clips train, {n_normal_test} normal clips "
        f"and {n_fault} clips of each fault are scored. CNN training accuracy at the last "
        f"epoch: {_cell([runs[s][('all', 'train_acc')] for s in seeds])} %.",
        "",
        "## AUC (%), each fault against the held-back normal clips",
        "",
        head,
    ]
    for name, label in (("cnn", "CNN"), ("gmm", "GMM per fan")):
        lines.append(row(f"{label}, both faults", f"{name} all"))
        lines += [row(f"{label}, {f}", f"{name} {f}") for f in faults]
    lines += [
        "",
        "## Checks",
        "",
        head,
        row("CNN names the right fan, held-back normal clips (%)", "id_acc"),
        row("Drift: later against earlier held-back normals, CNN AUC (%)", "cnn drift"),
        row("Drift: later against earlier held-back normals, GMM AUC (%)", "gmm drift"),
        row("Values outside the training range, normal (%)", "outside none"),
    ]
    n_control = sum(c.fault == CONTROL for c in clips)
    if n_control:
        spec = cfg.raw["data"]["control"]
        lines[-1:-1] = [
            row(
                "Another take's normal clips against held-back normals, CNN AUC (%)", "cnn control"
            ),
            row(
                "Another take's normal clips against held-back normals, GMM AUC (%)", "gmm control"
            ),
            row(
                "Both faults against the other take's normal clips, CNN AUC (%)",
                "cnn faults vs control",
            ),
            row(
                "Both faults against the other take's normal clips, GMM AUC (%)",
                "gmm faults vs control",
            ),
        ]
        note = (
            f" The other take is the {n_control} normal clips of session {spec['session']} "
            f"recorded at gain {spec['gain_shift']}: the same fan, taken off its mount and put "
            "back before the training take. They are normal, so about 50 % is expected; a high "
            "value means the models hear the take, not only the fault."
        )
    else:
        note = ""
    lines += [row(f"Values outside the training range, {f} (%)", f"outside {f}") for f in faults]
    lines += [
        "",
        "Drift compares two halves of normal audio, so about 50 % means no drift. The "
        "training range is the smallest to largest normalised log-mel value in the training "
        "clips, which is what the int8 input is calibrated on." + note,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="H3 pilot on one real-fan session.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, nargs="+", required=True)
    parser.add_argument("--threads", type=int, help="CPU threads (default: PyTorch's choice)")
    parser.add_argument("--out", type=Path, help="write the tables to this file")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    session = cfg.raw["data"]["session"]
    clips = read_session(cfg.data_root, session)
    feats = load_or_extract([c.clip for c in clips], cfg.logmel, CACHE_DIR, f"realfan_{session}")
    spec = cfg.raw["data"].get("control")
    if spec:
        control = read_control(cfg.data_root, spec)
        name = f"realfan_{spec['session']}_control"
        extra = load_or_extract([c.clip for c in control], cfg.logmel, CACHE_DIR, name)
        clips, feats = clips + control, concat([feats, extra])
    runs = {}
    for seed in args.seed:
        print(f"seed {seed}")
        runs[seed] = run_seed(cfg, clips, feats, seed)
    text = report(cfg, clips, runs)
    print(text)
    if args.out:
        command = (
            f"uv run python -m andon.h3_pilot --config {args.config.as_posix()} --seed "
            f"{' '.join(map(str, args.seed))} --out {args.out.as_posix()}"
        )
        header = f"<!-- Generated by `{command}` at commit {git_commit()}. -->\n\n"
        args.out.write_text(header + text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
