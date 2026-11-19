"""H3: train on the real fans' normal audio and score their faults, under fixed split rules.

The split is the same on dev and on eval, so a dev figure predicts what eval measures
(DECISIONS 2026-11-18). For a test session S, with a model that has never heard S:

- ``train``: every usable normal clip of the training sessions (main and control takes).
- ``limits``: the first ``limit_clips`` clips of S's main normal take. They set the light's
  limits (and the level and mount-centring references), as H6 calibrates on the device.
- ``normal``: the rest of S's main normal take, the running the light is judged on.
- ``control``: S's control take, normal audio after the fan was taken off and put back.
- ``fault``: S's fault takes. A take recorded out of protocol order is ``flagged``: scored
  and shown, but left out of the choice of model.

Dev has one fold per dev session (trained on the other dev sessions); eval trains on every
dev session and tests each eval session, only for a config marked ``final: true``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from andon import metrics, train_idcnn
from andon.data.dataset import Clip
from andon.data.features import SplitFeatures, load_or_extract
from andon.experiment import (
    RESULTS_LOG,
    Config,
    append_result,
    git_commit,
    load_config,
    read_results,
    require_final,
    set_seed,
)
from andon.models.classic import fit_and_score
from andon.realfan import usable_rows

ROLES = ("train", "limits", "normal", "control", "fault")
FAULTS = ("obstruction", "underspeed")
CACHE_DIR = Path("data/cache")
SCORES_DIR = CACHE_DIR / "h3_scores"  # window scores per run, for the light simulation
CONFIGS = ("h3_level", "h3_gmm", "h3_gmm_centred", "h3_idcnn_mixup")


@dataclass(frozen=True)
class Fold:
    train: tuple[str, ...]  # sessions whose normal clips train the model
    test: str  # the session scored

    @property
    def name(self) -> str:
        return f"{'+'.join(self.train)}>{self.test}"

    @property
    def slug(self) -> str:  # for file names: Windows refuses ">"
        return f"{'+'.join(self.train)}-to-{self.test}"


@dataclass(frozen=True)
class FanClip:
    clip: Clip
    session: str
    take: str
    fault: str  # "none" for normal clips
    order: int  # position in its take
    role: str  # one of ROLES
    flagged: bool = False  # a fault take recorded out of protocol order


def folds(recording: dict, split: str) -> list[Fold]:
    """The folds of a split, from the recording config's session lists."""
    dev = list(recording["sessions"]["dev"])
    if split == "dev_test":
        if len(dev) < 2:
            raise ValueError(
                "dev needs two sessions: each is tested by a model trained on the rest"
            )
        return [Fold(tuple(s for s in dev if s != test), test) for test in dev]
    if split == "eval":
        return [Fold(tuple(dev), test) for test in recording["sessions"]["eval"]]
    raise ValueError(f"split must be dev_test or eval, got {split}")


def read_recording(cfg: Config) -> dict:
    path = Path(cfg.raw["data"]["recording_config"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_manifest(root: Path) -> list[dict]:
    with (Path(root) / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_sessions(cfg: Config, recording: dict, fold: Fold, split: str) -> None:
    """Refuse a fold whose sessions don't belong to ``split``; eval only for a final config."""
    sessions = recording["sessions"]
    dev, ev = set(sessions["dev"]), set(sessions["eval"])
    if not set(fold.train) <= dev:
        raise ValueError(f"fold {fold.name}: training sessions must be dev sessions")
    if split == "dev_test" and fold.test not in dev:
        raise ValueError(f"fold {fold.name}: {fold.test} is not a dev session")
    if split == "eval":
        require_final(cfg, split)
        if fold.test not in ev:
            raise ValueError(f"fold {fold.name}: {fold.test} is not an eval session")


def _clip(root: Path, r: dict) -> Clip:
    return Clip(root / r["file"], "realfan", "test", r["fan_id"], r["condition"] == "anomaly")


def fold_clips(
    cfg: Config, recording: dict, rows: list[dict], fold: Fold, split: str
) -> list[FanClip]:
    """Every clip a fold uses, with its role. Rows go through ``usable_rows`` first."""
    _check_sessions(cfg, recording, fold, split)
    root = cfg.data_root
    limit_clips = int(cfg.raw["data"]["limit_clips"])
    out_of_order = set(recording.get("out_of_order_takes") or {})
    rows = usable_rows([r for r in rows if r["session"] in (*fold.train, fold.test)], recording)
    clips = []
    for r in rows:
        if r["session"] in fold.train and r["fault"] == "none":
            # Experiment rule 1: only normal audio trains. The file name says so as well.
            if r["condition"] != "normal" or Path(r["file"]).name.startswith("anomaly_"):
                raise ValueError(f"{r['file']}: a fault clip in a training set")
            clips.append(FanClip(_clip(root, r), r["session"], r["take"], "none", 0, "train"))
    test = [r for r in rows if r["session"] == fold.test]
    normal_takes: dict[str, dict[str, str]] = defaultdict(dict)  # fan -> take -> first clip time
    for r in test:
        if r["fault"] == "none":
            t = normal_takes[r["fan_id"]]
            t[r["take"]] = min(t.get(r["take"], r["recorded_utc"]), r["recorded_utc"])
    main = {}
    for fan, takes in normal_takes.items():
        if len(takes) != 2:
            raise ValueError(
                f"{fold.test} {fan}: {len(takes)} usable normal takes, need a main and a control"
            )
        main[fan] = min(takes, key=takes.get)  # the protocol records the main take first
    for r in test:
        order = int(r["clip_in_take"])
        if r["fault"] != "none":
            role = "fault"
        elif r["take"] != main[r["fan_id"]]:
            role = "control"
        else:
            role = "limits" if order < limit_clips else "normal"
        clips.append(
            FanClip(
                _clip(root, r),
                r["session"],
                r["take"],
                r["fault"],
                order,
                role,
                flagged=r["take"] in out_of_order,
            )
        )
    train_fans = {c.clip.machine_id for c in clips if c.role == "train"}
    if train_fans != set(main):
        raise ValueError(f"fold {fold.name}: fans differ between training and test sessions")
    for fan in main:
        if not any(c.role == "normal" and c.clip.machine_id == fan for c in clips):
            raise ValueError(f"{fold.test} {fan}: no normal clips left after the limit clips")
    return clips


def counts(clips: list[FanClip]) -> dict[tuple[str, str, str], int]:
    """Clips per (fan, role, fault), for the reports' count columns."""
    out: dict[tuple[str, str, str], int] = defaultdict(int)
    for c in clips:
        out[(c.clip.machine_id, c.role, c.fault)] += 1
    return dict(out)


# --- Scoring -------------------------------------------------------------------------------
# Every candidate scores the same windows (``window.frames`` every ``window.hop`` frames, as
# the CNN's patches), and a clip's score is the mean of its windows' scores.


def windows(feats: SplitFeatures, width: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Absolute first frame of every window, in clip order, and the clip each belongs to."""
    starts, owner = [], []
    for i in range(feats.n_clips):
        s = train_idcnn.patch_starts(int(feats.offsets[i + 1] - feats.offsets[i]), width, hop)
        starts.append(feats.offsets[i] + s)
        owner.append(np.full(len(s), i))
    return np.concatenate(starts), np.concatenate(owner)


def _window_sums(values: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    # A running sum makes every window's total two lookups; float64 keeps the difference exact.
    zero = np.zeros((1, *values.shape[1:]))
    cs = np.concatenate([zero, np.cumsum(values, axis=0, dtype=np.float64)])
    return cs[starts + width] - cs[starts]


def window_means(frames: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """Mean log-mel spectrum of each window, ``(n_windows, n_mels)``."""
    return _window_sums(frames, starts, width) / width


def window_levels(frames: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """Level of each window in dB: the mean mel power over its frames and bands."""
    power = (10.0 ** (frames.astype(np.float64) / 10.0)).mean(axis=1)
    return 10.0 * np.log10(_window_sums(power, starts, width) / width)


def _subset(feats: SplitFeatures, idx: np.ndarray | list[int]) -> SplitFeatures:
    parts = [feats.clip(int(i)) for i in idx]
    offsets = np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int64)
    return SplitFeatures(frames=np.concatenate(parts), offsets=offsets)


def _role_idx(clips: list[FanClip], role: str, fan: str) -> list[int]:
    return [i for i, c in enumerate(clips) if c.role == role and c.clip.machine_id == fan]


def centres(clips: list[FanClip], feats: SplitFeatures) -> np.ndarray:
    """The mean spectrum subtracted from each clip's windows (Gc), ``(n_clips, n_mels)``.

    A training clip gets its own take's mean; a test clip gets its fan's limit clips' mean,
    the only part of the tested session a calibration may use. The control take gets the
    same, because the light is not recalibrated after a remount.
    """
    out = np.empty((len(clips), feats.frames.shape[1]))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, c in enumerate(clips):
        groups[("take", c.take) if c.role == "train" else ("fan", c.clip.machine_id)].append(i)
    for (kind, key), idx in groups.items():
        source = idx if kind == "take" else _role_idx(clips, "limits", key)
        out[idx] = _subset(feats, source).frames.mean(axis=0, dtype=np.float64)
    return out


@dataclass
class Scored:
    owner: np.ndarray  # clip index of each test window
    scores: np.ndarray  # window scores
    extra: dict  # (fan or "all", key) -> value: CNN training accuracy and int8 range checks
    model: object = None  # the trained CNN, for its checkpoint

    def clip_scores(self, n_clips: int) -> np.ndarray:
        counts = np.bincount(self.owner, minlength=n_clips)
        sums = np.bincount(self.owner, weights=self.scores, minlength=n_clips)
        with np.errstate(invalid="ignore", divide="ignore"):
            return sums / counts  # NaN for training clips, which have no test windows


def score(
    cfg: Config, seed: int, clips: list[FanClip], feats: SplitFeatures, device: torch.device
) -> Scored:
    """Window scores of every clip that isn't a training clip, for one seed."""
    width, hop = int(cfg.raw["window"]["frames"]), int(cfg.raw["window"]["hop"])
    starts, owner = windows(feats, width, hop)
    roles = np.array([c.role for c in clips])
    fan_of = np.array([c.clip.machine_id for c in clips])
    fans = sorted(set(fan_of))
    test_w = roles[owner] != "train"
    kind = cfg.model
    extra: dict = {}
    model = None
    set_seed(seed)
    if kind == "level":
        level = window_levels(feats.frames, starts, width)
        ref = {f: level[np.isin(owner, _role_idx(clips, "limits", f))].mean() for f in fans}
        # Two-sided: underspeed is quieter than normal running and obstruction louder.
        s = np.abs(level - np.array([ref[f] for f in fan_of[owner]]))
    elif kind in ("gmm", "gmm_centred"):
        x = window_means(feats.frames, starts, width)
        if kind == "gmm_centred":
            x = x - centres(clips, feats)[owner]
        ids = fan_of[owner]
        s = np.full(len(starts), np.nan)
        s[test_w] = fit_and_score(
            dict(cfg.raw["model"], type="gmm"),
            seed,
            x[~test_w],
            list(ids[~test_w]),
            x[test_w],
            list(ids[test_w]),
        )
    elif kind == "id_cnn":
        train = np.flatnonzero(roles == "train")
        test = np.flatnonzero(roles != "train")
        f_train = _subset(feats, train)
        labels = np.array([fans.index(f) for f in fan_of[train]])
        model, history = train_idcnn.train_idcnn(f_train, labels, len(fans), cfg, device)
        true_ids = np.array([fans.index(f) for f in fan_of[test]])
        ws, own = train_idcnn.window_scores(
            model, _subset(feats, test), true_ids, width, hop, device
        )
        # Test clips keep their order in both lists, so their windows line up with test_w.
        if not (test[own] == owner[test_w]).all():
            raise RuntimeError("CNN windows do not line up with the test windows")
        s = np.full(len(starts), np.nan)
        s[test_w] = ws
        extra[("all", "train_acc")] = float(history[-1]["train_acc"])
        # The int8 input is calibrated on the training range; a value outside it would clip.
        x_train = model.normalize(torch.from_numpy(f_train.frames)).numpy()
        lo, hi = float(x_train.min()), float(x_train.max())
        for fan in fans:
            groups = {"normal": _role_idx(clips, "normal", fan)}
            for fault in FAULTS:
                groups[fault] = [
                    i for i in _role_idx(clips, "fault", fan) if clips[i].fault == fault
                ]
            for key, idx in groups.items():
                if idx:
                    x = model.normalize(torch.from_numpy(_subset(feats, idx).frames)).numpy()
                    extra[(fan, f"outside_{key}")] = float(((x < lo) | (x > hi)).mean())
    else:
        raise ValueError(f"unknown H3 model type {kind!r}")
    return Scored(owner[test_w], s[test_w], extra, model)


# --- Metrics -------------------------------------------------------------------------------


def _auc(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    x = np.r_[pos, neg]
    return metrics.auc(y, x), metrics.pauc(y, x)


def fold_rows(clips: list[FanClip], clip_score: np.ndarray) -> list[dict]:
    """Per fan: one row per fault and one check row (``fault=control``), AUCs as fractions.

    A fault's AUC is against all of the fan's test normals (same mount and control take);
    the check row's is the control take against the same-mount normals, which is about 0.5
    when a model doesn't hear the remount.
    """
    out = []
    for fan in sorted({c.clip.machine_id for c in clips}):
        normal = _role_idx(clips, "normal", fan)
        control = _role_idx(clips, "control", fan)
        both = clip_score[normal + control]
        for fault in FAULTS:
            idx = [i for i in _role_idx(clips, "fault", fan) if clips[i].fault == fault]
            if not idx:
                continue
            pos = clip_score[idx]
            auc, pauc = _auc(pos, both)
            out.append(
                {
                    "fan": fan,
                    "fault": fault,
                    "auc": auc,
                    "pauc": pauc,
                    "vs_same": _auc(pos, clip_score[normal])[0],
                    "vs_control": _auc(pos, clip_score[control])[0],
                    "flagged": int(any(clips[i].flagged for i in idx)),
                    "n_fault": len(idx),
                    "n_normal": len(normal),
                    "n_control": len(control),
                }
            )
        # Drift: the later half of the same-mount normals against the earlier half.
        ordered = sorted(normal, key=lambda i: clips[i].order)
        half = len(ordered) // 2
        auc, pauc = _auc(clip_score[control], clip_score[normal])
        out.append(
            {
                "fan": fan,
                "fault": "control",
                "auc": auc,
                "pauc": pauc,
                "drift": _auc(clip_score[ordered[half:]], clip_score[ordered[:half]])[0],
                "n_normal": len(normal),
                "n_control": len(control),
            }
        )
    return out


def _notes(row: dict, fold: Fold, extra: dict) -> str:
    parts = [f"fold={fold.name}"]
    for k, v in row.items():
        if k not in ("fan", "auc", "pauc"):
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    if row["fault"] == "control":
        for (fan, key), v in sorted(extra.items()):
            if fan in (row["fan"], "all"):
                parts.append(f"{key}={v:.4f}")
    return " ".join(parts)


def parse_notes(notes: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in notes.split() if "=" in p)


def _logged_runs(cfg: Config, split: str, log: Path) -> set[tuple[str, str]]:
    return {
        (parse_notes(r["notes"]).get("fold", ""), r["seed"])
        for r in read_results(log)
        if r["config"] == cfg.name and r["split"] == split
    }


def _save_scores(path: Path, clips: list[FanClip], scored: Scored) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        path=np.array([c.clip.path.as_posix() for c in clips]),
        fan=np.array([c.clip.machine_id for c in clips]),
        role=np.array([c.role for c in clips]),
        fault=np.array([c.fault for c in clips]),
        order=np.array([c.order for c in clips]),
        flagged=np.array([c.flagged for c in clips]),
        owner=scored.owner,
        scores=scored.scores,
    )


def run(
    cfg: Config,
    seeds: list[int],
    split: str,
    device: torch.device,
    log: Path = RESULTS_LOG,
    cache_dir: Path = CACHE_DIR,
    scores_dir: Path = SCORES_DIR,
    checkpoint_dir: Path = Path("checkpoints"),
) -> list[dict]:
    """Score every fold of ``split`` for each seed; append the rows; save window scores."""
    if split == "eval":
        require_final(cfg, split)  # before any audio is read
    if cfg.model == "level" and seeds != [0]:
        raise ValueError("the level baseline has no seed: run it with --seed 0 only")
    recording = read_recording(cfg)
    manifest = read_manifest(cfg.data_root)
    done = _logged_runs(cfg, split, log)
    logged = []
    for fold in folds(recording, split):
        clips = fold_clips(cfg, recording, manifest, fold, split)
        feats = load_or_extract(
            [c.clip for c in clips], cfg.logmel, cache_dir, f"realfan_h3_{fold.slug}"
        )
        for seed in seeds:
            if (fold.name, str(seed)) in done:
                print(f"{cfg.name} {fold.name} seed {seed}: already logged, skipped")
                continue
            print(f"{cfg.name} {fold.name} seed {seed}", flush=True)
            scored = score(cfg, seed, clips, feats, device)
            name = f"{cfg.name}_{split}_{fold.slug}_seed{seed}"
            _save_scores(scores_dir / f"{name}.npz", clips, scored)
            if scored.model is not None:
                fans = sorted({c.clip.machine_id for c in clips})
                path = checkpoint_dir / f"{name}.pt"
                train_idcnn.save_checkpoint(path, scored.model, cfg, fold.name, seed, fans, [])
            for r in fold_rows(clips, scored.clip_scores(len(clips))):
                notes = _notes(r, fold, scored.extra)
                logged.append(
                    append_result(cfg, r["fan"], split, seed, r["auc"], r["pauc"], notes, log=log)
                )
    return logged


# --- Tables --------------------------------------------------------------------------------


def _pct(values: list[float]) -> str:
    a = 100 * np.array(values, dtype=float)
    if len(a) == 1:
        return f"{a[0]:.2f}"
    return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}"


def cell_values(rows: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """(fan, fault) -> seed -> AUC averaged over the folds whose take isn't flagged."""
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        n = parse_notes(r["notes"])
        if n["fault"] in FAULTS and n["flagged"] == "0":
            acc[(r["machine_type"], n["fault"])][r["seed"]].append(float(r["auc"]))
    return {c: {s: float(np.mean(v)) for s, v in by_seed.items()} for c, by_seed in acc.items()}


def worst_cell(rows: list[dict]) -> tuple[float, tuple[str, str]]:
    """The keep rule's metric: the lowest cell mean over folds and seeds, and its cell."""
    means = {c: float(np.mean(list(v.values()))) for c, v in cell_values(rows).items()}
    cell = min(means, key=means.get)
    return means[cell], cell


def _grouped(rows: list[dict], want: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        n = parse_notes(r["notes"])
        if n["fault"] in want:
            groups[(n["fold"], r["machine_type"], n["fault"])].append(r | {"n": n})
    return dict(sorted(groups.items()))


def tables(configs: list[str], split: str, log: Path = RESULTS_LOG) -> str:
    """The H3 tables of one split, from the results log only."""
    rows = [r for r in read_results(log) if r["split"] == split and r["config"] in configs]
    by_cfg = {c: [r for r in rows if r["config"] == c] for c in configs}
    by_cfg = {c: rs for c, rs in by_cfg.items() if rs}
    if not by_cfg:
        raise ValueError(f"no {split} rows for {configs}")
    label = "Dev test" if split == "dev_test" else "Eval"
    seeds = "; ".join(
        f"`{c}` {', '.join(sorted({r['seed'] for r in rs}))}" for c, rs in by_cfg.items()
    )
    fans = sorted({r["machine_type"] for r in rows})
    cells = [(f, k) for f in fans for k in FAULTS]
    lines = [
        f"{label}, real fans. Seeds: {seeds}. Mean ± sample std over seeds; a single value "
        "where there is one seed.",
        "",
        "## AUC (%) per cell: the fault against all of the fan's test normals",
        "",
        "Each cell is averaged over the folds first, leaving out flagged takes (recorded out of "
        "protocol order). Worst cell: the lowest cell mean, the keep rule's metric.",
        "",
        "| Config | " + " | ".join(f"{f} {k}" for f, k in cells) + " | Worst cell |",
        "| --- |" + " ---: |" * (len(cells) + 1),
    ]
    for c, rs in by_cfg.items():
        cv = cell_values(rs)
        value, (wf, wk) = worst_cell(rs)
        vals = [_pct(list(cv[cell].values())) if cell in cv else "n/a" for cell in cells]
        lines.append(f"| `{c}` | " + " | ".join(vals) + f" | {100 * value:.2f} ({wf} {wk}) |")
    lines += [
        "",
        "## Per fold",
        "",
        "AUC (%) of each fault against all test normals, against the same-mount normals only, "
        "and against the control take only; pAUC (%) against all. Flagged: recorded out of "
        "protocol order, shown but not used to choose. Clips: fault / same-mount / control.",
        "",
        "| Config | Fold | Fan | Fault | AUC | vs same mount | vs control take | pAUC | Flagged "
        "| Clips |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for c, rs in by_cfg.items():
        for (fold, fan, fault), g in _grouped(rs, FAULTS).items():
            n = g[0]["n"]
            lines.append(
                f"| `{c}` | {fold} | {fan} | {fault} | {_pct([float(r['auc']) for r in g])} | "
                f"{_pct([float(r['n']['vs_same']) for r in g])} | "
                f"{_pct([float(r['n']['vs_control']) for r in g])} | "
                f"{_pct([float(r['pauc']) for r in g])} | "
                f"{'yes' if n['flagged'] == '1' else ''} | "
                f"{n['n_fault']} / {n['n_normal']} / {n['n_control']} |"
            )
    lines += [
        "",
        "## Checks",
        "",
        "Control take: AUC (%) of the control take (normal, after a remount) against the "
        "same-mount normals; about 50 means the model does not hear the remount. Drift: AUC (%) "
        "of the later half of the same-mount normals against the earlier half; about 50 means "
        "no drift over the take. CNN only: training accuracy at the last epoch (%, counted on "
        "mixup-blended inputs) and the share (%) of normalised inputs outside the training "
        "range, which the int8 input would clip.",
        "",
        "| Config | Fold | Fan | Control take | Drift | Train acc. | Outside range: normal / "
        "obstruction / underspeed |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for c, rs in by_cfg.items():
        for (fold, fan, _), g in _grouped(rs, ("control",)).items():
            ns = [r["n"] for r in g]
            acc = _pct([float(n["train_acc"]) for n in ns]) if "train_acc" in ns[0] else ""
            outside = ""
            if "outside_normal" in ns[0]:
                keys = ("normal", *FAULTS)
                outside = " / ".join(_pct([float(n[f"outside_{k}"]) for n in ns]) for k in keys)
            lines.append(
                f"| `{c}` | {fold} | {fan} | {_pct([float(r['auc']) for r in g])} | "
                f"{_pct([float(n['drift']) for n in ns])} | {acc} | {outside} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="H3: real-fan training and scoring.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="score one config on dev_test or eval")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--seed", type=int, nargs="+", required=True)
    p_run.add_argument("--split", choices=["dev_test", "eval"], default="dev_test")
    p_run.add_argument("--threads", type=int, help="CPU threads (default: PyTorch's choice)")
    p_tab = sub.add_parser("table", help="write one split's tables from the results log")
    p_tab.add_argument("--split", choices=["dev_test", "eval"], default="dev_test")
    p_tab.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.threads:
            torch.set_num_threads(args.threads)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = run(load_config(args.config), args.seed, args.split, device)
        print(f"{len(rows)} rows logged")
        for r in rows:
            print(f"{r['machine_type']} seed {r['seed']} auc {r['auc']} {r['notes']}")
        return
    text = tables(list(CONFIGS), args.split)
    print(text)
    if args.out:
        command = (
            f"uv run python -m andon.h3 table --split {args.split} --out {args.out.as_posix()}"
        )
        header = f"<!-- Generated by `{command}` at commit {git_commit()}. -->\n\n"
        args.out.write_text(header + text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
