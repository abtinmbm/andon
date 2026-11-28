"""H7's unseen fault: the final real-fan model scores the d4 tape take once.

No model or setting was chosen on any imbalance clip (the s1 pilot scored a smaller tape
piece; it led to the control takes, not to any model or setting), so this is a fault the
GMM was never tuned on.
The model is ``h3_gmm`` as scored on d3: one GMM per fan, trained on every usable normal
clip of d1 and d2 (``andon.h3.training_clips``). The unseen session holds one fan's fault
take and the control take recorded after it (the tape off, the fan lifted and put back),
so the control take is the only normal running it can be judged against. A remount alone
moves the level (DECISIONS 2026-11-20), so the fault is scored twice: as recorded, and
with every window of the session played at the control take's mean level
(``level_matched``), which leaves only the shape of the spectrum.

Per seed: the clip AUC and pAUC of fault clips against control clips, and the detection
rate at 1 % false alarms: the share of fault windows above the threshold that at most 1 %
of the control windows exceed. Logged once per config, as ``eval``: nothing is chosen
from it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from andon import h3, metrics
from andon.data.features import load_or_extract
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

CONFIGS = ("h7_unseen", "h7_unseen_levelmatched")


def unseen_clips(cfg: Config, recording: dict, rows: list[dict]) -> list[h3.FanClip]:
    """The training clips of the dev sessions, then the unseen session's fault and control
    clips. Refuses anything but one fan with one fault take and one normal take."""
    u = cfg.raw["unseen"]
    session, fan, fault = u["session"], u["fan"], u["fault"]
    sessions = recording["sessions"]
    if session not in (sessions.get("unseen") or []):
        raise ValueError(f"{session} is not an unseen session in the recording config")
    if session in set(sessions["dev"]) | set(sessions["eval"]):
        raise ValueError(f"{session} is also a dev or eval session")
    spec = (recording.get("unseen_faults") or {}).get(fault)
    if spec is None or session not in spec["sessions"]:
        raise ValueError(f"{fault} is not an unseen fault of {session}")
    clips = h3.training_clips(cfg, recording, rows, tuple(sessions["dev"]))
    test = usable_rows([r for r in rows if r["session"] == session], recording)
    if {r["fan_id"] for r in test} != {fan}:
        raise ValueError(
            f"{session} must hold {fan} only, got {sorted({r['fan_id'] for r in test})}"
        )
    takes: dict[str, set[str]] = {"fault": set(), "control": set()}
    for r in test:
        if r["fault"] not in ("none", fault):
            raise ValueError(f"{r['file']}: fault {r['fault']} in {session}")
        role = "control" if r["fault"] == "none" else "fault"
        takes[role].add(r["take"])
        clip = h3._clip(cfg.data_root, r)
        clips.append(h3.FanClip(clip, session, r["take"], r["fault"], int(r["clip_in_take"]), role))
    for role, names in takes.items():
        if len(names) != 1:
            raise ValueError(f"{session}: {len(names)} {role} takes, need exactly one")
    return clips


def detection_at(fault: np.ndarray, normal: np.ndarray, rate: float) -> tuple[float, float, float]:
    """Share of ``fault`` above the lowest threshold that at most ``rate`` of ``normal``
    exceeds; returns (detection rate, false-alarm rate reached, threshold)."""
    if not (np.isfinite(fault).all() and np.isfinite(normal).all()):
        raise ValueError("scores must be finite")
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"rate must be in [0, 1), got {rate}")
    ranked = np.sort(normal)[::-1]
    allowed = int(math.floor(rate * len(ranked)))  # normal windows allowed above
    threshold = float(ranked[allowed])
    return float((fault > threshold).mean()), float((normal > threshold).mean()), threshold


def score(cfg: Config, seed: int, clips: list[h3.FanClip], feats) -> dict:
    """One seed's metrics: clip AUC and pAUC, and window detection at the false-alarm rate."""
    width, hop = int(cfg.raw["window"]["frames"]), int(cfg.raw["window"]["hop"])
    starts, owner = h3.windows(feats, width, hop)
    roles = np.array([c.role for c in clips])
    ids = np.array([c.clip.machine_id for c in clips])[owner]
    x = h3.window_means(feats.frames, starts, width)
    if cfg.raw["model"].get("level_matched"):
        level = h3.window_levels(feats.frames, starts, width)
        x = h3.match_levels(x, level, owner, clips, ref_role="control")
    test_w = roles[owner] != "train"
    set_seed(seed)
    s = fit_and_score(
        dict(cfg.raw["model"], type="gmm"),
        seed,
        x[~test_w],
        list(ids[~test_w]),
        x[test_w],
        list(ids[test_w]),
    )
    w_owner = owner[test_w]
    clip_score = h3.Scored(w_owner, s, {}).clip_scores(len(clips))
    fault_c = np.flatnonzero(roles == "fault")
    control_c = np.flatnonzero(roles == "control")
    y = np.r_[np.ones(len(fault_c)), np.zeros(len(control_c))]
    cs = np.r_[clip_score[fault_c], clip_score[control_c]]
    fault_w = s[roles[w_owner] == "fault"]
    control_w = s[roles[w_owner] == "control"]
    rate = float(cfg.raw["unseen"]["false_alarm_rate"])
    det, fa, thr = detection_at(fault_w, control_w, rate)
    return {
        "auc": metrics.auc(y, cs),
        "pauc": metrics.pauc(y, cs),
        "det": det,
        "fa": fa,
        "threshold": thr,
        "n_fault": len(fault_c),
        "n_control": len(control_c),
        "n_fault_w": len(fault_w),
        "n_control_w": len(control_w),
    }


def run(
    cfg: Config, seeds: list[int], log: Path = RESULTS_LOG, cache_dir: Path = h3.CACHE_DIR
) -> list[dict]:
    """Score every seed and append one row each; refuses a config already logged."""
    require_final(cfg, "eval")  # before any audio is read
    if any(r["config"] == cfg.name for r in read_results(log)):
        raise ValueError(f"{cfg.name} is already logged: the unseen fault is scored once")
    recording = h3.read_recording(cfg)
    clips = unseen_clips(cfg, recording, h3.read_manifest(cfg.data_root))
    feats = load_or_extract([c.clip for c in clips], cfg.logmel, cache_dir, "realfan_h7_unseen")
    u = cfg.raw["unseen"]
    results = [(seed, score(cfg, seed, clips, feats)) for seed in seeds]
    logged = []
    for seed, m in results:  # every seed scored before any row is written: all or none
        train = "+".join(recording["sessions"]["dev"])
        target = float(u["false_alarm_rate"])
        notes = (
            f"session={u['session']} fault={u['fault']} train={train} "
            f"det_at_fa={m['det']:.4f} fa={m['fa']:.4f} fa_target={target:.4f} "
            f"threshold={m['threshold']:.4f} n_fault={m['n_fault']} n_control={m['n_control']} "
            f"n_fault_windows={m['n_fault_w']} n_control_windows={m['n_control_w']}"
        )
        logged.append(
            append_result(cfg, u["fan"], "eval", seed, m["auc"], m["pauc"], notes, log=log)
        )
    return logged


def _mean_sd(values: list[float]) -> str:
    a = 100 * np.array(values, dtype=float)
    return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}" if len(a) > 1 else f"{a[0]:.2f}"


def table(configs: tuple[str, ...] | list[str] = CONFIGS, log: Path = RESULTS_LOG) -> str:
    """The unseen-fault table from the log: mean ± sample std over the logged seeds."""
    lines = [
        "| Config | Fan | Fault | Seeds | AUC (%) | pAUC (%) | Detected at 1 % false alarms (%) "
        "| False alarms reached (%) | Fault / control clips | Fault / control windows |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = read_results(log)
    for name in configs:
        mine = [r for r in rows if r["config"] == name and r["split"] == "eval"]
        if not mine:
            continue
        notes = [h3.parse_notes(r["notes"]) for r in mine]
        n = notes[0]
        seeds = ", ".join(r["seed"] for r in mine)
        lines.append(
            f"| {name} | {mine[0]['machine_type']} | {n['fault']} | {seeds} "
            f"| {_mean_sd([float(r['auc']) for r in mine])} "
            f"| {_mean_sd([float(r['pauc']) for r in mine])} "
            f"| {_mean_sd([float(x['det_at_fa']) for x in notes])} "
            f"| {_mean_sd([float(x['fa']) for x in notes])} "
            f"| {n['n_fault']} / {n['n_control']} "
            f"| {n['n_fault_windows']} / {n['n_control_windows']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="H7: score the unseen fault once.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="score one config's seeds and log them (once)")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--seed", type=int, nargs="+", required=True)
    p_tab = sub.add_parser("table", help="write the table from the results log")
    p_tab.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        for r in run(load_config(args.config), args.seed):
            print(f"{r['config']} {r['machine_type']} seed {r['seed']} auc {r['auc']} {r['notes']}")
        return
    text = table()
    print(text)
    if args.out:
        command = f"uv run python -m andon.unseen table --out {args.out.as_posix()}"
        header = f"<!-- Generated by `{command}` at commit {git_commit()}. -->\n\n"
        args.out.write_text(header + text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
