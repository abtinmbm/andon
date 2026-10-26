"""Bar chart of evaluation AUC per machine type for the README, drawn from the results log.

    uv run --group dashboard python -m andon.chart --out docs/eval_auc.png

Each bar is the mean over seeds and its whisker the sample std: the same figures as the
README's evaluation table (:func:`andon.evaluate.type_stats`), so the chart can't drift
from the log.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from andon.data.dataset import MACHINE_TYPES
from andon.evaluate import type_stats
from andon.experiment import RESULTS_LOG

# The three models the README compares, in its column order. Colours are the first three
# slots of a colour-blind-checked categorical palette, assigned in order.
MODELS = {
    "phase3_baseline_eval": ("Autoencoder (DCASE baseline)", "#2a78d6"),
    "phase2_gmm_id": ("Gaussian mixture per machine", "#eb6834"),
    "phase3_idcnn_mixup": ("Machine-ID CNN (final)", "#1baf7a"),
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def draw(
    out: Path,
    models: dict[str, tuple[str, str]] = MODELS,
    split: str = "eval",
    log: Path = RESULTS_LOG,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    groups = [*MACHINE_TYPES, "mean"]
    x = np.arange(len(groups))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, (config, (name, colour)) in enumerate(models.items()):
        stats = type_stats(config, split, log=log)
        means = np.array([stats[g][0] for g in groups])
        sds = np.array([stats[g][1] for g in groups])
        offset = (i - (len(models) - 1) / 2) * width
        # A white edge leaves a thin gap between neighbouring bars.
        ax.bar(x + offset, means, width, yerr=sds, color=colour, edgecolor="#fcfcfb", lw=1.5,
               label=name, error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})  # fmt: skip
        # Label only the mean-of-six bars: the headline, without a number on every bar.
        ax.text(x[-1] + offset, means[-1] + sds[-1] + 1.5, f"{means[-1]:.2f}",
                ha="center", va="bottom", fontsize=8, color=INK)  # fmt: skip

    ax.axhline(50, color=MUTED, ls=":", lw=1)
    ax.text(-0.5, 51, "chance", fontsize=8, color=MUTED, va="bottom")
    ax.axvline(len(MACHINE_TYPES) - 0.5, color=GRID, lw=1)
    ax.set_xticks(x, [*MACHINE_TYPES, "Mean of six"])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("AUC (%)", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.legend(loc="upper center", ncols=len(models), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/eval_auc.png"))
    args = parser.parse_args(argv)
    draw(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
