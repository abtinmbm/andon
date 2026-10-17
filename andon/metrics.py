"""AUC and pAUC, the DCASE 2020 Task 2 metrics, written out in NumPy.

The official baseline and evaluator both call
``sklearn.metrics.roc_auc_score(y_true, y_score, max_fpr=0.1)`` for pAUC. With
``max_fpr`` set, scikit-learn does not return the raw partial area: it applies the
McClish standardization, which maps a random detector to 0.5 and a perfect one to
1.0. :func:`pauc` returns that same standardized value so our numbers compare
directly with the published ones; :func:`partial_auc` gives the raw area if needed.

Scores follow the DCASE convention: higher means more anomalous, label 1 = anomaly.
"""

from __future__ import annotations

import numpy as np

DCASE_MAX_FPR = 0.1


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False and true positive rates at every distinct threshold, starting at (0, 0).

    Tied scores share one threshold, so a tie between a normal and an anomalous
    clip draws a diagonal segment and earns half credit, as in the Mann-Whitney U.
    """
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} labels vs {y_score.shape} scores")
    n_pos = int(y_true.sum())
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need at least one normal and one anomalous example")

    order = np.argsort(-y_score, kind="mergesort")
    scores, labels = y_score[order], y_true[order]
    # Last index of each run of equal scores: the cumulative counts there are one ROC point.
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), scores.size - 1]
    tps = np.cumsum(labels)[ends]
    fps = (ends + 1) - tps
    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    return fpr, tpr


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    fpr, tpr = roc_curve(y_true, y_score)
    return float(np.trapezoid(tpr, fpr))


def partial_auc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = DCASE_MAX_FPR) -> float:
    """Raw ROC area for FPR in [0, max_fpr], interpolating the curve at max_fpr."""
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}")
    fpr, tpr = roc_curve(y_true, y_score)
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    if stop < fpr.size:
        # Cut the segment that crosses max_fpr. fpr[stop] > fpr[stop - 1], so no divide by zero.
        x0, x1, y0, y1 = fpr[stop - 1], fpr[stop], tpr[stop - 1], tpr[stop]
        tpr_cut = y0 + (y1 - y0) * (max_fpr - x0) / (x1 - x0)
        fpr = np.r_[fpr[:stop], max_fpr]
        tpr = np.r_[tpr[:stop], tpr_cut]
    return float(np.trapezoid(tpr, fpr))


def pauc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = DCASE_MAX_FPR) -> float:
    """McClish-standardized partial AUC, identical to scikit-learn's ``max_fpr`` result."""
    area = partial_auc(y_true, y_score, max_fpr)
    min_area = 0.5 * max_fpr**2  # the diagonal: a detector that guesses
    max_area = max_fpr  # a perfect detector
    return 0.5 * (1.0 + (area - min_area) / (max_area - min_area))
