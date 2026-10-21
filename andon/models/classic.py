"""Classic anomaly detectors on one vector per clip (the mean log-mel spectrum).

Each detector is fitted on normal clips only and returns a score per clip where higher
means more anomalous, the same convention as the autoencoder's reconstruction error.

- Mahalanobis: squared distance from the mean of normal clips, under a Ledoit-Wolf
  shrunk covariance. With about 1,000 clips per machine ID for 128 bands, the plain
  sample covariance is noisy in its smallest directions, which the distance weights most.
- GMM: negative log-likelihood under a Gaussian mixture.
- Isolation forest: the negated ``score_samples``, so short isolation paths score high.

All three are small enough to port to C later as fallbacks for the microcontroller.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture


class Detector(Protocol):
    def fit(self, x: np.ndarray) -> Detector: ...

    def score(self, x: np.ndarray) -> np.ndarray: ...


class Mahalanobis:
    def fit(self, x: np.ndarray) -> Mahalanobis:
        self._cov = LedoitWolf().fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return self._cov.mahalanobis(x)  # already squared


class GMM:
    def __init__(self, n_components: int, covariance_type: str, reg_covar: float, seed: int):
        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=seed,
        )

    def fit(self, x: np.ndarray) -> GMM:
        self._gmm.fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return -self._gmm.score_samples(x)

    @property
    def mixture(self) -> GaussianMixture:
        """The fitted scikit-learn model, for int8 quantization (Phase 4)."""
        return self._gmm


class IForest:
    def __init__(self, n_estimators: int, max_samples: int | str, seed: int):
        self._forest = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, random_state=seed
        )

    def fit(self, x: np.ndarray) -> IForest:
        self._forest.fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return -self._forest.score_samples(x)


def build_detector(model_cfg: dict[str, Any], seed: int) -> Detector:
    """A fresh, unfitted detector from the config's ``model`` section."""
    kind = model_cfg["type"]
    if kind == "mahalanobis":
        return Mahalanobis()
    if kind == "gmm":
        return GMM(
            n_components=int(model_cfg["n_components"]),
            covariance_type=str(model_cfg["covariance_type"]),
            reg_covar=float(model_cfg["reg_covar"]),
            seed=seed,
        )
    if kind == "isolation_forest":
        return IForest(
            n_estimators=int(model_cfg["n_estimators"]),
            max_samples=model_cfg["max_samples"],
            seed=seed,
        )
    raise ValueError(f"unknown classic model type: {kind}")


def fit_and_score(
    model_cfg: dict[str, Any],
    seed: int,
    x_train: np.ndarray,
    train_ids: list[str],
    x_test: np.ndarray,
    test_ids: list[str],
) -> np.ndarray:
    """Fit on normal training vectors and score the test vectors.

    ``fit_per: machine_type`` fits one detector on all IDs pooled, as the autoencoder
    was trained. ``fit_per: machine_id`` fits one per ID and scores each test clip with
    its own ID's detector; the ID is in every DCASE file name, and the hardware stage
    monitors a single machine.
    """
    fit_per = model_cfg["fit_per"]
    if fit_per == "machine_type":
        return build_detector(model_cfg, seed).fit(x_train).score(x_test)
    if fit_per != "machine_id":
        raise ValueError(f"fit_per must be machine_type or machine_id, got {fit_per}")

    train_ids_arr, test_ids_arr = np.asarray(train_ids), np.asarray(test_ids)
    missing = set(test_ids_arr) - set(train_ids_arr)
    if missing:
        raise ValueError(f"test clips for machine IDs with no training clips: {sorted(missing)}")
    scores = np.empty(len(x_test), dtype=np.float64)
    for machine_id in sorted(set(test_ids_arr)):
        detector = build_detector(model_cfg, seed).fit(x_train[train_ids_arr == machine_id])
        rows = test_ids_arr == machine_id
        scores[rows] = detector.score(x_test[rows])
    return scores
