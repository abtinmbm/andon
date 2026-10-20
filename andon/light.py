"""The andon light: SPC-style limits on the anomaly score, and a green/yellow/red state machine.

Limits are fitted the way a control chart is: score a stretch of normal running, take
the mean and standard deviation, and put the warning limit at ``mean + 2 sigma`` and
the control limit at ``mean + 3 sigma``. The fit can also be done on log scores, which
are less skewed, but for the ID classifier that backfires: windows it is very sure of
score near 0, their logs form a long left tail, and the inflated sigma pushes the
control limit above almost every fault (see DECISIONS.md, Phase 5).

The light never acts on one window. Yellow needs a run of windows above the warning
limit to come on and a run below it to go off, so a score hovering at the limit does
not flicker. Red needs a run above the control limit, and once on it stays on until
someone resets it, as a stopped line would.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Floor before the log: a window the model is certain about can score exactly 0.
_SCORE_FLOOR = 1e-12


class Light(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class Limits:
    """Warning and control limits, on the scale given by ``log_scale``."""

    center: float
    sigma: float
    warn: float
    control: float
    log_scale: bool

    def transform(self, scores: np.ndarray | float) -> np.ndarray:
        """Raw scores on the scale the limits live on."""
        s = np.asarray(scores, dtype=np.float64)
        return np.log(np.maximum(s, _SCORE_FLOOR)) if self.log_scale else s


def fit_limits(
    normal_scores: np.ndarray,
    warn_sigma: float = 2.0,
    control_sigma: float = 3.0,
    log_scale: bool = False,
) -> Limits:
    """Fit warning and control limits on window scores of normal audio only."""
    if not 0 < warn_sigma < control_sigma:
        raise ValueError(f"need 0 < warn_sigma < control_sigma, got {warn_sigma}, {control_sigma}")
    s = np.asarray(normal_scores, dtype=np.float64).ravel()
    if s.size < 2:
        raise ValueError("need at least two normal window scores to fit limits")
    if log_scale:
        s = np.log(np.maximum(s, _SCORE_FLOOR))
    center, sigma = float(s.mean()), float(s.std(ddof=1))
    return Limits(
        center=center,
        sigma=sigma,
        warn=center + warn_sigma * sigma,
        control=center + control_sigma * sigma,
        log_scale=log_scale,
    )


@dataclass(frozen=True)
class LightSettings:
    """How many windows in a row each change of state needs."""

    warn_on: int = 2  # above the warning limit: green -> yellow
    warn_off: int = 3  # at or below the warning limit: yellow -> green
    red_hold: int = 3  # above the control limit: -> red, latched

    def __post_init__(self) -> None:
        if min(self.warn_on, self.warn_off, self.red_hold) < 1:
            raise ValueError(f"every run length must be at least 1, got {self}")


class AndonLight:
    """Feed it one window score at a time; it returns the light after that window."""

    def __init__(self, limits: Limits, settings: LightSettings | None = None) -> None:
        self.limits = limits
        self.settings = settings or LightSettings()
        self.reset()

    def reset(self) -> None:
        """Back to green with every run count cleared, as an operator's reset would."""
        self.state = Light.GREEN
        self._above_warn = 0
        self._below_warn = 0
        self._above_control = 0

    def update(self, score: float) -> Light:
        if self.state is Light.RED:
            return self.state
        x = float(self.limits.transform(score))
        above_warn = x > self.limits.warn
        self._above_warn = self._above_warn + 1 if above_warn else 0
        self._below_warn = 0 if above_warn else self._below_warn + 1
        self._above_control = self._above_control + 1 if x > self.limits.control else 0

        st = self.settings
        if self._above_control >= st.red_hold:
            self.state = Light.RED
        elif self.state is Light.GREEN and self._above_warn >= st.warn_on:
            self.state = Light.YELLOW
        elif self.state is Light.YELLOW and self._below_warn >= st.warn_off:
            self.state = Light.GREEN
        return self.state

    def run(self, scores: np.ndarray) -> list[Light]:
        """The light after each score in turn (continues from the current state)."""
        return [self.update(float(s)) for s in np.asarray(scores).ravel()]
