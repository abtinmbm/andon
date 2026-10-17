import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from andon.metrics import auc, partial_auc, pauc, roc_curve


def random_case(seed: int, n: int = 300, ties: bool = False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    y[:2] = [0, 1]  # both classes present
    score = rng.normal(size=n) + 0.8 * y
    if ties:
        score = np.round(score, 1)  # many tied scores across classes
    return y, score


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("ties", [False, True])
def test_auc_matches_sklearn(seed, ties):
    y, s = random_case(seed, ties=ties)
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("max_fpr", [0.05, 0.1, 0.3, 1.0])
def test_pauc_matches_sklearn_max_fpr(seed, ties, max_fpr):
    y, s = random_case(seed, ties=ties)
    expected = roc_auc_score(y, s, max_fpr=max_fpr)
    assert pauc(y, s, max_fpr) == pytest.approx(expected, abs=1e-12)


def test_hand_computed_toy_case():
    # Normals 0.1, 0.4; anomalies 0.35, 0.8. Of the 4 (anomaly, normal) pairs, 3 rank correctly.
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    assert auc(y, s) == pytest.approx(0.75)
    fpr, tpr = roc_curve(y, s)
    np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
    # Up to FPR 0.1 the curve sits at TPR 0.5: raw area 0.05.
    assert partial_auc(y, s, 0.1) == pytest.approx(0.05)
    # McClish: 0.5 * (1 + (0.05 - 0.005) / (0.1 - 0.005))
    assert pauc(y, s, 0.1) == pytest.approx(0.5 * (1 + 0.045 / 0.095))


def test_perfect_random_and_inverted_detectors():
    y = np.r_[np.zeros(50), np.ones(50)]
    perfect = np.r_[np.zeros(50), np.ones(50)]
    assert auc(y, perfect) == 1.0
    assert pauc(y, perfect) == pytest.approx(1.0)
    constant = np.zeros(100)  # every score tied: the diagonal
    assert auc(y, constant) == pytest.approx(0.5)
    assert pauc(y, constant) == pytest.approx(0.5)
    assert auc(y, -perfect) == 0.0


def test_ties_earn_half_credit():
    assert auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_rejects_single_class_and_bad_input():
    with pytest.raises(ValueError, match="at least one"):
        auc([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="shape mismatch"):
        auc([0, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="max_fpr"):
        pauc([0, 1], [0.1, 0.2], max_fpr=0.0)
