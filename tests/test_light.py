import numpy as np
import pytest

from andon.light import AndonLight, Light, LightSettings, Limits, fit_limits

G, Y, R = Light.GREEN, Light.YELLOW, Light.RED
# Linear scale, so the test scores read directly: warning above 2, control above 3.
LIMITS = Limits(center=0.0, sigma=1.0, warn=2.0, control=3.0, log_scale=False)


def light(**settings) -> AndonLight:
    return AndonLight(LIMITS, LightSettings(**settings))


def test_fit_limits_linear_matches_mean_and_sample_std():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    lim = fit_limits(s, log_scale=False)
    assert lim.center == pytest.approx(2.5)
    assert lim.sigma == pytest.approx(np.std(s, ddof=1))
    assert lim.warn == pytest.approx(2.5 + 2 * lim.sigma)
    assert lim.control == pytest.approx(2.5 + 3 * lim.sigma)


def test_fit_limits_log_scale_fits_on_log_scores():
    rng = np.random.default_rng(0)
    s = np.exp(rng.normal(-3.0, 0.5, size=20_000))
    lim = fit_limits(s, log_scale=True)
    assert lim.center == pytest.approx(-3.0, abs=0.02)
    assert lim.sigma == pytest.approx(0.5, abs=0.02)
    # About 0.13% of normal windows exceed a 3-sigma limit on the log scale.
    assert np.mean(lim.transform(s) > lim.control) == pytest.approx(0.00135, abs=0.001)


def test_fit_limits_survives_zero_scores():
    lim = fit_limits(np.array([0.0, 1e-3, 1e-2, 1e-1]), log_scale=True)
    assert np.isfinite([lim.center, lim.sigma, lim.warn, lim.control]).all()


@pytest.mark.parametrize("bad", [(3.0, 2.0), (0.0, 3.0), (2.0, 2.0)])
def test_fit_limits_rejects_bad_sigmas(bad):
    with pytest.raises(ValueError):
        fit_limits(np.ones(5), warn_sigma=bad[0], control_sigma=bad[1])


def test_settings_reject_zero_runs():
    with pytest.raises(ValueError):
        LightSettings(red_hold=0)


def test_normal_scores_stay_green():
    assert light().run(np.zeros(50)) == [G] * 50


def test_single_spike_does_not_change_the_light():
    # One window above even the control limit is not enough for yellow or red.
    assert light().run([0, 5, 0, 0]) == [G, G, G, G]


def test_yellow_needs_warn_on_windows_in_a_row():
    assert light(warn_on=2).run([2.5, 0, 2.5, 2.5]) == [G, G, G, Y]


def test_yellow_clears_only_after_warn_off_windows_below():
    seq = [2.5, 2.5, 0, 0, 2.5, 0, 0, 0]
    assert light(warn_on=2, warn_off=3).run(seq) == [G, Y, Y, Y, Y, Y, Y, G]


def test_hovering_at_the_warning_limit_does_not_flicker():
    seq = [2.5, 2.5] + [1.9, 2.1] * 10
    states = light(warn_on=2, warn_off=3).run(seq)
    assert states[1:] == [Y] * (len(seq) - 1)


def test_red_needs_red_hold_windows_above_control():
    assert light(red_hold=3).run([3.5, 3.5, 0, 3.5, 3.5, 3.5]) == [G, Y, Y, Y, Y, R]


def test_red_latches_until_reset():
    lt = light(red_hold=2)
    assert lt.run([4, 4, 0, 0, 0, 0]) == [G, R, R, R, R, R]
    lt.reset()
    assert lt.state is G
    assert lt.run([0, 4]) == [G, G]  # run counts start again from zero after a reset


def test_red_can_come_straight_from_green():
    # With red_hold below warn_on, the control-limit run wins.
    assert light(warn_on=5, red_hold=2).run([4, 4]) == [G, R]


def test_log_scale_limits_are_applied_to_raw_scores():
    lim = Limits(center=0.0, sigma=1.0, warn=2.0, control=3.0, log_scale=True)
    lt = AndonLight(lim, LightSettings(red_hold=2))
    # log(30) = 3.4 is above control; 10 (log 2.3) is only above warn.
    assert lt.run([10, 30, 30]) == [G, Y, R]
