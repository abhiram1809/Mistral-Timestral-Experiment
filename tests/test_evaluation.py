import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import Calibrator, probability_up, simulate, sizes
from infer import split_origins


def test_embargo_all_calibration_labels_known_before_first_test_origin():
    cal, test = split_origins(2000)
    assert len(cal) == 200 and len(test) == 500
    assert cal[-1]+2 < test[0]
    assert test[-1]+2 == 1999


def test_cdf_interpolation():
    p = probability_up(np.array([[-2., 0., 2.], [1., 2., 3.], [-3., -2., -1.]]),
                       np.array([.1, .5, .9]))
    np.testing.assert_allclose(p, [.5, .9, .1])


def test_entry_exit_reversal_and_drift_costs():
    net, cost, turn = simulate([1, -1], [.1, -.1], [1, 1], .001, 0, 0)
    assert net[0] == pytest.approx(.099)
    drift = 1.1/1.099
    assert turn[1] == pytest.approx(1+drift+.9)
    assert net[1] == pytest.approx(.1-.001*(1+drift+.9))
    assert np.all(cost >= 0)


def test_no_position_no_profit_or_costs():
    net, cost, turn = simulate([0, 0], [.1, -.5], [1, 3], .01)
    np.testing.assert_array_equal(net, 0.)
    np.testing.assert_array_equal(cost, 0.)
    np.testing.assert_array_equal(turn, 0.)


def test_sizing_is_monotone_capped_and_symmetric():
    p = np.array([.5, .54, .55, .6, .65, .7, .9])
    w = sizes(p, np.full(len(p), .02), 3.)
    assert np.all(np.diff(w) >= 0) and max(w) == 3
    assert np.all(w[:3] == 0)
    np.testing.assert_allclose(sizes(1-p, np.full(len(p), .02), 3.), -w, atol=1e-12)


def test_calibration_is_frozen_for_later_predictions():
    p = np.linspace(.2, .8, 200)
    y = (np.arange(200) % 3 == 0).astype(int)
    calibrator = Calibrator().fit(p, y)
    prediction = calibrator.predict(np.array([.4]))
    calibrator.predict(np.array([.99, .01, .3]))
    np.testing.assert_array_equal(calibrator.predict(np.array([.4])), prediction)


def test_bankruptcy_cannot_resurrect_capital():
    net, _, _ = simulate([3., 3.], [-.5, 1.], [1, 1], 0.)
    assert np.prod(1+net) == 0


def test_passive_holding_has_only_entry_and_exit_cost():
    cost = .001
    r = np.array([.1, -.05, .2])
    net, _, turnover = simulate([1/(1+cost), 1., 1.], r, [1, 1, 1], cost, 0., 0.)
    assert turnover[1] == pytest.approx(0, abs=1e-12)
    assert np.prod(1+net) == pytest.approx(np.prod(1+r)*(1-cost)/(1+cost))
