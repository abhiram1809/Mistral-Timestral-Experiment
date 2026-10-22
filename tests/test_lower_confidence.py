"""Behavioral checks for sizing from frozen, cost-adjusted probabilities."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_lower_confidence import choose_exposure, replay_case, FIXED_COLUMNS


def test_each_exposure_must_pass_its_own_gate_and_flat_forecasts_stay_cash():
    p = [[.34, .33, .32], [.36, .39, .38], [.43, .42, .41], [.48, .47, .46], [.9, .8, .7]]
    np.testing.assert_array_equal(choose_exposure(p, [1, -1, 1, -1, 0], [.35, .40, .45]), [0, 1, 2, 3, 0])


def test_exact_gate_is_inclusive_and_own_cost_can_prevent_higher_exposure():
    p = [[.45, .44, .449], [.45, .44, .45]]
    np.testing.assert_array_equal(choose_exposure(p, [1, 1], [.35, .40, .45]), [2, 3])


def test_future_outcome_cannot_change_position_selection():
    row = {c: 0. for c in FIXED_COLUMNS}
    row.update(ticker="EXAMPLE", interval="1d", opportunity=1,
        entry_bar="2026-01-01", exit_bar="2026-01-02", entry_close=100., exit_close=101.,
        direction=1., calibrated_profit_confidence_1x=.48,
        calibrated_profit_confidence_2x=.47, calibrated_profit_confidence_3x=.46,
        holding_years=1 / 365.25)
    asset = {"ticker": "EXAMPLE", "label": "Example", "currency": "USD", "cost_bps": 5}
    profitable, _ = replay_case(pd.DataFrame([row]), asset, "lower")
    row["exit_close"] = 99.
    losing, _ = replay_case(pd.DataFrame([row]), asset, "lower")
    assert profitable.leverage.iloc[0] == losing.leverage.iloc[0] == 3
    assert profitable.net_return.iloc[0] > 0 > losing.net_return.iloc[0]


def test_invalid_probability_rejected_before_sizing():
    with pytest.raises(ValueError, match="Probabilities must be finite"):
        choose_exposure([[.5, np.nan, .4]], [1], [.35, .40, .45])


def test_sub_percent_gates_are_probabilities_not_whole_percentages():
    p = [[.0033, .0033, .0033], [.012, .012, .012], [.06, .059, .058], [.0033, .0033, .0033]]
    np.testing.assert_array_equal(choose_exposure(p, [1, 1, -1, 0], [.001, .01, .05]), [1, 2, 3, 0])
    np.testing.assert_array_equal(choose_exposure(p, [1, 1, -1, 0], [.001, .002, .003]), [3, 3, 3, 0])
