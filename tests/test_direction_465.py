"""Confidence event, embargo and exposure policy without model inference."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from direction_465 import DirectionCoordinator


def synthetic_coordinator():
    c = DirectionCoordinator.__new__(DirectionCoordinator)
    key = "TEST_1d"
    c.frames = {key: pd.DataFrame(
        {"Close": np.full(600, 100.)},
        index=pd.date_range("2020-01-01", periods=600, tz="UTC"))}
    c.info = {key: {}}
    for i in range(249, 551):
        c.info[key][i] = {
            "entry": 100., "exit": 101. if i < 449 else 99.,
            "direction": 1., "direction_probability": .75,
            "raw": np.full(3, .05), "returns": np.array([-.01, -.02, -.03]),
            "fees": np.array([.02, .04, .06]), "funding": np.zeros(3),
            "years": 1 / 365.25,
        }
    c.quantiles = {key: {550: np.linspace(90., 112., 21)}}
    c.ensure = lambda item, origins: None
    return c, {"ticker": "TEST", "interval": "1d"}


def test_direction_gates_do_not_confuse_correct_direction_with_net_profit():
    c, item = synthetic_coordinator()
    row = c.score(item, [550], 1).iloc[0]
    assert row.direction_confidence == 201 / 302
    assert row.profit_confidence_1x == 1 / 302
    assert row.executed and row.leverage == 2
    assert row.net_return < 0


def test_current_outcome_and_embargoed_outcome_cannot_change_confidence():
    c, item = synthetic_coordinator()
    before = c.score(item, [550], 1).iloc[0]
    c.info["TEST_1d"][549]["exit"] = 1000.
    c.info["TEST_1d"][550]["exit"] = 1000.
    after = c.score(item, [550], 1).iloc[0]
    assert after.direction_confidence == before.direction_confidence
    assert after.leverage == before.leverage
    assert after.calibration_latest_exit < after.entry_bar


def test_unchanged_realized_prices_are_not_counted_as_directional_successes():
    c, item = synthetic_coordinator()
    for value in c.info["TEST_1d"].values():
        value["exit"] = value["entry"]
    row = c.score(item, [550], 1).iloc[0]
    assert row.calibration_wins == 0
    assert row.direction_confidence == 1 / 302
    assert not row.executed
