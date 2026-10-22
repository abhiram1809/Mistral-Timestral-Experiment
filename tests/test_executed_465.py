import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from executed_465 import trim_at_execution_target, get_calibrator, THRESHOLDS, QUANTILE_LEVELS


def test_stop_counts_executions_and_retains_prior_skips():
    signals = pd.DataFrame({"executed": [False, True, False, False, True, False, True]})
    result = trim_at_execution_target(signals, already_executed=463)
    assert len(result) == 5 and result.executed.sum() == 2


def test_shortfall_does_not_duplicate_trades_or_relax_the_target():
    signals = pd.DataFrame({"executed": [False, True, False]})
    pd.testing.assert_frame_equal(trim_at_execution_target(signals, 1), signals)
    assert trim_at_execution_target(signals, 465).empty


def test_serialized_calibrator_preserves_clamp_and_ceiling():
    result = {"calibrator": {"a": 1.2, "b": -.8, "raw_min": .1, "raw_max": .6,
                             "center": 0., "base": .3}}
    fit = get_calibrator(result)
    np.testing.assert_allclose(fit.predict([.6, .99, 1.]), fit.predict(.6))
    assert fit.predict(.99) < THRESHOLDS[0]
    np.testing.assert_array_equal(THRESHOLDS, [.55, .60, .70])
    assert len(QUANTILE_LEVELS) == 21 and QUANTILE_LEVELS[10] == .5
