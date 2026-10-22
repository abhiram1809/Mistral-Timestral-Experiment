import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_more_yahoo import completed_mask


def test_live_us_daily_candle_is_not_a_completed_outcome():
    index = pd.DatetimeIndex(["2026-10-19", "2026-10-20", "2026-10-21"], tz="America/New_York")
    metadata = {"exchangeTimezoneName": "America/New_York", "currentTradingPeriod": {"regular": {
        "start": "2026-10-20T09:30:00-04:00", "end": "2026-10-20T16:00:00-04:00"}}}
    np.testing.assert_array_equal(completed_mask(index, "1d", metadata, pd.Timestamp("2026-10-20T18:58:00Z")), [True, False, False])


def test_completed_indian_session_can_be_included_on_its_date():
    index = pd.DatetimeIndex(["2026-10-19", "2026-10-20"], tz="Asia/Kolkata")
    metadata = {"exchangeTimezoneName": "Asia/Kolkata", "currentTradingPeriod": {"regular": {
        "start": "2026-10-20T09:15:00+05:30", "end": "2026-10-20T15:30:00+05:30"}}}
    np.testing.assert_array_equal(completed_mask(index, "1d", metadata, pd.Timestamp("2026-10-20T12:00:00Z")), [True, True])


def test_current_minute_is_excluded():
    index = pd.DatetimeIndex(["2026-10-20T18:57:00Z", "2026-10-20T18:58:00Z"])
    np.testing.assert_array_equal(completed_mask(index, "1m", {}, pd.Timestamp("2026-10-20T18:58:17Z")), [True, False])


def test_final_half_hour_bar_is_complete_at_session_close():
    index = pd.DatetimeIndex(["2026-10-20T19:30:00Z"])
    metadata = {"currentTradingPeriod": {"regular": {"start": "2026-10-20T13:30:00Z", "end": "2026-10-20T20:00:00Z"}}}
    assert not completed_mask(index, "1h", metadata, pd.Timestamp("2026-10-20T19:59:00Z"))[0]
    assert completed_mask(index, "1h", metadata, pd.Timestamp("2026-10-20T20:01:00Z"))[0]
