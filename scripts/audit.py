"""Verify saved forecasts, chronological labels and every simulated equity path."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from infer import ROOT, load_prices


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    counts = {"return_origins": 0, "price_origins": 0, "quantile_values": 0}
    for asset in manifest["assets"]:
        ticker = asset["ticker"]
        data = ROOT / "data" / f"{ticker}.csv"
        assert hashlib.sha256(data.read_bytes()).hexdigest() == asset["sha256"]
        assert load_prices(ticker).index.max() < pd.Timestamp(cfg["as_of_exclusive"])
        for kind, contexts in [("returns", cfg["contexts"]), ("prices", [None])]:
            for context in contexts:
                suffix = f"_{context}" if context else ""
                forecast = np.load(ROOT / "results" / f"{kind}_{ticker}{suffix}.npz")
                q = forecast["quantiles"]
                expected = 701 if kind == "returns" else 101
                assert q.shape == (expected, 20, 21), (ticker, context, q.shape)
                assert np.isfinite(q).all()
                assert (np.diff(q, axis=-1) >= -1e-6).all()
                counts["return_origins" if kind == "returns" else "price_origins"] += len(q)
                counts["quantile_values"] += q.size
    pred = pd.read_csv(ROOT / "results/test_predictions.csv")
    curves = pd.read_csv(ROOT / "results/equity_curves.csv")
    metrics = pd.read_csv(ROOT / "results/trading_metrics.csv")
    calibration = pd.read_csv(ROOT / "results/calibration.csv")
    assert len(pred) == 13500 and len(curves) == 121500 and len(metrics) == 486
    assert not pred.duplicated(["ticker", "context", "origin_date"]).any()
    assert (pd.to_datetime(pred.origin_date) < pd.to_datetime(pred.entry_date)).all()
    assert (pd.to_datetime(pred.entry_date) < pd.to_datetime(pred.exit_date)).all()
    assert (pd.to_datetime(calibration.calibration_last_outcome) <
            pd.to_datetime(calibration.test_first_origin)).all()
    assert (curves.weight.abs() <= cfg["maximum_leverage"] + 1e-10).all()
    assert (curves.cost_fraction >= 0).all()
    for (ticker, context), group in pred.groupby(["ticker", "context"]):
        assert len(group) == 500
        actual = np.log(load_prices(ticker)).diff()
        rederived = actual.reindex(pd.to_datetime(group.exit_date)).to_numpy()
        np.testing.assert_allclose(rederived, group.actual_log_return, atol=1e-12)
    merged = curves.merge(pred[["ticker", "context", "exit_date", "actual_log_return"]],
                          left_on=["ticker", "context", "date"],
                          right_on=["ticker", "context", "exit_date"], validate="many_to_one")
    np.testing.assert_allclose(merged.net_return,
        merged.weight*np.expm1(merged.actual_log_return)-merged.cost_fraction, atol=1e-12)
    for (ticker, context, strategy), group in curves.groupby(["ticker", "context", "strategy"]):
        equity = cfg["initial_capital"]*np.cumprod(1+group.net_return)
        np.testing.assert_allclose(equity, group.equity, rtol=1e-12)
        match = metrics[(metrics.ticker == ticker) & (metrics.context == context) &
                        (metrics.strategy == strategy) & (metrics.cost_multiplier == 1)]
        assert len(match) == 1
        np.testing.assert_allclose(equity.iloc[-1], match.ending_capital.iloc[0], rtol=1e-12)
    validation = json.loads((ROOT / "results/vulkan_validation.json").read_text())
    assert sum(row["origins"] for row in validation) == 36
    assert max(row["max_difference_divided_80_width"] for row in validation) < .001
    result = {"status": "passed", **counts, "test_predictions": len(pred),
              "daily_account_records": len(curves), "strategy_scenarios": len(metrics),
              "vulkan_validation_origins": 36,
              "checks": ["raw data hashes and cutoff", "finite ordered forecast quantiles",
                         "calibration embargo and delayed entries", "realized labels from source prices",
                         "position limits and costs", "daily P&L identity and compounded equity",
                         "final capital agrees with metrics", "Vulkan/CPU numerical agreement"]}
    (ROOT / "results/audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
