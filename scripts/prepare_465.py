"""Expand Yahoo history without modifying any preceding experiment's snapshot."""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from confidence_200 import ROOT, ASSETS, INTERVALS, bars_for

OUT = ROOT / "results/executed-465"
DATA = ROOT / "data/executed-465"
CUTOFF = pd.Timestamp("2026-10-20", tz="UTC")


def main():
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    prior_path = OUT / "source_manifest.json"
    prior = json.loads(prior_path.read_text())["cases"] if prior_path.exists() else []
    retry_keys = {(c["ticker"], c["interval"]) for c in prior if c.get("fallback_error")}
    manifest = {"cutoff_exclusive": str(CUTOFF), "retrieved_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    for interval in INTERVALS:
        for asset in ASSETS:
            ticker = asset["ticker"]; path = DATA / f"{ticker}_{interval}.csv"
            request = "saved minute snapshot" if interval == "1m" else "Yahoo maximum daily history" if interval == "1d" else "Yahoo two-year hourly history"
            fallback = None
            retry = (ticker, interval) in retry_keys
            if path.exists() and not retry:
                bars = pd.read_csv(path, index_col=0); bars.index = pd.to_datetime(bars.index, utc=True)
            elif interval == "1m":
                bars, _ = bars_for(ticker, interval)
            else:
                try:
                    window = {"period": "max"} if interval == "1d" else {"start": "2024-10-23"}
                    raw = yf.Ticker(ticker).history(**window, end="2026-10-20", interval=interval,
                        auto_adjust=False, actions=True, repair=False)
                    if raw.empty: raise RuntimeError("Yahoo returned no bars")
                    if interval == "1d":
                        raw.index = pd.to_datetime(raw.index.date, utc=True)
                        factor = raw["Adj Close"] / raw["Close"]
                        bars = raw[["Open", "High", "Low", "Close"]].multiply(factor, axis=0)
                        bars["Close"] = raw["Adj Close"]
                    else:
                        bars = raw[["Open", "High", "Low", "Close"]].copy()
                        bars.index = pd.to_datetime(bars.index, utc=True)
                except Exception as exc:
                    fallback = str(exc); bars, _ = bars_for(ticker, interval)
                    bars.index = pd.to_datetime(bars.index, utc=True)
            bars = bars.loc[bars.index < CUTOFF].sort_index()
            bars = bars.loc[np.isfinite(bars[["Open", "Close"]]).all(axis=1) & (bars.Open > 0) & (bars.Close > 0)]
            assert bars.index.is_unique and len(bars) >= 551
            if not path.exists() or retry: bars.to_csv(path)
            item = {**asset, "interval": interval, "source": str(path.relative_to(ROOT)),
                "request": request, "fallback_error": fallback, "bars": len(bars),
                "first": str(bars.index[0]), "last": str(bars.index[-1]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest["cases"].append(item)
            (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
            print(json.dumps(item), flush=True)


if __name__ == "__main__": main()
