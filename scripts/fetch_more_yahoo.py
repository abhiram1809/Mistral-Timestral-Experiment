"""Retrieve a larger auditable Yahoo snapshot without touching frozen model fits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
LABEL = "2026-10-21"  # User's client date in Asia/Calcutta.
OUT = ROOT / "results" / f"more-yahoo-{LABEL}"
DATA = ROOT / "data" / f"more-yahoo-{LABEL}"
ASSETS = [("RELIANCE.NS", "Reliance", "INR"), ("SPY", "SPY", "USD"), ("EURUSD=X", "EUR/USD", "USD")]


def instant(value):
    if isinstance(value, (int, float)): return pd.Timestamp(value, unit="s", tz="UTC")
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def completed_mask(index, interval, metadata, asof):
    """Use exchange session metadata to reject unfinished daily/intraday bars."""
    current = metadata.get("currentTradingPeriod", {}).get("regular", {})
    start = instant(current["start"]) if current.get("start") is not None else None
    end = instant(current["end"]) if current.get("end") is not None else None
    if interval == "1d":
        timezone = metadata.get("exchangeTimezoneName", str(index.tz))
        local_dates = np.array(index.tz_convert(timezone).date)
        today = asof.tz_convert(timezone).date()
        mask = local_dates < today
        if start is not None and end is not None and end <= asof:
            session_day = start.tz_convert(timezone).date()
            mask |= (local_dates == session_day) & (local_dates <= today)
        return mask
    delta = pd.Timedelta(minutes=1) if interval == "1m" else pd.Timedelta(hours=1)
    times = index.tz_convert("UTC"); finishes = times + delta
    if start is not None and end is not None:
        capped = (times >= start) & (times < end) & (finishes > end)
        finishes = finishes.where(~capped, end)
    return np.asarray(finishes <= asof)


def main():
    OUT.mkdir(parents=True, exist_ok=True); (DATA / "raw").mkdir(parents=True, exist_ok=True)
    asof = pd.Timestamp.now(tz="UTC").floor("s")
    old_manifest = json.loads((ROOT / "results/executed-465/source_manifest.json").read_text())
    old = {(i["ticker"], i["interval"]): i for i in old_manifest["cases"]}
    pinned = json.loads((ROOT / "results/frozen-465/configuration.json").read_text())["original_source_hashes"]
    for path, sha in pinned.items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha
    manifest = {"client_date": LABEL, "as_of_utc": str(asof), "calibration_refit": False,
        "gates": [.55, .60, .70], "minute_request_days": 6, "minute_retention_probe_days": 29,
        "cases": [], "original_source_hashes": pinned}
    requests = []
    for ticker, label, currency in ASSETS:
        instrument = yf.Ticker(ticker)
        for interval in ["1m", "1h", "1d"]:
            key = f"{ticker}_{interval}"; frames = []
            if interval == "1m":
                edges = [asof-pd.Timedelta(days=29)]
                while edges[-1] < asof: edges.append(min(edges[-1]+pd.Timedelta(days=6), asof))
                windows = [{"start": int(a.timestamp()), "end": int(b.timestamp()), "period": None}
                           for a, b in zip(edges[:-1], edges[1:])]
            elif interval == "1h":
                windows = [{"start": int((asof-pd.Timedelta(days=729)).timestamp()), "end": int(asof.timestamp()), "period": None}]
            else: windows = [{"period": "max"}]
            metadata = {}
            for number, window in enumerate(windows):
                record = {"ticker": ticker, "interval": interval, "window": window}
                try:
                    frame = instrument.history(**window, interval=interval, auto_adjust=False,
                        actions=True, repair=False, keepna=True, timeout=25)
                    if frame.empty: raise RuntimeError(getattr(instrument._price_history, "_last_error", None) or "Yahoo returned no bars")
                    frame.to_csv(DATA / "raw" / f"{key}_part{number+1}.csv")
                    metadata = instrument.history_metadata
                    frames.append(frame); record["rows"] = len(frame); record["status"] = "ok"
                except Exception as exc:
                    record.update(status="error", error=str(exc), rows=0)
                requests.append(record)
                (OUT / "requests.json").write_text(json.dumps(requests, indent=2, default=str))
            if not frames:
                manifest["cases"].append({"ticker": ticker, "interval": interval, "status": "no_data"})
                continue
            raw = pd.concat(frames).sort_index(); duplicate_rows = int(raw.index.duplicated(keep="last").sum())
            raw = raw.loc[~raw.index.duplicated(keep="last")]
            if raw.index.tz is None: raw.index = raw.index.tz_localize(metadata.get("exchangeTimezoneName", "UTC"))
            mask = completed_mask(raw.index, interval, metadata, asof)
            valid = np.isfinite(raw[["Open", "Close"]]).all(axis=1).to_numpy() & (raw.Open.to_numpy()>0) & (raw.Close.to_numpy()>0)
            closed = raw.loc[mask & valid].copy()
            if interval == "1d":
                factor = closed["Adj Close"]/closed.Close
                bars = closed[["Open", "High", "Low", "Close"]].multiply(factor, axis=0)
                bars["Close"] = closed["Adj Close"]
                bars.index = pd.to_datetime(closed.index.date, utc=True)
            else:
                bars = closed[["Open", "High", "Low", "Close"]].copy(); bars.index = bars.index.tz_convert("UTC")
            bars.index.name = "Timestamp"
            assert bars.index.is_monotonic_increasing and bars.index.is_unique and len(bars)
            path = DATA / f"{key}.csv"; bars.to_csv(path)
            (DATA / f"metadata_{key}.json").write_text(json.dumps(metadata, indent=2, default=str))
            baseline = old[ticker, interval]
            previous = pd.read_csv(ROOT / baseline["source"], index_col=0)
            previous.index = pd.to_datetime(previous.index, utc=True)
            new_index = bars.index.difference(previous.index)
            original_test = pd.read_csv(ROOT / "results/confidence-200" / f"opportunities_{key}.csv")
            last_evaluated = pd.to_datetime(original_test.exit_bar, utc=True).max()
            new_forward = bars.index[bars.index > last_evaluated]
            item = {"ticker": ticker, "label": label, "currency": currency, "interval": interval, "status": "ok",
                "source": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "raw_unique_rows": len(raw), "complete_rows": len(bars), "partial_rows_excluded": int(sum(~mask)),
                "invalid_complete_rows_excluded": int(sum(mask & ~valid)), "duplicate_rows_removed": duplicate_rows,
                "first": str(bars.index[0]), "last": str(bars.index[-1]), "previous_rows": len(previous),
                "additional_unique_rows": len(new_index), "newer_complete_rows": int(sum(bars.index > previous.index.max())),
                "older_or_gap_rows_added": int(sum(new_index <= previous.index.max())),
                "additional_rows_after_original_test": len(new_forward),
                "last_original_test_bar": str(last_evaluated),
                "successful_requests": sum(r["status"]=="ok" for r in requests if r["ticker"]==ticker and r["interval"]==interval),
                "requested_windows": len(windows)}
            manifest["cases"].append(item)
            (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            print(json.dumps(item), flush=True)
    for path, sha in pinned.items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha
    manifest["original_sources_unchanged"] = True
    manifest["total_complete_rows"] = sum(i.get("complete_rows", 0) for i in manifest["cases"])
    manifest["additional_unique_rows"] = sum(i.get("additional_unique_rows", 0) for i in manifest["cases"])
    manifest["status"] = "complete" if len(manifest["cases"])==9 and all(i["status"]=="ok" for i in manifest["cases"]) else "partial"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    pd.DataFrame(manifest["cases"]).to_csv(OUT / "data_inventory.csv", index=False)
    print("SNAPSHOT", manifest["status"], manifest["total_complete_rows"], "complete bars;", manifest["additional_unique_rows"], "new unique rows", flush=True)


if __name__ == "__main__": main()
