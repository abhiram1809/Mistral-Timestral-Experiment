"""Fixed original gates; stop at 465 completed trades combined, or exhaust history.

Extend the history, fit the same calibrator on the first 300 valid forecast origins
after 250-bar warmup, and freeze it. Never refit in response to test P&L or scarcity.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster

from confidence_200 import (ROOT, ASSETS, INTERVALS, CONTEXT, INITIAL, NCAL, THRESHOLDS,
    ProfitCalibrator, bars_for, information, select_leverage, equity_stats)

OUT = ROOT / "results/executed-465"
TARGET = 465
FIRST_TEST = CONTEXT + NCAL
QUANTILE_LEVELS = np.load(ROOT / "results/confidence-200/forecasts_SPY_1d.npz")["levels"]


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp.json"); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def read_bars(item):
    path = ROOT / item["source"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    bars = pd.read_csv(path, index_col=0); bars.index = pd.to_datetime(bars.index, utc=True)
    assert bars.index.is_unique and bars.index.is_monotonic_increasing
    assert (bars.Close > 0).all() and np.isfinite(bars.Close).all()
    return bars


def get_calibrator(result):
    fit = ProfitCalibrator()
    for key, value in result["calibrator"].items(): setattr(fit, key, value)
    return fit


class CachedInference:
    def __init__(self, backend, threads):
        self.backend = backend; self.model = None; self.engine = None; self.lookups = {}
        torch.set_num_threads(threads); torch.set_num_interop_threads(1)

    def load(self):
        if self.model is not None: return
        revision = json.loads((ROOT / "data/manifest.json").read_text())["model_revision"]
        self.model = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", revision=revision, local_files_only=True)
        if self.backend == "vulkan":
            from vulkan_backend import offload
            self.engine = offload(self.model.model)

    def forecasts(self, item, bars, origins):
        key = f"{item['ticker']}_{item['interval']}"; path = OUT / f"forecasts_{key}.npz"
        if key not in self.lookups:
            lookup = {}
            if path.exists():
                old = np.load(path)
                assert str(old["source_sha256"]) == item["sha256"]
                lookup.update({int(o): q for o, q in zip(old["origins"], old["quantiles"])})
            # Reuse original 500 forecasts only after verifying the complete input context.
            previous = ROOT / "results/confidence-200" / f"forecasts_{key}.npz"
            if previous.exists():
                old_bars, _ = bars_for(item["ticker"], item["interval"])
                old_bars.index = pd.to_datetime(old_bars.index, utc=True)
                old = np.load(previous)
                for timestamp, q in zip(old["timestamps"], old["quantiles"]):
                    t = pd.Timestamp(str(timestamp)); t = t.tz_localize("UTC") if t.tz is None else t
                    a = bars.index.get_indexer([t])[0]; b = old_bars.index.get_indexer([t])[0]
                    if a >= CONTEXT-1 and b >= CONTEXT-1:
                        x = bars.Close.iloc[a-CONTEXT+1:a+1].to_numpy(dtype="float32")
                        y = old_bars.Close.iloc[b-CONTEXT+1:b+1].to_numpy(dtype="float32")
                        if np.array_equal(x, y): lookup.setdefault(int(a), q)
            self.lookups[key] = lookup
        lookup = self.lookups[key]; pending = [int(t) for t in origins if int(t) not in lookup]
        prices = bars.Close.to_numpy()
        for start in range(0, len(pending), 48):
            self.load(); ids = pending[start:start+48]
            x = np.stack([prices[t-CONTEXT+1:t+1] for t in ids]).astype("float32")
            q = self.model.forecast(x, prediction_length=1).quantiles[:, 0, 0].cpu().numpy()
            assert np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-6).all()
            lookup.update(dict(zip(ids, q)))
            keys = sorted(lookup); tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, origins=np.array(keys), quantiles=np.stack([lookup[k] for k in keys]),
                source_sha256=np.array(item["sha256"]))
            tmp.replace(path)
            print("FORECAST", key, min(start+48, len(pending)), "/", len(pending), flush=True)
        return np.stack([lookup[int(t)] for t in origins])

    def close(self):
        if self.engine: self.engine.close()


def calibrate(item, inference):
    key = f"{item['ticker']}_{item['interval']}"; output = OUT / f"calibration_{key}.json"
    if output.exists():
        assert json.loads(output.read_text())["source_sha256"] == item["sha256"]
        return
    with (OUT / f"calibration_{key}.lock").open("w") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        bars = read_bars(item); origins = np.arange(CONTEXT-1, CONTEXT+NCAL-1)
        q = inference.forecasts(item, bars, origins)
        levels = QUANTILE_LEVELS
        data = information(bars, origins, q, levels, item["cost_bps"] / 1e4)
        valid = data["direction"] != 0
        fit = ProfitCalibrator().fit(data["raw"][valid], data["returns"][valid] > 0)
        ceiling = float(fit.predict(.99))
        result = {"ticker": item["ticker"], "interval": item["interval"], "source_sha256": item["sha256"],
            "calibrator": fit.__dict__, "calibration_origins": 300, "nonflat_calibration_origins": int(sum(valid)),
            "calibration_first": str(bars.index[origins[0]]), "calibration_last_exit": str(bars.index[origins[-1]+1]),
            "test_first_entry": str(bars.index[FIRST_TEST]), "maximum_confidence": ceiling,
            "can_qualify": bool(ceiling >= THRESHOLDS[0]), "future_candidate_bars": len(bars)-FIRST_TEST-1}
        assert bars.index[origins[-1]+1] < bars.index[FIRST_TEST]
        pd.DataFrame({"origin": origins, "entry_bar": bars.index[origins].astype(str),
            "exit_bar": bars.index[origins+1].astype(str), "direction": data["direction"],
            **{f"raw_profit_{i+1}x": data["raw"][:, i] for i in range(3)},
            **{f"net_return_{i+1}x": data["returns"][:, i] for i in range(3)}}).to_csv(OUT / f"calibration_{key}.csv", index=False)
        atomic_json(output, result)
        print("CALIBRATED", key, "ceiling", ceiling, "can_trade", result["can_qualify"], flush=True)


def score_block(item, bars, origins, inference, fit):
    q = inference.forecasts(item, bars, origins)
    data = information(bars, origins, q, QUANTILE_LEVELS, item["cost_bps"] / 1e4)
    p = fit.predict(data["raw"]); chosen = select_leverage(p); chosen[data["direction"] == 0] = 0
    active = chosen > 0
    def selected(values):
        out = np.zeros(len(chosen)); out[active] = values[active, chosen[active].astype(int)-1]; return out
    return pd.DataFrame({"ticker": item["ticker"], "interval": item["interval"], "origin": origins,
        "entry_bar": bars.index[origins].astype(str), "exit_bar": bars.index[origins+1].astype(str),
        "entry_close": data["entry"], "exit_close": data["exit"], "forecast_median": q[:, 10],
        "direction": data["direction"], **{f"calibrated_profit_confidence_{i+1}x": p[:, i] for i in range(3)},
        "leverage": chosen, "executed": active, "selected_confidence": selected(p),
        "net_return": selected(data["returns"]), "fees": selected(data["fees"]),
        "funding": selected(data["funding"]), "holding_years": data["years"]})


def trim_at_execution_target(block, already_executed, target=TARGET):
    needed = target - already_executed
    if needed <= 0: return block.iloc[:0]
    locations = np.flatnonzero(block.executed.to_numpy(dtype=bool))
    return block.iloc[:locations[needed-1]+1] if len(locations) >= needed else block


def test_history(items, inference):
    maps = {}; bars = {}; events = []
    for rank, item in enumerate(items):
        key = f"{item['ticker']}_{item['interval']}"
        path = OUT / f"calibration_{key}.json"
        if not path.exists(): raise RuntimeError(f"Calibration unfinished: {key}")
        result = json.loads(path.read_text()); bars[key] = read_bars(item); maps[key] = get_calibrator(result)
        if result["can_qualify"]:
            ids = np.arange(FIRST_TEST, len(bars[key])-1)
            events.append(pd.DataFrame({"key": key, "rank": rank, "origin": ids, "exit": bars[key].index[ids+1]}))
    columns = ["ticker", "interval", "origin", "entry_bar", "exit_bar", "entry_close", "exit_close", "forecast_median",
        "direction", *[f"calibrated_profit_confidence_{i}x" for i in (1, 2, 3)], "leverage", "executed",
        "selected_confidence", "net_return", "fees", "funding", "holding_years"]
    if events:
        events = pd.concat(events).sort_values(["exit", "rank"], kind="stable").reset_index(drop=True)
    else: events = pd.DataFrame(columns=["key", "rank", "origin", "exit"])
    lookup = {f"{i['ticker']}_{i['interval']}": i for i in items}; blocks = []; executed = 0
    for start in range(0, len(events), 192):
        group = events.iloc[start:start+192]; pieces = []
        for key, subset in group.groupby("key", sort=False):
            part = score_block(lookup[key], bars[key], subset.origin.to_numpy(), inference, maps[key])
            part["event_rank"] = subset.index.to_numpy(); pieces.append(part)
        block = pd.concat(pieces).sort_values("event_rank").drop(columns="event_rank").reset_index(drop=True)
        block = trim_at_execution_target(block, executed)
        blocks.append(block); executed += int(block.executed.sum())
        completed = pd.concat(blocks, ignore_index=True)
        completed.to_csv(OUT / "evaluated_signals.csv", index=False)
        atomic_json(OUT / "progress.json", {"target": TARGET, "executed": executed,
            "signals_scored": len(completed), "last_exit": completed.exit_bar.iloc[-1], "target_reached": executed >= TARGET})
        print("PROGRESS", executed, "/", TARGET, "executions;", len(completed), "signals scored; last", completed.exit_bar.iloc[-1], flush=True)
        if executed >= TARGET: break
    ledger = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=columns)
    ledger["global_trade_number"] = np.where(ledger.executed.astype(bool), np.cumsum(ledger.executed.astype(bool)), 0)
    # Separate accounts compound in their own chronological order.
    ledger["net_equity"] = INITIAL
    for _, indexes in ledger.groupby(["ticker", "interval"], sort=False).groups.items():
        ledger.loc[indexes, "net_equity"] = INITIAL * np.cumprod(1+ledger.loc[indexes, "net_return"].to_numpy(dtype=float))
    ledger.to_csv(OUT / "evaluated_signals.csv", index=False)
    ledger.loc[ledger.executed.astype(bool)].to_csv(OUT / "executed_trades.csv", index=False)
    atomic_json(OUT / "completion.json", {"target": TARGET, "executed": executed, "target_reached": executed >= TARGET,
        "history_exhausted": executed < TARGET, "qualified_cases": sum(json.loads((OUT / f"calibration_{k}.json").read_text())["can_qualify"] for k in bars),
        "last_evaluated_exit": ledger.exit_bar.iloc[-1] if len(ledger) else None})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["calibrate", "test"], required=True)
    parser.add_argument("--backend", choices=["cpu", "vulkan"], default="vulkan")
    parser.add_argument("--threads", type=int, default=2); parser.add_argument("--intervals", nargs="*")
    args = parser.parse_args(); items = json.loads((OUT / "source_manifest.json").read_text())["cases"]
    config = json.loads((OUT / "configuration.json").read_text())
    if config.get("status") == "cancelled":
        raise RuntimeError("Earlier-fit exploration was cancelled: the user requires original frozen calibration. Use scripts/frozen_465.py.")
    if args.intervals: items = [i for i in items if i["interval"] in args.intervals]
    inference = CachedInference(args.backend, args.threads)
    try:
        if args.stage == "calibrate":
            # Start with daily history, then hourly and minute, without choosing fits by their returns.
            for interval in ["1d", "1h", "1m"]:
                for item in items:
                    if item["interval"] == interval: calibrate(item, inference)
        else: test_history(items, inference)
    finally: inference.close()


if __name__ == "__main__": main()
