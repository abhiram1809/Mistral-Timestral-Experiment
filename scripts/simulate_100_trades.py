"""Plain model-directed 100-trade simulations: one instrument × three markets × three bars."""
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/100-trades"
DATA = ROOT / "data/100-trades"
ASSETS = [
    {"ticker": "RELIANCE.NS", "market": "India", "label": "Reliance", "currency": "INR", "cost_bps": 10},
    {"ticker": "SPY", "market": "US", "label": "SPY", "currency": "USD", "cost_bps": 5},
    {"ticker": "EURUSD=X", "market": "FX", "label": "EUR/USD", "currency": "USD", "cost_bps": 2},
]
INTERVALS = ["1m", "1h", "1d"]
CONTEXT = 250
INITIAL = 100000.


def load_bars(ticker, interval):
    if interval == "1d":
        path = ROOT / "data" / f"{ticker}.csv"
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
        raw = raw.loc[raw.index < pd.Timestamp("2026-10-20")]
        factor = raw["Adj Close"] / raw["Close"]
        bars = raw[["Open", "High", "Low", "Close"]].multiply(factor, axis=0)
        bars["Close"] = raw["Adj Close"]
    else:
        path = DATA / f"{ticker}_{interval}.csv"
        bars = pd.read_csv(path, index_col=0)
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars = bars.loc[bars.index < pd.Timestamp("2026-10-20", tz="UTC")]
    before = len(bars)
    bars = bars.loc[np.isfinite(bars[["Open", "Close"]]).all(axis=1) &
                    (bars.Open > 0) & (bars.Close > 0)].sort_index()
    assert bars.index.is_unique
    assert len(bars) >= CONTEXT+100
    return bars, {"source": str(path.relative_to(ROOT)), "source_rows": before,
                  "usable_rows": len(bars), "removed_rows": before-len(bars),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def forecasts(model, ticker, interval, closes, origins):
    path = OUT / f"forecasts_{ticker}_{interval}.npz"
    q = np.full((len(origins), 21), np.nan, dtype=np.float32)
    if path.exists():
        old = np.load(path)
        old_map = {int(o): row for o, row in zip(old["origins"], old["quantiles"])}
        for i, origin in enumerate(origins):
            if int(origin) in old_map:
                q[i] = old_map[int(origin)]
    # Daily price forecasts from the prior run used these identical 250 closes.
    old_price = ROOT / "results" / f"prices_{ticker}.npz"
    if interval == "1d" and old_price.exists():
        old = np.load(old_price)
        old_map = {int(o): row[0] for o, row in zip(old["origins"], old["quantiles"])}
        for i, origin in enumerate(origins):
            if int(origin) in old_map:
                q[i] = old_map[int(origin)]
    pending = np.flatnonzero(~np.isfinite(q).all(axis=1))
    for offset in range(0, len(pending), 32):
        ids = pending[offset:offset+32]
        x = np.stack([closes[t-CONTEXT+1:t+1] for t in origins[ids]]).astype("float32")
        pred = model.forecast(x, prediction_length=1)
        q[ids] = pred.quantiles[:, 0, 0].cpu().numpy()
        print(ticker, interval, f"{min(offset+32,len(pending))}/{len(pending)} forecasts", flush=True)
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-6).all()
    cache_rows = old_map if path.exists() else {}
    cache_rows.update({int(o): row for o,row in zip(origins,q)})
    cache_origins=np.array(sorted(cache_rows))
    np.savez_compressed(path, origins=cache_origins, quantiles=np.stack([cache_rows[int(o)] for o in cache_origins]),
                        levels=np.asarray(model.quantile_levels))
    return q


def trade_returns(direction, opening, closing, cost_bps):
    ratio = np.asarray(closing)/np.asarray(opening)
    c = cost_bps/1e4
    gross = np.asarray(direction)*(ratio-1)
    # Entry plus exit fees; entry notional leaves cash for the entry fee.
    net = (gross-c*(1+ratio))/(1+c)
    return gross, net


def run_job(model, asset, interval):
    ticker = asset["ticker"]
    lockpath = OUT / f"{ticker}_{interval}.lock"
    with lockpath.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        result_path = OUT / f"result_{ticker}_{interval}.json"
        if result_path.exists():
            saved_result=json.loads(result_path.read_text())
            if saved_result.get("execution") == "close_to_next_close":
                return
        bars, source = load_bars(ticker, interval)
        closes = bars.Close.to_numpy()
        candidates = 100
        while True:
            origins = np.arange(len(bars)-candidates-1, len(bars)-1)
            assert origins.min()+1 >= CONTEXT
            q = forecasts(model, ticker, interval, closes, origins)
            directions = np.sign(q[:, 10].astype(float)-closes[origins])
            selected = np.flatnonzero(directions != 0)
            if len(selected) >= 100:
                selected = selected[-100:]
                break
            candidates += max(20,100-len(selected))
        ties_skipped = int(sum(directions == 0))
        origins = origins[selected]
        q = q[selected]
        directions = directions[selected]
        targets = origins+1
        opening = closes[origins]
        closing = closes[targets]
        gross, net = trade_returns(directions, opening, closing, asset["cost_bps"])
        assert len(net) == 100 and np.all(net > -1)
        equity = INITIAL*np.cumprod(1+net)
        gross_equity = INITIAL*np.cumprod(1+gross)
        peak = np.maximum.accumulate(np.r_[INITIAL,equity])[1:]
        ledger = pd.DataFrame({
            "trade": np.arange(1,101), "ticker": ticker, "interval": interval,
            "signal_bar": [str(bars.index[t]) for t in origins],
            "entry_bar": [str(bars.index[t]) for t in origins],
            "exit_bar": [str(bars.index[t]) for t in targets],
            "last_close": closes[origins], "forecast_close": q[:, 10],
            "forecast_q10": q[:, 2], "forecast_q90": q[:, 18],
            "direction": np.where(directions>0,"long","short"),
            "entry_close": opening, "exit_close": closing,
            "gross_return": gross, "net_return": net,
            "gross_equity": gross_equity, "net_equity": equity,
        })
        ledger.to_csv(OUT/f"trades_{ticker}_{interval}.csv", index=False)
        result = {**asset, "interval": interval, "context": CONTEXT, "trades": len(net),
            "execution": "close_to_next_close",
            "first_trade_bar": str(bars.index[origins[0]]), "last_trade_bar": str(bars.index[targets[-1]]),
            "initial_capital": INITIAL, "gross_ending": float(gross_equity[-1]),
            "net_ending": float(equity[-1]), "gross_return": float(gross_equity[-1]/INITIAL-1),
            "net_return": float(equity[-1]/INITIAL-1), "net_profit": float(equity[-1]-INITIAL),
            "gross_win_rate": float(np.mean(gross>0)), "net_win_rate": float(np.mean(net>0)),
            "long_trades": int(sum(directions>0)), "short_trades": int(sum(directions<0)),
            "unchanged_quote_trades": int(sum(opening==closing)),
            "yahoo_open_equals_close_bars": int(sum(bars.Open.to_numpy()[targets]==closing)),
            "exact_forecast_ties_skipped": ties_skipped,
            "max_drawdown": float(np.min(equity/peak-1)),
            "mae": float(np.mean(abs(closing-q[:, 10]))),
            "rmse": float(np.sqrt(np.mean((closing-q[:, 10])**2))),
            "source": source}
        result_path.write_text(json.dumps(result,indent=2))
        print("COMPLETE",ticker,interval,"100 trades",f"gross={result['gross_return']:.3%}",
              f"net={result['net_return']:.3%}",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=["cpu","vulkan"],default="cpu")
    p.add_argument("--threads",type=int,default=3)
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    revision=json.loads((ROOT/"data/manifest.json").read_text())["model_revision"]
    model=TimestralForecaster.from_pretrained("mistralai/Timestral-209M",revision=revision,local_files_only=True)
    engine=None
    if args.backend == "vulkan":
        from vulkan_backend import offload
        engine=offload(model.model)
    for interval in INTERVALS:
        for asset in ASSETS:
            run_job(model,asset,interval)
    if engine:engine.close()
    print("Worker finished",flush=True)


if __name__ == "__main__":
    main()
