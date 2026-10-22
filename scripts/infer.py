"""Run the actual pinned 209M checkpoint. All forecasts are cached/resumable."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster

ROOT = Path(__file__).resolve().parents[1]


def load_prices(ticker):
    frame = pd.read_csv(ROOT / "data" / f"{ticker}.csv", index_col=0, parse_dates=True)
    cfg = json.loads((ROOT / "config.json").read_text())
    frame = frame.loc[frame.index < pd.Timestamp(cfg["as_of_exclusive"])]
    return frame.loc[frame["Adj Close"].notna() & (frame["Adj Close"] > 0), "Adj Close"]


def split_origins(length, calibration=200, test=500, horizon=2):
    test_end = length - horizon  # exclusive; last origin + horizon == length-1
    test_start = test_end - test
    cal_end = test_start - horizon  # last calibration outcome < first test origin
    cal = np.arange(cal_end - calibration, cal_end)
    test_ids = np.arange(test_start, test_end)
    assert cal[-1] + horizon < test_ids[0]
    return cal, test_ids


def atomic_save(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def infer_job(model, series, origins, context, path, batch, scale=1.0):
    # Independent CPU/Vulkan workers can share a queue without duplicating a job.
    with path.with_suffix(".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{path.stem}: another worker owns this job", flush=True)
            return
        _infer_locked(model, series, origins, context, path, batch, scale)


def _infer_locked(model, series, origins, context, path, batch, scale):
    levels = np.asarray(model.quantile_levels)
    result = np.full((len(origins), 20, len(levels)), np.nan, dtype=np.float32)
    if path.exists():
        old = np.load(path)
        assert np.array_equal(old["origins"], origins)
        result = old["quantiles"]
    assert origins.min() + 1 >= context
    pending = np.flatnonzero(~np.isfinite(result).all(axis=(1, 2)))
    start = time.perf_counter()
    for offset in range(0, len(pending), batch):
        ids = pending[offset:offset + batch]
        inputs = np.stack([series[t-context+1:t+1] * scale for t in origins[ids]])
        output = model.forecast(inputs.astype(np.float32), prediction_length=20,
                                clip_nonnegative=(scale == 1.0))
        q = output.quantiles[:, 0].cpu().numpy() / scale
        assert np.isfinite(q).all(), "Non-finite model outputs"
        assert (np.diff(q, axis=-1) >= -1e-6).all(), "Crossed quantiles"
        result[ids] = q
        atomic_save(path, origins=origins, quantiles=result, levels=levels)
        if offset % (batch * 5) == 0 or offset + batch >= len(pending):
            done = min(offset + batch, len(pending))
            elapsed = time.perf_counter() - start
            print(f"{path.stem}: {done}/{len(pending)} new forecasts; "
                  f"{elapsed:.1f}s elapsed; {elapsed/done:.3f}s/origin", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend", choices=["cpu", "vulkan"], default="cpu")
    parser.add_argument("--contexts", type=int, nargs="*")
    parser.add_argument("--tickers", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    cfg = json.loads((ROOT / "config.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    model = TimestralForecaster.from_pretrained(
        cfg["model_id"], revision=manifest["model_revision"], local_files_only=True,
    )
    print("Loaded parameters:", sum(p.numel() for p in model.model.parameters()), flush=True)
    engine = None
    if args.backend == "vulkan":
        from vulkan_backend import offload
        engine = offload(model.model)
    for context in [cfg["primary_context"]] + [c for c in cfg["contexts"] if c != cfg["primary_context"]]:
        if args.contexts and context not in args.contexts:
            continue
        for asset in cfg["assets"]:
            ticker = asset["ticker"]
            if args.tickers and ticker not in args.tickers:
                continue
            prices = load_prices(ticker)
            returns = np.diff(np.log(prices.to_numpy()))
            cal, test = split_origins(len(returns), cfg["calibration_observations"],
                                      cfg["test_observations"], cfg["execution_horizon"])
            origins = np.r_[cal, test, len(returns)-1]
            infer_job(model, returns, origins, context,
                      ROOT / "results" / f"returns_{ticker}_{context}.npz", args.batch, scale=100.)
    for asset in cfg["assets"]:
        ticker = asset["ticker"]
        if args.tickers and ticker not in args.tickers:
            continue
        prices = load_prices(ticker).to_numpy()
        # Price diagnostics on the most recent 100 completed one-step forecasts.
        origins = np.r_[np.arange(len(prices)-101, len(prices)-1), len(prices)-1]
        infer_job(model, prices, origins, cfg["primary_context"],
                  ROOT / "results" / f"prices_{ticker}.npz", args.batch)
    print("All inference complete", flush=True)
    if engine:
        engine.close()


if __name__ == "__main__":
    main()
