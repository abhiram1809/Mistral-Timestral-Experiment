"""Download an auditable Yahoo daily-bar snapshot and pin the model."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "assets": []}
    for asset in cfg["assets"]:
        ticker = asset["ticker"]
        path = ROOT / "data" / f"{ticker}.csv"
        if path.exists():
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            frame = yf.Ticker(ticker).history(
                start=cfg["data_start"], end=cfg["as_of_exclusive"],
                interval="1d", auto_adjust=False, actions=True, repair=False,
            )
            if frame.empty:
                raise RuntimeError(f"No Yahoo data: {ticker}")
            frame.index = pd.to_datetime(frame.index.date)
            frame.index.name = "Date"
            frame.to_csv(path)
        assert frame.index.is_monotonic_increasing and frame.index.is_unique
        cutoff_removed = int((frame.index >= pd.Timestamp(cfg["as_of_exclusive"])).sum())
        frame = frame.loc[frame.index < pd.Timestamp(cfg["as_of_exclusive"])]
        assert frame.index.max() < pd.Timestamp(cfg["as_of_exclusive"])
        valid = frame["Adj Close"].notna() & (frame["Adj Close"] > 0)
        clean = frame.loc[valid]
        if len(clean) < 1300:
            raise RuntimeError(f"Insufficient observations: {ticker}, {len(clean)}")
        returns = np.log(clean["Adj Close"]).diff()
        item = {
            **asset, "observations": len(clean), "removed_rows": int((~valid).sum()),
            "cutoff_rows_excluded": cutoff_removed,
            "first": str(clean.index.min().date()), "last": str(clean.index.max().date()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "max_abs_log_return": float(returns.abs().max()),
            "zero_return_fraction": float((returns == 0).mean()),
            "open_equals_close_fraction": float((clean["Open"] == clean["Close"]).mean()),
        }
        manifest["assets"].append(item)
        print(json.dumps(item), flush=True)
    info = HfApi().model_info(cfg["model_id"])
    manifest["model_revision"] = info.sha
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Downloading {cfg['model_id']} at {info.sha}", flush=True)
    snapshot_download(cfg["model_id"], revision=info.sha,
                      allow_patterns=["config.json", "model.safetensors", "LICENSE", "README.md"])
    print("Data and checkpoint ready", flush=True)


if __name__ == "__main__":
    main()
