"""Package the expanded Yahoo data and audit its effect on the frozen strategy."""
import base64
import hashlib
import json
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fetch_more_yahoo import ROOT, OUT, DATA
from confidence_200 import ProfitCalibrator
from replay_lower_confidence import formatted_table, NAMES


def main():
    manifest = json.loads((OUT / "manifest.json").read_text()); requests = json.loads((OUT / "requests.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["cases"]) == 9
    for path, sha in manifest["original_source_hashes"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha
    old_manifest = json.loads((ROOT / "results/executed-465/source_manifest.json").read_text())
    old = {(i["ticker"], i["interval"]): i for i in old_manifest["cases"]}
    (OUT / "datasets").mkdir(exist_ok=True)
    rows = []; frozen = []; missing = 0
    for item in manifest["cases"]:
        path = ROOT/item["source"]; assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        frame = pd.read_csv(path, index_col=0); frame.index = pd.to_datetime(frame.index, utc=True)
        assert len(frame) == item["complete_rows"] and frame.index.is_unique and frame.index.is_monotonic_increasing
        assert np.isfinite(frame[["Open", "High", "Low", "Close"]]).all().all() and (frame.Close > 0).all()
        previous = pd.read_csv(ROOT/old[item["ticker"], item["interval"]]["source"], index_col=0)
        previous.index = pd.to_datetime(previous.index, utc=True)
        missing_case = len(previous.index.difference(frame.index)); missing += missing_case
        assert len(frame.index.difference(previous.index)) == item["additional_unique_rows"]
        assert len(previous)+item["additional_unique_rows"]-missing_case == len(frame)
        export = OUT / "datasets" / path.name; shutil.copy2(path, export)
        item["export_csv"] = str(export.relative_to(OUT)); item["previous_rows_missing_from_fresh"] = missing_case
        saved = json.loads((ROOT/"results/confidence-200"/f"result_{item['ticker']}_{item['interval']}.json").read_text())
        fit = ProfitCalibrator()
        fit.a = saved["slope"]; fit.b = saved["intercept"]; fit.center = saved["calibration_logit_center"]
        fit.raw_min = saved["calibration_raw_min"]; fit.raw_max = saved["calibration_raw_max"]
        ceiling = float(fit.predict(.99))
        extra = item["additional_rows_after_original_test"]
        # No inference is needed if the frozen map cannot pass the minimum gate.
        assert not (extra and ceiling >= .55), "New gate-eligible data require model inference before declaring a trade count"
        frozen.append({"ticker": item["ticker"], "interval": item["interval"], "new_completed_intervals": extra,
            "frozen_maximum_probability": ceiling, "analytically_skipped": extra if ceiling < .55 else 0,
            "new_model_forecasts_required": 0, "calibration_refit": False})
        rows.append({"Instrument": item["label"], "Data": NAMES[item["interval"]],
            "Previous bars": item["previous_rows"], "Complete bars now": item["complete_rows"],
            "Additional timestamps": item["additional_unique_rows"], "First (UTC)": item["first"],
            "Last complete bar (UTC)": item["last"]})
    extra_forward = sum(r["new_completed_intervals"] for r in frozen)
    original_trades = pd.read_csv(ROOT/"results/confidence-200/all_executed_trades.csv")
    state = {"target_executed_trades": 465, "executed": len(original_trades), "new_executions": 0,
        "target_reached": False, "shortfall": 465-len(original_trades), "calibration_refit": False,
        "gates": [.55, .60, .70], "new_completed_intervals_checked": extra_forward,
        "new_intervals_skipped_by_frozen_ceiling": extra_forward, "new_gate_eligible_daily_bars": 0,
        "cases": frozen}
    (OUT / "frozen_strategy_check.json").write_text(json.dumps(state, indent=2))
    pd.DataFrame(frozen).to_csv(OUT / "frozen_strategy_check.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(manifest["cases"]).to_csv(OUT / "data_inventory.csv", index=False)
    original_trades.to_csv(OUT / "executed_trades.csv", index=False)
    total = manifest["total_complete_rows"]; added = manifest["additional_unique_rows"]
    additions_by_interval = pd.DataFrame(manifest["cases"]).groupby("interval").additional_unique_rows.sum()
    successful = sum(r["status"] == "ok" for r in requests)
    minute = pd.DataFrame(manifest["cases"]).query("interval == '1m'")
    plt.rcParams.update({"figure.facecolor": "#0b1220", "axes.facecolor": "#111c2f", "savefig.facecolor": "#0b1220",
        "text.color": "#e2e8f0", "axes.labelcolor": "#cbd5e1", "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
        "axes.edgecolor": "#334155", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True); x = np.arange(3)
    ax.bar(x-.19, minute.previous_rows, .38, label="Previous minute snapshot", color="#64748b")
    ax.bar(x+.19, minute.complete_rows, .38, label="Expanded minute snapshot", color="#38bdf8")
    ax.set_xticks(x, minute.label); ax.set(ylabel="Complete minute bars", title="More Yahoo minute history from six-day request chunks")
    ax.legend(frameon=False); fig.savefig(OUT/"minute_data.png", dpi=155); plt.close(fig)
    img = base64.b64encode((OUT/"minute_data.png").read_bytes()).decode()
    asof = pd.Timestamp(manifest["as_of_utc"]); local = asof.tz_convert("Asia/Kolkata").strftime("%d %B %Y, %H:%M:%S IST")
    links = "".join(f'<li><a href="{i["export_csv"]}">{i["label"]} · {NAMES[i["interval"]]} CSV</a> · {i["complete_rows"]:,} complete bars</li>' for i in manifest["cases"])
    doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yahoo Finance · expanded data · 21 October 2026</title><style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1380px;margin:auto;padding:40px 28px 65px}}h1{{font-size:clamp(30px,4vw,46px);line-height:1.2}}h2{{font-size:25px}}p{{max-width:1080px}}section{{margin:30px 0}}.eyebrow{{color:#38bdf8;text-transform:uppercase;letter-spacing:.1em;font-weight:700;font-size:13px}}.lead{{font-size:20px}}.box{{background:#111c2f;border:1px solid #334155;border-radius:14px;padding:20px 26px}}.scroll{{overflow-x:auto;border:1px solid #334155;border-radius:10px;margin:16px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}}th,td{{text-align:right;padding:11px 13px;border-bottom:1px solid #25334b}}th:first-child,td:first-child{{text-align:left}}th{{color:#93c5fd;background:#111c2f}}img{{width:100%;height:auto;border-radius:12px}}a{{color:#7dd3fc}}.muted{{color:#94a3b8;font-size:14px}}li{{margin:7px 0}}@media(max-width:760px){{main{{padding:26px 16px}}}}</style></head><body><main>
<div class="eyebrow">Yahoo Finance API · expanded snapshot · frozen calibration retained</div>
<h1>{total:,} complete bars.<br>{added:,} additional timestamps.</h1>
<p class="lead">Retrieved {local}. All {successful}/{len(requests)} requests succeeded.
The expansion adds {int(additions_by_interval['1m']):,} minute bars and {int(additions_by_interval['1h']):,} hourly bars compared with the preceding extended snapshot.</p>
<section><h2>What was downloaded</h2>{formatted_table(pd.DataFrame(rows))}
<img src="data:image/png;base64,{img}" alt="Previous and expanded complete minute bar counts for Reliance, SPY and EUR/USD">
<p>Minute data were requested across the last 29 days in five six-day-or-shorter chunks per instrument.
Hourly requests cover 729 days; daily requests use Yahoo’s maximum history. Every source response, exchange-session metadata,
request window, cleaned dataset and file hash is retained. The <a href="https://github.com/ranaroussi/yfinance/blob/main/yfinance/scrapers/history.py">yfinance history implementation</a>
documents the request handling; the actual returned dates and counts above determine this snapshot’s coverage.</p></section>
<section class="box"><h2>Effect on the 465-trade target</h2>
<p>The original 55% / 60% / 70% gates and all original calibration parameters remain unchanged.
There are {extra_forward:,} newly completed intervals after the original test endpoints. All belong to cases whose frozen confidence
ceiling is below 55%, so they are rejected analytically without unnecessary model inference. SPY daily, the only case able to pass 55%,
has no new completed daily candle in this snapshot.</p>
<p><strong>The trade count remains {len(original_trades)} of 465, with {465-len(original_trades)} still missing.</strong>
Older bars before the calibration dates cannot be presented as new causal trades using those later frozen fits.</p>
<p>The October 20 SPY daily candle was still in progress at retrieval and is retained only in the raw response, not the completed-bar dataset.
No calibration was fitted or updated, no forecast was substituted, and no additional trade was invented.</p></section>
<section><h2>Data quality and provenance</h2>
<p>Unfinished candles, duplicate timestamps and rows with missing or nonpositive opening/closing quotes were removed from the completed-bar datasets.
The raw responses remain available for audit. {missing} timestamps from the preceding snapshot were not returned as valid rows in this fresh response;
all preceding snapshot files remain untouched. “Additional timestamps” counts set differences, so it is not always the simple row-count increase.</p>
<p class="muted">Daily processed closes are Yahoo’s adjusted closes. Fresh responses can revise historical prices or rounding.
Existing simulated P&amp;L remains tied to its original saved inputs, not silently recomputed from revised quotes.</p></section>
<section><h2>Download the data</h2><p><a href="Yahoo-expanded-data.zip">Complete ZIP: cleaned CSVs, raw responses, metadata and audit files</a>.</p>
<ul>{links}</ul><p><a href="data_inventory.csv">Inventory CSV</a> · <a href="manifest.json">manifest and hashes</a> · <a href="requests.json">21 API request records</a> ·
<a href="frozen_strategy_check.json">frozen-strategy check</a> · <a href="audit.json">audit</a> · <a href="../frozen-465/report.html">preceding frozen-calibration report</a></p></section>
</main></body></html>'''
    (OUT/"report.html").write_text(doc)
    audit = {"status": "passed", "complete_bars": total, "additional_unique_timestamps": added,
        "successful_requests": successful, "requests": len(requests), "original_sources_unchanged": True,
        "calibration_refit": False, "executed_trades": len(original_trades), "target_reached": False,
        "checks": ["nine nonempty sorted unique datasets", "source and exported file hashes",
            "raw request windows and exchange metadata retained", "incomplete candles excluded",
            "additional and missing timestamp counts reconcile", "original calibration hashes unchanged",
            "new intervals rejected only when mathematically below the gate", "no new gate-eligible data left unforecast"]}
    (OUT/"audit.json").write_text(json.dumps(audit, indent=2))
    with zipfile.ZipFile(OUT/"Yahoo-expanded-data.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DATA.rglob("*")):
            if path.is_file(): archive.write(path, "data/"+str(path.relative_to(DATA)))
        for name in ["manifest.json", "requests.json", "data_inventory.csv", "frozen_strategy_check.json", "frozen_strategy_check.csv", "audit.json", "executed_trades.csv", "report.html"]:
            archive.write(OUT/name, "report/"+name)
        for path in sorted((OUT/"datasets").glob("*.csv")): archive.write(path, "report/datasets/"+path.name)
    print(json.dumps(audit, indent=2)); print("ZIP bytes", (OUT/"Yahoo-expanded-data.zip").stat().st_size)


if __name__ == "__main__": main()
