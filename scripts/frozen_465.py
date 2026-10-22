"""Audit the 465-execution target with the exact original frozen calibration."""
import base64
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_200 import ROOT, ASSETS, INTERVALS, THRESHOLDS, INITIAL, pnl, select_leverage, ProfitCalibrator
from replay_lower_confidence import formatted_table, NAMES

OUT = ROOT / "results/frozen-465"
SOURCE = ROOT / "results/confidence-200"
TARGET = 465


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_manifest = json.loads((ROOT / "results/lower-confidence-200/configuration.json").read_text())
    for path, expected in original_manifest["source_hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    ledger = pd.read_csv(SOURCE / "all_1800_opportunities.csv")
    probabilities = ledger[[f"calibrated_profit_confidence_{i}x" for i in (1, 2, 3)]].to_numpy()
    expected = select_leverage(probabilities); expected[ledger.direction == 0] = 0
    np.testing.assert_array_equal(expected, ledger.leverage)
    extended = json.loads((ROOT / "results/executed-465/source_manifest.json").read_text())
    inventory = {(i["ticker"], i["interval"]): i for i in extended["cases"]}
    rows = []; bounds = []; availability = []; calibration_hashes = {}
    for asset in ASSETS:
        for interval in INTERVALS:
            ticker = asset["ticker"]; key = f"{ticker}_{interval}"
            path = SOURCE / f"result_{key}.json"; result = json.loads(path.read_text())
            calibration_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            fit = ProfitCalibrator()
            fit.a = result["slope"]; fit.b = result["intercept"]; fit.center = result["calibration_logit_center"]
            fit.raw_min = result["calibration_raw_min"]; fit.raw_max = result["calibration_raw_max"]
            ceiling = float(fit.predict(.99)); assert fit.a >= 0
            np.testing.assert_allclose(ceiling, result["maximum_attainable_confidence"], atol=1e-12)
            d = ledger[(ledger.ticker == ticker) & (ledger.interval == interval)]
            t = d[d.executed]; item = inventory[ticker, interval]
            bars_path = ROOT / item["source"]
            assert hashlib.sha256(bars_path.read_bytes()).hexdigest() == item["sha256"]
            bars = pd.read_csv(bars_path, index_col=0); bars.index = pd.to_datetime(bars.index, utc=True)
            last = pd.to_datetime(d.exit_bar, utc=True).max()
            extra = int(sum(bars.index > last))
            assert extra == 0, "New post-calibration data need forecasts; do not call an untested extension exhausted"
            assert pd.Timestamp(result["calibration_last_exit"]) < pd.Timestamp(result["first_entry"])
            net, fees, funding, _ = pnl(d.exit_close/d.entry_close, d.direction, d.leverage,
                                       asset["cost_bps"]/1e4, d.holding_years)
            np.testing.assert_allclose(net, d.net_return, atol=1e-12)
            np.testing.assert_allclose(fees, d.fees, atol=1e-12)
            np.testing.assert_allclose(funding, d.funding, atol=1e-12)
            np.testing.assert_allclose(INITIAL*np.cumprod(1+net), d.net_equity, rtol=1e-12)
            rows.append({"Instrument": asset["label"], "Bars": NAMES[interval], "Trades": len(t),
                "Net return": float(d.net_equity.iloc[-1]/INITIAL-1),
                "Final balance": f"{d.net_equity.iloc[-1]:,.2f} {asset['currency']}",
                "Frozen confidence ceiling": ceiling})
            bounds.append({"ticker": ticker, "interval": interval, "maximum_calibrated_probability": ceiling,
                "can_trade_1x": bool(ceiling >= .55), "can_trade_2x": bool(ceiling >= .60), "can_trade_3x": bool(ceiling >= .70)})
            availability.append({"Instrument": asset["label"], "Bars": NAMES[interval],
                "Expanded data begin": item["first"], "Frozen calibration last exit": result["calibration_last_exit"],
                "First eligible test entry": result["first_entry"], "Last completed data bar": item["last"],
                "Additional bars after saved test": extra})
    trades = ledger.loc[ledger.executed].copy()
    trades = trades.sort_values("exit_bar").reset_index(drop=True)
    trades.insert(0, "combined_trade_number", np.arange(1, len(trades)+1))
    count = len(trades); assert count == 1
    result = {"target_executed_trades": TARGET, "scope": "combined across nine cases", "executed": count,
        "target_reached": count >= TARGET, "shortfall": TARGET-count, "stop_reason": "available completed history exhausted",
        "gates": [.55, .60, .70], "calibration": "exact original frozen parameters", "calibration_refit": False,
        "new_forecasts_required": False, "new_closed_bars_after_previous_test": 0,
        "original_source_hashes": original_manifest["source_hashes"], "original_calibration_hashes": calibration_hashes,
        "expanded_history_audit_source": "results/executed-465/source_manifest.json"}
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(bounds).to_csv(OUT / "confidence_ceilings.csv", index=False)
    pd.DataFrame(availability).to_csv(OUT / "history_availability.csv", index=False)
    trades.to_csv(OUT / "executed_trades.csv", index=False)
    (OUT / "configuration.json").write_text(json.dumps(result, indent=2))
    audit = {"status": "passed", "target_reached": False, "executed": count, "shortfall": TARGET-count,
        "checks": ["all original source and calibration hashes unchanged", "original gates and leverage decisions reproduced",
            "unique completed executions, no repeated trades", "strict calibration chronology", "frozen confidence upper bounds",
            "fees, funding and separate account equity", "extended history adds no later completed bars",
            "earlier calibration exploration stopped and excluded"]}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    plt.rcParams.update({"figure.facecolor": "#0b1220", "axes.facecolor": "#111c2f", "savefig.facecolor": "#0b1220",
        "text.color": "#e2e8f0", "axes.labelcolor": "#cbd5e1", "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
        "axes.edgecolor": "#334155", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ceilings = np.array([r["Frozen confidence ceiling"] for r in rows])
    ax.bar(np.arange(9), ceilings*100, color=np.where(ceilings >= .55, "#38bdf8", "#64748b"))
    ax.axhline(55, color="#fb923c", ls="--", label="Minimum gate: 55%")
    ax.set_xticks(np.arange(9), [f"{r['Instrument']}\n{r['Bars']}" for r in rows])
    ax.set(ylabel="Maximum calibrated profit probability (%)", ylim=(0, 80), title="Frozen calibration · only SPY daily can pass the minimum gate")
    ax.legend(frameon=False); fig.savefig(OUT / "confidence_ceilings.png", dpi=155); plt.close(fig)
    img = base64.b64encode((OUT / "confidence_ceilings.png").read_bytes()).decode()
    filled = trades[["combined_trade_number", "ticker", "interval", "entry_bar", "exit_bar", "direction", "leverage", "selected_confidence", "net_return", "net_equity"]].copy()
    filled.columns = ["Trade", "Instrument", "Bars", "Entry", "Exit", "Direction", "Leverage", "Profit confidence", "Net return", "Ending balance"]
    filled["Direction"] = filled.Direction.map({1.: "Long", -1.: "Short"})
    doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Timestral · frozen calibration · 465-trade target</title><style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:40px 28px 65px}}h1{{font-size:clamp(30px,4vw,46px);line-height:1.2}}h2{{font-size:25px}}p{{max-width:1070px}}section{{margin:30px 0}}.eyebrow{{color:#38bdf8;text-transform:uppercase;letter-spacing:.1em;font-weight:700;font-size:13px}}.lead{{font-size:20px}}.box{{background:#111c2f;border:1px solid #334155;border-radius:14px;padding:20px 26px}}.scroll{{overflow-x:auto;border:1px solid #334155;border-radius:10px;margin:16px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}}th,td{{text-align:right;padding:11px 13px;border-bottom:1px solid #25334b}}th:first-child,td:first-child{{text-align:left}}th{{color:#93c5fd;background:#111c2f}}img{{width:100%;height:auto;border-radius:12px}}a{{color:#7dd3fc}}.muted{{color:#94a3b8;font-size:14px}}li{{margin:7px 0}}@media(max-width:760px){{main{{padding:26px 16px}}}}</style></head><body><main>
<div class="eyebrow">Original confidence gates · original frozen calibration</div>
<h1>465 trades requested.<br>1 available. 464 short.</h1>
<p class="lead">The target is <strong>465 executed trades combined</strong>, with no fixed opportunity-count stop.
The original 55% / 60% / 70% gates and all original calibration parameters are retained. The available completed history is exhausted;
the trade target has <strong>not</strong> been reached.</p>
<section class="box"><h2>Why simply extending history cannot produce 465 trades</h2>
<p>Eight frozen calibration maps cannot output a profit confidence of 55%, regardless of the model forecast.
Only SPY daily can pass 55%; its maximum is 55.26325%, so even it cannot qualify for 2× or 3×.</p>
<p>Yahoo supplied older daily data back to 1993 for SPY, 1996 for Reliance and 2003 for EUR/USD, plus roughly two years of hourly data.
These add no completed bars after the saved test’s final bar, October 19, 2026. Applying a calibration learned in 2025–2026 to those
older dates would use future outcomes. They therefore cannot count as valid new trades with this calibration frozen.</p>
<p>The 200 saved later bars per case are the full eligible history at the retained cutoff, <strong>not a 200-opportunity stop rule</strong>.
The earlier-calibration exploration was stopped when you chose to keep calibration frozen; none of its fits affects this report.</p></section>
<section><h2>Actual executed trade</h2>{formatted_table(filled, percentages=("Profit confidence", "Net return"), money=("Ending balance",))}
<p>The SPY account started at $100,000 and ended at <strong>$100,126.25</strong>: a net gain of <strong>$126.25 (+0.12625%)</strong>.
This is one completed trade, not two entry/exit fills and not a repeatedly reused observation.</p></section>
<section><h2>All nine accounts, measured by executions</h2>
{formatted_table(pd.DataFrame(rows), percentages=("Net return", "Frozen confidence ceiling"))}
<img src="data:image/png;base64,{img}" alt="Frozen confidence ceilings for nine instrument and timeframe cases, with a 55 percent minimum trading gate"></section>
<section><h2>History availability audit</h2>{formatted_table(pd.DataFrame(availability))}
<p class="muted">The study retains the original cutoff before October 20, 2026. Older snapshots were checked for available dates;
the original saved prices, forecasts, probabilities and trades remain the source of reported P&amp;L.</p></section>
<section><h2>Unchanged rules and costs</h2><ul>
<li>Confidence is estimated probability of positive next-bar P&amp;L after fees, financing and borrow. The median forecast chooses direction.</li>
<li>Use 3× at its own confidence ≥70%, otherwise 2× at ≥60%, otherwise 1× at ≥55%; otherwise skip. Flat forecasts always skip.</li>
<li>Official pinned Timestral-209M, 250-close context, 300 earlier calibration origins, one-bar embargo and frozen calibration maps.</li>
<li>Ideal observed-close to next-close fills. Fees per fill: Reliance 10 basis points, SPY 5, EUR/USD 2; annual financing 8% above 1× and short borrow 3%.</li>
<li>Separate 100,000 accounts in INR for Reliance and USD for SPY and EUR/USD. No pooled currency P&amp;L, taxes, FX carry or intrabar liquidation.</li>
<li>No resampling, duplicate trades, lower gates, calibration refits or live orders. Historical training overlap remains unknown.</li></ul></section>
<section class="box"><h2>Downloads</h2><ul>
<li><a href="executed_trades.csv">Executed trades</a> · <a href="summary.csv">account summary</a> · <a href="confidence_ceilings.csv">confidence ceilings</a></li>
<li><a href="history_availability.csv">expanded-history audit</a> · <a href="configuration.json">frozen configuration and hashes</a> · <a href="audit.json">audit</a></li>
<li><a href="../confidence-200/report.html">Original one-trade report</a></li></ul>
<p>Audit passed. The 465-execution target is explicitly reported as unmet.</p></section></main></body></html>'''
    (OUT / "report.html").write_text(doc)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
