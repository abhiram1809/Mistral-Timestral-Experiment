"""Report executed trades, not a fixed opportunity count; never disguise shortfall."""
import base64
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_200 import ROOT, INITIAL, pnl, select_leverage, equity_stats
from executed_465 import OUT, TARGET, FIRST_TEST, read_bars, get_calibrator
from replay_lower_confidence import formatted_table, NAMES


def embedded(name, alt):
    data = base64.b64encode((OUT / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def main():
    source = json.loads((OUT / "source_manifest.json").read_text())
    completion = json.loads((OUT / "completion.json").read_text())
    ledger = pd.read_csv(OUT / "evaluated_signals.csv")
    trades = pd.read_csv(OUT / "executed_trades.csv")
    count = len(trades); reached = count >= TARGET
    assert reached == completion["target_reached"] and count == completion["executed"]
    assert count <= TARGET and not ledger.duplicated(["ticker", "interval", "origin"]).any()
    if count:
        np.testing.assert_array_equal(trades.global_trade_number, np.arange(1, count+1))
        assert pd.to_datetime(trades.exit_bar, utc=True).is_monotonic_increasing
    stop = pd.to_datetime(completion["last_evaluated_exit"], utc=True) if reached else pd.Timestamp(source["cutoff_exclusive"])
    rows = []; bounds = []; periods = []
    plt.rcParams.update({"figure.facecolor": "#0b1220", "axes.facecolor": "#111c2f",
        "savefig.facecolor": "#0b1220", "text.color": "#e2e8f0", "axes.labelcolor": "#cbd5e1",
        "xtick.color": "#94a3b8", "ytick.color": "#94a3b8", "axes.edgecolor": "#334155",
        "grid.color": "#334155", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)
    asset_rank = {"RELIANCE.NS": 0, "SPY": 1, "EURUSD=X": 2}
    interval_rank = {"1m": 0, "1h": 1, "1d": 2}
    for item in source["cases"]:
        key = f"{item['ticker']}_{item['interval']}"
        result = json.loads((OUT / f"calibration_{key}.json").read_text())
        old = json.loads((ROOT / "results/confidence-200" / f"result_{key}.json").read_text())
        bars = read_bars(item); fit = get_calibrator(result)
        assert pd.Timestamp(result["calibration_last_exit"]) < pd.Timestamp(result["test_first_entry"])
        assert fit.a >= 0
        np.testing.assert_allclose(fit.predict(.99), result["maximum_confidence"], atol=1e-12)
        eligible_time = bars.index[FIRST_TEST+1:] <= stop
        available = int(eligible_time.sum())
        d = ledger[(ledger.ticker == item["ticker"]) & (ledger.interval == item["interval"])].copy()
        t = d[d.executed.astype(bool)].copy()
        if len(d):
            probabilities = d[[f"calibrated_profit_confidence_{i}x" for i in (1, 2, 3)]].to_numpy()
            chosen = select_leverage(probabilities); chosen[d.direction == 0] = 0
            np.testing.assert_array_equal(chosen, d.leverage)
            net, fees, funding, _ = pnl(d.exit_close/d.entry_close, d.direction, d.leverage,
                                       item["cost_bps"]/1e4, d.holding_years)
            np.testing.assert_allclose(net, d.net_return, atol=1e-12)
            np.testing.assert_allclose(fees, d.fees, atol=1e-12)
            np.testing.assert_allclose(funding, d.funding, atol=1e-12)
            np.testing.assert_allclose(INITIAL*np.cumprod(1+net), d.net_equity, rtol=1e-12)
            assert (net > -1).all()
            assert pd.to_datetime(d.entry_bar, utc=True).min() > pd.Timestamp(result["calibration_last_exit"])
        analytical = available if not result["can_qualify"] else 0
        skipped = int(sum(~d.executed.astype(bool))) + analytical
        if len(t): stats, equity = equity_stats(t.net_return.to_numpy())
        else: stats = {"ending": INITIAL, "return": 0., "max_drawdown": 0.}; equity = np.array([])
        before = np.r_[INITIAL, d.net_equity.to_numpy()[:-1]] if len(d) else np.array([])
        rows.append({"ticker": item["ticker"], "label": item["label"], "interval": item["interval"],
            "currency": item["currency"], "executed": len(t), "skipped": skipped,
            "signals_forecast_and_scored": len(d), "analytically_skipped": analytical,
            "future_candidate_bars_available": len(bars)-FIRST_TEST-1,
            "trades_1x": int(sum(t.leverage == 1)), "trades_2x": int(sum(t.leverage == 2)),
            "trades_3x": int(sum(t.leverage == 3)), **stats,
            "profit_cash": stats["ending"]-INITIAL,
            "net_win_rate": float((t.net_return > 0).mean()) if len(t) else None,
            "mean_selected_confidence": float(t.selected_confidence.mean()) if len(t) else None,
            "fees_cash": float(np.sum(before*d.fees)) if len(d) else 0.,
            "funding_cash": float(np.sum(before*d.funding)) if len(d) else 0.,
            "status": "traded" if len(t) else "confidence ceiling below 55%" if not result["can_qualify"] and available else
                      "not reached before target stop" if reached and not available else "no qualifying trade in available history"})
        bounds.append({"Instrument": item["label"], "Bars": NAMES[item["interval"]],
            "Original fit maximum": old["maximum_attainable_confidence"],
            "Earlier fit maximum": result["maximum_confidence"], "Can reach 55%": result["can_qualify"]})
        periods.append({"Instrument": item["label"], "Bars": NAMES[item["interval"]], "Data begins": item["first"],
            "Calibration first origin": result["calibration_first"], "Calibration last exit": result["calibration_last_exit"],
            "Test can start": result["test_first_entry"], "Data ends": item["last"], "Bars available": item["bars"]})
        ax = axes[asset_rank[item["ticker"]], interval_rank[item["interval"]]]
        ax.plot(np.arange(len(t)+1), np.r_[INITIAL, equity]/1000, color="#38bdf8", lw=2)
        if not len(t):
            ax.set_xlim(0, 1)
            ax.text(.5, .55, "No executed trades", transform=ax.transAxes, ha="center", color="#94a3b8")
        ax.set(title=f"{item['label']} · {NAMES[item['interval']]}", xlabel="Executed trades",
               ylabel=f"Thousands of {item['currency']}"); ax.grid(alpha=.2)
    rows = sorted(rows, key=lambda r: (asset_rank[r["ticker"]], interval_rank[r["interval"]]))
    summary = pd.DataFrame(rows); bounds = pd.DataFrame(bounds); periods = pd.DataFrame(periods)
    assert summary.executed.sum() == count
    summary.to_csv(OUT / "summary.csv", index=False)
    bounds.to_csv(OUT / "confidence_ceilings.csv", index=False)
    periods.to_csv(OUT / "data_and_calibration_dates.csv", index=False)
    fig.suptitle(f"Original 55% / 60% / 70% gates · {count} executed trades · separate accounts", fontsize=16)
    fig.savefig(OUT / "equity.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    x = np.arange(9); b = bounds.set_index(["Instrument", "Bars"])
    labels = [(r["label"], NAMES[r["interval"]]) for r in rows]
    ax.bar(x-.18, [b.loc[label, "Original fit maximum"] for label in labels], .36, color="#94a3b8", label="Original fit")
    ax.bar(x+.18, [b.loc[label, "Earlier fit maximum"] for label in labels], .36, color="#38bdf8", label="Extended-history fit")
    ax.axhline(.55, color="#fb923c", ls="--", label="Minimum gate 55%")
    ax.set_xticks(x, [f"{a}\n{i}" for a, i in labels]); ax.set(ylabel="Maximum calibrated profit probability", ylim=(0, 1))
    ax.legend(ncol=3, frameon=False); ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "confidence_ceilings.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    audit = {"status": "passed", "target": TARGET, "executed": count, "target_reached": reached,
        "shortfall": max(0, TARGET-count), "history_exhausted": completion["history_exhausted"],
        "checks": ["original 55/60/70 probability gates", "unique completed trades, not skipped signals",
            "strict calibration/test chronology and one-bar embargo", "calibration maps frozen before evaluation",
            "source hashes", "model forecast cache tied to source hash", "confidence ceiling skip proof",
            "each exposure passes its own gate", "fees and funding and separate-account equity",
            "global chronology and exact execution stopping rule", "no duplicated or bootstrapped trades"]}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    display = pd.DataFrame([{"Instrument": r["label"], "Bars": NAMES[r["interval"]], "Trades": r["executed"],
        "Skips": r["skipped"], "1× / 2× / 3×": f"{r['trades_1x']} / {r['trades_2x']} / {r['trades_3x']}",
        "Net return": r["return"], "Final balance": f"{r['ending']:,.2f} {r['currency']}",
        "Max drawdown": r["max_drawdown"], "Status": r["status"]} for r in rows])
    confidence = pd.DataFrame([{"Instrument": r["label"], "Bars": NAMES[r["interval"]], "Trades": r["executed"],
        "Mean predicted net-profit probability": r["mean_selected_confidence"], "Observed net win rate": r["net_win_rate"],
        "Fees": f"{r['fees_cash']:,.2f} {r['currency']}", "Financing + borrow": f"{r['funding_cash']:,.2f} {r['currency']}"} for r in rows])
    title = f"465-trade target reached: {count} executions" if reached else f"465-trade target: {count} executions, {TARGET-count} short"
    outcome = (f"The simulation reached {count} actual completed trades and stopped at the 465th execution, on {trades.exit_bar.iloc[-1]}."
        if reached else f"The simulation exhausted the available history with {count} qualifying completed trades. It did not reach 465.")
    doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Timestral · 465 executed-trade target</title><style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1380px;margin:auto;padding:40px 28px 65px}}h1{{font-size:clamp(30px,4vw,46px);line-height:1.17}}h2{{font-size:26px}}p{{max-width:1080px}}section{{margin:32px 0}}.eyebrow{{color:#38bdf8;text-transform:uppercase;letter-spacing:.1em;font-weight:700;font-size:13px}}.lead{{font-size:20px}}.box{{background:#111c2f;border:1px solid #334155;border-radius:14px;padding:20px 26px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.card{{padding:20px;border:1px solid #334155;border-radius:12px;background:#111c2f}}.number{{font-size:32px;font-weight:750;color:#38bdf8}}.scroll{{overflow-x:auto;border:1px solid #334155;border-radius:10px;margin:16px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}}th,td{{text-align:right;padding:11px 13px;border-bottom:1px solid #25334b}}th:first-child,td:first-child{{text-align:left}}th{{color:#93c5fd;background:#111c2f}}img{{width:100%;height:auto;border:1px solid #25334b;border-radius:12px}}a{{color:#7dd3fc}}.muted{{color:#94a3b8;font-size:14px}}li{{margin:7px 0}}@media(max-width:760px){{main{{padding:26px 16px}}.cards{{grid-template-columns:1fr}}}}</style></head><body><main>
<div class="eyebrow">Timestral-209M · original gates · executed-trade count</div><h1>{title}</h1>
<p class="lead">{outcome} The target is <strong>465 combined across the nine cases</strong>. Skips do not count as trades.</p>
<div class="cards"><div class="card"><div class="number">465</div>requested executions combined</div>
<div class="card"><div class="number">{count}</div>actual executions</div>
<div class="card"><div class="number">55% / 60% / 70%</div>unchanged 1× / 2× / 3× gates</div></div>
<section class="box"><h2>What stayed fixed, and why calibration dates changed</h2>
<p>The original one-trade report used the same gates: trade 3× if its estimated probability of net profit is at least 70%, otherwise
2× at 60%, otherwise 1× at 55%, otherwise hold cash. The model’s median chooses direction. Confidence includes fees, financing and borrow.</p>
<p>Counting executions instead of opportunities does not create qualifying signals. With the original saved calibration parameters,
eight cases have a confidence ceiling below 55% for any possible model score. The ninth, SPY daily, had one execution in its 200 later bars.</p>
<p><strong>This extended-history test preserves the gates and calibration algorithm, but fits new parameters before the earlier test begins.</strong>
Using a late-2025 or 2026 calibration to trade in the 1990s would use future outcomes. Each extended case instead uses 250 bars of context,
then its earliest 300 forecast origins for calibration, one embargo bar, and a frozen map over the remaining history.
No rolling refits, retries with alternative calibration windows, threshold changes or return optimization were used to reach the target.</p></section>
<section><h2>Results on executed trades</h2><p>Each independent account starts with 100,000: INR for Reliance, USD for SPY and EUR/USD.
The x-axis below counts completed trades. INR and USD account profits are not added together.</p>
{formatted_table(display, percentages=("Net return", "Max drawdown"))}
{embedded('equity.png', 'Separate account equity against number of executed trades')}</section>
<section><h2>Which calibrated models can pass the gates?</h2>
<p>The calibrator is monotone and clamps raw scores to its training range. Its upper bound therefore proves whether a model
can ever clear 55% under that frozen fit. Cases below the bound are skipped analytically, avoiding redundant test inference.</p>
{formatted_table(bounds, percentages=("Original fit maximum", "Earlier fit maximum"))}
{embedded('confidence_ceilings.png', 'Maximum attainable calibrated probabilities against the fixed minimum gate')}
<p class="muted">A bound applies to the particular frozen calibration map. It is not a claim that the instrument can never be predicted.
If the trade target is reached before another case’s test starts, that case is reported as not reached rather than as a tested failure.</p></section>
<section><h2>Confidence, realized net wins and costs</h2>
{formatted_table(confidence, percentages=("Mean predicted net-profit probability", "Observed net win rate"))}
<p class="muted">Confidence is a profit probability, not expected return. Samples are serially dependent, and historical model-training overlap
is unknown. These ideal historical fills do not establish prospective profitability.</p></section>
<section><h2>History and calibration periods</h2>{formatted_table(periods)}
<p>Daily history was expanded to Yahoo’s returned maximum and hourly history to approximately two years. Minute data reuse the saved snapshot.
All data stop before October 20, 2026, preserving the earlier study’s cutoff. Intraday retention differs from daily history;
see the <a href="https://github.com/ranaroussi/yfinance/blob/main/yfinance/scrapers/history.py">yfinance history implementation</a>.</p>
<p>The chronological stopping rule orders completed trades by exit timestamp, with fixed case ordering for ties. It closes each accepted
position at the next observed close; model inference and signal construction use only that case’s past input prices.</p></section>
<section><h2>Unchanged execution assumptions</h2><ul>
<li>Same official pinned 209M model, 250-close context, next-close forecast, FP32 and default symmetric averaging.</li>
<li>Ideal current-close to next-close fills, including gaps. Same-close fills omit inference and order latency.</li>
<li>Per entry and exit: Reliance 10 basis points, SPY 5, EUR/USD 2. Financing: 8% annually on exposure above 1×;
short borrow: 3% on short notional, using elapsed calendar time. Cash earns zero interest.</li>
<li>Entry fees reduce investable exposure. No FX carry, taxes, margin calls or intrabar liquidation simulation.</li>
<li>Daily prices use retrospective adjustments. The very early historical tests may overlap the model’s pretraining data.</li>
<li>No duplicated observations, resampling, overlapping positions within a case or hypothetical fills are counted as completed trades.</li></ul></section>
<section class="box"><h2>Downloads and audit</h2><ul>
<li><a href="executed_trades.csv">Every executed trade</a> · <a href="summary.csv">nine account summaries</a></li>
<li><a href="evaluated_signals.csv">Forecasted and evaluated signals</a> · <a href="confidence_ceilings.csv">analytical confidence bounds</a></li>
<li><a href="configuration.json">configuration</a> · <a href="source_manifest.json">source hashes</a> · <a href="audit.json">audit</a> · <a href="PROTOCOL.md">protocol</a></li>
<li><a href="../confidence-200/report.html">original one-trade report</a></li></ul>
<p>Audit passed: trade uniqueness, strict calibration chronology, unchanged gates, confidence bounds, cost identities,
separate-account equity and the execution-count stopping rule.</p></section></main></body></html>'''
    (OUT / "report.html").write_text(doc)
    print(json.dumps(audit, indent=2)); print(summary.to_string(index=False))


if __name__ == "__main__": main()
