"""Replay frozen forecasts/profit probabilities with explicitly lower trading gates.

No model loading, data download, calibration fitting or return-based threshold search.
The original account is reproduced before the new threshold schedules are evaluated.
"""
import base64
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from confidence_200 import ASSETS, INTERVALS, INITIAL, ROOT, pnl, equity_stats

SOURCE = ROOT / "results/confidence-200"
OUT = ROOT / "results/lower-confidence-200"
NAMES = {"1m": "Minute", "1h": "Hourly", "1d": "Daily"}
PROBABILITIES = [f"calibrated_profit_confidence_{n}x" for n in (1, 2, 3)]
SCENARIOS = {
    "original": {"label": "Original", "thresholds": [.55, .60, .70], "color": "#94a3b8"},
    "lower": {"label": "Lower · primary", "thresholds": [.35, .40, .45], "color": "#38bdf8"},
    "looser": {"label": "Looser", "thresholds": [.25, .30, .35], "color": "#a78bfa"},
    "very_low": {"label": "Very low", "thresholds": [.10, .20, .30], "color": "#fb923c"},
    "unfiltered_1x": {"label": "Every non-flat signal · 1×", "thresholds": None, "color": "#64748b"},
}
FIXED_COLUMNS = ["ticker", "interval", "opportunity", "entry_bar", "exit_bar", "entry_close", "exit_close",
                 "forecast_median", "direction", "raw_direction_confidence", "raw_profit_confidence_1x",
                 *PROBABILITIES, "holding_years"]


def choose_exposure(probabilities, direction, thresholds):
    """Highest level passing its own cost-adjusted probability gate; ties stay cash."""
    probabilities = np.asarray(probabilities, dtype=float)
    direction = np.asarray(direction)
    if probabilities.shape != (len(direction), 3):
        raise ValueError("Expected one probability for each of three exposure levels")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (3,) or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Expected three probability gates between zero and one")
    chosen = np.zeros(len(direction))
    for column, leverage in enumerate((1., 2., 3.)):
        chosen = np.where(probabilities[:, column] >= thresholds[column], leverage, chosen)
    chosen[direction == 0] = 0
    return chosen


def replay_case(source, asset, scenario, rule=None):
    """Apply gates to saved predictions, then use outcomes only to score P&L."""
    rule = SCENARIOS[scenario] if rule is None else rule
    p = source[PROBABILITIES].to_numpy()
    direction = source.direction.to_numpy()
    leverage = (direction != 0).astype(float) if rule["thresholds"] is None else choose_exposure(
        p, direction, rule["thresholds"])
    active = leverage > 0
    net, fees, funding, exposure = pnl(source.exit_close.to_numpy() / source.entry_close.to_numpy(),
        direction, leverage, asset["cost_bps"] / 1e4, source.holding_years.to_numpy())
    assert (net > -1).all(), "Bankruptcy requires an explicit halted-account policy"
    stats, equity = equity_stats(net)
    equity_before = np.r_[INITIAL, equity[:-1]]
    selected_confidence = np.full(len(source), np.nan)
    selected_confidence[active] = p[np.flatnonzero(active), leverage[active].astype(int) - 1]
    selected_threshold = np.full(len(source), np.nan)
    if rule["thresholds"] is not None:
        selected_threshold[active] = np.asarray(rule["thresholds"])[leverage[active].astype(int) - 1]
    underlying = direction * (source.exit_close.to_numpy() / source.entry_close.to_numpy() - 1)
    frictionless, _ = equity_stats(leverage * underlying)
    matched_net, *_ = pnl(source.exit_close.to_numpy() / source.entry_close.to_numpy(), direction,
        active.astype(float), asset["cost_bps"] / 1e4, source.holding_years.to_numpy())
    matched, _ = equity_stats(matched_net)
    ledger = source[FIXED_COLUMNS].copy()
    ledger.insert(0, "scenario", scenario)
    for name, values in {
        "leverage": leverage, "actual_exposure": exposure, "executed": active,
        "trade_number": np.where(active, np.cumsum(active), 0),
        "selected_confidence": selected_confidence, "selected_threshold": selected_threshold,
        "gross_return_before_costs": exposure * underlying, "net_return": net,
        "fees": fees, "funding": funding, "equity_before": equity_before, "net_equity": equity,
        "fee_cash": equity_before * fees, "funding_cash": equity_before * funding,
        "net_profit_cash": equity_before * net,
        "skip_reason": np.where(direction == 0, "unchanged_forecast",
            np.where(active, "executed", "profit_confidence_below_threshold")),
    }.items():
        ledger[name] = values
    result = {"scenario": scenario, "ticker": asset["ticker"], "label": asset["label"],
        "currency": asset["currency"], "interval": source.interval.iloc[0], "opportunities": len(source),
        "executed": int(sum(active)), "skipped": int(sum(~active)),
        **{f"trades_{i}x": int(sum(leverage == i)) for i in (1, 2, 3)},
        "flat_skips": int(sum(direction == 0)), "initial": INITIAL, **stats,
        "profit_cash": float(equity[-1] - INITIAL), "fee_cash": float(sum(equity_before * fees)),
        "funding_cash": float(sum(equity_before * funding)),
        "mean_confidence_executed": float(np.mean(selected_confidence[active])) if active.any() else None,
        "net_win_rate": float(np.mean(net[active] > 0)) if active.any() else None,
        "frictionless_return": frictionless["return"], "matched_signals_1x_return": matched["return"],
        "mean_leverage_all_opportunities": float(leverage.mean()),
        "first_entry": source.entry_bar.iloc[0], "last_exit": source.exit_bar.iloc[-1],
    }
    np.testing.assert_allclose(net, ledger.gross_return_before_costs - fees - funding, atol=1e-12)
    np.testing.assert_allclose(sum(ledger.net_profit_cash), result["profit_cash"], atol=1e-7)
    assert (ledger.loc[~active, ["net_return", "fees", "funding"]].to_numpy() == 0).all()
    if rule["thresholds"] is not None:
        assert (selected_confidence[active] >= selected_threshold[active]).all()
    return ledger, result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formatted_table(frame, percentages=(), money=()):
    frame = frame.copy()
    for col in percentages:
        frame[col] = frame[col].map(lambda v: "—" if pd.isna(v) else f"{v:.2%}")
    for col in money:
        frame[col] = frame[col].map(lambda v: f"{v:,.2f}")
    return '<div class="scroll">' + frame.to_html(index=False, border=0, escape=True) + '</div>'


def image(name, alt):
    encoded = base64.b64encode((OUT / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'


def plots(ledgers, summary):
    plt.rcParams.update({"figure.facecolor": "#0b1220", "axes.facecolor": "#111c2f",
        "savefig.facecolor": "#0b1220", "text.color": "#e2e8f0", "axes.labelcolor": "#cbd5e1",
        "xtick.color": "#94a3b8", "ytick.color": "#94a3b8", "axes.edgecolor": "#334155",
        "grid.color": "#334155", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for row, asset in enumerate(ASSETS):
        for col, interval in enumerate(INTERVALS):
            ax = axes[row, col]
            for scenario, rule in SCENARIOS.items():
                d = ledgers[scenario, asset["ticker"], interval]
                ax.plot(np.arange(201), np.r_[INITIAL, d.net_equity] / 1000,
                    color=rule["color"], label=rule["label"], lw=2.4 if scenario == "lower" else 1.4,
                    ls="--" if scenario == "unfiltered_1x" else "-")
            ax.set(title=f"{asset['label']} · {NAMES[interval]}", xlabel="Opportunity",
                   ylabel=f"Thousands of {asset['currency']}")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Same 1,800 opportunities · lower probability gates · net of costs", fontsize=17)
    fig.savefig(OUT / "equity.png", dpi=155, bbox_inches="tight"); plt.close(fig)
    primary = summary[summary.scenario == "lower"]
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    bottom = np.zeros(9)
    for column, label, color in [("skipped", "Skipped", "#475569"), ("trades_1x", "1×", "#22c55e"),
            ("trades_2x", "2×", "#38bdf8"), ("trades_3x", "3×", "#a78bfa")]:
        values = primary[column].to_numpy()
        ax.bar(np.arange(9), values, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set_xticks(np.arange(9), [f"{r.label}\n{NAMES[r.interval]}" for r in primary.itertuples()])
    ax.set(ylabel="Count", ylim=(0, 220), title="Primary gates 35% / 40% / 45% · 200 opportunities in each case")
    ax.legend(ncol=4, frameon=False)
    fig.savefig(OUT / "allocation.png", dpi=155, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)
    for row, asset in enumerate(ASSETS):
        for col, interval in enumerate(INTERVALS):
            ax = axes[row, col]
            d = primary[(primary.ticker == asset["ticker"]) & (primary.interval == interval)].iloc[0]
            values = [d.matched_signals_1x_return, d["return"], d.frictionless_return]
            ax.bar(["Same trades\nat 1×", "Actual\n1–3× net", "Same 1–3×\nno costs"], values,
                color=["#94a3b8", "#38bdf8", "#a78bfa"])
            if d.executed == 0:
                ax.text(.5, .65, "No trades\n200 opportunities skipped", transform=ax.transAxes,
                        ha="center", va="center", color="#94a3b8")
            ax.axhline(0, color="#64748b", lw=.7)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(f"{asset['label']} · {NAMES[interval]}")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Primary replay · separate signal selection, leverage and trading costs", fontsize=16)
    fig.savefig(OUT / "leverage_and_costs.png", dpi=155, bbox_inches="tight"); plt.close(fig)


def write_report(summary, ledgers, manifest):
    primary = summary[summary.scenario == "lower"]
    executed = int(primary.executed.sum()); skipped = int(primary.skipped.sum())
    schedules = pd.DataFrame([{"Schedule": r["label"], "1× gate": f"{r['thresholds'][0]:.0%}",
        "2× gate": f"{r['thresholds'][1]:.0%}", "3× gate": f"{r['thresholds'][2]:.0%}"}
        for r in SCENARIOS.values() if r["thresholds"] is not None])
    overview = []
    sections = []
    for scenario, rule in SCENARIOS.items():
        subset = summary[summary.scenario == scenario]
        overview.append({"Schedule": rule["label"], "Executed": int(subset.executed.sum()),
            "Skipped": int(subset.skipped.sum()), "1×": int(subset.trades_1x.sum()),
            "2×": int(subset.trades_2x.sum()), "3×": int(subset.trades_3x.sum()),
            "Profitable accounts / 9": int(sum(subset["return"] > 1e-12)),
            "Loss-making accounts / 9": int(sum(subset["return"] < -1e-12))})
        display = pd.DataFrame([{"Instrument": r["label"], "Bars": NAMES[r["interval"]],
            "Trades": r["executed"], "Skips": r["skipped"],
            "1× / 2× / 3×": f"{r['trades_1x']} / {r['trades_2x']} / {r['trades_3x']}",
            "Net return": r["return"], "P&L": f"{r['profit_cash']:+,.2f} {r['currency']}",
            "Final balance": f"{r['ending']:,.2f} {r['currency']}", "Max drawdown": r["max_drawdown"]}
            for r in subset.to_dict("records")])
        body = formatted_table(display, percentages=("Net return", "Max drawdown"))
        if scenario == "lower":
            sections.append(f'<section><h2>Primary replay: 35% / 40% / 45%</h2>{body}</section>')
        else:
            sections.append(f'<details><summary>{html.escape(rule["label"])} · all nine accounts</summary>{body}</details>')
    overview = pd.DataFrame(overview)
    overview.to_csv(OUT / "scenario_totals.csv", index=False)
    sensitivity = []
    for asset in ASSETS:
        for interval in INTERVALS:
            row = {"Instrument": asset["label"], "Bars": NAMES[interval]}
            for scenario, rule in SCENARIOS.items():
                r = summary[(summary.scenario == scenario) & (summary.ticker == asset["ticker"]) &
                            (summary.interval == interval)].iloc[0]
                row[rule["label"]] = f"{r['return']:+.2%} ({r.executed} trades)"
            sensitivity.append(row)
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT / "threshold_comparison.csv", index=False)
    confidence = pd.DataFrame([{"Instrument": r["label"], "Bars": NAMES[r["interval"]],
        "Executed": r["executed"], "Mean predicted profit probability": r["mean_confidence_executed"],
        "Observed net win rate": r["net_win_rate"], "Fees paid": f"{r['fee_cash']:,.2f} {r['currency']}",
        "Financing + borrow": f"{r['funding_cash']:,.2f} {r['currency']}"}
        for r in primary.to_dict("records")])
    comparison = pd.DataFrame([{"Instrument": r["label"], "Bars": NAMES[r["interval"]],
        "Same selected signals at 1×": r["matched_signals_1x_return"],
        "Actual 1–3×, net": r["return"], "Same selections, no fees/funding": r["frictionless_return"]}
        for r in primary.to_dict("records")])
    comparison.to_csv(OUT / "leverage_and_costs.csv", index=False)
    dates = primary[["label", "interval", "first_entry", "last_exit"]].rename(
        columns={"label": "Instrument", "interval": "Bars", "first_entry": "First entry (UTC)", "last_exit": "Last exit (UTC)"})
    dates["Bars"] = dates.Bars.map(NAMES)
    losses = int(sum(primary["return"] < -1e-12)); gains = int(sum(primary["return"] > 1e-12))
    high = int(primary.trades_2x.sum() + primary.trades_3x.sum())
    original = summary[summary.scenario == "original"]
    changed = primary.merge(original, on=["ticker", "interval"], suffixes=("_new", "_old"))
    reduced = int(sum(changed["return_new"] < changed["return_old"] - 1e-12))
    source_hash = manifest["source_hashes"]["results/confidence-200/all_1800_opportunities.csv"]
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Timestral · lower confidence replay</title>
<style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px/1.65 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:45px 30px 70px}}h1{{font-size:clamp(30px,4vw,48px);line-height:1.16;max-width:1000px}}h2{{font-size:26px;margin:0 0 14px}}h3{{font-size:20px}}
p{{max-width:1060px}}.eyebrow{{color:#38bdf8;letter-spacing:.1em;font-size:13px;font-weight:700;text-transform:uppercase}}.muted{{color:#94a3b8}}.lead{{font-size:20px}}
section,details{{margin:32px 0}}.box{{border:1px solid #334155;background:#111c2f;padding:22px 26px;border-radius:14px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}}
.card{{background:#111c2f;border:1px solid #334155;padding:20px;border-radius:12px}}.number{{font-size:32px;font-weight:750;color:#38bdf8}}.small{{font-size:14px}}a{{color:#7dd3fc}}img{{width:100%;height:auto;border:1px solid #25334b;border-radius:12px;margin:14px 0}}
.scroll{{overflow-x:auto;border:1px solid #334155;border-radius:10px;margin:16px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}}th,td{{text-align:right;padding:12px 14px;border-bottom:1px solid #25334b}}th:first-child,td:first-child{{text-align:left}}th{{color:#93c5fd;background:#111c2f}}tr:hover td{{background:#142238}}summary{{cursor:pointer;color:#7dd3fc;padding:12px 0;font-weight:650}}code{{overflow-wrap:anywhere}}li{{margin:7px 0}}@media(max-width:760px){{main{{padding:26px 16px}}.cards{{grid-template-columns:repeat(2,1fr)}}.box{{padding:18px}}}}</style></head>
<body><main><div class="eyebrow">Timestral-209M · saved forecast replay</div>
<h1>Lower confidence gates, more trades.<br>Same market data and probabilities.</h1>
<p class="lead">The primary lower-threshold rule executes <strong>{executed:,}</strong> of the same 1,800 opportunities,
compared with 1 previously. It skips {skipped:,} and uses 2× or 3× exposure on {high:,} trades.</p>
<section class="box"><h2>First: what confidence measures</h2>
<p><strong>Confidence is the estimated probability that a proposed next-bar trade makes money after fees,
financing and short borrow.</strong> It is not the probability that the price simply moves in the predicted direction,
and it is not a measure of expected return. The model’s median forecast still chooses long or short.</p>
<p>The existing calculation turns the model’s 21 next-price quantiles into an approximate probability of crossing
the trade’s break-even price. That price includes costs at each of 1×, 2× and 3×. A monotone regularized sigmoid,
fitted separately for each instrument and timeframe using 300 earlier forecast origins, calibrates those probabilities.
There is a one-bar embargo before each later 200-opportunity sample. The calibration map is frozen and clamps inputs
to the calibration range. <strong>This replay reuses those saved probabilities exactly.</strong></p>
<p>Previously, the gates were 55% / 60% / 70%. Now they are <strong>35% / 40% / 45%</strong>.
Choose the highest exposure whose own probability clears its gate; otherwise hold cash. A flat median forecast always skips.
For example, probabilities 43% / 42% / 41% select 2× under the primary rule, while
48% / 47% / 46% select 3×. Each number is the probability for that exposure level’s costs.</p>
<p class="small muted">A 40% estimated win probability means roughly 4 profitable trades out of 10 if the estimate is accurate.
Lowering a gate does not improve confidence. Win rate alone does not determine profit; win and loss sizes also matter.
The earlier probability audit beat raw Brier scores in 4/9 cases and a constant historical probability in 6/9 cases.</p></section>
<div class="cards"><div class="card"><div class="number">1,800</div>unique opportunities</div>
<div class="card"><div class="number">{executed:,}</div>primary executions</div>
<div class="card"><div class="number">{skipped:,}</div>primary skips</div>
<div class="card"><div class="number">3×</div>maximum permitted exposure</div></div>
<section><h2>The result of relaxing the gates</h2>
<p>The primary rule has {gains} profitable, {losses} loss-making and {9-gains-losses} unchanged accounts.
It produces a lower net return than the original rule in {reduced}/9 accounts. Additional executions are not evidence
of a better strategy; read the account returns alongside the increased trade count.</p>
<p>Each account starts with <strong>100,000</strong>: INR for Reliance, USD for SPY and EUR/USD.
Accounts remain separate; INR and USD P&amp;L are not added together.</p>
{formatted_table(schedules)}
{formatted_table(overview)}
<p class="small muted">These schedules were fixed before calculating their replay P&amp;L, with the earlier report already known.
They were not chosen by maximizing returns. Each row replays the same opportunities; rows are not independent new market evidence.
This is a post-hoc sensitivity analysis, not a fresh out-of-sample validation. The unfiltered comparator keeps every non-flat signal at 1×.</p></section>
<section><h2>All timeframes at a glance</h2><p>Net compounded return, with executed trade counts in parentheses.</p>
{formatted_table(sensitivity)}</section>
{sections[1]}
{image('allocation.png', 'Primary execution counts and exposure across nine instrument/timeframe cases')}
<section><h2>Full account paths</h2>{image('equity.png', 'Account equity for all threshold schedules on the same opportunities')}
<p class="small muted">The x-axis counts opportunities, not equal calendar durations across timeframes. Cash earns zero interest.
Every selected position exits at the next observed close and the next opportunity may reopen a position; each new trade pays both fill costs.</p></section>
<section><h2>Did leverage help the selected trades?</h2>
<p>The first column reruns exactly the primary rule’s selected signals at 1×. The second is the actual confidence-sized 1–3× account.
The last removes trading fees, financing and borrow on the same selections and nominal leverage. All three compound independently.</p>
{formatted_table(comparison, percentages=tuple(comparison.columns[2:]))}
{image('leverage_and_costs.png', 'Returns with the same selected signals at one times exposure, confidence sized exposure, and without costs')}
<h3>Predicted confidence versus realized outcomes</h3>
{formatted_table(confidence, percentages=('Mean predicted profit probability', 'Observed net win rate'))}
<p class="small muted">Mean confidence uses the exposure actually selected. Observed wins require strictly positive net P&amp;L.
These are small, serially dependent samples; differences between predicted and realized rates can be substantial.</p></section>
<section><h2>Other schedules, using exactly the same numbers</h2>{sections[0]}{''.join(sections[2:])}</section>
<section><h2>Why some minute cases still skip</h2>
<p>The saved calibrated profit probabilities are very low on minute data because round-trip fees often exceed the next-bar movement.
Lowering a confidence gate does not remove those costs. The schedules above accept only probabilities at or above their stated gates;
the unfiltered 1× comparator shows what happens if all non-flat model signals are traded anyway.</p></section>
<section><h2>Execution assumptions and reproducibility</h2>
<p>The Yahoo Finance snapshot, Timestral revision, 250-close contexts, forecast medians, directions, calibration parameters,
probabilities, entry/exit timestamps and holding periods are identical to the preceding confidence report.
No model inference, calibration refit or new download was needed.</p>
<ul><li>200 opportunities per case: Reliance, SPY and EUR/USD × minute, hourly and daily bars.</li>
<li>Ideal observed-close to next-observed-close fills, including overnight/weekend gaps. Same-close signal execution excludes latency.</li>
<li>Fees per entry and exit: Reliance 10 basis points, SPY 5, EUR/USD 2. These are unchanged simulation assumptions.</li>
<li>8% annual financing on exposure above 1×; 3% annual short borrow; elapsed calendar time, including weekends.
No FX carry, taxes, separate exchange fees or margin-call/intrabar liquidation simulation.</li>
<li>Effective starting exposure = requested leverage / (1 + per-fill cost × requested leverage), leaving the entry fee in the account.
Exposure labels 1×/2×/3× refer to requested leverage.</li>
<li>Daily inputs use retrospective adjusted closes; model training overlap is unknown. No broker orders are placed.</li></ul>
{formatted_table(dates)}
<p class="small">Audit passed: original leverage decisions and all nine equity paths reproduced, 200 opportunities per case,
frozen inputs preserved, cost identities checked, skips charged zero, chronological entry/exit verified,
and every relaxed schedule has at least as many executions as its stricter predecessor.</p>
<p class="small muted">Source ledger SHA-256: <code>{source_hash}</code>.<br>
The need for separate evaluation after changing thresholds is described in
<a href="https://scikit-learn.org/stable/modules/classification_threshold.html">scikit-learn’s decision-threshold documentation</a>.</p></section>
<section class="box"><h2>Download the numbers</h2><ul>
<li><a href="all_1800_opportunities.csv">Primary replay · all 1,800 opportunities</a> and <a href="all_executed_trades.csv">executed trades only</a></li>
<li><a href="summary.csv">All nine account summaries for every schedule</a> and <a href="scenario_totals.csv">schedule execution counts</a></li>
<li><a href="threshold_comparison.csv">All timeframes and threshold schedules at a glance</a></li>
<li><a href="all_scenarios.csv">All five replays in one ledger</a> · 9,000 rows, 1,800 unique opportunities</li>
<li><a href="configuration.json">Frozen configuration and source hashes</a> · <a href="audit.json">audit</a> · <a href="PROTOCOL.md">protocol</a></li>
<li><a href="../confidence-200/report.html">Preceding confidence report</a></li></ul></section>
</main></body></html>'''
    (OUT / "report.html").write_text(document)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = SOURCE / "all_1800_opportunities.csv"
    source = pd.read_csv(source_file)
    original_config = json.loads((SOURCE / "configuration.json").read_text())
    paths = [source_file, SOURCE / "configuration.json"]
    paths += [SOURCE / f"{kind}_{a['ticker']}_{i}.{suffix}"
        for a in ASSETS for i in INTERVALS
        for kind, suffix in [("calibration", "csv"), ("forecasts", "npz"), ("result", "json")]]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    manifest = {"analysis": "post_hoc_threshold_sensitivity_replay", "primary_scenario": "lower",
        "unique_opportunities": 1800, "opportunities_per_case": 200,
        "model_inference_rerun": False, "calibration_refit": False, "new_market_data": False,
        "threshold_search_by_returns": False, "scenarios": SCENARIOS,
        "source_configuration": original_config, "source_hashes": hashes}
    # Save the fixed protocol before computing any scenario outcomes.
    (OUT / "configuration.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "PROTOCOL.md").write_text("""# Lower-confidence replay protocol

Reuse the saved 1,800 opportunity records and their frozen calibrated profit
probabilities. Do not rerun the model, refit calibration or refresh Yahoo data.
Primary thresholds: 35% / 40% / 45% for 1x / 2x / 3x. Sensitivity schedules:
25% / 30% / 35% and 10% / 20% / 30%. The highest qualifying exposure wins.
Unchanged median forecasts always skip. Reproduce the original 55% / 60% / 70%
rule and compare the saved unfiltered 1x baseline. Each case starts at 100,000
in its original quote currency; costs, holding periods and compounding are fixed.

Rules are fixed before this replay's P&L calculations, but earlier results have
already been inspected. This is post-hoc sensitivity analysis of the same sample,
not a new prospective or untouched validation set. No schedule is selected by
return optimization, and all schedules are reported. Nine independent simulated
accounts retain 200 opportunities each; executions and skips are separate.
Five scenario replays do not create 9,000 independent observations.
""")
    assert len(source) == 1800 and not source.duplicated(["ticker", "interval", "opportunity"]).any()
    ledgers = {}; rows = []
    for scenario in SCENARIOS:
        for asset in ASSETS:
            for interval in INTERVALS:
                case = source[(source.ticker == asset["ticker"]) & (source.interval == interval)].reset_index(drop=True)
                assert len(case) == 200 and case.opportunity.tolist() == list(range(1, 201))
                assert (pd.to_datetime(case.entry_bar, utc=True) < pd.to_datetime(case.exit_bar, utc=True)).all()
                assert pd.to_datetime(case.entry_bar, utc=True).is_monotonic_increasing and case.entry_bar.is_unique
                ledger, result = replay_case(case, asset, scenario)
                if scenario == "original":
                    np.testing.assert_array_equal(ledger.leverage, case.leverage)
                    np.testing.assert_allclose(ledger.net_return, case.net_return, atol=1e-12)
                    np.testing.assert_allclose(ledger.net_equity, case.net_equity, rtol=1e-12)
                if scenario == "unfiltered_1x":
                    np.testing.assert_allclose(ledger.net_return, case.baseline_net_return, atol=1e-12)
                    np.testing.assert_allclose(ledger.net_equity, case.baseline_equity, rtol=1e-12)
                if scenario == "lower":
                    ledger.to_csv(OUT / f"opportunities_{asset['ticker']}_{interval}.csv", index=False)
                for col in FIXED_COLUMNS:
                    pd.testing.assert_series_equal(ledger[col], case[col], check_names=True)
                ledgers[scenario, asset["ticker"], interval] = ledger
                rows.append(result)
    summary = pd.DataFrame(rows)
    counts = [int(summary.loc[summary.scenario == s, "executed"].sum()) for s in SCENARIOS]
    assert counts[0] == 1 and all(a <= b for a, b in zip(counts, counts[1:]))
    for asset in ASSETS:
        for interval in INTERVALS:
            previous = np.zeros(200)
            for scenario in SCENARIOS:
                current = ledgers[scenario, asset["ticker"], interval].leverage.to_numpy()
                if scenario != "unfiltered_1x":
                    assert (current >= previous).all()
                else:
                    assert ((current > 0) >= (previous > 0)).all()
                previous = current
    primary = pd.concat([d for (s, _, _), d in ledgers.items() if s == "lower"], ignore_index=True)
    primary.to_csv(OUT / "all_1800_opportunities.csv", index=False)
    primary.loc[primary.executed].to_csv(OUT / "all_executed_trades.csv", index=False)
    pd.concat(ledgers.values(), ignore_index=True).to_csv(OUT / "all_scenarios.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    plots(ledgers, summary)
    write_report(summary, ledgers, manifest)
    assert all(digest(ROOT / path) == expected for path, expected in hashes.items())
    audit = {"status": "passed", "unique_opportunities": 1800, "cases": 9,
        "scenario_replay_rows": len(summary) * 200,
        "executed_by_scenario": dict(zip(SCENARIOS, counts)),
        "checks": ["original policy and nine account equity paths reproduced", "unfiltered 1x baseline reproduced",
            "200 chronological opportunities per case", "frozen probabilities and inputs unchanged",
            "each level uses its own calibrated profit probability", "no fees or funding on skips",
            "return equals gross minus fees and funding", "cash P&L reconciles to ending equity",
            "relaxing gates never removes an execution or reduces requested exposure",
            "source file hashes unchanged after replay", "no bankrupt simulated account"]}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    print(summary.loc[summary.scenario == "lower", ["label", "interval", "executed", "skipped", "trades_1x",
        "trades_2x", "trades_3x", "return", "ending", "max_drawdown"]].to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
