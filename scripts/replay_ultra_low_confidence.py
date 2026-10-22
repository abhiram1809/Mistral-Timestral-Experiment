"""Further threshold sensitivity on the identical saved 1,800 opportunities."""
import base64
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from replay_lower_confidence import (
    ASSETS, INTERVALS, INITIAL, ROOT, SOURCE, FIXED_COLUMNS,
    NAMES, digest, formatted_table, replay_case,
)

OUT = ROOT / "results/ultra-low-confidence-200"
SCENARIOS = {
    "previous_lowest": {"label": "Previous lowest · 10% / 20% / 30%",
        "thresholds": [.10, .20, .30], "color": "#94a3b8"},
    "one_percent": {"label": "Lower again · 1% / 5% / 10%",
        "thresholds": [.01, .05, .10], "color": "#a78bfa"},
    "all_signals": {"label": "Primary · 0.1% / 1% / 5%",
        "thresholds": [.001, .01, .05], "color": "#38bdf8"},
    "max_exposure": {"label": "Lowest · 0.1% / 0.2% / 0.3%",
        "thresholds": [.001, .002, .003], "color": "#fb923c"},
}
PRIMARY = "all_signals"


def embedded(name, alt):
    data = base64.b64encode((OUT / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def make_charts(ledgers, summary):
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
                    color=rule["color"], lw=2.3 if scenario == PRIMARY else 1.4,
                    label=rule["label"], ls="--" if scenario == "max_exposure" else "-")
            ax.set(title=f"{asset['label']} · {NAMES[interval]}", xlabel="Opportunity",
                   ylabel=f"Thousands of {asset['currency']}")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Still lower confidence gates · same 1,800 opportunities · after costs", fontsize=17)
    fig.savefig(OUT / "equity.png", dpi=155, bbox_inches="tight"); plt.close(fig)
    primary = summary[summary.scenario == PRIMARY]
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    bottom = np.zeros(9)
    for column, label, color in [("trades_1x", "1×", "#22c55e"), ("trades_2x", "2×", "#38bdf8"),
            ("trades_3x", "3×", "#a78bfa"), ("skipped", "Skipped", "#475569")]:
        values = primary[column].to_numpy()
        ax.bar(np.arange(9), values, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set_xticks(np.arange(9), [f"{r.label}\n{NAMES[r.interval]}" for r in primary.itertuples()])
    ax.set(ylabel="Count", ylim=(0, 220), title="Primary gates: 0.1% / 1% / 5% · 200 opportunities per case")
    ax.legend(ncol=4, frameon=False)
    fig.savefig(OUT / "allocation.png", dpi=155, bbox_inches="tight"); plt.close(fig)


def write_report(summary, totals, comparison, manifest):
    primary = summary[summary.scenario == PRIMARY]
    executed = int(primary.executed.sum()); skipped = int(primary.skipped.sum())
    display = pd.DataFrame([{"Instrument": r["label"], "Bars": NAMES[r["interval"]],
        "Trades": r["executed"], "Skips": r["skipped"],
        "1× / 2× / 3×": f"{r['trades_1x']} / {r['trades_2x']} / {r['trades_3x']}",
        "Net return": r["return"], "Final balance": f"{r['ending']:,.2f} {r['currency']}",
        "Max drawdown": r["max_drawdown"]} for r in primary.to_dict("records")])
    confidence = pd.DataFrame([{"Instrument": r["label"], "Bars": NAMES[r["interval"]],
        "Mean estimated net profit probability": r["mean_confidence_executed"],
        "Observed net win rate": r["net_win_rate"],
        "Fees": f"{r['fee_cash']:,.2f} {r['currency']}",
        "Financing + borrow": f"{r['funding_cash']:,.2f} {r['currency']}"}
        for r in primary.to_dict("records")])
    comparison_table = comparison.rename(columns={s: rule["label"] for s, rule in SCENARIOS.items()})
    sizing = pd.DataFrame([{"Instrument": r["label"], "Bars": NAMES[r["interval"]],
        "Same signals at 1×, net": r["matched_signals_1x_return"],
        "Primary 1–3×, net": r["return"], "Primary selections, no costs": r["frictionless_return"]}
        for r in primary.to_dict("records")])
    sizing.to_csv(OUT / "leverage_and_costs.csv", index=False)
    period = primary[["label", "interval", "first_entry", "last_exit"]].rename(columns={
        "label": "Instrument", "interval": "Bars", "first_entry": "First entry (UTC)", "last_exit": "Last exit (UTC)"})
    period["Bars"] = period.Bars.map(NAMES)
    extra = []
    for scenario, rule in SCENARIOS.items():
        if scenario == PRIMARY:
            continue
        d = summary[summary.scenario == scenario]
        table = pd.DataFrame([{"Instrument": r["label"], "Bars": NAMES[r["interval"]],
            "Trades": r["executed"], "Skips": r["skipped"],
            "1× / 2× / 3×": f"{r['trades_1x']} / {r['trades_2x']} / {r['trades_3x']}",
            "Net return": r["return"], "Final balance": f"{r['ending']:,.2f} {r['currency']}",
            "Max drawdown": r["max_drawdown"]} for r in d.to_dict("records")])
        extra.append(f'<details><summary>{rule["label"]}</summary>{formatted_table(table, percentages=("Net return", "Max drawdown"))}</details>')
    losses = int(sum(primary["return"] < 0))
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Timestral · even lower confidence</title>
<style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px/1.65 system-ui,sans-serif}}
main{{max-width:1380px;margin:auto;padding:40px 28px 65px}}h1{{font-size:clamp(30px,4vw,48px);line-height:1.17}}h2{{font-size:26px}}p{{max-width:1080px}}section{{margin:32px 0}}
.eyebrow{{color:#38bdf8;text-transform:uppercase;letter-spacing:.1em;font-weight:700;font-size:13px}}.lead{{font-size:20px}}.box{{background:#111c2f;border:1px solid #334155;border-radius:14px;padding:20px 26px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.card{{padding:20px;border:1px solid #334155;border-radius:12px;background:#111c2f}}.number{{font-size:32px;font-weight:750;color:#38bdf8}}
.scroll{{overflow-x:auto;border:1px solid #334155;border-radius:10px;margin:16px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}}th,td{{text-align:right;padding:11px 13px;border-bottom:1px solid #25334b}}th:first-child,td:first-child{{text-align:left}}th{{color:#93c5fd;background:#111c2f}}tr:hover td{{background:#142238}}img{{width:100%;height:auto;border:1px solid #25334b;border-radius:12px}}a,summary{{color:#7dd3fc}}summary{{cursor:pointer;padding:12px 0;font-weight:650}}.muted{{color:#94a3b8;font-size:14px}}code{{overflow-wrap:anywhere}}li{{margin:7px 0}}@media(max-width:760px){{main{{padding:26px 16px}}.cards{{grid-template-columns:1fr}}.box{{padding:18px}}}}</style></head><body><main>
<div class="eyebrow">Timestral-209M · another frozen-data replay</div>
<h1>Lowered enough to trade every non-flat signal.</h1>
<p class="lead">The primary <strong>0.1% / 1% / 5%</strong> gates execute <strong>{executed:,}</strong> simulated trades
and skip <strong>{skipped}</strong> of the same 1,800 opportunities. All nine instrument/timeframe cases now trade.</p>
<section class="box"><h2>The confidence estimates are unchanged</h2>
<p>Confidence is the saved estimated probability of <strong>positive next-bar P&amp;L after fees, financing and short borrow</strong>
at each proposed exposure. Model quantiles give a break-even probability; a monotone sigmoid fitted on 300 earlier forecast origins
calibrates it separately for each instrument/timeframe. The forecast direction, probabilities and calibration remain frozen.</p>
<p>Only the admission gates changed. A <strong>0.1% gate means 0.001 probability</strong>, not 10%.
The highest passing level wins: 1× needs 0.1%, 2× needs 1%, and 3× needs 5% under the primary schedule.
For example, saved probabilities near 0.33% pass the 1× gate but fail the higher gates.</p>
<p>Lower gates accept trades with lower estimated chances of profit. They do not improve the estimates or their accuracy.
Mean confidence and realized net win rates are shown below, including minute cases.</p></section>
<div class="cards"><div class="card"><div class="number">{executed:,}</div>primary trades, up from 597 in the preceding primary replay</div>
<div class="card"><div class="number">{skipped}</div>skips: unchanged EUR/USD minute median forecasts</div>
<div class="card"><div class="number">{losses}/9</div>primary accounts lose money after costs</div></div>
<section><h2>Schedules and trade counts</h2>{formatted_table(totals)}
<p>The most permissive two schedules reach the same maximum trade count. Further relaxation then increases leverage rather than
creating more signals. At 0.1% / 0.2% / 0.3%, every non-flat signal qualifies for 3× in this saved sample.</p>
<p class="muted">Thresholds were declared before their new P&amp;L was calculated, after earlier results were known.
This is threshold sensitivity on the same historical sample. Four replays contain 7,200 rows but only 1,800 unique opportunities.
No return-optimized schedule or new validation set is claimed.</p></section>
<section><h2>Primary result · 0.1% / 1% / 5%</h2>
<p>Each independent account starts with 100,000: INR for Reliance, USD for SPY and EUR/USD. Returns compound after costs.
Currencies are not added together.</p>{formatted_table(display, percentages=("Net return", "Max drawdown"))}
{embedded('allocation.png', 'Primary executed trades by exposure and skipped signals across nine cases')}</section>
<section><h2>Every schedule, side by side</h2><p>Net compounded return, with executed trade counts in parentheses.</p>
{formatted_table(comparison_table)}{embedded('equity.png', 'Equity paths for the previous and three still lower gate schedules')}</section>
<section><h2>What leverage and costs changed</h2>
<p>The 1× column uses exactly the primary selected signals. The last column removes fees, financing and borrow on the same signals
and nominal exposure; the accounts compound independently.</p>
{formatted_table(sizing, percentages=tuple(sizing.columns[2:]))}
<h2>Predicted confidence versus actual net wins</h2>
{formatted_table(confidence, percentages=("Mean estimated net profit probability", "Observed net win rate"))}
<p class="muted">These are descriptive, serially dependent samples of at most 200 trades per case.
A win means strictly positive after-cost P&amp;L; win frequency alone does not determine expected return.</p></section>
<section><h2>Other account details</h2>{''.join(extra)}</section>
<section><h2>Unchanged simulation assumptions</h2><ul>
<li>One market per category: Reliance (India), SPY (U.S.) and EUR/USD (FX); 200 opportunities each on minute, hourly and daily bars.</li>
<li>Identical Yahoo prices, 250-close model contexts, directions, frozen calibration and holding periods as the earlier report.</li>
<li>Ideal current-close to next-observed-close fills; each selected trade pays a new entry and exit cost. Same-close execution excludes latency.</li>
<li>Per-fill fees: Reliance 10 basis points, SPY 5, EUR/USD 2. Annual financing: 8% on actual exposure above 1×;
short borrow: 3%. Elapsed calendar time includes weekends.</li>
<li>Effective exposure is nominal leverage / (1 + per-fill fee × nominal leverage), reserving the entry fee. Cash earns no interest.</li>
<li>No FX carry, taxes or intrabar margin liquidation. Daily prices are retrospectively adjusted; model training overlap is unknown.
No inference rerun, data download, calibration refit or real order was needed.</li></ul>
{formatted_table(period)}
<p class="muted">Audit: preceding lowest-gate replay reproduced; frozen values and source hashes preserved; each case has 200 opportunities;
all cost identities and ending balances reconcile; skips have zero costs; relaxing gates cannot remove trades or lower exposure.</p>
<p class="muted">Source ledger SHA-256: <code>{manifest['source_ledger_sha256']}</code></p></section>
<section class="box"><h2>Download the report data</h2><ul>
<li><a href="all_1800_opportunities.csv">Primary · all 1,800 opportunities</a> · <a href="all_executed_trades.csv">primary executed trades</a></li>
<li><a href="summary.csv">Every account and schedule</a> · <a href="scenario_totals.csv">execution counts</a> · <a href="threshold_comparison.csv">side-by-side returns</a></li>
<li><a href="all_scenarios.csv">All four scenario ledgers</a> · <a href="configuration.json">configuration and hashes</a> · <a href="audit.json">audit</a></li>
<li><a href="../lower-confidence-200/report.html">Preceding lower-confidence report</a> · <a href="../confidence-200/report.html">original calibration report</a></li></ul></section>
</main></body></html>'''
    (OUT / "report.html").write_text(document)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = SOURCE / "all_1800_opportunities.csv"
    previous_file = ROOT / "results/lower-confidence-200/all_scenarios.csv"
    paths = [source_file, SOURCE / "configuration.json", previous_file]
    previous_manifest = json.loads((ROOT / "results/lower-confidence-200/configuration.json").read_text())
    hashes = dict(previous_manifest["source_hashes"])
    assert all(digest(ROOT / path) == expected for path, expected in hashes.items())
    hashes.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
    assert all(digest(ROOT / path) == expected for path, expected in hashes.items())
    manifest = {"analysis": "further_post_hoc_threshold_sensitivity", "primary_scenario": PRIMARY,
        "source_ledger_sha256": digest(source_file), "scenarios": SCENARIOS,
        "source_hashes": hashes, "source_configuration": previous_manifest["source_configuration"],
        "new_inference": False, "calibration_refit": False, "new_data": False,
        "threshold_search_by_returns": False, "unique_opportunities": 1800}
    # Write the fixed rules before calculating their new P&L.
    (OUT / "configuration.json").write_text(json.dumps(manifest, indent=2))
    source = pd.read_csv(source_file)
    previous = pd.read_csv(previous_file).query("scenario == 'very_low'").reset_index(drop=True)
    assert len(source) == len(previous) == 1800
    ledgers = {}; rows = []
    for scenario, rule in SCENARIOS.items():
        for asset in ASSETS:
            for interval in INTERVALS:
                case = source[(source.ticker == asset["ticker"]) & (source.interval == interval)].reset_index(drop=True)
                assert len(case) == 200 and case.opportunity.tolist() == list(range(1, 201))
                entry = pd.to_datetime(case.entry_bar, utc=True)
                assert entry.is_monotonic_increasing and entry.is_unique
                assert (entry < pd.to_datetime(case.exit_bar, utc=True)).all()
                ledger, result = replay_case(case, asset, scenario, rule=rule)
                for column in FIXED_COLUMNS:
                    pd.testing.assert_series_equal(ledger[column], case[column])
                if scenario == "previous_lowest":
                    old = previous[(previous.ticker == asset["ticker"]) & (previous.interval == interval)]
                    np.testing.assert_array_equal(ledger.leverage, old.leverage)
                    np.testing.assert_allclose(ledger.net_equity, old.net_equity, rtol=1e-12)
                else:
                    predecessor = list(SCENARIOS)[list(SCENARIOS).index(scenario)-1]
                    assert (ledger.leverage.to_numpy() >= ledgers[predecessor, asset["ticker"], interval].leverage.to_numpy()).all()
                ledgers[scenario, asset["ticker"], interval] = ledger
                rows.append(result)
    summary = pd.DataFrame(rows)
    totals = pd.DataFrame([{"Schedule (1× / 2× / 3× gates)": rule["label"],
        **{column: int(summary.loc[summary.scenario == scenario, column].sum())
           for column in ("executed", "skipped", "trades_1x", "trades_2x", "trades_3x")}}
        for scenario, rule in SCENARIOS.items()])
    totals.columns = ["Schedule (1× / 2× / 3× gates)", "Executed", "Skipped", "1×", "2×", "3×"]
    comparison = []
    for asset in ASSETS:
        for interval in INTERVALS:
            row = {"Instrument": asset["label"], "Bars": NAMES[interval]}
            for scenario in SCENARIOS:
                r = summary[(summary.scenario == scenario) & (summary.ticker == asset["ticker"]) &
                            (summary.interval == interval)].iloc[0]
                row[scenario] = f"{r['return']:+.2%} ({r.executed} trades)"
            comparison.append(row)
    comparison = pd.DataFrame(comparison)
    all_replays = pd.concat(ledgers.values(), ignore_index=True)
    primary = all_replays[all_replays.scenario == PRIMARY]
    assert primary.executed.sum() == (source.direction != 0).sum() == 1790
    assert (all_replays.query("scenario == 'max_exposure' and executed").leverage == 3).all()
    summary.to_csv(OUT / "summary.csv", index=False)
    totals.to_csv(OUT / "scenario_totals.csv", index=False)
    comparison.to_csv(OUT / "threshold_comparison.csv", index=False)
    all_replays.to_csv(OUT / "all_scenarios.csv", index=False)
    primary.to_csv(OUT / "all_1800_opportunities.csv", index=False)
    primary.loc[primary.executed].to_csv(OUT / "all_executed_trades.csv", index=False)
    make_charts(ledgers, summary)
    write_report(summary, totals, comparison, manifest)
    assert all(digest(ROOT / path) == expected for path, expected in hashes.items())
    audit = {"status": "passed", "source_hashes_unchanged": True,
        "unique_opportunities": 1800, "scenario_replay_rows": len(all_replays),
        "primary_executed": int(primary.executed.sum()), "primary_skipped": int((~primary.executed).sum()),
        "checks": ["previous lowest-gate replay reproduced", "frozen probabilities and source hashes preserved",
            "200 chronological opportunities per case", "each exposure passes its own gate",
            "every non-flat signal executed under primary schedule", "flat medians remain skipped",
            "all cost and cash P&L identities reconcile", "relaxation never lowers exposure",
            "all non-flat signals qualify at 3x under lowest schedule", "no bankrupt account"]}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    print(totals.to_string(index=False))
    print(summary[summary.scenario == PRIMARY][["label", "interval", "executed", "skipped", "trades_1x",
        "trades_2x", "trades_3x", "return", "ending", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
