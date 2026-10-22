"""Generate a self-contained HTML research report and publication-size charts."""
import base64
import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from infer import ROOT, load_prices

OUT = ROOT / "results"
FIG = OUT / "figures"
COLORS = {
    "passive_1x": "#64748b", "raw_sign_1x": "#ef4444",
    "calibrated_sign_1x": "#eab308", "confidence_cap1": "#22c55e",
    "confidence_cap3": "#38bdf8", "momentum_1x": "#a78bfa",
}
LABELS = {
    "passive_1x": "Buy & hold", "raw_sign_1x": "Raw direction 1x",
    "calibrated_sign_1x": "Calibrated direction 1x",
    "confidence_cap1": "Confidence · cap 1x",
    "confidence_cap3": "Confidence · cap 3x", "momentum_1x": "Momentum 1x",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": "#0b1220", "axes.facecolor": "#111c2f",
        "savefig.facecolor": "#0b1220", "text.color": "#e2e8f0",
        "axes.labelcolor": "#cbd5e1", "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8", "axes.edgecolor": "#334155",
        "grid.color": "#334155", "font.family": "DejaVu Sans",
        "font.size": 10, "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3,
    })


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def table(df, percent_cols=(), decimals=3):
    copy = df.copy()
    for col in copy:
        if col in percent_cols:
            copy[col] = copy[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        elif pd.api.types.is_float_dtype(copy[col]):
            copy[col] = copy[col].map(lambda x: f"{x:,.{decimals}f}" if pd.notna(x) else "—")
    return copy.to_html(index=False, border=0, classes="data", escape=True)


def image(name, alt):
    b64 = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}"></figure>'


def main():
    FIG.mkdir(exist_ok=True)
    style()
    cfg = json.loads((ROOT / "config.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    fm = pd.read_csv(OUT / "forecast_metrics.csv")
    tm = pd.read_csv(OUT / "trading_metrics.csv")
    cv = pd.read_csv(OUT / "equity_curves.csv")
    ca = pd.read_csv(OUT / "calibration.csv")
    bins = pd.read_csv(OUT / "confidence_bins.csv")
    latest = pd.read_csv(OUT / "latest_signals.csv")
    pm = pd.read_csv(OUT / "price_metrics.csv")
    pf = pd.read_csv(OUT / "latest_price_forecasts.csv")
    pred = pd.read_csv(OUT / "test_predictions.csv")
    primary = cfg["primary_context"]
    tickers = [a["ticker"] for a in cfg["assets"]]
    f = fm[(fm.context == primary) & (fm.horizon == 2)].set_index("ticker").loc[tickers]
    t = tm[(tm.context == primary) & (tm.cost_multiplier == 1)]
    c = ca[ca.context == primary].set_index("ticker").loc[tickers]
    cap = t[t.strategy == "confidence_cap3"].set_index("ticker").loc[tickers]
    raw = t[t.strategy == "raw_sign_1x"].set_index("ticker").loc[tickers]
    passive = t[t.strategy == "passive_1x"].set_index("ticker").loc[tickers]
    wins = int((f.mae_skill_zero > 0).sum())
    significant = int((f.mae_improvement_ci_low > 0).sum())
    profit = int((cap.net_return > 0).sum())
    beatpassive = int((cap.net_return > passive.net_return).sum())
    summary = {
        "forecast_mae_winners_of_nine": wins,
        "mae_improvement_unadjusted_ci_above_zero": significant,
        "confidence_profitable_accounts_of_nine": profit,
        "confidence_beats_passive_of_nine": beatpassive,
        "mean_direction_accuracy": float(f.direction_accuracy.mean()),
        "median_raw_coverage80": float(f.coverage80.median()),
        "median_calibrated_coverage80": float(c.calibrated_coverage80.median()),
        "max_realized_leverage": float(cap.max_abs_exposure.max()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    for ax, ticker in zip(axes.flat, tickers):
        frame = cv[(cv.ticker == ticker) & (cv.context == primary)]
        for strategy in ["passive_1x", "raw_sign_1x", "calibrated_sign_1x", "confidence_cap3"]:
            rr = frame[frame.strategy == strategy]
            ax.plot(pd.to_datetime(rr.date), rr.equity/1000, lw=1.6,
                    label=LABELS[strategy], color=COLORS[strategy])
        ax.axhline(100, color="#94a3b8", ls=":", lw=.8)
        ax.set_title(ticker)
        ax.set_ylabel("Account value · thousands")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("500 delayed trading decisions · costs, financing and short borrow included", fontsize=17)
    save(fig, "equity")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].barh(tickers, f.mae_skill_zero*100,
                 color=np.where(f.mae_skill_zero >= 0, "#22c55e", "#ef4444"))
    axes[0].axvline(0, color="white", lw=.8)
    axes[0].set_title("Point forecasts vs zero return")
    axes[0].set_xlabel("MAE improvement (%) · higher is better")
    axes[1].barh(tickers, f.crps_skill_empirical*100,
                 color=np.where(f.crps_skill_empirical >= 0, "#22c55e", "#ef4444"))
    axes[1].axvline(0, color="white", lw=.8)
    axes[1].set_title("Distributions vs empirical history")
    axes[1].set_xlabel("Truncated CRPS improvement (%)")
    y = np.arange(9)
    axes[2].barh(y-.18, f.coverage80*100, .35, color="#a78bfa", label="Raw")
    axes[2].barh(y+.18, c.calibrated_coverage80*100, .35, color="#38bdf8", label="Calibrated")
    axes[2].set_yticks(y, tickers)
    axes[2].axvline(80, color="white", ls="--")
    axes[2].set_xlim(0, 100)
    axes[2].set_title("80% prediction intervals")
    axes[2].set_xlabel("Observed coverage (%)")
    axes[2].legend()
    save(fig, "forecast_quality")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, values, title, cmap in [
        (axes[0], fm[fm.horizon == 2].pivot(index="ticker", columns="context", values="mae_skill_zero")*100,
         "MAE improvement vs zero return (%)", "RdYlGn"),
        (axes[1], tm[(tm.strategy == "confidence_cap3") & (tm.cost_multiplier == 1)].pivot(
            index="ticker", columns="context", values="net_return")*100,
         "Confidence-sized account return (%)", "RdYlGn"),
    ]:
        values = values.reindex(tickers)
        limit = max(abs(values.to_numpy()).max(), 1)
        im = ax.imshow(values.to_numpy(), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(values.columns)), values.columns)
        ax.set_yticks(range(9), tickers)
        ax.set_xlabel("Historical observations supplied")
        ax.set_title(title)
        ax.grid(False)
        for row in range(9):
            for col in range(len(values.columns)):
                ax.text(col, row, f"{values.iloc[row,col]:+.1f}", ha="center", va="center",
                        color="#0b1220", fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, "context_sensitivity")

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for ax, ticker in zip(axes.flat, tickers):
        b = bins[(bins.ticker == ticker) & (bins.context == primary) & (bins["count"] > 0)]
        ax.plot([.5, 1], [.5, 1], "--", color="#64748b")
        ax.scatter(b.mean_confidence, b.hit_rate, s=np.maximum(b["count"], 20),
                   color="#38bdf8", alpha=.8)
        for label_index, (_, row) in enumerate(b.iterrows()):
            ax.annotate(f"n={int(row['count'])}", (row.mean_confidence, row.hit_rate),
                        xytext=(5, 5 + 12*(label_index % 2)), textcoords="offset points", fontsize=8)
        ax.set(xlim=(.49, .9), ylim=(.2, .9), title=ticker,
               xlabel="Predicted probability of chosen direction", ylabel="Observed hit rate")
    fig.suptitle("Does higher confidence mean more correct calls? · frozen calibration", fontsize=17)
    save(fig, "confidence_reliability")

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    for ax, ticker in zip(axes.flat, tickers):
        prices = load_prices(ticker)
        future = pf[pf.ticker == ticker]
        ax.plot(np.arange(-59, 1), prices.iloc[-60:], color="#e2e8f0", lw=1.8)
        ax.plot(future.horizon, future["median"], color="#38bdf8", lw=1.8)
        ax.fill_between(future.horizon, future.q10, future.q90, color="#38bdf8", alpha=.22)
        ax.axvline(0, color="#64748b", ls="--")
        ax.set(title=ticker, xlabel="Trading observations relative to latest close",
               ylabel="Adjusted price / FX quote")
    fig.suptitle("Next 20 observations · price median and uncalibrated 10–90% range", fontsize=17)
    save(fig, "forecast_fans")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for market in ["India", "US", "FX"]:
        p = pred[(pred.market == market) & (pred.context == primary)]
        axes[0].hist(p.calibrated_probability_up, bins=np.linspace(0, 1, 31), alpha=.5,
                     label=market)
        axes[1].hist(abs(p.weight_cap3), bins=np.linspace(0, 3, 31), alpha=.5, label=market)
    axes[0].axvspan(.45, .55, color="#64748b", alpha=.2, label="No-position band")
    axes[0].set(title="Calibrated probability of a positive return",
                xlabel="Probability", ylabel="Forecast count")
    axes[1].set(title="Actual gross exposure under the 3x cap",
                xlabel="Gross exposure / account equity", ylabel="Decision count")
    axes[0].legend(); axes[1].legend()
    save(fig, "allocation")

    overview = pd.DataFrame({
        "Instrument": tickers, "Market": f.market.to_numpy(),
        "MAE skill": f.mae_skill_zero.to_numpy(),
        "Direction accuracy": f.direction_accuracy.to_numpy(),
        "Raw 1x return": raw.net_return.to_numpy(),
        "Confidence return": cap.net_return.to_numpy(),
        "Ending capital": cap.ending_capital.to_numpy(),
        "Worst drawdown": cap.max_drawdown.to_numpy(),
        "Buy & hold": passive.net_return.to_numpy(),
    })
    overview.to_csv(OUT / "overview.csv", index=False)
    market_summary = overview.groupby("Market", sort=False).agg({
        "MAE skill": "mean", "Direction accuracy": "mean", "Raw 1x return": "mean",
        "Confidence return": "mean", "Buy & hold": "mean",
    }).reset_index()
    market_summary.to_csv(OUT / "market_summary.csv", index=False)
    next_table = latest[latest.context == primary][[
        "ticker", "as_of", "last_adjusted_close", "median_h2_log_return",
        "refreshed_probability_up_h2", "refreshed_weight_cap3",
    ]].rename(columns={
        "ticker": "Instrument", "as_of": "Data through", "last_adjusted_close": "Last adjusted close",
        "median_h2_log_return": "Day +2 log return", "refreshed_probability_up_h2": "P(up), refreshed",
        "refreshed_weight_cap3": "Research exposure",
    })
    next_table["Day +2 log return"] = next_table["Day +2 log return"].map(lambda x: f"{x:.3%}")
    next_table["Research exposure"] = next_table["Research exposure"].map(
        lambda x: "0.000×" if abs(x) < 1e-10 else f"{x:+.3f}×")
    next_prices = pf[pf.horizon == 1].merge(
        latest[latest.context == primary][["ticker", "last_adjusted_close"]], on="ticker")
    next_prices["median_change"] = next_prices["median"] / next_prices.last_adjusted_close - 1
    next_prices.to_csv(OUT / "next_price_snapshot.csv", index=False)
    next_price_table = next_prices[["ticker", "last_adjusted_close", "median", "q10", "q90", "median_change"]].copy()
    for col in ["last_adjusted_close", "median", "q10", "q90"]:
        next_price_table[col] = [f"{v:,.5f}" if ticker.endswith("=X") else f"{v:,.2f}"
                                 for ticker, v in zip(next_price_table.ticker, next_price_table[col])]
    next_price_table["median_change"] = next_price_table.median_change.map(lambda x: f"{x:+.3%}")
    next_price_table = next_price_table.rename(columns={"ticker": "Instrument", "last_adjusted_close": "Last price",
        "median": "Next close: median", "q10": "10th percentile", "q90": "90th percentile",
        "median_change": "Median change"})
    stress = tm[(tm.context == primary) & (tm.strategy == "confidence_cap3")].pivot(
        index="ticker", columns="cost_multiplier", values="net_return").reindex(tickers).reset_index()
    stress.columns = ["Instrument", "Base costs", "Double costs"]
    ablations = t[t.strategy.isin(["confidence_cap3", "base_rate_sized", "volatility_scaled_sign"])].pivot(
        index="ticker", columns="strategy", values="net_return").reindex(tickers).reset_index()
    calib_table = c.reset_index()[["ticker", "raw_brier", "calibrated_brier", "base_brier",
                                  "calibrated_coverage80", "coefficient"]]
    source_links = """
    <li><a href="https://mistral.ai/news/timestral/">Mistral release announcement</a></li>
    <li><a href="https://huggingface.co/mistralai/Timestral-209M">209M checkpoint and model card</a></li>
    <li><a href="https://github.com/mistralai/timestral">Official inference source</a></li>
    <li><a href="https://ranaroussi.github.io/yfinance/reference/index.html">yfinance data client</a></li>
    <li><a href="https://otexts.com/fpp3/accuracy.html">Forecast accuracy and scaled errors</a></li>"""
    verdict = ("The primary test does not establish a dependable trading edge."
               if significant < 5 else
               "Some predictive improvements are visible, but this remains exploratory historical evidence.")
    intro = (
        f"{wins}/9 instruments beat a zero-return forecast on MAE; "
        f"{significant}/9 have a positive MAE improvement interval before multiple-testing adjustment. "
        f"Confidence-sized accounts made money in {profit}/9 cases and beat passive holding in "
        f"{beatpassive}/9. Those counts do not establish statistical or prospective profitability."
    )
    content = f"""
    <header><div class="eyebrow">EMPIRICAL RESEARCH • 20 OCTOBER 2026</div>
    <h1>Can Timestral trade?</h1><p class="lead">A real 209M-parameter checkpoint, nine Yahoo Finance histories,
    three context lengths, and a confidence-aware simulated account.</p>
    <div class="tags"><span>209,312,256 parameters · FP32</span><span>100 / 250 / 500 observations</span>
    <span>200 calibration + 500 test origins</span><span>No real orders</span></div></header>
    <section><h2>{verdict}</h2><p>{intro}</p>
    <div class="cards"><div><b>{wins}/9</b><span>beat zero-return MAE</span></div>
    <div><b>{f.direction_accuracy.mean():.1%}</b><span>mean direction accuracy</span></div>
    <div><b>{profit}/9</b><span>profitable sized accounts</span></div>
    <div><b>{cap.max_abs_exposure.max():.2f}×</b><span>largest observed exposure</span></div></div>
    <p class="warning">This is historical research with a newly released pretrained model. The public
    training description does not establish exclusion of these histories, so this is not a certified
    unseen-data or prospective backtest. Model benchmark rankings are not trading evidence.</p>
    </section>
    <section><h2>What happened to 100,000?</h2>
    <p>Primary context: 250 daily log returns. Every instrument gets a separate 100,000-unit account:
    INR for Indian equities and USD/INR, USD for U.S. instruments and EUR/USD or GBP/USD.
    These are quote-currency simulations, not a consolidated multi-currency account.
    The confidence strategy has a 3× ceiling; it does not automatically use 3×.</p>
    {table(overview, ["MAE skill", "Direction accuracy", "Raw 1x return", "Confidence return",
                     "Worst drawdown", "Buy & hold"], 2)}
    <h3>Market-level view</h3><p>Arithmetic averages of three instrument-level results, not market-wide
    estimates or a currency-converted portfolio. The 500 realized daily outcomes run from
    October 14, 2024 in India, October 21, 2024 in the U.S., and November 12, 2024 in FX,
    through October 19, 2026.</p>
    {table(market_summary, ["MAE skill", "Direction accuracy", "Raw 1x return",
                           "Confidence return", "Buy & hold"])}
    {image("equity", "Net simulated account equity for nine instruments")}
    <p>All curves include execution costs. Cash starts and stays at 100,000 with zero interest.
    Buy & hold includes overnight exposure and adjusted-price corporate-action effects.</p></section>
    <section><h2>Is it a good forecast?</h2>
    <p>The primary target is the individual daily return two observations ahead. A positive MAE skill
    beats forecasting zero; a positive probabilistic skill beats quantiles from the trailing history.
    Return R² is negative in {int((f.return_r2 < 0).sum())}/9 primary instruments.
    The median raw 80% interval covers {f.coverage80.median():.1%}
    of outcomes, and the calibrated median covers {c.calibrated_coverage80.median():.1%}: reasonable central uncertainty estimates
    can coexist with little useful directional information.</p>
    <p>This experiment uses univariate zero-shot daily-return inputs and separate price-level inputs.
    The findings apply to this setup. Fine-tuning and richer inputs such as volume, cross-asset
    covariates or market events would require their own evaluation.</p>
    {image("forecast_quality", "Point forecast skill, probabilistic skill and interval coverage")}
    <details><summary>Full primary forecast scores</summary>{table(f.reset_index())}</details>
    <h3>The price R² trap</h3><p>A forecast can closely track the price level yet fail to improve on
    tomorrow’s price being today’s price. These independent price-level diagnostics use the latest
    100 completed one-step origins, with fewer observed outcomes at longer horizons. For example,
    MSFT price R² is 0.968, yet its MAE is 0.57% worse than simply repeating the last price.</p>
    {table(pm[pm.horizon == 1], ["mae_skill_random_walk"])}
    </section>
    <section><h2>Does confidence deserve a larger bet?</h2>
    <p>Raw probabilities come from the model’s return quantiles. A regularized logistic map is fitted
    on 200 earlier examples and frozen for the entire test. It can reverse the raw signal when earlier
    data indicate an inverse relationship; negative coefficients in the calibration table expose that.</p>
    <div class="formula">confidence = max(P(up), 1 − P(up))<br>
    conviction = clip((confidence − 0.55) / 0.15, 0, 1)<br>
    exposure = direction × min(cap, conviction × 10% / trailing annual volatility)</div>
    <p>Below 55% confidence: cash. From 55–70%: increasing risk allocation. At 70% and above:
    full volatility-scaled allocation, capped at 1× or 3×. Trailing volatility uses 60 completed
    observations and a 1% floor. This mapping was fixed before results, not optimized on the test.</p>
    {image("confidence_reliability", "Forecast confidence compared with observed directional hit rate")}
    <p>Marker size represents sample count. Empty confidence bands contain no supporting evidence.
    A 65% estimate that wins only 50% of the time is overconfident, not a reason to borrow more.</p>
    {image("allocation", "Probability and realized exposure distributions")}
    <details><summary>Calibration scores · Brier is better when lower</summary>
    {table(calib_table, ["calibrated_coverage80"])}</details>
    <h3>Is confidence adding information, or just reducing exposure?</h3>
    <p>Two diagnostic baselines were added during implementation: <code>base_rate_sized</code>
    sizes from the calibration period's constant up frequency, without model information;
    <code>volatility_scaled_sign</code> removes the confidence multiplier but keeps the calibrated
    direction and volatility target. These explanatory comparisons were not used to tune the strategy.</p>
    {table(ablations, ["confidence_cap3", "base_rate_sized", "volatility_scaled_sign"])}
    <p>SPY illustrates the distinction: confidence sizing earns 4.04%, while sizing from the
    historical up frequency alone earns 3.90%. Most of that result does not require useful
    model-dependent confidence. USD/INR reaches 2.57× exposure but loses 2.31% after costs;
    a larger position does not repair a weak forecast.</p>
    </section>
    <section><h2>Does more history help?</h2>
    {image("context_sensitivity", "Forecast and strategy results across three context lengths")}
    <p>All three settings are reported. The 250-observation setting was selected before seeing
    performance; picking whichever setting won this test would require a new untouched evaluation.</p>
    </section>
    <section><h2>Costs can erase the apparent edge</h2>
    <p>One-way execution costs: India 10 bps, U.S. 5 bps, EUR/USD and GBP/USD 2 bps,
    USD/INR 3 bps. Every entry, exit and rebalance is charged, including position drift.
    Financing is 8% annually on gross exposure above 1×; short borrow is 3% on short notional.
    Carry charges use calendar days. These are scenario assumptions, not broker quotes.</p>
    {table(stress, ["Base costs", "Double costs"])}
    <details><summary>All strategies and both cost scenarios</summary>
    {table(tm[tm.context == primary], ["net_return", "max_drawdown", "cagr"])}</details>
    </section>
    <section><h2>What does it forecast next?</h2>
    <p>The return signal predicts the individual daily return at observation +2. It waits one
    complete observation before entry. The latest-only calibration below is refreshed using the most
    recent 200 known test outcomes; this refresh never changes the historical performance table.
    Research exposure is a formula output, not a validated trade recommendation.</p>
    {table(next_table, ["P(up), refreshed"])}
    <h3>Next observed close from the price-level model</h3>
    <p>These are independent price-level forecasts from 250 adjusted closes, dated from the
    October 19 snapshot. The raw 10th–90th percentile range describes one future observation.</p>
    {table(next_price_table)}
    {image("forecast_fans", "Twenty-observation price forecasts with uncertainty bands")}
    <p>Price ranges above are raw marginal model intervals, not calibrated trading-risk limits.
    They do not include a joint path distribution or a promise that a whole future path remains inside
    the band. Horizons are trading observations, not calendar dates.</p></section>
    <section><h2>How to interpret the evidence</h2>
    <ol><li><strong>Beat simple baselines first.</strong> A powerful general forecasting model may
    have little useful information about daily asset returns.</li>
    <li><strong>Separate prediction from implementation.</strong> Better MAE does not automatically
    give better directional trades, and gross profit can disappear after turnover and borrowing.</li>
    <li><strong>Check calibration before increasing exposure.</strong> Small calibration samples and
    regime changes make probability estimates uncertain. Low realized exposure may be the correct output.</li>
    <li><strong>Demand forward evidence.</strong> Freeze the method and collect genuinely new outcomes
    after release before treating a historical winner as a deployable strategy.</li></ol>
    <p>Uncertainty intervals use 2,000 circular moving-block resamples of 20 observations. They are
    exploratory, conditional on the fitted calibration, and not corrected for all contexts, assets,
    horizons and strategies tested. They do not account for uncertainty in model training or deployment.</p>
    <p>Additional limitations: selected surviving instruments; retrospective Yahoo adjustments;
    indicative FX daily quotes; no bid/ask or executable intraday data; no actual FX carry or cross-currency
    conversion; simplified funding; no borrow-availability or margin-call simulation; daily drawdowns miss
    intraday stress. Overnight Indian short positions and FX exposure require suitable tradable contracts.
    No missing outcomes are forward-filled and no returns are winsorized.</p>
    </section>
    <section><h2>Reproduce and inspect</h2>
    <p>Checkpoint revision: <code>{manifest["model_revision"]}</code>. Inference uses official code,
    FP32 weights and default symmetric averaging. The CPU and a custom Vulkan matrix-multiplication
    backend share the work. The latter passed a 36-origin check against CPU forecasts across price/return
    inputs and all contexts; the maximum difference was approximately 0.0013% of interval width.
    Integer quantization was rejected after a numerical check; none of these results use it.</p>
    <p>Files alongside this report include every test prediction, account path, strategy metric,
    confidence bucket, latest forecast, source manifest and environment lock.
    <code>PROTOCOL.md</code> explains the predeclared design. Scripts can resume from forecast caches.</p>
    <details><summary>Data provenance</summary>{table(pd.DataFrame(manifest["assets"]))}</details>
    <h3>Primary references</h3><ul>{source_links}</ul></section>
    <footer>Timestral market experiment · research simulation · generated locally from recorded forecasts</footer>
    """
    css = """
    :root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#080f1c;color:#dbe5f3;
    font:16px/1.65 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:48px 28px}
    header{padding:35px 0 50px;border-bottom:1px solid #253247}h1{font-size:clamp(42px,7vw,76px);
    line-height:1.05;letter-spacing:-3px;margin:20px 0}h2{font-size:30px;line-height:1.25;color:#f1f5f9}
    h3{font-size:21px}.eyebrow{letter-spacing:3px;color:#38bdf8;font-size:12px;font-weight:700}
    .lead{font-size:23px;max-width:900px;color:#94a3b8}.tags{display:flex;flex-wrap:wrap;gap:10px}
    .tags span{background:#142238;border:1px solid #253b56;border-radius:20px;padding:5px 12px;
    font-size:12px}section{padding:35px 0;border-bottom:1px solid #253247}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}
    .cards div{background:#111d31;border:1px solid #243750;border-radius:12px;padding:24px}
    .cards b{font-size:42px;display:block;color:#38bdf8}.cards span{font-size:13px;color:#9aadc5}
    .warning{border-left:3px solid #eab308;background:#252310;padding:18px 22px}
    figure{margin:25px 0}img{width:100%;height:auto;border-radius:10px}
    .table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;
    display:block;overflow:auto;margin:20px 0}th{background:#1d2f47;text-align:left;white-space:nowrap}
    th,td{padding:10px 13px;border-bottom:1px solid #26354a}tr:nth-child(even){background:#111c2b}
    a{color:#38bdf8}code{overflow-wrap:anywhere;font-size:13px;color:#a5d8f5}
    .formula{font:14px/2 ui-monospace,monospace;white-space:normal;background:#111d31;
    padding:25px;border-radius:10px}details{padding:15px 0}summary{cursor:pointer;color:#7dd3fc}
    footer{padding:45px 0;color:#64748b;font-size:12px}
    @media(max-width:700px){main{padding:24px 15px}.cards{grid-template-columns:repeat(2,1fr)}
    .cards b{font-size:32px}h1{letter-spacing:-1px}}
    """
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Timestral-209M · Financial market evaluation</title><style>{css}</style></head>
    <body><main>{content}</main></body></html>"""
    (OUT / "report.html").write_text(document)
    markdown = f"""# Timestral-209M market assessment

{verdict}

{intro}

Open `results/report.html` for figures, per-instrument accounts, calibration,
confidence sizing, sensitivity comparisons and latest forecasts.

Primary context: 250 observations. Per instrument: 200 prior calibration
origins and 500 test origins; forecast r[t+2] using observations through t,
enter at close t+1 and exit at close t+2. All trading results include the
predeclared costs and financing assumptions.

The CSV `results/overview.csv` gives the headline values. `results/market_summary.csv`
contains arithmetic averages of three instruments per group, not a market index
or a pooled multi-currency portfolio.

The public model training description does not rule out historical-data overlap.
This is not a prospective validation or evidence of dependable future profits.

See [protocol](../PROTOCOL.md), [model card](https://huggingface.co/mistralai/Timestral-209M),
[official code](https://github.com/mistralai/timestral) and
[forecast metrics](https://otexts.com/fpp3/accuracy.html).
"""
    (OUT / "REPORT.md").write_text(markdown)
    print(json.dumps(summary, indent=2))
    print(overview.to_string(index=False))
    print("HTML report:", OUT / "report.html")


if __name__ == "__main__":
    main()
