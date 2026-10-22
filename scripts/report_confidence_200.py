"""Explain measured confidence first, then report 1,800 signal opportunities."""
import base64
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_200 import ASSETS, INTERVALS, OUT, ROOT, INITIAL, pnl, select_leverage, LEVERAGES, THRESHOLDS

NAMES={"1m":"Minute","1h":"Hourly","1d":"Daily"}


def table(frame,percent=(),money=()):
    d=frame.copy()
    for column in d:
        if column in percent:d[column]=d[column].map(lambda v:f"{v:+.2%}" if pd.notna(v) else "—")
        elif column in money:d[column]=d[column].map(lambda v:f"{v:,.2f}")
        elif pd.api.types.is_float_dtype(d[column]):d[column]=d[column].map(lambda v:f"{v:.4f}" if pd.notna(v) else "—")
    return d.to_html(index=False,border=0)


def embedded(name,alt):
    data=base64.b64encode((OUT/name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def main():
    rows=[];ledgers={};reliability=[]
    for asset in ASSETS:
        for interval in INTERVALS:
            key=f"{asset['ticker']}_{interval}"
            result=json.loads((OUT/f"result_{key}.json").read_text())
            ledger=pd.read_csv(OUT/f"opportunities_{key}.csv")
            cal=pd.read_csv(OUT/f"calibration_{key}.csv")
            assert result["counting_mode"]=="opportunities" and len(ledger)==200
            result["target_reached"]=True
            result["flat_forecast_skips"]=int(sum(ledger.direction==0))
            result["low_confidence_skips"]=int(sum((ledger.direction!=0)&(ledger.leverage==0)))
            ledger["skip_reason"]=np.where(ledger.direction==0,"unchanged_forecast",
                np.where(ledger.leverage==0,"profit_confidence_below_threshold","executed"))
            (OUT/f"result_{key}.json").write_text(json.dumps(result,indent=2))
            ledger.to_csv(OUT/f"opportunities_{key}.csv",index=False)
            ledger.loc[ledger.executed].to_csv(OUT/f"trades_{key}.csv",index=False)
            assert result["executed"]+result["skipped"]==200
            assert ledger.entry_bar.is_unique
            assert (pd.to_datetime(ledger.entry_bar,utc=True)<pd.to_datetime(ledger.exit_bar,utc=True)).all()
            assert pd.to_datetime(cal.exit_bar,utc=True).max()<pd.to_datetime(ledger.entry_bar,utc=True).min()
            expected=select_leverage(ledger[["calibrated_profit_confidence_1x","calibrated_profit_confidence_2x",
                                           "calibrated_profit_confidence_3x"]].to_numpy())
            probabilities=ledger[["calibrated_profit_confidence_1x","calibrated_profit_confidence_2x",
                                  "calibrated_profit_confidence_3x"]].to_numpy()
            assert (np.diff(probabilities,axis=1)<=1e-10).all()
            proposed=ledger.direction!=0
            assert (ledger.loc[proposed,"raw_profit_confidence_1x"]<=
                    ledger.loc[proposed,"raw_direction_confidence"]+1e-10).all()
            source_path=ROOT/result["source"]["source"]
            assert hashlib.sha256(source_path.read_bytes()).hexdigest()==result["source"]["sha256"]
            expected[ledger.direction==0]=0
            np.testing.assert_array_equal(expected,ledger.leverage)
            net,fees,funding,exposure=pnl(ledger.exit_close/ledger.entry_close,ledger.direction,
                                         ledger.leverage,asset["cost_bps"]/1e4,ledger.holding_years)
            np.testing.assert_allclose(net,ledger.net_return,atol=1e-12)
            np.testing.assert_allclose(fees,ledger.fees,atol=1e-12)
            np.testing.assert_allclose(funding,ledger.funding,atol=1e-12)
            np.testing.assert_allclose(INITIAL*np.cumprod(1+net),ledger.net_equity,rtol=1e-12)
            np.testing.assert_allclose(ledger.net_equity.iloc[-1],result["ending"],rtol=1e-12)
            assert (ledger.leverage<=3).all()
            assert (ledger.loc[ledger.leverage==0,"net_return"]==0).all()
            assert len(pd.read_csv(OUT/f"trades_{key}.csv"))==result["executed"]
            rows.append(result);ledgers[asset["ticker"],interval]=ledger
            valid=ledger[ledger.direction!=0]
            for kind,column in [("Raw","raw_profit_confidence_1x"),("Calibrated","calibrated_profit_confidence_1x")]:
                for lower,upper in zip(np.arange(0,1,.1),np.arange(.1,1.01,.1)):
                    group=valid[(valid[column]>=lower)&(valid[column]<upper)]
                    if len(group):
                        observed=group.hypothetical_1x_profit.mean();n=len(group);z=1.96
                        center=(observed+z*z/(2*n))/(1+z*z/n)
                        half=z*np.sqrt(observed*(1-observed)/n+z*z/(4*n*n))/(1+z*z/n)
                        reliability.append({"ticker":asset["ticker"],"interval":interval,"kind":kind,
                            "count":n,"mean_confidence":group[column].mean(),"observed_profit_rate":observed,
                            "low95":center-half,"high95":center+half})
    total=sum(r["opportunities"] for r in rows);executed=sum(r["executed"] for r in rows)
    assert total==1800
    all_ledger=pd.concat(ledgers.values(),ignore_index=True)
    all_ledger.to_csv(OUT/"all_1800_opportunities.csv",index=False)
    all_ledger.loc[all_ledger.executed].to_csv(OUT/"all_executed_trades.csv",index=False)
    summary=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows])
    summary.to_csv(OUT/"summary.csv",index=False)
    probability_scores=pd.DataFrame([{"ticker":r["ticker"],"interval":r["interval"],
        "raw_brier":r["raw_scores"]["brier"],"calibrated_brier":r["calibrated_scores"]["brier"],
        "constant_brier":r["constant_scores"]["brier"],"raw_log_loss":r["raw_scores"]["log_loss"],
        "calibrated_log_loss":r["calibrated_scores"]["log_loss"],
        "constant_log_loss":r["constant_scores"]["log_loss"]} for r in rows])
    probability_scores.to_csv(OUT/"probability_scores.csv",index=False)
    confidence_values=pd.DataFrame([{"Instrument":asset["label"],"Data":NAMES[interval],
        "Mean raw direction confidence":d.raw_direction_confidence.mean(),
        "Mean raw profit confidence":d.raw_profit_confidence_1x.mean(),
        "Mean calibrated profit confidence":d.calibrated_profit_confidence_1x.mean(),
        "Observed profit fraction":d.hypothetical_1x_profit.mean()}
        for asset in ASSETS for interval in INTERVALS
        for d in [ledgers[asset["ticker"],interval].query("direction != 0")]])
    confidence_values.to_csv(OUT/"confidence_measured.csv",index=False)
    reliability=pd.DataFrame(reliability);reliability.to_csv(OUT/"reliability_bins.csv",index=False)
    plt.rcParams.update({"figure.facecolor":"#0b1220","axes.facecolor":"#111c2f","savefig.facecolor":"#0b1220",
        "text.color":"#e2e8f0","axes.labelcolor":"#cbd5e1","xtick.color":"#94a3b8","ytick.color":"#94a3b8",
        "axes.edgecolor":"#334155","grid.color":"#334155","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for row,asset in enumerate(ASSETS):
        for col,interval in enumerate(INTERVALS):
            ax=axes[row,col];d=ledgers[asset["ticker"],interval];x=np.arange(201)
            for field,label,color in [("baseline_equity","Unfiltered 1×","#94a3b8"),
                    ("raw_policy_equity","Raw confidence policy","#a78bfa"),
                    ("net_equity","Calibrated policy","#38bdf8")]:
                ax.plot(x,np.r_[INITIAL,d[field]]/1000,label=label,color=color,lw=1.8)
            ax.axhline(100,color="#64748b",ls=":",lw=.8)
            ax.set(title=f"{asset['label']} · {NAMES[interval]}",xlabel="Signal opportunities",
                   ylabel=f"Thousands of {asset['currency']}");ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("Confidence, skipping and leverage · 200 opportunities per case · net of costs",fontsize=17)
    fig.savefig(OUT/"equity.png",dpi=170,bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    for row,asset in enumerate(ASSETS):
        for col,interval in enumerate(INTERVALS):
            ax=axes[row,col];b=reliability[(reliability.ticker==asset["ticker"])&(reliability.interval==interval)]
            ax.plot([0,1],[0,1],"--",color="#64748b")
            for kind,color in [("Raw","#a78bfa"),("Calibrated","#38bdf8")]:
                d=b[b.kind==kind]
                ax.scatter(d.mean_confidence,d.observed_profit_rate,s=np.maximum(d["count"],15),color=color,label=kind,alpha=.8)
                ax.errorbar(d.mean_confidence,d.observed_profit_rate,
                    yerr=np.maximum(0,np.stack([d.observed_profit_rate-d.low95,d.high95-d.observed_profit_rate])),
                    fmt="none",color=color,alpha=.5,capsize=2)
                for i,(_,record) in enumerate(d.iterrows()):
                    ax.annotate(str(int(record["count"])),(record.mean_confidence,record.observed_profit_rate),
                                xytext=(4,-12 if record.observed_profit_rate>.9 else 5+(12 if kind=="Calibrated" else 0)),
                                textcoords="offset points",fontsize=7)
            ax.axvline(.55,color="#fbbf24",ls=":",lw=1)
            ax.set(xlim=(-.02,1),ylim=(-.02,1),title=f"{asset['label']} · {NAMES[interval]}",
                   xlabel="Predicted profit probability",ylabel="Observed profitable fraction");ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle("Confidence check on later data · hypothetical 1× trades, including skipped signals",fontsize=16)
    fig.savefig(OUT/"reliability.png",dpi=170,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5),constrained_layout=True)
    labels=[f"{r['label']}\n{NAMES[r['interval']]}" for r in rows];bottom=np.zeros(9)
    for field,label,color in [("skipped","Skipped","#64748b"),("leverage_1x_trades","1×","#22c55e"),
                              ("leverage_2x_trades","2×","#38bdf8"),("leverage_3x_trades","3×","#a78bfa")]:
        values=np.array([r[field] for r in rows]);ax.bar(np.arange(9),values,bottom=bottom,color=color,label=label);bottom+=values
    ax.set_xticks(np.arange(9),labels);ax.set(ylabel="Opportunities",ylim=(0,220),title="What happened to each set of 200 signals?")
    ax.legend(ncol=4,frameon=False);fig.savefig(OUT/"allocation.png",dpi=170,bbox_inches="tight");plt.close(fig)
    results=pd.DataFrame([{"Instrument":r["label"],"Data":NAMES[r["interval"]],"Opportunities":r["opportunities"],
        "Executed":r["executed"],"Skipped":r["skipped"],"1× / 2× / 3×":f"{r['leverage_1x_trades']} / {r['leverage_2x_trades']} / {r['leverage_3x_trades']}",
        "Net return":r["return"],"Final balance":f"{r['ending']:,.2f} {r['currency']}","Drawdown":r["max_drawdown"]} for r in rows])
    confidence_display=confidence_values.copy()
    for column in confidence_display.columns[2:]:
        confidence_display[column]=confidence_display[column].map(lambda v:f"{v:.2%}")
    comparisons=pd.DataFrame([{"Instrument":r["label"],"Data":NAMES[r["interval"]],
        "Unfiltered 1×":r["baseline"]["return"],"Raw confidence":r["raw_policy"]["return"],
        "Calibrated, cap 1×":r["calibrated_1x"]["return"],"Calibrated, cap 3×":r["return"]} for r in rows])
    comparisons.to_csv(OUT/"comparison.csv",index=False)
    filled=all_ledger.loc[all_ledger.executed,["ticker","interval","entry_bar","exit_bar","direction",
        "leverage","selected_confidence","net_return","net_equity"]].copy()
    filled=filled.rename(columns={"selected_confidence":"Estimated profit probability",
        "net_return":"Net trade return","net_equity":"Ending balance"})
    filled["direction"]=filled.direction.map({1.:"Long",-1.:"Short"})
    filled["interval"]=filled.interval.map(NAMES)
    dates=summary[["label","interval","calibration_first","calibration_last_exit","first_entry","last_exit"]].copy()
    max_leverage=max([0]+all_ledger.leverage.tolist())
    improved=int((probability_scores.calibrated_brier<probability_scores.raw_brier).sum())
    beat_constant=int((probability_scores.calibrated_brier<probability_scores.constant_brier).sum())
    audit={"status":"passed","opportunities":1800,"executed":executed,"skipped":1800-executed,"cases":9,
        "checks":["200 opportunities per case","strict calibration/test embargo","direction and leverage rules",
                  "cost-adjusted probabilities","entry/exit chronology","fees and funding identity",
                  "cash has zero costs","all equity paths and final balances"]}
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2))
    explanation="""Confidence means <strong>the estimated probability that the proposed trade earns positive
    next-bar P&amp;L after the costs used here</strong>. It is measured separately at 1×, 2× and 3× exposure.
    It is different from the probability that price merely moves in the predicted direction.
    A 70% estimate should correspond to roughly 70 profitable trades per 100 comparable later calls;
    it is not a 70% return or a guarantee of profit."""
    content=f"""<div class="tag">TIMESTRAL-209M · CONFIDENCE CALIBRATION · DATA THROUGH 19 OCTOBER 2026</div>
    <h1>What does confidence mean?</h1><p class="lead">{explanation}</p>
    <section><h2>How we measure it</h2><ol>
    <li><b>Read the forecast distribution.</b> Feed the last 250 closes to the pinned 209M model.
    Its 21 next-close quantiles define an approximate price CDF. The median chooses long or short.</li>
    <li><b>Find the break-even price.</b> Include entry and exit fees, financing above 1×, and short borrow.
    For a long, raw profit confidence is the probability above that price; for a short, it is the probability below it.</li>
    <li><b>Calibrate using earlier outcomes.</b> The preceding 300 forecast origins supply proposed trades at all
    three exposure levels. Fit a monotone sigmoid between raw scores and realized profitable outcomes, giving
    each origin total weight one. Center and regularize the map; clamp later scores to the observed calibration range.</li>
    <li><b>Freeze and test.</b> The map is frozen before the 200 later opportunities. Those outcomes never fit
    the map or alter the thresholds. Reliability curves compare predicted probabilities with realized profit frequencies;
    Brier score and log loss assess probability quality against both raw scores and a constant historical rate.</li></ol>
    <div class="formula">raw long confidence = 1 − F(break-even price)<br>
    raw short confidence = F(break-even price)<br>
    calibrated confidence = sigmoid(a × centered log-odds(raw confidence) + b)</div>
    <p>Example: a price of 100 with a 0.10% fee on each fill needs a next price above about 100.20 to make a
    long trade profitable at 1×. If the forecast puts only 15% probability above that threshold, raw profit
    confidence is 15%, even if the probability of any upward move is above 50%.</p>
    <p class="note">Outside the 1st–99th percentile forecast grid, use boundary scores rather than inventing tail
    forecasts. A winning-trade probability also says nothing by itself about the size of wins versus losses.</p></section>
    <section><h2>How confidence controls skipping and exposure</h2>
    <table><tr><th>Action</th><th>Required calibrated profit probability at that exposure</th></tr>
    <tr><td>3×</td><td>At least 70%</td></tr><tr><td>2×</td><td>At least 60%, if 3× does not qualify</td></tr>
    <tr><td>1×</td><td>At least 55%, if higher levels do not qualify</td></tr><tr><td>Skip</td><td>No level qualifies, or the median forecast is unchanged</td></tr></table>
    <p>The policy chooses the highest qualifying level. Higher exposure faces its own financing-adjusted
    break-even threshold. Every accepted position closes at the next quote. Cash earns zero interest.</p></section>
    <section><h2>Results: 200 opportunities in each of nine cases</h2>
    <div class="cards"><div><b>1,800</b><span>opportunities</span></div><div><b>{executed}</b><span>executed trades</span></div>
    <div><b>{1800-executed}</b><span>skipped signals</span></div><div><b>{max_leverage:.0f}×</b><span>highest exposure actually used</span></div></div>
    <p>Same three instruments: Reliance, SPY and EUR/USD. Each account begins with 100,000 in its quote
    currency. As requested, 200 counts opportunities; skipped signals are not called trades.</p>
    <p>The calibrated policy executed {executed} trade and skipped {1800-executed} opportunities.
    There were {sum(r['leverage_2x_trades']+r['leverage_3x_trades'] for r in rows)} trades above 1×.
    This small executed sample gives very little evidence about trading performance or the higher-exposure tiers.</p>
    {table(results,percent=["Net return","Drawdown"])}
    <h3>Executed trade record</h3>
    {table(filled,percent=["Estimated profit probability","Net trade return"],money=["Ending balance"])}
    {embedded('allocation.png','Executed and skipped opportunities at each leverage level')}
    {embedded('equity.png','Net account values for calibrated and uncalibrated policies')}
    <h3>Same-period comparisons</h3><p>Every column uses the same 200 later opportunities for its case.
    Unfiltered 1× follows the raw median direction. Raw confidence applies the same thresholds without calibration.
    The calibrated 1× comparison isolates skipping from the additional exposure allowed by the 3× policy.</p>
    {table(comparisons,percent=["Unfiltered 1×","Raw confidence","Calibrated, cap 1×","Calibrated, cap 3×"])}</section>
    <section><h2>Did calibration measure confidence more accurately?</h2>
    <p>Calibrated Brier scores improve over raw scores in {improved}/9 cases and over the constant historical
    profit rate in {beat_constant}/9. Lower scores are better. Improvements against raw scores can come from
    learning the base win rate; they do not automatically establish useful ranking or profitable trades.</p>
    <p>Brier score is the average of <code>(predicted probability − outcome)²</code>, where outcome is 1
    for a profitable trade and 0 otherwise. A 70% call scores 0.09 when it wins and 0.49 when it loses,
    penalizing confident errors. Reliability bins compare mean predicted probability with the profitable fraction.</p>
    {table(confidence_display)}
    <details><summary>Brier scores and log loss</summary>{table(probability_scores)}</details>
    {embedded('reliability.png','Predicted profit probabilities versus observed later outcomes')}
    <p class="note">These checks use hypothetical 1× outcomes for every non-flat model proposal, including
    signals the policy skips. Marker labels show sample counts; error bars show 95% Wilson intervals for
    observed frequencies. These descriptive intervals do not adjust for serial dependence, calibration-fit
    or pretraining uncertainty. The yellow line is the 55% 1× gate.</p>
    <p>If the calibrated probability stays below the gate, the strategy remains in cash. A flat cash curve
    indicates abstention and preservation of nominal capital, not evidence that the model predicted returns well.</p></section>
    <section><h2>Costs, timing and exact records</h2><p>Per-fill cost assumptions are unchanged: Reliance 0.10%,
    SPY 0.05%, EUR/USD 0.02%. Annual financing is 8% on actual exposure above 1×; annual borrow is 3% on short
    notional. Charges use elapsed calendar time, including overnight and weekend gaps. Entry fees reduce
    investable equity. Actual broker terms and FX carry are not supplied by Yahoo.</p>
    <p>As in the previous report, these are ideal close-to-next-close paper fills, with no inference or
    execution latency. Daily prices are retrospectively adjusted, hourly session-end bars can be shorter,
    and missing prices are not filled. Model pretraining overlap remains unknown. The thresholds and
    calibration design were recorded before evaluating these outcomes.</p>
    <details><summary>Calibration and test dates</summary>{table(dates)}</details>
    <p><a href="all_1800_opportunities.csv">All 1,800 opportunities</a> ·
    <a href="all_executed_trades.csv">Executed trades only</a> · <a href="summary.csv">Summary CSV</a> ·
    <a href="probability_scores.csv">Confidence scores</a> · <a href="PROTOCOL.md">Protocol</a></p>
    <p class="note">Audit: passed. Chronology, counts, thresholds, fees, financing and equity paths verified.
    Reference: <a href="https://scikit-learn.org/stable/modules/calibration.html">probability calibration and reliability diagrams</a>.</p></section>"""
    css="""body{background:#0b1220;color:#e2e8f0;font:16px/1.65 system-ui;margin:0}main{max-width:1250px;margin:auto;padding:45px 25px}
    h1{font-size:55px;line-height:1.1;letter-spacing:-2px}h2{font-size:30px;line-height:1.25}h3{font-size:22px}
    .tag{color:#38bdf8;letter-spacing:2px;font-size:12px}.lead{font-size:21px;color:#bdcce0}section{border-top:1px solid #26374e;margin-top:32px;padding-top:20px}
    .formula{padding:20px;background:#14243a;border-left:3px solid #38bdf8;font:14px/2 monospace}.note{font-size:13px;color:#a5b6cd}
    table{display:block;overflow:auto;border-collapse:collapse;width:100%;font-size:12px;margin:25px 0}th,td{padding:10px 12px;border-bottom:1px solid #2b3c52;text-align:left;white-space:nowrap}
    th{background:#1b2d45}tr:nth-child(even){background:#111d30}img{width:100%;height:auto;margin:20px 0}a,summary{color:#38bdf8}summary{cursor:pointer}
    li{margin:14px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}.cards div{padding:20px;background:#142238;border-radius:10px}
    .cards b{display:block;font-size:40px;color:#38bdf8}.cards span{font-size:13px;color:#a5b6cd}@media(max-width:650px){.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:40px}}
    """
    (OUT/"report.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Timestral · Confidence, skipping and leverage</title><style>{css}</style></head><body><main>{content}</main></body></html>')
    (OUT/"run_summary.json").write_text(json.dumps({"opportunities":1800,"executed":executed,"skipped":1800-executed,
        "max_leverage":max_leverage,"brier_improves_raw_cases":improved,"brier_beats_constant_cases":beat_constant},indent=2))
    (OUT/"configuration.json").write_text(json.dumps({"model":"mistralai/Timestral-209M",
        "model_revision":json.loads((ROOT/"data/manifest.json").read_text())["model_revision"],
        "context":250,"calibration_origins":300,"test_opportunities_per_case":200,
        "counting_mode":"opportunities","leverage_levels":LEVERAGES.tolist(),
        "profit_probability_thresholds":THRESHOLDS.tolist(),"annual_financing_rate":.08,
        "annual_short_borrow_rate":.03,"assets":ASSETS,"intervals":INTERVALS},indent=2))
    print(results.to_string(index=False));print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
