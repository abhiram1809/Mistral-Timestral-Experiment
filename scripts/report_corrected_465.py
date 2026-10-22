"""Results and audit of calibrated directional confidence and after-cost P&L."""
import base64
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_200 import INITIAL, pnl, select_leverage, equity_stats
from corrected_465 import ROOT, WINDOW, TARGET, load_frame, calibrated_probability, Coordinator
from direction_465 import OUT
from replay_lower_confidence import formatted_table, NAMES


def embedded(name, alt):
    return '<img alt="'+alt+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'


def report():
    completion=json.loads((OUT/"completion.json").read_text());config=json.loads((OUT/"configuration.json").read_text())
    source=json.loads((OUT/"source_manifest.json").read_text());ledger=pd.read_csv(OUT/"evaluated_signals.csv")
    trades=pd.read_csv(OUT/"executed_trades.csv")
    assert not ledger.duplicated(["ticker","interval","origin"]).any()
    assert len(trades)==ledger.executed.sum()==completion["executed"]
    np.testing.assert_array_equal(trades.combined_trade_number,np.arange(1,len(trades)+1))
    pd.testing.assert_frame_equal(
        trades.drop(columns="combined_trade_number").reset_index(drop=True),
        ledger[ledger.executed].sort_values(["exit_bar","ticker","interval"],kind="stable").reset_index(drop=True))
    totals=ledger.groupby("round").executed.sum().cumsum()
    assert (totals.iloc[:-1]<TARGET).all()
    assert (pd.to_datetime(ledger.calibration_latest_exit,utc=True)<pd.to_datetime(ledger.entry_bar,utc=True)).all()
    pinned=json.loads((ROOT/"results/more-yahoo-2026-10-21/manifest.json").read_text())["original_source_hashes"]
    for path,value in pinned.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value
    rows=[];confidence_rows=[];forecast_rows=[];calibration_checks=[];reliability=[];quality_rows=[];influence_rows=[]
    cached=Coordinator(source["cases"])
    (OUT/"forecast-cache").mkdir(exist_ok=True)
    plt.rcParams.update({"figure.facecolor":"#0b1220","axes.facecolor":"#111c2f","savefig.facecolor":"#0b1220",
        "text.color":"#e2e8f0","axes.labelcolor":"#cbd5e1","xtick.color":"#94a3b8","ytick.color":"#94a3b8",
        "axes.edgecolor":"#334155","grid.color":"#334155","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for case_number,item in enumerate(source["cases"]):
        key=f"{item['ticker']}_{item['interval']}";path=ROOT/item["source"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
        bars=load_frame(path);d=ledger[(ledger.ticker==item["ticker"])&(ledger.interval==item["interval"])].copy()
        assert d.origin.is_monotonic_increasing
        active=d[d.executed].copy();scored=d[d.forecast_median.notna()].copy()
        assert len(scored)==len(d)
        ids=d.origin.to_numpy(dtype=int)
        q_all=np.stack([cached.quantiles[key][int(i)] for i in ids])
        np.testing.assert_allclose(q_all[:,10],d.forecast_median,rtol=1e-12)
        actual_returns=(d.exit_close/d.entry_close-1).to_numpy()
        predicted_returns=(d.forecast_median/d.entry_close-1).to_numpy()
        if item["interval"]=="1d":
            review_threshold=.05 if item["ticker"]=="EURUSD=X" else .5
            for pos in np.flatnonzero(abs(actual_returns)>review_threshold):
                quoted=d.iloc[int(pos)]
                quality_rows.append({"Instrument":item["label"],"Entry":quoted.entry_bar,"Exit":quoted.exit_bar,
                    "Entry quote":quoted.entry_close,"Exit quote":quoted.exit_close,"Quoted move":actual_returns[pos],
                    "Executed":bool(quoted.executed),"Review threshold":review_threshold})
        rw_mae=np.mean(np.abs(actual_returns))
        return_mae=np.mean(np.abs(predicted_returns-actual_returns))
        return_rmse=np.sqrt(np.mean((predicted_returns-actual_returns)**2))
        from executed_465 import QUANTILE_LEVELS
        cache_ids=np.arange(int(ids.min())-WINDOW-1,int(ids.max())+1)
        np.savez_compressed(OUT/"forecast-cache"/f"{key}.npz",origins=cache_ids,
            quantiles=np.stack([cached.quantiles[key][int(i)] for i in cache_ids]),
            levels=QUANTILE_LEVELS,source_sha256=np.array(item["sha256"]))
        lo=np.argmin(abs(QUANTILE_LEVELS-.1));hi=np.argmin(abs(QUANTILE_LEVELS-.9))
        coverage=np.mean((d.exit_close.to_numpy()>=q_all[:,lo])&(d.exit_close.to_numpy()<=q_all[:,hi]))
        q_returns=q_all/d.entry_close.to_numpy()[:,None]-1
        errors=actual_returns[:,None]-q_returns
        pinball=float(np.mean(np.maximum(QUANTILE_LEVELS*errors,(QUANTILE_LEVELS-1)*errors)))
        naive_errors=actual_returns[:,None]
        naive_pinball=float(np.mean(np.maximum(QUANTILE_LEVELS*naive_errors,(QUANTILE_LEVELS-1)*naive_errors)))
        np.testing.assert_allclose(naive_pinball,.5*rw_mae,rtol=1e-12)
        forecast_rows.append({"Instrument":item["label"],"Bars":NAMES[item["interval"]],
            "Return MAE":return_mae,"Return RMSE":return_rmse,
            "No-change return MAE":rw_mae,"MAE improvement over no-change":1-return_mae/rw_mae if rw_mae else np.nan,
            "80% interval coverage":coverage,"Unchanged realized prices":float(np.mean(actual_returns==0)),
            "Direction accuracy on price changes":float(d.loc[actual_returns!=0,"direction_correct"].mean()) if np.any(actual_returns!=0) else np.nan,
            "Mean quantile loss":pinball,"Quantile loss improvement over no-change":1-pinball/naive_pinball if naive_pinball else np.nan})
        predicted_from_counts=np.where(d.calibration_support>=20,(d.calibration_wins+1)/(d.calibration_support+2),.5)
        np.testing.assert_allclose(predicted_from_counts,d.direction_confidence,atol=1e-12)
        assert (active.calibration_support>=20).all()
        # Independently rebuild sampled calibration windows from the cached model quantiles.
        from confidence_200 import information
        from executed_465 import QUANTILE_LEVELS
        for test_row in d.iloc[np.unique(np.linspace(0,len(d)-1,min(5,len(d)),dtype=int))].itertuples():
            t=int(test_row.origin);ids=np.arange(t-WINDOW-1,t+1)
            assert set(map(int,ids)).issubset(cached.quantiles[key])
            q=np.stack([cached.quantiles[key][int(i)] for i in ids])
            values=information(bars,ids,q,QUANTILE_LEVELS,item["cost_bps"]/1e4)
            h=slice(0,WINDOW)
            raw=np.tile(values["direction_probability"][h,None],(1,3))
            labels=values["direction"][h]*(values["exit"][h]/values["entry"][h]-1)>0
            p,_,_,_=calibrated_probability(raw,np.tile(labels[:,None],(1,3)),values["direction"][h]!=0,
                [values["direction_probability"][-1]]*3)
            np.testing.assert_allclose(p[0],test_row.direction_confidence,atol=1e-12)
            calibration_checks.append({"ticker":item["ticker"],"interval":item["interval"],"origin":t})
        if len(scored):
            probabilities=np.tile(scored.direction_confidence.to_numpy()[:,None],(1,3))
            chosen=select_leverage(probabilities);chosen[scored.direction==0]=0
            np.testing.assert_array_equal(chosen,scored.leverage)
            n,f,c,_=pnl(scored.exit_close/scored.entry_close,scored.direction,scored.leverage,item["cost_bps"]/1e4,scored.holding_years)
            np.testing.assert_allclose(n,scored.net_return,atol=1e-12)
            np.testing.assert_allclose(f,scored.fees,atol=1e-12);np.testing.assert_allclose(c,scored.funding,atol=1e-12)
        assert (d.net_return>-1).all()
        stats,equity=equity_stats(d.net_return.to_numpy());np.testing.assert_allclose(equity,d.net_equity,rtol=1e-12)
        before=np.r_[INITIAL,equity[:-1]]
        if len(active):
            net1,*_=pnl(active.exit_close/active.entry_close,active.direction,1.,item["cost_bps"]/1e4,active.holding_years)
            one,_=equity_stats(net1)
            gross=active.leverage*active.direction*(active.exit_close/active.entry_close-1)
            gross_stats,_=equity_stats(gross.to_numpy())
        else:one={"return":0.};gross_stats={"return":0.}
        best=None;without_best=None
        if len(active) and active.net_return.max()>0:
            best=active.loc[active.net_return.idxmax()]
            without_best=float(np.prod(1+active.drop(index=best.name).net_return.to_numpy())-1)
        influence_rows.append({"Instrument":item["label"],"Bars":NAMES[item["interval"]],"Trades":len(active),
            "Net return":stats["return"],"Largest winning trade":float(best.net_return) if best is not None else None,
            "Winner entry":best.entry_bar if best is not None else None,
            "Winner exit":best.exit_bar if best is not None else None,"Return excluding that winner":without_best})
        shadow=select_leverage(d[[f"profit_confidence_{j}x" for j in (1,2,3)]].to_numpy());shadow[d.direction==0]=0
        shadow_net,*_=pnl(d.exit_close/d.entry_close,d.direction,shadow,item["cost_bps"]/1e4,d.holding_years)
        shadow_stats,_=equity_stats(shadow_net)
        buy_hold_net=float(pnl(float(d.exit_close.iloc[-1]/d.entry_close.iloc[0]),1.,1.,item["cost_bps"]/1e4,0.)[0])
        row={"ticker":item["ticker"],"label":item["label"],"interval":item["interval"],"currency":item["currency"],
            "executed":len(active),"skipped":len(d)-len(active),"signals_processed":len(d),"actual_forecast_decisions":len(scored),
            "trades_1x":int(sum(active.leverage==1)),"trades_2x":int(sum(active.leverage==2)),
            "trades_3x":int(sum(active.leverage==3)),**stats,"profit_cash":stats["ending"]-INITIAL,
            "fees_cash":float(np.sum(before*d.fees)),"funding_cash":float(np.sum(before*d.funding)),
            "mean_selected_confidence":float(active.selected_confidence.mean()) if len(active) else None,
            "net_win_rate":float((active.net_return>0).mean()) if len(active) else None,
            "direction_hit_rate":float(active.direction_correct.mean()) if len(active) else None,
            "profit_gated_shadow_executions":int(sum(shadow>0)),"profit_gated_shadow_return":shadow_stats["return"],
            "same_selected_trades_1x_return":one["return"],"same_selected_trades_without_costs":gross_stats["return"],
            "buy_hold_return":buy_hold_net,
            "without_largest_winner_return":without_best,
            "first_entry":d.entry_bar.iloc[0],"last_exit":d.exit_bar.iloc[-1],
            "return_mae":return_mae,"return_rmse":return_rmse,"return_pinball":pinball,
            "return_pinball_skill":1-pinball/naive_pinball if naive_pinball else None,
            "raw_direction_brier":float(np.mean((scored.raw_direction_confidence-scored.direction_correct.astype(float))**2)),
            "calibrated_direction_brier":float(np.mean((scored.direction_confidence-scored.direction_correct.astype(float))**2))}
        rows.append(row)
        confidence_rows.append({"Instrument":item["label"],"Bars":NAMES[item["interval"]],"Trades":len(active),
            "Mean estimated direction probability":row["mean_selected_confidence"],"Observed direction accuracy":row["direction_hit_rate"],
            "Observed net win rate":row["net_win_rate"],"Raw direction Brier":row["raw_direction_brier"],
            "Calibrated direction Brier":row["calibrated_direction_brier"]})
        if len(active):
            for lower in np.arange(.5,1,.1):
                part=active[(active.selected_confidence>=lower)&(active.selected_confidence<lower+.1)]
                if len(part):reliability.append({"ticker":item["ticker"],"interval":item["interval"],"n":len(part),
                    "mean_confidence":part.selected_confidence.mean(),"direction_accuracy":part.direction_correct.mean(),
                    "net_win_rate":(part.net_return>0).mean()})
        ax=axes[case_number//3,case_number%3];x=np.arange(1,len(d)+1)
        ax.plot(np.r_[0,x],np.r_[INITIAL,equity]/1000,color="#38bdf8",lw=2,label="Confidence-sized, net")
        if len(active):
            matched=np.zeros(len(d));matched[np.flatnonzero(d.executed.to_numpy())]=net1
            _,matched_equity=equity_stats(matched)
            ax.plot(np.r_[0,x],np.r_[INITIAL,matched_equity]/1000,color="#94a3b8",lw=1.4,ls="--",label="Same trades at 1×")
        ax.set(title=f"{item['label']} · {NAMES[item['interval']]}",xlabel="Chronological signals",ylabel=f"Thousands of {item['currency']}")
        ax.grid(alpha=.2)
    handles,labels=max((ax.get_legend_handles_labels() for ax in axes.flat),key=lambda pair:len(pair[0]))
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle(f"{len(trades)} actual trades · causal calibration · original 55% / 60% / 70% gates",fontsize=16)
    fig.savefig(OUT/"equity.png",dpi=155,bbox_inches="tight");plt.close(fig)
    summary=pd.DataFrame(rows);summary.to_csv(OUT/"summary.csv",index=False)
    pd.DataFrame(forecast_rows).to_csv(OUT/"forecast_metrics.csv",index=False)
    pd.DataFrame(quality_rows).to_csv(OUT/"data_quality_flags.csv",index=False)
    pd.DataFrame(influence_rows).to_csv(OUT/"winner_sensitivity.csv",index=False)
    pd.DataFrame(reliability).to_csv(OUT/"reliability.csv",index=False)
    fig,ax=plt.subplots(figsize=(13,5),constrained_layout=True);bottom=np.zeros(9)
    for column,label,color in [("trades_1x","1×","#22c55e"),("trades_2x","2×","#38bdf8"),("trades_3x","3×","#a78bfa")]:
        values=summary[column].to_numpy();ax.bar(np.arange(9),values,bottom=bottom,label=label,color=color);bottom+=values
    ax.set_xticks(np.arange(9),[f"{r['label']}\n{NAMES[r['interval']]}" for r in rows]);ax.set(ylabel="Executed trades",title="Completed trades, excluding every skipped signal")
    ax.legend(ncol=3,frameon=False);fig.savefig(OUT/"executions.png",dpi=155,bbox_inches="tight");plt.close(fig)
    audit={"status":"passed","target":TARGET,"executed":len(trades),"target_reached":len(trades)>=TARGET,
        "unique_trades":True,"original_gates":[.55,.60,.70],"source_hashes_unchanged":True,
        "checks":["unique case and origin keys","complete historical trades, skips excluded","past-only calibration with strict embargo",
            "minimum support and original leverage gates","calibration estimates reconstructed from prior model predictions",
            "fee and funding identities","separate account compounding","original experiment hashes preserved","fixed full-round stopping rule"]}
    audit["independently_rebuilt_calibration_windows"]=len(calibration_checks)
    audit.update(source_quality_verified=False,posthoc_daily_quote_review_flags=len(quality_rows),
        directly_executed_flagged_intervals=sum(r["Executed"] for r in quality_rows),
        source_quality_note="Arithmetic and chronology pass; extreme Yahoo quotes remain uncorrected, including their effect on later contexts and calibration.")
    assert len(trades)<TARGET or not completion["history_exhausted"]
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2))
    results=pd.DataFrame([{"Instrument":r["label"],"Bars":NAMES[r["interval"]],"Trades":r["executed"],"Skips":r["skipped"],
        "1× / 2× / 3×":f"{r['trades_1x']} / {r['trades_2x']} / {r['trades_3x']}","Net return":r["return"],
        "Ending balance":f"{r['ending']:,.2f} {r['currency']}","Max drawdown":r["max_drawdown"]} for r in rows])
    comparison=pd.DataFrame([{"Instrument":r["label"],"Bars":NAMES[r["interval"]],"Same executions at 1×":r["same_selected_trades_1x_return"],
        "Confidence-sized, net":r["return"],"Same leverage, no costs":r["same_selected_trades_without_costs"],
        "Buy-and-hold, net":r["buy_hold_return"],
        "Fees paid":f"{r['fees_cash']:,.2f} {r['currency']}","Financing + borrow":f"{r['funding_cash']:,.2f} {r['currency']}"} for r in rows])
    dates=pd.DataFrame([{"Instrument":r["label"],"Bars":NAMES[r["interval"]],"First entry (UTC)":r["first_entry"],"Last exit (UTC)":r["last_exit"]} for r in rows])
    forecast_frame=pd.DataFrame(forecast_rows)
    accuracy_columns=["Instrument","Bars","Return MAE","Return RMSE","No-change return MAE","MAE improvement over no-change"]
    probability_columns=["Instrument","Bars","Mean quantile loss","Quantile loss improvement over no-change","80% interval coverage","Unchanged realized prices","Direction accuracy on price changes"]
    fx_influence=next(r for r in influence_rows if r["Instrument"]=="EUR/USD" and r["Bars"]=="Daily")
    ecb_doc=""
    if (OUT/"ecb-check-2008-09.csv").exists():
        reference=pd.read_csv(OUT/"ecb-check-2008-09.csv");comparison_rows=[]
        fx_bars=cached.frames["EURUSD=X_1d"]
        for observation in reference.itertuples():
            stamp=pd.Timestamp(observation.TIME_PERIOD,tz="UTC")
            quote=float(fx_bars.loc[stamp,"Close"]) if stamp in fx_bars.index else np.nan
            comparison_rows.append({"Date":observation.TIME_PERIOD,"ECB USD per EUR":observation.OBS_VALUE,
                "Yahoo daily close":quote,"Relative difference":quote/observation.OBS_VALUE-1})
        ecb_doc='<h3>Independent reference check</h3>'+formatted_table(pd.DataFrame(comparison_rows),percentages=("Relative difference",))+'''<p>ECB reference fixings and Yahoo daily closes have different observation times, so exact agreement is not expected.
The 2008-09-08 discrepancy is approximately 5.9%: Yahoo reports 1.5050 while the ECB reference rate is 1.4214 USD per EUR.
That Yahoo entry quote drives the largest winning trade, a 2× short exiting at 1.41485 on 2008-09-09.
This is an unresolved source-data concern, not a validated trading gain. The original Yahoo inputs remain unchanged.</p>
<p><a href="https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=2008-09-05&amp;endPeriod=2008-09-10&amp;format=csvdata">ECB primary data</a>
 · <a href="ecb-check-2008-09.csv">saved reference response</a> · <a href="ecb-check-source.json">comparison provenance</a></p>'''
    gained=int(sum(summary["return"]>0));lost=int(sum(summary["return"]<0))
    active_cases=int(sum(summary.executed>0))
    leverage_better=int(sum((summary.executed>0)&(summary["return"]>summary.same_selected_trades_1x_return+1e-12)))
    leverage_worse=int(sum((summary.executed>0)&(summary["return"]<summary.same_selected_trades_1x_return-1e-12)))
    mae_better=sum(r["MAE improvement over no-change"]>0 for r in forecast_rows)
    status="Target reached" if len(trades)>=TARGET else "Available history exhausted before target"
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Timestral · directional confidence · {len(trades)} trades</title><style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:40px 28px 65px}}h1{{font-size:clamp(30px,4vw,46px);line-height:1.2}}h2{{font-size:25px}}p{{max-width:1100px}}section{{margin:30px 0}}.eyebrow{{color:#38bdf8;text-transform:uppercase;letter-spacing:.1em;font-weight:700;font-size:13px}}.lead{{font-size:20px}}.box{{background:#111c2f;border:1px solid #334155;border-radius:14px;padding:20px 26px}}.scroll{{overflow-x:auto;border:1px solid #334155;border-radius:10px;margin:16px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}}th,td{{text-align:right;padding:11px 13px;border-bottom:1px solid #25334b}}th:first-child,td:first-child{{text-align:left}}th{{color:#93c5fd;background:#111c2f}}img{{width:100%;height:auto;border-radius:12px}}a{{color:#7dd3fc}}.muted{{color:#94a3b8;font-size:14px}}li{{margin:7px 0}}@media(max-width:760px){{main{{padding:26px 16px}}}}</style></head><body><main>
<div class="eyebrow">Timestral-209M · rolling directional confidence · simulated executions</div>
<h1>{status}: {len(trades):,} completed trades.</h1>
<p class="lead">The target was at least 465 executions combined. This run processed {len(ledger):,} chronological signals across nine independent
accounts and executed {len(trades):,} trades. Skipped signals are excluded. The gates remain <strong>55% / 60% / 70%</strong> for 1× / 2× / 3×.</p>
<section class="box"><h2>First: how confidence is measured now</h2>
<p><strong>Confidence now estimates the probability that the model gets the next price direction right.</strong>
Timestral’s median chooses long or short. Its 21 quantiles give a raw probability of price moving in that direction, using the CDF at the current close.
The CDF is linearly interpolated between quantiles, with endpoint values of 1% and 99% outside their range.
Profit after fees is measured separately and is not promised by a high directional confidence.</p>
<p>For each decision, take the preceding 300 forecast origins whose outcomes exit strictly before the current entry. Put their raw direction
probabilities into ten fixed bins: 0–10%, 10–20%, and so on. Count prior correct directions W and observations N in the
current raw-score bin. The smoothed estimate is <strong>(W + 1) / (N + 2)</strong>, a Beta(1,1) posterior mean.</p>
<p>A bin needs at least 20 observations. Otherwise it uses the whole past window; if that also has fewer than 20, confidence is 50% and
the trade skips. Flat realized price changes count as incorrect. Leverage does not change the price-direction event, so all levels use this same
directional probability. Select the highest level meeting its own fixed gate.</p>
<p>This replaces the old frozen, input-range-clamped pooled sigmoid. The new counts update using completed past observations.
Estimates come from that changing past window, and no future price or current test outcome enters calibration. The original one-trade run is preserved.
<strong>The event definition changed from positive net profit to correct forecast direction.</strong> This is a distinct strategy, not a reproduction
of the original profit-gated experiment. Each row also retains a separate estimated net-profit probability at 1×, 2× and 3×.</p>
<p class="muted">This is a newly specified custom estimator, not an official Mistral confidence function. A 60% estimate is an estimated
direction probability, not 60% statistical certainty or a promise of profit. Its observed calibration is measured below.</p></section>
<section><h2>Actual simulated account results</h2><p>Each account starts with 100,000: INR for Reliance and USD for SPY and EUR/USD.
{gained} accounts gain, {lost} lose and {9-gained-lost} stay unchanged. The results below include costs; different currencies are not added together.</p>
<div class="box"><strong>Source-data limitation:</strong> these are simulations on Yahoo's quoted history, including unresolved daily quote anomalies.
The only positive account, EUR/USD daily, returns {fx_influence['Net return']:.2%}; excluding its single largest winner changes that to
<strong>{fx_influence['Return excluding that winner']:.2%}</strong>. That winner uses a Yahoo entry quote that differs sharply from an independent ECB reference.
The numerical result does not establish reliable market profitability. Details and sensitivity checks follow below.</div>
<p class="muted">A case with zero executions stays in cash. Its 0% account return does not demonstrate a profitable forecasting strategy;
forecast errors are still assessed for its skipped signals below.</p>
<p><strong>Assessment:</strong> {active_cases} cases execute trades. Confidence-based leverage improves the return versus the same executions
at 1× in {leverage_better} cases and worsens it in {leverage_worse}. The model beats a no-change forecast on return MAE in {mae_better} of nine cases.
Directional confidence alone therefore needs to be judged alongside move size, costs and the actual net account results.</p>
{formatted_table(results,percentages=("Net return","Max drawdown"))}
{embedded('executions.png','Actual completed trade counts and leverage for all nine cases')}
{embedded('equity.png','Separate simulated account paths, net confidence-sized results versus the same executions at one times leverage')}</section>
<section><h2>Leverage, cost effects and a passive reference</h2>{formatted_table(comparison,percentages=("Same executions at 1×","Confidence-sized, net","Same leverage, no costs","Buy-and-hold, net"))}
<p>The 1× and cost-free counterfactuals execute exactly the same selected signals. They compound independently. Removing costs is a diagnostic,
not an alternative net result. The separate buy-and-hold reference stays long at 1× from the first evaluated entry to the last exit,
charging one entry and exit, with the same adjusted prices and fee assumption.</p></section>
<section><h2>Data quality and dependence on individual winners</h2>
<p>Post hoc quote review flags daily FX moves above 5% and daily equity moves above 50% in absolute value. This flags {len(quality_rows)} intervals,
including Reliance's quoted +337% jump and subsequent −77% reversal in July 2005. These rules are diagnostic review flags, not trading filters.
The primary 561-trade run retains every original input. Untraded anomalous bars can still affect later model contexts and calibration windows.</p>
{formatted_table(pd.DataFrame(quality_rows),percentages=("Quoted move","Review threshold"))}
{ecb_doc}
{formatted_table(pd.DataFrame(influence_rows),percentages=("Net return","Largest winning trade","Return excluding that winner"))}
<p class="muted">The winner-exclusion calculation is a post hoc fragility diagnostic. It removes one executed round trip and compounds
the other recorded returns with their original signals and exposures. It is neither an executable selection rule nor a backtest on repaired data.
Proper repair would require validating the quotes and rerunning all affected model contexts and causal calibration.</p></section>
<section><h2>Does directional confidence match later outcomes?</h2>
{formatted_table(pd.DataFrame(confidence_rows),percentages=("Mean estimated direction probability","Observed direction accuracy","Observed net win rate"))}
<p class="muted">Brier scores use every model forecast in each evaluated span, including skipped trades. Lower Brier is better.
Always estimating a 50% probability gives a Brier score of 0.25. Directional accuracy and after-cost win rate answer different questions;
the latter can be much lower when fees exceed a small correct move.</p>
<p>Calibration can still be inaccurate on later data. Neither a high recent win frequency nor a high trade count establishes profitable
forecasting. The observed net win rates, losses and drawdowns above are the assessment.</p>
<h3>Forecast error against a no-change prediction</h3>
{formatted_table(forecast_frame[accuracy_columns],percentages=tuple(accuracy_columns[2:]))}
{formatted_table(forecast_frame[probability_columns],percentages=tuple(probability_columns[2:]))}
<p class="muted">Errors are measured on next-bar fractional returns, avoiding deceptively high price-level R² from persistent price levels.
Positive MAE improvement means lower error than predicting the next close equals the current close; a negative value means worse.
The model’s 10th–90th percentile interval should cover approximately 80% of outcomes if its marginal intervals are calibrated.
Mean quantile loss averages pinball loss over the 21 forecast quantiles in return units. Lower is better; its improvement compares against
putting every quantile at the current close. These metrics include skipped signals. The conditional direction-accuracy column excludes unchanged realized prices only for that diagnostic;
the trading policy and its calibration still count unchanged prices as directional misses.</p>
<h3>Separate profit-gated diagnostic on the same data</h3>
{formatted_table(pd.DataFrame([{"Instrument":r["label"],"Bars":NAMES[r["interval"]],"Direction-gated trades":r["executed"],"Profit-gated trades":r["profit_gated_shadow_executions"],"Profit-gated net return":r["profit_gated_shadow_return"]} for r in rows]),percentages=("Profit-gated net return",))}
<p class="muted">The diagnostic applies 55% / 60% / 70% to estimated positive net-profit probability instead. It has a separate account;
its executions are not added to the primary 465-trade target.</p></section>
<section><h2>Sampling and dates</h2>{formatted_table(dates)}
<p>The cases are independent chronological backtests. Each round advances 128 candidate bars per active case. The run stops after the first
complete round with at least 465 executions, so the total may exceed 465. This predetermined allocation gives every case a chance to trade;
it is not a single synchronized portfolio or the first 465 trades in global calendar time.</p>
<p>Windows, bins, prior, minimum sample size, case order, gates and stopping rule were fixed before evaluating this run’s P&amp;L.
They were not selected by return optimization. This methodology redesign follows the already-reviewed earlier results, so it is a new
retrospective sensitivity study, not a clean untouched validation set.</p></section>
<section><h2>Model, data and assumptions</h2><ul>
<li>Actual pinned official Timestral-209M checkpoint, FP32, symmetric averaging; 250 closing prices predict the next close.</li>
<li>Yahoo’s expanded daily, hourly and minute snapshots. Old quote values were retained where changes were only tiny rounding differences.
Forecast caches are reused only after the complete 250-value float32 context matches exactly.</li>
<li>Ideal current-close to next-observed-close fills, including calendar gaps. Holding durations used for cost calculations are realized
historical intervals. Same-close fills omit inference and execution latency.</li>
<li>Each selected signal opens and fully closes a one-bar position. Adjacent selected signals reopen and pay another entry fee,
including when the model keeps the same direction. Positions are not netted across signals.</li>
<li>Per entry and exit: Reliance 10 basis points, SPY 5, FX 2. Annual financing: 8% on actual exposure above 1×; short borrow: 3%.
Entry fees reduce investable exposure. Cash interest, taxes, FX carry and intrabar margin liquidation are omitted.</li>
<li>Retrospective adjusted prices and possible model-pretraining overlap limit prospective conclusions. There are no live orders,
duplicated trades, forced threshold passes or bootstrap copies.</li></ul>
<p class="muted">For calibration background, see <a href="https://proceedings.mlr.press/v70/guo17a.html">Guo et al., On Calibration of Modern Neural Networks</a>.
The particular rolling histogram and smoothing rules used here are specified in the protocol.</p></section>
<section class="box"><h2>Download every result</h2><ul>
<li><a href="executed_trades.csv">Every executed trade</a> · <a href="summary.csv">account summary</a> · <a href="evaluated_signals.csv">every signal, including skips</a></li>
<li><a href="reliability.csv">confidence reliability bins</a> · <a href="configuration.json">configuration</a> · <a href="source_manifest.json">data hashes</a></li>
<li><a href="forecast_metrics.csv">Forecast errors and interval coverage</a></li>
<li><a href="data_quality_flags.csv">Quote review flags</a> · <a href="winner_sensitivity.csv">largest-winner sensitivity</a></li>
<li><a href="PROTOCOL.md">protocol</a> · <a href="audit.json">audit</a> · <a href="corrected-465-results.zip">research bundle ZIP</a></li>
<li><a href="../confidence-200/report.html">Original one-trade report</a> · <a href="../frozen-465/report.html">Frozen-calibration assessment</a>
 · <a href="../more-yahoo-2026-10-21/report.html">Yahoo data download</a></li></ul>
<p>Arithmetic and chronology audit passed: unique completed trades, causal calibration, unchanged gates, sample support,
independently rebuilt prior calibration windows, fees, funding and account balances. Source-quote quality remains unresolved as documented above.</p>
<p class="muted">The ZIP includes exact input CSVs, all forecasts used for testing and calibration, research scripts and the result tables.
Model weights are cached separately in the workspace.</p></section></main></body></html>'''
    (OUT/"report.html").write_text(doc)
    with zipfile.ZipFile(OUT/"corrected-465-results.zip","w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in ["executed_trades.csv","evaluated_signals.csv","summary.csv","forecast_metrics.csv","configuration.json","source_manifest.json","PROTOCOL.md","audit.json","reliability.csv","data_quality_flags.csv","winner_sensitivity.csv","equity.png","executions.png","report.html"]:
            z.write(OUT/name,name)
        for name in ["ecb-check-2008-09.csv","ecb-check-source.json"]:
            if (OUT/name).exists():z.write(OUT/name,name)
        for item in source["cases"]:
            z.write(ROOT/item["source"],item["source"])
        for path in sorted((OUT/"forecast-cache").glob("*.npz")):
            z.write(path,str(path.relative_to(OUT)))
        for path in sorted((ROOT/"scripts").glob("*.py")):
            z.write(path,str(path.relative_to(ROOT)))
    print(json.dumps(audit,indent=2));print(summary[["label","interval","executed","return","ending","max_drawdown"]].to_string(index=False))


if __name__=="__main__":report()
