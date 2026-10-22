"""Compact report for exactly 100 model-only trades in each of nine cases."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulate_100_trades import ASSETS, INTERVALS, INITIAL, OUT, ROOT, trade_returns

NAMES={"1m":"1 minute","1h":"1 hour","1d":"1 day"}


def main():
    rows=[]; ledgers={}
    for asset in ASSETS:
        for interval in INTERVALS:
            name=f"{asset['ticker']}_{interval}"
            result=json.loads((OUT/f"result_{name}.json").read_text())
            assert result["execution"] == "close_to_next_close", name
            ledger=pd.read_csv(OUT/f"trades_{name}.csv")
            assert len(ledger)==100 and list(ledger.trade)==list(range(1,101))
            assert (pd.to_datetime(ledger.entry_bar,utc=True)<pd.to_datetime(ledger.exit_bar,utc=True)).all()
            directions=np.where(ledger.direction=="long",1,-1)
            np.testing.assert_array_equal(directions,np.sign(ledger.forecast_close-ledger.last_close))
            gross,net=trade_returns(directions,ledger.entry_close,ledger.exit_close,asset["cost_bps"])
            np.testing.assert_allclose(gross,ledger.gross_return,atol=1e-12)
            np.testing.assert_allclose(net,ledger.net_return,atol=1e-12)
            np.testing.assert_allclose(INITIAL*np.cumprod(1+net),ledger.net_equity,rtol=1e-12)
            np.testing.assert_allclose(ledger.net_equity.iloc[-1],result["net_ending"],rtol=1e-12)
            rows.append(result);ledgers[asset["ticker"],interval]=ledger
    summary=pd.DataFrame([{k:v for k,v in r.items() if k!="source"} for r in rows])
    summary.to_csv(OUT/"summary.csv",index=False)
    pd.concat(ledgers.values(),ignore_index=True).to_csv(OUT/"all_900_trades.csv",index=False)
    plt.rcParams.update({"figure.facecolor":"#0b1220","axes.facecolor":"#111c2f",
        "savefig.facecolor":"#0b1220","text.color":"#e2e8f0","axes.labelcolor":"#cbd5e1",
        "xtick.color":"#94a3b8","ytick.color":"#94a3b8","axes.edgecolor":"#334155",
        "grid.color":"#334155","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for ri,asset in enumerate(ASSETS):
        for ci,interval in enumerate(INTERVALS):
            d=ledgers[asset["ticker"],interval];ax=axes[ri,ci]
            ax.plot(range(101),np.r_[INITIAL,d.gross_equity]/1000,color="#94a3b8",label="Before costs")
            ax.plot(range(101),np.r_[INITIAL,d.net_equity]/1000,color="#38bdf8",lw=2,label="After costs")
            ax.axhline(100,color="#64748b",ls=":",lw=.8)
            ax.set(title=f"{asset['label']} · {NAMES[interval]}",xlabel="Completed trades",
                   ylabel=f"Account · thousands of {asset['currency']}")
            ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle("Timestral-209M · 100 trades per case · fixed 1× positions",fontsize=18)
    fig.savefig(OUT/"equity_100_trades.png",dpi=170,bbox_inches="tight");plt.close(fig)
    table=summary[["market","label","interval","trades","gross_return","net_return","net_ending","gross_win_rate"]].copy()
    table["interval"]=table.interval.map(NAMES)
    for col in ["gross_return","net_return","gross_win_rate"]:
        table[col]=table[col].map(lambda x:f"{x:+.2%}" if col!="gross_win_rate" else f"{x:.0%}")
    table["net_ending"]=[f"{v:,.2f} {currency}" for v,currency in zip(table.net_ending,summary.currency)]
    table.columns=["Category","Instrument","Data","Trades","Before costs","After costs","Final account","Gross win rate"]
    b64=base64.b64encode((OUT/"equity_100_trades.png").read_bytes()).decode()
    dates=summary[["label","interval","first_trade_bar","last_trade_bar"]].copy()
    dates["interval"]=dates.interval.map(NAMES)
    dates.columns=["Instrument","Data","First entry bar","Last exit bar"]
    audit={"status":"passed","cases":9,"trades_per_case":100,"total_trades":900,
           "checks":["model-only direction","entry before exit","100 distinct trades per case",
                     "two-fill trading costs","compounded equity","summary agrees with trade ledger"]}
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2))
    text=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Timestral · 100-trade simulation</title>
    <style>body{{background:#0b1220;color:#e2e8f0;font:16px/1.6 system-ui;margin:0}}main{{max-width:1200px;margin:auto;padding:45px 25px}}
    h1{{font-size:46px;line-height:1.1}}h2{{margin-top:40px}}.tag{{color:#38bdf8;letter-spacing:2px;font-size:12px}}
    .lead{{font-size:20px;color:#a9bbd0}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:25px 0;display:block;overflow:auto}}
    th,td{{padding:11px 13px;text-align:left;border-bottom:1px solid #28384e;white-space:nowrap}}th{{background:#1b2d45}}
    tr:nth-child(even){{background:#111d30}}img{{width:100%;height:auto}}a{{color:#38bdf8}}.note{{color:#a9bbd0;font-size:14px}}
    .rule{{background:#14243a;padding:18px 22px;border-left:3px solid #38bdf8}}summary{{cursor:pointer;color:#38bdf8}}
    </style></head><body><main><div class="tag">TIMESTRAL-209M · YAHOO FINANCE · DATA THROUGH 19 OCTOBER 2026</div>
    <h1>100 trades. Three markets.<br>Minute, hourly and daily data.</h1>
    <p class="lead">One instrument per category: Reliance for India, SPY for the U.S., and EUR/USD for currencies.
    Nine cases, exactly 100 completed trades each. Every account starts at 100,000 in its quote currency.</p>
    <p class="rule">Feed the model the last 250 closing prices. If its next-close median is higher, go long;
    if lower, go short. Hold until the next closing quote and close the trade. Position size stays at 1×.
    Exact no-change predictions are skipped until 100 actual trades are collected.</p>
    <p>No confidence calibration, trade filter, volatility sizing or discretionary signal changes are used.</p>
    <h2>Results after 100 trades</h2>{table.to_html(index=False,border=0)}
    <p class="note">“Before costs” is the raw price simulation. “After costs” charges both entry and exit:
    0.10% per fill for Reliance, 0.05% for SPY, and 0.02% for EUR/USD. Profit is compounded between trades.
    These are cost assumptions; funding, borrow fees and FX carry are excluded.</p>
    <img src="data:image/png;base64,{b64}" alt="Nine account curves, each covering exactly 100 trades">
    <h2>Simulation assumptions</h2><p>This is a model-only paper simulation with ideal fills at the last
    observed closing quote immediately after the forecast. It does not model inference or order latency.
    Yahoo's EUR/USD minute OHLC bars are point quotes, so consecutive closes are used consistently for all cases.</p>
    <p class="note">The bar size changes the elapsed time covered by 100 trades. Yahoo hourly equity data
    can include a shorter final bar at the session close. No missing prices are filled.
    Daily stock closes are adjusted for corporate actions. The released 209M checkpoint is used with its default
    forecast settings and no fine-tuning; historical training overlap remains unknown.</p>
    <details><summary>Exact test periods</summary>{dates.to_html(index=False,border=0)}</details>
    <p><a href="summary.csv">Download results</a> · <a href="all_900_trades.csv">All 900 trades</a> ·
    <a href="equity_100_trades.png">Export chart</a></p>
    <p class="note">Audit passed: 100 trades in every case, signal direction, timestamps, costs and account balances verified.</p>
    </main></body></html>"""
    (OUT/"report.html").write_text(text)
    print(table.to_string(index=False));print("Audit passed: 900 trades")


if __name__ == "__main__":
    main()
