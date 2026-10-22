"""Causal probability calibration and at least 465 unique executions combined.

Nine independent chronological streams share a fixed round-robin sampling rule.
Gates remain 55/60/70. Only past outcomes update Beta-binomial histogram estimates.
"""
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from confidence_200 import ROOT, ASSETS, INTERVALS, INITIAL, CONTEXT, information, pnl, select_leverage
from executed_465 import CachedInference, QUANTILE_LEVELS

OUT = ROOT / "results/corrected-465"
DATA = ROOT / "data/corrected-465"
WINDOW = 300
MIN_SUPPORT = 20
TARGET = 465
ROUND_SIZE = 128
FIRST_TEST = CONTEXT + WINDOW
EDGES = np.linspace(0., 1., 11)


def write_json(path, value):
    tmp = path.with_suffix(".tmp.json"); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frame(path):
    b = pd.read_csv(path, index_col=0); b.index = pd.to_datetime(b.index, utc=True)
    return b.sort_index()


def calibrated_probability(raw_history, outcomes, valid, current):
    """Posterior mean in fixed probability bins; small bins back off to all past rows."""
    raw_history = np.asarray(raw_history); outcomes = np.asarray(outcomes); valid = np.asarray(valid)
    old_bins = np.clip(np.searchsorted(EDGES, raw_history, side="right")-1, 0, 9)
    new_bins = np.clip(np.searchsorted(EDGES, current, side="right")-1, 0, 9)
    probability = []; supports = []; wins = []; fallback = []
    for j in range(3):
        group = valid & (old_bins[:, j] == new_bins[j])
        backoff = int(sum(group)) < MIN_SUPPORT
        if backoff: group = valid
        n = int(sum(group)); w = int(sum(outcomes[group, j]))
        p = (w+1)/(n+2) if n >= MIN_SUPPORT else .5
        probability.append(p); supports.append(n); wins.append(w); fallback.append(backoff)
    # Positive-profit events shrink with leverage because financing costs increase.
    return np.minimum.accumulate(probability), supports, wins, fallback


def oracle_history(prices, times, cost):
    years = np.diff(times.asi8) / (365.25*86400*1e9)
    ratio = prices[1:]/prices[:-1]
    long = pnl(ratio, 1., 1., cost, years)[0]
    short = pnl(ratio, -1., 1., cost, years)[0]
    return np.r_[0, np.cumsum(np.maximum(long, short) > 0)], years


def oracle_count(prefix, origin):
    # Origin t-2 exits at t-1, strictly before decision t: one-bar embargo.
    return int(prefix[origin-1]-prefix[origin-WINDOW-1])


def prepare():
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    (OUT/"jobs").mkdir(exist_ok=True); (OUT/"forecasts").mkdir(exist_ok=True)
    config = {"target": TARGET, "target_scope": "at least 465 executed trades combined across nine independent cases",
        "gates": [.55, .60, .70], "context": CONTEXT, "calibration_window": WINDOW,
        "calibration": "Causal rolling Beta(1,1) histogram posterior mean of net-profit frequency",
        "bins": EDGES.tolist(), "minimum_support": MIN_SUPPORT,
        "fallback": "All prior 300 non-flat predictions if bin has fewer than 20; probability 0.5 if total support below 20",
        "leverage_probability_constraint": "Cumulative minimum over 1x, 2x, 3x probabilities",
        "embargo": "Most recent calibration outcome exits strictly before the current entry bar",
        "refit": "Counts update with each newly completed past observation; no future labels",
        "stopping_rule": "Process 128 next chronological signals per active case per round; stop at the end of the first full round with >=465 executions, or exhausted data",
        "accounts": "Independent 100000 quote-currency accounts; combined count is not a pooled chronological portfolio",
        "no_tuning_by_profit": True, "no_threshold_relaxation": True,
        "model_revision": json.loads((ROOT/"data/manifest.json").read_text())["model_revision"],
        "costs": "Unchanged: per-fill 10/5/2 bps, annual financing 8% above 1x, short borrow 3%",
        "original_experiment_preserved": True}
    write_json(OUT/"configuration.json", config)
    (OUT/"PROTOCOL.md").write_text("""# Corrected confidence overlay: executed-trade target

Keep the 55% / 60% / 70% gates and model directions. The user's latest instruction
authorizes fixing the failed confidence overlay; the original frozen experiment
remains preserved as a separate result. Replace its input-range-clamped pooled
sigmoid with the predeclared causal histogram estimator in configuration.json.
Use the preceding 300 forecast origins with a one-bar embargo. Fixed ten bins,
Beta(1,1) smoothing, minimum support 20; sparse bins back off to the whole window.
Use only non-flat model directions as calibration examples. Calculate each
leverage's positive-after-cost label separately; enforce nested profit probabilities.
No test P&L or desired count chooses bins, windows, gates, prior, or direction.

Run nine independent chronological streams in fixed case order, 128 candidate
bars per active case per round. Stop after the first full round whose combined
executions reach at least 465. This is a sample count across independent accounts,
not a pooled time-synchronized portfolio. Report each case's actual date span.
If available history is exhausted first, report the shortfall without fabricated,
duplicated, bootstrapped or forced trades.

For computational savings a decision can be rejected before inference if fewer
than 12 of its prior 300 bars could have beaten costs even with perfect direction.
Any admissible calibration estimate has at least 20 observations; with <=11
possible wins its Beta(1,1) mean is at most 12/22 < 55%. Only past oracle bounds
are used, never the current or future outcome. Unknown model probabilities and
directions on such skipped rows remain missing, rather than invented.
""")
    fresh = json.loads((ROOT/"results/more-yahoo-2026-10-21/manifest.json").read_text())
    old = json.loads((ROOT/"results/executed-465/source_manifest.json").read_text())
    prior = {(r["ticker"],r["interval"]): r for r in old["cases"]}
    cases = []
    from confidence_200 import bars_for
    for asset in ASSETS:
        for interval in INTERVALS:
            source = next(r for r in fresh["cases"] if r["ticker"]==asset["ticker"] and r["interval"]==interval)
            path = ROOT/source["source"]; assert sha(path)==source["sha256"]
            b = load_frame(path); snapshots = [load_frame(ROOT/prior[asset["ticker"],interval]["source"])]
            original, _ = bars_for(asset["ticker"], interval); original.index = pd.to_datetime(original.index, utc=True)
            snapshots.append(original)
            preserved = 0
            # Preserve old quote values when differences are only insignificant rounding.
            for previous in snapshots:
                common = b.index.intersection(previous.index)
                relative = np.max(np.abs(b.loc[common,"Close"].to_numpy()/previous.loc[common,"Close"].to_numpy()-1)) if len(common) else 0
                if relative <= 1e-5:
                    b = pd.concat([b, previous[["Open","High","Low","Close"]]])
                    b = b.loc[~b.index.duplicated(keep="last")].sort_index(); preserved += len(previous)
            target = DATA/f"{asset['ticker']}_{interval}.csv"; b.to_csv(target)
            cases.append({**asset, "interval": interval, "source": str(target.relative_to(ROOT)),
                "sha256": sha(target), "bars": len(b), "first": str(b.index[0]), "last": str(b.index[-1]),
                "rounding_compatible_saved_rows_preserved": preserved})
    write_json(OUT/"source_manifest.json", {"cases": cases, "fresh_source_manifest": str((ROOT/"results/more-yahoo-2026-10-21/manifest.json").relative_to(ROOT))})
    print("PREPARED", len(cases), "cases; protocol saved", flush=True)


def worker(backend, threads):
    engine = CachedInference(backend, threads); frames = {}
    print("WORKER", backend, "ready", flush=True)
    try:
        while not (OUT/"STOP").exists():
            worked = False
            for jobfile in sorted((OUT/"jobs").glob("*.json")):
                if jobfile.name.endswith(".tmp.json"): continue
                output = OUT/"forecasts"/(jobfile.stem+".npz")
                if output.exists(): continue
                with jobfile.with_suffix(".lock").open("w") as lock:
                    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError: continue
                    if output.exists(): continue
                    job = json.loads(jobfile.read_text()); key=job["key"]
                    if key not in frames:
                        path=ROOT/job["source"]; assert sha(path)==job["source_sha256"]
                        frames[key]=load_frame(path).Close.to_numpy()
                    prices=frames[key]; ids=np.asarray(job["origins"],dtype=int)
                    x=np.stack([prices[t-CONTEXT+1:t+1] for t in ids]).astype("float32")
                    engine.load(); q=engine.model.forecast(x,prediction_length=1).quantiles[:,0,0].cpu().numpy()
                    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-6).all()
                    tmp=output.with_suffix(".tmp.npz")
                    np.savez_compressed(tmp,origins=ids,quantiles=q,source_sha256=np.array(job["source_sha256"]))
                    tmp.replace(output); worked=True
                    print("DONE",backend,jobfile.stem,len(ids),flush=True)
            if not worked: time.sleep(1)
    finally: engine.close()


class Coordinator:
    def __init__(self, items):
        self.items=items; self.frames={}; self.quantiles={}; self.info={}; self.prefix={}; self.counter=0
        for item in items:
            key=f"{item['ticker']}_{item['interval']}"; b=load_frame(ROOT/item["source"])
            assert sha(ROOT/item["source"])==item["sha256"]
            self.frames[key]=b; self.quantiles[key]={}; self.info[key]={}
            self.prefix[key],_=oracle_history(b.Close.to_numpy(),b.index,item["cost_bps"]/1e4)
        self.reuse_saved()

    def reuse_saved(self):
        from confidence_200 import bars_for
        old_manifest=json.loads((ROOT/"results/executed-465/source_manifest.json").read_text())
        for item in self.items:
            key=f"{item['ticker']}_{item['interval']}"; b=self.frames[key]
            sources=[]
            original,_=bars_for(item["ticker"],item["interval"]);original.index=pd.to_datetime(original.index,utc=True)
            sources.append((ROOT/"results/confidence-200"/f"forecasts_{key}.npz",original,"timestamps"))
            previous=next(r for r in old_manifest["cases"] if r["ticker"]==item["ticker"] and r["interval"]==item["interval"])
            sources.append((ROOT/"results/executed-465"/f"forecasts_{key}.npz",load_frame(ROOT/previous["source"]),"origins"))
            for path,old,field in sources:
                if not path.exists():continue
                z=np.load(path)
                for origin,q in zip(z[field],z["quantiles"]):
                    if field=="origins":oldpos=int(origin);stamp=old.index[oldpos]
                    else:
                        stamp=pd.Timestamp(str(origin));stamp=stamp.tz_localize("UTC") if stamp.tz is None else stamp
                        oldpos=old.index.get_indexer([stamp])[0]
                    pos=b.index.get_indexer([stamp])[0]
                    if min(pos,oldpos)>=CONTEXT-1 and np.array_equal(b.Close.iloc[pos-CONTEXT+1:pos+1].to_numpy(dtype="float32"),old.Close.iloc[oldpos-CONTEXT+1:oldpos+1].to_numpy(dtype="float32")):
                        self.quantiles[key][int(pos)]=q
            published=ROOT/"results/direction-confidence-465/forecast-cache"/f"{key}.npz"
            cached_paths=([published] if published.exists() else [])+sorted((OUT/"forecasts").glob(f"*_{key}_*.npz"))
            for path in cached_paths:
                if path.name.endswith(".tmp.npz"):continue
                z=np.load(path);assert str(z["source_sha256"])==item["sha256"]
                self.quantiles[key].update({int(t):q for t,q in zip(z["origins"],z["quantiles"])})
            print("REUSED",key,len(self.quantiles[key]),flush=True)

    def ensure(self, item, origins):
        key=f"{item['ticker']}_{item['interval']}";lookup=self.quantiles[key]
        missing=sorted(set(map(int,origins))-set(lookup));outputs=[]
        for offset in range(0,len(missing),48):
            ids=missing[offset:offset+48]
            identity=hashlib.sha256(np.array(ids,dtype="int64").tobytes()+item["sha256"].encode()).hexdigest()[:12]
            stem=f"J_{key}_{ids[0]:06d}_{identity}"; output=OUT/"forecasts"/(stem+".npz")
            write_json(OUT/"jobs"/(stem+".json"),{"key":key,"source":item["source"],"source_sha256":item["sha256"],"origins":ids})
            outputs.append(output)
        while outputs:
            for path in outputs[:]:
                if path.exists():
                    z=np.load(path);lookup.update({int(t):q for t,q in zip(z["origins"],z["quantiles"])})
                    outputs.remove(path)
            if outputs:time.sleep(1)
        pending=sorted(set(map(int,origins))-set(self.info[key]))
        if pending:
            ids=np.array(pending);q=np.stack([lookup[t] for t in pending])
            values=information(self.frames[key],ids,q,QUANTILE_LEVELS,item["cost_bps"]/1e4)
            for i,t in enumerate(pending):self.info[key][t]={name:row[i] for name,row in values.items()}

    def score(self,item,origins,round_number):
        key=f"{item['ticker']}_{item['interval']}";b=self.frames[key];records=[]
        needs=[]
        for t in origins:
            if oracle_count(self.prefix[key],t)>=12:needs.extend(range(t-WINDOW-1,t+1))
        self.ensure(item,needs)
        for t in origins:
            base={"ticker":item["ticker"],"interval":item["interval"],"round":round_number,"origin":int(t),
                "entry_bar":str(b.index[t]),"exit_bar":str(b.index[t+1]),"entry_close":float(b.Close.iloc[t]),
                "exit_close":float(b.Close.iloc[t+1]),"calibration_first_origin":str(b.index[t-WINDOW-1]),
                "calibration_latest_exit":str(b.index[t-1]),"oracle_possible_wins":oracle_count(self.prefix[key],t)}
            if base["oracle_possible_wins"]<12:
                base.update(executed=False,leverage=0.,net_return=0.,fees=0.,funding=0.,
                    skip_reason="past_window_cannot_support_55_percent",confidence_upper_bound=(base["oracle_possible_wins"]+1)/22)
            else:
                values=self.info[key][t];history=[self.info[key][i] for i in range(t-WINDOW-1,t-1)]
                raw=np.stack([r["raw"] for r in history]);outcomes=np.stack([r["returns"]>0 for r in history])
                valid=np.array([r["direction"]!=0 for r in history])
                p,n,w,fallback=calibrated_probability(raw,outcomes,valid,values["raw"])
                leverage=float(select_leverage(p[None,:])[0]) if values["direction"]!=0 else 0.
                j=int(leverage)-1;active=leverage>0
                base.update(forecast_median=float(self.quantiles[key][t][10]),direction=float(values["direction"]),
                    raw_direction_confidence=float(values["direction_probability"]),hypothetical_1x_profit=bool(values["returns"][0]>0),
                    **{f"profit_confidence_{i+1}x":float(p[i]) for i in range(3)},
                    **{f"support_{i+1}x":n[i] for i in range(3)},**{f"wins_{i+1}x":w[i] for i in range(3)},
                    executed=active,leverage=leverage,selected_confidence=float(p[j]) if active else None,
                    net_return=float(values["returns"][j]) if active else 0.,fees=float(values["fees"][j]) if active else 0.,
                    funding=float(values["funding"][j]) if active else 0.,holding_years=float(values["years"]),
                    skip_reason="executed" if active else "flat_forecast" if values["direction"]==0 else "profit_confidence_below_gates")
            records.append(base)
        return pd.DataFrame(records)


def coordinate(output_dir=OUT, coordinator_class=Coordinator):
    (OUT/"STOP").unlink(missing_ok=True)
    items=json.loads((OUT/"source_manifest.json").read_text())["cases"]
    coordinator=coordinator_class(items);pieces=[];executed=0;round_number=0
    try:
        while executed<TARGET:
            round_number+=1;active_cases=0
            for item in items:
                key=f"{item['ticker']}_{item['interval']}";start=FIRST_TEST+(round_number-1)*ROUND_SIZE
                end=min(start+ROUND_SIZE,len(coordinator.frames[key])-1)
                if end<=start:continue
                active_cases+=1;block=coordinator.score(item,range(start,end),round_number);pieces.append(block)
                executed+=int(block.executed.sum())
                print("CASE",key,"round",round_number,"executions",int(block.executed.sum()),"TOTAL",executed,flush=True)
                pd.concat(pieces,ignore_index=True).to_csv(output_dir/"evaluated_signals.csv",index=False)
                write_json(output_dir/"progress.json",{"target":TARGET,"executed":executed,"round":round_number,
                    "latest_case":key,"signals_processed":sum(len(p) for p in pieces),"round_completed":False})
            print("ROUND",round_number,"executions",executed,flush=True)
            if not active_cases:break
        ledger=pd.concat(pieces,ignore_index=True);ledger["net_equity"]=INITIAL
        for _,indexes in ledger.groupby(["ticker","interval"],sort=False).groups.items():
            assert (ledger.loc[indexes,"net_return"]>-1).all()
            ledger.loc[indexes,"net_equity"]=INITIAL*np.cumprod(1+ledger.loc[indexes,"net_return"].to_numpy())
        trades=ledger[ledger.executed].sort_values(["exit_bar","ticker","interval"],kind="stable").copy()
        trades.insert(0,"combined_trade_number",np.arange(1,len(trades)+1))
        ledger.to_csv(output_dir/"evaluated_signals.csv",index=False);trades.to_csv(output_dir/"executed_trades.csv",index=False)
        write_json(output_dir/"completion.json",{"target":TARGET,"executed":len(trades),"target_reached":len(trades)>=TARGET,
            "rounds":round_number,"signals_processed":len(ledger),"history_exhausted":len(trades)<TARGET,
            "stopping_rule":"Full round of independent case streams; not a shared calendar portfolio"})
        print("COMPLETE",len(trades),"executed trades",flush=True)
    finally:(OUT/"STOP").write_text("Coordinator stopped\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","worker","run"])
    p.add_argument("--backend",choices=["cpu","vulkan"],default="vulkan");p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    if args.mode=="prepare":prepare()
    elif args.mode=="worker":worker(args.backend,args.threads)
    else:coordinate()


if __name__=="__main__":main()
