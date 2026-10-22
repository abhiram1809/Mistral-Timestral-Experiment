"""Chronological profit-probability calibration, abstention and 1x/2x/3x simulation."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from timestral import TimestralForecaster

from simulate_100_trades import ASSETS, INTERVALS, CONTEXT, INITIAL, ROOT, load_bars

OUT=ROOT/"results/confidence-200"
LEVERAGES=np.array([1.,2.,3.])
THRESHOLDS=np.array([.55,.60,.70])
NCAL=300
FINANCING=.08
BORROW=.03


def pnl(ratio, direction, leverage, cost, years):
    ratio=np.asarray(ratio);direction=np.asarray(direction);leverage=np.asarray(leverage)
    exposure=leverage/(1+cost*leverage)
    fees=exposure*cost*(1+ratio)
    funding=(FINANCING*np.maximum(exposure-1,0)+BORROW*exposure*(direction<0))*years
    net=exposure*direction*(ratio-1)-fees-funding
    return net,fees,funding,exposure


def profit_threshold(entry,direction,leverage,cost,years):
    exposure=leverage/(1+cost*leverage)
    carry=(FINANCING*np.maximum(exposure-1,0)/exposure+BORROW*(direction<0))*years
    ratio=np.where(direction>0,(1+cost+carry)/(1-cost),(1-cost-carry)/(1+cost))
    return np.asarray(entry)*ratio


def cdf(quantiles,levels,prices):
    return np.array([np.interp(p,q,levels) for q,p in zip(quantiles,prices)])


class ProfitCalibrator:
    """Monotone sigmoid, L2 slope penalty, two virtual opposing intercept labels."""
    def fit(self,raw,outcomes):
        raw=np.clip(np.asarray(raw),.01,.99)
        self.raw_min=float(raw.min());self.raw_max=float(raw.max())
        self.center=float(np.mean(logit(raw)))
        x=(logit(raw)-self.center).ravel()
        y=np.asarray(outcomes,dtype=float).ravel()
        weight=np.full(len(y),1/3.)
        self.base=float((np.sum(weight*y)+1)/(sum(weight)+2))
        def objective(ab):
            a,b=ab;z=a*x+b;p=expit(z)
            loss=np.sum(weight*(np.logaddexp(0,z)-y*z))+.5*a*a+np.logaddexp(0,b)+np.logaddexp(0,-b)
            grad=np.array([np.sum(weight*(p-y)*x)+a,np.sum(weight*(p-y))+2*expit(b)-1])
            return loss,grad
        fit=minimize(objective,[0.,logit(self.base)],jac=True,method="L-BFGS-B",
                     bounds=[(0,None),(None,None)],options={"maxiter":300,"ftol":1e-12})
        if not fit.success:
            raise RuntimeError(f"Calibration failed: {fit.message}")
        self.a,self.b=map(float,fit.x)
        return self

    def predict(self,raw):
        bounded=np.clip(raw,self.raw_min,self.raw_max)
        return expit(self.a*(logit(bounded)-self.center)+self.b)


def select_leverage(probabilities):
    p=np.asarray(probabilities)
    selected=np.zeros(len(p))
    for i,level in enumerate(LEVERAGES):
        selected=np.where(p[:,i]>=THRESHOLDS[i],level,selected)
    return selected


def bars_for(ticker,interval):
    if interval!="1h":return load_bars(ticker,interval)
    path=ROOT/"data/confidence-200"/f"{ticker}_1h.csv"
    bars=pd.read_csv(path,index_col=0);bars.index=pd.to_datetime(bars.index,utc=True)
    bars=bars.loc[bars.index<pd.Timestamp("2026-10-20",tz="UTC")]
    valid=np.isfinite(bars[["Open","Close"]]).all(axis=1)&(bars.Open>0)&(bars.Close>0)
    clean=bars.loc[valid].sort_index()
    assert clean.index.is_unique
    return clean,{"source":str(path.relative_to(ROOT)),"source_rows":len(bars),
        "usable_rows":len(clean),"removed_rows":int((~valid).sum()),
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def get_forecasts(model,bars,ticker,interval,origins,batch=48):
    path=OUT/f"forecasts_{ticker}_{interval}.npz"
    lookup={}
    if path.exists():
        old=np.load(path)
        lookup={str(t):q for t,q in zip(old["timestamps"],old["quantiles"])}
    previous=ROOT/"results/100-trades"/f"forecasts_{ticker}_{interval}.npz"
    if previous.exists():
        old_bars,_=load_bars(ticker,interval)
        old=np.load(previous)
        for origin,q in zip(old["origins"],old["quantiles"]):
            timestamp=str(old_bars.index[int(origin)])
            # Cache reuse requires the same complete context, including hourly re-downloads.
            pos=bars.index.get_indexer([old_bars.index[int(origin)]])[0]
            if pos>=CONTEXT-1 and int(origin)>=CONTEXT-1:
                x=bars.Close.to_numpy()[pos-CONTEXT+1:pos+1]
                old_x=old_bars.Close.to_numpy()[int(origin)-CONTEXT+1:int(origin)+1]
                if np.array_equal(x.astype("float32"),old_x.astype("float32")):
                    lookup.setdefault(timestamp,q)
    timestamps=np.array([str(bars.index[t]) for t in origins])
    pending=[i for i,t in enumerate(timestamps) if t not in lookup]
    closes=bars.Close.to_numpy()
    for start in range(0,len(pending),batch):
        ids=pending[start:start+batch]
        x=np.stack([closes[t-CONTEXT+1:t+1] for t in origins[ids]]).astype("float32")
        forecast=model.forecast(x,prediction_length=1)
        q=forecast.quantiles[:,0,0].cpu().numpy()
        assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-6).all()
        lookup.update({timestamps[i]:row for i,row in zip(ids,q)})
        keys=sorted(lookup)
        tmp=path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,timestamps=np.array(keys),quantiles=np.stack([lookup[k] for k in keys]),
                            levels=np.array(model.quantile_levels))
        tmp.replace(path)
        print(ticker,interval,f"{min(start+batch,len(pending))}/{len(pending)} new forecasts",flush=True)
    return np.stack([lookup[t] for t in timestamps])


def information(bars,origins,quantiles,levels,cost):
    prices=bars.Close.to_numpy();entry=prices[origins];exit=prices[origins+1]
    direction=np.sign(quantiles[:,10].astype(float)-entry)
    times=pd.to_datetime(bars.index,utc=True)
    years=np.array([(times[t+1]-times[t]).total_seconds()/(365.25*86400) for t in origins])
    raw=np.zeros((len(origins),3));returns=np.zeros_like(raw);fees=np.zeros_like(raw);funding=np.zeros_like(raw)
    for i,leverage in enumerate(LEVERAGES):
        threshold=profit_threshold(entry,direction,leverage,cost,years)
        raw[:,i]=np.where(direction>0,1-cdf(quantiles,levels,threshold),cdf(quantiles,levels,threshold))
        returns[:,i],fees[:,i],funding[:,i],_=pnl(exit/entry,direction,leverage,cost,years)
    raw[direction==0]=.5
    direction_probability=np.where(direction>0,1-cdf(quantiles,levels,entry),cdf(quantiles,levels,entry))
    return {"entry":entry,"exit":exit,"direction":direction,"years":years,
            "raw":raw,"returns":returns,"fees":fees,"funding":funding,"direction_probability":direction_probability}


def scores(probability,labels):
    p=np.clip(probability,1e-9,1-1e-9);y=np.asarray(labels,dtype=float)
    return {"brier":float(np.mean((p-y)**2)),
            "log_loss":float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}


def equity_stats(returns):
    returns=np.asarray(returns,dtype=float)
    equity=INITIAL*np.cumprod(1+np.maximum(returns,-1))
    peak=np.maximum.accumulate(np.r_[INITIAL,equity])[1:]
    return {"ending":float(equity[-1]),"return":float(equity[-1]/INITIAL-1),
            "max_drawdown":float(np.min(equity/peak-1))},equity


def run_case(model,asset,interval,mode,target=200,pool=1000):
    ticker=asset["ticker"];key=f"{ticker}_{interval}"
    with (OUT/f"{key}.lock").open("w") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        result_path=OUT/f"result_{key}.json"
        if result_path.exists() and json.loads(result_path.read_text()).get("counting_mode")==mode:return
        bars,source=bars_for(ticker,interval)
        available=len(bars)-CONTEXT-NCAL-1
        count=target if mode=="opportunities" else min(pool,available)
        assert count>=target
        start=len(bars)-1-count
        calibration_origins=np.arange(start-NCAL-1,start-1)
        test_origins=np.arange(start,len(bars)-1)
        assert calibration_origins.min()>=CONTEXT-1
        assert calibration_origins[-1]+1<test_origins[0]
        cal_q=get_forecasts(model,bars,ticker,interval,calibration_origins)
        levels=np.asarray(model.quantile_levels);cost=asset["cost_bps"]/1e4
        cal=information(bars,calibration_origins,cal_q,levels,cost)
        valid=cal["direction"]!=0
        calibrator=ProfitCalibrator().fit(cal["raw"][valid],cal["returns"][valid]>0)
        print("CALIBRATED",key,"base",calibrator.base,"slope",calibrator.a,
              "maximum attainable confidence",float(calibrator.predict(.99)),flush=True)
        q_blocks=[];chosen_count=0;used=0
        for offset in range(0,len(test_origins),100):
            ids=test_origins[offset:offset+100]
            q=get_forecasts(model,bars,ticker,interval,ids)
            data=information(bars,ids,q,levels,cost)
            probability=calibrator.predict(data["raw"])
            chosen=select_leverage(probability);chosen[data["direction"]==0]=0
            if mode=="executions" and chosen_count+np.count_nonzero(chosen)>=target:
                end=np.flatnonzero(chosen>0)[target-chosen_count-1]+1
                q=q[:end];q_blocks.append(q);used+=len(q);break
            q_blocks.append(q);used+=len(q);chosen_count+=np.count_nonzero(chosen)
        q=np.concatenate(q_blocks);origins=test_origins[:used]
        data=information(bars,origins,q,levels,cost);probability=calibrator.predict(data["raw"])
        chosen=select_leverage(probability);chosen[data["direction"]==0]=0
        raw_chosen=select_leverage(data["raw"]);raw_chosen[data["direction"]==0]=0
        one_chosen=np.where((probability[:,0]>=.55)&(data["direction"]!=0),1.,0.)
        def selected(values,leverage):
            result=np.zeros(len(leverage));active=leverage>0
            result[active]=values[active,leverage[active].astype(int)-1]
            return result
        main_return=selected(data["returns"],chosen)
        base_return=np.where(data["direction"]!=0,data["returns"][:,0],0.)
        raw_return=selected(data["returns"],raw_chosen)
        one_return=selected(data["returns"],one_chosen)
        gross_return=chosen*data["direction"]*(data["exit"]/data["entry"]-1)
        main_stat,equity=equity_stats(main_return);base_stat,base_equity=equity_stats(base_return)
        raw_stat,raw_equity=equity_stats(raw_return);one_stat,one_equity=equity_stats(one_return)
        _,gross_equity=equity_stats(gross_return)
        valid_test=data["direction"]!=0;y=data["returns"][:,0]>0
        base_probability=float((sum(cal["returns"][valid,0]>0)+1)/(sum(valid)+2))
        ledger=pd.DataFrame({"ticker":ticker,"interval":interval,"opportunity":np.arange(1,used+1),
            "entry_bar":[str(bars.index[t]) for t in origins],"exit_bar":[str(bars.index[t+1]) for t in origins],
            "entry_close":data["entry"],"exit_close":data["exit"],"forecast_median":q[:,10],
            "direction":data["direction"],"raw_direction_confidence":data["direction_probability"],
            "raw_profit_confidence_1x":data["raw"][:,0],"calibrated_profit_confidence_1x":probability[:,0],
            "calibrated_profit_confidence_2x":probability[:,1],"calibrated_profit_confidence_3x":probability[:,2],
            "leverage":chosen,"executed":chosen>0,"trade_number":np.where(chosen>0,np.cumsum(chosen>0),0),
            "selected_confidence":selected(probability,chosen),"net_return":main_return,"gross_return":gross_return,
            "fees":selected(data["fees"],chosen),"funding":selected(data["funding"],chosen),
            "net_equity":equity,"gross_equity":gross_equity,"baseline_net_return":base_return,"baseline_equity":base_equity,
            "raw_policy_equity":raw_equity,"calibrated_1x_equity":one_equity,
            "hypothetical_1x_profit":y,"raw_policy_leverage":raw_chosen,"holding_years":data["years"]})
        ledger.to_csv(OUT/f"opportunities_{key}.csv",index=False)
        ledger.loc[ledger.executed].to_csv(OUT/f"trades_{key}.csv",index=False)
        calibration=pd.DataFrame({"entry_bar":[str(bars.index[t]) for t in calibration_origins],
            "exit_bar":[str(bars.index[t+1]) for t in calibration_origins],"direction":cal["direction"],
            "raw_profit_1x":cal["raw"][:,0],"profit_1x":cal["returns"][:,0]>0})
        calibration.to_csv(OUT/f"calibration_{key}.csv",index=False)
        result={**asset,"interval":interval,"counting_mode":mode,"target":target,"opportunities":used,
            "executed":int(sum(chosen>0)),"skipped":int(sum(chosen==0)),
            "target_reached":bool(used>=target if mode=="opportunities" else sum(chosen>0)>=target),
            "first_entry":str(bars.index[origins[0]]),"last_exit":str(bars.index[origins[-1]+1]),
            "calibration_first":str(bars.index[calibration_origins[0]]),
            "calibration_last_exit":str(bars.index[calibration_origins[-1]+1]),
            "calibration_observations":int(sum(valid)),"slope":calibrator.a,"intercept":calibrator.b,
            "calibration_logit_center":calibrator.center,"calibration_raw_min":calibrator.raw_min,
            "calibration_raw_max":calibrator.raw_max,
            "maximum_attainable_confidence":float(calibrator.predict(.99)),
            "constant_profit_probability":base_probability,"initial":INITIAL,
            **main_stat,"gross_ending":float(gross_equity[-1]),"baseline":base_stat,"raw_policy":raw_stat,"calibrated_1x":one_stat,
            "leverage_1x_trades":int(sum(chosen==1)),"leverage_2x_trades":int(sum(chosen==2)),
            "leverage_3x_trades":int(sum(chosen==3)),"mean_leverage":float(np.mean(chosen[chosen>0])) if np.any(chosen) else 0.,
            "net_win_rate":float(np.mean(main_return[chosen>0]>0)) if np.any(chosen) else None,
            "raw_scores":scores(data["raw"][valid_test,0],y[valid_test]),
            "calibrated_scores":scores(probability[valid_test,0],y[valid_test]),
            "constant_scores":scores(np.full(sum(valid_test),base_probability),y[valid_test]),"source":source}
        result_path.write_text(json.dumps(result,indent=2))
        print("COMPLETE",key,"executions",result["executed"],"skips",result["skipped"],"return",result["return"],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--backend",choices=["cpu","vulkan"],default="cpu")
    parser.add_argument("--threads",type=int,default=3);parser.add_argument("--mode",choices=["executions","opportunities"],default="opportunities")
    parser.add_argument("--pool",type=int,default=1000);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    revision=json.loads((ROOT/"data/manifest.json").read_text())["model_revision"]
    model=TimestralForecaster.from_pretrained("mistralai/Timestral-209M",revision=revision,local_files_only=True)
    engine=None
    if args.backend=="vulkan":
        from vulkan_backend import offload
        engine=offload(model.model)
    for interval in INTERVALS:
        for asset in ASSETS:run_case(model,asset,interval,args.mode,pool=args.pool)
    if engine:engine.close()


if __name__=="__main__":main()
