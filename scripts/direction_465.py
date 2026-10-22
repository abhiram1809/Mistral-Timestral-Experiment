"""Calibrated forecast-direction confidence; measure trading profit after all costs."""
import json
import shutil

import numpy as np
import pandas as pd

import corrected_465 as base
from confidence_200 import select_leverage

OUT=base.ROOT/"results/direction-confidence-465"


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((base.OUT/"configuration.json").read_text())
    config.update(confidence_event="Model median direction is correct at the next close; exact zero movement is not a success",
        profit_probability="Separate diagnostic: estimate positive net P&L at each exposure; never relabel it directional confidence",
        calibration="Causal rolling Beta(1,1) histogram of directional correctness",
        raw_score="Quantile-CDF probability that price moves in the median-predicted direction",
        leverage_probability_constraint="All exposure levels use the same directional probability; exposure does not change price direction",
        methodology_change="Explicit event-definition redesign after observing trade scarcity; neither old frozen calibration nor a clean untouched study",
        calibration_updates=True,
        no_return_hyperparameter_search=True)
    base.write_json(OUT/"configuration.json",config)
    shutil.copy2(base.OUT/"source_manifest.json",OUT/"source_manifest.json")
    (OUT/"PROTOCOL.md").write_text("""# Directional confidence and at least 465 actual executions

The user authorized fixing the confidence overlay and completing the test.
Confidence now means probability that the model's predicted direction is right
at the next close. This is an explicit change from after-cost profit probability.
Costs are still fully charged in P&L. Keep the 55%/60%/70% gates for 1x/2x/3x.

Raw score: the model quantile CDF at the current close, in its median direction.
Calibrate on the last 300 forecast origins whose outcomes exit strictly before
the current entry. Use ten fixed width-0.1 bins, Beta(1,1) posterior mean
(wins+1)/(n+2), minimum support 20. A sparse bin backs off to all valid non-flat
predictions in that past window; total support below 20 yields probability 0.5.
Zero realized moves count as failures. No scores are clamped to an observed
training range. Update counts only as completed past observations become usable.
The same price-direction probability controls all exposure levels; leverage
cannot change the probability of the price direction. Choose highest level
whose fixed gate is passed, otherwise cash. Unchanged forecast medians skip.

Also record the probability of positive net P&L at each level, using the same
causal histogram method, as a separate diagnostic. It does not control the
direction-confidence policy or get relabeled as forecast confidence.

Run nine independent case streams, each chronologically from the earliest
eligible test origin after 250 context observations and 300 calibration origins,
with one-bar embargo. Advance 128 bars per active case per round. Stop after
the first complete round with at least 465 unique executions combined, or when
all available data are exhausted. This is a sample count across separate quote-
currency accounts, not a globally synchronized shared portfolio. Skips never
count as executions. No forced passes, duplicated trades or resampling.

Model, fees, financing, borrow and same-close paper fills remain unchanged.
The method definition changed after earlier test results were inspected. This
is retrospective strategy research with possible model-training overlap, not
a clean prospective validation. All previous experiments remain preserved.
""")


class DirectionCoordinator(base.Coordinator):
    def score(self,item,origins,round_number):
        key=f"{item['ticker']}_{item['interval']}";b=self.frames[key]
        self.ensure(item,range(min(origins)-base.WINDOW-1,max(origins)+1))
        records=[]
        for t in origins:
            value=self.info[key][t];history=[self.info[key][i] for i in range(t-base.WINDOW-1,t-1)]
            raw=np.tile(np.array([r["direction_probability"] for r in history])[:,None],(1,3))
            correct=np.array([r["direction"]*(r["exit"]/r["entry"]-1)>0 for r in history])
            outcomes=np.tile(correct[:,None],(1,3));valid=np.array([r["direction"]!=0 for r in history])
            p,n,w,fallback=base.calibrated_probability(raw,outcomes,valid,[value["direction_probability"]]*3)
            net_raw=np.stack([r["raw"] for r in history]);net_labels=np.stack([r["returns"]>0 for r in history])
            net_p,*_=base.calibrated_probability(net_raw,net_labels,valid,value["raw"])
            leverage=float(select_leverage(p[None,:])[0]) if value["direction"]!=0 else 0.
            j=int(leverage)-1;active=leverage>0
            records.append({"ticker":item["ticker"],"interval":item["interval"],"round":round_number,"origin":int(t),
                "entry_bar":str(b.index[t]),"exit_bar":str(b.index[t+1]),"entry_close":float(value["entry"]),"exit_close":float(value["exit"]),
                "calibration_first_origin":str(b.index[t-base.WINDOW-1]),"calibration_latest_exit":str(b.index[t-1]),
                "forecast_median":float(self.quantiles[key][t][10]),"direction":float(value["direction"]),
                "raw_direction_confidence":float(value["direction_probability"]),"direction_confidence":float(p[0]),
                "direction_correct":bool(value["direction"]*(value["exit"]/value["entry"]-1)>0),
                "hypothetical_1x_profit":bool(value["returns"][0]>0),
                **{f"profit_confidence_{i+1}x":float(net_p[i]) for i in range(3)},
                "calibration_support":n[0],"calibration_wins":w[0],"calibration_backoff":bool(fallback[0]),
                "executed":active,"leverage":leverage,"selected_confidence":float(p[0]) if active else None,
                "net_return":float(value["returns"][j]) if active else 0.,"fees":float(value["fees"][j]) if active else 0.,
                "funding":float(value["funding"][j]) if active else 0.,"holding_years":float(value["years"]),
                "skip_reason":"executed" if active else "flat_forecast" if value["direction"]==0 else "direction_confidence_below_gates"})
        return pd.DataFrame(records)


def main():
    prepare()
    base.coordinate(output_dir=OUT,coordinator_class=DirectionCoordinator)


if __name__=="__main__":main()
