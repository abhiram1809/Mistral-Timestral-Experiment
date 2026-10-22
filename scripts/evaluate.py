"""Chronological calibration, forecast scoring and friction-aware simulation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from infer import ROOT, load_prices


def probability_up(q, levels):
    return np.array([1 - np.interp(0., row, levels) for row in q])


class Calibrator:
    def fit(self, p, y):
        self.constant = float((np.sum(y) + 1) / (len(y) + 2))
        self.model = None
        if len(np.unique(y)) > 1:
            self.model = LogisticRegression(C=1., solver="lbfgs")
            self.model.fit(self.features(p), y)
        return self

    @staticmethod
    def features(p):
        p = np.clip(p, .01, .99)
        return np.log(p / (1-p)).reshape(-1, 1)

    def predict(self, p):
        if self.model is None:
            return np.full(len(p), self.constant)
        return self.model.predict_proba(self.features(p))[:, 1]

    def metadata(self):
        return {"base_rate": self.constant,
                "coefficient": float(self.model.coef_[0, 0]) if self.model else 0.,
                "intercept": float(self.model.intercept_[0]) if self.model else 0.}


def interval_scale(q, y, levels, coverage=.8):
    mid = q[:, np.argmin(abs(levels-.5))]
    half = np.maximum((q[:, np.argmin(abs(levels-.9))] -
                       q[:, np.argmin(abs(levels-.1))]) / 2, 1e-8)
    score = np.abs(y - mid) / half
    rank = min(int(np.ceil((len(score) + 1) * coverage)), len(score))
    return float(np.sort(score)[rank-1])


def sizes(probability, volatility, cap, target=.10):
    confidence = np.maximum(probability, 1-probability)
    conviction = np.clip((confidence-.55)/.15, 0, 1)
    return np.sign(probability-.5) * np.minimum(
        cap, conviction * target / np.maximum(volatility, .01)
    )


def simulate(weights, returns, calendar_days, cost, financing=.08, borrow=.03):
    """Weights target notional / pre-cost equity; rebalance from drifted exposure.

    Financing is a simple exposure surcharge, not a model of an actual margin
    account. Trading costs are paid from equity. End-of-period gross exposure
    is carried into the next rebalance. Liquidate the final position.
    """
    weights = np.asarray(weights, dtype=float)
    returns = np.asarray(returns, dtype=float)
    net = np.zeros(len(weights))
    costs = np.zeros(len(weights))
    turnover = np.zeros(len(weights))
    drifted = 0.
    bankrupt = False
    for i, (w, r, days) in enumerate(zip(weights, returns, calendar_days)):
        if bankrupt:
            continue
        turnover[i] = abs(w - drifted)
        costs[i] = cost * turnover[i] + (
            financing * max(abs(w)-1, 0) + borrow * max(-w, 0)
        ) * days / 365.25
        net[i] = w * r - costs[i]
        if i == len(weights)-1:
            exit_turnover = abs(w * (1+r))
            costs[i] += cost * exit_turnover
            turnover[i] += exit_turnover
            net[i] -= cost * exit_turnover
        if net[i] <= -1:
            net[i] = -1.
            bankrupt = True
            drifted = 0.
        else:
            drifted = w * (1+r) / (1+net[i])
    return net, costs, turnover


def block_ci(values, seed=42, count=2000, block=20):
    """Circular moving-block CI for the mean, preserving local dependence."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, (count, int(np.ceil(n/block))))
    indices = (starts[..., None] + np.arange(block)) % n
    means = values[indices.reshape(count, -1)[:, :n]].mean(axis=1)
    return np.quantile(means, [.025, .975])


def trading_stats(net, weights, turnover, days, initial):
    equity = initial * np.cumprod(1+net)
    peak = np.maximum.accumulate(np.r_[initial, equity])[1:]
    annvol = float(np.std(net, ddof=1) * np.sqrt(252))
    mean_ci = block_ci(net)
    return {
        "ending_capital": float(equity[-1]),
        "net_return": float(equity[-1] / initial - 1),
        "cagr": float((equity[-1]/initial)**(365.25/sum(days))-1),
        "sharpe": float(np.mean(net)*252 / annvol) if annvol > 1e-12 else 0.,
        "max_drawdown": float(np.min(equity/peak-1)),
        "annual_volatility": annvol,
        "average_abs_exposure": float(np.mean(abs(weights))),
        "max_abs_exposure": float(np.max(abs(weights))),
        "turnover": float(sum(turnover)),
        "active_days": int(np.count_nonzero(weights)),
        "winning_active_fraction": float(np.mean(net[weights != 0] > 0)) if np.any(weights) else 0.,
        "daily_mean_ci_low": float(mean_ci[0]),
        "daily_mean_ci_high": float(mean_ci[1]),
    }, equity


def forecast_metrics(q, levels, y, empirical, train_scale):
    mid = q[:, np.argmin(abs(levels-.5))]
    residual = y[:, None] - q
    pinball = np.maximum(levels*residual, (levels-1)*residual)
    br = y[:, None] - empirical
    base_loss = np.maximum(levels*br, (levels-1)*br)
    # Integrate only available quantiles, .01-.99; do not invent tail forecasts.
    crps = 2*np.trapezoid(pinball, levels, axis=1)
    crps_base = 2*np.trapezoid(base_loss, levels, axis=1)
    mse = np.mean((y-mid)**2)
    mae = np.mean(abs(y-mid))
    naive_mae = np.mean(abs(y))
    ci = block_ci(abs(y)-abs(y-mid))
    return {
        "n": len(y), "mae": float(mae), "rmse": float(np.sqrt(mse)),
        "mae_skill_zero": float(1-mae/naive_mae),
        "mse_skill_zero": float(1-mse/np.mean(y*y)),
        "return_r2": float(1-mse/np.mean((y-y.mean())**2)),
        "mase": float(mae/train_scale),
        "direction_accuracy": float(np.mean((mid > 0) == (y > 0))),
        "nonzero_direction_accuracy": float(np.mean((mid[y != 0] > 0) == (y[y != 0] > 0))),
        "zero_outcomes": int(np.sum(y == 0)),
        "pinball": float(pinball.mean()), "truncated_crps": float(crps.mean()),
        "crps_skill_empirical": float(1-crps.mean()/crps_base.mean()),
        "coverage80": float(np.mean((y >= q[:, np.argmin(abs(levels-.1))]) &
                                    (y <= q[:, np.argmin(abs(levels-.9))]))),
        "width80": float(np.mean(q[:, np.argmin(abs(levels-.9))] -
                                q[:, np.argmin(abs(levels-.1))])),
        "mae_improvement_ci_low": float(ci[0]), "mae_improvement_ci_high": float(ci[1]),
    }


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    metrics, trades, curve_rows, calibration_rows, reliability_rows = [], [], [], [], []
    latest_rows, predictions, price_metrics, price_latest = [], [], [], []
    ncal = cfg["calibration_observations"]
    ntest = cfg["test_observations"]
    for asset in cfg["assets"]:
        ticker = asset["ticker"]
        prices = load_prices(ticker)
        r = np.diff(np.log(prices.to_numpy()))
        dates = prices.index[1:]
        for context in cfg["contexts"]:
            path = ROOT / "results" / f"returns_{ticker}_{context}.npz"
            if not path.exists():
                continue
            saved = np.load(path)
            q, origins, levels = saved["quantiles"], saved["origins"], saved["levels"]
            if not np.isfinite(q).all():
                continue
            cal_idx, test_idx = origins[:ncal], origins[ncal:ncal+ntest]
            assert cal_idx[-1] + 2 < test_idx[0]
            yc = r[cal_idx+2]
            yt = r[test_idx+2]
            qc, qt = q[:ncal, 1], q[ncal:ncal+ntest, 1]
            rawc, rawt = probability_up(qc, levels), probability_up(qt, levels)
            calibrator = Calibrator().fit(rawc, (yc > 0).astype(int))
            prob = calibrator.predict(rawt)
            scale = interval_scale(qc, yc, levels)
            median = qt[:, np.argmin(abs(levels-.5))]
            half = (qt[:, np.argmin(abs(levels-.9))]-qt[:, np.argmin(abs(levels-.1))])/2
            coverage = np.mean(abs(yt-median) <= scale*half)
            common = {"ticker": ticker, "market": asset["market"], "context": context}
            calibration_rows.append({
                **common, **calibrator.metadata(), "interval_scale80": scale,
                "calibration_first_origin": str(dates[cal_idx[0]].date()),
                "calibration_last_outcome": str(dates[cal_idx[-1]+2].date()),
                "test_first_origin": str(dates[test_idx[0]].date()),
                "test_first_exit": str(dates[test_idx[0]+2].date()),
                "test_last_exit": str(dates[test_idx[-1]+2].date()),
                "raw_brier": float(np.mean((rawt-(yt>0))**2)),
                "calibrated_brier": float(np.mean((prob-(yt>0))**2)),
                "base_brier": float(np.mean((calibrator.constant-(yt>0))**2)),
                "calibrated_coverage80": float(coverage),
                "probability_min": float(prob.min()), "probability_max": float(prob.max()),
            })
            for horizon in [1, 2, 5, 20]:
                valid = test_idx + horizon < len(r)
                origins_h = test_idx[valid]
                target = r[origins_h+horizon]
                empirical = np.stack([np.quantile(r[t-context+1:t+1], levels) for t in origins_h])
                train_scale = np.mean(abs(r[:test_idx[0]+1] - np.r_[0., r[:test_idx[0]]]))
                score = forecast_metrics(q[ncal:ncal+ntest, horizon-1][valid],
                                         levels, target, empirical, train_scale)
                metrics.append({**common, "horizon": horizon, **score})
            annual_vol = np.array([np.std(r[t-59:t+1], ddof=1)*np.sqrt(252) for t in test_idx])
            weights = {
                "cash": np.zeros(ntest), "passive_1x": np.ones(ntest),
                "momentum_1x": np.array([np.sign(sum(r[t-19:t+1])) for t in test_idx]),
                "raw_sign_1x": np.sign(rawt-.5),
                "calibrated_sign_1x": np.sign(prob-.5),
                "confidence_cap1": sizes(prob, annual_vol, 1, cfg["volatility_target"]),
                "confidence_cap3": sizes(prob, annual_vol, cfg["maximum_leverage"], cfg["volatility_target"]),
                # Diagnostic ablations: separate calibrated confidence from drift/volatility.
                "base_rate_sized": sizes(np.full(ntest, calibrator.constant), annual_vol,
                                         cfg["maximum_leverage"], cfg["volatility_target"]),
                "volatility_scaled_sign": np.sign(prob-.5) * np.minimum(
                    cfg["maximum_leverage"], cfg["volatility_target"]/np.maximum(annual_vol, .01)),
            }
            # Passive 1x is actual holding with no routine daily rebalancing.
            days = np.array([(dates[t+2]-dates[t+1]).days for t in test_idx])
            simple = np.expm1(yt)
            for strategy, w in weights.items():
                for cost_multiplier in [1, 2]:
                    unit_cost = asset["one_way_cost_bps"]/1e4*cost_multiplier
                    w_sim = w.copy()
                    if strategy == "passive_1x":
                        # Spend cash on shares AND entry fee; thereafter fully invested.
                        w_sim[0] = 1/(1+unit_cost)
                    net, cost, turn = simulate(
                        w_sim, simple, days, unit_cost,
                        cfg["annual_financing_rate"], cfg["annual_short_borrow_rate"],
                    )
                    stat, equity = trading_stats(net, w_sim, turn, days, cfg["initial_capital"])
                    trades.append({**common, "strategy": strategy, "cost_multiplier": cost_multiplier,
                                   "currency": asset["currency"], **stat})
                    if cost_multiplier == 1:
                        for j, t in enumerate(test_idx):
                            curve_rows.append({
                                **common, "strategy": strategy, "date": str(dates[t+2].date()),
                                "net_return": net[j], "equity": equity[j], "weight": w_sim[j],
                                "cost_fraction": cost[j], "turnover": turn[j],
                            })
            confidence = np.maximum(prob, 1-prob)
            hit = ((prob > .5) == (yt > 0)).astype(int)
            for lower, upper in [(0.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, 1.001)]:
                mask = (confidence >= lower) & (confidence < upper)
                reliability_rows.append({
                    **common, "bin_lower": lower, "bin_upper": upper, "count": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                    "hit_rate": float(hit[mask].mean()) if mask.any() else np.nan,
                    "mean_signed_return": float((np.sign(prob-.5)*simple)[mask].mean()) if mask.any() else np.nan,
                })
            for j, t in enumerate(test_idx):
                predictions.append({
                    **common, "origin_date": str(dates[t].date()),
                    "entry_date": str(dates[t+1].date()), "exit_date": str(dates[t+2].date()),
                    "actual_log_return": yt[j], "predicted_log_return": median[j],
                    "raw_probability_up": rawt[j], "calibrated_probability_up": prob[j],
                    "q10": qt[j, np.argmin(abs(levels-.1))],
                    "q90": qt[j, np.argmin(abs(levels-.9))],
                    "weight_cap1": weights["confidence_cap1"][j],
                    "weight_cap3": weights["confidence_cap3"][j],
                })
            lastp_raw = probability_up(q[-1, 1][None], levels)
            lastp = calibrator.predict(lastp_raw)
            # Latest-only refresh; NEVER used for the historical performance table.
            fresh_calibrator = Calibrator().fit(rawt[-ncal:], (yt[-ncal:] > 0).astype(int))
            freshp = fresh_calibrator.predict(lastp_raw)
            lastvol = np.std(r[-60:], ddof=1)*np.sqrt(252)
            latest_rows.append({
                **common, "as_of": str(dates[-1].date()), "last_adjusted_close": prices.iloc[-1],
                "raw_probability_up_h2": lastp_raw[0], "calibrated_probability_up_h2": lastp[0],
                "median_h2_log_return": q[-1, 1, np.argmin(abs(levels-.5))],
                "q10_h2": q[-1, 1, np.argmin(abs(levels-.1))],
                "q90_h2": q[-1, 1, np.argmin(abs(levels-.9))],
                "frozen_calibration_weight_cap3": sizes(lastp, np.array([lastvol]), 3)[0],
                "refreshed_probability_up_h2": freshp[0],
                "refreshed_weight_cap3": sizes(freshp, np.array([lastvol]), 3)[0],
                "annualized_volatility": lastvol,
            })
        price_path = ROOT / "results" / f"prices_{ticker}.npz"
        if price_path.exists():
            p = np.load(price_path)
            if np.isfinite(p["quantiles"]).all():
                pp, oo, ll = p["quantiles"], p["origins"], p["levels"]
                for h in [1, 5, 20]:
                    mask = oo[:-1]+h < len(prices)
                    origin = oo[:-1][mask]
                    actual = prices.to_numpy()[origin+h]
                    pred = pp[:-1, h-1, np.argmin(abs(ll-.5))][mask]
                    naive = prices.to_numpy()[origin]
                    price_metrics.append({
                        "ticker": ticker, "market": asset["market"], "horizon": h, "n": len(actual),
                        "price_r2": float(1-np.mean((actual-pred)**2)/np.var(actual)),
                        "price_mae": float(np.mean(abs(actual-pred))),
                        "naive_mae": float(np.mean(abs(actual-naive))),
                        "mae_skill_random_walk": float(1-np.mean(abs(actual-pred))/np.mean(abs(actual-naive))),
                    })
                for h in range(1, 21):
                    price_latest.append({
                        "ticker": ticker, "market": asset["market"], "horizon": h,
                        "as_of": str(prices.index[-1].date()),
                        "median": pp[-1, h-1, np.argmin(abs(ll-.5))],
                        "q10": pp[-1, h-1, np.argmin(abs(ll-.1))],
                        "q90": pp[-1, h-1, np.argmin(abs(ll-.9))],
                    })
    tables = {
        "forecast_metrics": metrics, "trading_metrics": trades, "equity_curves": curve_rows,
        "calibration": calibration_rows, "confidence_bins": reliability_rows,
        "latest_signals": latest_rows, "test_predictions": predictions,
        "price_metrics": price_metrics, "latest_price_forecasts": price_latest,
    }
    for name, rows in tables.items():
        if rows:
            pd.DataFrame(rows).to_csv(ROOT / "results" / f"{name}.csv", index=False)
            print(name, len(rows), flush=True)
    print("Evaluation complete", flush=True)


if __name__ == "__main__":
    main()
