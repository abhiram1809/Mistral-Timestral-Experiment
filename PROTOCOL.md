# Timestral-209M financial forecasting experiment

Protocol written before model results were examined. Reference date: 2026-10-20
(Asia/Kolkata); exclude that date to avoid incomplete daily bars.

## Questions and sample

Does the 209M checkpoint beat zero-return / random-walk forecasts? Are its
probabilities calibrated? Does confidence sizing improve net simulated trading?
Nine deliberately selected liquid instruments cover India, the US and FX; this
is a small convenience sample, not a representative market index study.

Use the last 100, 250 and 500 daily observations as context. The predeclared
primary context is 250. For each context and instrument, take 200 earlier
origins for calibration, leave an embargo so every calibration label is known
before the first test forecast, then evaluate 500 consecutive historical
origins. No shuffled splits, no fitting on test results, no choosing the winning
context afterward. Longer-horizon outcomes are scored only when observed.

## Forecast and execution timing

Primary model input is daily adjusted-close log return. Forecast the next 20
individual daily returns; these are marginal forecasts, not cumulative-return
quantiles. At close t, forecast r[t+2]; submit the position for close t+1,
hold until close t+2. Thus the entire signal and risk estimate are known one
full bar before the entry. Position returns use exp(r[t+2])-1. This is a
research model of close execution, with explicit friction, not broker fills.

Independent price-level forecasts provide a diagnostic against the random
walk. High price R² alone cannot establish predictable returns. The latest
price forecast fan is a marginal price interval at each horizon.

## Calibration and position sizes

Interpolate the return quantile CDF at zero to estimate P(return > 0). Clamp
outside the reported tail quantiles; do not assert knowledge of the tails.
Fit regularized logistic calibration on the earlier 200 examples, separately
per instrument and context. Freeze it for the entire 500-observation test.
Calibration is allowed to learn an inverse relationship, but this is reported.
Also calibrate an 80% central interval using earlier standardized residuals.

Calibrated confidence = max(p, 1-p); direction = sign(p-0.5).
No position below 55% confidence. From 55% to 70%, allocation increases
linearly, reaching the full risk allocation at 70%.
Risk allocation = 10% / trailing 60-observation annualized volatility,
with a 1% annualized volatility floor. Cap gross exposure at 1x or 3x,
depending on the comparison. Confidence is a probability estimate, not a
guarantee; confidence sizing is itself under evaluation.

Compare cash, passive 1x holding, lagged 20-day momentum, raw-sign 1x,
calibrated-sign 1x, confidence-sized cap 1x, and confidence-sized cap 3x.
Charge costs on initial entry, every rebalance including reversals, and final
exit. Rebalance turnover uses the previous position after price drift.
Charge 8% annual financing on exposure above 1x and 3% annual short borrow
on short exposure, using elapsed calendar days. These are explicit illustrative
assumptions, not venue-specific quotes; double execution costs in a stress run.
No interest credited to cash. FX uses quote-currency P&L on a synthetic
exposure; actual carry, forward points, broker margin and conversion are not
modeled. Indian overnight shorts similarly require a suitable instrument or
borrowing arrangement. No claim of direct implementation through cash stocks.

## Scores

Report return MAE/RMSE, MAE and MSE skill versus zero, return R², directional
accuracy, Brier score versus a calibration-period constant probability,
pinball loss, quantile-integrated CRPS approximation, and 80% interval
coverage/width. Compare probabilistic loss with trailing empirical-return
quantiles. Report price MAE skill versus random walk separately.

Trading: ending capital, net total return, annualized return, Sharpe at zero
risk-free rate, worst close-to-close drawdown, realized volatility, exposure,
turnover, active days and winning active days. A position held through adjacent
days is not counted as multiple independent bets. Use circular moving-block
bootstrap intervals (20-observation blocks; 2,000 resamples) for mean net
returns and predictive improvements. The many comparisons are exploratory;
isolated apparent winners are not evidence after multiple testing.

## Limits and reproducibility

Cache raw Yahoo responses as CSV, record hashes, checkpoint revision,
upstream code commit, installed package versions, configuration and predictions.
Yahoo data are accessed through the unofficial yfinance client, not an official
broker execution feed. Adjustments are downloaded retrospectively; this is not
a point-in-time corporate-action database. Flag suspicious returns and missing
rows, do not silently winsorize or fill targets.

The model was released only on 2026-10-15. Its real-data training composition
does not establish exclusion of this historical sample. Chronological
evaluation prevents leakage in this experiment's calibration and trading
logic, but cannot certify that the pretrained weights never saw the history.
There are too few post-release observations for a genuine prospective test.
Survivorship, convenience-sample selection, changing regimes, thin calibration
samples and uncertain execution all limit generalization.

## Diagnostic additions during implementation

After the first two instruments completed, two additional fixed baselines were
added to distinguish predictive confidence from simply holding less risk:
`base_rate_sized` uses the calibration period's constant up probability, with
the same confidence and volatility formula; `volatility_scaled_sign` uses the
calibrated direction and volatility allocation without the confidence multiplier.
These are explanatory ablations, not predeclared strategy selection candidates.
The original strategy rules and primary context were not changed.

The user's requested Vulkan acceleration was implemented as an FP32 offload of
138 large Linear layers. A 36-origin price/return comparison across three
contexts checked agreement against official CPU inference. CPU and Vulkan
workers share a file-locked queue; a forecast job is owned by one worker at a
time. The model, precision, default symmetric averaging and test design remain
the same. This local backend is not supplied or certified by Mistral.
