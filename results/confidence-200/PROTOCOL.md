# Confidence, skipping and leverage: 200 per case

Written before this experiment's forecasts and outcomes are evaluated.

- Instruments: RELIANCE.NS, SPY, EURUSD=X; intervals: 1 minute, 1 hour, 1 day.
- Model: pinned official Timestral-209M checkpoint, FP32, default symmetric
  averaging; input: last 250 adjusted closing prices; horizon: next close.
- Same ideal close-to-next-close paper fills as the 100-trade report.
- Each independent account starts with 100,000 in its quote currency.
- User-confirmed counting mode: 200 opportunities per case, 1,800 opportunities
  total. Executions and skipped opportunities are reported separately. An
  abstention is never labelled an executed trade.

## What confidence means

Direction confidence and profit confidence are different. Direction confidence
is the model's probability of a price move in its median-predicted direction.
Profit confidence is the estimated probability that a proposed position earns
positive P&L after execution fees, financing and short borrow in this simulation.
Profit confidence controls the new strategy. A high probability of winning does
not by itself establish positive expected return; payoff sizes still matter.

Timestral supplies 21 price quantiles. Linearly interpolate this quantile CDF
at the current price for direction probability and at the position's break-even
price for profit probability. The reported tails stop at 1% and 99%; do not
extrapolate beyond them. Flat median forecasts propose no trade.

Use the preceding 300 completed forecast origins as calibration data. Simulate
the proposed direction at 1x, 2x and 3x on those earlier outcomes. Fit a monotone
regularized sigmoid map from raw profit probability to observed profitable
outcomes, pooling these levels with weight 1/3 per level so each origin has unit
weight. Center log odds on the calibration mean. A nonnegative sigmoid slope
preserves probability ordering; two virtual opposing labels at that mean
stabilize the intercept when successes are rare. Clamp test scores to the raw
calibration score range, avoiding unsupported confidence extrapolation. Freeze this
map before evaluating later origins; no test outcomes adjust the map or rules.

## Rules fixed before evaluation

- Take 3x only when its calibrated probability of positive net P&L is >=70%.
- Otherwise take 2x only when its probability is >=60%.
- Otherwise take 1x only when its probability is >=55%.
- Otherwise skip. Never lower a threshold to manufacture a trade count.
- Trade direction remains the model median direction, without discretionary
  changes. Close each accepted trade at the next quote and compound its P&L.

Use the same per-fill cost assumptions as the earlier report: India 10 bps,
US 5 bps, FX 2 bps. Charge both entry and exit. Financing: 8% annually on actual
exposure above 1x. Short borrow: 3% annually on short notional. Use elapsed
calendar time. No interest on cash; FX carry and executable spreads are not
available in this quote data. Entry fees reduce investable capital.

## Report

Show actual executions, skipped signals, leverage counts, gross/net returns,
ending balance and drawdown. Include unfiltered 1x and uncalibrated versions on
the same chronological evaluation opportunities so periods remain comparable.
Measure probability quality on later outcomes with Brier score, log loss,
reliability bins and sample counts. Include a calibration-period constant
probability baseline. Evaluate every opportunity's hypothetical outcome even if
the strategy skips it, avoiding a selected-trades-only calibration assessment.

Yahoo data through October 19, 2026; source hashes and forecast caches retained.
Corporate-action adjustments are retrospective. Pretraining overlap is unknown.
This is research on ideal quoted fills and does not model inference/order
latency, margin calls, forced liquidation or actual contract availability.
