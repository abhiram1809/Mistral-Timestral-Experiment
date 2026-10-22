# Corrected confidence overlay: executed-trade target

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
