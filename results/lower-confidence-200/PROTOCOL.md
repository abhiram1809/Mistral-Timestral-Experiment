# Lower-confidence replay protocol

Reuse the saved 1,800 opportunity records and their frozen calibrated profit
probabilities. Do not rerun the model, refit calibration or refresh Yahoo data.
Primary thresholds: 35% / 40% / 45% for 1x / 2x / 3x. Sensitivity schedules:
25% / 30% / 35% and 10% / 20% / 30%. The highest qualifying exposure wins.
Unchanged median forecasts always skip. Reproduce the original 55% / 60% / 70%
rule and compare the saved unfiltered 1x baseline. Each case starts at 100,000
in its original quote currency; costs, holding periods and compounding are fixed.

Rules are fixed before this replay's P&L calculations, but earlier results have
already been inspected. This is post-hoc sensitivity analysis of the same sample,
not a new prospective or untouched validation set. No schedule is selected by
return optimization, and all schedules are reported. Nine independent simulated
accounts retain 200 opportunities each; executions and skips are separate.
Five scenario replays do not create 9,000 independent observations.
