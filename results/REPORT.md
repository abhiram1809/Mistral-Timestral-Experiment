# Timestral-209M market assessment

The primary test does not establish a dependable trading edge.

4/9 instruments beat a zero-return forecast on MAE; 0/9 have a positive MAE improvement interval before multiple-testing adjustment. Confidence-sized accounts made money in 3/9 cases and beat passive holding in 3/9. Those counts do not establish statistical or prospective profitability.

Open `results/report.html` for figures, per-instrument accounts, calibration,
confidence sizing, sensitivity comparisons and latest forecasts.

Primary context: 250 observations. Per instrument: 200 prior calibration
origins and 500 test origins; forecast r[t+2] using observations through t,
enter at close t+1 and exit at close t+2. All trading results include the
predeclared costs and financing assumptions.

The CSV `results/overview.csv` gives the headline values. `results/market_summary.csv`
contains arithmetic averages of three instruments per group, not a market index
or a pooled multi-currency portfolio.

The public model training description does not rule out historical-data overlap.
This is not a prospective validation or evidence of dependable future profits.

See [protocol](../PROTOCOL.md), [model card](https://huggingface.co/mistralai/Timestral-209M),
[official code](https://github.com/mistralai/timestral) and
[forecast metrics](https://otexts.com/fpp3/accuracy.html).
