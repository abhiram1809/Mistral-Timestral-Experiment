# Timestral-209M · Yahoo Finance experiment

[Read the report online](https://abhiram1809.github.io/Mistral-Timestral-Experiment/)
or [download the research bundle](results/direction-confidence-465/corrected-465-results.zip).

Latest assessment: [561 completed simulated trades across nine cases](results/direction-confidence-465/report.html),
with [every trade](results/direction-confidence-465/executed_trades.csv) and a
[research bundle](results/direction-confidence-465/corrected-465-results.zip)
containing inputs, forecasts, scripts and diagnostics. The target was at least
465 executions combined; the fixed full-round stopping rule produced 561, with
30,312 skipped signals. Original 55% / 60% / 70% leverage gates are retained.

This is a **distinct rolling directional-confidence strategy**, using only
completed past outcomes. It changes both the calibration updates and the
confidence event from the earlier frozen net-profit calibration. That original
one-trade experiment remains preserved and is not relabeled as a 561-trade run.

Five accounts lose, three stay in cash, and EUR/USD daily is +7.08% on the quoted
data. That positive result is fragile: excluding its largest winner gives
−4.28%. Its Yahoo entry quote differs sharply from an independently downloaded
ECB reference rate. The report flags extreme daily quotes and preserves the raw
inputs; it does not establish a reliable trading edge. Arithmetic, chronology,
source hashes and 45 calibration windows are audited; 42 tests pass.

The repository includes the data snapshots and consolidated forecast caches used
by the completed run. Model weights, local environments and temporary worker
queues are excluded. Install the pinned dependencies and upstream inference code
as described below, then reproduce the completed run from its saved forecasts:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/direction_465.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/report_corrected_465.py
```

Earlier data refresh: [expanded Yahoo minute, hourly and daily snapshot](results/more-yahoo-2026-10-21/report.html),
with downloadable cleaned CSVs, raw responses and a ZIP archive. Original confidence
gates and calibration parameters remain frozen. Reproduce with:

```bash
.venv/bin/python scripts/fetch_more_yahoo.py
.venv/bin/python scripts/report_more_yahoo.py
```

The preceding request: [465 executed trades combined, with original gates and original
calibration frozen](results/frozen-465/report.html). The full available eligible
history supports one execution; the report explicitly records the shortfall of
464, rather than lowering gates or using future calibration on earlier dates.
Reproduce with:

```bash
.venv/bin/python scripts/frozen_465.py
```

The preceding replay: [even lower confidence gates, including sub-percent
levels](results/ultra-low-confidence-200/report.html), on the same saved 1,800
opportunities. Primary gates are 0.1% / 1% / 5% for 1x / 2x / 3x; comparisons
include 1% / 5% / 10% and 0.1% / 0.2% / 0.3%. Reproduce with:

```bash
.venv/bin/python scripts/replay_ultra_low_confidence.py
```

The preceding replay: [lower confidence gates on the same saved 1,800
opportunities](results/lower-confidence-200/report.html). Primary gates are
35% / 40% / 45% for 1x / 2x / 3x, with two looser schedules also reported.
The saved forecasts and calibrated probabilities remain frozen. Reproduce with:

```bash
.venv/bin/python scripts/replay_lower_confidence.py
```

The preceding experiment: [confidence, skipping and leverage over 200
opportunities per case](results/confidence-200/report.html), using Reliance,
SPY and EUR/USD with minute, hourly and daily data. Executed trades and skipped
signals are counted separately. Its confidence definition and rules are in
[the experiment protocol](results/confidence-200/PROTOCOL.md).

Reproduce the confidence run from the saved data and forecast cache:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/confidence_200.py --mode opportunities
.venv/bin/python scripts/report_confidence_200.py
```

The preceding [plain 100-trade simulation](results/100-trades/report.html) uses
fixed 1x positions. The broader original daily-market study is described below.

The preceding ungated simulation used **one instrument per category, minute/hourly/daily
data, exactly 100 trades per case**. Open
[`results/100-trades/report.html`](results/100-trades/report.html) or
[`results/100-trades/summary.csv`](results/100-trades/summary.csv).
It uses raw next-price model directions and fixed 1x positions, with no confidence
or volatility overlay. `all_900_trades.csv` contains the complete ledger.

To rerun that simulation using the saved inputs:

```bash
.venv/bin/python scripts/prepare_100_trades.py
.venv/bin/python scripts/simulate_100_trades.py --backend cpu --threads 3
.venv/bin/python scripts/report_100_trades.py
```

The earlier broader daily study is documented below.

An empirical assessment of Mistral's actual 209,312,256-parameter checkpoint
on three Indian equities, three U.S. instruments and three currency pairs.
Includes probabilistic forecast evaluation, chronological calibration,
confidence-dependent position sizes and net simulated accounts.

**Start with [`results/report.html`](results/report.html)** after the run.
The HTML is self-contained and works offline. Figures are also standalone PNGs.
The design and limitations are in [`PROTOCOL.md`](PROTOCOL.md).

## Reproduce

The pinned experiment uses Python 3.14.7; use Python 3.14 for these exact dependency
versions. The upstream model itself supports Python 3.10+. CPU installation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/mistralai/timestral.git vendor/timestral
git -C vendor/timestral checkout a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
.venv/bin/python -m pip install -e vendor/timestral -r requirements.txt
mkdir -p data results
.venv/bin/python scripts/prepare.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/infer.py --batch 64 --threads 4
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/audit.py
.venv/bin/python scripts/report.py
.venv/bin/python -m pytest -q tests
```

The saved experiment pins model revision
`9874d252ab0aae16f7f7c999f98f98d6ad7d8365`. `prepare.py` resolves the current
revision on a fresh run; preserve `data/manifest.json` and the cached Yahoo
CSVs to reproduce this exact snapshot. Never reuse old forecasts after changing
configuration, data or checkpoint; use a separate results directory / checkout.

Forecasts are saved atomically after each batch, so rerunning inference resumes.
The original float32 model and default symmetric averaging are used.
Data are requested up to but excluding 2026-10-20; the cutoff is enforced
again locally because Yahoo FX responses included an extra bar.

## Optional Vulkan kernel

`scripts/vulkan_linear.cpp` and `.comp` implement an FP32 Vulkan matrix
multiplication backend for the transformer's large linear projections. The
remaining official operations run in PyTorch on CPU. This is a custom inference
optimization, not an official Mistral backend and not quantization.

```bash
git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git vendor/Vulkan-Headers
glslc -O scripts/vulkan_linear.comp -o scripts/vulkan_linear.spv
g++ -O3 -std=c++17 -fPIC -shared scripts/vulkan_linear.cpp \
  -I vendor/Vulkan-Headers/include -Wl,-l:libvulkan.so.1 \
  -o scripts/libvulkan_linear.so
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_vulkan.py
# Only after validation passes and measured performance is acceptable:
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/infer.py --backend vulkan --threads 2 --batch 64
```

This implementation chooses a discrete Vulkan device, supports inference only,
and uses a synchronous shared workspace (not thread safe). The validation checks
irregular matrix shapes and end-to-end price/return forecasts across all three
context lengths. Compare `results/benchmark*.log` for measured throughput.

## Outputs

| File | Contents |
|---|---|
| `results/report.html` | Offline research report and all charts |
| `results/overview.csv` | Primary-context headline results |
| `results/forecast_metrics.csv` | Accuracy and probabilistic scores, all contexts/horizons |
| `results/trading_metrics.csv` | Seven original strategies plus two diagnostic baselines, base/double costs |
| `results/test_predictions.csv` | Every test signal, dates, realized outcome and sized exposure |
| `results/equity_curves.csv` | Daily account value, costs, exposure and turnover |
| `results/calibration.csv` | Frozen calibration fits, probabilities, Brier and interval coverage |
| `results/confidence_bins.csv` | Whether high-confidence directions actually win more often |
| `results/latest_signals.csv` | Frozen and refreshed latest-only calibrated signals |
| `results/latest_price_forecasts.csv` | Next 20 model price forecasts and marginal intervals |
| `results/next_price_snapshot.csv` | Readable next-close price snapshot |
| `results/price_metrics.csv` | Price-level diagnostic against random walk |
| `results/figures/` | Exportable PNG charts |
| `data/manifest.json` | Yahoo snapshot dates, hashes, quality checks, model revision |
| `results/environment.txt` | Exact installed package inventory |
| `results/audit.json` | Verified data, chronology, P&L and numerical checks |

The predefined primary setting is 250 return observations. The test comprises
500 daily decisions following 200 calibration examples, with an embargo.
Signals formed at close t trade at close t+1 and exit at close t+2.
“High confidence → more leverage” is tested, not assumed to work.
No money is deposited, no brokerage is connected, and no actual orders are sent.

Historical training overlap is unknown. These results cannot establish
prospective trading profitability; read the report's uncertainty and execution
limitations before interpreting any apparently attractive result.
