# Predictive Business Intelligence for Internal Operations — Q1 2026 Revenue Forecast

## Description

A board-ready monthly revenue forecast for **Q1 2026** (January, February, March),
produced for the firm as a whole and broken down across its seven sectors. The project
takes five years of project-level records, aggregates them to a monthly revenue series,
trains and validates a forecasting model, and produces the charts and written summaries a
leadership team needs to plan the quarter — each with an honest uncertainty range rather
than a single false-precision number.

The whole pipeline is written to be **read as well as run**: every step is commented in
plain language explaining what it does and *why*, as a learning artifact.

## Dataset

- **File:** `Predictive_BI_for_internal_operation_data.csv`
- **A proxy dataset** — a stand-in that mirrors the structure of the firm's real internal
  project and finance records, approved for prototyping while the live data connection is
  arranged. It lets the method be built and validated now; swapping in the real extract
  later is a change of input file, not a rewrite.
- **1,897 project records**, one row per signed project.
- **60 consecutive months**, January 2021 – December 2025. No gaps.
- **7 sectors:** Technology and Innovation, Climate and Carbon Markets, Agriculture,
  Energy, Finance and Insurance, Creative Economy and Tourism, Public Sector.
- Full column definitions are in
  [`data_dictionary_predictive_BI_for_internal_operations.md`](data_dictionary_predictive_BI_for_internal_operations.md).

The forecast uses four columns — `start_date`, `contract_value_usd`, `sector`, `status`.
Projects with `status == 'Cancelled'` (signed but never delivered, so no revenue earned)
are removed before anything is counted, to avoid inflating the forecast.

## Method

1. **Data prep** — confirm the 60-month span, remove cancelled projects (logged), and
   aggregate to monthly total contract value, both firm-wide and per sector.
2. **Feature engineering** — calendar month (seasonality), a time index (trend), lags of
   1–3 months (momentum), a 3-month rolling average (smoothing), a Q4 flag, and sector
   one-hot encoding. The rolling average and lag-3 make the first three months unusable,
   leaving **57 usable months, 45 for training**.
3. **Modelling** — a strictly chronological 80/20 split (no shuffling), with
   `GridSearchCV` tuning a Random Forest over `TimeSeriesSplit(n_splits=5)` so no fold
   ever trains on data that comes *after* the data it is validated on.
4. **Benchmark** — a Linear Regression, so the more complex model has to prove its worth.
5. **Forecast** — recursive month-by-month (each predicted month feeds the next month's
   lag features), with prediction intervals at 80% and 95%.

## Key results

| | Q1 2026 forecast |
|---|---|
| **Total** | **$26.4M** |
| 80% confidence range | $16.6M – $38.8M |
| vs Q1 2025 actual ($16.8M) | +58% |
| Monthly split | Jan $8.0M · Feb $9.1M · Mar $9.4M |

**Model selection (12-month holdout):**

| Model | MAE | RMSE | MAPE | vs benchmark (MAE) |
|---|---|---|---|---|
| Linear Regression (benchmark) | $2.31M | $3.02M | 32.3% | — |
| Random Forest (tuned) | $2.55M | $2.93M | 34.1% | **−10.2%** |
| **Random Forest + trend (hybrid, selected)** | **$1.86M** | **$2.37M** | **27.7%** | **+19.4%** |

- The **plain Random Forest lost to the simple benchmark**, because a tree can never
  predict a value outside its training range — a fatal flaw for a business that grows
  every year. The fix (and the model shipped) fits a linear trend first and lets the
  forest predict only the movement around it.
- Typical accuracy: within **~28% (~$1.9M)** of the actual month.
- Drivers are roughly balanced: the growth trend, recent months' revenue, and the calendar
  month each contribute ~19% of the model's predictive power — no single dominant factor.

## Business implications

- **January is reliably the weakest month; April is the strongest.** Plan Q1 cash and
  resourcing against the lower end of the range.
- **The strong season is spring (Q2), not the year-end.** In this data Q4 runs *below* the
  annual average, so a December-led "year-end push" narrative is not supported — Q1 is the
  run-up to the spring peak, and capacity should be readied for that.
- **Watch the least-predictable sectors.** Public Sector and Finance and Insurance are
  forecast least reliably (backtest error ~104% and ~87%, vs ~44% for the most
  predictable sector); tighter pipeline reporting there improves the whole forecast most.
- Treat the **range as the forecast** and the single figure as its midpoint.

## Tech Stack

- **Python 3.11**
- **pandas**, **numpy** — data handling
- **scikit-learn** — Random Forest, Linear Regression, `GridSearchCV`, `TimeSeriesSplit`,
  partial dependence
- **quantile-forest** — attempted for prediction intervals (see *Learning*)
- **matplotlib** — executive charts

Exact versions are pinned in [`requirements.txt`](requirements.txt).

## Files

| File | What it is |
|---|---|
| `forecast_q1_2026.py` | The full, heavily-commented pipeline. |
| `Predictive_BI_for_internal_operation_data.csv` | The proxy dataset. |
| `data_dictionary_predictive_BI_for_internal_operations.md` | Column-by-column definitions. |
| `forecast_results.csv` | History, holdout actual vs predicted, and the forecast with intervals. |
| `summary.md` | Five short paragraphs for the CEO — no jargon. |
| `assumptions.md` | Limitations, usable sample size, and what changes with real data. |
| `chart_firmwide_forecast.png` | Firm-wide history + Q1 2026 forecast with confidence bands. |
| `chart_sector_forecast.png` | The same, as small multiples for all seven sectors. |
| `chart_feature_importance.png` | What drives the forecast. |
| `chart_model_comparison.png` | Random Forest vs Linear Regression on the holdout. |
| `chart_partial_dependence.png` | How the top three drivers move the prediction. |

## How to reproduce

```bash
pip install -r requirements.txt

# Place the proxy dataset CSV in the project folder (already included here),
# then run the pipeline:
python3 forecast_q1_2026.py      # ~2 minutes; writes the CSV, all charts, and both .md files
```

Every number in `summary.md` and `assumptions.md`, and every chart, is generated by the
run — nothing is typed by hand — so re-running keeps them all consistent. To use the real
data later, drop in a CSV with the same four columns (`start_date`, `contract_value_usd`,
`sector`, `status`) and re-run; the numbers update themselves.

## Learning

- **A more powerful model is not automatically a better one.** A plain Random Forest,
  carefully tuned, lost to a one-line Linear Regression here — because tree models cannot
  extrapolate a trend. Always keep a simple benchmark; it is what tells you whether the
  complexity is earning its keep.
- **Validation for time series is different.** Shuffling rows leaks the future into the
  past. `TimeSeriesSplit` and a chronological holdout are what make the reported accuracy
  honest.
- **The preferred tool does not always win.** `quantile-forest` installed fine and was
  tried first, but with only 57 data points — and a Q1 2026 whose inputs sit outside the
  training range — its intervals came out ~1.4× wider than empirical intervals derived
  from the model's own backtest errors, for the same coverage. Both were implemented and
  compared; the pipeline states which it used.
- **Errors compound in a recursive forecast**, so March is measurably less certain than
  January — measured with a rolling-origin backtest, not assumed.
- **Let the data write the story.** Seasonal claims are computed from the data, not
  hardcoded, precisely so the write-up can never contradict the numbers.

## Next step

- Swap the proxy dataset for the live internal extract and re-validate accuracy on real
  data (do not assume the ~28% error carries over).
- Add features the calendar cannot capture: open pipeline value, win rates, headcount, and
  sales activity are usually far more predictive than time alone.
- Consider weekly (rather than monthly) buckets if data volume allows — it would roughly
  quadruple the training rows and ease the biggest constraint here (only 45 training months).
- Extend the same framework to the other four models the data dictionary anticipates
  (staffing/capacity, delivery risk, sector demand, client renewal).
