# Q1 2026 Revenue Forecast

A monthly revenue forecast for Q1 2026 — firm-wide and across seven sectors — built from
five years of project-level history, with prediction intervals, model validation, and
board-ready outputs.

> **The data is synthetic.** Every figure here describes an invented company. See
> [`assumptions.md`](assumptions.md) before quoting any number.

## Headline

**Q1 2026 forecast: $58.6M** — $16.6M (Jan), $20.6M (Feb), $21.3M (Mar).
80% confidence range $33.9M – $73.8M. Typical error in testing: 26% (~$4.3M) per month.

## Quick start

```bash
pip install -r requirements.txt
python3 generate_synthetic_data.py   # builds synthetic_project_data.csv (seed=42)
python3 forecast_q1_2026.py          # runs everything, writes all outputs (~2 min)
```

Both scripts are heavily commented and meant to be read top to bottom.

## Files

| File | What it is |
|---|---|
| `generate_synthetic_data.py` | Builds the 1,897-project dataset. Fixed seed, fully reproducible. |
| `data_dictionary.md` | Every column explained, plus the patterns deliberately built into the data. |
| `forecast_q1_2026.py` | The pipeline: data prep → features → modelling → forecast → outputs. |
| `forecast_results.csv` | History, holdout actual vs predicted, and the forecast with intervals. |
| `summary.md` | Five paragraphs for the CEO. No jargon. |
| `assumptions.md` | Limitations, sample size, and what changes with real data. |
| `chart_firmwide_forecast.png` | Five years of history plus the Q1 2026 forecast with confidence bands. |
| `chart_sector_forecast.png` | The same, as small multiples across all seven sectors. |
| `chart_feature_importance.png` | What drives the forecast. |
| `chart_model_comparison.png` | Random Forest vs Linear Regression on the holdout. |
| `chart_partial_dependence.png` | How the top three drivers move the prediction. |

## What the pipeline does

1. **Data prep** — verifies the 60-month span, removes 118 cancelled projects (signed but
   never delivered, so no revenue was earned), aggregates to monthly totals.
2. **Features** — calendar month (seasonality), time index (trend), lags 1–3 (momentum),
   3-month rolling average (smoothing), Q4 flag, sector one-hot. Dropping rows without a
   full 3-month history leaves **57 usable months, 45 for training**.
3. **Modelling** — chronological 80/20 split, no shuffling. `GridSearchCV` with
   `TimeSeriesSplit(n_splits=5)` so no fold ever trains on data that comes after what it
   validates on.
4. **Forecast** — recursive: each predicted month feeds the next month's lag features.
   Error compounding is measured, not assumed, via a rolling-origin backtest.

## Two findings worth knowing

**A plain Random Forest lost to Linear Regression.** Trees predict by averaging training
values, so they cannot output a number outside the range they were trained on. In a
growing business that makes them structurally incapable of forecasting forward — the plain
forest under-predicted the holdout by ~$3.6M/month. The fix, and the model actually
shipped, fits a linear trend first and lets the forest predict only the movement around it.

**`quantile-forest` installed fine but lost the interval bake-off.** With 57 data points
and a January 2026 whose inputs sit outside the training range, its bands came out ~2.2×
wider than empirical intervals derived from the model's own backtest errors, for the same
coverage. Both methods are implemented; the pipeline measures each and picks on evidence,
stating clearly which it used.
