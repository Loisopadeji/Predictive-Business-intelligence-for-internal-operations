# Assumptions and Limitations

*Companion to `summary.md`. Read this before quoting any figure from this analysis.*

## 1. This is a proxy dataset

The analysis runs on `Predictive_BI_for_internal_operation_data.csv`, a **proxy
dataset**: a stand-in that mirrors the structure of the firm's real internal project and
finance records, approved for prototyping while the live data connection is being
arranged. It is used so the pipeline, the validation and the outputs can be built and
stress-tested now, ahead of the real extract. Swapping in the real data later is a
change of input file, not a rewrite.

**What this means for the figures.** The numbers in `summary.md` describe the proxy
dataset, not confirmed live business performance. The dataset was prepared independently
of the model - the modelling code never sees how the data was produced and cannot be
influenced by it - so the method and its accuracy carry over to the real data; only the
specific dollar figures will change once the live extract replaces this file.

## 2. What the dataset looks like

These are properties **observed** in the proxy dataset (not settings we chose - we did
not produce this data). They are worth stating because they shape the forecast:

- **7 sectors**, ranging from Technology and Innovation (the largest by revenue) to
  Creative Economy and Tourism (the smallest). Sector mix and deal sizes vary widely.
- **A strong upward trend.** Firm-wide revenue grew from $34.0M in 2021 to $89.9M in
  2025 - it has more than doubled across the five years. This growth is the single most
  important feature of the series, and Section 4 explains why it dictated the choice of
  model.
- **A repeating yearly (seasonal) pattern.** January is the weakest month of the year
  and April the strongest; the calendar month accounts for about 18% of the model's
  predictive power - real, but not dominant. Note the strong season here is spring, not
  the year-end, so do not assume a Q4 peak.
- **Right-skewed contract values.** Most projects are modest, with a small number of
  very large contracts, so a few big deals can move a whole month - typical of
  professional services and a source of month-to-month lumpiness.
- **A 2.4% cancellation rate** (45 of 1,897 projects), which the pipeline removes before
  any revenue is counted.

## 3. Usable sample size

This is the most important limitation in the whole analysis, and it is easy to miss
because 1,897 sounds like a lot of data.

| Stage | Firm-wide rows | Per-sector rows |
|---|---|---|
| Projects in the raw file | 1,897 | 1,897 |
| After removing 45 cancelled projects | 1,852 | 1,852 |
| **After aggregating to monthly totals** | **60** | **420** |
| After dropping rows with no 3-month history | 57 | 399 |
| Available to train the model | **45** | **315** |
| Held back to test it | 12 | 84 |

Aggregating to monthly totals is what collapses 1,897 rows into 60. **The model is
effectively learning from 45 observations**, which is a very small sample by any
machine-learning standard, and it is the root cause of most of the limitations below.
With only 45 training months the model sees fewer than four examples of each calendar
month, so its view of, say, "what February looks like" rests on a handful of data
points.

## 4. Model limitations

**Trees cannot extrapolate.** A Random Forest predicts by averaging training values, so
it can never output a number outside the range it was trained on. Because this business
grows every year, a plain Random Forest was structurally incapable of predicting 2025
correctly - it under-forecast the holdout by 1,963,005 dollars a month on average, and
lost to simple Linear Regression (2,547,972 vs 2,311,540 MAE). The selected model fixes
this by fitting a straight-line trend first and letting the forest predict only the
movement around it. **This is the single most important technical finding in the
analysis** and it is why the headline forecast does not come from a plain Random Forest.

**Accuracy is modest in absolute terms.** The selected model was typically within 28%
($1.9M) of the actual month across the 12-month holdout. Monthly revenue in this dataset
ranges from $1.2M to $11.9M, so that error is real. The forecast supports direction-
setting and range-based planning; it does not support committing to a precise number.

**Error compounds across the quarter.** The forecast is recursive: February's prediction
uses January's prediction as an input, and March's uses both. Measured across 10
rolling-origin backtests, error grew from $2.4M one month out to $2.4M three months out
(x1.00). March is meaningfully less reliable than January, which is why the confidence
bands widen across the quarter rather than running parallel.

**A measured bias correction is applied.** The model under-forecast throughout the
backtest, because 2025 grew faster than the straight-line trend fitted to 2021-2024. We
add the measured correction ($-212K at one month out, rising to $106K at three) to each
forecast month. This is legitimate and standard, but it assumes the recent acceleration
continues. **If growth reverts to the longer-run trend, this forecast will be too
high.**

**The intervals are empirical, not theoretical.** Empirical (conformal) intervals,
derived from the spread of the model's own errors across a 30-forecast rolling-origin
backtest. The quantile-forest package (v1.4.2) WAS installed and was tried first, but
its bands did not transfer to the out-of-range Q1 2026 rows and came out roughly 1.4x
wider for the same coverage. Coverage was verified at 77% against a nominal 80%. Note
that quantile-forest was tried first, as the preferred method, and rejected on evidence
rather than preference - with 57 data points and a January 2026 whose inputs sit outside
the training range, its bands came out about 1.4x wider for the same coverage.

**The quarterly interval is deliberately conservative.** $16.6M to $38.8M is the sum of
the three monthly intervals, which assumes all three months miss in the same direction
together. In practice some cancellation is likely, so the true 80% range for the quarter
is probably narrower than stated.

**The two models do not have to agree.** The firm-wide model and the per-sector model
are separate. Before reconciliation the sector forecasts summed to about +15% of the
firm-wide figure. We kept the firm-wide total as the headline (aggregation cancels
noise, and it is the number we validated most carefully) and scaled the sectors
proportionally to match. The sector split is therefore a share-out of a validated total,
not seven independent forecasts.

**Only the signing date is modelled.** Revenue is booked entirely against `start_date`.
A project signed in March for $1.2M counts as $1.2M of March revenue, even if the work
and the cash span the following year. This is a bookings forecast, not a recognised-
revenue or cash forecast, and it should not be used for cash planning without being
reworked.

**Nothing outside the data is modelled.** No pipeline, no headcount, no win rates, no
pricing, no marketing spend, no competitor activity, no economic indicators. The model
only knows the calendar and the recent revenue history.

## 5. What changes when you use real data

1. **Expect the accuracy to be different, and check it before trusting it.** Real data
   is messier than a curated proxy dataset. Re-read the holdout comparison the pipeline
   prints - do not assume the 28% error carries over.
2. **Check the column names first.** The pipeline expects `start_date`,
   `contract_value_usd`, `sector` and `status`. It also expects `Cancelled` to be
   spelled that way; real systems often use several codes for the same thing.
3. **Expect real data quality problems.** Duplicate projects, missing dates, currency
   mixing, contract values revised after signing, projects re-entered under a new ID.
   Budget more time for cleaning than for modelling - that ratio is normal.
4. **The bias correction must be re-measured, not carried over.** It is specific to this
   dataset's growth pattern.
5. **Add the features you actually have.** Pipeline value at the start of each month,
   win rates, headcount and sales activity are usually far more predictive than the
   calendar alone. The calendar is what is left when you have nothing else.
6. **Re-examine the aggregation.** If your business is materially larger, weekly buckets
   would give roughly 4x the training rows and remove the single biggest constraint
   here.
7. **Re-run the whole pipeline rather than editing outputs.** Every number in
   `summary.md` and in this file is generated from the data. Re-running keeps them
   consistent; hand- editing guarantees they will eventually contradict each other.

---

*Generated 14 September 2026 by `forecast_q1_2026.py`. Model: Random Forest + trend
(hybrid). Training months: 45. Holdout months: 12.*
