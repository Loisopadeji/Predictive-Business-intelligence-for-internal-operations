# Data Dictionary — `synthetic_project_data.csv`

**Grain:** one row = one signed client project.
**Rows:** 1,897 projects
**Coverage:** 60 consecutive months, January 2021 → December 2025
**Sectors:** 7
**Source:** synthetic — produced by `generate_synthetic_data.py` (seed = 42). No real
client, employee, or financial data appears anywhere in this file.

---

## Columns

| # | Column | Type | Description | Notes for analysis |
|---|--------|------|-------------|--------------------|
| 1 | `project_id` | string | Unique project key, format `PRJ-00001`. | Primary key. Never null, never duplicated. |
| 2 | `client_id` | string | The client the project belongs to, format `CLI-0001`. | ~420 distinct clients; a client can have many projects. |
| 3 | `sector` | category | Industry the client operates in. One of 7 values (below). | The dimension used for the per-sector forecast. |
| 4 | `region` | category | Delivery region: `North America`, `EMEA`, `APAC`, `LATAM`. | Descriptive only — not used in the Q1 2026 model. |
| 5 | `client_type` | category | `New` or `Existing` at the time of signing. | ~63% Existing. |
| 6 | `delivery_mode` | category | `On-site`, `Hybrid`, or `Remote`. | Descriptive only. |
| 7 | `start_date` | date (`YYYY-MM-DD`) | Date the project started. | **The date the revenue is booked against.** The `month` column in the pipeline is derived from this. |
| 8 | `end_date` | date (`YYYY-MM-DD`) | Actual or scheduled completion date. | `end_date` = `start_date` + `duration_months`. |
| 9 | `duration_months` | integer | Project length in whole months, 1–24. | Correlated with `contract_value_usd` — bigger deals run longer. |
| 10 | `contract_value_usd` | float | **Total signed contract value in US dollars.** | **This is the forecast target.** Rounded to the nearest $100. Right-skewed (many mid-size deals, a few very large ones). |
| 11 | `status` | category | `Completed`, `In Progress`, or `Cancelled`. | See the critical note below. |
| 12 | `team_headcount` | integer | People assigned to the project, 1–40. | Descriptive only. |
| 13 | `billable_hours` | integer | Total billable hours booked. | Descriptive only. |
| 14 | `margin_pct` | float | Gross margin as a decimal (`0.28` = 28%). | Descriptive only. Mean ≈ 0.28. |

---

## The 7 sectors

| Sector | Projects | Character in the data |
|--------|----------|-----------------------|
| Financial Services | 355 | Largest by volume; steady growth. |
| Healthcare | 328 | Second by volume; above-average growth. |
| Public Sector | 276 | Slowest growth, but the strongest December spike (year-end budget flush). |
| Manufacturing | 273 | Middle of the pack on both size and growth. |
| Energy & Utilities | 251 | Largest average deal size, slowest-but-steady growth. |
| Retail & Consumer | 217 | Smallest average deal size. |
| Technology & Telecom | 197 | Lowest volume, but the fastest growth rate and a high average deal size. |

---

## ⚠️ Critical note on `status`

`status == 'Cancelled'` means the contract was **signed and entered into the system, but
the work was never delivered and no revenue was ever earned.** There are **118 such rows
(6.2% of the file)**.

These rows must be **removed before any revenue aggregation**. Leaving them in treats
signed-but-dead contracts as earned revenue and inflates every forecast produced from
the data. The pipeline drops them in its first data-prep step and logs the count.

`Completed` and `In Progress` both represent real, earned or in-flight revenue and are
**kept**.

---

## Known structure deliberately built into the data

Because this dataset is synthetic, we know the true underlying patterns. They are stated
here so you can check whether the model actually recovers them:

1. **Upward trend.** Both project count (+0.45%/month) and average deal size (+0.25% to
   +0.90%/month depending on sector) grow over the 60 months.
2. **Seasonality.** A repeating 12-month pattern: January and the July–August summer
   period are the weakest months; October, November and December are the strongest, with
   December the peak (~1.35× an average month).
3. **A Q4 effect.** The three Q4 months are all above 1.0 on the seasonal index — the
   reason the pipeline includes a binary Q4 flag as a feature.
4. **A one-off shock.** A softer patch across roughly June–September 2022 (~12% below
   trend), so the series is not unrealistically smooth.
5. **Right-skewed deal sizes.** Contract values are drawn from a lognormal distribution,
   so the mean sits above the median and a handful of very large contracts exist.

---

## Reproducing the file

```bash
python3 generate_synthetic_data.py
```

The random seed is fixed at 42, so the output is identical on every run and on every
machine.
