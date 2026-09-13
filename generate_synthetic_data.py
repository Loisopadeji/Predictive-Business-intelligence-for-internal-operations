"""
generate_synthetic_data.py
==========================

WHY THIS FILE EXISTS
--------------------
The forecasting pipeline (`forecast_q1_2026.py`) needs a dataset called
`synthetic_project_data.csv`. This script builds that dataset from scratch so the
whole project is reproducible: anyone can clone the repo, run this file, and get
byte-for-byte the same CSV, because we fix the random seed.

"Synthetic" means the numbers are invented by a computer, not measured from a real
company. We invent them using a *data-generating process* (DGP) - a set of rules
that mimic how consulting-style project revenue actually behaves:

    revenue = baseline level x growth trend x seasonal pattern x random noise

Because we WROTE the rules, we know the true answer. That is genuinely useful for
learning: when the model later says "seasonality matters", we can check that against
the seasonality we deliberately put in. With real data you never get that luxury.

HOW TO READ THIS FILE
---------------------
Every block has a comment explaining WHAT it does and WHY it is there.
Run it with:  python3 generate_synthetic_data.py
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# 1. REPRODUCIBILITY
# ----------------------------------------------------------------------------------
# A "seed" fixes the starting point of the random number generator. Without it, every
# run would produce different numbers and your forecast would change each time you
# ran the code - which makes debugging impossible and results impossible to defend in
# a board meeting. With it, randomness becomes *repeatable* randomness.
SEED = 42
rng = np.random.default_rng(SEED)

# The spec we are matching, kept in named constants rather than scattered through the
# code. If a number needs to change, it changes in exactly one place.
N_PROJECTS = 1897                 # total rows in the final CSV
START_MONTH = "2021-01"           # first month of history
END_MONTH = "2025-12"             # last month of history  -> 60 months inclusive
OUTPUT_CSV = "synthetic_project_data.csv"

# ----------------------------------------------------------------------------------
# 2. THE MONTH GRID
# ----------------------------------------------------------------------------------
# pd.period_range gives us calendar months as first-class objects (2021-01, 2021-02...).
# We use Periods rather than raw strings because Periods know how to sort and how to do
# arithmetic ("what is 3 months after 2021-11?"), which strings do not.
months = pd.period_range(START_MONTH, END_MONTH, freq="M")
n_months = len(months)
assert n_months == 60, f"Expected 60 months, built {n_months}"

# ----------------------------------------------------------------------------------
# 3. THE SEVEN SECTORS
# ----------------------------------------------------------------------------------
# Each sector gets three dials, because in real life sectors differ in three ways:
#   share        - how much of the firm's project VOLUME it accounts for (must sum to 1)
#   base_value   - the typical size of one contract in that sector (USD)
#   growth       - compound MONTHLY growth in deal size (0.004 = +0.4%/month = ~+4.9%/yr)
# Public Sector, for example, is given the strongest Q4 behaviour later on, because
# government budgets genuinely do get spent before the fiscal year closes.
SECTORS = {
    "Financial Services":  {"share": 0.20, "base_value": 310_000, "growth": 0.0055},
    "Healthcare":          {"share": 0.16, "base_value": 245_000, "growth": 0.0070},
    "Energy & Utilities":  {"share": 0.13, "base_value": 420_000, "growth": 0.0035},
    "Public Sector":       {"share": 0.15, "base_value": 275_000, "growth": 0.0025},
    "Manufacturing":       {"share": 0.14, "base_value": 265_000, "growth": 0.0040},
    "Retail & Consumer":   {"share": 0.12, "base_value": 190_000, "growth": 0.0050},
    "Technology & Telecom":{"share": 0.10, "base_value": 355_000, "growth": 0.0090},
}
sector_names = list(SECTORS.keys())
sector_shares = np.array([SECTORS[s]["share"] for s in sector_names])
sector_shares = sector_shares / sector_shares.sum()   # normalise so it sums to exactly 1

# ----------------------------------------------------------------------------------
# 4. SEASONALITY: WHICH MONTHS ARE BUSY?
# ----------------------------------------------------------------------------------
# A "seasonal index" is a multiplier per calendar month. 1.00 = an average month,
# 1.30 = 30% busier than average. These values encode a pattern that is very common in
# B2B services:
#   - January is slow (budgets not yet released, holidays)
#   - there is a summer dip (July/August holidays)
#   - Q4 spikes hard, and December hardest (use-it-or-lose-it budget spending)
# We put this in ON PURPOSE so the model has a real seasonal signal to discover.
SEASONAL_INDEX = np.array([
    0.82,  # Jan
    0.90,  # Feb
    1.05,  # Mar - end of Q1, some budget release
    0.98,  # Apr
    1.02,  # May
    1.08,  # Jun - end of H1 push
    0.85,  # Jul - summer lull
    0.80,  # Aug - summer lull
    1.06,  # Sep - post-summer restart
    1.15,  # Oct - Q4 begins
    1.22,  # Nov
    1.35,  # Dec - year-end budget flush
])

# ----------------------------------------------------------------------------------
# 5. HOW MANY PROJECTS START IN EACH MONTH?
# ----------------------------------------------------------------------------------
# Project COUNT also grows over time (the firm is winning more work) and follows the
# same seasonal shape. We build a relative "intensity" per month, then convert those
# intensities into actual project counts that sum to exactly N_PROJECTS.
time_index = np.arange(n_months)                    # 0, 1, 2, ... 59
volume_trend = 1.0 + 0.0045 * time_index            # +0.45% more projects per month
month_numbers = np.array([m.month for m in months]) # 1..12 repeating
volume_seasonal = SEASONAL_INDEX[month_numbers - 1] # -1 because arrays are 0-indexed

# A one-off shock: a real business is never a smooth curve. We add a mild slowdown in
# mid-2022 (a "market wobble") so the series is not unrealistically clean.
shock = np.ones(n_months)
shock[17:21] *= 0.88                                # Jun-Sep 2022 softer

monthly_intensity = volume_trend * volume_seasonal * shock
monthly_probability = monthly_intensity / monthly_intensity.sum()

# rng.multinomial distributes N_PROJECTS across the 60 months according to those
# probabilities. It guarantees the counts add up to exactly N_PROJECTS - a plain
# rounding of probabilities would not.
projects_per_month = rng.multinomial(N_PROJECTS, monthly_probability)

# ----------------------------------------------------------------------------------
# 6. BUILD ONE ROW PER PROJECT
# ----------------------------------------------------------------------------------
# We loop month by month, and inside each month create that month's projects. Building
# a list of dictionaries and converting once at the end is much faster than appending
# to a DataFrame row by row (which recopies the whole table every time).
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
REGION_WEIGHTS = [0.42, 0.31, 0.19, 0.08]
DELIVERY_MODES = ["On-site", "Hybrid", "Remote"]
DELIVERY_WEIGHTS = [0.24, 0.46, 0.30]

rows = []
project_counter = 1

for i, month in enumerate(months):
    n_this_month = projects_per_month[i]

    # Assign each of this month's projects to a sector, weighted by sector share.
    sectors_this_month = rng.choice(sector_names, size=n_this_month, p=sector_shares)

    for sector in sectors_this_month:
        cfg = SECTORS[sector]

        # --- Contract value ------------------------------------------------------
        # Deal sizes are RIGHT-SKEWED in real life: lots of small/medium projects and a
        # few very large ones. A lognormal distribution produces exactly that shape,
        # which is why we use it instead of a normal (bell curve) distribution - a
        # normal curve would imply as many tiny deals as huge ones, and could even
        # generate negative contract values, which is nonsense.
        size_multiplier = rng.lognormal(mean=0.0, sigma=0.45)

        # Trend: deal sizes compound upward month by month at the sector's own rate.
        trend_multiplier = (1.0 + cfg["growth"]) ** i

        # Seasonality on VALUE (separate from seasonality on COUNT): busy months also
        # tend to carry slightly bigger deals. We damp it to 60% so we are not
        # double-counting the seasonality already present in project counts.
        seas = SEASONAL_INDEX[month.month - 1]
        value_seasonal = 1.0 + (seas - 1.0) * 0.6

        # Public Sector gets an extra December kick - year-end budget flush is a real
        # and well-documented government procurement behaviour.
        if sector == "Public Sector" and month.month == 12:
            value_seasonal *= 1.18

        contract_value = cfg["base_value"] * size_multiplier * trend_multiplier * value_seasonal
        contract_value = float(np.round(contract_value, -2))   # round to nearest $100

        # --- Dates ---------------------------------------------------------------
        # Spread start dates across the days of the month so start_date looks like a
        # real date column rather than everything landing on the 1st.
        day = int(rng.integers(1, 29))          # 1-28 keeps every month valid
        start_date = pd.Timestamp(year=month.year, month=month.month, day=day)

        # Bigger contracts take longer. We tie duration to value so the two columns are
        # realistically correlated instead of independent.
        base_duration = 3 + (contract_value / 200_000)
        duration_months = int(np.clip(np.round(rng.normal(base_duration, 2.0)), 1, 24))
        end_date = start_date + pd.DateOffset(months=duration_months)

        # --- Status --------------------------------------------------------------
        # ~6% of signed projects get cancelled. These are the rows the forecasting
        # pipeline must REMOVE: the contract was signed (so it is in the system) but no
        # work was delivered and no revenue was earned. Leaving them in would inflate
        # every forecast.
        roll = rng.random()
        if roll < 0.061:
            status = "Cancelled"
        elif end_date > pd.Timestamp("2025-12-31"):
            status = "In Progress"      # still running at the end of our history window
        else:
            status = "Completed"

        # --- Other descriptive columns -------------------------------------------
        # These are not used by the forecast but make the dataset realistic, and give
        # you extra columns to practise exploratory analysis on.
        headcount = int(np.clip(np.round(contract_value / 95_000 + rng.normal(0, 1.2)), 1, 40))
        billable_hours = int(np.round(headcount * duration_months * 150 * rng.uniform(0.75, 1.05)))
        margin_pct = float(np.round(np.clip(rng.normal(0.28, 0.07), 0.02, 0.55), 4))
        client_type = "Existing" if rng.random() < 0.63 else "New"

        rows.append({
            "project_id":         f"PRJ-{project_counter:05d}",
            "client_id":          f"CLI-{int(rng.integers(1, 421)):04d}",
            "sector":             sector,
            "region":             str(rng.choice(REGIONS, p=REGION_WEIGHTS)),
            "client_type":        client_type,
            "delivery_mode":      str(rng.choice(DELIVERY_MODES, p=DELIVERY_WEIGHTS)),
            "start_date":         start_date.strftime("%Y-%m-%d"),
            "end_date":           end_date.strftime("%Y-%m-%d"),
            "duration_months":    duration_months,
            "contract_value_usd": contract_value,
            "status":             status,
            "team_headcount":     headcount,
            "billable_hours":     billable_hours,
            "margin_pct":         margin_pct,
        })
        project_counter += 1

# ----------------------------------------------------------------------------------
# 7. SAVE
# ----------------------------------------------------------------------------------
df = pd.DataFrame(rows)

# Sort by start_date so the file reads chronologically - convenient for humans opening
# it in Excel, and it costs nothing.
df = df.sort_values("start_date").reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)

# A short report so you can sanity-check the output without opening the file.
print("=" * 70)
print("SYNTHETIC DATASET GENERATED")
print("=" * 70)
print(f"Rows (projects)     : {len(df):,}")
print(f"Columns             : {len(df.columns)}")
print(f"Date range          : {df['start_date'].min()} to {df['start_date'].max()}")
print(f"Distinct months     : {df['start_date'].str[:7].nunique()}")
print(f"Sectors             : {df['sector'].nunique()}")
print(f"Total contract value: ${df['contract_value_usd'].sum():,.0f}")
print("\nStatus breakdown:")
print(df["status"].value_counts().to_string())
print("\nProjects per sector:")
print(df["sector"].value_counts().to_string())
print(f"\nSaved to: {OUTPUT_CSV}")
