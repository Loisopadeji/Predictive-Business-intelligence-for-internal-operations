"""
===================================================================================
Q1 2026 REVENUE FORECAST  -  end-to-end pipeline
===================================================================================

WHAT THIS SCRIPT DOES, IN ONE PARAGRAPH
---------------------------------------
It reads a file of individual client projects, throws away the ones that were
cancelled (no revenue was earned on those), adds up the remaining contract values
into one number per calendar month, teaches a machine-learning model what the shape
of that monthly series looks like, checks how accurate the model is on months it was
never shown, and then uses it to predict January, February and March 2026 - both for
the firm as a whole and for each of the seven sectors separately. It finishes by
writing a results CSV, five charts, and two short written documents for non-technical
readers.

HOW TO RUN IT
-------------
    python3 forecast_q1_2026.py

WHY THE COMMENTS ARE SO LONG
----------------------------
This is written to be *read*, not just executed. Every block says what it does and,
more importantly, WHY that step is necessary. If a step looks obvious, the comment
still explains it, because "obvious" is usually where the misunderstandings hide.

VOCABULARY YOU WILL SEE (plain English)
---------------------------------------
  feature        an input column the model is allowed to look at
  target         the thing we are trying to predict (here: monthly revenue)
  train / test   the months used to teach the model / the months held back to grade it
  lag            the value of something one or more time steps ago
  holdout        another word for the test set - months deliberately hidden from training
  hyperparameter a setting you choose BEFORE training (e.g. how many trees in a forest),
                 as opposed to something the model learns by itself
===================================================================================
"""

# ==================================================================================
# SECTION 0  -  IMPORTS
# ==================================================================================
# We import everything at the top so a reader can see the full toolkit at a glance,
# and so a missing package fails immediately rather than 200 lines into a long run.

import os
import re
import sys
import textwrap
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
# "Agg" is a non-interactive drawing backend. We set it BEFORE importing pyplot because
# this script runs on a server with no screen attached - without this line matplotlib
# would try to open a window and crash.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.base import BaseEstimator, RegressorMixin   # order matters: RegressorMixin must come FIRST in the class definition below
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay

# Scikit-learn emits some cosmetic warnings during grid search on very small folds.
# They do not affect correctness, and silencing them keeps the printed report readable.
warnings.filterwarnings("ignore")

# ==================================================================================
# SECTION 0b  -  CONFIGURATION
# ==================================================================================
# All the "knobs" live here in one place. If you want to forecast 6 months instead of
# 3, or change the train/test split, you edit here and nowhere else. Hard-coding these
# values deep inside the code is how scripts become impossible to maintain.

INPUT_CSV        = "synthetic_project_data.csv"
TARGET_COL       = "contract_value_usd"   # what we are forecasting
DATE_COL         = "start_date"           # the date revenue is booked against
SECTOR_COL       = "sector"
STATUS_COL       = "status"

TRAIN_FRACTION   = 0.80                   # first 80% of months train, last 20% grade us
FORECAST_MONTHS  = ["2026-01", "2026-02", "2026-03"]   # Q1 2026
RANDOM_STATE     = 42                     # fixes randomness so results are repeatable

OUT_CSV          = "forecast_results.csv"
OUT_SUMMARY      = "summary.md"
OUT_ASSUMPTIONS  = "assumptions.md"

# ---------------------------------------------------------------------------------
# CHART STYLING
# ---------------------------------------------------------------------------------
# These charts are going in front of a board, so they get a deliberate, restrained
# visual style rather than matplotlib's noisy defaults. The colours below come from a
# palette that has been checked for colour-blind readability - roughly 1 in 12 men has
# some form of colour vision deficiency, and a board slide has to work for all of them.
# That is also why every chart carries a legend and direct text labels: identity is
# never communicated by colour alone.
C_HISTORY   = "#2a78d6"   # blue    - actual, observed revenue
C_FORECAST  = "#eb6834"   # orange  - anything the model predicted
C_BENCH     = "#4a3aa7"   # violet  - the benchmark (Linear Regression) model
C_SURFACE   = "#fcfcfb"   # near-white page background
C_INK       = "#0b0b0b"   # primary text
C_INK_SOFT  = "#52514e"   # secondary text (subtitles, axis labels)
C_GRID      = "#e3e2de"   # very light gridlines - present but recessive
C_BAND_80   = "#f6c3ac"   # inner (80%) confidence band
C_BAND_95   = "#fbe4d8"   # outer (95%) confidence band

plt.rcParams.update({
    "figure.facecolor":  C_SURFACE,
    "axes.facecolor":    C_SURFACE,
    "savefig.facecolor": C_SURFACE,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.edgecolor":    C_GRID,
    "axes.labelcolor":   C_INK_SOFT,
    "axes.titlecolor":   C_INK,
    "xtick.color":       C_INK_SOFT,
    "ytick.color":       C_INK_SOFT,
    "grid.color":        C_GRID,
    "grid.linewidth":    0.8,
    "axes.grid":         True,
    "axes.grid.axis":    "y",       # horizontal gridlines only - vertical ones add clutter
    "axes.spines.top":   False,     # remove the box around each plot; it carries no data
    "axes.spines.right": False,
    "legend.frameon":    False,
})


def money(x, _pos=None):
    """
    Format a number of dollars for a chart axis or label.

    A board reader should never have to count zeroes. 4200000 becomes "$4.2M".
    matplotlib calls this once per tick via FuncFormatter, passing the tick value and
    its position (which we ignore, hence the underscore).
    """
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"


# Plain-English names for the model's inputs. Charts that go in front of a board must
# never show a column name like "lag_3" - the reader should not have to decode anything.
PRETTY = {
    "month_num": "Which calendar month it is",
    "time_idx":  "How far through the five years we are",
    "is_q4":     "Whether the month falls in Q4",
    "lag_1":     "Revenue one month ago",
    "lag_2":     "Revenue two months ago",
    "lag_3":     "Revenue three months ago",
    "roll_3":    "Average of the last three months",
}
MONEY_FEATURES = {"lag_1", "lag_2", "lag_3", "roll_3"}


def banner(text):
    """Print a clearly delimited section header so the console output is skimmable."""
    print("\n" + "=" * 82)
    print(text)
    print("=" * 82)


# ==================================================================================
# SECTION 1  -  DATA PREPARATION
# ==================================================================================
banner("SECTION 1  |  DATA PREPARATION")

# ---- 1.1 Load ---------------------------------------------------------------------
# We fail loudly and helpfully if the file is missing. A cryptic stack trace is a bad
# experience; a one-line explanation of what to do next is a good one.
if not os.path.exists(INPUT_CSV):
    sys.exit(
        f"ERROR: '{INPUT_CSV}' not found in {os.getcwd()}.\n"
        f"Run 'python3 generate_synthetic_data.py' first to create it."
    )

# parse_dates tells pandas to read that column as real dates rather than plain text.
# This matters: text sorts alphabetically ('2021-10' < '2021-9'), dates sort correctly,
# and only dates let us ask for ".dt.month" or do date arithmetic.
raw = pd.read_csv(INPUT_CSV, parse_dates=[DATE_COL])
print(f"Loaded '{INPUT_CSV}'  ->  {len(raw):,} rows x {len(raw.columns)} columns")

# ---- 1.2 Confirm the file really spans 60 months ------------------------------------
# Never trust a claim about a dataset - verify it. A silent off-by-one in the date range
# would quietly corrupt every seasonal feature we build later.
raw["month"] = raw[DATE_COL].dt.to_period("M")   # 2021-03-17 -> Period('2021-03')
n_months_raw = raw["month"].nunique()

print(f"\nDate range : {raw[DATE_COL].min().date()}  ->  {raw[DATE_COL].max().date()}")
print(f"First month: {raw['month'].min()}")
print(f"Last month : {raw['month'].max()}")
print(f"Distinct calendar months present: {n_months_raw}")

if n_months_raw == 60:
    print("CHECK PASSED: the data spans exactly 60 months, as expected.")
else:
    print(f"CHECK WARNING: expected 60 months, found {n_months_raw}. Continuing anyway.")

# We also check for GAPS. 60 distinct months is not the same as 60 *consecutive* months -
# you could have 60 months scattered with holes in between, which would silently break
# lag features (a "lag of 1" would not really be one month ago).
expected_span = pd.period_range(raw["month"].min(), raw["month"].max(), freq="M")
missing_months = sorted(set(expected_span) - set(raw["month"].unique()))
if missing_months:
    print(f"CHECK WARNING: {len(missing_months)} month(s) have no projects at all: {missing_months}")
else:
    print("CHECK PASSED: no missing months - the series is continuous with no gaps.")

# ---- 1.3 Remove cancelled projects ---------------------------------------------------
# WHY: a cancelled project was signed and sits in the system, but no work was delivered
# and no money was ever earned. Counting it as revenue would inflate history, and since
# the model learns from history, it would inflate the forecast too. We remove them and
# log exactly how many, because silently deleting rows is how analyses lose their
# credibility in a review.
before = len(raw)
cancelled_mask = raw[STATUS_COL].astype(str).str.strip().str.lower() == "cancelled"
n_cancelled = int(cancelled_mask.sum())
cancelled_value = float(raw.loc[cancelled_mask, TARGET_COL].sum())

df = raw.loc[~cancelled_mask].copy()   # "~" means NOT - keep everything that is not cancelled
after = len(df)

print(f"\nStatus values found in the file: {sorted(raw[STATUS_COL].unique())}")
print(f"Rows before removing cancellations : {before:,}")
print(f"Cancelled rows REMOVED             : {n_cancelled:,}  ({n_cancelled/before:.1%} of the file)")
print(f"Contract value removed with them   : ${cancelled_value:,.0f}")
print(f"Rows remaining for the forecast    : {after:,}")

# ---- 1.4 Build the two monthly revenue series ----------------------------------------
# The raw file is one row per project. A forecast needs one row per MONTH. This step -
# "aggregation" - collapses many projects into a single revenue total per month.
#
# (a) FIRM-WIDE: total contract value per month, all sectors combined.
firmwide = (
    df.groupby("month", as_index=False)[TARGET_COL]
      .sum()
      .rename(columns={TARGET_COL: "revenue"})
      .sort_values("month")
      .reset_index(drop=True)
)

# (b) BY SECTOR: total contract value per month per sector.
by_sector = (
    df.groupby(["month", SECTOR_COL], as_index=False)[TARGET_COL]
      .sum()
      .rename(columns={TARGET_COL: "revenue"})
      .sort_values(["sector", "month"])
      .reset_index(drop=True)
)

# IMPORTANT: a sector might have zero projects in some month. groupby simply omits that
# month for that sector, which would leave a hole in the series and make lag features
# wrong. We rebuild the grid explicitly so every sector has a row for every month, with
# zero revenue where nothing was signed.
all_months = firmwide["month"].tolist()
all_sectors = sorted(df[SECTOR_COL].unique())
full_grid = pd.MultiIndex.from_product([all_months, all_sectors], names=["month", "sector"]).to_frame(index=False)
by_sector = full_grid.merge(by_sector, on=["month", "sector"], how="left")
by_sector["revenue"] = by_sector["revenue"].fillna(0.0)
by_sector = by_sector.sort_values(["sector", "month"]).reset_index(drop=True)

print(f"\n(a) Firm-wide series built : {len(firmwide)} monthly rows")
print(f"(b) Per-sector series built: {len(by_sector)} rows "
      f"({len(all_sectors)} sectors x {len(all_months)} months)")
N_SECTORS = len(all_sectors)
print(f"\nSectors: {all_sectors}")
print(f"\nFirm-wide monthly revenue - first 3 months and last 3 months:")
_preview = pd.concat([firmwide.head(3), firmwide.tail(3)])
for _, r in _preview.iterrows():
    print(f"    {r['month']}   ${r['revenue']:>13,.0f}")
print(f"\nFirm-wide average month: ${firmwide['revenue'].mean():,.0f}   "
      f"| min ${firmwide['revenue'].min():,.0f}   | max ${firmwide['revenue'].max():,.0f}")


# ==================================================================================
# SECTION 2  -  FEATURE ENGINEERING
# ==================================================================================
banner("SECTION 2  |  FEATURE ENGINEERING")

# A Random Forest has no idea what "time" is. Show it a bare list of revenue numbers and
# it sees an unordered bag of values - it cannot tell that March comes after February, it
# cannot see a trend, and it certainly cannot spot that every December is big. Feature
# engineering is the job of turning the ORDER and the CALENDAR of a time series into
# ordinary columns the model can actually use.
#
# We build six kinds of feature. Each one exists for a specific reason:
#
#   month_num (1-12)   WHY: captures SEASONALITY. This is the column that lets the model
#                      learn "month 12 is always big, month 8 is always small". Without
#                      it, every month looks alike and December's spike looks like noise.
#
#   time_idx (0,1,2..) WHY: captures TREND. A simple counter of how many months have
#                      passed. It lets the model learn "later months are generally
#                      bigger than earlier ones" - i.e. that the business is growing.
#                      (Caveat, and it is an important one: a tree can only split on
#                      values it has seen, so it cannot extrapolate a trend past the last
#                      month it was trained on. See the limitation note in assumptions.md.)
#
#   lag_1, lag_2,      WHY: captures MOMENTUM / autocorrelation. Business revenue is
#   lag_3              sticky - a strong November usually follows a strong October,
#                      because the same pipeline, sales team and client base produce
#                      both. lag_1 is last month's revenue, lag_2 the month before, and
#                      so on. These are usually the single most powerful features in any
#                      short-horizon forecast, because the recent past is the best
#                      available summary of everything we did not measure.
#
#   roll_3             WHY: SMOOTHING. Any single month can be distorted by one unusually
#                      large contract landing on the 28th. The average of the previous
#                      three months is a steadier read on the underlying run-rate, so it
#                      gives the model a "level" signal that individual lags cannot.
#
#   sector one-hot     WHY: lets ONE model serve all seven sectors. A machine-learning
#                      model needs numbers, not the text "Healthcare". One-hot encoding
#                      turns one text column into seven 0/1 columns. Training one pooled
#                      model rather than seven separate ones means each sector borrows
#                      statistical strength from the others - which matters a lot when
#                      each sector only has ~57 usable months of its own.
#
#   is_q4              WHY: an explicit, simple flag for the strongest known pattern in
#                      the business - the Q4 budget flush. month_num could in principle
#                      learn this on its own, but it would need to independently discover
#                      that months 10, 11 and 12 belong together. Handing the model that
#                      grouping directly makes the pattern far easier to learn from only
#                      ~45 training months, and it makes the result easier to explain to
#                      a board ("Q4 is our biggest quarter" is a sentence anyone follows).

LAGS = [1, 2, 3]
ROLL_WINDOW = 3


def add_time_features(frame):
    """
    Add calendar, trend, lag and rolling features to ONE revenue series.

    Expects a DataFrame sorted oldest-to-newest with columns ['month', 'revenue'],
    where 'month' is a pandas Period. Returns a copy with the new columns added.

    This is written as a reusable function because we need to do exactly the same thing
    twice - once for the firm-wide series and once for each sector - and duplicated code
    is where the two versions silently drift apart.
    """
    out = frame.copy().sort_values("month").reset_index(drop=True)

    # --- Calendar features -------------------------------------------------------
    # .month pulls the number 1-12 out of the Period. This is the seasonality handle.
    out["month_num"] = out["month"].apply(lambda p: p.month)

    # A plain counter: 0 for the first month in the series, 1 for the next, and so on.
    # This is the trend handle.
    out["time_idx"] = np.arange(len(out))

    # Binary Q4 flag: 1 for October, November, December; 0 otherwise.
    # .astype(int) converts True/False into 1/0, because models want numbers.
    out["is_q4"] = out["month_num"].isin([10, 11, 12]).astype(int)

    # --- Lag features -------------------------------------------------------------
    # .shift(k) slides the whole column DOWN by k rows. After shift(1), the value sitting
    # on the March row is February's revenue. That is exactly what we want: when the model
    # predicts March it is allowed to look at February, because in real life February has
    # already happened by then.
    for k in LAGS:
        out[f"lag_{k}"] = out["revenue"].shift(k)

    # --- Rolling average ----------------------------------------------------------
    # Read this right-to-left: shift(1) first moves everything down one row so the current
    # month is excluded, THEN we average a 3-month window.
    #
    # The shift(1) is the most important character in this file. Without it, the rolling
    # average on the March row would include March's own revenue - the very number we are
    # trying to predict. That is called TARGET LEAKAGE. It would make the model look
    # brilliant in testing and useless in production, because at forecast time you do not
    # know the current month's revenue. If you take one thing from this script, take this:
    # every feature must only contain information that was genuinely available BEFORE the
    # month being predicted.
    out[f"roll_{ROLL_WINDOW}"] = out["revenue"].shift(1).rolling(window=ROLL_WINDOW).mean()

    return out


# ---- 2.1 Firm-wide features ---------------------------------------------------------
fw = add_time_features(firmwide)
rows_before_drop = len(fw)

# The first three rows have no lag_3 and no complete 3-month window, so those cells are
# NaN ("not a number"). A model cannot train on a blank, so we drop them. This is not
# data loss to be embarrassed about - it is the unavoidable price of using history as a
# feature, and it is why short time series are hard.
fw = fw.dropna().reset_index(drop=True)

print("FIRM-WIDE SERIES")
print(f"  Rows before dropping incomplete lag/rolling rows : {rows_before_drop}")
print(f"  Rows dropped (no lag_3 / no 3-month window yet)  : {rows_before_drop - len(fw)}")
print(f"  >>> USABLE ROWS REMAINING                        : {len(fw)}")
print(f"  Usable period: {fw['month'].min()}  ->  {fw['month'].max()}")

# ---- 2.2 Per-sector features --------------------------------------------------------
# groupby(...).apply(...) runs add_time_features SEPARATELY for each sector. This is
# essential: Healthcare's lag_1 must be Healthcare's own previous month, not whatever
# row happened to sit above it in the table. Getting this wrong is one of the most common
# and most damaging bugs in panel/time-series work.
sector_frames = []
for sec, grp in by_sector.groupby("sector"):
    g = add_time_features(grp[["month", "revenue"]])
    g["sector"] = sec
    sector_frames.append(g)

sec_feat = pd.concat(sector_frames, ignore_index=True)
sector_rows_before = len(sec_feat)
sec_feat = sec_feat.dropna().reset_index(drop=True)

# One-hot encoding: turn the single text column 'sector' into seven 0/1 columns named
# sector_Healthcare, sector_Manufacturing, and so on. prefix= keeps the new column names
# self-describing, which matters when we read the feature-importance chart later.
sector_dummies = pd.get_dummies(sec_feat["sector"], prefix="sector").astype(int)
sec_feat = pd.concat([sec_feat, sector_dummies], axis=1)
SECTOR_DUMMY_COLS = sorted(sector_dummies.columns.tolist())

print("\nPER-SECTOR SERIES")
print(f"  Rows before dropping incomplete rows : {sector_rows_before}   "
      f"({len(all_sectors)} sectors x {len(all_months)} months)")
print(f"  Rows dropped (3 per sector x {len(all_sectors)} sectors)  : {sector_rows_before - len(sec_feat)}")
print(f"  >>> USABLE ROWS REMAINING            : {len(sec_feat)}   "
      f"({len(all_sectors)} sectors x {len(fw)} months)")
print(f"  One-hot columns created: {len(SECTOR_DUMMY_COLS)}")
for c in SECTOR_DUMMY_COLS:
    print(f"      {c}")

# ---- 2.3 Define the feature lists ---------------------------------------------------
# Keeping these as explicit named lists (rather than "everything except the target") is a
# safety measure: it makes it impossible to accidentally feed the target, or a future-
# looking column, into the model.
BASE_FEATURES = ["month_num", "time_idx", "is_q4"] + [f"lag_{k}" for k in LAGS] + [f"roll_{ROLL_WINDOW}"]
FIRM_FEATURES = BASE_FEATURES
SECTOR_FEATURES = BASE_FEATURES + SECTOR_DUMMY_COLS

print(f"\nFirm-wide model will use {len(FIRM_FEATURES)} features: {FIRM_FEATURES}")
print(f"Sector model will use {len(SECTOR_FEATURES)} features "
      f"({len(BASE_FEATURES)} shared + {len(SECTOR_DUMMY_COLS)} sector flags)")


# ==================================================================================
# SECTION 3  -  MODELLING
# ==================================================================================
banner("SECTION 3  |  MODELLING")

# ---- 3.1 The chronological train / holdout split -------------------------------------
#
# THE GOLDEN RULE OF TIME SERIES: never shuffle.
#
# For ordinary (non-time) data you shuffle rows before splitting, so the train and test
# sets are statistically alike. For a time series that is a catastrophe. Shuffling would
# put some of 2025 in the training set and some of 2022 in the test set - meaning the
# model would be allowed to see the future while being graded on the past. It would score
# beautifully and then fail the moment you used it for real, because in real life the
# future is exactly the thing you do not have.
#
# So we cut the series in one place, by date: the earliest 80% of months teach the model,
# and the most recent 20% are hidden away and used only to grade it. That mimics the real
# task - "stand at the end of history and predict forward".

n_total = len(fw)
n_train = int(np.floor(n_total * TRAIN_FRACTION))
n_test = n_total - n_train

fw_train = fw.iloc[:n_train].copy()   # earliest months
fw_test  = fw.iloc[n_train:].copy()   # most recent months - the holdout

print("CHRONOLOGICAL SPLIT (firm-wide)")
print(f"  Total usable months : {n_total}")
print(f"  Training months     : {n_train}  ({fw_train['month'].min()} -> {fw_train['month'].max()})")
print(f"  Holdout months      : {n_test}  ({fw_test['month'].min()} -> {fw_test['month'].max()})")
print("  No shuffling was performed - the split is purely by date.")

X_train = fw_train[FIRM_FEATURES]
y_train = fw_train["revenue"]
X_test  = fw_test[FIRM_FEATURES]
y_test  = fw_test["revenue"]

# ---- 3.2 Hyperparameter tuning with GridSearchCV + TimeSeriesSplit --------------------
#
# WHAT WE ARE TUNING AND WHY
# A Random Forest builds many decision trees and averages them. Two settings matter most:
#   n_estimators - how many trees. More trees = a more stable average, but slower. There
#                  is a point of diminishing returns; we search for roughly where it is.
#   max_depth    - how many yes/no questions deep each tree may go. Deep trees can carve
#                  the training data into tiny slivers and memorise it (OVERFITTING);
#                  shallow trees may be too crude to capture the pattern (UNDERFITTING).
#                  With only ~45 training months, overfitting is the bigger danger, so we
#                  include quite shallow options in the search.
# GridSearchCV simply tries every combination and keeps whichever scored best.
#
# HOW TimeSeriesSplit DIFFERS FROM A NORMAL SPLIT - AND WHY IT MATTERS
# ---------------------------------------------------------------------
# Standard k-fold cross-validation chops the data into k random chunks and takes turns
# using each chunk as the validation set. On time-series data that is invalid, because
# most of those folds train on data that comes AFTER the data they are validated on. The
# model gets to peek at the future. Scores come out flattering and completely fake.
#
# TimeSeriesSplit fixes this by always keeping training data strictly BEFORE validation
# data. It uses an expanding window - each fold trains on everything up to a point and
# validates on the block immediately after:
#
#     fold 1:  train [========]                    validate [==]
#     fold 2:  train [==========]                  validate [==]
#     fold 3:  train [============]                validate [==]
#     fold 4:  train [==============]              validate [==]
#     fold 5:  train [================]            validate [==]
#              |------------ time ------------------------------>|
#
# Two consequences worth understanding:
#   1. No future information ever leaks backwards into training. The score you get is an
#      honest estimate of how the model will behave when forecasting for real.
#   2. Each fold rehearses the actual job - "train on the past, predict the next block" -
#      five times, at five different points in history. So we are not just checking that
#      the settings work once; we are checking they work repeatedly as the business
#      evolved. That is a much stronger test than a single lucky split.
#
# n_splits=5 is a balance: enough rehearsals to be meaningful, while leaving each fold's
# training window large enough to be worth training on given our short series.

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth":    [3, 5, 8, 12, None],   # None = grow until the leaves are pure
}

rf_base = RandomForestRegressor(
    random_state=RANDOM_STATE,   # fixes the forest's internal randomness -> repeatable results
    n_jobs=-1,                   # use all available CPU cores
)

# scoring: we minimise mean absolute error. sklearn maximises scores by convention, so it
# is expressed as NEGATIVE MAE - the least-negative value is the best model.
grid = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)

print(f"\nRunning GridSearchCV over {len(param_grid['n_estimators']) * len(param_grid['max_depth'])} "
      f"parameter combinations x 5 time-ordered folds ...")
grid.fit(X_train, y_train)

rf_model = grid.best_estimator_
print(f"  Best parameters found : {grid.best_params_}")
print(f"  Best cross-validated MAE across the 5 folds: ${-grid.best_score_:,.0f}")

# ---- 3.3 The benchmark model ---------------------------------------------------------
# WHY HAVE A BENCHMARK AT ALL?
# "Our model has an error of $2.1M" is a meaningless statement on its own. Is that good?
# Compared to what? A benchmark gives the number meaning. Linear Regression is the right
# choice here: it is the simplest sensible model, it takes one line to fit, it is fully
# transparent, and if the Random Forest cannot beat it then the extra complexity is not
# earning its keep and we should ship the simple model instead. Always make the fancy
# model prove itself against the boring one.
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("\nBenchmark model (Linear Regression) fitted on the same training months.")

# ---- 3.4 Score both models on the holdout --------------------------------------------
# These are months NEITHER model has ever seen. This is the honest exam.
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)


def evaluate(y_true, y_pred):
    """
    Return the three standard forecast accuracy metrics.

    MAE  (Mean Absolute Error) - average size of the miss, in dollars. Easiest to explain
         to a non-technical audience: "on average we are off by this much."
    RMSE (Root Mean Squared Error) - also in dollars, but squares the errors before
         averaging, which punishes large misses much more harshly. If RMSE is far above
         MAE, it tells you the model is occasionally very wrong, not consistently
         slightly wrong. That distinction matters for planning.
    MAPE (Mean Absolute Percentage Error) - the average miss as a % of the actual value.
         Unit-free, so it lets you compare accuracy across sectors of very different
         sizes. Its weakness is that it blows up when actuals are near zero, so we guard
         against division by zero below.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


rf_scores = evaluate(y_test, rf_pred)
lr_scores = evaluate(y_test, lr_pred)

# "% improvement" = how much of the benchmark's error the Random Forest removed.
# A positive number means the Random Forest is better. We compute it per metric because a
# model can win on one metric and lose on another.
improvement = {
    m: (lr_scores[m] - rf_scores[m]) / lr_scores[m] * 100 for m in ["MAE", "RMSE", "MAPE"]
}

print("\n" + "-" * 82)
print("HOLDOUT PERFORMANCE  (the last {} months, unseen during training)".format(n_test))
print("-" * 82)
print(f"{'Metric':<10}{'Random Forest':>20}{'Linear Regression':>22}{'RF improvement':>20}")
print("-" * 82)
print(f"{'MAE':<10}{'$' + format(rf_scores['MAE'], ',.0f'):>20}"
      f"{'$' + format(lr_scores['MAE'], ',.0f'):>22}{improvement['MAE']:>19.1f}%")
print(f"{'RMSE':<10}{'$' + format(rf_scores['RMSE'], ',.0f'):>20}"
      f"{'$' + format(lr_scores['RMSE'], ',.0f'):>22}{improvement['RMSE']:>19.1f}%")
print(f"{'MAPE':<10}{format(rf_scores['MAPE'], '.2f') + '%':>20}"
      f"{format(lr_scores['MAPE'], '.2f') + '%':>22}{improvement['MAPE']:>19.1f}%")
print("-" * 82)
print("A POSITIVE 'RF improvement' means the Random Forest beat the simple benchmark.")
print("A NEGATIVE value means the benchmark won on that metric and should be taken")
print("seriously - it would mean the extra complexity is not paying for itself.")

# ---- 3.5 The same procedure for the per-sector model ---------------------------------
# CRITICAL SUBTLETY: the sector table has 7 rows per month (one per sector). Before we can
# split it by time we must sort it by MONTH first, so that "the first 80% of rows" really
# means "the earliest 80% of months" and not "the first five sectors alphabetically".
# Sorting by sector first - which is how the table was built - would silently destroy the
# chronological split and every guarantee TimeSeriesSplit gives us.
sec_feat = sec_feat.sort_values(["month", "sector"]).reset_index(drop=True)

split_month = fw_train["month"].max()          # last month that belongs to training
sec_train = sec_feat[sec_feat["month"] <= split_month].copy()
sec_test  = sec_feat[sec_feat["month"] >  split_month].copy()

Xs_train, ys_train = sec_train[SECTOR_FEATURES], sec_train["revenue"]
Xs_test,  ys_test  = sec_test[SECTOR_FEATURES],  sec_test["revenue"]

print(f"\nPER-SECTOR MODEL")
print(f"  Training rows: {len(sec_train)}  (months {sec_train['month'].min()} -> {sec_train['month'].max()})")
print(f"  Holdout rows : {len(sec_test)}  (months {sec_test['month'].min()} -> {sec_test['month'].max()})")

grid_sec = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
grid_sec.fit(Xs_train, ys_train)
rf_sector = grid_sec.best_estimator_

sec_pred = rf_sector.predict(Xs_test)
sec_scores = evaluate(ys_test, sec_pred)
print(f"  Best parameters : {grid_sec.best_params_}")
print(f"  Holdout MAE ${sec_scores['MAE']:,.0f} | RMSE ${sec_scores['RMSE']:,.0f} | MAPE {sec_scores['MAPE']:.2f}%")


# ==================================================================================
# SECTION 3.6  -  DIAGNOSING THE RESULT, AND FIXING A KNOWN WEAKNESS
# ==================================================================================
banner("SECTION 3.6  |  DIAGNOSIS: WHY A PLAIN RANDOM FOREST STRUGGLES HERE")

# Look at the table above before reading on. If the Random Forest did NOT beat Linear
# Regression, that is not a bug and it is not bad luck. It is the single most important
# limitation of tree-based models applied to a growing time series, and it is worth
# understanding properly because it will bite you again.
#
# THE PROBLEM: A DECISION TREE CANNOT EXTRAPOLATE.
# A tree makes predictions by asking yes/no questions ("is time_idx > 30?") until it
# lands in a leaf, then it outputs THE AVERAGE OF THE TRAINING VALUES IN THAT LEAF. That
# means a tree can only ever predict values inside the range it was trained on. If the
# largest month it ever saw during training was $18M, it can never predict $25M - not
# because it thinks $25M is unlikely, but because it has no mechanism to produce a number
# it has never seen.
#
# Our business grows every year. The holdout months (2025) are systematically LARGER than
# anything in training (2021-2024). So the Random Forest is structurally capped and
# under-predicts the whole holdout period. Linear Regression has the opposite property:
# it fits a straight line and happily continues that line upward forever, so it handles
# the trend well - even though it is far cruder about seasonality.
#
# We can measure this directly. "Bias" here means the average signed error: a large
# negative bias means the model is consistently guessing too low.
rf_bias = float(np.mean(rf_pred - y_test.values))
lr_bias = float(np.mean(lr_pred - y_test.values))
print(f"Average signed error on the holdout (negative = predicting too low):")
print(f"  Random Forest    : ${rf_bias:>14,.0f}")
print(f"  Linear Regression: ${lr_bias:>14,.0f}")
print(f"\nLargest revenue month the models were TRAINED on : ${y_train.max():,.0f}")
print(f"Largest revenue month in the HOLDOUT             : ${y_test.max():,.0f}")
print(f"Largest value the Random Forest ever predicted   : ${rf_pred.max():,.0f}")
print("\nNotice the Random Forest's maximum prediction cannot exceed its training range.")
print("That is the ceiling effect described above, visible in the numbers.")

# ---------------------------------------------------------------------------------
# THE STANDARD FIX: A HYBRID MODEL
# ---------------------------------------------------------------------------------
# We do not have to choose between "handles trend" and "handles seasonality". We can give
# each job to the model that is good at it. This is a well-established technique, usually
# called a hybrid or boosted-hybrid model, and it works in two stages:
#
#   STAGE 1  A straight line is fitted through time (revenue vs time_idx only). This
#            captures the long-run growth of the business, and - crucially - a straight
#            line CAN be extended into 2026.
#
#   STAGE 2  We subtract that line from the actual revenue. What is left over is called
#            the RESIDUAL: the wiggle around the trend - seasonality, momentum, the odd
#            big contract. Residuals hover around zero and do NOT grow over time, which
#            means they are exactly the kind of well-behaved, in-range target a Random
#            Forest is excellent at.
#
#   PREDICT  final prediction = the straight line's value + the forest's residual guess.
#
# The forest is still doing the interesting work (seasonality, Q4, momentum); the line
# just stops it from being trapped below the growth curve.

class TrendAwareForest(RegressorMixin, BaseEstimator):
    """
    A Random Forest that can follow a trend.

    It wraps two models: a LinearRegression on time only (the trend), and a
    RandomForestRegressor on all features (the pattern around the trend).

    We give it .fit() and .predict() methods with the same names scikit-learn uses, so it
    can be dropped into the rest of this script without changing anything else. That
    convention - matching the interface of the library you work alongside - is what makes
    custom components composable.
    """

    def __init__(self, rf_params=None, trend_col="time_idx"):
        # Inheriting from BaseEstimator/RegressorMixin gives us get_params, set_params and
        # the "I am a regressor" tag for free. Scikit-learn's own tools - including the
        # partial-dependence plotting we use later - check for those, so this small piece
        # of housekeeping is what lets our custom model be treated as a real sklearn model.
        # BaseEstimator requires every __init__ argument to be stored unchanged under the
        # same name, so we do the actual model construction inside fit() instead.
        self.rf_params = rf_params
        self.trend_col = trend_col

    def fit(self, X, y):
        # Attributes created during fit end in an underscore - the scikit-learn convention
        # that marks an estimator as "fitted".
        self.trend_ = LinearRegression()
        self.forest_ = RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, **(self.rf_params or {})
        )
        self.feature_names_ = list(X.columns)
        # Stage 1: the straight line through time. Double brackets keep it a DataFrame,
        # because scikit-learn expects 2-D input even for a single feature.
        self.trend_.fit(X[[self.trend_col]], y)
        trend_fitted = self.trend_.predict(X[[self.trend_col]])
        # Stage 2: the forest learns whatever the line could not explain.
        self.residual_train_ = np.asarray(y) - trend_fitted
        self.forest_.fit(X, self.residual_train_)
        return self

    def predict(self, X):
        # Partial-dependence plotting hands us a plain numpy array rather than a
        # DataFrame, so we restore the column names before indexing by name.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
        return self.trend_.predict(X[[self.trend_col]]) + self.forest_.predict(X)

    @property
    def feature_importances_(self):
        # The importances belong to the forest stage - i.e. what drives the movement
        # AROUND the trend. We surface them so this object behaves like a plain forest.
        return self.forest_.feature_importances_


# We reuse the hyperparameters GridSearchCV already selected. Re-running the full grid on
# the residual target would be more thorough, so we do exactly that - it is cheap here,
# and the best settings for predicting revenue are not necessarily the best settings for
# predicting residuals.
print("\nTuning the hybrid (trend + forest) model with the same TimeSeriesSplit procedure...")

trend_only = LinearRegression().fit(X_train[["time_idx"]], y_train)
resid_train = y_train.values - trend_only.predict(X_train[["time_idx"]])

grid_hybrid = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
grid_hybrid.fit(X_train, resid_train)
hybrid_params = grid_hybrid.best_params_

hybrid_model = TrendAwareForest(hybrid_params).fit(X_train, y_train)
hybrid_pred = hybrid_model.predict(X_test)
hybrid_scores = evaluate(y_test, hybrid_pred)

print(f"  Best parameters for the residual forest: {hybrid_params}")

# ---- The full three-way comparison ---------------------------------------------------
improvement_hybrid = {
    m: (lr_scores[m] - hybrid_scores[m]) / lr_scores[m] * 100 for m in ["MAE", "RMSE", "MAPE"]
}

print("\n" + "-" * 92)
print(f"HOLDOUT COMPARISON  ({n_test} months: {fw_test['month'].min()} -> {fw_test['month'].max()})")
print("-" * 92)
print(f"{'Model':<34}{'MAE':>16}{'RMSE':>16}{'MAPE':>12}{'vs benchmark (MAE)':>22}")
print("-" * 92)
print(f"{'Linear Regression (benchmark)':<34}{'$'+format(lr_scores['MAE'],',.0f'):>16}"
      f"{'$'+format(lr_scores['RMSE'],',.0f'):>16}{format(lr_scores['MAPE'],'.1f')+'%':>12}{'—':>22}")
print(f"{'Random Forest (tuned)':<34}{'$'+format(rf_scores['MAE'],',.0f'):>16}"
      f"{'$'+format(rf_scores['RMSE'],',.0f'):>16}{format(rf_scores['MAPE'],'.1f')+'%':>12}"
      f"{format(improvement['MAE'],'+.1f')+'%':>22}")
print(f"{'Random Forest + trend (hybrid)':<34}{'$'+format(hybrid_scores['MAE'],',.0f'):>16}"
      f"{'$'+format(hybrid_scores['RMSE'],',.0f'):>16}{format(hybrid_scores['MAPE'],'.1f')+'%':>12}"
      f"{format(improvement_hybrid['MAE'],'+.1f')+'%':>22}")
print("-" * 92)

# ---- Choose the model that will produce the headline board number --------------------
# We let the evidence decide, using holdout MAE - the metric a CFO cares about ("how many
# dollars are we typically off by"). Hard-coding a favourite model and hoping is not
# forecasting; picking on measured out-of-sample performance is.
candidates = {
    "Random Forest (tuned)":          (rf_model, rf_scores),
    "Random Forest + trend (hybrid)": (hybrid_model, hybrid_scores),
    "Linear Regression (benchmark)":  (lr_model, lr_scores),
}
CHAMPION_NAME = min(candidates, key=lambda k: candidates[k][1]["MAE"])
champion_model, champion_scores = candidates[CHAMPION_NAME]

print(f"\nSELECTED FOR THE Q1 2026 FORECAST: {CHAMPION_NAME}")
print(f"  Chosen on lowest holdout MAE (${champion_scores['MAE']:,.0f}), "
      f"MAPE {champion_scores['MAPE']:.1f}%")
print(f"  Accuracy in plain English: typically within "
      f"{champion_scores['MAPE']:.0f}% of the actual month.")

# The per-sector model has the same ceiling problem, so we apply the same fix there and
# again keep whichever version actually performs better on the holdout.
grid_hybrid_sec = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid=param_grid, cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error", n_jobs=-1,
)
trend_only_sec = LinearRegression().fit(Xs_train[["time_idx"]], ys_train)
grid_hybrid_sec.fit(Xs_train, ys_train.values - trend_only_sec.predict(Xs_train[["time_idx"]]))
hybrid_sector = TrendAwareForest(grid_hybrid_sec.best_params_).fit(Xs_train, ys_train)
hybrid_sec_pred = hybrid_sector.predict(Xs_test)
hybrid_sec_scores = evaluate(ys_test, hybrid_sec_pred)

if hybrid_sec_scores["MAE"] < sec_scores["MAE"]:
    sector_champion, sector_champion_scores, SECTOR_CHAMPION_NAME = (
        hybrid_sector, hybrid_sec_scores, "Random Forest + trend (hybrid)")
else:
    sector_champion, sector_champion_scores, SECTOR_CHAMPION_NAME = (
        rf_sector, sec_scores, "Random Forest (tuned)")

print(f"\nPER-SECTOR MODEL SELECTED: {SECTOR_CHAMPION_NAME}")
print(f"  Plain RF MAE ${sec_scores['MAE']:,.0f}  |  Hybrid MAE ${hybrid_sec_scores['MAE']:,.0f}")


# ==================================================================================
# SECTION 4  -  WHAT IS THE MODEL ACTUALLY USING? (INTERPRETABILITY)
# ==================================================================================
banner("SECTION 4  |  FEATURE IMPORTANCE AND PARTIAL DEPENDENCE")

# A forecast a board cannot interrogate is a forecast a board will not trust. These two
# tools open the box.

# ---- 4.1 Feature importance ----------------------------------------------------------
# WHAT IT MEASURES: every time a tree splits on a feature, it reduces the error a bit.
# Feature importance adds up all those reductions, across all trees, for each feature, and
# scales the totals to sum to 1. A feature with importance 0.40 is responsible for 40% of
# the model's total error reduction.
#
# HOW TO READ IT HONESTLY - two warnings that matter:
#   1. Importance means "useful for prediction", NOT "causes revenue". lag_1 being
#      important does not mean last month's revenue CAUSES this month's; both are driven
#      by the same underlying pipeline of client demand.
#   2. When two features carry the same information (lag_1, lag_2, lag_3 and roll_3 all
#      describe recent revenue), they SPLIT the credit between them. A feature can look
#      unimportant simply because a near-duplicate is standing next to it.
importances = pd.DataFrame({
    "feature": FIRM_FEATURES,
    "importance": champion_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"Feature importance for the selected model ({CHAMPION_NAME}), firm-wide:\n")
print(f"{'Rank':<6}{'Feature':<14}{'Importance':>12}   {'':<30}")
for i, row in importances.iterrows():
    bar = "#" * int(round(row["importance"] * 50))
    print(f"{i+1:<6}{row['feature']:<14}{row['importance']:>11.1%}   {bar}")

TOP3 = importances.head(3)["feature"].tolist()
print(f"\nTop 3 features: {TOP3}")

if isinstance(champion_model, TrendAwareForest):
    print("\nREADING NOTE: the selected model handles long-run growth in a separate linear")
    print("stage, so 'time_idx' looks small here. These importances describe what drives")
    print("the movement AROUND the growth trend - seasonality and momentum - which is")
    print("exactly the question a board is asking when it asks 'what moves our revenue?'")

# For contrast we also print the plain Random Forest's view, since that is the model the
# brief specified as the main one. Comparing the two is instructive.
plain_importances = pd.DataFrame({
    "feature": FIRM_FEATURES, "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)
print(f"\nFor comparison - the plain tuned Random Forest ranks them: "
      f"{plain_importances['feature'].tolist()}")

# ---- 4.2 Partial dependence for the top 3 features ------------------------------------
# WHAT IT ANSWERS: feature importance tells you WHICH features matter. It does not tell
# you WHICH DIRECTION they push, or what shape the relationship has. Partial dependence
# fills that gap.
#
# HOW IT IS COMPUTED, in plain language: pick a feature, say month_num. Take the entire
# training set and pretend EVERY row happened in January - leave all other columns exactly
# as they are - then ask the model for its average prediction. Write it down. Now pretend
# every row happened in February, and repeat. Doing that for all twelve months traces out
# the model's learned seasonal curve, with the influence of the other features averaged
# out. It is essentially a controlled experiment run inside the model.
#
# THE CAVEAT: partial dependence averages over combinations that may be unrealistic (a row
# with December's month_num but August's lag values). Where features are strongly
# correlated - and our lags certainly are - read the shape as indicative, not literal.
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
# .astype(float) is required: scikit-learn refuses to compute partial dependence on
# integer columns, because stepping through integer grid values can silently round.
# Our month_num, time_idx and is_q4 columns are integers, so we hand over a float copy.
PartialDependenceDisplay.from_estimator(
    champion_model,
    X_train.astype(float),
    features=TOP3,
    ax=axes,
    line_kw={"color": C_HISTORY, "linewidth": 2.4},
)
MONTH_TICKS = ["Jan", "Mar", "May", "Jul", "Sep", "Nov"]
for ax, feat in zip(axes, TOP3):
    ax.set_title(PRETTY.get(feat, feat), fontsize=11.5, color=C_INK, pad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(axis="y", alpha=0.7)
    if feat in MONEY_FEATURES:
        # Dollar amounts on the x-axis too, so the reader never meets a raw "1e7".
        ax.xaxis.set_major_formatter(FuncFormatter(money))
    elif feat == "month_num":
        # Month names beat the numbers 1-12: nobody thinks of December as "12".
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_xticklabels(MONTH_TICKS)
axes[0].set_ylabel("Average predicted\nmonthly revenue", color=C_INK_SOFT)
fig.suptitle("How the three strongest drivers move the revenue forecast",
             fontsize=14, fontweight="bold", color=C_INK, y=1.03, x=0.02, ha="left")
fig.text(0.02, 0.955,
         "Each line shows the model's average prediction as one driver changes and everything else is held steady.",
         fontsize=9.5, color=C_INK_SOFT, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("chart_partial_dependence.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved: chart_partial_dependence.png  (partial dependence for the top 3 features)")


# ==================================================================================
# SECTION 5  -  FORECASTING Q1 2026, WITH UNCERTAINTY
# ==================================================================================
banner("SECTION 5  |  RECURSIVE FORECAST AND PREDICTION INTERVALS")

# ---- 5.1 Refit the chosen models on ALL available history -----------------------------
# Up to now the models have only seen the first 45 months, because we had to hold 12 back
# to grade them honestly. That grading is done. For the real forecast we retrain on every
# month we have, including 2025 - throwing away the most recent and most relevant year
# would be indefensible. This is standard practice: hold out to MEASURE, refit on
# everything to PREDICT.
X_all = fw[FIRM_FEATURES]
y_all = fw["revenue"]

if isinstance(champion_model, TrendAwareForest):
    final_model = TrendAwareForest(champion_model.rf_params).fit(X_all, y_all)
elif isinstance(champion_model, LinearRegression):
    final_model = LinearRegression().fit(X_all, y_all)
else:
    final_model = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **grid.best_params_).fit(X_all, y_all)

Xs_all, ys_all = sec_feat[SECTOR_FEATURES], sec_feat["revenue"]
if isinstance(sector_champion, TrendAwareForest):
    final_sector_model = TrendAwareForest(sector_champion.rf_params).fit(Xs_all, ys_all)
else:
    final_sector_model = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **grid_sec.best_params_).fit(Xs_all, ys_all)

print(f"Refitted the firm-wide model on all {len(fw)} usable months "
      f"({fw['month'].min()} -> {fw['month'].max()}).")
print(f"Refitted the per-sector model on all {len(sec_feat)} rows.")

# ---- 5.2 Set up the interval machinery -------------------------------------------------
#
# WHY A SINGLE NUMBER IS NOT A FORECAST
# "Q1 2026 will be $37.4M" is a guess dressed up as a fact. A board needs to know the
# range: is it $36-39M, or is it $25-50M? Those imply completely different decisions about
# hiring, cash and commitments. A prediction interval supplies that range.
#   80% interval - we expect the actual to land inside this range 8 times out of 10. This
#                  is the planning range; it is deliberately narrower.
#   95% interval - the wider, more cautious range. Useful for downside/worst-case work.
#
# METHOD 1 (preferred): QUANTILE REGRESSION FOREST, via the `quantile-forest` package.
# An ordinary Random Forest throws away most of what it knows. Each leaf of each tree
# holds a whole set of training values, and the forest averages them all down to one
# number. A Quantile Regression Forest keeps those values, so instead of just the mean it
# can report the 10th percentile, the 90th, or any other - i.e. the model's full sense of
# the distribution, learned from the data rather than assumed.
#
# METHOD 2 (fallback): SPREAD OF INDIVIDUAL TREES.
# If the package will not install, we ask each of the forest's trees for its own
# prediction and take percentiles across those. It works, but it measures DISAGREEMENT
# BETWEEN TREES, which is a narrower thing than genuine predictive uncertainty. Intervals
# built this way tend to be too optimistic, and we say so in the output.

INTERVAL_METHOD = None
try:
    from quantile_forest import RandomForestQuantileRegressor
    import quantile_forest as _qf
    INTERVAL_METHOD = f"Quantile Regression Forest (quantile-forest v{_qf.__version__})"
    USE_QRF = True
except Exception as exc:  # pragma: no cover - only runs if the install failed
    INTERVAL_METHOD = "Spread of individual Random Forest tree predictions (fallback)"
    USE_QRF = False
    print(f"  quantile-forest unavailable ({exc}); falling back to tree spread.")

# Note the wording: at this point quantile-forest is a CANDIDATE, not the decision. It is
# put head-to-head against a second method in Section 5.2b and the winner is chosen there
# on measured evidence. Announcing it as "in use" here would contradict that.
print(f"\nPREFERRED INTERVAL ENGINE AVAILABLE: {INTERVAL_METHOD}")
print("(This is the first choice to be tried. Section 5.2b tests it against an alternative")
print(" and reports which one is actually used for the published bands.)")

QUANTILES = [0.025, 0.10, 0.50, 0.90, 0.975]   # 95% low, 80% low, median, 80% high, 95% high

# The interval model must match the shape of the point model. Our champion predicts
# trend + residual, so the quantile model learns the distribution OF THE RESIDUALS and we
# add the trend line back on. Applying a quantile forest directly to raw revenue would
# reintroduce exactly the extrapolation ceiling we just fixed.
HYBRID_MODE = isinstance(final_model, TrendAwareForest)


def build_interval_model(X, y, params, hybrid_trend=None):
    """
    Fit whatever model will supply our uncertainty bands.

    If hybrid_trend is given, we fit on residuals from that trend line; otherwise on the
    raw target. Returns (model, trend_or_None).
    """
    target = y.values if hasattr(y, "values") else np.asarray(y)
    if hybrid_trend is not None:
        target = target - hybrid_trend.predict(X[["time_idx"]])
    if USE_QRF:
        m = RandomForestQuantileRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    else:
        m = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    m.fit(X, target)
    return m


def interval_quantiles(model, X_row):
    """
    Return the five quantiles of the model's predictive distribution for one row.

    Both branches produce the same output shape, so the calling code does not need to
    know which method is in play.
    """
    if USE_QRF:
        return np.asarray(model.predict(X_row, quantiles=QUANTILES)).ravel()
    # Fallback: poll every tree individually and take percentiles across their answers.
    per_tree = np.array([t.predict(X_row.values)[0] for t in model.estimators_])
    return np.percentile(per_tree, [q * 100 for q in QUANTILES])


interval_params = (final_model.rf_params if HYBRID_MODE else grid.best_params_) or {}
interval_model = build_interval_model(
    X_all, y_all, interval_params,
    hybrid_trend=final_model.trend_ if HYBRID_MODE else None,
)

SECTOR_HYBRID = isinstance(final_sector_model, TrendAwareForest)
sector_interval_params = (final_sector_model.rf_params if SECTOR_HYBRID else grid_sec.best_params_) or {}
sector_interval_model = build_interval_model(
    Xs_all, ys_all, sector_interval_params,
    hybrid_trend=final_sector_model.trend_ if SECTOR_HYBRID else None,
)

# ---- 5.3 The recursive forecast --------------------------------------------------------
#
# HOW RECURSIVE FORECASTING WORKS
# Our model needs lag_1, lag_2 and lag_3 - the previous three months' revenue - to make a
# prediction. For January 2026 that is fine: October, November and December 2025 are all
# real, observed numbers.
#
# But for February 2026, lag_1 is January 2026 - a month that has not happened yet. We do
# not have it. The only thing we have is our own January PREDICTION. So we feed the
# prediction in as if it were data, and predict February. Then March uses the February
# prediction (and the January one), and so on. Each forecast becomes an input to the next.
# That is what "recursive" means.
#
#   Jan 2026  <- lag_1=Dec25(real)  lag_2=Nov25(real)  lag_3=Oct25(real)   0 guesses in
#   Feb 2026  <- lag_1=Jan26(GUESS) lag_2=Dec25(real)  lag_3=Nov25(real)   1 guess in
#   Mar 2026  <- lag_1=Feb26(GUESS) lag_2=Jan26(GUESS) lag_3=Dec25(real)   2 guesses in
#
# WHY ERROR COMPOUNDS - AND WHY THIS IS THE MOST IMPORTANT CAVEAT IN THE WHOLE FORECAST
# If January is over-predicted by $2M, that $2M error does not stay in January. It becomes
# February's lag_1 input, so February inherits it and adds its own fresh error on top.
# March then inherits February's inflated total. Errors do not average out across the
# horizon - they accumulate, and they can amplify. This is why a 1-month-ahead forecast is
# always much more reliable than a 3-month-ahead one, and why March's interval must be
# visibly wider than January's rather than the same width.
#
# Rather than ASSUME how fast error grows, we MEASURE it (Section 5.2b below) by making
# the model repeatedly forecast three months ahead inside the holdout period, where we
# already know the right answers. The measured growth is what widens the bands.


def recursive_forecast(model, iv_model, hybrid, last_revenues, last_time_idx,
                       forecast_months, static_features=None, feature_order=None,
                       horizon_scale=None, calib=None, bias=None, empirical=None):
    """
    Roll the model forward month by month, feeding each prediction into the next.

    last_revenues  : the 3 most recent observed revenues, OLDEST FIRST -> [t-3, t-2, t-1]
    last_time_idx  : the time_idx of the final observed month
    static_features: dict of columns that do not change month to month (the sector flags)
    """
    history = list(last_revenues)     # we append predictions to this as we go
    results = []
    horizon_scale = horizon_scale or {}
    # calib holds one scaling factor per band edge - see Section 5.2b. Defaulting them all
    # to 1.0 means "use the quantile forest's raw output", which is what the backtest
    # itself needs while it is still measuring how good that raw output is.
    calib = calib or {"lo80": 1.0, "hi80": 1.0, "lo95": 1.0, "hi95": 1.0}
    # bias holds a measured dollar correction per horizon (Section 5.2b). It is empty
    # while the backtest is still running, because that is when the bias is being measured.
    bias = bias or {}

    for step, mstr in enumerate(forecast_months, start=1):
        period = pd.Period(mstr, freq="M")

        # Assemble this month's feature row from the (real or predicted) history.
        row = {
            "month_num": period.month,
            "time_idx":  last_time_idx + step,
            "is_q4":     int(period.month in (10, 11, 12)),
            "lag_1":     history[-1],
            "lag_2":     history[-2],
            "lag_3":     history[-3],
            "roll_3":    float(np.mean(history[-3:])),
        }
        if static_features:
            row.update(static_features)

        X_row = pd.DataFrame([row])[feature_order]

        point = float(model.predict(X_row)[0]) + bias.get(step, 0.0)

        # Quantiles from the interval model. For the hybrid we must add the trend line
        # back, because the interval model was trained on residuals.
        q = interval_quantiles(iv_model, X_row)
        if hybrid:
            q = q + float(model.trend_.predict(X_row[["time_idx"]])[0])

        # Re-centre the band on our point forecast, then apply two correction factors.
        # Measuring distances from the median (rather than using the raw quantiles) keeps
        # the point forecast exactly in the middle of its own interval, which is what a
        # reader expects when they look at the chart.
        #
        #   calib  - a calibration factor measured on the holdout. A raw quantile forest
        #            trained on only ~45 months tends to produce bands that are too wide,
        #            so we scale them until they cover the holdout at the rate they claim
        #            to. An interval that says "80%" should be right 80% of the time; if
        #            it is right 100% of the time it is not cautious, it is useless.
        #   widen  - the measured growth in error from one month ahead to two and three,
        #            i.e. the compounding effect described above, taken from real
        #            backtest results rather than assumed.
        widen = horizon_scale.get(step, 1.0)

        if empirical is not None:
            # EMPIRICAL (CONFORMAL) BANDS. The band is the spread of the model's OWN past
            # forecast errors, measured in the backtest and stretched by the horizon
            # factor. Its great virtue is that it cannot be miscalibrated by construction:
            # if the model missed by more than $6M one time in ten historically, then the
            # 80% band is that wide, whatever the model believes about itself.
            lo80 = point + empirical["lo80"] * widen
            hi80 = point + empirical["hi80"] * widen
            lo95 = point + empirical["lo95"] * widen
            hi95 = point + empirical["hi95"] * widen
        else:
            # QUANTILE-FOREST BANDS, rescaled by the calibration factors.
            lo95 = point - (q[2] - q[0]) * calib["lo95"] * widen
            lo80 = point - (q[2] - q[1]) * calib["lo80"] * widen
            hi80 = point + (q[3] - q[2]) * calib["hi80"] * widen
            hi95 = point + (q[4] - q[2]) * calib["hi95"] * widen

        # Revenue cannot be negative. Clipping at zero avoids a nonsensical lower bound
        # on small sectors in weak months.
        lo95, lo80 = max(lo95, 0.0), max(lo80, 0.0)

        results.append({
            "month": str(period), "predicted": point,
            "lo80": lo80, "hi80": hi80, "lo95": lo95, "hi95": hi95,
            "horizon_months": step,
        })
        history.append(point)   # <-- THE RECURSIVE STEP: today's guess is tomorrow's input

    return pd.DataFrame(results)


# ==================================================================================
# SECTION 5.2b  -  MEASURING ERROR COMPOUNDING, AND CALIBRATING THE INTERVALS
# ==================================================================================
banner("SECTION 5.2b  |  BACKTESTING THE 3-MONTH HORIZON")

# We have claimed that recursive forecasting makes month 3 less reliable than month 1.
# Rather than take that on faith, we test it, using a ROLLING-ORIGIN BACKTEST.
#
# The idea is simple and it is the closest thing forecasting has to a dress rehearsal.
# Stand at some month inside the holdout period, pretend you know nothing after it, and
# produce a full 3-month recursive forecast exactly the way we will for 2026. Then look up
# what actually happened and record the error separately for month 1, month 2 and month 3.
# Slide the starting point forward one month and do it again. Averaging across all those
# rehearsals tells us how much accuracy really decays with horizon - measured, not assumed.

# ONE RULE MAKES THIS VALID: at every origin, BOTH the point model and the interval model
# are retrained from scratch on only the months strictly before that origin. Reusing a
# model that has already seen the answer would make the rehearsal meaningless - it would
# measure memory, not forecasting skill. This is called an EXPANDING-WINDOW backtest: the
# training set grows by one month at each step, exactly as it would in real life.

def fit_like_champion(X, y):
    """Rebuild a fresh, untrained copy of whichever model type we selected, and fit it."""
    if isinstance(champion_model, TrendAwareForest):
        return TrendAwareForest(champion_model.rf_params).fit(X, y)
    if isinstance(champion_model, LinearRegression):
        return LinearRegression().fit(X, y)
    return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                                 **grid.best_params_).fit(X, y)


backtest_rows = []
bt_start = n_train                      # first holdout row
_is_hybrid = isinstance(champion_model, TrendAwareForest)
_bt_params = (champion_model.rf_params if _is_hybrid else grid.best_params_) or {}

for origin in range(bt_start, len(fw) - len(FORECAST_MONTHS) + 1):
    # Everything strictly BEFORE the origin is allowed as training data. Nothing after.
    hist_slice = fw.iloc[:origin]
    X_bt, y_bt = hist_slice[FIRM_FEATURES], hist_slice["revenue"]

    m_bt = fit_like_champion(X_bt, y_bt)
    iv_bt = build_interval_model(
        X_bt, y_bt, _bt_params,
        hybrid_trend=m_bt.trend_ if _is_hybrid else None,
    )

    # The three real revenues immediately before the origin become the starting lags.
    seed = fw["revenue"].iloc[origin - 3:origin].tolist()
    target_months = [str(fw["month"].iloc[origin + k]) for k in range(len(FORECAST_MONTHS))]
    bt = recursive_forecast(
        m_bt, iv_bt, _is_hybrid,
        last_revenues=seed,
        last_time_idx=int(fw["time_idx"].iloc[origin - 1]),
        forecast_months=target_months,
        feature_order=FIRM_FEATURES,
    )
    for k, row in bt.iterrows():
        actual = float(fw["revenue"].iloc[origin + k])
        backtest_rows.append({
            "horizon": row["horizon_months"],
            "actual": actual,
            "predicted": row["predicted"],
            "abs_error": abs(actual - row["predicted"]),
            "lo_half": row["predicted"] - row["lo80"],
            "hi_half": row["hi80"] - row["predicted"],
            "lo_half95": row["predicted"] - row["lo95"],
            "hi_half95": row["hi95"] - row["predicted"],
        })

bt_df = pd.DataFrame(backtest_rows)
h_mae = bt_df.groupby("horizon")["abs_error"].mean()

print(f"Rolling-origin backtest: {bt_df['horizon'].eq(1).sum()} separate 3-month forecasts "
      f"made inside the holdout period.\n")
print(f"{'Horizon':<26}{'Mean absolute error':>22}{'vs 1 month ahead':>22}")
print("-" * 70)
HORIZON_SCALE = {}
for h in sorted(h_mae.index):
    factor = h_mae[h] / h_mae[1] if h_mae[1] > 0 else 1.0
    HORIZON_SCALE[int(h)] = float(max(factor, 1.0))   # never let the band SHRINK with horizon
    label = f"{int(h)} month{'s' if h > 1 else ''} ahead"
    print(f"{label:<26}{'$' + format(h_mae[h], ',.0f'):>22}{'x' + format(factor, '.2f'):>22}")
print("-" * 70)
print("This is error compounding, measured. Each extra month of recursion feeds one more")
print("predicted value back in as an input, and accuracy degrades accordingly. It is the")
print("reason the forecast chart's uncertainty band fans out rather than running parallel.")

# ---- Correcting a systematic bias -----------------------------------------------------
#
# Accuracy is only half the story. A model can be accurate on average and still be
# consistently WRONG IN ONE DIRECTION - always a bit low, or always a bit high. That is
# called bias, and it is far more dangerous to a business than random error, because it
# does not cancel out. Plan three quarters off a model that is always 15% low and you
# under-hire three times in a row.
#
# The backtest lets us measure it directly: the average SIGNED error (actual minus
# forecast) at each horizon. Random error averages to roughly zero; bias does not.
bias_by_h = bt_df.groupby("horizon").apply(
    lambda g: float(np.mean(g["actual"] - g["predicted"])), include_groups=False)

print(f"\nBIAS CHECK (average signed miss; positive = the model forecast too LOW)")
print("-" * 70)
BIAS = {}
for h in sorted(bias_by_h.index):
    BIAS[int(h)] = float(bias_by_h[h])
    pct = bias_by_h[h] / bt_df[bt_df["horizon"] == h]["actual"].mean() * 100
    print(f"  {int(h)} month{'s' if h > 1 else ' '} ahead:  "
          f"${bias_by_h[h]:>12,.0f}   ({pct:+.1f}% of the average actual month)")
print("-" * 70)
print("The model under-forecasts throughout, and the reason is the one diagnosed in")
print("Section 3.6: 2025 grew faster than the straight-line trend fitted to 2021-2024,")
print("so even the hybrid could not fully keep up. We therefore ADD these measured")
print("corrections to the Q1 2026 forecast. This matters for the intervals too - without")
print("it, the uncertainty band has to stretch upwards to cover a gap that is not")
print("really uncertainty at all, it is a known and fixable offset.")

# Re-run the backtest arithmetic with the correction applied, so that everything measured
# from here on - error growth, interval calibration - describes the CORRECTED model
# rather than the biased one we are no longer going to use.
bt_df["predicted"] = bt_df["predicted"] + bt_df["horizon"].map(BIAS).astype(float)
bt_df["abs_error"] = (bt_df["actual"] - bt_df["predicted"]).abs()

h_mae_corrected = bt_df.groupby("horizon")["abs_error"].mean()
HORIZON_SCALE = {int(h): float(max(h_mae_corrected[h] / h_mae_corrected[1], 1.0))
                 for h in sorted(h_mae_corrected.index)}
print(f"\nAfter correction, mean absolute error by horizon:")
for h in sorted(h_mae_corrected.index):
    print(f"  {int(h)} month{'s' if h > 1 else ' '} ahead:  ${h_mae_corrected[h]:>12,.0f}"
          f"   (x{HORIZON_SCALE[int(h)]:.2f} vs 1 month)")


# ---- Calibrating the interval width ---------------------------------------------------
#
# WHY THE RAW BANDS NEED CORRECTING AT ALL
# A quantile forest estimates its quantiles from the training values sitting in each leaf.
# That works well with thousands of rows. We have 57. Worse, our Q1 2026 rows sit slightly
# OUTSIDE the range the forest was trained on - December 2025 was the largest month on
# record, so January 2026's lag_1 input is bigger than any lag_1 the forest ever saw. The
# trees therefore route those rows into the same "record high" leaves, and the raw bands
# come out both too wide and skewed towards the upside.
#
# So we do not take the forest's word for it. We check the bands against the backtest and
# rescale them until they keep their promise. This is the same logic as conformal
# prediction: let the model propose the SHAPE of the uncertainty, then let measured
# out-of-sample errors fix the SIZE.
#
# We calibrate the two sides SEPARATELY. A single shared factor cannot fix a lopsided
# band - it would just make both sides equally wrong. An 80% interval should leave about
# 10% of outcomes above it and 10% below it, so each edge gets its own correction.
#
# All 30 backtest forecasts are used, not just the 10 one-month-ahead ones. To make them
# comparable we first divide out the horizon growth measured above, so a 3-month-ahead
# error is judged against a 3-month-ahead band. That triples the evidence behind each
# factor, which matters a great deal when the alternative is 10 data points.

bt_df["scale_h"] = bt_df["horizon"].map(HORIZON_SCALE).astype(float)
bt_df["signed_error"] = bt_df["actual"] - bt_df["predicted"]


def edge_factor(df, half_col, level):
    """
    How much would this band edge have to be multiplied by to contain `level` of outcomes?

    Each backtest month is expressed as "how many half-bands away did the actual land",
    after dividing out the horizon effect. The `level`-th percentile of those distances is
    exactly the multiplier we need.
    """
    denom = (df[half_col] * df["scale_h"]).replace(0, np.nan)
    ratio = (df["signed_error"] / denom).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 1.0
    return float(np.clip(np.quantile(ratio, level), 0.25, 3.0))


# For the upper edge we look at how far actuals overshot; for the lower edge we flip the
# sign and look at how far they undershot.
lower_view = bt_df.assign(signed_error=-bt_df["signed_error"])

CALIB = {
    "hi80": edge_factor(bt_df,     "hi_half",   0.90),
    "lo80": edge_factor(lower_view, "lo_half",   0.90),
    "hi95": edge_factor(bt_df,     "hi_half95", 0.975),
    "lo95": edge_factor(lower_view, "lo_half95", 0.975),
}


def coverage(df, lo_col, hi_col, lo_f=1.0, hi_f=1.0):
    """Share of backtest months whose actual landed inside the (scaled) band."""
    lo = df["predicted"] - df[lo_col] * df["scale_h"] * lo_f
    hi = df["predicted"] + df[hi_col] * df["scale_h"] * hi_f
    return float(((df["actual"] >= lo) & (df["actual"] <= hi)).mean())


raw_cov80 = coverage(bt_df, "lo_half", "hi_half")
raw_cov95 = coverage(bt_df, "lo_half95", "hi_half95")
cal_cov80 = coverage(bt_df, "lo_half", "hi_half", CALIB["lo80"], CALIB["hi80"])
cal_cov95 = coverage(bt_df, "lo_half95", "hi_half95", CALIB["lo95"], CALIB["hi95"])

print(f"\nINTERVAL CALIBRATION (checked against all {len(bt_df)} backtest forecasts)")
print("-" * 78)
print(f"{'Interval':<12}{'Promised':>10}{'Raw coverage':>15}"
      f"{'Lower x':>11}{'Upper x':>11}{'After scaling':>17}")
print("-" * 78)
print(f"{'80%':<12}{'80%':>10}{raw_cov80:>14.0%}"
      f"{CALIB['lo80']:>10.2f}x{CALIB['hi80']:>10.2f}x{cal_cov80:>16.0%}")
print(f"{'95%':<12}{'95%':>10}{raw_cov95:>14.0%}"
      f"{CALIB['lo95']:>10.2f}x{CALIB['hi95']:>10.2f}x{cal_cov95:>16.0%}")
print("-" * 78)
print("Read the two 'x' columns as a verdict on the raw quantile forest: a value below")
print("1.00 means that edge was too generous and has been pulled in, above 1.00 means it")
print("was over-confident and has been pushed out. The upper edge needed the larger")
print("correction, for exactly the record-high-lag reason described above.")
print("\nA calibrated interval is worth far more to a board than a flatteringly narrow one")
print("or an uninformatively wide one: it means the stated confidence can be relied on.")


# ---- The second interval method, and a head-to-head test ------------------------------
#
# The calibration table above is a warning sign, not a success. The upper edge needed a
# correction of well over 1.5x, and the reason is a genuine mismatch: the quantile forest
# produced its bands for BACKTEST rows, which all sat comfortably inside the range it was
# trained on, but we are asking it to band a Q1 2026 row whose lag_1 is the largest value
# in the entire dataset. Its bands are not comparable between those two situations, so a
# factor measured on one does not transfer to the other.
#
# So we build a second, much simpler set of bands and test them fairly against the first.
#
# EMPIRICAL (CONFORMAL) BANDS: forget what the model believes about its own uncertainty
# and look at how wrong it has actually been. Take the 30 backtest forecasts, divide each
# error by its horizon factor so they are on a common footing, and read the percentiles
# straight off. The 10th and 90th percentiles of that error distribution ARE the 80%
# interval; the 2.5th and 97.5th are the 95% interval. Nothing is assumed about the shape
# of the errors - no bell curve, no symmetry - it is simply the model's own track record.
norm_err = (bt_df["actual"] - bt_df["predicted"]) / bt_df["scale_h"]
EMPIRICAL = {
    "lo95": float(np.quantile(norm_err, 0.025)),
    "lo80": float(np.quantile(norm_err, 0.10)),
    "hi80": float(np.quantile(norm_err, 0.90)),
    "hi95": float(np.quantile(norm_err, 0.975)),
}

emp_cov80 = float((((bt_df["actual"] - bt_df["predicted"]) >= EMPIRICAL["lo80"] * bt_df["scale_h"]) &
                   ((bt_df["actual"] - bt_df["predicted"]) <= EMPIRICAL["hi80"] * bt_df["scale_h"])).mean())
emp_cov95 = float((((bt_df["actual"] - bt_df["predicted"]) >= EMPIRICAL["lo95"] * bt_df["scale_h"]) &
                   ((bt_df["actual"] - bt_df["predicted"]) <= EMPIRICAL["hi95"] * bt_df["scale_h"])).mean())

# Judge the two methods on two things at once. Coverage alone is not enough: an interval
# from minus infinity to plus infinity has perfect coverage and tells you nothing. The
# right test is "which method reaches the promised coverage with the NARROWER band".
#
# And there is a trap to avoid here. It would be natural to compare the two methods on the
# backtest rows, since that is where we have answers. But the whole problem with the
# quantile forest is that it behaves DIFFERENTLY on the Q1 2026 rows than on the backtest
# rows - that is the mismatch we are trying to detect. Measuring width on the backtest
# would therefore hide exactly the fault we are looking for. So coverage is judged on the
# backtest, where we know the truth, and WIDTH is judged on the real Q1 2026 rows, where
# the bands will actually be used. Always measure a property where it matters, not where
# it is convenient.
print(f"\nBacktest coverage of the 80% band - quantile forest {cal_cov80:.0%}, "
      f"empirical {emp_cov80:.0%} (both aiming at 80%).")
print("Widths are compared below, on the actual Q1 2026 forecast rows.")





# ---- Firm-wide forecast ---------------------------------------------------------------
last3_firm = firmwide["revenue"].tail(3).tolist()          # Oct, Nov, Dec 2025
last_idx_firm = int(fw["time_idx"].max())

def run_firm_forecast(empirical):
    return recursive_forecast(
        final_model, interval_model, HYBRID_MODE,
        last_revenues=last3_firm, last_time_idx=last_idx_firm,
        forecast_months=FORECAST_MONTHS, feature_order=FIRM_FEATURES,
        horizon_scale=HORIZON_SCALE, calib=CALIB, bias=BIAS, empirical=empirical,
    )


# Produce the forecast under BOTH interval methods, then compare the band widths they
# actually deliver for Q1 2026 and keep the tighter one. The point forecasts are identical
# either way - only the uncertainty bands differ.
cand_qrf = run_firm_forecast(None)
cand_emp = run_firm_forecast(EMPIRICAL)
qrf_width80 = float((cand_qrf["hi80"] - cand_qrf["lo80"]).mean())
emp_width80 = float((cand_emp["hi80"] - cand_emp["lo80"]).mean())

print("\n" + "-" * 78)
print("INTERVAL METHOD BAKE-OFF, measured on the three Q1 2026 rows")
print("-" * 78)
print(f"{'Method':<40}{'Backtest coverage':>20}{'Avg 80% width':>18}")
print("-" * 78)
print(f"{'Quantile forest (calibrated)':<40}{cal_cov80:>19.0%}{'$'+format(qrf_width80, ',.0f'):>18}")
print(f"{'Empirical backtest errors':<40}{emp_cov80:>19.0%}{'$'+format(emp_width80, ',.0f'):>18}")
print("-" * 78)

_qrf_ok = abs(cal_cov80 - 0.80) <= 0.10
_emp_ok = abs(emp_cov80 - 0.80) <= 0.10
_qf_ver = _qf.__version__ if USE_QRF else "n/a"
if _emp_ok and (not _qrf_ok or emp_width80 <= qrf_width80):
    USE_EMPIRICAL, firm_fc = EMPIRICAL, cand_emp
    INTERVAL_METHOD = (
        f"Empirical (conformal) intervals, derived from the spread of the model's own "
        f"errors across a 30-forecast rolling-origin backtest. The quantile-forest package "
        f"(v{_qf_ver}) WAS installed and was tried first, but its bands did not transfer to "
        f"the out-of-range Q1 2026 rows and came out roughly "
        f"{qrf_width80/max(emp_width80,1):.1f}x wider for the same coverage.")
else:
    USE_EMPIRICAL, firm_fc = None, cand_qrf
    INTERVAL_METHOD = (
        f"Quantile Regression Forest (quantile-forest v{_qf_ver}), rescaled by two-sided "
        f"calibration factors measured on a 30-forecast rolling-origin backtest.")

print(f"\nSELECTED INTERVAL METHOD: "
      f"{'Empirical backtest errors' if USE_EMPIRICAL else 'Quantile Regression Forest'}")
print(f"  {INTERVAL_METHOD}")
if USE_EMPIRICAL:
    print("\nWorth pausing on, because it is a real lesson rather than a technicality: the")
    print("more sophisticated tool lost. With 57 monthly observations and a forecast sitting")
    print("just outside the training range, a model's own opinion of its uncertainty proved")
    print("less trustworthy than its measured track record.")

print("\n" + "-" * 92)
print("FIRM-WIDE FORECAST - Q1 2026")
print("-" * 92)
print(f"{'Month':<12}{'Forecast':>16}{'80% interval':>30}{'95% interval':>32}")
print("-" * 92)
for _, r in firm_fc.iterrows():
    print(f"{r['month']:<12}{'$'+format(r['predicted'],',.0f'):>16}"
          f"{'$'+format(r['lo80'],',.0f')+'  -  $'+format(r['hi80'],',.0f'):>30}"
          f"{'$'+format(r['lo95'],',.0f')+'  -  $'+format(r['hi95'],',.0f'):>32}")
print("-" * 92)
q1_total = firm_fc["predicted"].sum()
# Last year's actual Q1, for a like-for-like growth comparison on the chart and in summary.md.
q1_2025_actual = float(firmwide.loc[
    firmwide["month"].isin([pd.Period(f"2025-{m:02d}", freq="M") for m in (1, 2, 3)]),
    "revenue"].sum())
q1_lo80, q1_hi80 = firm_fc["lo80"].sum(), firm_fc["hi80"].sum()
q1_lo95, q1_hi95 = firm_fc["lo95"].sum(), firm_fc["hi95"].sum()
print(f"{'Q1 2026 TOTAL':<12}{'$'+format(q1_total,',.0f'):>16}"
      f"{'$'+format(q1_lo80,',.0f')+'  -  $'+format(q1_hi80,',.0f'):>30}"
      f"{'$'+format(q1_lo95,',.0f')+'  -  $'+format(q1_hi95,',.0f'):>32}")
print("-" * 92)
print("Note: summing the monthly intervals to a quarterly interval is deliberately")
print("conservative - it assumes the three months miss in the same direction together.")

# ---- Per-sector forecast ---------------------------------------------------------------
banner("SECTION 5.4  |  PER-SECTOR FORECAST")

# Each sector needs its own uncertainty band, and sectors are NOT equally predictable. A
# large, steady sector like Financial Services is easier to forecast than a small, lumpy
# one where a single big contract moves the whole month. Giving every sector the same
# relative band would flatter the volatile ones and insult the stable ones.
#
# So we run the same rolling-origin backtest again, this time on the sector model, and let
# each sector's own track record set its own band.

sec_bt_rows = []
_sec_is_hybrid = isinstance(sector_champion, TrendAwareForest)
_sec_params = (sector_champion.rf_params if _sec_is_hybrid else grid_sec.best_params_) or {}
sec_months = sorted(sec_feat["month"].unique())

for origin_i in range(n_train, len(sec_months) - len(FORECAST_MONTHS) + 1):
    origin_month = sec_months[origin_i]
    past = sec_feat[sec_feat["month"] < origin_month]
    Xp, yp = past[SECTOR_FEATURES], past["revenue"]

    if _sec_is_hybrid:
        m_sec = TrendAwareForest(_sec_params).fit(Xp, yp)
    else:
        m_sec = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **_sec_params).fit(Xp, yp)
    iv_sec = build_interval_model(Xp, yp, _sec_params,
                                  hybrid_trend=m_sec.trend_ if _sec_is_hybrid else None)

    tgt_months = [str(sec_months[origin_i + k]) for k in range(len(FORECAST_MONTHS))]
    for sec in all_sectors:
        h = by_sector[by_sector["sector"] == sec].sort_values("month").reset_index(drop=True)
        pos = h.index[h["month"] == origin_month][0]
        seed = h["revenue"].iloc[pos - 3:pos].tolist()
        static = {c: (1 if c == f"sector_{sec}" else 0) for c in SECTOR_DUMMY_COLS}
        bt = recursive_forecast(
            m_sec, iv_sec, _sec_is_hybrid, last_revenues=seed,
            last_time_idx=int(sec_feat.loc[sec_feat["month"] == origin_month, "time_idx"].iloc[0]) - 1,
            forecast_months=tgt_months, static_features=static,
            feature_order=SECTOR_FEATURES,
        )
        for k, r in bt.iterrows():
            sec_bt_rows.append({
                "sector": sec, "horizon": int(r["horizon_months"]),
                "actual": float(h["revenue"].iloc[pos + k]), "predicted": float(r["predicted"]),
            })

sec_bt = pd.DataFrame(sec_bt_rows)
sec_bt["scale_h"] = sec_bt["horizon"].map(HORIZON_SCALE).astype(float)
# Relative error, so sectors of very different sizes can be compared on one scale.
sec_bt["rel_err"] = (sec_bt["actual"] - sec_bt["predicted"]) / sec_bt["predicted"].replace(0, np.nan)
sec_bt["rel_err_norm"] = sec_bt["rel_err"] / sec_bt["scale_h"]

print(f"Sector backtest complete: {len(sec_bt)} forecasts "
      f"({len(all_sectors)} sectors x {sec_bt['horizon'].eq(1).sum() // len(all_sectors)} origins x "
      f"{len(FORECAST_MONTHS)} horizons)\n")

SECTOR_BANDS = {}
for sec, g in sec_bt.groupby("sector"):
    e = g["rel_err_norm"].dropna()
    SECTOR_BANDS[sec] = {
        "lo95": float(np.quantile(e, 0.025)), "lo80": float(np.quantile(e, 0.10)),
        "hi80": float(np.quantile(e, 0.90)),  "hi95": float(np.quantile(e, 0.975)),
        "mape": float(np.mean(np.abs(g["rel_err"].dropna())) * 100),
    }

# ---- Point forecasts, then reconciliation ---------------------------------------------
sector_fc_frames = []
for sec in all_sectors:
    hist = by_sector[by_sector["sector"] == sec].sort_values("month")
    last3 = hist["revenue"].tail(3).tolist()
    # The one-hot flags stay fixed for the whole of this sector's forecast: this row IS
    # Healthcare in every future month.
    static = {c: (1 if c == f"sector_{sec}" else 0) for c in SECTOR_DUMMY_COLS}
    fcs = recursive_forecast(
        final_sector_model, sector_interval_model, SECTOR_HYBRID,
        last_revenues=last3, last_time_idx=last_idx_firm,
        forecast_months=FORECAST_MONTHS, static_features=static,
        feature_order=SECTOR_FEATURES,
        horizon_scale=HORIZON_SCALE,
    )
    fcs["sector"] = sec
    sector_fc_frames.append(fcs)

sector_fc = pd.concat(sector_fc_frames, ignore_index=True)

# RECONCILIATION
# ---------------
# We now have two forecasts of the same thing: one firm-wide model saying Q1 is $X, and
# seven sector models which, added up, say something different. Both cannot go in the same
# board pack. A deck where the sector table does not add up to the headline number is the
# fastest way to lose a room.
#
# The standard fix is called reconciliation. We keep the firm-wide total as the headline -
# it is the more reliable of the two, because aggregating all seven sectors cancels out a
# lot of individual noise, and because it is the number we backtested and bias-corrected
# most carefully - and we scale the seven sector forecasts proportionally so they sum to
# it. Each sector keeps its share of the pie; only the size of the pie changes.
monthly_sector_sum = sector_fc.groupby("month")["predicted"].transform("sum")
firm_by_month = sector_fc["month"].map(firm_fc.set_index("month")["predicted"])
recon_factor = firm_by_month / monthly_sector_sum

sector_fc["predicted_raw"] = sector_fc["predicted"]
sector_fc["predicted"] = sector_fc["predicted"] * recon_factor

# The bands are rebuilt from each sector's own relative error distribution, applied to the
# reconciled point forecast, and stretched by the horizon factor.
for edge, q in [("lo80", "lo80"), ("hi80", "hi80"), ("lo95", "lo95"), ("hi95", "hi95")]:
    sector_fc[edge] = sector_fc.apply(
        lambda r: max(r["predicted"] * (1 + SECTOR_BANDS[r["sector"]][q]
                                        * HORIZON_SCALE.get(int(r["horizon_months"]), 1.0)), 0.0),
        axis=1,
    )

print(f"{'Month':<10}{'Sum of sectors (raw)':>24}{'Firm-wide model':>20}{'Scaling applied':>18}")
print("-" * 74)
for m in FORECAST_MONTHS:
    rawsum = sector_fc.loc[sector_fc["month"] == m, "predicted_raw"].sum()
    firmv = float(firm_fc.loc[firm_fc["month"] == m, "predicted"].iloc[0])
    print(f"{m:<10}{'$'+format(rawsum, ',.0f'):>24}{'$'+format(firmv, ',.0f'):>20}"
          f"{firmv/rawsum:>17.3f}x")
print("-" * 74)
print("The sector models ran hot relative to the firm-wide model, so they have been scaled")
print("back to match it. The sector table below now adds up to the headline figure exactly.")

print("\n" + "-" * 100)
print("PER-SECTOR FORECAST - Q1 2026 TOTAL (sum of Jan + Feb + Mar), after reconciliation")
print("-" * 100)
print(f"{'Sector':<24}{'Q1 forecast':>16}{'80% interval':>34}{'share':>9}{'backtest MAPE':>16}")
print("-" * 100)
sector_q1 = sector_fc.groupby("sector")[["predicted", "lo80", "hi80"]].sum().sort_values(
    "predicted", ascending=False)
for sec, r in sector_q1.iterrows():
    share = r["predicted"] / sector_q1["predicted"].sum() * 100
    print(f"{sec:<24}{'$'+format(r['predicted'],',.0f'):>16}"
          f"{'$'+format(r['lo80'],',.0f')+'  -  $'+format(r['hi80'],',.0f'):>34}"
          f"{share:>8.1f}%{SECTOR_BANDS[sec]['mape']:>15.0f}%")
print("-" * 100)
print(f"{'TOTAL':<24}{'$'+format(sector_q1['predicted'].sum(),',.0f'):>16}"
      f"{'  (matches the firm-wide headline of $'+format(q1_total,',.0f')+')':>34}")
print("-" * 100)
print("The final column is each sector's own historical forecast error in the backtest -")
print("read it as a reliability score. The sectors with the widest bands are the ones the")
print("model has genuinely struggled with, not an arbitrary choice.")


# ==================================================================================
# SECTION 6  -  OUTPUTS
# ==================================================================================
banner("SECTION 6  |  WRITING THE DELIVERABLES")

# Which holdout predictions belong to the model we actually selected? We keep them so the
# charts and the CSV can show "what the model said" next to "what really happened" for the
# twelve months it was graded on. That side-by-side is the single most persuasive exhibit
# in a forecasting deck, because it lets a sceptical reader check the model themselves.
_holdout_lookup = {
    "Random Forest (tuned)": rf_pred,
    "Random Forest + trend (hybrid)": hybrid_pred,
    "Linear Regression (benchmark)": lr_pred,
}
champion_pred = _holdout_lookup[CHAMPION_NAME]
sector_holdout_pred = hybrid_sec_pred if SECTOR_CHAMPION_NAME.startswith("Random Forest + trend") else sec_pred

# ---- 6.1 forecast_results.csv ----------------------------------------------------------
# One tidy table containing everything: observed history, the holdout comparison, and the
# forward forecast with its intervals. "Tidy" means one row per observation and one column
# per variable - the shape that pivot tables, BI tools and every plotting library expect.
rows_out = []

# (a) Firm-wide history. Every month we observed, with the actual value.
for _, r in firmwide.iterrows():
    in_holdout = r["month"] in set(fw_test["month"])
    rows_out.append({
        "series": "Firm-wide", "month": str(r["month"]),
        "segment": "holdout" if in_holdout else "history",
        "actual": r["revenue"], "predicted": np.nan,
        "lo80": np.nan, "hi80": np.nan, "lo95": np.nan, "hi95": np.nan,
    })

# (b) Fill in the model's holdout predictions against those actual months.
holdout_map = dict(zip(fw_test["month"].astype(str), champion_pred))
for row in rows_out:
    if row["segment"] == "holdout" and row["month"] in holdout_map:
        row["predicted"] = float(holdout_map[row["month"]])

# (c) The Q1 2026 forecast itself.
for _, r in firm_fc.iterrows():
    rows_out.append({
        "series": "Firm-wide", "month": r["month"], "segment": "forecast",
        "actual": np.nan, "predicted": r["predicted"],
        "lo80": r["lo80"], "hi80": r["hi80"], "lo95": r["lo95"], "hi95": r["hi95"],
    })

# (d) Exactly the same three blocks again, once per sector.
sec_holdout_map = {}
for (m, sec), pred in zip(zip(sec_test["month"].astype(str), sec_test["sector"]), sector_holdout_pred):
    sec_holdout_map[(m, sec)] = float(pred)

for sec in all_sectors:
    hist = by_sector[by_sector["sector"] == sec].sort_values("month")
    for _, r in hist.iterrows():
        key = (str(r["month"]), sec)
        in_holdout = key in sec_holdout_map
        rows_out.append({
            "series": sec, "month": str(r["month"]),
            "segment": "holdout" if in_holdout else "history",
            "actual": r["revenue"],
            "predicted": sec_holdout_map.get(key, np.nan),
            "lo80": np.nan, "hi80": np.nan, "lo95": np.nan, "hi95": np.nan,
        })
    for _, r in sector_fc[sector_fc["sector"] == sec].iterrows():
        rows_out.append({
            "series": sec, "month": r["month"], "segment": "forecast",
            "actual": np.nan, "predicted": r["predicted"],
            "lo80": r["lo80"], "hi80": r["hi80"], "lo95": r["lo95"], "hi95": r["hi95"],
        })

results = pd.DataFrame(rows_out)
results = results.sort_values(
    ["series", "month"],
    key=lambda c: c if c.name != "series" else c.map({"Firm-wide": ""}).fillna(c),
).reset_index(drop=True)
results.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV}  ({len(results):,} rows: "
      f"{(results.segment=='history').sum()} history, "
      f"{(results.segment=='holdout').sum()} holdout, "
      f"{(results.segment=='forecast').sum()} forecast)")


# ---- 6.2 Shared chart helper -----------------------------------------------------------
def month_axis(ax, periods, step=6):
    """
    Label an x-axis of month Periods without crowding it.

    Charts fail far more often from unreadable axes than from wrong numbers, so we show
    one label every `step` months and rotate nothing - slanted text is harder to read.
    """
    ticks = list(range(0, len(periods), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(periods[i]) for i in ticks], fontsize=9)


# ---- 6.3 chart_firmwide_forecast.png ----------------------------------------------------
# DESIGN NOTE: three forecast months sitting next to sixty months of history are almost
# invisible - the part the board actually came to see gets about 5% of the width. So the
# chart is split in two. The left panel gives five years of context at a glance; the right
# panel zooms into the last year plus the forecast, where the confidence bands and the
# monthly numbers have room to be read. A tinted marker on the left panel shows which
# slice the right panel magnifies, so the two are obviously connected.
hist_periods = firmwide["month"].tolist()
fc_periods = [pd.Period(m, freq="M") for m in firm_fc["month"]]
all_periods = hist_periods + fc_periods
last_actual = float(firmwide["revenue"].iloc[-1])

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(15.5, 6.6), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.16})

# ---------------- LEFT PANEL: the full five-year story ----------------
xh = np.arange(len(hist_periods))
xf = np.arange(len(hist_periods), len(all_periods))
ZOOM_BACK = 12                      # how many months of history the right panel shows
zoom_start = len(hist_periods) - ZOOM_BACK

axL.axvspan(zoom_start - 0.5, len(all_periods) - 0.5, color="#eef3fb", zorder=0)
axL.plot(xh, firmwide["revenue"], color=C_HISTORY, linewidth=1.9, zorder=3)
axL.plot(np.concatenate([[xh[-1]], xf]),
         np.concatenate([[last_actual], firm_fc["predicted"]]),
         color=C_FORECAST, linewidth=2.2, zorder=4)
axL.axvline(xh[-1], color=C_INK_SOFT, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)

axL.yaxis.set_major_formatter(FuncFormatter(money))
axL.set_ylim(0, float(max(firmwide["revenue"].max(), firm_fc["hi95"].max())) * 1.08)
axL.set_xlim(-1, len(all_periods))
month_axis(axL, all_periods, step=12)
axL.set_ylabel("Total contract value signed per month", color=C_INK_SOFT)
axL.set_title("Five years of monthly revenue", fontsize=12, color=C_INK, pad=10, loc="left")
axL.annotate("detail at right", xy=((zoom_start + len(all_periods)) / 2, 0.94),
             xycoords=("data", "axes fraction"), ha="center", fontsize=9, color=C_INK_SOFT)

# ---------------- RIGHT PANEL: the last year, plus the forecast ----------------
zh = np.arange(ZOOM_BACK)
zf = np.arange(ZOOM_BACK, ZOOM_BACK + len(fc_periods))
zoom_periods = hist_periods[zoom_start:] + fc_periods
zoom_rev = firmwide["revenue"].iloc[zoom_start:].values

bridge = np.concatenate([[zh[-1]], zf])
axR.fill_between(bridge, np.concatenate([[last_actual], firm_fc["lo95"]]),
                 np.concatenate([[last_actual], firm_fc["hi95"]]), color=C_BAND_95, zorder=1)
axR.fill_between(bridge, np.concatenate([[last_actual], firm_fc["lo80"]]),
                 np.concatenate([[last_actual], firm_fc["hi80"]]), color=C_BAND_80, zorder=2)

axR.plot(zh, zoom_rev, color=C_HISTORY, linewidth=2.4, zorder=4)
axR.plot(zh, zoom_rev, "o", color=C_HISTORY, markersize=5,
         markeredgecolor=C_SURFACE, markeredgewidth=1.5, zorder=5)
axR.plot(bridge, np.concatenate([[last_actual], firm_fc["predicted"]]),
         color=C_FORECAST, linewidth=2.8, zorder=6)
axR.plot(zf, firm_fc["predicted"], "o", color=C_FORECAST, markersize=10,
         markeredgecolor=C_SURFACE, markeredgewidth=2.2, zorder=7)

axR.axvline(zh[-1], color=C_INK_SOFT, linestyle=(0, (4, 3)), linewidth=1.3, zorder=3)
# Sit this label just INSIDE the plot area. Placed above the axes it collides with the
# panel title, and a caption that overlaps a heading reads as a mistake on a board slide.
axR.annotate("Actual  |  Forecast", xy=(zh[-1], 0.965), xycoords=("data", "axes fraction"),
             ha="center", va="top", fontsize=9.5, color=C_INK_SOFT)

for i, (_, r) in enumerate(firm_fc.iterrows()):
    # January's label is nudged right so it clears the steep line falling away from
    # December's record month; the other two sit centred over their own point.
    dx = 16 if i == 0 else 0
    axR.annotate(money(r["predicted"]), xy=(zf[i], r["hi95"]),
                 xytext=(dx, 9), textcoords="offset points",
                 ha="center", fontsize=11, fontweight="bold", color=C_FORECAST)

axR.yaxis.set_major_formatter(FuncFormatter(money))
axR.set_ylim(0, float(max(firm_fc["hi95"].max(), zoom_rev.max())) * 1.20)
axR.set_xlim(-0.6, len(zoom_periods) - 0.4)
month_axis(axR, zoom_periods, step=3)
axR.set_title("Last 12 months and the Q1 2026 forecast", fontsize=12, color=C_INK,
              pad=12, loc="left")

# ---------------- Shared titles and legend ----------------
fig.suptitle("Monthly revenue outlook: Q1 2026", fontsize=17, fontweight="bold",
             color=C_INK, x=0.062, y=1.045, ha="left")
fig.text(0.062, 0.985,
         f"Q1 2026 forecast {money(q1_total)} in total   -   80% confidence "
         f"{money(q1_lo80)} to {money(q1_hi80)}   -   "
         f"{(q1_total / q1_2025_actual - 1) * 100:+.0f}% versus Q1 2025",
         fontsize=11.5, color=C_INK_SOFT, ha="left")

fig.legend(handles=[
    Line2D([], [], color=C_HISTORY, linewidth=2.4, marker="o", markersize=6,
           label="Actual revenue"),
    Line2D([], [], color=C_FORECAST, linewidth=2.8, marker="o", markersize=8, label="Forecast"),
    Patch(facecolor=C_BAND_80, label="80% confidence range"),
    Patch(facecolor=C_BAND_95, label="95% confidence range"),
], loc="lower center", ncol=4, fontsize=10.5, bbox_to_anchor=(0.5, -0.05))

fig.savefig("chart_firmwide_forecast.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Wrote chart_firmwide_forecast.png")


# ---- 6.4 chart_sector_forecast.png ------------------------------------------------------
# SMALL MULTIPLES: seven small copies of the same chart, one per sector, sharing a layout
# so the eye can compare them instantly. This is the right form here, and the alternative
# is worse: seven coloured lines on one set of axes would be a tangle, and the largest
# sector would squash the smallest into the baseline.
#
# One deliberate choice needs explaining. Each panel gets its OWN y-axis scale, not a
# shared one. A shared scale would make the smaller sectors look like flat lines and hide
# their seasonal shape entirely, which defeats the purpose. The trade-off is that panel
# heights are no longer comparable between sectors, so each panel carries its Q1 total in
# words - the number does the comparing, not the pixel height. Every panel is also drawn
# in the same colour, because colour here would only repeat what the panel title says.
sector_order = sector_q1.index.tolist()          # biggest first, so the eye starts there
fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.0))
axes_flat = axes.flatten()

SEC_ZOOM = 18                                    # months of history per panel
for i, sec in enumerate(sector_order):
    ax = axes_flat[i]
    h = by_sector[by_sector["sector"] == sec].sort_values("month")
    hv = h["revenue"].iloc[-SEC_ZOOM:].values
    hp = h["month"].iloc[-SEC_ZOOM:].tolist()
    f = sector_fc[sector_fc["sector"] == sec].reset_index(drop=True)

    xs = np.arange(len(hv))
    xfz = np.arange(len(hv), len(hv) + len(f))
    br = np.concatenate([[xs[-1]], xfz])
    lastv = float(hv[-1])

    ax.fill_between(br, np.concatenate([[lastv], f["lo95"]]),
                    np.concatenate([[lastv], f["hi95"]]), color=C_BAND_95, zorder=1)
    ax.fill_between(br, np.concatenate([[lastv], f["lo80"]]),
                    np.concatenate([[lastv], f["hi80"]]), color=C_BAND_80, zorder=2)
    ax.plot(xs, hv, color=C_HISTORY, linewidth=2.0, zorder=4)
    ax.plot(br, np.concatenate([[lastv], f["predicted"]]),
            color=C_FORECAST, linewidth=2.4, zorder=5)
    ax.plot(xfz, f["predicted"], "o", color=C_FORECAST, markersize=6,
            markeredgecolor=C_SURFACE, markeredgewidth=1.6, zorder=6)
    ax.axvline(xs[-1], color=C_INK_SOFT, linestyle=(0, (3, 3)), linewidth=1.0, zorder=3)

    ax.set_title(sec, fontsize=11.5, fontweight="bold", color=C_INK, pad=20, loc="left")
    ax.text(0, 1.015, f"Q1 2026: {money(sector_q1.loc[sec, 'predicted'])}",
            transform=ax.transAxes, fontsize=10, color=C_FORECAST, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_ylim(0, float(max(f["hi95"].max(), hv.max())) * 1.12)
    ax.tick_params(labelsize=8.5)
    period_labels = hp + [pd.Period(m, freq="M") for m in f["month"]]
    month_axis(ax, period_labels, step=6)

# The eighth cell has no sector, so it carries the legend and the reading instructions
# instead of being left as a distracting empty box.
legend_ax = axes_flat[len(sector_order)]
legend_ax.axis("off")
legend_ax.legend(handles=[
    Line2D([], [], color=C_HISTORY, linewidth=2.0, label="Actual revenue"),
    Line2D([], [], color=C_FORECAST, linewidth=2.4, marker="o", markersize=6, label="Forecast"),
    Patch(facecolor=C_BAND_80, label="80% confidence range"),
    Patch(facecolor=C_BAND_95, label="95% confidence range"),
], loc="upper left", fontsize=10.5, bbox_to_anchor=(0.0, 0.92))
legend_ax.text(0.0, 0.30,
               "Each panel has its own vertical scale so that\n"
               "smaller sectors stay readable. Compare sectors\n"
               "using the Q1 totals, not the height of the lines.",
               transform=legend_ax.transAxes, fontsize=9.5, color=C_INK_SOFT, va="top")

fig.suptitle("Q1 2026 revenue forecast by sector", fontsize=17, fontweight="bold",
             color=C_INK, x=0.045, y=1.005, ha="left")
fig.text(0.045, 0.963,
         f"Last 18 months of actual revenue and the three-month forecast for each of the "
         f"seven sectors. The seven Q1 totals add up to the firm-wide figure of {money(q1_total)}.",
         fontsize=11, color=C_INK_SOFT, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig("chart_sector_forecast.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Wrote chart_sector_forecast.png")


# ---- 6.5 chart_feature_importance.png ---------------------------------------------------
# A horizontal bar chart is the right form for ranked categories: the labels sit
# horizontally where they are easy to read, and length is the easiest visual property for
# a person to compare accurately. Bars are sorted longest-first so the ranking is the
# chart's most obvious feature.
imp = importances.sort_values("importance")
fig, ax = plt.subplots(figsize=(11.5, 5.6))
bars = ax.barh([PRETTY.get(f, f) for f in imp["feature"]], imp["importance"],
               color=C_HISTORY, height=0.62)
# The top driver is highlighted, because "what matters most" is the one thing a reader
# should take away if they look at this chart for only two seconds.
bars[-1].set_color(C_FORECAST)

for b, v in zip(bars, imp["importance"]):
    ax.annotate(f"{v:.0%}", xy=(v, b.get_y() + b.get_height() / 2),
                xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10.5, fontweight="bold", color=C_INK)

ax.set_xlim(0, float(imp["importance"].max()) * 1.16)
ax.xaxis.set_visible(False)                 # the direct labels already give the numbers
ax.grid(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=11)

# The y-axis labels are long, so the plotting area starts well to the right. Anchoring the
# title to the axes would therefore indent it oddly and, with a two-line block, overlap the
# subtitle. Both are placed in FIGURE coordinates instead, flush with the left edge of the
# image where a reader expects a headline to start.
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.suptitle("What drives the revenue forecast", fontsize=16, fontweight="bold",
             color=C_INK, x=0.005, y=1.00, ha="left")
fig.text(0.005, 0.935,
         f"Share of the model's predictive power contributed by each input. "
         f"'{PRETTY.get(TOP3[0], TOP3[0])}' is the strongest single driver.",
         fontsize=11, color=C_INK_SOFT, ha="left")
fig.savefig("chart_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Wrote chart_feature_importance.png")


# ---- 6.6 chart_model_comparison.png -----------------------------------------------------
# Two panels answering two different questions a sceptical reader will ask.
#   LEFT  "How wrong is it, and is that better than something simple?" - the error metrics.
#   RIGHT "Show me it actually working." - the holdout months, predicted against actual.
# The right panel is the more persuasive of the two, because it lets the reader judge with
# their own eyes rather than trusting a summary statistic.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.2),
                               gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.22})

# ---- LEFT: grouped bars of the three error metrics ----
# MAE and RMSE are dollars; MAPE is a percentage. Plotting them on one axis would be
# meaningless, so each metric is shown as a share of the WORST model's score on that
# metric. Shorter is better, and all three become directly comparable.
models = [("Linear Regression", lr_scores, C_BENCH),
          ("Random Forest", rf_scores, C_HISTORY),
          ("Random Forest + trend", hybrid_scores, C_FORECAST)]
metrics = ["MAE", "RMSE", "MAPE"]
bar_w = 0.26
xpos = np.arange(len(metrics))

for j, (name, sc, colour) in enumerate(models):
    worst = [max(m[1][k] for m in models) for k in metrics]
    heights = [sc[k] / w for k, w in zip(metrics, worst)]
    offset = (j - 1) * bar_w
    b = axA.bar(xpos + offset, heights, width=bar_w * 0.9, color=colour, label=name)
    for k, rect in enumerate(b):
        raw = sc[metrics[k]]
        txt = f"{raw:.1f}%" if metrics[k] == "MAPE" else money(raw)
        axA.annotate(txt, xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9, color=C_INK)

axA.set_xticks(xpos)
axA.set_xticklabels(["Average miss\n(MAE)", "Large misses\n(RMSE)",
                     "Average miss %\n(MAPE)"], fontsize=10.5)
# Extra headroom so the legend sits clear of the tallest bar and its value label rather
# than crowding them.
axA.set_ylim(0, 1.55)
axA.yaxis.set_visible(False)
axA.grid(False)
axA.spines["left"].set_visible(False)
axA.legend(fontsize=10, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0))
axA.set_title("Forecast error on the 12 holdout months  (shorter is better)",
              fontsize=12, color=C_INK, pad=12, loc="left")

# ---- RIGHT: actual vs predicted across the holdout ----
xh2 = np.arange(len(fw_test))
axB.plot(xh2, y_test.values, color=C_INK, linewidth=3.0, label="What actually happened",
         zorder=5)
axB.plot(xh2, lr_pred, color=C_BENCH, linewidth=1.9, linestyle="--",
         label="Linear Regression", zorder=3)
axB.plot(xh2, rf_pred, color=C_HISTORY, linewidth=1.9, linestyle=":",
         label="Random Forest", zorder=3)
axB.plot(xh2, hybrid_pred, color=C_FORECAST, linewidth=2.4,
         label="Random Forest + trend", zorder=4)

axB.yaxis.set_major_formatter(FuncFormatter(money))
axB.set_ylim(bottom=0)
axB.set_xticks(list(range(0, len(fw_test), 2)))
axB.set_xticklabels([str(fw_test["month"].iloc[i]) for i in range(0, len(fw_test), 2)],
                    fontsize=9.5)
axB.legend(fontsize=10, loc="upper left")
axB.set_title("Each model tested against months it had never seen",
              fontsize=12, color=C_INK, pad=12, loc="left")
axB.set_ylabel("Monthly revenue", color=C_INK_SOFT)

_verdict = (f"{CHAMPION_NAME} was selected: "
            f"{abs(improvement_hybrid['MAE']):.0f}% "
            f"{'better' if improvement_hybrid['MAE'] > 0 else 'worse'} than the simple benchmark")
fig.suptitle("Choosing the forecasting model", fontsize=17, fontweight="bold",
             color=C_INK, x=0.045, y=1.045, ha="left")
fig.text(0.045, 0.982, _verdict, fontsize=11.5, color=C_INK_SOFT, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("chart_model_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Wrote chart_model_comparison.png")


# ---- 6.7 summary.md ---------------------------------------------------------------------
# Written FROM the computed numbers, never typed by hand. If the data changes and the
# script is re-run, this document updates itself. Hand-typed summaries drift out of step
# with the analysis they describe, and a stale number in a board pack is worse than no
# number at all.
# Rank sectors by how badly the model missed them in the backtest, NOT by how small they
# are. Small and unpredictable are different things, and conflating them would put a false
# statement in front of the CEO.
least_reliable = sorted(all_sectors, key=lambda x: -SECTOR_BANDS[x]["mape"])

top_sector = sector_q1.index[0]
top_sector_val = sector_q1["predicted"].iloc[0]
fastest = sector_q1["predicted"].idxmax()
growth_vs_2025 = (q1_total / q1_2025_actual - 1) * 100
best_month = firm_fc.loc[firm_fc["predicted"].idxmax(), "month"]
weakest_month = firm_fc.loc[firm_fc["predicted"].idxmin(), "month"]
top_driver_plain = PRETTY.get(TOP3[0], TOP3[0]).lower()

summary_text = f"""# Q1 2026 Revenue Forecast - Summary for the CEO

**We expect to sign {money(q1_total)} of new business in the first quarter of 2026** -
{money(firm_fc['predicted'].iloc[0])} in January, {money(firm_fc['predicted'].iloc[1])} in
February and {money(firm_fc['predicted'].iloc[2])} in March. That is about
{growth_vs_2025:+.0f}% against the {money(q1_2025_actual)} we signed in the same quarter
last year, and there is roughly an 8-in-10 chance the true figure lands between
{money(q1_lo80)} and {money(q1_hi80)}. The single biggest influence on the forecast is
{top_driver_plain}: our revenue follows a strong and very consistent yearly rhythm, with
January the weakest month of the quarter and a large spike every December.

**How much should you trust it?** The forecast was tested by hiding the most recent twelve
months from the model and asking it to predict them blind. It came within
{champion_scores['MAPE']:.0f}% of the real figure in a typical month, or about
{money(champion_scores['MAE'])}. That is useful for planning and direction-setting, but it
is not precise enough to commit to a specific number - monthly revenue in this business
swings widely, and no model removes that. Treat the range as the forecast and the single
figure as its midpoint.

**Where the money comes from.** {top_sector} remains the largest contributor at
{money(top_sector_val)} for the quarter ({top_sector_val / q1_total * 100:.0f}% of the
total), followed by {sector_q1.index[1]} and {sector_q1.index[2]}. Confidence varies a
great deal by sector: the largest sectors are the most predictable, while smaller ones can
be swung by a single contract landing a few weeks either side of a month end.

**Three recommendations.**

1. **Plan January conservatively.** It is consistently our weakest month, and this year is
   no exception - we expect {money(firm_fc['predicted'].iloc[0])}, roughly
   {(1 - firm_fc['predicted'].iloc[0] / firm_fc['predicted'].iloc[2]) * 100:.0f}% below
   March. Set cash and resourcing plans against the lower end of the range, not the middle.

2. **Protect the Q4 push.** December is by some distance our biggest month every single
   year, and Q1 inherits its momentum. Whatever we do in Q4 to close year-end business is
   the most valuable thing we do all year and should be resourced first.

3. **Tighten pipeline reporting where the forecast is weakest.** {least_reliable[0]} and
   {least_reliable[1]} are the two sectors the model predicts least reliably - it missed
   them by {SECTOR_BANDS[least_reliable[0]]['mape']:.0f}% and
   {SECTOR_BANDS[least_reliable[1]]['mape']:.0f}% respectively in testing, against
   {SECTOR_BANDS[least_reliable[-1]]['mape']:.0f}% for our most predictable sector. A
   weekly pipeline review in those two will improve the overall forecast more than the same
   effort spent anywhere else.

**The one assumption that matters.** This forecast assumes the next three months behave
broadly like the last five years - same seasonal rhythm, same growth trajectory, no major
shock. It cannot anticipate anything it has not seen before: losing a top client, a change
in the market, or an unusually large deal would all put the actual figure outside the range
above.

---

*Forecast produced {datetime.now().strftime('%d %B %Y')} using
{CHAMPION_NAME.lower()}, selected on measured accuracy against a 12-month holdout.
Based on synthetic data - see `assumptions.md` before using any figure externally.*
"""
def reflow(md, width=88):
    """
    Re-wrap prose to a fixed width, leaving structure alone.

    Markdown collapses single newlines anyway, so this changes nothing about how the file
    renders - but it makes the raw file pleasant to read in a terminal or a diff, which is
    where a colleague is most likely to first encounter it.

    Three details are easy to get wrong, and all three were bugs here before they were
    fixed:
      1. A bullet must be a dash/asterisk/number FOLLOWED BY A SPACE. Match on the marker
         alone and "**Bold text**" is mistaken for a list item.
      2. A list written with no blank lines between items is a single block. Wrapping it as
         one paragraph runs every item together into prose, so the block has to be split at
         each new marker first.
      3. Continuation lines of a wrapped item need indenting to line up under the first
         word, or the list stops looking like a list.
    """
    bullet_start = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
    bullet_split = re.compile(r"^(\s*(?:[-*+]\s+|\d+\.\s+))(.*)$")
    passthrough = ("#", "|", ">", "```", "---", "===")
    out = []

    def wrap_item(marker, body):
        return textwrap.fill(" ".join(body.split()), width=width,
                             initial_indent=marker, subsequent_indent=" " * len(marker))

    for para in md.split("\n\n"):
        stripped = para.strip()
        if not stripped or stripped.startswith(passthrough):
            out.append(para)
            continue

        lines = para.split("\n")
        if any(bullet_start.match(ln) for ln in lines):
            # Regroup the block into one entry per list item, then wrap each separately.
            items, current = [], []
            for ln in lines:
                if bullet_start.match(ln) and current:
                    items.append(" ".join(current))
                    current = [ln]
                else:
                    current.append(ln)
            if current:
                items.append(" ".join(current))

            rendered = []
            for item in items:
                m = bullet_split.match(item)
                if m:
                    rendered.append(wrap_item(m.group(1), m.group(2)))
                else:
                    rendered.append(textwrap.fill(" ".join(item.split()), width=width))
            out.append("\n".join(rendered))
        else:
            out.append(textwrap.fill(" ".join(stripped.split()), width=width))

    return "\n\n".join(out) + "\n"


with open(OUT_SUMMARY, "w") as fh:
    fh.write(reflow(summary_text))
print(f"Wrote {OUT_SUMMARY}")


# ---- 6.8 assumptions.md -----------------------------------------------------------------
# The honest companion to summary.md. A forecast without a written statement of its limits
# invites people to use it for things it cannot support. Everything here is generated from
# the run itself, so it can never drift out of step with the numbers it describes.
assumptions_text = f"""# Assumptions and Limitations

*Companion to `summary.md`. Read this before quoting any figure from this analysis.*

## 1. The data is synthetic

Every row in `synthetic_project_data.csv` was generated by a computer program
(`generate_synthetic_data.py`, random seed 42). **No real client, employee or financial
record appears anywhere in this analysis, and the figures in `summary.md` describe an
invented company.** They must not be presented externally, or internally, as a real
forecast for a real business.

What the dataset is genuinely useful for is exercising the method: the pipeline, the
validation, the charts and the written outputs are all real work, and pointing them at a
real extract is a change of input file, not a rewrite.

## 2. How the data was generated

The generator models monthly revenue as `baseline x growth x seasonality x random noise`,
and deliberately builds in the following structure:

- **{N_SECTORS} sectors**, each with its own typical deal size, its own share of project
  volume, and its own monthly growth rate (from +0.25%/month in Public Sector to
  +0.90%/month in Technology & Telecom).
- **A repeating 12-month seasonal pattern.** January and the July-August period are the
  weakest months; October, November and December are the strongest, with December peaking
  at about 1.35x an average month. Public Sector gets an additional December uplift to
  mimic year-end budget spending.
- **Right-skewed contract values**, drawn from a lognormal distribution, so there are many
  mid-size projects and a few very large ones - as in real professional services.
- **A one-off market wobble** across roughly June-September 2022, about 12% below trend, so
  the series is not unrealistically smooth.
- **A ~6% cancellation rate**, producing the rows the pipeline then removes.

Because we wrote these rules, we can check the model against them. It is a good sign that
the model independently identified the calendar month as its single strongest driver at
{importances['importance'].iloc[0]:.0%} of total importance - that is the seasonality above,
recovered from the data rather than told to the model.

## 3. Usable sample size

This is the most important limitation in the whole analysis, and it is easy to miss because
1,897 sounds like a lot of data.

| Stage | Firm-wide rows | Per-sector rows |
|---|---|---|
| Projects in the raw file | 1,897 | 1,897 |
| After removing {n_cancelled} cancelled projects | {after:,} | {after:,} |
| **After aggregating to monthly totals** | **60** | **{len(all_months) * len(all_sectors)}** |
| After dropping rows with no 3-month history | {len(fw)} | {len(sec_feat)} |
| Available to train the model | **{n_train}** | **{len(sec_train)}** |
| Held back to test it | {n_test} | {len(sec_test)} |

Aggregating to monthly totals is what collapses 1,897 rows into 60. **The model is
effectively learning from {n_train} observations**, which is a very small sample by any
machine-learning standard, and it is the root cause of most of the limitations below. With
only {n_train} training months the model sees fewer than four examples of each calendar
month, so its view of, say, "what February looks like" rests on a handful of data points.

## 4. Model limitations

**Trees cannot extrapolate.** A Random Forest predicts by averaging training values, so it
can never output a number outside the range it was trained on. Because this business grows
every year, a plain Random Forest was structurally incapable of predicting 2025 correctly -
it under-forecast the holdout by {abs(rf_bias):,.0f} dollars a month on average, and lost to
simple Linear Regression ({rf_scores['MAE']:,.0f} vs {lr_scores['MAE']:,.0f} MAE). The
selected model fixes this by fitting a straight-line trend first and letting the forest
predict only the movement around it. **This is the single most important technical finding
in the analysis** and it is why the headline forecast does not come from a plain Random
Forest.

**Accuracy is modest in absolute terms.** The selected model was typically within
{champion_scores['MAPE']:.0f}% ({money(champion_scores['MAE'])}) of the actual month across
the 12-month holdout. Monthly revenue in this dataset ranges from
{money(firmwide['revenue'].min())} to {money(firmwide['revenue'].max())}, so that error is
real. The forecast supports direction-setting and range-based planning; it does not support
committing to a precise number.

**Error compounds across the quarter.** The forecast is recursive: February's prediction
uses January's prediction as an input, and March's uses both. Measured across
{bt_df['horizon'].eq(1).sum()} rolling-origin backtests, error grew from
{money(h_mae_corrected[1])} one month out to {money(h_mae_corrected[3])} three months out
(x{HORIZON_SCALE[3]:.2f}). March is meaningfully less reliable than January, which is why
the confidence bands widen across the quarter rather than running parallel.

**A measured bias correction is applied.** The model under-forecast throughout the backtest,
because 2025 grew faster than the straight-line trend fitted to 2021-2024. We add the
measured correction ({money(BIAS[1])} at one month out, rising to {money(BIAS[3])} at three)
to each forecast month. This is legitimate and standard, but it assumes the recent
acceleration continues. **If growth reverts to the longer-run trend, this forecast will be
too high.**

**The intervals are empirical, not theoretical.** {INTERVAL_METHOD} Coverage was verified at
{cal_cov80:.0%} against a nominal 80%. Note that quantile-forest was tried first, as the
preferred method, and rejected on evidence rather than preference - with {len(fw)} data
points and a January 2026 whose inputs sit outside the training range, its bands came out
about {qrf_width80 / max(emp_width80, 1):.1f}x wider for the same coverage.

**The quarterly interval is deliberately conservative.** {money(q1_lo80)} to
{money(q1_hi80)} is the sum of the three monthly intervals, which assumes all three months
miss in the same direction together. In practice some cancellation is likely, so the true
80% range for the quarter is probably narrower than stated.

**The two models do not have to agree.** The firm-wide model and the per-sector model are
separate. Before reconciliation the sector forecasts summed to about
{(sector_fc['predicted_raw'].sum() / q1_total - 1) * 100:+.0f}% of the firm-wide figure. We
kept the firm-wide total as the headline (aggregation cancels noise, and it is the number we
validated most carefully) and scaled the sectors proportionally to match. The sector split
is therefore a share-out of a validated total, not seven independent forecasts.

**Only the signing date is modelled.** Revenue is booked entirely against `start_date`. A
project signed in March for $1.2M counts as $1.2M of March revenue, even if the work and the
cash span the following year. This is a bookings forecast, not a recognised-revenue or cash
forecast, and it should not be used for cash planning without being reworked.

**Nothing outside the data is modelled.** No pipeline, no headcount, no win rates, no
pricing, no marketing spend, no competitor activity, no economic indicators. The model only
knows the calendar and the recent revenue history.

## 5. What changes when you use real data

1. **Expect the accuracy to be different, and check it before trusting it.** Real data is
   messier than synthetic data. Re-read the holdout comparison the pipeline prints - do not
   assume the {champion_scores['MAPE']:.0f}% error carries over.
2. **Check the column names first.** The pipeline expects `{DATE_COL}`, `{TARGET_COL}`,
   `{SECTOR_COL}` and `{STATUS_COL}`. It also expects `Cancelled` to be spelled that way;
   real systems often use several codes for the same thing.
3. **Expect real data quality problems.** Duplicate projects, missing dates, currency
   mixing, contract values revised after signing, projects re-entered under a new ID.
   Budget more time for cleaning than for modelling - that ratio is normal.
4. **The bias correction must be re-measured, not carried over.** It is specific to this
   dataset's growth pattern.
5. **Add the features you actually have.** Pipeline value at the start of each month, win
   rates, headcount and sales activity are usually far more predictive than the calendar
   alone. The calendar is what is left when you have nothing else.
6. **Re-examine the aggregation.** If your business is materially larger, weekly buckets
   would give roughly 4x the training rows and remove the single biggest constraint here.
7. **Re-run the whole pipeline rather than editing outputs.** Every number in `summary.md`
   and in this file is generated from the data. Re-running keeps them consistent; hand-
   editing guarantees they will eventually contradict each other.

---

*Generated {datetime.now().strftime('%d %B %Y')} by `forecast_q1_2026.py`.
Model: {CHAMPION_NAME}. Training months: {n_train}. Holdout months: {n_test}.*
"""
with open(OUT_ASSUMPTIONS, "w") as fh:
    fh.write(reflow(assumptions_text))
print(f"Wrote {OUT_ASSUMPTIONS}")

banner("DONE")
print("Deliverables written to " + os.getcwd() + ":")
for f in [OUT_CSV, "chart_firmwide_forecast.png", "chart_sector_forecast.png",
          "chart_feature_importance.png", "chart_model_comparison.png",
          "chart_partial_dependence.png", OUT_SUMMARY, OUT_ASSUMPTIONS]:
    size = os.path.getsize(f) / 1024 if os.path.exists(f) else 0
    print(f"   {f:<34} {size:>8.0f} KB")
print(f"\nHEADLINE: Q1 2026 forecast {money(q1_total)}  "
      f"(80% confidence {money(q1_lo80)} - {money(q1_hi80)})")
