# File for enriching prediction files before ES testing in R

# %%
# Load libraries
from matplotlib.pylab import norm
import pandas as pd
import numpy as np

from settings import DATA_PATH, TEST_SET, TRAIN_VALIDATION_SPLIT, VALIDATION_TEST_SPLIT
from shared.mdn import calculate_es_for_quantile
from shared.conf_levels import format_cl

CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.98]


# %%
# prepare the return data
df = pd.read_csv(DATA_PATH)
# Ensure the Date column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Remove .O suffix from tickers
df["Symbol"] = df["Symbol"].str.replace(".O", "")

# Sort the dataframe by both Date and Symbol
df = df.sort_values(["Symbol", "Date"])

# Calculate log returns for each instrument separately using groupby
df["LogReturn"] = (
    df.groupby("Symbol")["Close"].apply(lambda x: np.log(x / x.shift(1))).droplevel(0)
)
df["Total Return Test"] = (
    df.groupby("Symbol")["Close"].apply(lambda x: (x / x.shift(1)) - 1).droplevel(0)
)
# Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
df = df[~df["LogReturn"].isnull()]

df["SquaredReturn"] = df["LogReturn"] ** 2

# Set date and symbol as index
df: pd.DataFrame = df.set_index(["Date", "Symbol"])

dates = df.index.get_level_values("Date")
df_return = (
    df[(dates >= TRAIN_VALIDATION_SPLIT) & (dates < VALIDATION_TEST_SPLIT)]
    if TEST_SET == "validation"
    else df[(dates >= VALIDATION_TEST_SPLIT)]
)
df_return

# %%
# define y_true
y_true_log = df_return["LogReturn"].values
y_true_total = df_return["Total Return Test"].values

# %%
# HAR models
har_preds = pd.read_csv("predictions/HAR_R.csv")
har_preds["Date"] = pd.to_datetime(har_preds["Date"])
har_preds = har_preds.set_index(["Date", "Symbol"])
har_dates = har_preds.index.get_level_values("Date")
har_preds = har_preds[
    (
        (har_dates >= TRAIN_VALIDATION_SPLIT)
        & (har_dates < VALIDATION_TEST_SPLIT)
        if TEST_SET == "validation"
        else (har_dates >= VALIDATION_TEST_SPLIT)
    )
]
har_vol_preds = har_preds["HAR_vol_R"].values
mus = np.zeros_like(har_vol_preds)

for cl in CONFIDENCE_LEVELS:
    alpha = 1 - cl
    z_alpha = norm.ppf(1 - alpha / 2)
    lb = mus - z_alpha * har_vol_preds
    ub = mus + z_alpha * har_vol_preds
    har_preds[f"LB_{format_cl(cl)}"] = lb
    har_preds[f"UB_{format_cl(cl)}"] = ub
    es_alpha = alpha / 2
    har_preds[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
        np.ones_like(mus).reshape(-1, 1),
        mus.reshape(-1, 1),
        har_vol_preds.reshape(-1, 1),
        lb,
    )

# write enriched HAR predictions back to the same file
har_preds.reset_index(inplace=True)
har_preds.to_csv("predictions/HAR_R.csv", index=False)


