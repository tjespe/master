# %%
# Define parameters
from ..settings import LOOKBACK_DAYS, SUFFIX, TEST_ASSET, DATA_PATH, TRAIN_TEST_SPLIT

# %%
import numpy as np
import statsmodels.api as sm
import pandas as pd
import warnings

# %%
df = pd.read_csv(DATA_PATH)

if not "Symbol" in df.columns:
    df["Symbol"] = TEST_ASSET

# Ensure the Date column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort the dataframe by both Date and Symbol
df = df.sort_values(["Symbol", "Date"])

# Calculate log returns for each instrument separately using groupby
df["LogReturn"] = (
    df.groupby("Symbol")["Close"]
    .apply(lambda x: np.log(x / x.shift(1)))
    .reset_index()["Close"]
)

# Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
df = df[~df["LogReturn"].isnull()]

df["SquaredReturn"] = df["LogReturn"] ** 2

# Set date and symbol as index
df: pd.DataFrame = df.set_index(["Date", "Symbol"])
df

# %%
# Check for NaN values
nan_mask = df[["LogReturn"]].isnull().sum(axis=1).gt(0)
df[nan_mask]

# %%
# Add feature: is next day trading day or not
df["NextDayTradingDay"] = (
    df.index.get_level_values("Date")
    .shift(1, freq="D")
    .isin(df.index.get_level_values("Date"))
)
df["NextDayTradingDay"]

# %%
# Filter away data before 2001
df = df[df.index.get_level_values("Date") >= "2001-01-01"]
df

# %%
# Regress NextDayTradingDay on absolute log returns to check for significance
X = df["NextDayTradingDay"].astype(int)
X = sm.add_constant(X)
y = np.abs(df["LogReturn"])
model = sm.OLS(y, X, missing="drop")
results = model.fit()
results.summary()

# %%
# Check monday effect
df["DayOfWeek"] = df.index.get_level_values("Date").dayofweek
df["Monday"] = df["DayOfWeek"] == 0
df["Tuesday"] = df["DayOfWeek"] == 1
df["Wednesday"] = df["DayOfWeek"] == 2
df["Thursday"] = df["DayOfWeek"] == 3
df["Friday"] = df["DayOfWeek"] == 4

# %%
# Regress Monday on absolute log returns to check for significance
X = df["Monday"].astype(int)
X = sm.add_constant(X)
y = np.abs(df["LogReturn"])
model = sm.OLS(y, X, missing="drop")
results = model.fit()
results.summary()

# %%
# Regress Friday on absolute log returns to check for significance
X = df["Friday"].astype(int)
X = sm.add_constant(X)
y = np.abs(df["LogReturn"])
model = sm.OLS(y, X, missing="drop")
results = model.fit()
results.summary()

# %%
