# %%
# Define parameters
from settings import LOOKBACK_DAYS, TEST_ASSET, DATA_PATH, TRAIN_TEST_SPLIT

# %%
import numpy as np
import pandas as pd
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

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
# Check if TEST_ASSET is in the data
if TEST_ASSET not in df.index.get_level_values("Symbol"):
    raise ValueError(f"TEST_ASSET '{TEST_ASSET}' not found in the data")

# %%
# Filter away data before 1990
df = df[df.index.get_level_values("Date") >= "1990-01-01"]
df

# %%
# Fit a GARCH-GJR(1,1) model and make predictions
# Filter the data for the selected symbol
df_filtered = df.xs(TEST_ASSET, level="Symbol")

# Training data
returns_train = df_filtered["LogReturn"].loc[:TRAIN_TEST_SPLIT]
returns_train = returns_train * 100  # Scale to percentages

# Test data
returns_test = df_filtered["LogReturn"].loc[TRAIN_TEST_SPLIT:]
scaled_returns_test = returns_test * 100  # Scale to percentages

# Initialize an empty list to store forecasts
gjr_vol_pred = []

# Combine the training and test data
returns_combined = pd.concat([returns_train, scaled_returns_test])

# Perform rolling forecasts
for i in range(len(scaled_returns_test)):
    if i % 20 == 0:
        print(f"Progress: {i/len(scaled_returns_test):.2%}", end="\r")

    # Update the model with data up to the current point in time
    end = len(returns_train) + i
    returns_sample = returns_combined.iloc[:end]

    # Fit the GJR-GARCH model
    am = arch_model(returns_sample, vol="GARCH", p=1, q=1, o=1, mean="Zero")
    res = am.fit(disp="off")

    # Forecast the next time point
    forecast = res.forecast(horizon=1)
    forecast_var = forecast.variance.iloc[-1].values[0]
    forecast_vol = np.sqrt(forecast_var) / 100  # Adjust scaling
    gjr_vol_pred.append(forecast_vol)

# Convert the list to a numpy array
gjr_vol_pred = np.array(gjr_vol_pred)

# %%
# Save GJR-GARCH predictions to file
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Volatility"] = gjr_vol_pred
df_test["Mean"] = 0  # Assume mean is 0
df_test.to_csv(
    f"predictions/gjr_garch_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)

# %%
