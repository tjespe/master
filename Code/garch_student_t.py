# %%
# Define parameters
from settings import (
    LOOKBACK_DAYS,
    TEST_ASSET,
    DATA_PATH,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)

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
    df.groupby("Symbol")["Close"].apply(lambda x: np.log(x / x.shift(1))).droplevel(0)
)

# Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
df = df[~df["LogReturn"].isnull()]

df["SquaredReturn"] = df["LogReturn"] ** 2

# remove .O at the end of all symbols
df['Symbol'] = df['Symbol'].str.replace('.O', '')
# Set date and symbol as index
df: pd.DataFrame = df.set_index(["Date", "Symbol"])
df

# %%
# Filter away data before 1990
df = df[df.index.get_level_values("Date") >= "1990-01-01"]
df
# %%
# Fit a GARCH(1,1) model with t-distribution and make predictions, do it for each asset
symbols = df.index.get_level_values("Symbol").unique()
# initialize an empty list to store forecasts
garch_vol_pred = []

for symbol in symbols:
    print(f"Processing {symbol}")
    df_filtered = df.xs(symbol, level="Symbol")

    # Training data
    returns_train = df_filtered["LogReturn"].loc[:TRAIN_VALIDATION_SPLIT]
    returns_train = returns_train * 100  # Scale to percentages

    # Test data
    returns_validation = df_filtered["LogReturn"].loc[
        TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
    ]
    scaled_returns_test = returns_validation * 100  # Scale to percentages

    # Initialize an empty list to store forecasts

    # Combine the training and test data
    returns_combined = pd.concat([returns_train, scaled_returns_test])

    # Perform rolling forecasts
    for i in range(len(scaled_returns_test)):
        if i % 20 == 0:
            print(f"Progress: {i/len(scaled_returns_test):.2%}", end="\r")

        # Update the model with data up to the current point in time
        end = len(returns_train) + i
        returns_sample = returns_combined.iloc[:end]

        # Fit the GARCH model
        am = arch_model(returns_sample, vol="GARCH", p=1, q=1, mean="Zero", dist="t")
        res = am.fit(disp="off")

        # Forecast the next time point
        forecast = res.forecast(horizon=1)
        forecast_var = forecast.variance.iloc[-1].values[0]
        forecast_vol = np.sqrt(forecast_var) / 100  # Adjust scaling
        garch_vol_pred.append(forecast_vol)

    # Convert the list to a numpy array

garch_vol_pred = np.array(garch_vol_pred)

# %%
# Save GARCH predictions to file
df_validation = df[df.index.get_level_values("Date") >= TRAIN_VALIDATION_SPLIT]
df_validation = df_validation[df_validation.index.get_level_values("Date") < VALIDATION_TEST_SPLIT]
df_validation.reset_index(inplace=True)
# remove all coloumns except Symbol, Date, LogReturn, SquaredReturn
df_validation = df_validation[["Symbol", "Date", "LogReturn", "SquaredReturn"]]
df_validation
# %%
df_validation["GARCH_t_Vol"] = garch_vol_pred
df_validation["GARCH_t_Mean"] = 0
df_validation
# %%
df_validation.to_csv("predictions/garch_predictions_student_t.csv")
