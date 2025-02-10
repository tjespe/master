# %%
# Define parameters
from settings import LOOKBACK_DAYS, DATA_PATH, TEST_ASSET

# %%
import numpy as np
import pandas as pd
import os
from arch import arch_model
import warnings
from joblib import Parallel, delayed

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
# Filter away data before 2000
df = df[df.index.get_level_values("Date") >= "2000-01-01"]

# Get list of unique symbols
symbols = df.index.get_level_values("Symbol").unique()

# Define output file
base_path = DATA_PATH.replace(".csv", "")
output_path = f"{base_path}_garch_{LOOKBACK_DAYS}_days.csv"

# Check if a previous run exists
if os.path.exists(output_path):
    df_existing = pd.read_csv(
        output_path, index_col=["Date", "Symbol"], parse_dates=["Date"]
    )
    completed_symbols = df_existing.index.get_level_values("Symbol").unique()
    print(
        f"Resuming from previous progress. Skipping {len(completed_symbols)} already processed symbols."
    )
else:
    df_existing = None
    completed_symbols = set()


# %%
def process_symbol(symbol):
    """Fits GARCH(1,1) for a given symbol and makes rolling predictions for every time step."""
    if symbol in completed_symbols:
        print(f"Skipping {symbol}, already processed.")
        return None

    print(f"Processing {symbol}...")

    # Filter data for the symbol
    df_filtered = df.xs(symbol, level="Symbol")

    # Store predictions
    garch_vol_pred = []

    # Perform rolling estimation for every time step
    for i in range(1, len(df_filtered)):  # Start from index 1 to have enough data
        if i % 20 == 0:
            print(f"{symbol}: {i/len(df_filtered):.2%} completed", end="\r")

        # Use data up to the current time step
        returns_sample = df_filtered["LogReturn"].iloc[:i] * 100  # Scale to percentages

        try:
            # Fit the GARCH model
            am = arch_model(returns_sample, vol="GARCH", p=1, q=1, mean="Zero")
            res = am.fit(disp="off")

            # Forecast next time point
            forecast = res.forecast(horizon=1)
            forecast_var = forecast.variance.iloc[-1].values[0]
            forecast_vol = np.sqrt(forecast_var) / 100  # Adjust scaling

            garch_vol_pred.append(forecast_vol)

        except Exception as e:
            print(f"Error processing {symbol} at index {i}: {e}")
            garch_vol_pred.append(np.nan)  # Store NaN if error occurs

    # Create a DataFrame for results
    df_result = df_filtered.iloc[1:].copy()  # Align with predictions
    df_result["Volatility"] = garch_vol_pred
    df_result["Mean"] = 0  # Assume mean is 0
    df_result["Symbol"] = symbol

    # Save progress after each symbol
    if os.path.exists(output_path):
        df_result.to_csv(output_path, mode="a", header=False)  # Append without header
    else:
        df_result.to_csv(output_path)

    print(f"Finished {symbol}, saved progress.")
    return df_result


# %%
# Run in parallel for all symbols
results = Parallel(n_jobs=-1)(delayed(process_symbol)(symbol) for symbol in symbols)

# %%
print(f"Predictions saved to {output_path}")
