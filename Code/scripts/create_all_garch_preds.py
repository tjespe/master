# %%
# Define parameters
from settings import DATA_PATH, TEST_ASSET

# %%
import numpy as np
import pandas as pd
import os
import sys  # Import sys to handle command-line arguments
from arch import arch_model
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# %%
# Read input data
df = pd.read_csv(DATA_PATH)

if "Symbol" not in df.columns:
    df["Symbol"] = TEST_ASSET

# Ensure Date column is datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort dataframe
df = df.sort_values(["Symbol", "Date"])

# Compute log returns
df["LogReturn"] = (
    df.groupby("Symbol")["Close"].apply(lambda x: np.log(x / x.shift(1))).droplevel(0)
)

# Drop NaN rows
df = df[~df["LogReturn"].isnull()]

df["SquaredReturn"] = df["LogReturn"] ** 2

# Set index
df = df.set_index(["Date", "Symbol"])

# %%
# Remove null dates
df = df[~df.index.get_level_values("Date").isnull()]

# %%
# Define model type (default to "GARCH")
garch_type = sys.argv[1] if len(sys.argv) > 1 else "GARCH"
valid_models = {"GARCH", "EGARCH"}

if garch_type not in valid_models:
    raise ValueError(f"Invalid model type '{garch_type}'. Choose from {valid_models}.")

# Output file
base_path = DATA_PATH.replace(".csv", "")
output_path = f"{base_path}_{garch_type.lower()}.csv"

# Check existing results
if os.path.exists(output_path):
    df_existing = pd.read_csv(
        output_path, index_col=["Date", "Symbol"], parse_dates=["Date"]
    )
    completed_symbols = df_existing.index.get_level_values("Symbol").unique()
    print(f"Resuming: Skipping {len(completed_symbols)} symbols.")
else:
    df_existing = None
    completed_symbols = set()


# %%
def process_symbol(symbol):
    """Fits specified GARCH model (GARCH or EGARCH) and makes rolling predictions."""
    if symbol in completed_symbols:
        print(f"Skipping {symbol}, already processed.")
        return None

    print(f"Processing {symbol}...")

    df_filtered = df.xs(symbol, level="Symbol")
    garch_vol_pred = []

    for i in range(1, len(df_filtered)):  # Rolling window
        if i % 20 == 0:
            print(f"{symbol}: {i/len(df_filtered):.2%} completed", end="\r")

        returns_sample = df_filtered["LogReturn"].iloc[:i] * 100  # Scale to %

        try:
            # Fit model based on user input
            am = arch_model(returns_sample, vol=garch_type, p=1, q=1, mean="Zero")
            res = am.fit(disp="off")

            # Forecast next time point
            forecast = res.forecast(horizon=1)
            forecast_var = forecast.variance.iloc[-1].values[0]
            forecast_vol = np.sqrt(forecast_var) / 100  # Rescale

            garch_vol_pred.append(forecast_vol)

        except Exception as e:
            print(f"Error {symbol} at {i}: {e}")
            garch_vol_pred.append(np.nan)

    # Save results
    df_result = df_filtered.iloc[1:].copy()
    df_result[f"{garch_type}_Vol"] = garch_vol_pred
    df_result[f"{garch_type}_Mean"] = 0
    df_result["Symbol"] = symbol

    # Append to file
    if os.path.exists(output_path):
        df_result.to_csv(output_path, mode="a", header=False)
    else:
        df_result.to_csv(output_path)

    print(f"Finished {symbol}, saved progress.")
    return df_result


# %%
# Run in parallel
results = Parallel(n_jobs=-1)(
    delayed(process_symbol)(symbol)
    for symbol in df.index.get_level_values("Symbol").unique()
)

# %%
print(f"Predictions saved to {output_path}")
