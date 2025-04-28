# %%
# Define parameters
import sys
from settings import (
    LOOKBACK_DAYS,
    TEST_ASSET,
    DATA_PATH,
    TRAIN_VALIDATION_SPLIT,
    BASEDIR,
)

# %%
import numpy as np
import pandas as pd
from arch import arch_model
import warnings
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

# %%
if DATA_PATH.startswith(f"{BASEDIR}/data/dow_jones"):
    capire_df = pd.read_csv(
        f"{BASEDIR}/data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv"
    )
    print("✅ Successfully loaded capire data")
# %%
capire_df
# %%
capire_df["Date"] = pd.to_datetime(capire_df["Date"])

# Convert RV and RV_5 to daily persentages
capire_df["RV"] = (capire_df["RV"] / 100) / 252.0
capire_df["RV_5"] = (capire_df["RV_5"] / 100) / 252.0
capire_df

# %%
return_df = pd.read_csv(DATA_PATH)

return_df["Symbol"] = return_df["Symbol"].str.replace(".O", "")

# Ensure the Date column is in datetime format
return_df["Date"] = pd.to_datetime(return_df["Date"])

# Sort the dataframe by both Date and Symbol
return_df = return_df.sort_values(["Symbol", "Date"])

# Calculate log returns for each instrument separately using groupby
return_df["LogReturn"] = (
    return_df.groupby("Symbol")["Close"]
    .apply(lambda x: np.log(x / x.shift(1)))
    .droplevel(0)
)

# Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
return_df = return_df[~return_df["LogReturn"].isnull()]

return_df["SquaredReturn"] = return_df["LogReturn"] ** 2

# Set date and symbol as index
return_df: pd.DataFrame = return_df.set_index(["Date", "Symbol"])
return_df

# %%
# Filter away data before 1990
return_df = return_df[return_df.index.get_level_values("Date") >= "1990-01-01"]
return_df

# %% Merge the two dataframes
print(f"Capire data shape: {capire_df.shape}")
print(f"Return data shape: {return_df.shape}")

# %%
# Merge the two dataframes
df = return_df.merge(capire_df, on=["Date", "Symbol"], how="inner")

# %%
print(f"Merged data shape: {df.shape}")
df


# %% GET ALL UNIQUE SYMBOLS
# Ensure "Symbol" is in the index
if "Symbol" not in df.index.names:
    df = df.set_index(
        ["Date", "Symbol"]
    )  # Make sure "Date" and "Symbol" are multi-indexed

symbols = df.index.get_level_values("Symbol").unique()

# Create an empty DataFrame to store all predictions
df_all_predictions = pd.DataFrame()


# %% FUNCTION: Define Realized GARCH Log-Likelihood Function
def realized_garch_loglik(params, r, rv, return_h=False):
    """
    Computes the negative log-likelihood for a Realized GARCH(1,1) model.

    Parameters:
      - omega, beta, alpha, gamma: volatility dynamics
      - xi, phi, tau1, tau2, sigma_eta: measurement equation
      - return_h: If True, returns variance estimates h instead of log-likelihood.

    The model equations are:
      - h_t = omega + beta * h_{t-1} + alpha * r_{t-1}^2 + gamma * (rv_{t-1} - h_{t-1})
      - log(rv_t) = xi + phi * log(h_t) + tau1 * z_t + tau2 * (z_t^2 - 1) + eta_t, eta_t ~ N(0, sigma_eta^2)

    Returns:
      - Negative log-likelihood if return_h=False
      - Variance estimates h if return_h=True
    """
    omega, beta, alpha, gamma, xi, phi, tau1, tau2, sigma_eta = params
    T = len(r)
    h = np.zeros(T)

    # Initialize variance with sample variance
    h[0] = np.var(r)
    nll = 0.0  # Negative log-likelihood accumulator

    # Iterate through time
    for t in range(1, T):
        h[t] = max(
            1e-6,
            omega
            + beta * h[t - 1]
            + alpha * r[t - 1] ** 2
            + gamma * (rv[t - 1] - h[t - 1]),
        )
        if h[t] <= 0:  # Prevent non-positive variance
            return 1e6

        # Compute standardized residuals
        z_t = r[t] / np.sqrt(h[t])

        # Measurement equation mean
        m_t = xi + phi * np.log(h[t]) + tau1 * z_t + tau2 * (z_t**2 - 1)

        # Log-likelihood contributions
        nll -= norm.logpdf(r[t], loc=0, scale=np.sqrt(h[t]))  # Return equation
        nll -= norm.logpdf(
            np.log(rv[t]), loc=m_t, scale=sigma_eta
        )  # Measurement equation

    if return_h:
        return h  # Return estimated variance values
    return nll  # Return negative log-likelihood


# %% TRAIN & PREDICT REALIZED GARCH FOR ALL SYMBOLS
for symbol in symbols:
    print(f"Processing {symbol}...")

    # Filter data for the current symbol
    df_filtered = df.xs(symbol, level="Symbol")

    # Training & test data
    returns_train = df_filtered["LogReturn"].loc[:TRAIN_VALIDATION_SPLIT] * 100
    realized_vol_train = df_filtered["RV_5"].loc[:TRAIN_VALIDATION_SPLIT]

    returns_test = df_filtered["LogReturn"].loc[TRAIN_VALIDATION_SPLIT:] * 100
    realized_vol_test = df_filtered["RV_5"].loc[TRAIN_VALIDATION_SPLIT:]

    # Combine returns for rolling estimation
    returns_combined = pd.concat([returns_train, returns_test])
    realized_vol_combined = pd.concat([realized_vol_train, realized_vol_test])

    # List to store volatility predictions
    realized_garch_vol_pred = []

    # PERFORM ROLLING FORECAST (This is now fully within the loop)
    for i in range(len(returns_test)):
        print(i)
        if i % 20 == 0:
            print(f"Progress {symbol}: {i/len(returns_test):.2%}", end="\r")
            sys.stdout.flush()  # Force immediate update

        # Update the model with available data
        end = len(returns_train) + i
        returns_sample = returns_combined.iloc[:end]
        realized_vol_sample = realized_vol_combined.iloc[:end]

        # Initial parameter guesses
        init_params = [0.05, 0.8, 0.02, 0.01, 0.0, 0.5, 0.0, 0.0, 0.05]

        # Parameter bounds
        bounds = [
            (1e-6, None),  # omega > 0
            (1e-6, 1),  # beta in (0,1)
            (1e-6, None),  # alpha > 0
            (None, None),  # gamma can be pos/neg
            (None, None),  # xi unbounded
            (None, None),  # phi unbounded
            (None, None),  # tau1 unbounded
            (None, None),  # tau2 unbounded
            (1e-6, None),
        ]  # sigma_eta > 0
        print(f"Initial parameters: {init_params}")
        print(f"Bounds: {bounds}")
        print(f"First 5 returns: {returns_sample.head().values}")
        print(f"First 5 realized volatilities: {realized_vol_sample.head().values}")

        test_loglik = realized_garch_loglik(
            init_params, returns_sample, realized_vol_sample
        )
        print(f"Initial log-likelihood: {test_loglik}")

        print("BEFORE MINIMIZER \n")
        # Optimize parameters using MLE
        res = minimize(
            realized_garch_loglik,
            init_params,
            args=(returns_sample, realized_vol_sample),
            bounds=bounds,
            method="TNC",
            options={"disp": True, "maxiter": 20},
        )
        print("AFTER MINIMIZER")

        # Extract estimated parameters
        omega, beta, alpha, gamma, xi, phi, tau1, tau2, sigma_eta = res.x

        # Compute estimated variances using the optimized parameters
        h_estimated = realized_garch_loglik(
            res.x, returns_sample, realized_vol_sample, return_h=True
        )

        # Predict next step variance using the last estimated variance value
        h_next = (
            omega
            + beta * h_estimated[-1]
            + alpha * returns_sample.iloc[-1] ** 2
            + gamma * (realized_vol_sample.iloc[-1] - h_estimated[-1])
        )
        forecast_vol = np.sqrt(h_next) / 100  # Scale back

        realized_garch_vol_pred.append(forecast_vol)

    # Convert list to numpy array
    realized_garch_vol_pred = np.array(realized_garch_vol_pred)

    # STORE PREDICTIONS FOR CURRENT SYMBOL
    df_validation = df.xs(symbol, level="Symbol").loc[TRAIN_VALIDATION_SPLIT:].copy()
    df_validation["Volatility"] = realized_garch_vol_pred
    df_validation["Mean"] = 0  # Assume mean is 0
    df_validation["Symbol"] = symbol  # Add symbol for tracking

    # Append to the main DataFrame
    df_all_predictions = pd.concat([df_all_predictions, df_validation])

# %% SAVE ALL PREDICTIONS TO A SINGLE FILE
df_all_predictions.to_csv(
    f"predictions/realized_garch_predictions_all_assets_{LOOKBACK_DAYS}_days.csv"
)

print("✅ All predictions saved successfully!")

# %%
