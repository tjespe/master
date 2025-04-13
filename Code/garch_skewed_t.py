# Using the skewed student-t distribution for GARCH model proposed by Hansen (1994)

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
from arch.univariate.distribution import SkewStudent  # uses Hansen skew-t
from shared.loss import crps_skewed_t
warnings.filterwarnings("ignore")
from tqdm import tqdm

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
garch_vol_pred = []    # will hold forecasted volatility
nu_values = []         # degrees of freedom estimates
skew_values = []       # skewness parameter estimates
for symbol in symbols:
    print(f"Processing {symbol}")
    df_filtered = df.xs(symbol, level="Symbol")
    
    # Define training and test periods
    returns_train = df_filtered["LogReturn"].loc[:VALIDATION_TEST_SPLIT]
    returns_train = returns_train * 100  # Scale to percentages for numerical stability
    
    returns_test = df_filtered["LogReturn"].loc[VALIDATION_TEST_SPLIT:]
    scaled_returns_test = returns_test * 100
    
    # Combine the training and test periods
    returns_combined = pd.concat([returns_train, scaled_returns_test])
    
    # Rolling forecast over the test period:
    for i in range(len(scaled_returns_test)):
        if i % 20 == 0:
            print(f"Progress: {i/len(scaled_returns_test):.2%}", end="\r")
        
        # Update window end point: training + i observations
        end = len(returns_train) + i
        returns_sample = returns_combined.iloc[:end]
        
        # Fit the GARCH(1,1) model with skewed t distribution:
        am = arch_model(returns_sample, vol="GARCH", p=1, q=1, mean="Zero", dist="skewt")
        res = am.fit(disp="off")
        
        # Extract the degrees-of-freedom (nu) and skew parameter ("lambda")
        nu_value = res.params.get("nu")
        skew_value = res.params.get("lambda")
        nu_values.append(nu_value)
        skew_values.append(skew_value)
        
        # Forecast variance for the next step and convert back to original scaling
        forecast = res.forecast(horizon=1)
        forecast_var = forecast.variance.iloc[-1].values[0]
        forecast_vol = np.sqrt(forecast_var) / 100
        garch_vol_pred.append(forecast_vol)

# Convert lists to numpy arrays
garch_vol_pred = np.array(garch_vol_pred)
nu_values = np.array(nu_values)
skew_values = np.array(skew_values)

# %%
# Save GARCH predictions to file
df_test = df[df.index.get_level_values("Date") >= VALIDATION_TEST_SPLIT]
df_test.reset_index(inplace=True)
# remove all coloumns except Symbol, Date, LogReturn, SquaredReturn
df_test = df_test[["Symbol", "Date", "LogReturn", "SquaredReturn"]]
df_test
# Append the forecasted volatility and estimated parameters
df_test["GARCH_skewt_Vol"] = garch_vol_pred
df_test["GARCH_skewt_Nu"] = nu_values
df_test["GARCH_skewt_Skew"] = skew_values
df_test["GARCH_skewt_Mean"] = 0  # using the zero-mean assumption

# %%
crps_values = [
    crps_skewed_t(x, mu, sigma, nu, lam)
    for x, mu, sigma, nu, lam in tqdm(
        zip(
            df_test["LogReturn"],
            df_test["GARCH_skewt_Mean"],
            df_test["GARCH_skewt_Vol"],
            df_test["GARCH_skewt_Nu"],
            df_test["GARCH_skewt_Skew"],
        ),
        total=len(df_test),
        desc="Computing CRPS"
    )
]

df_test["GARCH_skewt_CRPS"] = crps_values
df_test

# %%
# calculate upper and lower bounds for given quantiles
def format_cl(cl):
    return f"{100*cl:1f}".rstrip("0").rstrip(".")

confidence_levels = [0.67, 0.90, 0.95, 0.98]
# For each confidence level, compute lower/upper bounds and expected shortfall using simulation
for cl in confidence_levels:
    alpha = 1 - cl
    lower_quantiles = []
    upper_quantiles = []
    es_values = []
    for mu, sigma, nu, lam in zip(
        df_test["GARCH_skewt_Mean"],
        df_test["GARCH_skewt_Vol"],
        df_test["GARCH_skewt_Nu"],
        df_test["GARCH_skewt_Skew"]
    ):
        skewt = SkewStudent()
        # Calculate lower and upper quantiles using the skewed t percent point function (ppf)
        lb = mu + sigma * skewt.ppf(alpha / 2, nu=nu, lam=lam)
        ub = mu + sigma * skewt.ppf(1 - alpha / 2, nu=nu, lam=lam)
        lower_quantiles.append(lb)
        upper_quantiles.append(ub)
        
        # Estimate Expected Shortfall (ES) via Monte Carlo:
        nsim_es = 1000  # number of draws to estimate ES
        z_samples = skewt.rvs(nsim_es, nu=nu, lam=lam)
        samples = mu + sigma * z_samples
        # VaR for the lower tail
        var_level = np.percentile(samples, (alpha / 2) * 100)
        es = np.mean(samples[samples <= var_level])
        es_values.append(es)
    
    df_test[f"LB_{format_cl(cl)}"] = lower_quantiles
    df_test[f"UB_{format_cl(cl)}"] = upper_quantiles
    df_test[f"ES_{format_cl(1 - alpha/2)}"] = es_values


df_test
# %%
df_test.to_csv("predictions/garch_predictions_skewed_t.csv")


# %%
