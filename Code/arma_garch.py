# %%
# Define version paramaters
AR_LAGS = 1
DIST = "normal"  # "normal" or "t"

VERSION = f"AR({AR_LAGS})-GARCH({1},{1})-{DIST}"

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
from scipy.stats import t
from shared.loss import crps_student_t, crps_normal_univariate
from scipy.stats import norm
from shared.mdn import calculate_es_for_quantile
from joblib import Parallel, delayed
from tqdm import tqdm

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
df["Symbol"] = df["Symbol"].str.replace(".O", "")

# Set date and symbol as index
df: pd.DataFrame = df.set_index(["Date", "Symbol"])
df

# %%
# Filter away data before 1990
df = df[df.index.get_level_values("Date") >= "1990-01-01"]
df

# %%
# FIT THE AR GARCH MODEL AND PREDICT
symbols = df.index.get_level_values("Symbol").unique()

# %% FUNCTION: process one symbol


def process_symbol(symbol):
    print(f"Processing {symbol}")
    df_filtered = df.xs(symbol, level="Symbol")

    returns_train = df_filtered["LogReturn"].loc[:VALIDATION_TEST_SPLIT] * 100
    returns_validation = df_filtered["LogReturn"].loc[VALIDATION_TEST_SPLIT:]
    scaled_returns_test = returns_validation * 100

    returns_combined = pd.concat([returns_train, scaled_returns_test])

    garch_vol_pred = []
    garch_mean_pred = []
    nu_values = [] if DIST == "t" else None

    for i in range(len(scaled_returns_test)):
        if i % 20 == 0:
            print(f"Progress {symbol}: {i/len(scaled_returns_test):.2%}", end="\r")

        end = len(returns_train) + i
        returns_sample = returns_combined.iloc[:end]

        am = arch_model(
            returns_sample, vol="GARCH", p=1, q=1, mean="ARX", dist=DIST, lags=AR_LAGS
        )
        res = am.fit(disp="off")

        if DIST == "t":
            nu_value = res.params.get("nu")
            nu_values.append(nu_value)

        forecast = res.forecast(horizon=1)
        forecast_var = forecast.variance.iloc[-1].values[0]
        forecast_vol = np.sqrt(forecast_var) / 100
        forecast_mean = forecast.mean.iloc[-1].values[0] / 100

        garch_vol_pred.append(forecast_vol)
        garch_mean_pred.append(forecast_mean)

    df_validation_symbol = df_filtered.loc[VALIDATION_TEST_SPLIT:].copy()
    df_validation_symbol = df_validation_symbol.reset_index()
    df_validation_symbol = df_validation_symbol[["Date", "LogReturn", "SquaredReturn"]]
    df_validation_symbol["Symbol"] = symbol

    df_validation_symbol["AR_GARCH_Vol"] = np.array(garch_vol_pred)
    if DIST == "t":
        df_validation_symbol["AR_GARCH_Nu"] = np.array(nu_values)
    df_validation_symbol["AR_GARCH_Mean"] = np.zeros_like(garch_vol_pred)

    return df_validation_symbol


# %% RUN IN PARALLEL

results = Parallel(n_jobs=-1)(delayed(process_symbol)(symbol) for symbol in symbols)

# Combine all results
df_validation = pd.concat(results)

# %%
# Compute CRPS for each observation with a progress bar based on what distribution is used
if DIST == "normal":
    crps_values = crps_normal_univariate(
        df_validation["LogReturn"].values.astype(np.float32),
        df_validation["AR_GARCH_Mean"].values.astype(np.float32),
        df_validation["AR_GARCH_Vol"].values.astype(np.float32),
    ).numpy()

elif DIST == "t":
    crps_values = [
        crps_student_t(x, mu, sigma, nu)
        for x, mu, sigma, nu in tqdm(
            zip(
                df_validation["LogReturn"],
                df_validation["AR_GARCH_Mean"],
                df_validation["AR_GARCH_Vol"],
                df_validation["AR_GARCH_Nu"],
            ),
            total=len(df_validation),
            desc="Computing CRPS",
        )
    ]

df_validation["AR_GARCH_CRPS"] = crps_values


# %%
# calculate upper and lower bounds for given quantiles
def format_cl(cl):
    return f"{100*cl:1f}".rstrip("0").rstrip(".")


confidence_levels = [0.67, 0.90, 0.95, 0.98]
for cl in confidence_levels:
    alpha = 1 - cl
    if DIST == "normal":
        lb = (
            df_validation["AR_GARCH_Mean"]
            - norm.ppf(1 - alpha / 2) * df_validation["AR_GARCH_Vol"]
        )
        ub = (
            df_validation["AR_GARCH_Mean"]
            + norm.ppf(1 - alpha / 2) * df_validation["AR_GARCH_Vol"]
        )
        df_validation[f"LB_{format_cl(cl)}"] = lb
        df_validation[f"UB_{format_cl(cl)}"] = ub

        es_alpha = alpha / 2
        df_validation[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
            np.ones_like(df_validation["AR_GARCH_Mean"]).reshape(-1, 1),
            df_validation["AR_GARCH_Mean"].values.reshape(-1, 1),
            df_validation["AR_GARCH_Vol"].values.reshape(-1, 1),
            lb.to_numpy(),
        )

    elif DIST == "t":
        lb = (
            df_validation["AR_GARCH_Mean"]
            + t.ppf(alpha / 2, df=df_validation["AR_GARCH_Nu"])
            * df_validation["AR_GARCH_Vol"]
        )
        ub = (
            df_validation["AR_GARCH_Mean"]
            + t.ppf(1 - alpha / 2, df=df_validation["AR_GARCH_Nu"])
            * df_validation["AR_GARCH_Vol"]
        )
        df_validation[f"LB_{format_cl(cl)}"] = lb
        df_validation[f"UB_{format_cl(cl)}"] = ub

        es_alpha = alpha / 2
        z_p = t.ppf(es_alpha, df=df_validation["AR_GARCH_Nu"])
        t_pdf_z = t.pdf(z_p, df=df_validation["AR_GARCH_Nu"])
        es = df_validation["AR_GARCH_Mean"] - df_validation["AR_GARCH_Vol"] * (
            (df_validation["AR_GARCH_Nu"] + z_p**2) / (df_validation["AR_GARCH_Nu"] - 1)
        ) * (t_pdf_z / es_alpha)
        df_validation[f"ES_{format_cl(1 - es_alpha)}"] = es

# %%
df_validation.to_csv(f"predictions/predictions_{VERSION}.csv")

# %%
