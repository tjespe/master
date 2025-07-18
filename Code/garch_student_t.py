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
from shared.loss import crps_student_t
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# %%
df = pd.read_csv(DATA_PATH)

if not "Symbol" in df.columns:
    df["Symbol"] = TEST_ASSET

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Symbol", "Date"])

df["LogReturn"] = (
    df.groupby("Symbol")["Close"].apply(lambda x: np.log(x / x.shift(1))).droplevel(0)
)
df = df[~df["LogReturn"].isnull()]
df["SquaredReturn"] = df["LogReturn"] ** 2

df["Symbol"] = df["Symbol"].str.replace(".O", "")
df: pd.DataFrame = df.set_index(["Date", "Symbol"])
df = df[df.index.get_level_values("Date") >= "1990-01-01"]

# %%
symbols = df.index.get_level_values("Symbol").unique()


def process_symbol(symbol):
    df_filtered = df.xs(symbol, level="Symbol")
    returns_train = df_filtered["LogReturn"].loc[:VALIDATION_TEST_SPLIT] * 100
    returns_test = df_filtered["LogReturn"].loc[VALIDATION_TEST_SPLIT:] * 100
    returns_combined = pd.concat([returns_train, returns_test])

    garch_vol_pred = []
    nu_values = []

    for i in range(len(returns_test)):
        if i % 20 == 0:
            print(f"{symbol}: {i/len(returns_test):.2%} completed", end="\r")
        end = len(returns_train) + i
        returns_sample = returns_combined.iloc[:end]

        try:
            am = arch_model(
                returns_sample, vol="GARCH", p=1, q=1, mean="Zero", dist="t"
            )
            res = am.fit(disp="off")

            nu_value = res.params.get("nu")
            forecast = res.forecast(horizon=1)
            forecast_var = forecast.variance.iloc[-1].values[0]
            forecast_vol = np.sqrt(forecast_var) / 100

            garch_vol_pred.append(forecast_vol)
            nu_values.append(nu_value)
        except Exception as e:
            print(f"Error {symbol} at {i}: {e}")
            garch_vol_pred.append(np.nan)
            nu_values.append(np.nan)

    df_result = df_filtered.loc[df_filtered.index >= VALIDATION_TEST_SPLIT].copy()
    df_result = df_result.reset_index()
    df_result = df_result[["Date", "LogReturn", "SquaredReturn"]]
    df_result["Symbol"] = symbol
    df_result["GARCH_t_Vol"] = garch_vol_pred
    df_result["GARCH_t_Nu"] = nu_values
    df_result["GARCH_t_Mean"] = 0
    return df_result


# %%
results = Parallel(n_jobs=-1)(delayed(process_symbol)(symbol) for symbol in symbols)

# %%
df_test = pd.concat(results, ignore_index=True)

# %%
from tqdm import tqdm

crps_values = Parallel(n_jobs=-1)(
    delayed(crps_student_t)(x, mu, sigma, nu)
    for x, mu, sigma, nu in tqdm(
        zip(
            df_test["LogReturn"],
            df_test["GARCH_t_Mean"],
            df_test["GARCH_t_Vol"],
            df_test["GARCH_t_Nu"],
        ),
        total=len(df_test),
        desc="Computing CRPS",
    )
)
df_test["GARCH_t_CRPS"] = crps_values


# %%
def format_cl(cl):
    return f"{100*cl:1f}".rstrip("0").rstrip(".")


confidence_levels = [0.67, 0.90, 0.95, 0.98]
for cl in confidence_levels:
    alpha = 1 - cl
    lb = (
        df_test["GARCH_t_Mean"]
        + t.ppf(alpha / 2, df=df_test["GARCH_t_Nu"]) * df_test["GARCH_t_Vol"]
    )
    ub = (
        df_test["GARCH_t_Mean"]
        + t.ppf(1 - alpha / 2, df=df_test["GARCH_t_Nu"]) * df_test["GARCH_t_Vol"]
    )
    df_test[f"LB_{format_cl(cl)}"] = lb
    df_test[f"UB_{format_cl(cl)}"] = ub

    es_alpha = alpha / 2
    z_p = t.ppf(es_alpha, df=df_test["GARCH_t_Nu"])
    t_pdf_z = t.pdf(z_p, df=df_test["GARCH_t_Nu"])
    es = df_test["GARCH_t_Mean"] - df_test["GARCH_t_Vol"] * (
        (df_test["GARCH_t_Nu"] + z_p**2) / (df_test["GARCH_t_Nu"] - 1)
    ) * (t_pdf_z / es_alpha)
    df_test[f"ES_{format_cl(1 - es_alpha)}"] = es

# %%
df_test.to_csv("predictions/garch_predictions_student_t.csv", index=False)
