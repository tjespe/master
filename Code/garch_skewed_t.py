# Using the skewed student-t distribution for GARCH model proposed by Hansen (1994)

# %%
# Define parameters
from settings import (
    TEST_ASSET,
    DATA_PATH,
    VALIDATION_TEST_SPLIT,
)

# %%
import numpy as np
import pandas as pd
from arch import arch_model
import warnings
from arch.univariate.distribution import SkewStudent
from shared.loss import crps_skewt
from tqdm import tqdm
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
    skew_values = []

    for i in range(len(returns_test)):
        if i % 20 == 0:
            print(f"{symbol}: {i/len(returns_test):.2%} completed", end="\r")
        end = len(returns_train) + i
        returns_sample = returns_combined.iloc[:end]

        try:
            am = arch_model(
                returns_sample, vol="GARCH", p=1, q=1, mean="Zero", dist="skewt"
            )
            res = am.fit(disp="off")

            nu_value = res.params.get("eta")  # nu is called eta
            skew_value = res.params.get("lambda")

            forecast = res.forecast(horizon=1)
            forecast_var = forecast.variance.iloc[-1].values[0]
            forecast_vol = np.sqrt(forecast_var) / 100

            garch_vol_pred.append(forecast_vol)
            nu_values.append(nu_value)
            skew_values.append(skew_value)
        except Exception as e:
            print(f"Error {symbol} at {i}: {e}")
            garch_vol_pred.append(np.nan)
            nu_values.append(np.nan)
            skew_values.append(np.nan)

    df_result = df_filtered.loc[df_filtered.index >= VALIDATION_TEST_SPLIT].copy()
    df_result = df_result.reset_index()
    df_result = df_result[["Date", "LogReturn", "SquaredReturn"]]
    df_result["Symbol"] = symbol
    df_result["GARCH_skewt_Vol"] = garch_vol_pred
    df_result["GARCH_skewt_Nu"] = nu_values
    df_result["GARCH_skewt_Skew"] = skew_values
    df_result["GARCH_skewt_Mean"] = 0
    return df_result


# %%
results = Parallel(n_jobs=-1)(delayed(process_symbol)(symbol) for symbol in symbols)

# %%
df_test = pd.concat(results, ignore_index=True)


# %%
def safe_crps(x, mu, sigma, nu, lam):
    def is_real_number(v):
        return isinstance(v, (int, float)) and np.isfinite(v)

    if is_real_number(nu) and is_real_number(lam):
        return crps_skewt(x, mu, sigma, nu, lam)
    else:
        return np.nan


crps_values = Parallel(n_jobs=-1)(
    delayed(safe_crps)(x, mu, sigma, nu, lam)
    for x, mu, sigma, nu, lam in tqdm(
        zip(
            df_test["LogReturn"],
            df_test["GARCH_skewt_Mean"],
            df_test["GARCH_skewt_Vol"],
            df_test["GARCH_skewt_Nu"],
            df_test["GARCH_skewt_Skew"],
        ),
        total=len(df_test),
        desc="Computing CRPS",
    )
)
df_test["GARCH_skewt_CRPS"] = crps_values


# %%
def format_cl(cl):
    return f"{100*cl:1f}".rstrip("0").rstrip(".")


skewt = SkewStudent()


def estimate_quantile(mu, sigma, nu, lam, q):
    return mu + sigma * np.array(
        [skewt.ppf(q, parameters=[nu, lam]) for nu, lam in zip(nu, lam)]
    )


confidence_levels = (
    # The ones used in the paper
    [0.67, 0.90, 0.95, 0.98]
    +
    # Not used in the paper but are useful for testing
    [0.99, 0.995]
)
# For each confidence level, compute lower/upper bounds and expected shortfall using simulation
for cl in confidence_levels:
    alpha = (1 - cl) / 2

    lb = estimate_quantile(
        df_test["GARCH_skewt_Mean"],
        df_test["GARCH_skewt_Vol"],
        df_test["GARCH_skewt_Nu"],
        df_test["GARCH_skewt_Skew"],
        alpha,
    )

    ub = estimate_quantile(
        df_test["GARCH_skewt_Mean"],
        df_test["GARCH_skewt_Vol"],
        df_test["GARCH_skewt_Nu"],
        df_test["GARCH_skewt_Skew"],
        1 - alpha,
    )

    # To calculate ES, estimate several quantiles below the lower bound and take the mean
    n_quantiles = 100
    quantiles = np.linspace(0, alpha, n_quantiles + 1)[1:-1]  # Exclude 0 and alpha

    quantile_values = np.array(
        Parallel(n_jobs=-1)(
            delayed(estimate_quantile)(
                df_test["GARCH_skewt_Mean"],
                df_test["GARCH_skewt_Vol"],
                df_test["GARCH_skewt_Nu"],
                df_test["GARCH_skewt_Skew"],
                q,
            )
            for q in quantiles
        )
    )

    # Add lb to quantile_values
    quantile_values = np.array(lb)[np.newaxis, :] + quantile_values

    # Calculate ES as the mean of the quantiles
    es_values = np.mean(quantile_values, axis=0)

    assert (
        es_values < lb
    ).all(), f"ES breaches lower bound for CL {cl} (ES {format_cl(1-alpha)})"

    df_test[f"LB_{format_cl(cl)}"] = lb
    df_test[f"UB_{format_cl(cl)}"] = ub
    df_test[f"ES_{format_cl(1 - alpha)}"] = es_values

# %%
# Sense-check the results
for cl in confidence_levels:
    VaR = df_test[f"LB_{format_cl(cl)}"]
    alpha = (1 - cl) / 2
    ES = df_test[f"ES_{format_cl(1 - alpha)}"]
    breaches = df_test[ES > VaR]
    assert (
        len(breaches) == 0
    ), f"ES breaches VaR for {len(breaches)} samples at {cl}: {breaches}"

# %%
# Make a little plot to evaluate correctness
df_test.set_index(["Date", "Symbol"]).xs("AAPL", level="Symbol").sort_index()[
    ["LB_90", "ES_95", "LB_98", "ES_99"]
].rename(
    columns={
        "LB_90": "95% VaR",
        "ES_95": "95% ES",
        "LB_98": "99% VaR",
        "ES_99": "99% ES",
    }
).plot(
    title="99% VaR and ES for AAPL",
    # Color appropriately
    color=["#ffaaaa", "#ff0000", "#aaaaff", "#0000ff"],
    figsize=(12, 6),
)

# %%
df_test.to_csv("predictions/garch_predictions_skewed_t.csv", index=False)
