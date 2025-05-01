# File implementing a HAR quantile regression model as a benchmark
# Using lags of avg previous 1, 5, and 22 days for the volatility forecast like Bollerslev et al. (2016)
# Using the RV_5 minute sampling following Bollerselev et al. (2016)
# %%
# Important libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from shared.conf_levels import format_cl

import warnings

warnings.filterwarnings("ignore")

# %%
# Define parameters
from settings import DATA_PATH, TEST_SET, TRAIN_VALIDATION_SPLIT, VALIDATION_TEST_SPLIT

# %%
# Import return data
df = pd.read_csv(DATA_PATH)

# remove all coloumns except: Date, Close, Symbol
df = df[["Date", "Close", "Symbol", "LogReturn"]]

# Ensure the Date column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort the dataframe by both Date and Symbol
df = df.sort_values(["Symbol", "Date"])

# remove .O at the end of all symbols
df["Symbol"] = df["Symbol"].str.replace(".O", "")

# ungroup the dataframe
df = df.reset_index(drop=True)

df


# %%
# Import the capire data with realized volatility
ivol_df = pd.read_csv("data/dow_jones/processed_data/processed_ivol_data.csv")

# sort the dataframe by both Date and Symbol
ivol_df = ivol_df.sort_values(["Symbol", "Date"])

ivol_df

# %%
# Format the ivol data, remove all coloumns expect: Date, Symbol, 10 Day Call IVOL
ivol_df = ivol_df[["Date", "Symbol", "10 Day Call IVOL"]]

# Ensure the Date column is in datetime format
ivol_df["Date"] = pd.to_datetime(ivol_df["Date"])

# transform ivol feature to become daily variance
annual_pct_vol = ivol_df["10 Day Call IVOL"]
annual_vol = annual_pct_vol / 100
annual_var = annual_vol**2
daily_var = annual_var / 252
ivol_df["Variance"] = daily_var

ivol_df
# %%
# Merge the two dataframes
df = pd.merge(df, ivol_df, on=["Date", "Symbol"], how="inner")
df

# %%
# Print the number of entries per Symbol in the new dataframe and the capire dataframe. Print it side by side in a table
df["Symbol"].value_counts().to_frame().join(
    ivol_df["Symbol"].value_counts().to_frame(), lsuffix="_df", rsuffix="_capire"
)


# %%
# create HAR features
def create_har_features(group):
    group = group.copy()
    group["Variance_lag1"] = group["Variance"].shift(1)
    group["Variance_lag5"] = group["Variance"].shift(1).rolling(window=5).mean()
    group["Variance_lag22"] = group["Variance"].shift(1).rolling(window=22).mean()
    return group


# apply feature creation to each group
df = df.groupby("Symbol").apply(create_har_features)
df

# %%
# ungroup the dataframe
df = df.reset_index(drop=True)
df

# %%
# drop rows with NaN values
df = df.dropna()
df

# %%
# define training and validation data
if TEST_SET == "test":
    # use the test data as the validation data
    training_data = df[df["Date"] < VALIDATION_TEST_SPLIT]
    validation_data = df[(df["Date"] >= VALIDATION_TEST_SPLIT)]
elif TEST_SET == "validation":
    # use the validation data as the test data
    training_data = df[df["Date"] < TRAIN_VALIDATION_SPLIT]
    validation_data = df[
        (df["Date"] >= TRAIN_VALIDATION_SPLIT) & (df["Date"] < VALIDATION_TEST_SPLIT)
    ]

validation_data
# %%
# combine training and validation data
combined_data = pd.concat([training_data, validation_data])
# reset the index
combined_data = combined_data.reset_index(drop=True)
combined_data


# %%
def forecast_symbol(symbol):
    print(f"Forecasting for symbol: {symbol}")
    symbol_data = combined_data[combined_data["Symbol"] == symbol].copy()
    symbol_data = symbol_data.reset_index(drop=True)
    training_data_symbol = training_data[training_data["Symbol"] == symbol].copy()
    validation_data_symbol = validation_data[validation_data["Symbol"] == symbol].copy()

    # Define confidence levels and ES levels
    CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.975, 0.98]
    ES_LEVELS = [1 - (1 - cl) / 2 for cl in CONFIDENCE_LEVELS]
    n_es_points = 5

    # Build quantile list
    quantiles = sorted(
        [(1 - cl) / 2 for cl in CONFIDENCE_LEVELS]
        + [(1 + cl) / 2 for cl in CONFIDENCE_LEVELS]
    )
    es_extra_quantiles = set()
    for alpha in ES_LEVELS:
        lower_q = 1 - alpha
        small_qs = np.linspace(0, lower_q, n_es_points + 1)[1:]
        es_extra_quantiles.update(small_qs)

    quantiles = sorted(set(quantiles).union(es_extra_quantiles))

    all_results = []

    # Rolling forecast
    for i in range(len(validation_data_symbol)):
        end = len(training_data_symbol) + i
        sample_data = symbol_data.iloc[:end]

        X = sample_data[["Variance_lag1", "Variance_lag5", "Variance_lag22"]]
        X = sm.add_constant(X)
        y = sample_data["LogReturn"]
        X_pred = symbol_data[["Variance_lag1", "Variance_lag5", "Variance_lag22"]].iloc[
            end : end + 1
        ]
        X_pred = sm.add_constant(X_pred, has_constant="add")

        # Fit quantile models
        quantile_models = {q: sm.QuantReg(y, X).fit(q=q) for q in quantiles}

        # VaR predictions with custom naming
        var_preds = {}
        for cl in CONFIDENCE_LEVELS:
            lb_q = (1 - cl) / 2
            ub_q = (1 + cl) / 2
            lb_pred = quantile_models[lb_q].predict(X_pred).iloc[0]
            ub_pred = quantile_models[ub_q].predict(X_pred).iloc[0]
            var_preds[f"LB_{format_cl(cl)}"] = lb_pred
            var_preds[f"UB_{format_cl(cl)}"] = ub_pred

        # ES predictions
        es_preds = {}
        for alpha in ES_LEVELS:
            lower_q = 1 - alpha
            es_qs = np.linspace(0, lower_q, n_es_points + 1)[1:]
            es_vals = [(quantile_models[q].predict(X_pred).iloc[0]) for q in es_qs]
            es_preds[f"ES_{alpha * 100:.1f}".rstrip("0").rstrip(".")] = np.mean(es_vals)

        # Collect results
        result = {
            "Date": symbol_data.iloc[end]["Date"],
            "Symbol": symbol,
            "Index_in_validation": i,
        }
        result.update(var_preds)
        result.update(es_preds)
        all_results.append(result)

        if i % 20 == 0:
            print(f"Progress: {i/len(validation_data_symbol):.2%}", end="\r")

    print(f"Finished forecasting for symbol: {symbol}")
    return pd.DataFrame(all_results)


# %%
# Make predictions in parallel for all symbols
symbols = df["Symbol"].unique()
results = Parallel(n_jobs=-1)(delayed(forecast_symbol)(symbol) for symbol in symbols)
results_df = pd.concat(results, ignore_index=True)

# %%
# save the dataframe
results_df.to_csv(f"predictions/HAR_IVOL_qreg_{TEST_SET}.csv", index=False)
