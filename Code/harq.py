# File implementing the HARQ model as a benchmark
# Using lags of avg previous 1, 5, and 22 days for the volatility forecast like Bollerslev et al. (2016)
# Using the RV_5 minute sampling following Bollerselev et al. (2016)
# %%
# Important libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from joblib import Parallel, delayed
from shared.conf_levels import format_cl
from scipy.stats import norm
from shared.mdn import calculate_es_for_quantile


# %%
# Define parameters
from settings import DATA_PATH, TEST_SET, TRAIN_VALIDATION_SPLIT, VALIDATION_TEST_SPLIT

# %%
# Import return data
df = pd.read_csv(DATA_PATH)

# remove all coloumns except: Date, Close, Symbol, Total Return
df = df[["Date", "Close", "Symbol", "Total Return", "LogReturn"]]

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
capire_df = pd.read_csv(
    "data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv"
)

# sort the dataframe by both Date and Symbol
capire_df = capire_df.sort_values(["Symbol", "Date"])

capire_df

# %%
# Format the capire data, remove all coloumns expect: Date, Symbol, RV_5
capire_df = capire_df[["Date", "Symbol", "RV_5", "RQ_5"]]

# Ensure the Date column is in datetime format
capire_df["Date"] = pd.to_datetime(capire_df["Date"])

# transform the RV to become log_daily_rv
capire_df["RV"] = (capire_df["RV_5"] / 100) / 252.0

# transform the RQ to become log_daily_rq
capire_df["RQ"] = capire_df["RQ_5"] / (
    100.0**4
)  # annual percent^4 --> annual decimal^4
capire_df["RQ"] = capire_df["RQ"] / (
    252.0**2
)  # annual decimal^4 / 252^2 --> daily decimal^4
capire_df
# %%
# Merge the two dataframes
df = pd.merge(df, capire_df, on=["Date", "Symbol"], how="inner")
df

# %%
# Print the number of entries per Symbol in the new dataframe and the capire dataframe. Print it side by side in a table
df["Symbol"].value_counts().to_frame().join(
    capire_df["Symbol"].value_counts().to_frame(), lsuffix="_df", rsuffix="_capire"
)


# %%
# create HAR features
def create_harq_features(group):
    group = group.copy()
    group["RV_lag1"] = group["RV"].shift(1)
    group["RV_lag5"] = group["RV"].shift(1).rolling(window=5).mean()
    group["RV_lag22"] = group["RV"].shift(1).rolling(window=22).mean()
    group["RQ_lag1"] = group["RQ"].shift(1)
    return group


# apply feature creation to each group
df = df.groupby("Symbol").apply(create_harq_features)
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
# scale to percentages
df["RV"] = df["RV"]
df["RV_lag1"] = df["RV_lag1"]
df["RV_lag5"] = df["RV_lag5"]
df["RV_lag22"] = df["RV_lag22"]
df["RQ"] = df["RQ"]
df["RQ_lag1"] = df["RQ_lag1"]

# create the RV lag1 * RQ lag1
df["RV_lag1_RQ_lag1"] = df["RV_lag1"] * np.sqrt(df["RQ_lag1"])


# %%
# define training and validation data
if TEST_SET == "validation":
    training_data = df[df["Date"] < TRAIN_VALIDATION_SPLIT]
    validation_data = df[
        (df["Date"] >= TRAIN_VALIDATION_SPLIT) & (df["Date"] < VALIDATION_TEST_SPLIT)
    ]
elif TEST_SET == "test":
    # use the test data as the validation data
    training_data = df[df["Date"] < VALIDATION_TEST_SPLIT]
    validation_data = df[(df["Date"] >= VALIDATION_TEST_SPLIT)]

validation_data
# %%
# define empty result list
volatality_preds = []

# combine training and validation data
combined_data = pd.concat([training_data, validation_data])
# reset the index
combined_data = combined_data.reset_index(drop=True)
combined_data
# %%
# loop over the symbols and make predictions
symbols = df["Symbol"].unique()
symbols


# %%
def forecast_symbol(symbol):
    print(f"Forecasting for symbol: {symbol}")
    symbol_data = combined_data[combined_data["Symbol"] == symbol].copy()
    symbol_data = symbol_data.reset_index(drop=True)
    training_data_symbol = training_data[training_data["Symbol"] == symbol].copy()
    validation_data_symbol = validation_data[validation_data["Symbol"] == symbol].copy()

    volatality_preds = []

    # perform rolling forecast
    for i in range(len(validation_data_symbol)):
        # update the model with data up to the current date
        end = len(training_data_symbol) + i  # the point we would like to predict
        # print(f"End: {end}")
        sample_data = symbol_data.iloc[
            :end
        ]  # train on all data up to the point we would like to predict

        # independent variables
        X = sample_data[["RV_lag1", "RV_lag5", "RV_lag22", "RV_lag1_RQ_lag1"]]
        X = sm.add_constant(X)

        # dependent variable
        y = sample_data["RV"]

        # fit the OLS model
        model = sm.OLS(y, X).fit()

        # make the prediction data
        # print(symbol_data.iloc[end:end+1])
        X_pred = symbol_data[
            ["RV_lag1", "RV_lag5", "RV_lag22", "RV_lag1_RQ_lag1"]
        ].iloc[end : end + 1]
        X_pred = sm.add_constant(X_pred, has_constant="add")

        # print(X_pred)
        # print("X columns:", X.columns.tolist())
        # print("X_pred columns:", X_pred.columns.tolist())
        # print("Model params index:", model.params.index.tolist())
        # forecast the next time point
        forecast_var = model.predict(X_pred)  # forecast the next time point
        # scale down
        forecast_var = forecast_var

        # print(forecast_var)
        # print(f"Forecast: {forecast_var.iloc[0]}")
        # print(f"Forecast vol: {np.sqrt(forecast_var.iloc[0])}")

        volatality_preds.append(np.sqrt(forecast_var.iloc[0]))

        if i % 20 == 0:
            print(f"Progress {symbol}: {i/len(validation_data_symbol):.2%}", end="\r")

    validation_data_symbol["HARQ_vol_python"] = volatality_preds
    validation_data_symbol["Mean"] = 0  # Assume mean is 0
    return validation_data_symbol


# %%
# Make predictions in parallel for all symbols
results = Parallel(n_jobs=-1)(
    delayed(forecast_symbol)(symbol) for symbol in tqdm(symbols)
)
results_df = pd.concat(results, ignore_index=True)

# %%
# save the dataframe
results_df.to_csv("predictions/HARQ_python.csv", index=False)


# %%
# load the dataframe
results_df = pd.read_csv("predictions/HARQ_python.csv")

# %%
# Add VaR and ES estimates assuming normality
CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.98]
ALL_CONFIDENCE_LEVELS = CONFIDENCE_LEVELS + [0.99, 0.995]
mu = results_df["Mean"].values
har_vol_pred = results_df["HARQ_vol_python"].values
for cl in ALL_CONFIDENCE_LEVELS:
    alpha = 1 - cl
    z_alpha = norm.ppf(1 - alpha / 2)
    lb = mu - z_alpha * har_vol_pred
    ub = mu + z_alpha * har_vol_pred
    results_df[f"LB_{format_cl(cl)}"] = lb
    results_df[f"UB_{format_cl(cl)}"] = ub
    es_alpha = alpha / 2
    results_df[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
        np.ones_like(mu).reshape(-1, 1),
        mu.reshape(-1, 1),
        har_vol_pred.reshape(-1, 1),
        lb,
    )

# %%
# Save the dataframe with the new columns
results_df.to_csv("predictions/HARQ_python.csv", index=False)
