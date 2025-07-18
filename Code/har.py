# File implementing the HAR model as a benchmark
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

import warnings

warnings.filterwarnings("ignore")

# %%
# Define parameters
from settings import (
    LOOKBACK_DAYS,
    TEST_ASSET,
    DATA_PATH,
    TEST_SET,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)

# %%
# Import return data
df = pd.read_csv(DATA_PATH)

# remove all coloumns except: Date, Close, Symbol, Total Return
df = df[["Date", "Close", "Symbol", "Total Return", "LogReturn"]]
df["Total Return"] = df["Total Return"] / 100

# Ensure the Date column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort the dataframe by both Date and Symbol
df = df.sort_values(["Symbol", "Date"])

# remove .O at the end of all symbols
df["Symbol"] = df["Symbol"].str.replace(".O", "")

df["Total Return Test"] = (
    df.groupby("Symbol")["Close"].apply(lambda x: (x / x.shift(1)) - 1).droplevel(0)
)

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
capire_df = capire_df[["Date", "Symbol", "RV_5"]]

# Ensure the Date column is in datetime format
capire_df["Date"] = pd.to_datetime(capire_df["Date"])

# transform the RV to become daily_rv
capire_df["RV"] = (
    capire_df["RV_5"] / 100
) / 252.0  # annual percentage^2 --> daily decimal^2

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
def create_har_features(group):
    group = group.copy()
    group["RV_lag1"] = group["RV"].shift(1)
    group["RV_lag5"] = group["RV"].shift(1).rolling(window=5).mean()
    group["RV_lag22"] = group["RV"].shift(1).rolling(window=22).mean()
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

    # perform rolling forecast
    for i in range(len(validation_data_symbol)):
        # update the model with data up to the current date
        end = len(training_data_symbol) + i  # the point we would like to predict
        # print(f"End: {end}")
        sample_data = symbol_data.iloc[
            :end
        ]  # train on all data up to the point we would like to predict

        # independent variables
        X = sample_data[["RV_lag1", "RV_lag5", "RV_lag22"]]
        X = sm.add_constant(X)

        # dependent variable
        y = sample_data["RV"]

        # fit the OLS model
        model = sm.OLS(y, X).fit()

        # make the prediction data
        # print(symbol_data.iloc[end:end+1])
        X_pred = symbol_data[["RV_lag1", "RV_lag5", "RV_lag22"]].iloc[end : end + 1]
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
            print(f"Progress: {i/len(validation_data_symbol):.2%}", end="\r")

    print(f"Finished forecasting for symbol: {symbol}")
    validation_data_symbol["HAR_vol_python"] = volatality_preds[
        -len(validation_data_symbol) :
    ]
    return validation_data_symbol


# %%
# Make predictions in parallel for all symbols
results = Parallel(n_jobs=-1)(
    delayed(forecast_symbol)(symbol) for symbol in tqdm(symbols)
)
results_df = pd.concat(results, ignore_index=True)


# %%
# add the predictions to the dataframe
results_df["Mean"] = 0  # Assume mean is 0

# %%
# save the dataframe
results_df.to_csv("predictions/HAR_python.csv", index=False)

# %%
# load the dataframe
results_df = pd.read_csv("predictions/HAR_python.csv")

# %%
# Add VaR and ES estimates assuming normality
CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.98]
ALL_CONFIDENCE_LEVELS = CONFIDENCE_LEVELS + [0.99, 0.995]
mu = results_df["Mean"].values
har_vol_pred = results_df["HAR_vol_python"].values
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
results_df.to_csv("predictions/HAR_python.csv", index=False)

# %%
# plot som example distributions
import seaborn as sns
import matplotlib.pyplot as plt

# plot  3 different plots of the Total Return Test for the validation data over time for the first 3 symbols
for symbol in results_df["Symbol"].unique()[:3]:
    symbol_data = results_df[results_df["Symbol"] == symbol]
    sns.lineplot(x="Date", y="Total Return Test", data=symbol_data)
    # plot the mean prediction
    sns.lineplot(x="Date", y="Mean", data=symbol_data)
    # plot two standard deviations
    plt.fill_between(
        symbol_data["Date"],
        symbol_data["Mean"] - 2 * symbol_data["HAR_vol_python"],
        symbol_data["Mean"] + 2 * symbol_data["HAR_vol_python"],
        alpha=0.5,
        color="purple",
    )
    # plot the volatility of the mean as the standard deviation
    plt.fill_between(
        symbol_data["Date"],
        symbol_data["Mean"] - symbol_data["HAR_vol_python"],
        symbol_data["Mean"] + symbol_data["HAR_vol_python"],
        alpha=0.6,
        color="red",
    )
    # plot legends based on the colors
    plt.legend(
        ["Total Return Test", "Mean", "2 Standard Deviations", "Volatility of the Mean"]
    )

    plt.title(f"Total Return Test for {symbol}")
    plt.show()

# %%
# plot the total return test vs the total return over time for the first 3 symbols
for symbol in results_df["Symbol"].unique()[:3]:
    symbol_data = results_df[results_df["Symbol"] == symbol]
    sns.lineplot(x="Date", y="Total Return", data=symbol_data, color="blue")
    sns.lineplot(x="Date", y="Total Return Test", data=symbol_data, color="red")
    plt.legend(["Total Return Test", "Total Return"])
    plt.title(f"Total Return Test vs Total Return for {symbol}")
    plt.show()


# %%
# check how simiar the Total Return Test and Total Return are
results_df["Total Return Test"].corr(results_df["Total Return"])
# %%
# plot acutal RV_5 vs predicted HAR_var
sns.scatterplot(x="RV_5", y="HAR_vol_python", data=results_df)
plt.title("Actual RV_5 vs Predicted vol")
plt.show()
# %%
print(np.log(0.01))
print(np.log(1.01))
# %%
