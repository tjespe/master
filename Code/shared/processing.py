# %%
# Define parameters (imported from your settings)
from shared.numerical_mixture_moments import numerical_mixture_moments
from shared.loss import mdn_loss_numpy, mdn_loss_tf
from shared.crps import crps_mdn_numpy
from settings import LOOKBACK_DAYS, SUFFIX, TEST_ASSET, DATA_PATH, TRAIN_TEST_SPLIT

RVOL_DATA_PATH = "data/RVOL.csv"
VIX_DATA_PATH = "data/VIX.csv"
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os


def get_lstm_train_test():
    """
    Prepare data for LSTM
    """
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
    # Read RVOL data
    rvol_df = pd.read_csv(RVOL_DATA_PATH)
    rvol_df["Date"] = pd.to_datetime(rvol_df["Date"])
    rvol_df = rvol_df.set_index("Date")
    rvol_df

    # %%
    # Read VIX data
    vix_df = pd.read_csv(VIX_DATA_PATH)
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    vix_df = vix_df.set_index("Date")
    vix_df

    # %%
    # Check if TEST_ASSET is in the data
    if TEST_ASSET not in df.index.get_level_values("Symbol"):
        raise ValueError(f"TEST_ASSET '{TEST_ASSET}' not found in the data")

    # %%
    # Filter away data we don't have RVOL data for
    df = df[df.index.get_level_values("Date") >= rvol_df.index[0]]
    df = df[df.index.get_level_values("Date") <= rvol_df.index[-1]]
    df

    # %%
    # Filter away data we don't have VIX data for
    df = df[df.index.get_level_values("Date") >= vix_df.index[0]]
    df = df[df.index.get_level_values("Date") <= vix_df.index[-1]]
    df

    # %%
    # Add RVOL data to the dataframe
    df = df.join(rvol_df, how="left", rsuffix="_RVOL")
    df

    # %%
    # Add VIX data to the dataframe
    df = df.join(vix_df, how="left", rsuffix="_VIX")

    # %%
    # If we are looking at stocks, enrich with industry codes
    if DATA_PATH == "data/sp500_stocks.csv":
        meta_df = pd.read_csv("data/sp500_stocks_meta.csv")
        meta_df = meta_df.set_index("Symbol")
        df = df.join(meta_df, how="left", rsuffix="_META")

        # Check for nans
        nan_mask = df[["GICS Sector"]].isnull().sum(axis=1).gt(0)
        nan_rows = df[nan_mask]
        nan_rows.groupby(nan_rows.index.get_level_values("Symbol")).count()

    # %%
    # Check for NaN values
    nan_mask = df[["LogReturn", "Close_RVOL", "Close_VIX"]].isnull().sum(axis=1).gt(0)
    df[nan_mask]

    # %%
    # Impute missing RVOL values using the last available value
    df["Close_RVOL"] = df["Close_RVOL"].fillna(method="ffill")
    df[nan_mask]

    # %%
    # Impute missing VIX values using the last available value
    df["Close_VIX"] = df["Close_VIX"].fillna(method="ffill")
    df[nan_mask]

    # %%
    # Add feature: is next day trading day or not
    df["NextDayTradingDay"] = (
        df.index.get_level_values("Date")
        .shift(1, freq="D")
        .isin(df.index.get_level_values("Date"))
    )
    df["NextDayTradingDay"]

    # %%
    # Check for NaN values
    df[df[["LogReturn", "Close_RVOL", "Close_VIX"]].isnull().sum(axis=1).gt(0)]

    # %%
    # Prepare data for LSTM
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # If we have GICS sectors, add them as a one-hot encoded feature
    if "GICS Sector" in df.columns:
        df["GICS Sector"] = df["GICS Sector"].astype("str")
        df["IDY_CODE"] = df["GICS Sector"].astype("category").cat.codes

    # Group by symbol to handle each instrument separately
    for symbol, group in df.groupby(level="Symbol"):
        # Extract the log returns and squared returns for the current group
        returns = group["LogReturn"].values.reshape(-1, 1)

        # Use the log of squared returns as the second input feature, because the
        # NN should output the log of variance
        log_sq_returns = np.log(
            group["SquaredReturn"].values.reshape(-1, 1)
            # For numerical stability: add a small constant to avoid log(0) equal to 0.1% squared
            + (0.1 / 100) ** 2
        )

        # Sign of return to capture the direction of the return
        sign_return = np.sign(returns)

        # Use whether the next day is a trading day or not as a feature
        next_day_trading_day = group["NextDayTradingDay"].values.reshape(-1, 1)

        # Extract realized volatility and transform it to a similar scale
        rvol_annualized = group["Close_RVOL"].values.reshape(-1, 1) / 100
        rvol_daily = rvol_annualized / np.sqrt(252)
        rvol = np.log(rvol_daily**2 + (0.1 / 100) ** 2)

        # Calculate one day change in RVOL
        rvol_change_1d = np.diff(rvol, axis=0, prepend=rvol[0, 0])
        rvol_change_2d = rvol - np.vstack([rvol[:2], rvol[:-2]])
        rvol_change_7d = rvol - np.vstack([rvol[:7], rvol[:-7]])

        # Extract VIX and transform it to a similar scale
        vix_annualized = group["Close_VIX"].values.reshape(-1, 1) / 100
        vix_daily = vix_annualized / np.sqrt(252)
        vix = np.log(vix_daily**2 + (0.1 / 100) ** 2)

        # Calculate one day change in VIX
        vix_change_1d = np.diff(vix, axis=0, prepend=vix[0, 0])
        vix_change_2d = vix - np.vstack([vix[:2], vix[:-2]])
        vix_change_7d = vix - np.vstack([vix[:7], vix[:-7]])

        # Find date to split on
        train_test_split_index = len(
            group[group.index.get_level_values("Date") < TRAIN_TEST_SPLIT]
        )

        # Stack returns and squared returns together
        data = np.hstack(
            (
                log_sq_returns,
                sign_return,
                rvol,
                rvol_change_1d,
                rvol_change_2d,
                rvol_change_7d,
                vix_change_1d,
                vix_change_2d,
                vix_change_7d,
                next_day_trading_day,
            )
        )

        # If we have GICS sectors, add them as a feature
        if "IDY_CODE" in group.columns:
            one_hot_sector = np.zeros((len(group), len(df["IDY_CODE"].unique())))
            one_hot_sector[np.arange(len(group)), group["IDY_CODE"]] = 1
            data = np.hstack((data, one_hot_sector))

        # Create training sequences of length 'sequence_length'
        for i in range(LOOKBACK_DAYS, train_test_split_index):
            X_train.append(data[i - LOOKBACK_DAYS : i])
            y_train.append(returns[i, 0])

        # Add the test data
        if symbol == TEST_ASSET:
            for i in range(train_test_split_index, len(data)):
                X_test.append(data[i - LOOKBACK_DAYS : i])
                y_test.append(returns[i, 0])

    # Convert X and y to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Ensure float32 for X and y
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Print shapes
    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"y_test.shape: {y_test.shape}")

    return df, X_train, X_test, y_train, y_test
