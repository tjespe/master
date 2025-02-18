# %%
# Define parameters (imported from your settings)
from dataclasses import dataclass
from functools import cached_property
from collections import OrderedDict
from settings import (
    LOOKBACK_DAYS,
    TEST_ASSET,
    DATA_PATH,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)

RVOL_DATA_PATH = "data/RVOL.csv"
VIX_DATA_PATH = "data/VIX.csv"
# %%
import numpy as np
import pandas as pd
import gc

# %%
# Define consistent sector mapping for GICS
GICS_SECTOR_MAPPING = {
    "Communication Services": 0,
    "Consumer Discretionary": 1,
    "Consumer Staples": 2,
    "Energy": 3,
    "Financials": 4,
    "Health Care": 5,
    "Industrials": 6,
    "Information Technology": 7,
    "Materials": 8,
    "Real Estate": 9,
    "Utilities": 10,
}


@dataclass
class LabelledTestSet:
    ticker: str
    df: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    y_dates: list[str]


@dataclass
class ProcessedData:
    df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    validation_sets: OrderedDict[str, LabelledTestSet]
    test_sets: OrderedDict[str, LabelledTestSet]

    @cached_property
    def X_val_combined(self) -> np.ndarray:
        return np.concatenate([val_set.X for val_set in self.validation_sets.values()])

    @cached_property
    def y_val_combined(self) -> np.ndarray:
        return np.concatenate([val_set.y for val_set in self.validation_sets.values()])

    @cached_property
    def validation_tickers(self) -> list[str]:
        return [t for s in self.validation_sets.values() for t in [s.ticker] * len(s.y)]

    @cached_property
    def validation_dates(self) -> list[pd.Timestamp]:
        return [d for s in self.validation_sets.values() for d in s.y_dates]

    def get_validation_range(self, ticker: str):
        from_idx = self.validation_tickers.index(ticker)
        to_idx = next(
            (
                i
                for i in range(from_idx + 1, len(self.validation_tickers))
                if self.validation_tickers[i] != ticker
            ),
        )
        return from_idx, to_idx

    def __str__(self):
        return f"ProcessedData(X_train.shape={self.X_train.shape}, y_train.shape={self.y_train.shape}, X_val_combined.shape={self.X_val_combined.shape}, y_val_combined.shape={self.y_val_combined.shape}, validation_sets={len(self.validation_sets)}, test_sets={len(self.test_sets)})"


def get_lstm_train_test_new(multiply_by_beta=False) -> ProcessedData:
    """
    Prepare data for LSTM
    """
    # %%
    df = pd.read_csv(DATA_PATH)

    if not "Symbol" in df.columns:
        df["Symbol"] = TEST_ASSET

    # Ensure the Date column is in datetime format
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # Sort the dataframe by both Date and Symbol
    df.sort_values(["Symbol", "Date"], inplace=True)
    df

    # %%
    # Join in the S&P 500 data
    spx_df = pd.read_csv("data/spx.csv")
    spx_df["Date"] = pd.to_datetime(spx_df["Date"]).dt.date
    spx_df.set_index("Date", inplace=True)
    df[["Close_SPX"]] = spx_df[["Close"]].loc[df["Date"].values].values
    df[["Close", "Close_SPX"]]

    # %%
    # Calculate log returns for each instrument separately using groupby
    df.sort_values(["Symbol", "Date"], inplace=True)
    df["LogReturn"] = (
        df.groupby("Symbol")["Close"]
        .apply(lambda x: np.log(x / x.shift(1)))
        .droplevel(0)
    )
    df["PctReturn"] = df.groupby("Symbol")["Close"].pct_change()
    df["MonthlyReturn"] = (
        df.groupby("Symbol")["Close"].apply(lambda x: x / x.shift(21) - 1).droplevel(0)
    )
    df["SPX_PctReturn"] = df.groupby("Symbol")["Close_SPX"].pct_change()
    df["SPX_LogReturn"] = np.log(df["Close_SPX"] / df["Close_SPX"].shift(1))
    df["SPX_MonthlyReturn"] = df["Close_SPX"] / df["Close_SPX"].shift(21) - 1

    # Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
    df = df[~df["LogReturn"].isnull()]

    df["SquaredReturn"] = df["LogReturn"] ** 2

    # Set date and symbol as index
    df.set_index(["Date", "Symbol"], inplace=True)
    df[
        [
            "Close",
            "LogReturn",
            "PctReturn",
            "MonthlyReturn",
            "Close_SPX",
            "SPX_LogReturn",
            "SPX_PctReturn",
            "SPX_MonthlyReturn",
        ]
    ]

    # %%
    # Compute rolling beta for each stock
    def compute_beta(g, window=251):
        g = g.reset_index().set_index("Date")
        rolling_df = g[["PctReturn", "SPX_PctReturn"]].rolling(
            window, min_periods=window
        )
        cov = rolling_df.cov().unstack()
        beta = (
            cov[("PctReturn", "SPX_PctReturn")]
            / cov[("SPX_PctReturn", "SPX_PctReturn")]
        )
        g["Beta"] = beta
        g = g.reset_index().set_index(["Date", "Symbol"])
        return g

    df = df.groupby("Symbol").apply(lambda g: compute_beta(g)).droplevel(0)
    df["Beta"]

    # %%
    # Add more features:
    # Downside Volatility
    df["DownsideVol"] = (
        df.groupby(level="Symbol")["LogReturn"]
        .apply(lambda x: x.where(x < 0).rolling(20, min_periods=1).std())
        .droplevel(0)
    )
    df[["Close", "LogReturn", "DownsideVol"]]

    # %%
    # If we have GARCH predictions, calculate skewness and kurtosis based on GARCH residuals
    if "GARCH_Vol" in df.columns:
        df["GARCH_Resid"] = df["LogReturn"] / df["GARCH_Vol"]

        # Apply to each Symbol group
        ewm_stats = df.groupby(level="Symbol")["GARCH_Resid"].apply(
            compute_ewm_skew_kurt
        )

        # Remove extra level in the multi-index
        ewm_stats = ewm_stats.droplevel(0)

        # `ewm_stats` now has a multi-index: (Symbol, Date).
        # We can join it back to df (which is indexed by (Date, Symbol) as well) directly:
        df = df.join(ewm_stats)

        # Drop first 20 rows for each instrument
        df = df.groupby("Symbol").apply(lambda x: x.iloc[20:])

        # Remove extra level in the multi-index
        df = df.droplevel(0)
        df

    # %%
    # Read RVOL data
    rvol_df = pd.read_csv(RVOL_DATA_PATH)
    rvol_df["Date"] = pd.to_datetime(rvol_df["Date"]).dt.date
    rvol_df = rvol_df.set_index("Date")
    rvol_df["RVOL_Std"] = rvol_df["Close"].rolling(10).std()
    rvol_df = rvol_df.dropna()
    rvol_df

    # %%
    # Read VIX data
    vix_df = pd.read_csv(VIX_DATA_PATH)
    vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.date
    vix_df = vix_df.set_index("Date")
    vix_df

    # %%
    # Read Fear & Greed Index data
    fng_df = pd.read_csv("data/fear-greed.csv")
    fng_df["Date"] = pd.to_datetime(fng_df["Date"]).dt.date
    fng_df = fng_df.set_index("Date")
    fng_df

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
    # Filter away data we don't have Fear & Greed Index data for
    df = df[df.index.get_level_values("Date") >= fng_df.index[0]]
    df = df[df.index.get_level_values("Date") <= fng_df.index[-1]]
    df

    # %%
    # Add RVOL data to the dataframe
    df = df.join(rvol_df, how="left", rsuffix="_RVOL")
    df

    # %%
    # Add VIX data to the dataframe
    df = df.join(vix_df, how="left", rsuffix="_VIX")

    # %%
    # Add Fear & Greed Index data to the dataframe
    df = df.join(fng_df, how="left")
    df[["LogReturn", "Fear Greed"]]

    # %%
    # If we are looking at stocks, enrich with industry codes
    if DATA_PATH.startswith("data/sp500_stocks"):
        meta_df = pd.read_csv("data/sp500_stocks_meta.csv")
        meta_df = meta_df.set_index("Symbol")
        df = df.join(meta_df, how="left", rsuffix="_META")

        # Check for nans
        nan_mask = df[["GICS Sector"]].isnull().sum(axis=1).gt(0)
        nan_rows = df[nan_mask]
        nan_rows.groupby(nan_rows.index.get_level_values("Symbol")).count()

    # %%
    # Check for NaN values
    important_cols = [
        "LogReturn",
        "Close_RVOL",
        "Close_VIX",
        "GARCH_Vol",
        "RVOL_Std",
        "DownsideVol",
        "Beta",
        "Skew_EWM",
        "Kurt_EWM",
    ]
    important_cols = [col for col in important_cols if col in df.columns]
    nan_mask = df[important_cols].isnull().sum(axis=1).gt(0)
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing RVOL values using the last available value
    df["Close_RVOL"] = df["Close_RVOL"].fillna(method="ffill")
    df["RVOL_Std"] = df["RVOL_Std"].fillna(method="ffill")
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing VIX values using the last available value
    df["Close_VIX"] = df["Close_VIX"].fillna(method="ffill")
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing DownsideVol values using the last available value
    df["DownsideVol"] = df["DownsideVol"].fillna(method="ffill")
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing Fear Greed values using the last available value
    df["Fear Greed"] = df["Fear Greed"].fillna(method="ffill")
    df[["LogReturn", "Fear Greed"]].loc[nan_mask]

    # %%
    # Add feature: is next day trading day or not
    dates = pd.to_datetime(df.index.get_level_values("Date"))
    df["NextDayTradingDay"] = dates.shift(1, freq="D").isin(dates)
    df["NextDayTradingDay"]

    # %%
    # Check for NaN values
    df[important_cols][df[important_cols].isnull().sum(axis=1).gt(0)]

    # %%
    # Check for infinite values
    df[important_cols][df[important_cols].eq(np.inf).any(axis=1)]

    # %%
    # Drop any rows with NaNs in important columns
    df = df.dropna(subset=important_cols)

    # %%
    # Change Date to datetime
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index(["Date", "Symbol"], inplace=True)

    # %%
    # Prepare data for LSTM
    X_train = []
    y_train = []
    validation_sets = OrderedDict()
    test_sets = OrderedDict()

    # If we have GICS sectors, add consistently encoded industry codes (integers) that
    # can be converted to one-hot vectors
    if "GICS Sector" in df.columns:
        df["IDY_CODE"] = df["GICS Sector"].map(GICS_SECTOR_MAPPING)

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

        # New Features
        rvol_std = group["RVOL_Std"].values.reshape(-1, 1)
        downside_vol = group["DownsideVol"].values.reshape(-1, 1)
        downside_log_var = np.log(downside_vol**2 + (0.1 / 100) ** 2)
        vix_rvol_diff = vix - rvol
        fear_greed = group["Fear Greed"].values.reshape(-1, 1) / 100
        fear_greed_1d = np.diff(fear_greed, axis=0, prepend=fear_greed[0, 0])
        fear_greed_7d = fear_greed - np.vstack([fear_greed[:7], fear_greed[:-7]])
        beta = group["Beta"].values.reshape(-1, 1)

        # Find dates to split on
        dates = pd.to_datetime(group.index.get_level_values("Date"))
        TRAIN_VALIDATION_SPLIT_index = len(group[dates < TRAIN_VALIDATION_SPLIT])
        VALIDATION_TEST_SPLIT_index = len(group[dates < VALIDATION_TEST_SPLIT])

        # Stack returns and squared returns together
        market_feature_factor = beta if multiply_by_beta else 1
        data = np.hstack(
            (
                log_sq_returns,
                sign_return,
                rvol * market_feature_factor,
                rvol_change_1d * market_feature_factor,
                rvol_change_2d * market_feature_factor,
                rvol_change_7d * market_feature_factor,
                vix_change_1d * market_feature_factor,
                vix_change_2d * market_feature_factor,
                vix_change_7d * market_feature_factor,
                next_day_trading_day * market_feature_factor,
                rvol_std * market_feature_factor,
                downside_log_var,
                vix_rvol_diff * market_feature_factor,
                returns,
                fear_greed * market_feature_factor,
                fear_greed_1d * 10 * market_feature_factor,
                fear_greed_7d * 10 * market_feature_factor,
                beta,
            )
        )

        # If we have GICS sectors, add them as a feature
        if "IDY_CODE" in group.columns:
            num_sectors = 11  # There are 11 GICS sectors
            one_hot_sector = np.zeros((len(group), num_sectors))
            one_hot_sector[np.arange(len(group)), group["IDY_CODE"]] = 1
            data = np.hstack((data, one_hot_sector))

        # If we have GARCH predictions, add them as a feature
        if "GARCH_Vol" in group.columns:
            garch = group["GARCH_Vol"].values.reshape(-1, 1)
            log_sq_garch = np.log(garch**2 + (0.1 / 100) ** 2)
            garch_skewness = group["Skew_EWM"].values.reshape(-1, 1)
            garch_kurtosis = group["Kurt_EWM"].values.reshape(-1, 1)
            data = np.hstack((data, log_sq_garch, garch_skewness, garch_kurtosis))

        # Include relative high-low difference as a feature if available to indicate volatility
        if (
            "High" in group.columns
            and "Low" in group.columns
            and not pd.isna(group["High"]).any()
            and not pd.isna(group["Low"]).any()
        ):
            high_low_diff = (
                (group["High"] - group["Low"]) / group["Close"]
            ).values.reshape(-1, 1)
            data = np.hstack((data, high_low_diff))

        # Create training sequences of length 'sequence_length'
        for i in range(LOOKBACK_DAYS, TRAIN_VALIDATION_SPLIT_index):
            X_train.append(data[i - LOOKBACK_DAYS : i])
            y_train.append(returns[i, 0])

        # Add test and validation data
        X_val = []
        y_val = []
        y_val_dates = []
        for i in range(TRAIN_VALIDATION_SPLIT_index, VALIDATION_TEST_SPLIT_index):
            point = data[i - LOOKBACK_DAYS : i]
            if len(point) != LOOKBACK_DAYS:
                continue
            X_val.append(point)
            y_val.append(returns[i, 0])
            y_val_dates.append(dates[i])
        if len(X_val) > 0:
            validation_sets[symbol] = LabelledTestSet(
                symbol,
                group.loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT],
                np.array(X_val).astype(np.float32),
                np.array(y_val).astype(np.float32),
                y_val_dates,
            )

        X_test = []
        y_test = []
        y_test_dates = []
        for i in range(VALIDATION_TEST_SPLIT_index, len(data)):
            point = data[i - LOOKBACK_DAYS : i]
            if len(point) != LOOKBACK_DAYS:
                continue
            X_test.append(point)
            y_test.append(returns[i, 0])
            y_test_dates.append(dates[i])
        if len(X_test) > 0:
            test_sets[symbol] = LabelledTestSet(
                symbol,
                group.loc[VALIDATION_TEST_SPLIT:],
                np.array(X_test).astype(np.float32),
                np.array(y_test).astype(np.float32),
                y_test_dates,
            )

    # %%
    # Convert X and y to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Ensure float32 for X and y
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    # Print shapes
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")

    if False:
        """
        Describe the data
        """
        # %%
        np.set_printoptions(suppress=True)
        print("Means:\n", list(float(n) for n in np.mean(X_train[:, -1, :], axis=0)))
        print("Stds:\n", list(float(n) for n in np.std(X_train[:, -1, :], axis=0)))

    gc.collect()

    # %%
    # Create return object
    return ProcessedData(df, X_train, y_train, validation_sets, test_sets)


def get_lstm_train_test_old(include_log_returns=False, include_fng=True):
    """
    Prepare data for LSTM
    """
    # %%
    df = pd.read_csv(DATA_PATH)

    if not "Symbol" in df.columns:
        df["Symbol"] = TEST_ASSET

    # Ensure the Date column is in datetime format
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # Sort the dataframe by both Date and Symbol
    df.sort_values(["Symbol", "Date"], inplace=True)
    df

    # %%
    # Join in the S&P 500 data
    spx_df = pd.read_csv("data/spx.csv")
    spx_df["Date"] = pd.to_datetime(spx_df["Date"]).dt.date
    spx_df.set_index("Date", inplace=True)
    df[["Close_SPX"]] = spx_df[["Close"]].loc[df["Date"].values].values
    df[["Close", "Close_SPX"]]

    # %%
    # Calculate log returns for each instrument separately using groupby
    df.sort_values(["Symbol", "Date"], inplace=True)
    df["LogReturn"] = (
        df.groupby("Symbol")["Close"]
        .apply(lambda x: np.log(x / x.shift(1)))
        .droplevel(0)
    )
    df["PctReturn"] = df.groupby("Symbol")["Close"].pct_change()
    df["MonthlyReturn"] = (
        df.groupby("Symbol")["Close"].apply(lambda x: x / x.shift(21) - 1).droplevel(0)
    )
    df["SPX_PctReturn"] = df.groupby("Symbol")["Close_SPX"].pct_change()
    df["SPX_LogReturn"] = np.log(df["Close_SPX"] / df["Close_SPX"].shift(1))
    df["SPX_MonthlyReturn"] = df["Close_SPX"] / df["Close_SPX"].shift(21) - 1

    # Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
    df = df[~df["LogReturn"].isnull()]

    df["SquaredReturn"] = df["LogReturn"] ** 2

    # Set date and symbol as index
    df.set_index(["Date", "Symbol"], inplace=True)
    df[
        [
            "Close",
            "LogReturn",
            "PctReturn",
            "MonthlyReturn",
            "Close_SPX",
            "SPX_LogReturn",
            "SPX_PctReturn",
            "SPX_MonthlyReturn",
        ]
    ]

    # %%
    # Compute rolling beta for each stock
    def compute_beta(g, window=251):
        g = g.reset_index().set_index("Date")
        rolling_df = g[["PctReturn", "SPX_PctReturn"]].rolling(
            window, min_periods=window
        )
        cov = rolling_df.cov().unstack()
        beta = (
            cov[("PctReturn", "SPX_PctReturn")]
            / cov[("SPX_PctReturn", "SPX_PctReturn")]
        )
        g["Beta"] = beta
        g = g.reset_index().set_index(["Date", "Symbol"])
        return g

    df = df.groupby("Symbol").apply(lambda g: compute_beta(g)).droplevel(0)
    df["Beta"]

    # %%
    # Add more features:
    # Downside Volatility
    df["DownsideVol"] = (
        df.groupby(level="Symbol")["LogReturn"]
        .apply(lambda x: x.where(x < 0).rolling(20, min_periods=1).std())
        .droplevel(0)
    )
    df[["Close", "LogReturn", "DownsideVol"]]

    # %%
    # If we have GARCH predictions, calculate skewness and kurtosis based on GARCH residuals
    if "GARCH_Vol" in df.columns:
        df["GARCH_Resid"] = df["LogReturn"] / df["GARCH_Vol"]

        # Apply to each Symbol group
        ewm_stats = df.groupby(level="Symbol")["GARCH_Resid"].apply(
            compute_ewm_skew_kurt
        )

        # Remove extra level in the multi-index
        ewm_stats = ewm_stats.droplevel(0)

        # `ewm_stats` now has a multi-index: (Symbol, Date).
        # We can join it back to df (which is indexed by (Date, Symbol) as well) directly:
        df = df.join(ewm_stats)

        # Drop first 20 rows for each instrument
        df = df.groupby("Symbol").apply(lambda x: x.iloc[20:])

        # Remove extra level in the multi-index
        df = df.droplevel(0)
        df

    # %%
    # Read RVOL data
    rvol_df = pd.read_csv(RVOL_DATA_PATH)
    rvol_df["Date"] = pd.to_datetime(rvol_df["Date"]).dt.date
    rvol_df = rvol_df.set_index("Date")
    rvol_df["RVOL_Std"] = rvol_df["Close"].rolling(10).std()
    rvol_df = rvol_df.dropna()
    rvol_df

    # %%
    # Read VIX data
    vix_df = pd.read_csv(VIX_DATA_PATH)
    vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.date
    vix_df = vix_df.set_index("Date")
    vix_df

    # %%
    # Read Fear & Greed Index data
    fng_df = pd.read_csv("data/fear-greed.csv")
    fng_df["Date"] = pd.to_datetime(fng_df["Date"]).dt.date
    fng_df = fng_df.set_index("Date")
    fng_df

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
    # Filter away data we don't have Fear & Greed Index data for
    df = df[df.index.get_level_values("Date") >= fng_df.index[0]]
    df = df[df.index.get_level_values("Date") <= fng_df.index[-1]]
    df

    # %%
    # Add RVOL data to the dataframe
    df = df.join(rvol_df, how="left", rsuffix="_RVOL")
    df

    # %%
    # Add VIX data to the dataframe
    df = df.join(vix_df, how="left", rsuffix="_VIX")

    # %%
    # Add Fear & Greed Index data to the dataframe
    df = df.join(fng_df, how="left")
    df[["LogReturn", "Fear Greed"]]

    # %%
    # If we are looking at stocks, enrich with industry codes
    if DATA_PATH.startswith("data/sp500_stocks"):
        meta_df = pd.read_csv("data/sp500_stocks_meta.csv")
        meta_df = meta_df.set_index("Symbol")
        df = df.join(meta_df, how="left", rsuffix="_META")

        # Check for nans
        nan_mask = df[["GICS Sector"]].isnull().sum(axis=1).gt(0)
        nan_rows = df[nan_mask]
        nan_rows.groupby(nan_rows.index.get_level_values("Symbol")).count()

    # %%
    # Check for NaN values
    important_cols = [
        "LogReturn",
        "Close_RVOL",
        "Close_VIX",
        "GARCH_Vol",
        "RVOL_Std",
        "DownsideVol",
        "Beta",
        "Skew_EWM",
        "Kurt_EWM",
    ]
    important_cols = [col for col in important_cols if col in df.columns]
    nan_mask = df[important_cols].isnull().sum(axis=1).gt(0)
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing RVOL values using the last available value
    df["Close_RVOL"] = df["Close_RVOL"].fillna(method="ffill")
    df["RVOL_Std"] = df["RVOL_Std"].fillna(method="ffill")
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing VIX values using the last available value
    df["Close_VIX"] = df["Close_VIX"].fillna(method="ffill")
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing DownsideVol values using the last available value
    df["DownsideVol"] = df["DownsideVol"].fillna(method="ffill")
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing Fear Greed values using the last available value
    df["Fear Greed"] = df["Fear Greed"].fillna(method="ffill")
    df[["LogReturn", "Fear Greed"]].loc[nan_mask]

    # %%
    # Add feature: is next day trading day or not
    dates = pd.to_datetime(df.index.get_level_values("Date"))
    df["NextDayTradingDay"] = dates.shift(1, freq="D").isin(dates)
    df["NextDayTradingDay"]

    # %%
    # Check for NaN values
    df[important_cols][df[important_cols].isnull().sum(axis=1).gt(0)]

    # %%
    # Check for infinite values
    df[important_cols][df[important_cols].eq(np.inf).any(axis=1)]

    # %%
    # Drop any rows with NaNs in important columns
    df = df.dropna(subset=important_cols)

    # %%
    # Prepare data for LSTM
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # If we have GICS sectors, add consistently encoded industry codes (integers) that
    # can be converted to one-hot vectors
    if "GICS Sector" in df.columns:
        df["IDY_CODE"] = df["GICS Sector"].map(GICS_SECTOR_MAPPING)

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

        # New Features
        rvol_std = group["RVOL_Std"].values.reshape(-1, 1)
        downside_vol = group["DownsideVol"].values.reshape(-1, 1)
        downside_log_var = np.log(downside_vol**2 + (0.1 / 100) ** 2)
        vix_rvol_diff = vix - rvol
        fear_greed = group["Fear Greed"].values.reshape(-1, 1) / 100
        fear_greed_1d = np.diff(fear_greed, axis=0, prepend=fear_greed[0, 0])
        fear_greed_7d = fear_greed - np.vstack([fear_greed[:7], fear_greed[:-7]])
        beta = group["Beta"].values.reshape(-1, 1)

        # Find dates to split on
        dates = pd.to_datetime(group.index.get_level_values("Date"))
        TRAIN_VALIDATION_SPLIT_index = len(group[dates < TRAIN_VALIDATION_SPLIT])
        VALIDATION_TEST_SPLIT_index = len(group[dates < VALIDATION_TEST_SPLIT])

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
                rvol_std,
                downside_log_var,
                vix_rvol_diff,
            )
        )

        if not (beta == 1).all():
            # If beta is not 1, add it as a feature
            # (It might be 1 if we are looking at the S&P 500 index)
            data = np.hstack((data, beta))

        if include_log_returns:
            data = np.hstack((data, returns))

        if include_fng:
            data = np.hstack((data, fear_greed, fear_greed_1d * 10, fear_greed_7d * 10))

        # If we have GICS sectors, add them as a feature
        if "IDY_CODE" in group.columns:
            num_sectors = 11  # There are 11 GICS sectors
            one_hot_sector = np.zeros((len(group), num_sectors))
            one_hot_sector[np.arange(len(group)), group["IDY_CODE"]] = 1
            data = np.hstack((data, one_hot_sector))

        # If we have GARCH predictions, add them as a feature
        if "GARCH_Vol" in group.columns:
            garch = group["GARCH_Vol"].values.reshape(-1, 1)
            log_sq_garch = np.log(garch**2 + (0.1 / 100) ** 2)
            garch_skewness = group["Skew_EWM"].values.reshape(-1, 1)
            garch_kurtosis = group["Kurt_EWM"].values.reshape(-1, 1)
            data = np.hstack((data, log_sq_garch, garch_skewness, garch_kurtosis))

        # Include relative high-low difference as a feature if available to indicate volatility
        if (
            "High" in group.columns
            and "Low" in group.columns
            and not pd.isna(group["High"]).any()
            and not pd.isna(group["Low"]).any()
        ):
            high_low_diff = (
                (group["High"] - group["Low"]) / group["Close"]
            ).values.reshape(-1, 1)
            data = np.hstack((data, high_low_diff))

        # Create training sequences of length 'sequence_length'
        for i in range(LOOKBACK_DAYS, TRAIN_VALIDATION_SPLIT_index):
            X_train.append(data[i - LOOKBACK_DAYS : i])
            y_train.append(returns[i, 0])

        # Add the test data
        if symbol == TEST_ASSET:
            for i in range(TRAIN_VALIDATION_SPLIT_index, VALIDATION_TEST_SPLIT_index):
                X_test.append(data[i - LOOKBACK_DAYS : i])
                y_test.append(returns[i, 0])

    # %%
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

    if False:
        """
        Describe the data
        """
        # %%
        np.set_printoptions(suppress=True)
        print("Means:\n", list(float(n) for n in np.mean(X_train[:, -1, :], axis=0)))
        print("Stds:\n", list(float(n) for n in np.std(X_train[:, -1, :], axis=0)))

    # %%
    # Change Date to datetime
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index(["Date", "Symbol"], inplace=True)

    gc.collect()

    # %%
    return df, X_train, X_test, y_train, y_test


# %%
def get_cgan_train_test():
    """
    Prepare data for CGAN
    """
    # %%
    df, X_train, X_test, y_train, y_test = get_lstm_train_test_old(
        include_log_returns=True
    )
    # Normalize log squared returns and RVOL
    scaling_mean = X_train[:, :, 0].mean()
    scaling_std = X_train[:, :, 0].std()

    # Normalize X[:, :, 0] (log squared returns) using y_train_mean and y_train_std
    X_train[:, :, 0] = (X_train[:, :, 0] - scaling_mean) / scaling_std
    X_test[:, :, 0] = (X_test[:, :, 0] - scaling_mean) / scaling_std

    # Normalize RVOL (X[:, :, 2]) using y_train_mean and y_train_std
    X_train[:, :, 2] = (X_train[:, :, 2] - scaling_mean) / scaling_std
    X_test[:, :, 2] = (X_test[:, :, 2] - scaling_mean) / scaling_std

    # Scale y and log returns by 100 to improve scale
    y_train = y_train * 100
    y_test = y_test * 100
    X_train[:, :, 10] = X_train[:, :, 10] * 100

    # Scale other features (except indicators) by 10 to improve scale
    for i in range(3, 9):
        X_train[:, :, i] = X_train[:, :, i] * 10
        X_test[:, :, i] = X_test[:, :, i] * 10

    return df, X_train, X_test, y_train, y_test, scaling_mean, scaling_std


# %%
# Define a helper that returns a DataFrame of skew and kurt for a given residual series
def compute_ewm_skew_kurt(series: pd.Series, alpha=0.06):
    """
    alpha is the smoothing factor for the exponentially weighted moments.
    Adjust it to put more (or less) weight on recent data.
    """
    ewm_mean = series.ewm(alpha=alpha).mean()
    ewm_mean2 = (series**2).ewm(alpha=alpha).mean()
    ewm_mean3 = (series**3).ewm(alpha=alpha).mean()
    ewm_mean4 = (series**4).ewm(alpha=alpha).mean()

    ewm_var = ewm_mean2 - ewm_mean**2
    ewm_std = np.sqrt(ewm_var)

    # Pearson's moment coefficient of skewness and excess kurtosis
    ewm_skew = (ewm_mean3 - 3 * ewm_mean * ewm_mean2 + 2 * ewm_mean**3) / (ewm_std**3)
    ewm_kurt = (
        ewm_mean4
        - 4 * ewm_mean * ewm_mean3
        + 6 * (ewm_mean**2) * ewm_mean2
        - 3 * ewm_mean**4
    ) / (ewm_std**4)

    # Return as a DataFrame so it's easy to join back
    return pd.DataFrame({"Skew_EWM": ewm_skew, "Kurt_EWM": ewm_kurt})


# %%
