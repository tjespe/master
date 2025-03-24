# %%
# Define parameters (imported from your settings)
from dataclasses import dataclass
from functools import cached_property
from collections import OrderedDict
from typing import Iterable, Union
import os
import tensorflow as tf
from settings import (
    LOOKBACK_DAYS,
    TEST_ASSET,
    DATA_PATH,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
    BASEDIR,
)
from shared.fred import get_fred_md


RVOL_DATA_PATH = f"{BASEDIR}/data/RVOL.csv"
VIX_DATA_PATH = f"{BASEDIR}/data/VIX.csv"

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
class LabelledDataSet:
    ticker: str
    X: np.ndarray  # [n_samples, sequence_length, n_features]
    col_names: list[str]  # Column names for the X matrix
    y: np.ndarray
    y_dates: list[pd.Timestamp]

    @cached_property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": self.y_dates,
                "Symbol": [self.ticker] * len(self.y_dates),
                "ActualReturn": self.y,
                **{col: self.X[:, -1, i] for i, col in enumerate(self.col_names)},
            }
        ).set_index(["Date", "Symbol"])

    def __str__(self):
        return f"LabelledDataSet(ticker={self.ticker}, X.shape={self.X.shape}, y.shape={self.y.shape})"

    def __repr__(self):
        return str(self)


@dataclass
class DataSetCollection:
    sets: OrderedDict[str, LabelledDataSet]

    @cached_property
    def X(self) -> np.ndarray:
        return np.concatenate([s.X for s in self.sets.values()])

    @cached_property
    def y(self) -> np.ndarray:
        return np.concatenate([s.y for s in self.sets.values()])

    @cached_property
    def tickers(self) -> list[str]:
        return np.array([t for s in self.sets.values() for t in [s.ticker] * len(s.y)])

    @cached_property
    def dates(self) -> np.ndarray[pd.Timestamp]:
        return np.array([d for s in self.sets.values() for d in s.y_dates])

    def get_range(self, ticker: str):
        from_idx = np.where(self.tickers == ticker)[0][0]
        to_idx = np.where(self.tickers == ticker)[0][-1] + 1
        return from_idx, to_idx

    def __str__(self):
        return f"DataSetCollection(X.shape={self.X.shape}, y.shape={self.y.shape}, sets={len(self.sets)})"

    def __repr__(self):
        return str(self)

    def filter_by_tickers(self, tickers: Iterable[str]) -> "DataSetCollection":
        return DataSetCollection(
            OrderedDict((k, v) for k, v in self.sets.items() if k in tickers)
        )

    @cached_property
    def df(self) -> pd.DataFrame:
        return pd.concat([s.df for s in self.sets.values()])


@dataclass
class ProcessedData:
    train: DataSetCollection
    validation: DataSetCollection
    test: DataSetCollection

    _ticker_lookup = None

    def _initialize_ticker_lookup(self):
        if self._ticker_lookup is None:
            self._ticker_lookup = tf.keras.layers.StringLookup(
                vocabulary=sorted(set(self.train.tickers)), num_oov_indices=1
            )

    def encode_tickers(self, tickers: Iterable[str]) -> tf.Tensor:
        self._initialize_ticker_lookup()
        return self._ticker_lookup(tickers)

    @cached_property
    def train_ticker_ids(self):
        return self.encode_tickers(self.train.tickers)

    @cached_property
    def validation_ticker_ids(self):
        return self.encode_tickers(self.validation.tickers)

    @cached_property
    def test_ticker_ids(self):
        return self.encode_tickers(self.test.tickers)

    @cached_property
    def ticker_ids_dim(self):
        self._initialize_ticker_lookup()
        return self._ticker_lookup.vocabulary_size()

    @cached_property
    def df(self) -> pd.DataFrame:
        train_df = self.train.df.copy()
        train_df["Set"] = "Train"
        validation_df = self.validation.df.copy()
        validation_df["Set"] = "Validation"
        test_df = self.test.df.copy()
        test_df["Set"] = "Test"
        return pd.concat([train_df, validation_df, test_df])


def get_lstm_train_test_new(
    multiply_by_beta=False,
    include_fng=False,
    include_spx_data=False,
    include_returns=False,
    include_industry=False,
    include_garch=False,
    include_beta=False,
    include_others=False,
    include_fred_md=False,
    include_fred_qd=False,
    include_ivol_cols: Union[None, list[str]] = None,
    include_5min_rv=True,
    include_1min_rv=False,
) -> ProcessedData:
    """
    Prepare data for LSTM
    """
    print("Processing data...")
    print("Multiply by beta:", multiply_by_beta)
    print("Include FNG:", include_fng)
    print("Include SPX data:", include_spx_data)
    print("Include returns:", include_returns)
    print("Include industry:", include_industry)
    print("Include GARCH:", include_garch)
    print("Include beta:", include_beta)
    print("Include others:", include_others)
    print("Include FRED-MD:", include_fred_md)
    print("Include FRED-QD:", include_fred_qd)
    print("Include IVOL cols:", include_ivol_cols)
    print("Include 5min RV:", include_5min_rv)
    print("Include 1min RV:", include_1min_rv)
    # %%
    df = pd.read_csv(DATA_PATH)

    # Ensure the Date column is in datetime format
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # Sort the dataframe by both Date and Symbol
    df.sort_values(["Symbol", "Date"], inplace=True)
    df

    # %%
    # Remove .O from the symbol names
    df["Symbol"] = df["Symbol"].str.replace(r"\.O$", "", regex=True)
    df

    # %%
    # Temporary: remove NaT date rows
    df = df[~df["Date"].isnull()]
    df

    # %%
    # Read the S&P 500 data
    spx_df = pd.read_csv(
        f"{BASEDIR}/data/spx/processed_data/spx_19900101_to_today_20250219_cleaned.csv"
    )
    spx_df["Date"] = pd.to_datetime(spx_df["Date"]).dt.date
    spx_df.set_index("Date", inplace=True)

    # %%
    # Temporary: filter away rows outside the S&P 500 data range
    spx_end = spx_df.index[-1]
    df = df[df["Date"] <= spx_end].copy()  # Copy to avoid SettingWithCopyWarning
    df

    # %%
    # Temporary: remove October 25th 2003
    df = df[~df["Date"].astype(str).str.startswith("2003-10-25")].copy()
    # Inspect weekdays
    df["Weekday"] = df["Date"].astype("datetime64[ns]").dt.day_name()
    df["Weekday"].value_counts()

    # %%
    # Join in the S&P 500 data
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
    df = df[~df["LogReturn"].isnull()].copy()  # Copy to avoid SettingWithCopyWarning

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
    # Join in FRED-MD data
    if include_fred_md:
        fred_md_df = get_fred_md()
        fred_md_df = (
            fred_md_df.reindex(df.index.get_level_values("Date"))
            .groupby("Symbol")
            .ffill()
        )
        fred_md_df.index = df.index
        df = df.join(fred_md_df)
    df

    # %%
    # Join in IVOL data
    if include_ivol_cols:
        ivol_df = pd.read_csv(
            f"{BASEDIR}/data/dow_jones/processed_data/processed_ivol_data.csv"
        )
        ivol_df["Date"] = pd.to_datetime(ivol_df["Date"]).dt.date
        ivol_df.set_index(["Date", "Symbol"], inplace=True)
        ivol_df = ivol_df[ivol_df.columns.intersection(include_ivol_cols)]
        df = df.join(ivol_df, how="left")
    df

    # %%
    # Compute rolling beta for each stock
    def compute_beta(g, window=251 * 5):
        g = g.reset_index().set_index("Date")
        rolling_df = g[["PctReturn", "SPX_PctReturn"]].rolling(
            window, min_periods=window
        )
        cov = rolling_df.cov().unstack()
        beta = (
            cov[("PctReturn", "SPX_PctReturn")]
            / cov[("SPX_PctReturn", "SPX_PctReturn")]
        )
        return beta

    df["Beta_5y"] = df.groupby("Symbol").apply(lambda g: compute_beta(g)).values
    df["Beta_60d"] = df.groupby("Symbol").apply(lambda g: compute_beta(g, 60)).values
    df[["Beta_5y", "Beta_60d"]]

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
    # Define columns that must be non-nan
    important_cols = [
        "LogReturn",
    ] + (include_ivol_cols or [])

    if include_spx_data:
        important_cols += [
            "Close_RVOL",
            "Close_VIX",
            "RVOL_Std",
        ]

    if include_beta:
        important_cols += [
            "Beta_5y",
            "Beta_60d",
        ]

    # %%
    # If we have GARCH predictions, calculate skewness nd kurtosis based on GARCH residuals
    for garch_type in ["GARCH", "EGARCH"]:
        if f"{garch_type}_Vol" in df.columns:
            df[f"{garch_type}_Resid"] = df["LogReturn"] / df[f"{garch_type}_Vol"]

            # Apply to each Symbol group
            ewm_stats = df.groupby(level="Symbol")[f"{garch_type}_Resid"].apply(
                compute_ewm_skew_kurt
            )

            # Remove extra level in the multi-index
            ewm_stats = ewm_stats.droplevel(0)

            # `ewm_stats` now has a multi-index: (Symbol, Date).
            # We can join it back to df (which is indexed by (Date, Symbol) as well) directly:
            df = df.join(ewm_stats.add_suffix(f"_{garch_type}"))

            # Drop first 20 rows for each instrument
            df = df.groupby("Symbol").apply(lambda x: x.iloc[20:])

            # Remove extra level in the multi-index
            df = df.droplevel(0)

            important_cols += [
                f"{garch_type}_Vol",
                f"Skew_EWM_{garch_type}",
                f"Kurt_EWM_{garch_type}",
            ]
    df

    # %%
    # If we are using the Dow Jones dataset, read in the realized volatility data
    if DATA_PATH.startswith(f"{BASEDIR}/data/dow_jones"):
        capire_df = pd.read_csv(
            f"{BASEDIR}/data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv"
        )
        capire_df["Date"] = pd.to_datetime(capire_df["Date"]).dt.date
        capire_df.set_index(["Date", "Symbol"], inplace=True)
        df = df.join(capire_df, how="left")

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
    fng_df = pd.read_csv(f"{BASEDIR}/data/fear-greed.csv")
    fng_df["Date"] = pd.to_datetime(fng_df["Date"]).dt.date
    fng_df = fng_df.set_index("Date")
    fng_df

    # %%
    # Filter away data we don't have RVOL data for
    if include_spx_data:
        df = df[df.index.get_level_values("Date") >= rvol_df.index[0]]
        df = df[df.index.get_level_values("Date") <= rvol_df.index[-1]]
    df

    # %%
    # Filter away data we don't have VIX data for
    if include_spx_data:
        df = df[df.index.get_level_values("Date") >= vix_df.index[0]]
        df = df[df.index.get_level_values("Date") <= vix_df.index[-1]]
    df

    # %%
    # Filter away data we don't have Fear & Greed Index data for
    if include_fng:
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
    if DATA_PATH.startswith(f"{BASEDIR}/data/sp500_stocks") or DATA_PATH.startswith(
        f"{BASEDIR}/data/dow_jones"
    ):
        print("Adding industry codes")
        meta_df = pd.read_csv(f"{BASEDIR}/data/sp500_stocks_meta.csv")
        meta_df = meta_df.set_index("Symbol")
        df = df.join(meta_df, how="left", rsuffix="_META")

        # Check for nans
        nan_mask = df[["GICS Sector"]].isnull().sum(axis=1).gt(0)
        nan_rows = df[nan_mask]
        if nan_rows.shape[0] > 0:
            print(
                "NaNs:",
                nan_rows.groupby(nan_rows.index.get_level_values("Symbol")).count(),
            )

    # %%
    # Check for NaN values
    important_cols = [col for col in important_cols if col in df.columns]
    nan_mask = df[important_cols].isnull().sum(axis=1).gt(0)
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing RVOL values using the last available value
    df["Close_RVOL"] = df.groupby("Symbol")["Close_RVOL"].ffill()
    df["RVOL_Std"] = df.groupby("Symbol")["RVOL_Std"].ffill()
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing VIX values using the last available value
    df["Close_VIX"] = df.groupby("Symbol")["Close_VIX"].ffill()
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing DownsideVol values using the last available value
    df["DownsideVol"] = df.groupby("Symbol")["DownsideVol"].ffill()
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing Fear Greed values using the last available value
    df["Fear Greed"] = df.groupby("Symbol")["Fear Greed"].ffill()
    df[["LogReturn", "Fear Greed"]].loc[nan_mask]

    # %%
    # Add feature: is next day trading day or not
    dates = pd.to_datetime(df.index.get_level_values("Date"))
    df["NextDayTradingDay"] = dates.shift(1, freq="D").isin(dates)
    df["NextDayTradingDay"]

    # %%
    # If we use realized volatility data, remove rows outside range
    if "RV" in df.columns:
        rv_data = df[~df["RV"].isna()]
        rv_min_date = rv_data.index.get_level_values("Date").min()
        rv_max = rv_data.index.get_level_values("Date").max()
        print(f"RV data range: {rv_min_date} - {rv_max}")
        df = df[
            (df.index.get_level_values("Date") >= rv_min_date)
            & (df.index.get_level_values("Date") <= rv_max)
        ]

        important_cols += ["RV"]
    df

    # %%
    # Check for missing beta values
    beta_nan_df = df[df["Beta_5y"].isna()].reset_index()
    beta_nans_per_day = beta_nan_df.groupby("Date")["Date"].count()
    # for ticker in beta_nan_df["Symbol"].unique():
    #     df.xs(ticker, level="Symbol")["Beta_5y"].plot(label=ticker + " beta")
    # import matplotlib.pyplot as plt
    # plt.legend()
    # beta_nan_df.groupby("Symbol")["Symbol"].count()
    beta_nans_per_day

    # %%
    # Remove rows with nan beta
    df = df[~df["Beta_5y"].isna()]
    df

    # %%
    # For any IVOL cols, front-fill with latest value if missing
    if include_ivol_cols:
        for col in include_ivol_cols:
            # Make sure we don't ffill across assets
            df[col] = df.groupby("Symbol")[col].ffill()
    df

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
    train_sets = OrderedDict()
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
        beta_5y = group["Beta_5y"].values.reshape(-1, 1)
        beta_60d = group["Beta_60d"].values.reshape(-1, 1)

        # Find dates to split on
        dates = pd.to_datetime(group.index.get_level_values("Date"))
        TRAIN_VALIDATION_SPLIT_index = len(group[dates < TRAIN_VALIDATION_SPLIT])
        VALIDATION_TEST_SPLIT_index = len(group[dates < VALIDATION_TEST_SPLIT])

        # Stack returns and squared returns together
        market_feature_factor = beta_5y if multiply_by_beta else 1

        # Start out with an empty data array
        data = np.zeros((len(group), 0))
        col_names = []

        if include_others:
            data = np.hstack(
                (
                    log_sq_returns,
                    sign_return,
                    next_day_trading_day,
                )
            )
            col_names = ["LogSquaredReturn", "SignReturn", "NextDayTradingDay"]

        if include_beta:
            data = np.hstack(
                (
                    data,
                    beta_5y,
                    beta_60d,
                )
            )
            col_names += ["Beta5y", "Beta60d"]

        if include_returns:
            data = np.hstack(
                (
                    data,
                    returns,
                )
            )
            col_names += ["LogReturn"]

        if include_spx_data:
            data = np.hstack(
                (
                    rvol * market_feature_factor,
                    rvol_change_1d * market_feature_factor,
                    rvol_change_2d * market_feature_factor,
                    rvol_change_7d * market_feature_factor,
                    vix_change_1d * market_feature_factor,
                    vix_change_2d * market_feature_factor,
                    vix_change_7d * market_feature_factor,
                    rvol_std * market_feature_factor,
                    vix_rvol_diff * market_feature_factor,
                )
            )
            col_names += [
                "RVOL_SP",
                "RVOL_SP_Change1d",
                "RVOL_SP_Change2d",
                "RVOL_SP_Change7d",
                "VIXChange1d",
                "VIXChange2d",
                "VIXChange7d",
                "RVOL_SP_Std",
                "VIXRVOLDiff",
            ]

        if include_fred_md:
            fred_features = fred_md_df.columns
            data = np.hstack(
                (
                    data,
                    group[fred_features].values,
                )
            )
            col_names += fred_features

        if include_fng:
            data = np.hstack(
                (
                    data,
                    fear_greed * market_feature_factor,
                    fear_greed_1d * 10 * market_feature_factor,
                    fear_greed_7d * 10 * market_feature_factor,
                )
            )
            col_names += ["FearGreed", "FearGreed1d", "FearGreed7d"]

        # If we have RVOL data, add it as a feature
        if "RV" in group.columns:
            # 1) Variance-like measures: RV, BPV, Good, Bad (and their 5-min versions)
            variance_keys = [
                *(["RV", "BPV", "Good", "Bad"] if include_1min_rv else []),
                *(["RV_5", "BPV_5", "Good_5", "Bad_5"] if include_5min_rv else []),
            ]
            for key in variance_keys:
                # annualized variance in "percent^2" => convert to decimal => then daily
                annual_var_pct2 = group[key].values.reshape(
                    -1, 1
                )  # e.g. 8.31 => 8.31%²
                annual_var_decimal = annual_var_pct2 / 100.0  # => 0.0831
                daily_var_decimal = annual_var_decimal / 252.0  # => daily variance
                log_daily_var = np.log(
                    daily_var_decimal + 1e-10
                )  # small offset to avoid log(0)
                data = np.hstack((data, log_daily_var))
                col_names.append(key)

            # 2) Quarticity measures: RQ (and RQ_5)
            quarticity_keys = (["RQ"] if include_1min_rv else []) + (
                ["RQ_5"] if include_5min_rv else []
            )
            for key in quarticity_keys:
                # annualized quarticity in "percent^4" => decimal => daily
                annual_q_pct4 = group[key].values.reshape(-1, 1)  # e.g. 263 => 263%²
                annual_q_decimal = annual_q_pct4 / (100.0**2)  # => 2.63 in decimal^2
                data = np.hstack(
                    (
                        data,
                        # Log for scale
                        np.log(annual_q_decimal + 1e-12),
                    )
                )
                col_names.append(key)

            # 3) Estimate realized skewness and kurtosis
            if include_others:
                daily_good_var = (group["Good"].values / 100) / 252.0
                daily_bad_var = (group["Bad"].values / 100) / 252.0
                daily_rv = (group["RV"].values / 100) / 252.0
                daily_rq = (group["RQ"].values / 10000.0) / (252.0**2)
                daily_skew = (
                    (1.5 * (daily_good_var - daily_bad_var)) / (daily_rv**1.5 + 1e-12)
                ).reshape(-1, 1)
                daily_kurt = (daily_rq / (daily_rv**2 + 1e-12)).reshape(-1, 1)

                data = np.hstack(
                    (
                        data,
                        daily_skew / 100,
                        daily_kurt / 10,
                    )
                )
                col_names += ["Skew", "Kurt"]

        if include_industry and "IDY_CODE" in group.columns:
            # If we have GICS sectors, add them as a feature
            num_sectors = 11  # There are 11 GICS sectors
            one_hot_sector = np.zeros((len(group), num_sectors))
            one_hot_sector[np.arange(len(group)), group["IDY_CODE"]] = 1
            data = np.hstack((data, one_hot_sector))
            col_names += [
                next(k for k, v in GICS_SECTOR_MAPPING.items() if v == i)
                for i in range(num_sectors)
            ]

        # If we have GARCH predictions, add them as a feature
        for garch_type in ["GARCH", "EGARCH"]:
            if include_garch and f"{garch_type}_Vol" in group.columns:
                garch = group[f"{garch_type}_Vol"].values.reshape(-1, 1)
                log_sq_garch = np.log(garch**2 + (0.1 / 100) ** 2)
                garch_skewness = group[f"Skew_EWM_{garch_type}"].values.reshape(-1, 1)
                garch_kurtosis = group[f"Kurt_EWM_{garch_type}"].values.reshape(-1, 1)
                data = np.hstack(
                    (
                        data,
                        log_sq_garch,
                        garch_skewness,
                        garch_kurtosis,
                    )
                )
                col_names += [
                    f"{garch_type}_Vol",
                    f"{garch_type}_Skew",
                    f"{garch_type}_Kurt",
                ]

        # Include relative high-low difference as a feature if available to indicate volatility
        if (
            "High" in group.columns
            and "Low" in group.columns
            and not pd.isna(group["High"]).any()
            and not pd.isna(group["Low"]).any()
            and include_others
        ):
            high_low_diff = (
                (group["High"] - group["Low"]) / group["Close"]
            ).values.reshape(-1, 1)
            data = np.hstack((data, high_low_diff))
            col_names += ["HighLowDiff"]

        if include_ivol_cols:
            for ivol_col in include_ivol_cols:
                annual_pct_vol = group[ivol_col].values.reshape(-1, 1)
                annual_vol = annual_pct_vol / 100
                annual_var = annual_vol**2
                daily_var = annual_var / 252
                log_daily_var = np.log(daily_var + 1e-10)
                data = np.hstack((data, log_daily_var))
                col_names.append(ivol_col)

        # Create training sequences of length 'sequence_length'
        X_train = []
        y_train = []
        y_train_dates = []
        for i in range(LOOKBACK_DAYS, TRAIN_VALIDATION_SPLIT_index):
            point = data[i - LOOKBACK_DAYS : i]
            if len(point) != LOOKBACK_DAYS:
                continue
            X_train.append(point)
            y_train.append(returns[i, 0])
            y_train_dates.append(dates[i])
        if len(X_train) > 0:
            train_sets[symbol] = LabelledDataSet(
                symbol,
                np.array(X_train).astype(np.float32),
                col_names,
                np.array(y_train).astype(np.float32),
                y_train_dates,
            )

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
            validation_sets[symbol] = LabelledDataSet(
                symbol,
                np.array(X_val).astype(np.float32),
                col_names,
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
            test_sets[symbol] = LabelledDataSet(
                symbol,
                np.array(X_test).astype(np.float32),
                col_names,
                np.array(y_test).astype(np.float32),
                y_test_dates,
            )

    if False:
        """
        Describe the data
        """
        # %%
        np.set_printoptions(suppress=True)
        print("Means:\n", list(float(n) for n in np.mean(X_train[:, -1, :], axis=0)))
        print("Stds:\n", list(float(n) for n in np.std(X_train[:, -1, :], axis=0)))

    # %%
    gc.collect()

    # %%
    # Create return object
    return ProcessedData(
        DataSetCollection(train_sets),
        DataSetCollection(validation_sets),
        DataSetCollection(test_sets),
    )


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
    spx_df = pd.read_csv(
        f"{BASEDIR}/data/spx/processed_data/spx_19900101_to_today_20250219_cleaned.csv"
    )
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
    fng_df = pd.read_csv(f"{BASEDIR}/data/fear-greed.csv")
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
    if DATA_PATH.startswith(f"{BASEDIR}/data/sp500_stocks"):
        meta_df = pd.read_csv(f"{BASEDIR}/data/sp500_stocks_meta.csv")
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
    df["Close_RVOL"] = df.groupby("Symbol")["Close_RVOL"].ffill()
    df["RVOL_Std"] = df.groupby("Symbol")["RVOL_Std"].fillna(method="ffill")
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing VIX values using the last available value
    df["Close_VIX"] = df.groupby("Symbol")["Close_VIX"].fillna(method="ffill")
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing DownsideVol values using the last available value
    df["DownsideVol"] = df.groupby("Symbol")["DownsideVol"].fillna(method="ffill")
    df[important_cols].loc[nan_mask]

    # %%
    # Impute missing Fear Greed values using the last available value
    df["Fear Greed"] = df.groupby("Symbol")["Fear Greed"].fillna(method="ffill")
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
def compute_ewm_skew_kurt(series: pd.Series, alpha=0.06, clip=5) -> pd.DataFrame:
    """
    Compute exponentially weighted skewness and kurtosis for a given residual series.

    Parameters:
    - series: the residual series to compute skewness and kurtosis for. Calculated as the
        true log return divided by the GARCH volatility.
    - alpha is the smoothing factor for the exponentially weighted moments. Adjust
        it to put more (or less) weight on recent data.
    - clip is the maximum value for the residuals. This is used to clip extreme values
        that could skew the results.
    """
    # Clip extreme residuals
    series = series.clip(lower=-clip, upper=clip)

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
