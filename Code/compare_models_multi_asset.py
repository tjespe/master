# %%
import sys
from scipy import stats
from sklearn.linear_model import LinearRegression
from shared.styling_guidelines_graphs import colors
from shared.skew_t import skewt_nll
from shared.adequacy import (
    christoffersen_test,
    pooled_bayer_dimitriadis_test,
)
from shared.jupyter import is_notebook
from shared.mdn import calculate_es_for_quantile
from shared.conf_levels import format_cl
from shared.loss import (
    al_loss,
    crps_normal_univariate,
    crps_skewt,
    ece_gaussian,
    ece_skewt,
    ece_student_t,
    fz_loss,
    nll_loss_mean_and_vol,
    student_t_nll,
)
from settings import (
    DATA_PATH,
    LOOKBACK_DAYS,
    SUFFIX,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
    TEST_SET,
)
from data.tickers import IMPORTANT_TICKERS
from scipy.stats import ttest_rel
from scipy.stats import linregress
from scipy.stats import ttest_1samp
from arch.bootstrap import MCS
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# %%
# Define which confidence levels to look at in tests
CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.98]

# %%
# Define all confidence levels that we want to read from the files
ALL_CONFIDENCE_LEVELS = CONFIDENCE_LEVELS + [0.99, 0.995]

# %%
# Select whether to only filter on important tickers
FILTER_ON_IMPORTANT_TICKERS = True

# %%
# Set threshold for how many NaN values are allowed in the predictions
MAX_NAN_THRESH = 0.001  # 0.1%

# %%
# Select wheter or not to print all of the christoffersen tests while testing
PRINT_CHRISTOFFERSEN_TESTS = False

# %%
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# %%
df = pd.read_csv(DATA_PATH)

# Ensure the Date column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Remove .O suffix from tickers
df["Symbol"] = df["Symbol"].str.replace(".O", "")

# Sort the dataframe by both Date and Symbol
df = df.sort_values(["Symbol", "Date"])

# Calculate log returns for each instrument separately using groupby
df["LogReturn"] = (
    df.groupby("Symbol")["Close"].apply(lambda x: np.log(x / x.shift(1))).droplevel(0)
)

df["Total Return Test"] = (
    df.groupby("Symbol")["Close"].apply(lambda x: (x / x.shift(1)) - 1).droplevel(0)
)

# Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
df = df[~df["LogReturn"].isnull()]

df["SquaredReturn"] = df["LogReturn"] ** 2

# Set date and symbol as index
df: pd.DataFrame = df.set_index(["Date", "Symbol"])

# %%
# load capire data
capire_df = pd.read_csv(
    "data/dow_jones/processed_data/processed_capire_stock_data_dow_jones.csv"
)
capire_df["Date"] = pd.to_datetime(capire_df["Date"])
capire_df = capire_df.set_index(["Date", "Symbol"])
# merge with df based on date and symbol
df = df.merge(capire_df, on=["Date", "Symbol"], how="inner")
# transform RV to daily decimal
df["RV_5_daily"] = (df["RV_5"] / 100) / 252.0
# %%
# Filter away data before 1990
df = df[df.index.get_level_values("Date") >= "1990-01-01"]

# %%
# Get validation part of df
dates = df.index.get_level_values("Date")
df_validation = (
    df[(dates >= TRAIN_VALIDATION_SPLIT) & (dates < VALIDATION_TEST_SPLIT)]
    if TEST_SET == "validation"
    else df[(dates >= VALIDATION_TEST_SPLIT)]
)

# %%
# Filter on important tickers
if FILTER_ON_IMPORTANT_TICKERS:
    df_validation = df_validation[
        df_validation.index.get_level_values("Symbol").isin(IMPORTANT_TICKERS)
    ]


# %%
# Collect models and their predictions (in order of model complexity)
preds_per_model = []

# GARCH Model
for garch_type in ["GARCH", "EGARCH"]:
    try:
        garch_vol_pred = df_validation[f"{garch_type}_Vol"].values
        y_true = df_validation["LogReturn"].values
        mus = np.zeros_like(garch_vol_pred)
        log_vars = np.log(garch_vol_pred**2)

        entry = {
            "name": garch_type,
            "mean_pred": mus,
            "volatility_pred": garch_vol_pred,
            "symbols": df_validation.index.get_level_values("Symbol"),
            "dates": df_validation.index.get_level_values("Date"),
            "nll": nll_loss_mean_and_vol(
                y_true,
                mus,
                garch_vol_pred,
            ),
            "crps": crps_normal_univariate(y_true, mus, garch_vol_pred),
            "ece": ece_gaussian(y_true, mus, log_vars),
        }

        for cl in ALL_CONFIDENCE_LEVELS:
            alpha = 1 - cl
            z_alpha = norm.ppf(1 - alpha / 2)
            lb = mus - z_alpha * garch_vol_pred
            ub = mus + z_alpha * garch_vol_pred
            entry[f"LB_{format_cl(cl)}"] = lb
            entry[f"UB_{format_cl(cl)}"] = ub
            es_alpha = alpha / 2
            entry[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
                np.ones_like(mus).reshape(-1, 1),
                mus.reshape(-1, 1),
                garch_vol_pred.reshape(-1, 1),
                lb,
            )

        preds_per_model.append(entry)
        nans = np.isnan(garch_vol_pred).sum()
        if nans > 0:
            print(f"{garch_type} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"{garch_type} predictions not found")

# GARCH Student-t Model
try:
    garch_t_vol_pred = pd.read_csv("predictions/garch_predictions_student_t.csv")
    garch_t_vol_pred["Date"] = pd.to_datetime(garch_t_vol_pred["Date"])
    garch_t_vol_pred = garch_t_vol_pred.set_index(["Date", "Symbol"])
    garch_t_dates = garch_t_vol_pred.index.get_level_values("Date")
    garch_t_vol_pred = garch_t_vol_pred[
        (
            (garch_t_dates >= TRAIN_VALIDATION_SPLIT)
            & (garch_t_dates < VALIDATION_TEST_SPLIT)
            if TEST_SET == "validation"
            else (garch_t_dates >= VALIDATION_TEST_SPLIT)
        )
    ]
    if np.isnan(garch_t_vol_pred).all().all():
        raise FileNotFoundError("All GARCH Student-t predictions are NaN")
    combined_df = df_validation.join(garch_t_vol_pred, how="left", rsuffix="_GARCH_t")
    garch_t_vol_pred = combined_df["GARCH_t_Vol"].values
    y_true = combined_df["LogReturn"].values
    mus = np.zeros_like(garch_t_vol_pred)
    nus = combined_df["GARCH_t_Nu"].values
    crps = combined_df["GARCH_t_CRPS"].values

    entry = {
        "name": "GARCH Student-t",
        "mean_pred": mus,
        "volatility_pred": garch_t_vol_pred,
        "symbols": combined_df.index.get_level_values("Symbol"),
        "dates": combined_df.index.get_level_values("Date"),
        "nll": student_t_nll(
            y_true,
            mus,
            garch_t_vol_pred,
            nus,
        ),
        "crps": crps,
        "ece": ece_student_t(y_true, mus, garch_t_vol_pred, nus),
        "LB_67": combined_df["LB_67"].values,
        "UB_67": combined_df["UB_67"].values,
        "LB_90": combined_df["LB_90"].values,
        "UB_90": combined_df["UB_90"].values,
        "LB_95": combined_df["LB_95"].values,
        "UB_95": combined_df["UB_95"].values,
        "LB_98": combined_df["LB_98"].values,
        "UB_98": combined_df["UB_98"].values,
        "ES_83.5": combined_df["ES_83.5"].values,
        "ES_95": combined_df["ES_95"].values,
        "ES_97.5": combined_df["ES_97.5"].values,
        "ES_99": combined_df["ES_99"].values,
    }

    preds_per_model.append(entry)
    nans = np.isnan(garch_t_vol_pred).sum()
    if nans > 0:
        print(f"GARCH Student-t has {nans} NaN predictions")
except FileNotFoundError:
    print("GARCH Student-t predictions not found")

# GARCH Skewed-t Model
try:
    garch_skewt_df = pd.read_csv("predictions/garch_predictions_skewed_t.csv")
    garch_skewt_df["Date"] = pd.to_datetime(garch_skewt_df["Date"])
    garch_skewt_df = garch_skewt_df.set_index(["Date", "Symbol"])
    garch_skewt_dates = garch_skewt_df.index.get_level_values("Date")
    garch_skewt_df = garch_skewt_df[
        (
            (garch_skewt_dates >= TRAIN_VALIDATION_SPLIT)
            & (garch_skewt_dates < VALIDATION_TEST_SPLIT)
            if TEST_SET == "validation"
            else (garch_skewt_dates >= VALIDATION_TEST_SPLIT)
        )
    ]
    if np.isnan(garch_skewt_df).all().all():
        raise FileNotFoundError("All GARCH Skewed-t predictions are NaN")
    combined_df = df_validation.join(garch_skewt_df, how="left", rsuffix="_GARCH_skewt")
    garch_skewt_vol_pred = combined_df["GARCH_skewt_Vol"].values
    y_true = combined_df["LogReturn"].values
    mus = np.zeros_like(garch_skewt_vol_pred)
    nus = combined_df["GARCH_skewt_Nu"].values
    skew = combined_df["GARCH_skewt_Skew"].values
    crps = combined_df[
        "GARCH_skewt_CRPS"
    ].values  # crps_skewt(y_true, mus, garch_skewt_vol_pred, nus, skew)
    nll = combined_df[
        "NLL"
    ].values  # skewt_nll(y_true, garch_skewt_vol_pred, nus, skew, reduce=False)
    ece = ece_skewt(y_true, mus, garch_skewt_vol_pred, nus, skew)

    entry = {
        "name": "GARCH Skewed-t",
        "mean_pred": mus,
        "volatility_pred": garch_skewt_vol_pred,
        "symbols": combined_df.index.get_level_values("Symbol"),
        "dates": combined_df.index.get_level_values("Date"),
        "nll": nll,
        "crps": crps,
        "ece": ece,
        "LB_67": combined_df["LB_67"].values,
        "UB_67": combined_df["UB_67"].values,
        "LB_90": combined_df["LB_90"].values,
        "UB_90": combined_df["UB_90"].values,
        "LB_95": combined_df["LB_95"].values,
        "UB_95": combined_df["UB_95"].values,
        "LB_98": combined_df["LB_98"].values,
        "UB_98": combined_df["UB_98"].values,
        "ES_83.5": combined_df["ES_83.5"].values,
        "ES_95": combined_df["ES_95"].values,
        "ES_97.5": combined_df["ES_97.5"].values,
        "ES_99": combined_df["ES_99"].values,
    }
    preds_per_model.append(entry)
    nans = np.isnan(garch_skewt_vol_pred).sum()
    if nans > 0:
        print(f"GARCH Skewed-t has {nans} NaN predictions")
except FileNotFoundError:
    print("GARCH Skewed-t predictions not found")

# AR GARCH Model
for version in [
    "AR(3)-GARCH(1,1)-normal",
    "AR(3)-GARCH(1,1)-t",
    "AR(10)-GARCH(1,1)-normal",
]:
    try:
        ar_garch_preds = pd.read_csv(f"predictions/predictions_{version}.csv")
        ar_garch_preds["Date"] = pd.to_datetime(ar_garch_preds["Date"])
        ar_garch_preds = ar_garch_preds.set_index(["Date", "Symbol"])
        ar_garch_dates = ar_garch_preds.index.get_level_values("Date")
        ar_garch_preds = ar_garch_preds[
            (
                (ar_garch_dates >= TRAIN_VALIDATION_SPLIT)
                & (ar_garch_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (ar_garch_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(ar_garch_preds["AR_GARCH_Vol"]).all():
            raise FileNotFoundError(f"All {version} predictions are NaN")
        combined_df = df_validation.join(
            ar_garch_preds, how="left", rsuffix="_AR_GARCH"
        )
        ar_garch_vol_pred = combined_df["AR_GARCH_Vol"].values
        y_true = combined_df["LogReturn"].values
        mus = combined_df["AR_GARCH_Mean"].values
        crps = combined_df.get("AR_GARCH_CRPS")
        model_dist = version.split("-")[-1]
        if model_dist == "t":
            nus = combined_df["AR_GARCH_Nu"].values

        entry = {
            "name": version,
            "mean_pred": mus,
            "volatility_pred": ar_garch_vol_pred,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "nll": (
                student_t_nll(
                    y_true,
                    mus,
                    ar_garch_vol_pred,
                    nus,
                )
                if model_dist == "t"
                else nll_loss_mean_and_vol(
                    y_true,
                    mus,
                    ar_garch_vol_pred,
                )
            ),
            "crps": crps,
            "ece": (
                ece_student_t(y_true, mus, ar_garch_vol_pred, nus)
                if model_dist == "t"
                else ece_gaussian(y_true, mus, np.log(ar_garch_vol_pred**2))
            ),
            "LB_67": combined_df["LB_67"].values,
            "UB_67": combined_df["UB_67"].values,
            "LB_90": combined_df["LB_90"].values,
            "UB_90": combined_df["UB_90"].values,
            "LB_95": combined_df["LB_95"].values,
            "UB_95": combined_df["UB_95"].values,
            "LB_98": combined_df["LB_98"].values,
            "UB_98": combined_df["UB_98"].values,
            "ES_83.5": combined_df["ES_83.5"].values,
            "ES_95": combined_df["ES_95"].values,
            "ES_97.5": combined_df["ES_97.5"].values,
            "ES_99": combined_df["ES_99"].values,
        }

        preds_per_model.append(entry)
        nans = np.isnan(ar_garch_vol_pred).sum()
        if nans > 0:
            print(f"{version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"{version} predictions not found")

# HAR Model
for version in [
    "python",
    # "R",
]:
    try:
        har_preds = pd.read_csv(f"predictions/HAR_{version}.csv")
        har_preds["Date"] = pd.to_datetime(har_preds["Date"])
        har_preds = har_preds.set_index(["Date", "Symbol"])
        har_dates = har_preds.index.get_level_values("Date")
        har_preds = har_preds[
            (
                (har_dates >= TRAIN_VALIDATION_SPLIT)
                & (har_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (har_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(har_preds[f"HAR_vol_{version}"]).all():
            raise FileNotFoundError(f"All HAR_{version} predictions are NaN")
        combined_df = df_validation.join(har_preds, how="left", rsuffix="_HAR")
        har_vol_pred = combined_df[f"HAR_vol_{version}"].values
        y_true = combined_df["Total Return Test"].values
        mus = np.zeros_like(har_vol_pred)

        entry = {
            "name": f"HAR_{version}",
            "mean_pred": mus,
            "volatility_pred": har_vol_pred,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "nll": nll_loss_mean_and_vol(
                y_true,
                mus,
                har_vol_pred,
            ),
            "crps": crps_normal_univariate(y_true, mus, har_vol_pred),
            "ece": ece_gaussian(y_true, mus, np.log(har_vol_pred**2)),
        }

        for cl in ALL_CONFIDENCE_LEVELS:
            alpha = 1 - cl
            z_alpha = norm.ppf(1 - alpha / 2)
            lb = mus - z_alpha * har_vol_pred
            ub = mus + z_alpha * har_vol_pred
            entry[f"LB_{format_cl(cl)}"] = lb
            entry[f"UB_{format_cl(cl)}"] = ub
            es_alpha = alpha / 2
            entry[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
                np.ones_like(mus).reshape(-1, 1),
                mus.reshape(-1, 1),
                har_vol_pred.reshape(-1, 1),
                lb,
            )

        preds_per_model.append(entry)
        nans = np.isnan(har_vol_pred).sum()
        if nans > 0:
            print(f"HAR_{version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"HAR_{version} predictions not found")

# HARQ Model
for version in ["python"]:
    try:
        harq_preds = pd.read_csv(f"predictions/HARQ_{version}.csv")
        harq_preds["Date"] = pd.to_datetime(harq_preds["Date"])
        harq_preds = harq_preds.set_index(["Date", "Symbol"])
        harq_dates = harq_preds.index.get_level_values("Date")
        harq_preds = harq_preds[
            (
                (harq_dates >= TRAIN_VALIDATION_SPLIT)
                & (harq_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (harq_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(harq_preds[f"HARQ_vol_{version}"]).all():
            raise FileNotFoundError(f"All HARQ_{version} predictions are NaN")
        combined_df = df_validation.join(harq_preds, how="left", rsuffix="_HARQ")
        harq_vol_pred = combined_df[f"HARQ_vol_{version}"].values
        y_true = combined_df["Total Return Test"].values
        mus = np.zeros_like(harq_vol_pred)

        entry = {
            "name": f"HARQ_{version}",
            "mean_pred": mus,
            "volatility_pred": harq_vol_pred,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "nll": nll_loss_mean_and_vol(
                y_true,
                mus,
                harq_vol_pred,
            ),
            "crps": crps_normal_univariate(y_true, mus, harq_vol_pred),
            "ece": ece_gaussian(y_true, mus, np.log(harq_vol_pred**2)),
        }

        for cl in ALL_CONFIDENCE_LEVELS:
            alpha = 1 - cl
            z_alpha = norm.ppf(1 - alpha / 2)
            lb = mus - z_alpha * harq_vol_pred
            ub = mus + z_alpha * harq_vol_pred
            entry[f"LB_{format_cl(cl)}"] = lb
            entry[f"UB_{format_cl(cl)}"] = ub
            es_alpha = alpha / 2
            entry[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
                np.ones_like(mus).reshape(-1, 1),
                mus.reshape(-1, 1),
                harq_vol_pred.reshape(-1, 1),
                lb,
            )

        preds_per_model.append(entry)
        nans = np.isnan(harq_vol_pred).sum()
        if nans > 0:
            print(f"HARQ_{version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"HARQ_{version} predictions not found")

# HAR quantile regression model
for version in ["", "Q", "_IVOL"]:
    try:
        har_qreg_preds = pd.read_csv(f"predictions/HAR{version}_qreg_{TEST_SET}.csv")
        har_qreg_preds["Date"] = pd.to_datetime(har_qreg_preds["Date"])
        har_qreg_preds = har_qreg_preds.set_index(["Date", "Symbol"])
        har_qreg_dates = har_qreg_preds.index.get_level_values("Date")
        har_qreg_preds = har_qreg_preds[
            (
                (har_qreg_dates >= TRAIN_VALIDATION_SPLIT)
                & (har_qreg_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (har_qreg_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(har_qreg_preds["LB_67"]).all():
            raise FileNotFoundError(f"All HAR{version}_QREG predictions are NaN")
        combined_df = df_validation.join(
            har_qreg_preds, how="left", rsuffix="_HAR_QREG"
        )

        entry = {
            "name": f"HAR{version}-QREG",
            "mean_pred": np.nan,
            "volatility_pred": np.nan,
            "nll": np.nan,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
        }

        for cl in ALL_CONFIDENCE_LEVELS:
            es_alpha = (1 - cl) / 2
            for key in [
                f"LB_{format_cl(cl)}",
                f"UB_{format_cl(cl)}",
                f"ES_{format_cl(1-es_alpha)}",
            ]:
                if key not in combined_df.columns:
                    print(f"Missing {key} for HAR{version}_QREG predictions")
                    entry[key] = np.nan
                else:
                    entry[key] = combined_df[key].values

        preds_per_model.append(entry)
    except FileNotFoundError:
        print("HAR QREG predictions not found")


# Realized GARCH
for version in ["norm", "std"]:
    try:
        realized_garch_preds = pd.read_csv(
            f"predictions/realized_garch_forecast_{version}.csv"
        )
        realized_garch_preds["Date"] = pd.to_datetime(realized_garch_preds["Date"])
        realized_garch_preds = realized_garch_preds.set_index(["Date", "Symbol"])
        realized_garch_dates = realized_garch_preds.index.get_level_values("Date")
        realized_garch_preds = realized_garch_preds[
            (
                (realized_garch_dates >= TRAIN_VALIDATION_SPLIT)
                & (realized_garch_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (realized_garch_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(realized_garch_preds["Forecast_Volatility"]).all():
            raise FileNotFoundError("All Realized GARCH predictions are NaN")
        combined_df = df_validation.join(
            realized_garch_preds, how="left", rsuffix="_Realized_GARCH"
        )
        realized_garch_preds = combined_df["Forecast_Volatility"].values
        y_true = combined_df["LogReturn"].values
        mus = combined_df["Mean"].values

        entry = {
            "name": f"Realized GARCH {version}",
            "mean_pred": mus,
            "volatility_pred": realized_garch_preds,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "nll": nll_loss_mean_and_vol(
                y_true,
                mus,
                realized_garch_preds,
            ),
            "crps": crps_normal_univariate(y_true, mus, realized_garch_preds),
            "ece": ece_gaussian(y_true, mus, np.log(realized_garch_preds**2)),
        }

        for cl in ALL_CONFIDENCE_LEVELS:
            alpha = 1 - cl
            z_alpha = norm.ppf(1 - alpha / 2)
            lb = mus - z_alpha * realized_garch_preds
            ub = mus + z_alpha * realized_garch_preds
            entry[f"LB_{format_cl(cl)}"] = lb
            entry[f"UB_{format_cl(cl)}"] = ub
            es_alpha = alpha / 2
            entry[f"ES_{format_cl(1-es_alpha)}"] = calculate_es_for_quantile(
                np.ones_like(mus).reshape(-1, 1),
                mus.reshape(-1, 1),
                realized_garch_preds.reshape(-1, 1),
                lb,
            )

        preds_per_model.append(entry)
        nans = np.isnan(realized_garch_preds).sum()
        if nans > 0:
            print(f"Realized GARCH {version} has {nans} NaN predictions")
    except FileNotFoundError:
        print("Realized GARCH predictions not found")

# LSTM MDN, new naming convention
for version in [
    "ivol-final-rolling",
    "rv-final-rolling",
    "rv-and-ivol-final-rolling",
]:
    try:
        fname = f"predictions/lstm_mdn_ensemble{SUFFIX}_v{version}_{TEST_SET}.csv"
        lstm_mdn_df = pd.read_csv(fname)
        lstm_mdn_df["Symbol"] = lstm_mdn_df["Symbol"].str.replace(".O", "")
        lstm_mdn_df["Date"] = pd.to_datetime(lstm_mdn_df["Date"])
        lstm_mdn_df = lstm_mdn_df.set_index(["Date", "Symbol"])
        lstm_mdn_dates = lstm_mdn_df.index.get_level_values("Date")
        lstm_mdn_df = lstm_mdn_df[
            (
                (lstm_mdn_dates >= TRAIN_VALIDATION_SPLIT)
                & (lstm_mdn_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (lstm_mdn_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(lstm_mdn_df["Mean_SP"]).all():
            raise Exception(f"All LSTM MDN {version} predictions are NaN")
        combined_df = df_validation.join(lstm_mdn_df, how="left", rsuffix="_LSTM_MDN")
        ece_col = combined_df.get("ECE")
        entry = {
            "name": f"LSTM MDN {version}",
            "mean_pred": combined_df["Mean_SP"].values,
            "volatility_pred": combined_df["Vol_SP"].values,
            "nll": combined_df.get("NLL", combined_df.get("loss")).values,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "crps": (
                crps.values if (crps := combined_df.get("CRPS")) is not None else None
            ),
            "p_up": combined_df.get("Prob_Increase"),
            "ece": ece_col.median() if ece_col is not None else None,
            "epistemic_var": combined_df.get("EpistemicVarMean"),
        }
        for cl in ALL_CONFIDENCE_LEVELS:
            lb = combined_df.get(f"LB_{format_cl(cl)}")
            ub = combined_df.get(f"UB_{format_cl(cl)}")
            if lb is None or ub is None:
                print(f"Missing {format_cl(cl)}% interval for LSTM MDN {version}")
            entry[f"LB_{format_cl(cl)}"] = lb
            entry[f"UB_{format_cl(cl)}"] = ub
            alpha = 1 - (1 - cl) / 2
            entry[f"ES_{format_cl(alpha)}"] = combined_df.get(f"ES_{format_cl(alpha)}")
        preds_per_model.append(entry)
        nans = combined_df["Mean_SP"].isnull().sum()
        if nans > 0:
            print(f"LSTM MDN {version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"LSTM MDN {version} predictions not found")
    except Exception as e:
        print(f"Issue loading LSTM MDN {version} predictions: {e}")

# Transformer MDN, new naming convention
for version in [
    "rvol",
    "ivol",
    "rvol-ivol",
]:
    try:
        transformer_df = pd.read_csv(
            f"predictions/transformer_mdn_ensemble_{version}_test_expanding.csv"
        )
        transformer_df["Symbol"] = transformer_df["Symbol"].str.replace(".O", "")
        transformer_df["Date"] = pd.to_datetime(transformer_df["Date"])
        transformer_df = transformer_df.set_index(["Date", "Symbol"])
        transformer_dates = transformer_df.index.get_level_values("Date")
        transformer_df = transformer_df[
            (
                (transformer_dates >= TRAIN_VALIDATION_SPLIT)
                & (transformer_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (transformer_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(transformer_df["Mean_SP"]).all():
            raise FileNotFoundError(
                f"All Transformer MDN {version} predictions are NaN"
            )
        combined_df = df_validation.join(
            transformer_df, how="left", rsuffix="_Transformer_MDN"
        )
        ece_col = combined_df.get("ECE")
        entry = {
            "name": f"Transformer MDN {version} expanding",
            "mean_pred": combined_df["Mean_SP"].values,
            "volatility_pred": combined_df["Vol_SP"].values,
            "epistemic_var": combined_df.get("EpistemicVarMean"),
            "nll": combined_df.get("NLL", combined_df.get("loss")).values,
            "dates": combined_df.index.get_level_values("Date"),
            "symbols": combined_df.index.get_level_values("Symbol"),
            "crps": (
                crps.values if (crps := combined_df.get("CRPS")) is not None else None
            ),
            "p_up": combined_df.get("Prob_Increase"),
            "ece": ece_col.median() if ece_col is not None else None,
        }
        for cl in ALL_CONFIDENCE_LEVELS:
            lb = combined_df.get(f"LB_{format_cl(cl)}")
            ub = combined_df.get(f"UB_{format_cl(cl)}")
            if lb is None or ub is None:
                print(
                    f"Missing {format_cl(cl)}% interval for Transformer MDN {version} expanding"
                )
            entry[f"LB_{format_cl(cl)}"] = lb
            entry[f"UB_{format_cl(cl)}"] = ub
            alpha = 1 - (1 - cl) / 2
            entry[f"ES_{format_cl(alpha)}"] = combined_df.get(f"ES_{format_cl(alpha)}")
        preds_per_model.append(entry)
        nans = combined_df["Mean_SP"].isnull().sum()
        if nans > 0:
            print(f"Transformer MDN {version} expanding has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"Transformer MDN {version} expanding predictions not found")

# Ensemble MDN
for version in ["iv", "rv", "rv-iv"]:
    try:
        fname = f"predictions/mdn_ensemble_{version}_{TEST_SET}_expanding.csv"
        ensemble_df = pd.read_csv(fname)
        ensemble_df["Symbol"] = ensemble_df["Symbol"].str.replace(".O", "")
        ensemble_df["Date"] = pd.to_datetime(ensemble_df["Date"])
        ensemble_df = ensemble_df.set_index(["Date", "Symbol"])
        ensemble_dates = ensemble_df.index.get_level_values("Date")
        ensemble_df = ensemble_df[
            (
                (ensemble_dates >= TRAIN_VALIDATION_SPLIT)
                & (ensemble_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (ensemble_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(ensemble_df["Mean_SP"]).all():
            raise FileNotFoundError(f"All Ensemble MDN {version} predictions are NaN")
        combined_df = df_validation.join(
            ensemble_df, how="left", rsuffix="_Ensemble_MDN"
        )
        ece_col = combined_df.get("ECE")
        entry = {
            "name": f"Ensemble MDN {version}",
            "mean_pred": combined_df["Mean_SP"].values,
            "volatility_pred": combined_df["Vol_SP"].values,
            "epistemic_var": combined_df.get("EpistemicVarMean"),
            "nll": combined_df.get("NLL", combined_df.get("loss")).values,
            "dates": combined_df.index.get_level_values("Date"),
            "symbols": combined_df.index.get_level_values("Symbol"),
            "crps": (
                crps.values if (crps := combined_df.get("CRPS")) is not None else None
            ),
            "p_up": combined_df.get("Prob_Increase"),
            "ece": ece_col.median() if ece_col is not None else None,
        }
        for cl in ALL_CONFIDENCE_LEVELS:
            lb = combined_df.get(f"LB_{format_cl(cl)}")
            ub = combined_df.get(f"UB_{format_cl(cl)}")
            if lb is None or ub is None:
                print(f"Missing {format_cl(cl)}% interval for Ensemble MDN {version}")
            entry[f"LB_{format_cl(cl)}"] = lb
            entry[f"UB_{format_cl(cl)}"] = ub
            alpha = 1 - (1 - cl) / 2
            entry[f"ES_{format_cl(alpha)}"] = combined_df.get(f"ES_{format_cl(alpha)}")
        preds_per_model.append(entry)
        nans = combined_df["Mean_SP"].isnull().sum()
        if nans > 0:
            print(f"Ensemble MDN {version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"Ensemble MDN {version} predictions not found")

# LSTM QR
for version in ["iv", "rv", "rv-iv"]:
    try:
        fname = f"predictions/lstm_qr{SUFFIX}_{version}.csv"
        lstm_qr_df = pd.read_csv(fname)
        lstm_qr_df["Symbol"] = lstm_qr_df["Symbol"].str.replace(".O", "")
        lstm_qr_df["Date"] = pd.to_datetime(lstm_qr_df["Date"])
        lstm_qr_df = lstm_qr_df.set_index(["Date", "Symbol"])
        lstm_mdn_dates = lstm_qr_df.index.get_level_values("Date")
        lstm_qr_df = lstm_qr_df[
            (
                (lstm_mdn_dates >= TRAIN_VALIDATION_SPLIT)
                & (lstm_mdn_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (lstm_mdn_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(lstm_qr_df["LB_90"]).all():
            raise Exception(f"All LSTM QREG {version} predictions are NaN")
        combined_df = df_validation.join(lstm_qr_df, how="left", rsuffix="_LSTM_MDN")
        ece_col = combined_df.get("ECE")
        combined_df["Mean"] = np.nan
        entry = {
            "name": f"LSTM QREG {version}",
            "mean_pred": combined_df["Mean"],
            "volatility_pred": None,
            "nll": None,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "crps": None,
            "ece": None,
        }
        for cl in ALL_CONFIDENCE_LEVELS:
            lb = combined_df.get(f"LB_{format_cl(cl)}")
            ub = combined_df.get(f"UB_{format_cl(cl)}")
            if lb is None or ub is None:
                print(f"Missing {format_cl(cl)}% interval for LSTM QREG {version}")
            entry[f"LB_{format_cl(cl)}"] = lb
            entry[f"UB_{format_cl(cl)}"] = ub
            alpha = 1 - (1 - cl) / 2
            entry[f"ES_{format_cl(alpha)}"] = combined_df.get(f"ES_{format_cl(alpha)}")
        preds_per_model.append(entry)
        nans = combined_df["LB_90"].isnull().sum()
        if nans > 0:
            print(f"LSTM QREG {version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"LSTM QREG {version} predictions not found")
    except Exception as e:
        print(f"Issue loading LSTM QREG {version} predictions: {e}")

# VAE based models
for version in [4]:
    for predictor in [
        # "simple_regressor_on_means",
        # "simple_regressor_on_samples",
        # "simple_regressor_on_means_and_std",
        # "ffnn_on_latent_means",
        # "ffnn_on_latent_means_and_std",
        # "ffnn_on_latent_samples",
    ]:
        try:
            pred_df = pd.read_csv(
                f"predictions/vae_lstm_mdm_{LOOKBACK_DAYS}_days{SUFFIX}_v{version}_{predictor}.csv"
            )
            pred_df["Symbol"] = pred_df["Symbol"].str.replace(".O", "")
            pred_df["Date"] = pd.to_datetime(pred_df["Date"])
            pred_df = pred_df.set_index(["Date", "Symbol"])
            dates = pred_df.index.get_level_values("Date")
            pred_df = pred_df[
                (
                    (dates >= TRAIN_VALIDATION_SPLIT) & (dates < VALIDATION_TEST_SPLIT)
                    if TEST_SET == "validation"
                    else (dates >= VALIDATION_TEST_SPLIT)
                )
            ]
            if np.isnan(pred_df["Pred_Mean"]).all():
                raise FileNotFoundError(f"All VAE MDN {version} predictions are NaN")
            combined_df = df_validation.join(
                pred_df, how="left", rsuffix="_Transformer_MDN"
            )
            name = f"VAE MDN {version} {predictor}"
            ece_col = combined_df.get("ECE")
            entry = {
                "name": name,
                "mean_pred": combined_df["Pred_Mean"].values,
                "volatility_pred": combined_df["Pred_Std"].values,
                "nll": combined_df.get("NLL", combined_df.get("loss")).values,
                "symbols": combined_df.index.get_level_values("Symbol"),
                "dates": combined_df.index.get_level_values("Date"),
                "crps": (
                    crps.values
                    if (crps := combined_df.get("CRPS")) is not None
                    else None
                ),
                "p_up": combined_df.get("Prob_Increase"),
                "ece": ece_col.median() if ece_col is not None else None,
            }
            for cl in ALL_CONFIDENCE_LEVELS:
                lb = combined_df.get(f"LB_{format_cl(cl)}")
                ub = combined_df.get(f"UB_{format_cl(cl)}")
                if lb is None or ub is None:
                    print(f"Missing {format_cl(cl)}% interval for {name}")
                entry[f"LB_{format_cl(cl)}"] = lb
                entry[f"UB_{format_cl(cl)}"] = ub
                alpha = 1 - (1 - cl) / 2
                entry[f"ES_{format_cl(alpha)}"] = combined_df.get(
                    f"ES_{format_cl(alpha)}"
                )
            preds_per_model.append(entry)
            nans = combined_df["Pred_Mean"].isnull().sum()
            if nans > 0:
                print(f"{name} has {nans} NaN predictions")
        except FileNotFoundError:
            print(f"{name} predictions not found")


# LSTM MAF
for version in []:  # ["v2", "v3", "v4"]:
    try:
        lstm_maf_preds = pd.read_csv(f"predictions/lstm_MAF_{version}{SUFFIX}.csv")
        lstm_maf_preds["Date"] = pd.to_datetime(lstm_maf_preds["Date"])
        lstm_maf_preds = lstm_maf_preds.set_index(["Date", "Symbol"])
        lstm_maf_dates = lstm_maf_preds.index.get_level_values("Date")
        lstm_maf_preds = lstm_maf_preds[
            (
                (lstm_maf_dates >= TRAIN_VALIDATION_SPLIT)
                & (lstm_maf_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (lstm_maf_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(lstm_maf_preds["Mean_SP"]).all():
            raise FileNotFoundError(f"All LSTM MAF {version} predictions are NaN")
        combined_df = df_validation.join(lstm_maf_preds, how="left", rsuffix="_MAF")
        ece_col = combined_df.get("ECE")
        preds_per_model.append(
            {
                "name": f"LSTM MAF {version}",
                "mean_pred": combined_df["Mean_SP"].values,
                "volatility_pred": combined_df["Vol_SP"].values,
                "LB_50": combined_df["LB_50"].values,
                "UB_50": combined_df["UB_50"].values,
                "LB_67": combined_df["LB_67"].values,
                "UB_67": combined_df["UB_67"].values,
                "LB_90": combined_df["LB_90"].values,
                "UB_90": combined_df["UB_90"].values,
                "LB_95": combined_df["LB_95"].values,
                "UB_95": combined_df["UB_95"].values,
                "LB_97.5": combined_df["LB_97"].values,
                "UB_97.5": combined_df["UB_97"].values,
                "LB_99": combined_df["LB_99"].values,
                "UB_99": combined_df["UB_99"].values,
                "nll": combined_df["NLL"].values,
                "symbols": combined_df.index.get_level_values("Symbol"),
                "dates": combined_df.index.get_level_values("Date"),
                # "crps": lstm_mdn_preds["CRPS"].values.mean(),
                "ece": ece_col.median() if ece_col is not None else None,
            }
        )
        nans = combined_df["Mean_SP"].isnull().sum()
        if nans > 0:
            print(f"LSTM MAF {version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"LSTM MAF {version} predictions not found")

for version in ["v1"]:
    try:
        vae = pd.read_csv(f"predictions/vae_lstm_{version}{SUFFIX}.csv")
        vae["Date"] = pd.to_datetime(vae["Date"])
        vae = vae.set_index(["Date", "Symbol"])
        vae_dates = vae.index.get_level_values("Date")
        vae = vae[
            (
                (vae_dates >= TRAIN_VALIDATION_SPLIT)
                & (vae_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (vae_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(vae["Mean_SP"]).all():
            raise FileNotFoundError(f"All VAE {version} predictions are NaN")
        combined_df = df_validation.join(vae, how="left", rsuffix="_VAE")
        ece_col = combined_df.get("ECE")
        preds_per_model.append(
            {
                "name": f"VAE {version}",
                "mean_pred": combined_df["Mean_SP"].values,
                "volatility_pred": combined_df["Vol_SP"].values,
                "LB_67": combined_df["LB_67"].values,
                "UB_67": combined_df["UB_67"].values,
                "LB_90": combined_df["LB_90"].values,
                "UB_90": combined_df["UB_90"].values,
                "LB_95": combined_df["LB_95"].values,
                "UB_95": combined_df["UB_95"].values,
                "LB_99": combined_df["LB_99"].values,
                "UB_99": combined_df["UB_99"].values,
                "nll": nll_loss_mean_and_vol(
                    y_true,
                    mus,
                    garch_vol_pred,
                ),
                "symbols": combined_df.index.get_level_values("Symbol"),
                "dates": combined_df.index.get_level_values("Date"),
                # "crps": lstm_mdn_preds["CRPS"].values.mean(),
                "ece": ece_col.median() if ece_col is not None else None,
            }
        )
        nans = combined_df["Mean_SP"].isnull().sum()
        if nans > 0:
            print(f"VAE {version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"VAE {version} predictions not found")

# BENCHMARK MODELS
#############################################
for version in ["RV", "IV", "RV_IV"]:
    try:
        catboost_preds = pd.read_csv(f"predictions/CatBoost_{version}_4y.csv")
        catboost_preds["Date"] = pd.to_datetime(catboost_preds["Date"])
        catboost_preds = catboost_preds.set_index(["Date", "Symbol"])
        catboost_dates = catboost_preds.index.get_level_values("Date")
        catboost_preds = catboost_preds[
            (
                (catboost_dates >= TRAIN_VALIDATION_SPLIT)
                & (catboost_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (catboost_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(catboost_preds).all().all():
            raise FileNotFoundError(f"All Catboost {version} _4y predictions are NaN")
        combined_df = df_validation.join(
            catboost_preds, how="left", rsuffix="_Catboost_4y"
        )
        # make a Mean_SP column full of 0s for now
        combined_df["Mean_SP"] = np.nan
        # same for Vol_SP
        combined_df["Vol_SP"] = np.nan
        # same for NLL
        combined_df["nll"] = np.nan

        preds_per_model.append(
            {
                "name": f"Benchmark Catboost {version}",
                "mean_pred": combined_df["Mean_SP"].values,
                "volatility_pred": combined_df["Vol_SP"].values,
                "nll": combined_df["nll"].values,
                "symbols": combined_df.index.get_level_values("Symbol"),
                "dates": combined_df.index.get_level_values("Date"),
                "LB_98": combined_df.get("Quantile_0.010"),
                "UB_98": combined_df.get("Quantile_0.990"),
                "LB_95": combined_df.get("Quantile_0.025"),
                "UB_95": combined_df.get("Quantile_0.975"),
                "LB_90": combined_df.get("Quantile_0.050"),
                "UB_90": combined_df.get("Quantile_0.950"),
                "LB_67": combined_df.get("Quantile_0.165"),
                "UB_67": combined_df.get("Quantile_0.835"),
                "ES_83.5": combined_df.get("ES_0.165"),
                "ES_16.5": combined_df.get("ES_0.835"),
                "ES_99": combined_df.get("ES_0.010"),
                "ES_97.5": combined_df.get("ES_0.025"),
                "ES_95": combined_df.get("ES_0.050"),
                "ES_0.05": combined_df.get("ES_0.950"),
                "ES_0.025": combined_df.get("ES_0.975"),
                "ES_0.01": combined_df.get("ES_0.990"),
            }
        )
        # nans = combined_df["Mean_SP"].isnull().sum()
        nans = 0
        if nans > 0:
            print(f"Catboost_4y has {nans} NaN predictions")
    except FileNotFoundError:
        print("Catboost_4y predictions not found")

for version in ["RV", "IV", "RV_IV"]:
    try:
        lightGBMpreds = pd.read_csv(f"predictions/LightGBM_{version}_4y.csv")
        lightGBMpreds["Date"] = pd.to_datetime(lightGBMpreds["Date"])
        lightGBMpreds = lightGBMpreds.set_index(["Date", "Symbol"])
        lightGBM_dates = lightGBMpreds.index.get_level_values("Date")
        lightGBMpreds = lightGBMpreds[
            (
                (lightGBM_dates >= TRAIN_VALIDATION_SPLIT)
                & (lightGBM_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (lightGBM_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(lightGBMpreds).all().all():
            raise FileNotFoundError(f"All LightGBM_4y {version} predictions are NaN")
        combined_df = df_validation.join(
            lightGBMpreds, how="left", rsuffix="_LIGHTGBM_4y"
        )
        # make a Mean_SP column full of 0s for now
        combined_df["Mean_SP"] = np.nan
        # same for Vol_SP
        combined_df["Vol_SP"] = np.nan
        # same for NLL
        combined_df["nll"] = np.nan

        preds_per_model.append(
            {
                "name": f"Benchmark LightGBM {version}",
                "mean_pred": combined_df["Mean_SP"].values,
                "volatility_pred": combined_df["Vol_SP"].values,
                "nll": combined_df["nll"].values,
                "symbols": combined_df.index.get_level_values("Symbol"),
                "dates": combined_df.index.get_level_values("Date"),
                "LB_98": combined_df.get("Quantile_0.010"),
                "UB_98": combined_df.get("Quantile_0.990"),
                "LB_95": combined_df.get("Quantile_0.025"),
                "UB_95": combined_df.get("Quantile_0.975"),
                "LB_90": combined_df.get("Quantile_0.050"),
                "UB_90": combined_df.get("Quantile_0.950"),
                "LB_67": combined_df.get("Quantile_0.165"),
                "UB_67": combined_df.get("Quantile_0.835"),
                "ES_83.5": combined_df.get("ES_0.165"),
                "ES_16.5": combined_df.get("ES_0.835"),
                "ES_99": combined_df.get("ES_0.010"),
                "ES_97.5": combined_df.get("ES_0.025"),
                "ES_95": combined_df.get("ES_0.050"),
                "ES_0.05": combined_df.get("ES_0.950"),
                "ES_0.025": combined_df.get("ES_0.975"),
                "ES_0.01": combined_df.get("ES_0.990"),
            }
        )
        # nans = combined_df["Mean_SP"].isnull().sum()
        nans = 0
        if nans > 0:
            print(f"LightGBM_4y has {nans} NaN predictions")
    except FileNotFoundError:
        print("LightGBM_4y predictions not found")

for version in ["RV", "IV", "RV_IV"]:
    try:
        xg_boost_preds = pd.read_csv(f"predictions/XGBoost_{version}_4y.csv")
        xg_boost_preds["Date"] = pd.to_datetime(xg_boost_preds["Date"])
        xg_boost_preds = xg_boost_preds.set_index(["Date", "Symbol"])
        xg_boost_dates = xg_boost_preds.index.get_level_values("Date")
        xg_boost_preds = xg_boost_preds[
            (
                (xg_boost_dates >= TRAIN_VALIDATION_SPLIT)
                & (xg_boost_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (xg_boost_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(xg_boost_preds).all().all():
            raise FileNotFoundError(f"All XGBoost_4y {version} predictions are NaN")
        combined_df = df_validation.join(
            xg_boost_preds, how="left", rsuffix="_XGBoost_4y"
        )
        # make a Mean_SP column full of 0s for now
        combined_df["Mean_SP"] = np.nan
        # same for Vol_SP
        combined_df["Vol_SP"] = np.nan
        # same for NLL
        combined_df["nll"] = np.nan

        preds_per_model.append(
            {
                "name": f"Benchmark XGBoost {version}",
                "mean_pred": combined_df["Mean_SP"].values,
                "volatility_pred": combined_df["Vol_SP"].values,
                "nll": combined_df["nll"].values,
                "symbols": combined_df.index.get_level_values("Symbol"),
                "dates": combined_df.index.get_level_values("Date"),
                "LB_98": combined_df.get("Quantile_0.010"),
                "UB_98": combined_df.get("Quantile_0.990"),
                "LB_95": combined_df.get("Quantile_0.025"),
                "UB_95": combined_df.get("Quantile_0.975"),
                "LB_90": combined_df.get("Quantile_0.050"),
                "UB_90": combined_df.get("Quantile_0.950"),
                "LB_67": combined_df.get("Quantile_0.165"),
                "UB_67": combined_df.get("Quantile_0.835"),
                "ES_83.5": combined_df.get("ES_0.165"),
                "ES_16.5": combined_df.get("ES_0.835"),
                "ES_99": combined_df.get("ES_0.010"),
                "ES_97.5": combined_df.get("ES_0.025"),
                "ES_95": combined_df.get("ES_0.050"),
                "ES_0.05": combined_df.get("ES_0.950"),
                "ES_0.025": combined_df.get("ES_0.975"),
                "ES_0.01": combined_df.get("ES_0.990"),
            }
        )
        # nans = combined_df["Mean_SP"].isnull().sum()
        nans = 0
        if nans > 0:
            print(f"XGBoost_4y has {nans} NaN predictions")
    except FileNotFoundError:
        print("XGBoost_4y predictions not found")

for version in ["RV", "IV", "RV_IV"]:
    try:
        DB_preds = pd.read_csv(f"predictions/DB_{version}.csv")
        DB_preds["Date"] = pd.to_datetime(DB_preds["Date"])
        DB_preds = DB_preds.set_index(["Date", "Symbol"])
        DB_dates = DB_preds.index.get_level_values("Date")
        DB_preds = DB_preds[
            (
                (DB_dates >= TRAIN_VALIDATION_SPLIT)
                & (DB_dates < VALIDATION_TEST_SPLIT)
                if TEST_SET == "validation"
                else (DB_dates >= VALIDATION_TEST_SPLIT)
            )
        ]
        if np.isnan(DB_preds).all().all():
            raise FileNotFoundError(f"All DB RV_only predictions are NaN")
        combined_df = df_validation.join(DB_preds, how="left", rsuffix="_DB")
        combined_df["Mean_SP"] = np.nan
        # same for Vol_SP
        combined_df["Vol_SP"] = np.nan
        # same for NLL
        combined_df["nll"] = np.nan

        key = "RV_set + IV_set" if version == "RV_IV" else f"{version}_set"

        preds_per_model.append(
            {
                "name": f"Benchmark DB {version}",
                "mean_pred": combined_df["Mean_SP"].values,
                "volatility_pred": combined_df["Vol_SP"].values,
                "nll": combined_df["nll"].values,
                "symbols": combined_df.index.get_level_values("Symbol"),
                "dates": combined_df.index.get_level_values("Date"),
                "LB_98": combined_df[f"DB_{key}_0.01"].values,
                "UB_98": combined_df[f"DB_{key}_0.99"].values,
                "LB_95": combined_df[f"DB_{key}_0.025"].values,
                "UB_95": combined_df[f"DB_{key}_0.975"].values,
                "LB_90": combined_df[f"DB_{key}_0.05"].values,
                "UB_90": combined_df[f"DB_{key}_0.95"].values,
                "LB_67": combined_df[f"DB_{key}_0.165"].values,
                "UB_67": combined_df[f"DB_{key}_0.835"].values,
                "ES_99": combined_df[f"DB_{key}_ES_0.01"].values,
                "ES_97.5": combined_df[f"DB_{key}_ES_0.025"].values,
                "ES_95": combined_df[f"DB_{key}_ES_0.05"].values,
                "ES_83.5": combined_df[f"DB_{key}_ES_0.165"].values,
                "ES_16.5": combined_df[f"DB_{key}_ES_0.835"].values,
                "ES_0.05": combined_df[f"DB_{key}_ES_0.95"].values,
                "ES_0.025": combined_df[f"DB_{key}_ES_0.975"].values,
                "ES_0.01": combined_df[f"DB_{key}_ES_0.99"].values,
            }
        )
    except FileNotFoundError:
        print(f"DB {version} predictions not found")
###########################################

# %%
# Exclude models with incomplete periods if requested
if MAX_NAN_THRESH:
    keep = []
    for model in preds_per_model:
        # if the model name does not contain "Benchmark"
        if "Benchmark" in model["name"]:
            nans = 0
        else:
            nans = pd.Series(model["LB_67"]).isnull().sum()

        if nans > len(model["LB_67"]) * MAX_NAN_THRESH:
            print(f"Excluding {model['name']} because it has {nans} NaN predictions")
            continue
        keep.append(model)
    preds_per_model = keep
    print("Remaining models:", len(preds_per_model))
    for i, entry in enumerate(preds_per_model):
        print(i, entry["name"])


# %%
# Create function for inspecting entries
def inspect_entry():
    for i, entry in enumerate(preds_per_model):
        print(i, entry["name"], flush=True)
    i = int(input("Enter index: "))
    m = preds_per_model[i]
    print("=========================")
    print("Selected", m["name"])
    print("=========================")
    print("Keys in entry:")
    print(list(m.keys()))
    return m


# %%
def calculate_picp(y_true, lower_bounds, upper_bounds):
    invalid = np.isnan(lower_bounds) | np.isnan(upper_bounds)
    # Mark entries with any NaN bounds as NaN so they don't contribute to the average.
    within_bounds = np.where(
        invalid,
        np.nan,
        np.logical_and(y_true >= lower_bounds, y_true <= upper_bounds),
    )
    picp = np.nanmean(within_bounds)
    return picp, within_bounds


def calculate_mpiw(lower_bounds, upper_bounds):
    mpiw = np.nanmean(upper_bounds - lower_bounds)
    return mpiw


def calculate_interval_score(y_true, lower_bounds, upper_bounds, alpha):
    valid = ~np.isnan(lower_bounds) & ~np.isnan(upper_bounds)
    interval_width = upper_bounds - lower_bounds
    penalties = (2 / alpha) * (
        (lower_bounds - y_true) * (y_true < lower_bounds)
        + (y_true - upper_bounds) * (y_true > upper_bounds)
    )
    interval_scores = interval_width + penalties
    # Exclude scores where any bound is NaN
    interval_scores = np.where(valid, interval_scores, np.nan)
    mean_interval_score = np.nanmean(interval_scores)
    return mean_interval_score


def calculate_uncertainty_error_correlation(y_true, mean_pred, interval_width):
    prediction_errors = np.abs(y_true - mean_pred)
    correlation = pd.Series(prediction_errors).corr(pd.Series(np.array(interval_width)))
    return correlation


def interpret_christoffersen_stat(p_value):
    if p_value < 0.05:
        return "❌"
    elif p_value > 0.05:
        return "✅"
    else:
        return "?"


def interpret_christoffersen_test(result):
    return pd.DataFrame(
        {
            "Test": ["Unconditional Coverage", "Independence", "Conditional Coverage"],
            "H0": [
                "VaR exceedances matches expected rate",
                "VaR exceedances occur independently over time",
                "VaR exceedances are both as expected and independent",
            ],
            "Statistic": [result["LR_uc"], result["LR_ind"], result["LR_cc"]],
            "p-value": [
                round(result["p_value_uc"], 3),
                round(result["p_value_ind"], 3),
                round(result["p_value_cc"], 3),
            ],
            "Good?": [
                interpret_christoffersen_stat(result["p_value_uc"]),
                interpret_christoffersen_stat(result["p_value_ind"]),
                interpret_christoffersen_stat(result["p_value_cc"]),
            ],
        }
    )


def interpret_bayer_dimitriadis_stat(p_value):
    if p_value < 0.05:
        return "❌"
    elif p_value > 0.05:
        return "✅"
    else:
        return "?"


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2))


# %%
# Evaluate models
y_test_actual = df_validation["LogReturn"].values
y_test_RV = df_validation["RV_5_daily"].values
abs_returns_test = np.abs(y_test_actual)

for entry in preds_per_model:
    # Calculate prediction intervals
    if "HAR" in entry["name"]:
        y_test_actual = df_validation["Total Return Test"].values
        print(f"Using Total Return Test for {entry['name']}")
    for cl in CONFIDENCE_LEVELS:
        cl_str = format_cl(cl)
        if entry.get(f"LB_{cl_str}") is None or entry.get(f"UB_{cl_str}") is None:
            continue

        # Calculate PICP and MPIW
        picp, within_bounds = calculate_picp(
            y_test_actual, entry[f"LB_{cl_str}"], entry[f"UB_{cl_str}"]
        )
        mpiw = calculate_mpiw(entry[f"LB_{cl_str}"], entry[f"UB_{cl_str}"])
        entry[f"picp_{cl_str}"] = picp
        entry[f"mpiw_{cl_str}"] = mpiw
        entry[f"within_bounds_{cl_str}"] = within_bounds

        # Calculate lower tail exceedances and picp
        q = (1 - cl) / 2
        q_str = format_cl(q)
        lb_pred = entry[f"LB_{cl_str}"]
        lower_picp, above_lb = calculate_picp(
            y_test_actual, lb_pred, np.inf * np.ones_like(lb_pred)
        )
        entry[f"below_{q_str}"] = ~(above_lb.astype(bool))
        entry[f"picp_lb_{q_str}"] = lower_picp

        # Calculate Interval Scores
        interval_score = calculate_interval_score(
            y_test_actual, entry[f"LB_{cl_str}"], entry[f"UB_{cl_str}"], cl
        )
        entry[f"interval_score_{cl_str}"] = interval_score

        # Calculate quantile loss for the lower bound
        entry[f"quantile_loss_{format_cl(q)}"] = np.where(
            y_test_actual >= lb_pred,
            q * (y_test_actual - lb_pred),
            (1 - q) * (lb_pred - y_test_actual),
        )

        # Calculate interval pinball loss (symmetric interval score)
        entry[f"interval_pinball_loss_{cl_str}"] = np.maximum(
            y_test_actual - entry[f"UB_{cl_str}"],
            entry[f"LB_{cl_str}"] - y_test_actual,
        )

        # Calculate Lopez loss function
        entry[f"lopez_loss_{cl_str}"] = np.nanmean(
            np.where(
                y_test_actual < entry[f"LB_{cl_str}"],
                1 + (entry[f"LB_{cl_str}"] - y_test_actual) ** 2,
                0,
            )
        )

        # Uncertainty-Error Correlation
        interval_width = entry[f"UB_{cl_str}"] - entry[f"LB_{cl_str}"]

        exceedance_df = pd.DataFrame(
            {
                "Symbol": entry["symbols"],
                "Within Bounds": above_lb,
            }
        )
        exceedance_df = exceedance_df.dropna(subset=["Within Bounds"])
        if len(exceedance_df) == 0:
            print(f"No valid data for {entry['name']}")
            continue

        chr_results = []
        for symbol, within_bounds in exceedance_df.groupby("Symbol")["Within Bounds"]:
            if within_bounds.isna().any():
                print(f"Skipping {symbol} due to NaN values")
                continue
            exceedances = ~within_bounds.astype(bool)
            result = christoffersen_test(exceedances, q)
            chr_results.append(
                {**result, "Symbol": symbol, "Coverage": within_bounds.mean()}
            )
        chr_results_df = pd.DataFrame(chr_results).set_index("Symbol")
        chr_results_df["uc_pass"] = np.where(
            chr_results_df["p_value_uc"].isna(),
            np.nan,
            chr_results_df["p_value_uc"] > 0.05,
        )
        chr_results_df["ind_pass"] = np.where(
            chr_results_df["p_value_ind"].isna(),
            np.nan,
            chr_results_df["p_value_ind"] > 0.05,
        )
        chr_results_df["cc_pass"] = np.where(
            chr_results_df["p_value_cc"].isna(),
            np.nan,
            chr_results_df["p_value_cc"] > 0.05,
        )
        chr_results_df["all_pass"] = np.where(
            np.isnan(chr_results_df["uc_pass"])
            | np.isnan(chr_results_df["ind_pass"])
            | np.isnan(chr_results_df["cc_pass"]),
            np.nan,
            chr_results_df["uc_pass"].astype(bool)
            & chr_results_df["ind_pass"].astype(bool)
            & chr_results_df["cc_pass"].astype(bool),
        )

        # Print Christoffersen's Test Results
        if PRINT_CHRISTOFFERSEN_TESTS:
            print(
                f"\n{entry['name']} Average Christoffersen's Test Results ({cl_str}%):"
            )

        entry[f"chr_results_df_{q_str}"] = chr_results_df
        entry[f"uc_passes_{q_str}"] = int(chr_results_df["uc_pass"].sum())
        entry[f"uc_fails_{q_str}"] = (chr_results_df["uc_pass"] == 0).sum()
        entry[f"uc_nans_{q_str}"] = chr_results_df["uc_pass"].isna().sum()
        entry[f"ind_passes_{q_str}"] = int(chr_results_df["ind_pass"].sum())
        entry[f"ind_fails_{q_str}"] = (chr_results_df["ind_pass"] == 0).sum()
        entry[f"ind_nans_{q_str}"] = chr_results_df["ind_pass"].isna().sum()
        entry[f"cc_passes_{q_str}"] = int(chr_results_df["cc_pass"].sum())
        entry[f"cc_fails_{q_str}"] = (chr_results_df["cc_pass"] == 0).sum()
        entry[f"cc_nans_{q_str}"] = chr_results_df["cc_pass"].isna().sum()
        if PRINT_CHRISTOFFERSEN_TESTS:
            print(
                f"Unconditional Coverage:\t{entry[f'uc_passes_{q_str}']} passes,\t{entry[f'uc_fails_{q_str}']} fails,\t{entry[f'uc_nans_{q_str}']} indeterminate\n"
                f"Independence:\t\t{entry[f'ind_passes_{q_str}']} passes,\t{entry[f'ind_fails_{q_str}']} fails,\t{entry[f'ind_nans_{q_str}']} indeterminate\n"
                f"Conditional Coverage:\t{entry[f'cc_passes_{q_str}']} passes,\t{entry[f'cc_fails_{q_str}']} fails,\t{entry[f'cc_nans_{q_str}']} indeterminate\n"
            )

        es_alpha = 1 - (1 - cl) / 2
        es_str = format_cl(es_alpha)
        es_pred = entry.get(f"ES_{es_str}")
        if es_pred is not None:
            # Code left in for now, but we have not found academic support for this approach.
            pool_df = pd.DataFrame(
                {
                    "Symbol": entry["symbols"],
                    "Date": entry["dates"],
                    "Actual": y_test_actual,
                    "VaR": np.array(entry[f"LB_{cl_str}"]),
                    "ES": np.array(es_pred),
                }
            )
            pool_df.dropna(subset=["ES"], inplace=True)
            pooled_bayer_dimitriadis_result = pooled_bayer_dimitriadis_test(
                pool_df.pivot_table("Actual", "Symbol", "Date"),
                pool_df.pivot_table("VaR", "Symbol", "Date"),
                pool_df.pivot_table("ES", "Symbol", "Date"),
                cl,
            )
            entry[f"pooled_bayer_dimitriadis_{es_str}"] = (
                pooled_bayer_dimitriadis_result
            )
            entry[f"pooled_bd_p_value_{es_str}"] = pooled_bayer_dimitriadis_result[
                "p_value"
            ]
            entry[f"pooled_bd_mean_violation_{es_str}"] = (
                pooled_bayer_dimitriadis_result["mean_z"]
            )

            es_df = pd.DataFrame(
                {
                    "Symbol": entry["symbols"],
                    "Actual": y_test_actual,
                    "LB": np.array(entry[f"LB_{cl_str}"]),
                    "ES": np.array(es_pred),
                }
            )
            passes = 0
            fails = 0
            indeterminate = 0
            # We should use both 0.05 (since it's the default) and 0.10 (which is stricter) to show that GARCH underestimates risk
            pass_threshold = 0.05
            # bd_results = {}
            # for symbol, group in es_df.groupby("Symbol"):
            #     test_stat, p = auxiliary_esr_test(
            #         group["Actual"].values, group["LB"].values, group["ES"].values, cl
            #     )
            #     if not np.isfinite(p):
            #         print(
            #             "WARNING: nan p-value in Bayer-Dimitriadis test for ",
            #             symbol,
            #             "at",
            #             cl_str,
            #         )
            #         indeterminate += 1
            #     elif p < pass_threshold:
            #         fails += 1
            #     else:
            #         passes += 1
            #     result = {
            #         "p_value": p,
            #         "test_stat": test_stat,
            #     }
            #     bd_results[symbol] = result
            #     result["n_violations"] = np.sum(group["Actual"] < group["LB"])
            # entry[f"bayer_dim_passes_{es_str}"] = passes
            # entry[f"bayer_dim_fails_{es_str}"] = fails
            # entry[f"bayer_dim_indeterminate_{es_str}"] = indeterminate
            # bd_results_df = pd.DataFrame(bd_results).T
            # bd_results_df["Pass"] = bd_results_df["p_value"] > pass_threshold
            # entry[f"bayer_dim_results_{es_str}"] = bd_results_df

            quantile = 1 - es_alpha
            entry[f"FZ0_{es_str}"] = fz_loss(
                y_test_actual, entry[f"LB_{cl_str}"], es_pred, quantile
            )
            entry[f"AL_{es_str}"] = al_loss(
                y_test_actual, entry[f"LB_{cl_str}"], es_pred, quantile
            )
        else:
            print(
                f"No ES_{es_str} predictions available for Pooled Bayer-Dimitriadis test."
            )

    # Calculate RMSE
    rmse = (
        calculate_rmse(y_test_actual, entry["mean_pred"])
        if entry["mean_pred"] is not None
        else np.nan
    )
    entry["rmse"] = rmse

    rmse_RV = (
        calculate_rmse(y_test_RV, entry["volatility_pred"] ** 2)
        if entry["volatility_pred"] is not None
        else np.nan
    )
    entry["rmse_RV"] = rmse_RV

    correlation = calculate_uncertainty_error_correlation(
        y_test_actual, entry["mean_pred"], interval_width
    )
    entry["uncertainty_error_correlation"] = correlation

    # Calculate sign of return accuracy
    sign_accuracy = np.nanmean(np.sign(y_test_actual) == np.sign(entry["mean_pred"]))
    entry["sign_accuracy"] = sign_accuracy

    # setting it back to correct value
    y_test_actual = df_validation["LogReturn"].values

# %%
# Compile results into DataFrame
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
interval_metric_keys = [
    "PICP",
    "PICP Miss",
    "Mean width (MPIW)",
    "Interval Score",
    "Interval Pinball Loss",
    "Lopez Loss",
]
es_metric_keys = [
    "Pooled Bayer-Dimitriadis pass?",
    "Pooled Bayer-Dimitriadis p-value",
    "Pooled Bayer-Dimitriadis mean violation",
    "Pooled Bayer-Dimitriadis violation SD",
    "Bayer-Dimitriadis passes",
    "Bayer-Dimitriadis fails",
    "Bayer-Dimitriadis indeterminate",
    "Bayer-Dimitriadis pass rate",
    "FZ Loss",
    "AL Loss",
]
quantile_metric_keys = [
    "QL",
    "Lower PICP",
    "Lower PICP Miss",
    "UC pass pct",
    "UC passes",
    "UC fails",
    "UC indeterminate",
    "Ind passes",
    "Ind fails",
    "Ind indeterminate",
    "CC passes",
    "CC fails",
    "CC indeterminate",
]
results = {
    "Model": [],
    # Non-quantile-based metrics
    "NLL": [],
    "ECE": [],
    "CRPS": [],
    "RMSE": [],
    "RMSE_RV": [],
    "Sign accuracy": [],
    "Correlation (vol. vs. errors)": [],
    # Interval based metrics
    **(
        {
            f"[{format_cl(cl)}] {key}": []
            for cl in CONFIDENCE_LEVELS
            for key in interval_metric_keys
        }
    ),
    # ES metrics
    **(
        {
            f"[{format_cl(1-(1-cl)/2)}] {key}": []
            for cl in CONFIDENCE_LEVELS
            for key in es_metric_keys
        }
    ),
    # Quantile metrics
    **(
        {
            f"[{format_cl((1-cl)/2)}] {key}": []
            for cl in CONFIDENCE_LEVELS
            for key in quantile_metric_keys
        }
    ),
}

for entry in preds_per_model:
    results["Model"].append(entry["name"])
    results["NLL"].append(
        np.nanmean(entry["nll"]) if entry["nll"] is not None else None
    )
    results["CRPS"].append(
        np.nanmean(crps) if (crps := entry.get("crps")) is not None else None
    )
    results["ECE"].append(entry.get("ece"))
    results["RMSE"].append(entry["rmse"])
    results["RMSE_RV"].append(entry["rmse_RV"])
    results["Sign accuracy"].append(entry["sign_accuracy"])
    results["Correlation (vol. vs. errors)"].append(
        entry["uncertainty_error_correlation"]
    )
    for cl in CONFIDENCE_LEVELS:
        cl_str = format_cl(cl)
        q = (1 - cl) / 2
        q_str = format_cl(q)
        if entry.get(f"picp_{cl_str}") is None:
            for key in interval_metric_keys:
                results[f"[{cl_str}] {key}"].append(np.nan)
            for key in quantile_metric_keys:
                results[f"[{q_str}] {key}"].append(np.nan)
            continue
        picp_miss = entry[f"picp_{cl_str}"] - cl
        results[f"[{cl_str}] PICP"].append(entry[f"picp_{cl_str}"])
        results[f"[{cl_str}] PICP Miss"].append(picp_miss)
        results[f"[{cl_str}] Mean width (MPIW)"].append(entry[f"mpiw_{cl_str}"])
        results[f"[{cl_str}] Interval Score"].append(entry[f"interval_score_{cl_str}"])
        results[f"[{cl_str}] Interval Pinball Loss"].append(
            np.nanmean(entry[f"interval_pinball_loss_{cl_str}"])
        )
        results[f"[{cl_str}] Lopez Loss"].append(entry[f"lopez_loss_{cl_str}"])
        results[f"[{q_str}] QL"].append(np.nanmean(entry.get(f"quantile_loss_{q_str}")))
        results[f"[{q_str}] Lower PICP"].append(entry[f"picp_lb_{q_str}"])
        results[f"[{q_str}] Lower PICP Miss"].append(
            entry[f"picp_lb_{q_str}"] - (1 - q)
        )
        uc_passes = entry[f"uc_passes_{q_str}"]
        uc_fails = entry[f"uc_fails_{q_str}"]
        uc_pass_pct = uc_passes / (uc_passes + uc_fails)
        results[f"[{q_str}] UC pass pct"].append(uc_pass_pct)
        results[f"[{q_str}] UC passes"].append(uc_passes)
        results[f"[{q_str}] UC fails"].append(uc_fails)
        results[f"[{q_str}] UC indeterminate"].append(entry[f"uc_nans_{q_str}"])
        results[f"[{q_str}] Ind passes"].append(entry[f"ind_passes_{q_str}"])
        results[f"[{q_str}] Ind fails"].append(entry[f"ind_fails_{q_str}"])
        results[f"[{q_str}] Ind indeterminate"].append(entry[f"ind_nans_{q_str}"])
        results[f"[{q_str}] CC passes"].append(entry[f"cc_passes_{q_str}"])
        results[f"[{q_str}] CC fails"].append(entry[f"cc_fails_{q_str}"])
        results[f"[{q_str}] CC indeterminate"].append(entry[f"cc_nans_{q_str}"])
    for cl in CONFIDENCE_LEVELS:
        es_alpha = 1 - (1 - cl) / 2
        es_str = format_cl(es_alpha)
        if entry.get(f"pooled_bayer_dimitriadis_{es_str}") is None:
            results[f"[{format_cl(es_alpha)}] Pooled Bayer-Dimitriadis pass?"].append(
                np.nan
            )
            results[f"[{format_cl(es_alpha)}] Pooled Bayer-Dimitriadis p-value"].append(
                np.nan
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis passes"].append(np.nan)
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis fails"].append(np.nan)
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis indeterminate"].append(
                np.nan
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis pass rate"].append(
                np.nan
            )
            results[
                f"[{format_cl(es_alpha)}] Pooled Bayer-Dimitriadis mean violation"
            ].append(np.nan)
            results[
                f"[{format_cl(es_alpha)}] Pooled Bayer-Dimitriadis violation SD"
            ].append(np.nan)
            results[f"[{format_cl(es_alpha)}] FZ Loss"].append(np.nan)
            results[f"[{format_cl(es_alpha)}] AL Loss"].append(np.nan)
        else:
            pooled_bd_test_result = entry[f"pooled_bayer_dimitriadis_{es_str}"]
            results[f"[{format_cl(es_alpha)}] Pooled Bayer-Dimitriadis pass?"].append(
                interpret_bayer_dimitriadis_stat(pooled_bd_test_result["p_value"])
            )
            results[f"[{format_cl(es_alpha)}] Pooled Bayer-Dimitriadis p-value"].append(
                pooled_bd_test_result["p_value"]
            )
            results[
                f"[{format_cl(es_alpha)}] Pooled Bayer-Dimitriadis mean violation"
            ].append(pooled_bd_test_result["mean_z"])
            results[
                f"[{format_cl(es_alpha)}] Pooled Bayer-Dimitriadis violation SD"
            ].append(pooled_bd_test_result["std_z"])
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis passes"].append(
                passes := entry.get(f"bayer_dim_passes_{es_str}")
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis fails"].append(
                fails := entry.get(f"bayer_dim_fails_{es_str}")
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis indeterminate"].append(
                entry.get(f"bayer_dim_indeterminate_{es_str}")
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis pass rate"].append(
                passes / (passes + fails) if passes or fails else np.nan
            )
            results[f"[{format_cl(es_alpha)}] FZ Loss"].append(
                np.mean(entry[f"FZ0_{es_str}"])
            )
            results[f"[{format_cl(es_alpha)}] AL Loss"].append(
                np.mean(entry[f"AL_{es_str}"])
            )


results_df = pd.DataFrame(results)
results_df = results_df.set_index("Model")

# Remove inadequate models
for model in results_df.index:
    if "GARCH" in model:
        continue
    if "Benchmark" in model:
        continue
    if "HAR" in model:
        continue
    if "DB" in model:
        continue
    # if "VAE MDN" in model:
    #     continue
    passes = 0
    fails = 0
    for cl in CONFIDENCE_LEVELS:
        q = (1 - cl) / 2
        q_str = format_cl(q)
        passes += results_df.loc[model, f"[{q_str}] CC passes"]
        fails += results_df.loc[model, f"[{q_str}] CC fails"]
    # if fails / (passes + fails) > 0.3:
    #     print(f"Removing {model} due to CC fails")
    #     results_df.drop(model, inplace=True)

# Identify winners
inadequate_models = {
    "HAR_R",
    "HARQ_R",
    "Benchmark XGBoost RV",
    "Benchmark LightGBM RV",
    "Benchmark LightGBM RV_IV",
}
# For some metrics, it is very important that we only compare adequate models.
# For others, we want to see how the inadequate models perform.
adequate_df = results_df.drop(index=inadequate_models, errors="ignore")
results_df.loc["Winner", "NLL"] = results_df["NLL"].idxmin()
results_df.loc["Winner", "ECE"] = results_df["ECE"].idxmin()
results_df.loc["Winner", "CRPS"] = results_df["CRPS"].idxmin()
results_df.loc["Winner", "Correlation (vol. vs. errors)"] = results_df[
    "Correlation (vol. vs. errors)"
].idxmax()
results_df.loc["Winner", "RMSE"] = results_df["RMSE"].idxmin()
results_df.loc["Winner", "RMSE_RV"] = results_df["RMSE_RV"].idxmin()
results_df.loc["Winner", "Sign accuracy"] = results_df["Sign accuracy"].idxmax()
for cl in CONFIDENCE_LEVELS:
    cl_str = format_cl(cl)
    results_df.loc["Winner", f"[{cl_str}] PICP"] = (
        results_df[f"[{cl_str}] PICP Miss"].abs().idxmin()
    )
    results_df.loc["Winner", f"[{cl_str}] PICP Miss"] = (
        results_df[f"[{cl_str}] PICP Miss"].abs().idxmin()
    )
    results_df.loc["Winner", f"[{cl_str}] Mean width (MPIW)"] = adequate_df[
        f"[{cl_str}] Mean width (MPIW)"
    ].idxmin()
    results_df.loc["Winner", f"[{cl_str}] Interval Score"] = adequate_df[
        f"[{cl_str}] Interval Score"
    ].idxmin()
    results_df.loc["Winner", f"[{cl_str}] Interval Pinball Loss"] = results_df[
        f"[{cl_str}] Interval Pinball Loss"
    ].idxmin()
    results_df.loc["Winner", f"[{cl_str}] Lopez Loss"] = results_df[
        f"[{cl_str}] Lopez Loss"
    ].idxmin()
    q = (1 - cl) / 2
    q_str = format_cl(q)
    for chr_test in ["UC", "Ind", "CC"]:
        results_df.loc["Winner", f"[{q_str}] {chr_test} passes"] = results_df[
            f"[{q_str}] {chr_test} passes"
        ].idxmax()
        results_df.loc["Winner", f"[{q_str}] {chr_test} fails"] = results_df[
            f"[{q_str}] {chr_test} fails"
        ].idxmin()
    results_df.loc["Winner", f"[{q_str}] UC pass pct"] = results_df[
        f"[{q_str}] UC pass pct"
    ].idxmax()
    results_df.loc["Winner", f"[{q_str}] QL"] = results_df[f"[{q_str}] QL"].idxmin()
    results_df.loc["Winner", f"[{q_str}] Lower PICP"] = (
        results_df[f"[{q_str}] Lower PICP Miss"].abs().idxmin()
    )
    results_df.loc["Winner", f"[{q_str}] Lower PICP Miss"] = (
        results_df[f"[{q_str}] Lower PICP Miss"].abs().idxmin()
    )
    es_alpha = 1 - (1 - cl) / 2
    es_str = format_cl(es_alpha)
    results_df.loc["Winner", f"[{es_str}] Pooled Bayer-Dimitriadis p-value"] = (
        results_df[f"[{es_str}] Pooled Bayer-Dimitriadis p-value"].idxmax()
    )
    results_df.loc["Winner", f"[{es_str}] Pooled Bayer-Dimitriadis mean violation"] = (
        results_df[f"[{es_str}] Pooled Bayer-Dimitriadis mean violation"].abs().idxmin()
    )
    results_df.loc["Winner", f"[{es_str}] Bayer-Dimitriadis passes"] = results_df[
        f"[{es_str}] Bayer-Dimitriadis passes"
    ].idxmax()
    results_df.loc["Winner", f"[{es_str}] Bayer-Dimitriadis fails"] = results_df[
        f"[{es_str}] Bayer-Dimitriadis fails"
    ].idxmin()
    results_df.loc["Winner", f"[{es_str}] FZ Loss"] = results_df[
        f"[{es_str}] FZ Loss"
    ].idxmin()
    results_df.loc["Winner", f"[{es_str}] AL Loss"] = results_df[
        f"[{es_str}] AL Loss"
    ].idxmin()
results_df = results_df.T
adequate_df = adequate_df.T

passing_models = [
    entry for entry in preds_per_model if entry["name"] in results_df.columns
]

results_df.to_csv(f"results/comp_results{SUFFIX}_{TEST_SET}.csv")


def underline_winner(row):
    return [
        (
            "text-decoration: underline; font-weight: bold; color: gold;"
            if col == row["Winner"]
            or row["Winner"] in row
            and row[col] == row[row["Winner"]]
            else ""
        )
        for col in row.index
    ]


results_df.style.apply(underline_winner, axis=1)

# %%
# Look at key adequacy metrics for each model
results_df.T[["[1] CC fails", "[2.5] CC fails", "[5] CC fails"]].style.apply(
    underline_winner, axis=0
)

# %%
# Look at lower PICP for each model
results_df.T[["[1] Lower PICP", "[2.5] Lower PICP", "[5] Lower PICP"]].style.apply(
    underline_winner, axis=0
)


# %%
# Examine if failures are caused by wrong UC or failure of independence
def color_uc_ind_dominator(row):
    # Color the UC stat in red if it is worse than the Ind stat, and vice versa,
    # for each confidence level (90, 95, 98).
    if row.name == "Winner":
        return [""] * 6
    styles = []
    for q in [0.01, 0.025, 0.05]:
        q_str = format_cl(q)
        uc_stat = row[f"[{q_str}] UC fails"]
        ind_stat = row[f"[{q_str}] Ind fails"]
        if uc_stat > ind_stat:
            styles.append("color:red")
            styles.append("")
        elif ind_stat > uc_stat:
            styles.append("")
            styles.append("color:red")
        else:
            styles.append("")
            styles.append("")
    print(row.name, styles)
    return styles


results_df.T[
    [
        "[1] UC fails",
        "[1] Ind fails",
        "[2.5] UC fails",
        "[2.5] Ind fails",
        "[5] UC fails",
        "[5] Ind fails",
    ]
].style.apply(color_uc_ind_dominator, axis=1)
# %%
# Calculate each model's rank in each metric, taking into account whether higher or lower is better
# Initialize a dictionary to hold ranking series for each metric.
# Each metric’s ranking is computed such that rank 1 is best.
rankings = {}
model_cols = results_df.columns.drop("Winner")

for metric in results_df.index:
    # Get the values for this metric for each model.
    values = results_df.loc[metric, model_cols]

    # Skip PICP Miss because it is redundant with PICP.
    if (
        "PICP Miss" in metric
        or "indeterminate" in metric
        or "pass?" in metric
        or "violation SD" in metric
        # Temporarily remove CRPS from ranking because it is not available for all models
        or "CRPS" in metric
    ):
        continue

    if values.isnull().all():
        continue

    # Determine ranking rule for each metric.
    if "PICP" in metric:
        # Rank by closeness to the target coverage.
        # Since picp_miss = target - picp, |picp - target| is equivalent.
        cl_str = metric.split(" ")[0].strip("[]")
        picp_target = float(cl_str) / 100
        key = (values - picp_target).abs()
    elif "mean violation" in metric or "mean bias" in metric:
        # Rank by closeness to zero.
        print("metric", metric)
        print("values", values)
        key = values.abs()
        print("key", key)
        print("best", key.min())
        print("worst", key.max())
    elif any(
        s in metric
        for s in [
            "Correlation (vol. vs. errors)",
            "QL",
            "Sign Accuracy",
            "p-value",
            "passes",
            "pass pct",
        ]
    ):
        # For these, higher is better.
        key = -values
    else:
        # For all other metrics, lower is better.
        key = values

    best = key.min()
    worst = key.max()
    if best == worst:
        # If all values are the same, set all rankings to 0.
        rankings[metric] = 0
        continue

    percentile = (key - best) / (worst - best)

    # Compute the ranking
    rankings[metric] = percentile

# Create a DataFrame of rankings with models as the index and metrics as columns.
# Each cell shows the rank of that model for that metric (1 = best).
rankings_df = pd.DataFrame(rankings, index=model_cols).T
# Fill nan values with 1
rankings_df = rankings_df.fillna(1)
# Cast to simple floats (not scientific notation)
rankings_df = rankings_df.applymap(lambda x: round(x * 100) / 100)
# Add a row for the sum of ranks
rankings_df.loc["Mean quantile"] = rankings_df.mean()


def color_cells(val):
    """
    Returns a background color string where:
    0   = green,
    0.5 = white,
    1   = red
    """
    if val < 0 or val > 1:
        return ""
    if val <= 0.5:
        # interpolate from green to white
        ratio = val / 0.5
        r = int(255 * ratio)
        g = 255
        b = int(255 * ratio)
    else:
        # interpolate from white to red
        ratio = (val - 0.5) / 0.5
        r = 255
        g = int(255 * (1 - ratio))
        b = int(255 * (1 - ratio))
    return f"color: rgb({r}, {g}, {b})"


# Apply the color style to the rankings DataFrame
rankings_df.style.applymap(color_cells)

# %%
print("Lowest mean quantile:", rankings_df.loc["Mean quantile"].idxmin())

# %%
# Save rankings to CSV
rankings_df.to_csv(f"results/comp_ranking{SUFFIX}_{TEST_SET}.csv")

# %%
# Define the most important loss functions
loss_fns = [
    "nll",
    "crps",
    "FZ0_95",
    "AL_95",
    "FZ0_97.5",
    "AL_97.5",
    "FZ0_99",
    "AL_99",
    "quantile_loss_1",
    "quantile_loss_2.5",
    "quantile_loss_5",
]


# %%
# Perform Model Confidence Set analysis
# For each model, we calculate the average loss across symbols at each time index to be
# able to simply perform the MCS procedure.
# This approach is supported by Patton, A. J., & Timmermann, A. (2007) [https://doi.org/10.1093/rfs/hhm004]:
#
# |  We average the forecast errors over the cross-section of assets to obtain a measure
# |  of overall performance... This is appropriate when no single asset is of primary
# |  interest and we wish to evaluate models on their generalizability.
#
mcs_results = {0.05: None, 0.25: None, 0.5: None}
loss_dfs = {}
all_models = [e["name"] for e in preds_per_model]
for alpha in mcs_results.keys():
    all_results = pd.DataFrame(index=all_models, columns=loss_fns)
    for metric in loss_fns:
        # Construct a DataFrame with columns = each model's losses for this metric, plus symbol
        df_losses = pd.DataFrame(index=df_validation.index)
        nan_mask = np.array([False] * len(df_validation.index))
        for entry in preds_per_model:
            entry_df = pd.DataFrame(
                index=[entry["symbols"], entry["dates"]],
                columns=[entry["name"]],
            )
            loss_vals = entry.get(metric)
            if loss_vals is None:
                continue
            if not isinstance(loss_vals, np.ndarray):
                loss_vals = np.array(loss_vals)
            nans = np.isnan(loss_vals)
            if nans.any():
                print(
                    f"NaNs found in {entry['name']} for {metric} ({100*np.round(np.mean(np.isnan(loss_vals)),4)}%)",
                    end=", ",
                )
                if np.mean(nans) < MAX_NAN_THRESH:
                    print("masking")
                    nan_mask = np.logical_or(nan_mask, nans)
                else:
                    print("excluding from MCS")
                    continue
            entry_df[entry["name"]] = loss_vals
            df_losses = df_losses.join(entry_df, how="left")

        # Remove rows that have NaNs in any model's losses
        df_losses = df_losses[~nan_mask]
        loss_dfs[metric] = df_losses.copy()

        # Now pivot to have rows = time index, columns = model losses (index will be time_idx)
        # We take the mean across symbols at each time index for each model
        avg_losses_by_time = df_losses.groupby(
            df_losses.index.get_level_values("Date")
        ).mean(numeric_only=True)

        # Remove degenenerate columns (no variation in losses)
        avg_losses_by_time = avg_losses_by_time.loc[:, avg_losses_by_time.nunique() > 1]
        # print(f"Competitors for {metric}:", list(avg_losses_by_time.columns))

        # Convert the DataFrame of average losses to a numpy array for MCS (T x M)
        loss_matrix = avg_losses_by_time.to_numpy()  # shape: [T, num_models]

        # Ensure no NaNs or infs
        assert np.isfinite(loss_matrix).all(), "loss_matrix contains NaN or inf"

        # Skip if no models remain
        if loss_matrix.shape[1] == 0:
            print(f"No models remain for {metric}, skipping MCS")
            continue

        # Perform the MCS procedure at 5% significance (95% confidence)
        # Choose a block length for bootstrap (e.g., sqrt(T) or a value based on autocorrelation analysis)
        T = loss_matrix.shape[0]
        block_len = int(
            T**0.5
        )  # using sqrt(T) as a rule of thumb&#8203;:contentReference[oaicite:13]{index=13}
        mcs = MCS(
            loss_matrix,
            size=alpha,
            reps=1000,
            block_size=block_len,
            bootstrap="stationary",
            method="R",
        )
        mcs.compute()  # run the bootstrap elimination procedure

        # Get indices of models included in the MCS and their p-values
        included_indices = mcs.included  # list of column indices that remain
        pvals = mcs.pvalues.values  # array of p-values for each model
        included_models = [avg_losses_by_time.columns[i] for i in included_indices]

        # print(f"\nMCS results for {metric}:")
        # print("Included models (95% MCS):", included_models)
        for model in all_models:
            if model in list(avg_losses_by_time.columns):
                all_results.loc[model, metric] = False
            if model in included_models:
                all_results.loc[model, metric] = True
        # Optionally, print p-values for reference
        # for model_name, pval in zip(avg_losses_by_time.columns, pvals):
        #     print(f"  p-value for {model_name}: {pval}")
    mcs_results[alpha] = all_results


# %%
def color_mcs_cells(val):
    if np.isnan(val):
        return "color: gray"
    elif val:
        return "color: gold"
    else:
        return ""


# %%
for alpha in mcs_results.keys():
    cl = format_cl(1 - alpha)
    print(f"{cl}% MCS results")
    styled_df = mcs_results[alpha].style.applymap(color_mcs_cells)
    try:
        display(styled_df)
    except Exception as e:
        print(styled_df)


# %%
# To better understand MCS results: calculate p-values of pairwise outperformance but
# averaged per day as in the MCS test.
for loss_fn in loss_fns:
    loss_df = loss_dfs[loss_fn]
    per_day_p_value_df = pd.DataFrame(index=loss_df.columns, columns=loss_df.columns)
    per_day_p_value_df.index.name = "Benchmark"
    per_day_p_value_df.columns.name = "Challenger"
    flat_p_value_df = pd.DataFrame(index=loss_df.columns, columns=loss_df.columns)
    flat_p_value_df.index.name = "Benchmark"
    flat_p_value_df.columns.name = "Challenger"

    for benchmark in loss_df.columns:
        for challenger in loss_df.columns:
            if benchmark == challenger:
                continue
            benchmark_values = loss_df[benchmark]
            challenger_values = loss_df[challenger]

            # Flat test
            flat_t_stat, flat_p_value = ttest_rel(
                challenger_values, benchmark_values, alternative="less"
            )
            flat_p_value_df.loc[benchmark, challenger] = flat_p_value

            # Avg per day
            benchmark_avg_per_day = np.array(
                benchmark_values.groupby(
                    benchmark_values.index.get_level_values("Date")
                ).mean()
            )
            challenger_avg_per_day = np.array(
                challenger_values.groupby(
                    challenger_values.index.get_level_values("Date")
                ).mean()
            )
            per_day_t_stat, per_day_p_value = ttest_rel(
                challenger_avg_per_day, benchmark_avg_per_day, alternative="less"
            )
            per_day_p_value_df.loc[benchmark, challenger] = per_day_p_value

    print("\n\n===== Loss function:", loss_fn, "=====")
    total_ps = per_day_p_value_df.fillna(1).sum(axis=0).T.sort_values().reset_index()
    total_ps.columns = ["Model", "Sum of p-values"]
    total_ps = total_ps.T
    total_ps.columns.name = "Ranking"
    try:

        def color_p_values(val):
            if val < 0.05:
                return "color: gold"
            else:
                return ""  # No styling for other values

        print("PER DAY TEST:")
        styled_df = per_day_p_value_df.style.applymap(color_p_values)
        display(styled_df)
        print("FLAT TEST:")
        styled_df = flat_p_value_df.style.applymap(color_p_values)
        display(styled_df)
    except Exception as e:
        print(per_day_p_value_df)

# %%
# Alternative MCS approach: do it per symbol and count inclusions
mcs_per_stock_results = {0.05: None, 0.25: None}
all_models = [e["name"] for e in preds_per_model]
for alpha in mcs_per_stock_results.keys():
    print(f"Alpha: {alpha}")
    all_results = pd.DataFrame(index=all_models, columns=loss_fns)
    for metric in loss_fns:
        print(f"\tMetric: {metric}")
        df_losses = loss_dfs[metric].copy()
        for symbol in df_losses.index.get_level_values("Symbol").unique():
            print(f"\t\tSymbol: {symbol}", end="\r")
            symbol_losses = df_losses.xs(symbol, level="Symbol")

            # Remove degenenerate columns (no variation in losses)
            symbol_losses = symbol_losses.loc[:, symbol_losses.nunique() > 1]
            # print(f"Competitors for {metric}:", list(symbol_losses.columns))

            # Convert the DataFrame of average losses to a numpy array for MCS (T x M)
            loss_matrix = symbol_losses.to_numpy()  # shape: [T, num_models]

            # Ensure no NaNs or infs
            assert np.isfinite(loss_matrix).all(), "loss_matrix contains NaN or inf"

            # Skip if no models remain
            if loss_matrix.shape[1] == 0:
                print(f"No models remain for {metric}, skipping MCS")
                continue

            # Perform the MCS procedure at 5% significance (95% confidence)
            # Choose a block length for bootstrap (e.g., sqrt(T) or a value based on autocorrelation analysis)
            T = loss_matrix.shape[0]
            # using sqrt(T) as a rule of thumb&#8203;:contentReference[oaicite:13]{index=13}
            block_len = int(T**0.5)
            mcs = MCS(
                loss_matrix,
                size=alpha,
                reps=1000,
                block_size=block_len,
                bootstrap="stationary",
                method="max",
            )
            mcs.compute()  # run the bootstrap elimination procedure

            # Get indices of models included in the MCS and their p-values
            included_indices = mcs.included  # list of column indices that remain
            pvals = mcs.pvalues.values  # array of p-values for each model
            included_models = [symbol_losses.columns[i] for i in included_indices]
            for model in symbol_losses.columns:
                if np.isnan(all_results.loc[model, metric]):
                    all_results.loc[model, metric] = 0
                if model in included_models:
                    all_results.loc[model, metric] += 1
    mcs_per_stock_results[alpha] = all_results

# %%
# Display the results
for alpha in mcs_per_stock_results.keys():
    cl = format_cl(1 - alpha)
    print(f"{cl}% MCS results")

    def color_percentage_cells(val):
        """
        Returns a background color on a scale from red to green:
        0   = red,
        50  = white,
        100 = green
        """
        if np.isnan(val):
            return "color: gray"
        # interpolate from red to white to green
        ratio = val / 100
        r = int(255 * (1 - ratio))
        g = int(255 * ratio)
        b = 0
        return f"color: rgb({r}, {g}, {b})"

    res = mcs_per_stock_results[alpha].copy()
    non_nan = res.notna()
    res[non_nan] = 100 * res[non_nan] / 29
    res["Distributional Perf"] = np.nanmean(res[["nll", "crps"]].fillna(0), axis=1)
    res["ES Perf"] = np.nanmean(
        res[[col for col in res.columns if "FZ0_" in col or "AL_" in col]].fillna(0),
        axis=1,
    )
    res["VaR Perf"] = np.nanmean(
        res[[col for col in res.columns if "quantile_loss" in col]].fillna(0), axis=1
    )
    # Convert to nullable floats
    res = res.astype("Float64")
    # Convert to nullable integers
    res = res.astype("Int64")

    styled_df = res.sort_values(
        by=["ES Perf", "VaR Perf", "Distributional Perf"], ascending=False
    ).style.applymap(color_percentage_cells)
    try:
        display(styled_df)
    except Exception as e:
        print(styled_df)

# %%
# Generate tables for Latex document
our = [
    ("LSTM-MDN-RV", "LSTM MDN rv-final-rolling"),
    ("LSTM-MDN-IV", "LSTM MDN ivol-final-rolling"),
    ("LSTM-MDN-RV-IV", "LSTM MDN rv-and-ivol-final-rolling"),
    ("Transformer-MDN-RV", "Transformer MDN rvol expanding"),
    ("Transformer-MDN-IV", "Transformer MDN ivol expanding"),
    ("Transformer-MDN-RV-IV", "Transformer MDN rvol-ivol expanding"),
    ("Ensemble-L+T-MDN-RV", "Ensemble MDN rv"),
    ("Ensemble-L+T-MDN-IV", "Ensemble MDN iv"),
    ("Ensemble-L+T-MDN-RV-IV", "Ensemble MDN rv-iv"),
]
traditional = [
    ("GARCH", "GARCH"),
    ("GARCH-t", "GARCH Student-t"),
    ("GARCH Skewed-t", "GARCH Skewed-t"),
    ("EGARCH", "EGARCH"),
    ("RV-GARCH", "Realized GARCH norm"),
    ("RV-GARCH-t", "Realized GARCH std"),
    ("AR-GARCH", "AR(1)-GARCH(1,1)-normal"),
    ("AR-GARCH-t", "AR(1)-GARCH(1,1)-t"),
    ("HAR-QREG", "HAR-QREG"),
    ("HARQ-QREG", "HARQ-QREG"),
    ("HAR-IV-QREG", "HAR_IVOL-QREG"),
    ("DB-RV", "Benchmark DB RV"),
    ("DB-IV", "Benchmark DB IV"),
    ("DB-RV-IV", "Benchmark DB RV_IV"),
]
ml_benchmarks = [
    ("XGBoost-RV", "Benchmark XGBoost RV"),
    ("XGBoost-IV", "Benchmark XGBoost IV"),
    ("XGBoost-RV-IV", "Benchmark XGBoost RV_IV"),
    ("CatBoost-RV", "Benchmark Catboost RV"),
    ("CatBoost-IV", "Benchmark Catboost IV"),
    ("CatBoost-RV-IV", "Benchmark Catboost RV_IV"),
    ("LightGBM-RV", "Benchmark LightGBM RV"),
    ("LightGBM-IV", "Benchmark LightGBM IV"),
    ("LightGBM-RV-IV", "Benchmark LightGBM RV_IV"),
    ("LSTM-RV", "LSTM QREG rv"),
    ("LSTM-IV", "LSTM QREG iv"),
    ("LSTM-RV-IV", "LSTM QREG rv-iv"),
]

# %%
# Table 2: Distribution Accuracy and Calibration Metrics
# Columns:
# Model,	NLL,	ECE,	CRPS,	PICP 67%,	PICP 90%,	PICP 95%,	PICP 98%
print("======================================================")
print("TABLE 2: Distribution Accuracy and Calibration Metrics")
print("======================================================")
comparison_keys = [
    "NLL",
    "ECE",
    "CRPS",
    "[67] PICP Miss",
    "[67] Mean width (MPIW)",
    "[90] PICP Miss",
    "[90] Mean width (MPIW)",
    "[95] PICP Miss",
    "[95] Mean width (MPIW)",
    "[98] PICP Miss",
    "[98] Mean width (MPIW)",
]
# Values for all benchmark models, shape: [num_models, num_metrics]
benchmark_vals = np.array(
    [
        [results_df.get(model_name, {}).get(key) for key in comparison_keys]
        for _, model_name in [*traditional, *ml_benchmarks]
    ],
    dtype=float,
)
# Take the absolute value of the PICP Miss values
benchmark_vals[:, 3:] = np.abs(benchmark_vals[:, 3:])
# Set NaNs to inf so that they are not considered in the comparison
benchmark_vals[np.isnan(benchmark_vals)] = np.inf
for model_set in [our, traditional, ml_benchmarks]:
    print("")
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            print(display_name, "&", " & ".join(["-"] * 6), "\\\\")
            continue

        conf_levels = [0.67, 0.90, 0.95, 0.98]
        comp_numbers = results_df.loc[comparison_keys, model_name].values
        comp_numbers[3:] = np.abs(comp_numbers[3:])
        display_value_keys = [k.replace("PICP Miss", "PICP") for k in comparison_keys]
        display_numbers = results_df.loc[display_value_keys, model_name].values

        metrics = [
            f"{val:.4f}" if val is not None and not np.isnan(val) else "-"
            for val in display_numbers[:3]
        ] + [
            (
                (f"{val * 100:.2f}\\%" if i % 2 == 0 else f"{val:.4f}")
                if val is not None and not np.isnan(val)
                else "-"
            )
            for i, val in enumerate(display_numbers[3:])
        ]
        underline = []
        for k, row in results_df.loc[display_value_keys].iterrows():
            underline.append(row.get(model_name) == row[row["Winner"]])
        bold = (comp_numbers < benchmark_vals).all(axis=0)
        print(display_name, end=" ")
        for val, under, b in zip(metrics, underline, bold):
            if under:
                val = f"\\underline{{{val}}}"
            if b and model_set == our:
                val = f"\\textbf{{{val}}}"
            print("&", val, end=" ")
        print("\\\\")

# %%
# Table: Model Confidence Set Analysis
# Columns:
# Model,	NLL MCS 95%,	CRPS MCS 95%,	Perf score 95%,	NLL MCS 75%,	CRPS MCS 75%,	Perf score 75%
print("================================================================")
print("TABLE: Model Confidence Set Analysis for Distributional Accuracy")
print("================================================================")
for model_set in [our, traditional]:
    print("")
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        mcs_95 = mcs_per_stock_results[0.05].copy()
        mcs_75 = mcs_per_stock_results[0.25].copy()
        if (
            entry is None
            or model_name not in mcs_95.index
            and model_name not in mcs_75.index
            or all(
                np.isnan(mat.loc[model_name, metric])
                for metric in ["nll", "crps"]
                for mat in [mcs_95, mcs_75]
            )
        ):
            continue
        vals_95 = mcs_95.loc[model_name]
        vals_75 = mcs_75.loc[model_name]
        metrics = [
            vals_75["nll"],
            vals_75["crps"],
            np.nanmean([vals_75["nll"], vals_75["crps"]]),
            vals_95["nll"],
            vals_95["crps"],
            np.nanmean([vals_95["nll"], vals_95["crps"]]),
        ]
        print(
            display_name,
            # "&",
            # " & ".join(f"{int(100*val/29)}\\%" for val in metrics),
            # "\\\\",
            end=" ",
        )
        for val in metrics:
            pct = int(100 * val / 29)
            formatted = f"{pct}\\%" if not np.isnan(pct) else "-"
            if pct == 100:
                formatted = f"\\textbf{{{formatted}}}"
            print("&", formatted, end=" ")
        print("\\\\")


# %%
# Table: VaR adequacy (Christoffersen test)
# Columns:
# Model,	95% passes, 95% fails, 95% inconclusives, 95% fail rate,	97.5% passes, 97.5% fails, 97.5% inconclusives, 97.5% fail rate,	99% passes, 99% fails, 99% inconclusives, 99% fail rate
print("=========================================")
print("Table: VaR adequacy (Christoffersen test)")
print("=========================================")
for model_set in [our, traditional, ml_benchmarks]:
    print("")
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            print(display_name, "&", " & ".join(["-"] * 12), "\\\\")
            continue

        print(display_name, end=" ")

        # Calculate the number of passes, fails, and inconclusives for each confidence level
        for q in [0.05, 0.025, 0.01]:
            q_str = format_cl(q)
            chr_results_df = entry.get(f"chr_results_df_{q_str}")
            if chr_results_df is None:
                print("&", " & ".join(["-"] * 4), end="")
                continue
            passes = (chr_results_df["cc_pass"] == True).sum()
            fails = (chr_results_df["cc_pass"] == False).sum()
            inconclusives = chr_results_df["cc_pass"].isna().sum()
            total = passes + fails
            fail_rate = fails / total if total > 0 else 0
            metrics = [
                passes,
                fails,
                inconclusives,
                f"{100*fail_rate:.1f}\\%",
            ]
            print(
                "&",
                " & ".join(str(val) for val in metrics),
                end=" ",
            )

        print("\\\\")

# %%
# Table: Determine cause of failures (UC vs. Ind)
# For each model, look at the series where the model failed CC and calculate how often UC and Ind failed.
# Columns:
# Model, % UC fails 95% VaR, % Ind fails 95% VaR, % UC fails 97.5% VaR, % Ind fails 97.5% VaR, % UC fails 99% VaR, % Ind fails 99% VaR
table_qs = [0.05, 0.025, 0.01]
table_str = (
    """
\\begin{table}[H]
    \\centering
    \\caption[Causes of Christoffersen's test failures]{Causes of Christoffersen's test failures}
    \\caption*{\\makebox[\\textwidth][c]{\\parbox{0.9\\textwidth}{\\centering\\small\\textit{For stocks failing the Conditional Coverage test: percentage that also fail Unconditional Coverage or Independence tests (by VaR level and model)}}}}

    \\label{table:var_adequacy_explanation}
    \\begin{adjustbox}{width=1\\textwidth,center}
    \\begin{tabular}{
        p{0.26\\textwidth}"""
    + """
        >{\\centering\\arraybackslash}p{0.12\\textwidth}
        >{\\centering\\arraybackslash}p{0.12\\textwidth}"""
    * len(table_qs)
    + """
    }
        \\toprule
        \\textbf{Model} & \\multicolumn{2}{c}{\\textbf{95\\% VaR}} & \\multicolumn{2}{c}{\\textbf{97.5\\% VaR}} & \\multicolumn{2}{c}{\\textbf{99\\% VaR}} \\\\
        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}
        & UC & Ind.
        & UC & Ind.
        & UC & Ind.\\\\
        \\midrule
"""
)
for set_i, (set_name, model_set) in enumerate(
    (
        {
            "Probabilistic AI Models": our,
            "Traditional Benchmarks": traditional,
            "Machine Learning Benchmarks": ml_benchmarks,
        }
    ).items()
):
    if set_i != 0:
        table_str += """
        \\addlinespace
        \\hdashline[0.2pt/3pt]
        \\addlinespace
        """

    table_str += f"""
        \\multicolumn{{{1+len(table_qs)*2}}}{{l}}{{\\textbf{{{set_name}}}}} \\\\\n"""
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        table_str += "\n        "
        if entry is None:
            table_str += " ".join([display_name, "&", " & ".join(["-"] * 6), "\\\\"])
            continue

        table_str += display_name

        # Calculate the number of passes, fails, and inconclusives for each confidence level
        for q in table_qs:
            q_str = format_cl(q)
            chr_results_df = entry.get(f"chr_results_df_{q_str}")
            if chr_results_df is None:
                table_str += " ".join(["&", " & ".join(["-"] * 2)])
                continue
            masked_df = chr_results_df[chr_results_df["cc_pass"] == False]
            fails = masked_df.shape[0]
            passes = chr_results_df[chr_results_df["cc_pass"] == True].shape[0]
            fail_rate = fails / (passes + fails) if (passes + fails) > 0 else 0
            if fail_rate < 0.15:
                table_str += " & * & *"
                continue
            uc_fails = (masked_df["uc_pass"] == False).sum()
            ind_fails = (masked_df["ind_pass"] == False).sum()
            uc_fail_rate = (
                uc_fails / masked_df.shape[0] if masked_df.shape[0] > 0 else 0
            )
            ind_fail_rate = (
                ind_fails / masked_df.shape[0] if masked_df.shape[0] > 0 else 0
            )
            metrics = [100 * uc_fail_rate, 100 * ind_fail_rate]
            bold = np.array(metrics) == np.max(metrics)
            for val, b in zip(metrics, bold):
                val = f"{val:.0f}\\%"
                if b:
                    val = f"\\textbf{{{val}}}"
                table_str += " & " + val

        table_str += "\\\\"
table_str += """
        \\bottomrule
    \end{tabular}
    \end{adjustbox}
    \par\\vspace{0.3em} % Forces space between table and footnote
    {\\raggedright\\footnotesize{
        Notes: \hspace{0.3em}%
        \\begin{minipage}[t]{0.92\\textwidth}
        \\textbf{Bold} indicates the dominating source of failure for each model at each confidence level.\\\\
        \\textbf{*} the model failed the CC test for 4 or fewer stocks and is assumed to be adequate at the tested level.\\\\
        \end{minipage}
    }}
\end{table}"""
print(table_str)

# %%
# Table: VaR accuracy: Quantile Loss
# Columns:
# Model,	Quantile Loss 95%, Quantile Loss 97.5%, Quantile Loss 99%
print("===================================")
print("TABLE: VaR accuracy: Quantile Loss")
print("===================================")
table_qs = [0.05, 0.025, 0.01]
adequacy_per_q = {q: [] for q in table_qs}
for model_set in [our, traditional, ml_benchmarks]:
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            continue
        for q in table_qs:
            chr_results_df = entry.get(f"chr_results_df_{q_str}")
            if chr_results_df is None:
                continue
            passes = chr_results_df[chr_results_df["cc_pass"] == True].shape[0]
            fails = chr_results_df[chr_results_df["cc_pass"] == False].shape[0]
            fail_rate = fails / (passes + fails) if (passes + fails) > 0 else 0
            if fail_rate > 0.15 and fails > 3:
                # It is not adequate
                continue
            else:
                adequacy_per_q[q].append(entry)
# Values for all adequate benchmark models, index: [num_models, num_metrics]
adequate_scores = pd.DataFrame(
    [
        [q, entry["name"], np.nanmean(entry.get(f"quantile_loss_{format_cl(q)}"))]
        for q in table_qs
        for entry in adequacy_per_q[q]
    ],
    columns=[
        "Quantile",
        "Model Name",
        "Quantile Loss",
    ],
).pivot_table(
    index="Model Name",
    columns="Quantile",
    values=["Quantile Loss"],
)
adequate_scores.columns = adequate_scores.columns.droplevel(0)
# Sort columns in same order as table_qs
adequate_scores = adequate_scores[table_qs]
# Set NaNs to inf so that they are not considered in the comparison
adequate_scores[np.isnan(adequate_scores)] = np.inf
# Create a filtered df with only the benchmarks
benchmark_vals = pd.DataFrame(
    [
        row
        for name, row in adequate_scores.iterrows()
        if name
        in [
            model_name
            for benchmark_set in [traditional, ml_benchmarks]
            for _, model_name in benchmark_set
        ]
    ],
)
# Get best values for each metric to decide underlining
best_vals = adequate_scores.min(axis=0)
for model_set in [our, traditional, ml_benchmarks]:
    print("")
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        print(display_name, end=" ")

        if entry is None:
            print("&", " & ".join(["-"] * 3), "\\\\")
            continue

        # Calculate the interval score and mean width for each confidence level
        numbers = np.array(
            [np.nanmean(entry.get(f"quantile_loss_{format_cl(q)}")) for q in table_qs]
        )
        bold = (numbers < benchmark_vals).all(axis=0)
        underline = numbers == best_vals
        for i, (val, u, b, q) in enumerate(zip(numbers, underline, bold, table_qs)):
            val = f"{val:.6f}"
            if u:
                val = f"\\underline{{{val}}}"
            if b and model_set == our:
                val = f"\\textbf{{{val}}}"
            print("&", val, end=" ")

        print("\\\\")

# %%
# Table: MCS for quantile loss
# Columns:
# Model, 95% VaR MCS inclusion rate (pct of stocks), 97.5% VaR MCS, 99% VaR MCS, All
print("==========================================")
print("TABLE: MCS for VaR accuracy: Quantile Loss")
print("==========================================")

for model_set in [our, traditional, ml_benchmarks]:
    print("")
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            print(display_name, "&", " & ".join(["-"] * 4), "\\\\")
            continue

        print(display_name, end=" ")

        # Calculate percentage of stocks for which the model was included
        for mcs_alpha in [0.25, 0.05]:
            mcs = mcs_per_stock_results[mcs_alpha].copy()
            if model_name not in mcs.index:
                print("&", " & ".join(["-"] * 4), end="")
                continue
            row = mcs.loc[model_name]
            if row.isna().all():
                print("&", " & ".join(["-"] * 4), end="")
                continue
            values = []
            for q in [0.05, 0.025, 0.01]:
                q_str = format_cl(q)
                inclusions = row[f"quantile_loss_{q_str}"]
                val = 100 * inclusions / 29
                values.append(val)

            # Calculate the average inclusion rate across all quantiles
            avg_inclusion_rate = np.nanmean(values)
            values.append(avg_inclusion_rate)

            # Print
            for val in values:
                fmtd = f"{val:.0f}\\%"
                if int(val) == 100:
                    fmtd = f"\\textbf{{{fmtd}}}"
                print("&", fmtd, end=" ")
        print("\\\\")


# %%
# Table: Fissler-Ziegel (FZ) and Acerbi-Laeven (AL) Scoring Rules for ES Accuracy
# Columns:
# Model,	FZ0 95%,	AL 95%,	FZ0 97.5%,	AL 97.5%,	FZ0 99%,	AL 99%
print(
    "================================================================================="
)
print("TABLE: Fissler-Ziegel (FZ) and Acerbi-Laeven (AL) Scoring Rules for ES Accuracy")
print(
    "================================================================================="
)

for model_set in [our, traditional, ml_benchmarks]:
    print("")
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            print(display_name, "&", " & ".join(["-"] * 6), "\\\\")
            continue

        print(display_name, end=" ")

        # Calculate the FZ and AL scores for each confidence level
        for cl in [0.95, 0.975, 0.99]:
            cl_str = format_cl(cl)
            fz_score = entry.get(f"FZ0_{cl_str}")
            al_score = entry.get(f"AL_{cl_str}")
            if fz_score is None or al_score is None:
                print("&", " & ".join(["-"] * 2), end="")
                continue
            fz_score = np.nanmean(fz_score)
            al_score = np.nanmean(al_score)
            res_df_keys = [
                f"[{cl_str}] FZ Loss",
                f"[{cl_str}] AL Loss",
            ]
            benchmarks = pd.DataFrame(
                [
                    [benchmark_name, key, np.nanmean(scores)]
                    for key in ["FZ0", "AL"]
                    for benchmarks in [traditional, ml_benchmarks]
                    for _, benchmark_name in benchmarks
                    if (
                        benchmark_entry := next(
                            (e for e in preds_per_model if e["name"] == benchmark_name),
                            None,
                        )
                    )
                    and (scores := benchmark_entry.get(f"{key}_{cl_str}")) is not None
                ],
                columns=["Model", "Metric", "Score"],
            ).pivot_table(index="Model", columns="Metric", values="Score")
            underline = []
            for k, row in results_df.loc[res_df_keys].iterrows():
                # Underline if it is the absolute best
                underline.append(row.get(model_name) == row[row["Winner"]])
            # Bold if it is better than all benchmarks
            bolden = [
                (fz_score < benchmarks["FZ0"]).all(),
                (al_score < benchmarks["AL"]).all(),
            ]
            # Format with 4 decimal places
            metrics = [
                f"{fz_score:.4f}",
                f"{al_score:.4f}",
            ]
            for val, bold, under in zip(metrics, bolden, underline):
                if bold:
                    val = f"\\textbf{{{val}}}"
                if under:
                    val = f"\\underline{{{val}}}"
                print("& ", val, end=" ")

        print("\\\\")


# %%
# Table 8: MCS for FZ and AL
# Columns:
# Model,	# of FZ0 wins MCS 95%,	# of AL MCS wins 95%,	# of FZ0 wins MCS 75%,	# of AL wins MCS 75%
print("=========================================================")
print("TABLE 8: MCS for FZ and AL Scoring Rules for ES Accuracy")
print("=========================================================")

for model_set in [our, traditional, ml_benchmarks]:
    print("")
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            print(display_name, "&", " & ".join(["-"] * 14), "\\\\")
            continue

        print(display_name, end=" ")

        # Calculate the number of wins for each scoring rule at each confidence level
        for alpha in [0.25, 0.05]:
            mcs = mcs_per_stock_results[alpha].copy()
            if model_name not in mcs.index or mcs.loc[model_name].isna().all():
                print("&", " & ".join(["-"] * 7), end="")
                continue
            row = mcs.loc[model_name].fillna(False)
            values = []
            for cl in [0.95, 0.975, 0.99]:
                cl_str = format_cl(cl)
                values += [row[f"FZ0_{cl_str}"], row[f"AL_{cl_str}"]]
            values += [np.mean(values)]
            for v in values:
                v = int(100 * v / 29)
                s = f"{v}\\%"
                if v == 100:
                    s = f"\\textbf{{{s}}}"
                print("&", s, end=" ")

        print("\\\\")


# %%
# Table 9: Aleatoric and Epistemic variance
# Columns:
# Model,	Average aleatoric variance,	Average epistemic variance,	Average total variance
print("=========================================================")
print("TABLE 9: Aleatoric and Epistemic variance (average across symbols)")
print("=========================================================")
print("")


def sci_notation_latex(val, decimals=1):
    """Return val in the form 5.8 \\times $10^-9$, or '-' if val is NaN."""
    if np.isnan(val):
        return "-"
    # format into standard Python 1e format, e.g. '5.8e-09'
    s = f"{val:.{decimals}e}"
    mantissa, exp_str = s.split("e")
    exp = int(exp_str)  # convert e.g. '-09' -> -9
    # build the desired string, e.g. '5.8 \times $10^-9$'
    return f"${mantissa} \\times 10^{{{exp}}}$"


for display_name, model_name in our:
    entry = next(
        (entry for entry in preds_per_model if entry["name"] == model_name), None
    )
    if entry is None:
        print(display_name, "&", " & ".join(["-"] * 3), "\\\\")
        continue

    # Calculate the average aleatoric and epistemic variance
    aleatoric_var = entry.get("volatility_pred") ** 2
    epistemic_var = entry.get("epistemic_var")
    avg_aleatoric_var = np.mean(aleatoric_var) if aleatoric_var is not None else np.nan
    avg_epistemic_var = np.mean(epistemic_var) if epistemic_var is not None else np.nan
    avg_total_var = avg_aleatoric_var + avg_epistemic_var

    print(display_name, end=" ")
    if avg_aleatoric_var > 1:
        break

    for val in [avg_aleatoric_var, avg_epistemic_var, avg_total_var]:
        # Transform from variance of log returns to variance of normal returns
        log_sd = np.sqrt(val)
        sd = np.exp(log_sd) - 1
        var = sd**2
        print("&", sci_notation_latex(var), end=" ")

    print("\\\\")

# %%
# Look at how different loss functions change over time for the best performing models of each type
for variant in ["", "cumulative_"]:
    for title, loss_fn in [
        ("Negative Log-Likelihood", "nll"),
        ("Fissler-Ziegel loss (FZ) for 95% ES", "FZ0_95"),
        ("Fissler-Ziegel loss (FZ) for 97.5% ES", "FZ0_97.5"),
        ("Quantile Loss (QL) for the 2.5% quantile", "quantile_loss_2.5"),
        ("Quantile Loss (QL) for the 1% quantile", "quantile_loss_1"),
    ]:
        plt.figure(figsize=(7, 4))
        series = []
        models = [
            # "HAR_IVOL-QREG",
            "Benchmark DB IV",
            "Benchmark Catboost RV_IV",
            "GARCH",
            "GARCH Skewed-t",
            "LSTM MDN ivol-final-rolling",
            "Transformer MDN ivol expanding",
        ]
        for model_name in models:
            entry = next(
                (entry for entry in preds_per_model if entry["name"] == model_name),
                None,
            )
            s = pd.DataFrame(
                {
                    "Date": entry["dates"],
                    "Symbol": entry["symbols"],
                    model_name: entry[loss_fn],
                }
            ).set_index(["Date", "Symbol"])[model_name]
            if s.isnull().all():
                continue
            series.append(s)
        # Merge all series into a single DataFrame
        loss_df = pd.concat(series, axis=1)
        # Loop through the models and plot them
        for zorder, name in list(enumerate(models))[::-1]:
            display_name = model_name
            for model_set in [our, traditional, ml_benchmarks]:
                for d_name, model_name in model_set:
                    if name == model_name:
                        display_name = d_name
                        break
            entry = next(entry for entry in passing_models if entry["name"] == name)
            if name not in loss_df.columns:
                print(f"Model {name} not found in loss_df")
                continue
            x = loss_df.groupby("Date")[name].mean()
            if variant == "cumulative_":
                cum_sums = loss_df.groupby("Date").mean().cumsum()
                x = x.cumsum() - cum_sums.max(axis=1)
            else:
                x = x.rolling(30).mean()
            plt.plot(
                x,
                label=display_name,
                linewidth=1,
                alpha=0.8,
                zorder=zorder,
            )
        if variant == "cumulative_":
            title = f"Cumulative {title}"
        plt.title(title)
        plt.tight_layout()
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=4,
            fontsize=10,
            frameon=False,
        )
        plt.savefig(f"results/loss/{variant}{loss_fn}.pdf")

# %%
# Generate time series chart with confidence intervals for each model
example_tickers = ["AAPL", "WMT"]
# Include more conf levels here because it is interesting to see
conf_levels = CONFIDENCE_LEVELS + [0.99, 0.995]

for model_set in [our, traditional, ml_benchmarks]:
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            print(f"Model {model_name} not found in preds_per_model")
            continue
        model_name = entry["name"]
        log_df = pd.DataFrame(
            index=[entry["symbols"], entry["dates"]],
        )
        log_df.index.names = ["Symbol", "Date"]
        log_df["Mean"] = entry.get("mean_pred")
        for cl in conf_levels:
            if (lb := entry.get(f"LB_{format_cl(cl)}")) is not None:
                log_df[f"LB_{format_cl(cl)}"] = np.array(lb)
            if (ub := entry.get(f"UB_{format_cl(cl)}")) is not None:
                log_df[f"UB_{format_cl(cl)}"] = np.array(ub)
        df = np.exp(log_df) - 1
        for ticker in example_tickers:
            true_log_ret = df_validation.xs(ticker, level="Symbol")["LogReturn"]
            true_ret = np.exp(true_log_ret) - 1
            ticker_df = df.xs(ticker, level="Symbol")
            plt.figure(figsize=(7, 4))
            plt.plot(
                true_ret, label="Actual Returns", color="black", alpha=0.5, linewidth=1
            )
            plt.plot(
                ticker_df["Mean"],
                label="Predicted Mean",
                color=colors["secondary"],
                linewidth=1,
            )
            for i, cl in enumerate(conf_levels):
                lb = ticker_df.get(f"LB_{format_cl(cl)}")
                ub = ticker_df.get(f"UB_{format_cl(cl)}")
                if lb is None or ub is None or lb.isnull().any() or ub.isnull().any():
                    # Skip if any of the bounds are NaN
                    print(
                        f"Skipping {model_name} for {ticker} at {cl} due to NaN values in bounds"
                    )
                    continue
                alpha = 0.75 - i * 0.14
                plt.fill_between(
                    lb.index,
                    lb,
                    ub,
                    color=colors["primary"],
                    alpha=alpha,
                    label=f"{format_cl(cl)}% Interval",
                )
                # Mark violations
                violations = np.logical_or(
                    true_ret < lb,
                    true_ret > ub,
                )
                mark_color = (0.3 + i * 0.1,) * 3
                # Commented out now because it is too crowded
                # plt.scatter(
                #     true_log_ret[violations].index,
                #     true_log_ret[violations],
                #     marker="x",
                #     label=f"Exceedances",
                #     color=mark_color,
                #     s=100,
                #     zorder=20 - i,
                # )
            plt.ylim(-0.2, 0.2)
            # Format y ticks as pct
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            plt.xlim(ticker_df.index.min(), ticker_df.index.max())
            plt.title(f"{display_name} predictions for {ticker} on test data")
            # Place legend below plot
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=4,
                fontsize=10,
                frameon=False,
            )
            # Ensure everything fits in the figure
            plt.tight_layout()
            plt.savefig(f"results/time_series/{ticker}_{model_name}.pdf")
            if is_notebook():
                plt.show()
            plt.close()


# %%
# Plot epistemic variance for the models that have that
for model_set in [our, traditional, ml_benchmarks]:
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            continue
        epistemic_var = entry.get("epistemic_var")
        if epistemic_var is None or pd.isnull(epistemic_var).all():
            continue
        log_df = pd.DataFrame(
            index=[entry["symbols"], entry["dates"]],
        )
        log_df.index.names = ["Symbol", "Date"]
        log_df["Mean"] = entry.get("mean_pred")
        log_df["EpistemicSD"] = np.sqrt(np.array(epistemic_var))
        df = np.exp(log_df) - 1
        for ticker in example_tickers:
            true_log_ret = df_validation.xs(ticker, level="Symbol")["LogReturn"]
            true_ret = np.exp(true_log_ret) - 1
            ticker_df = df.xs(ticker, level="Symbol")
            dates = ticker_df.index
            filtered_mean = ticker_df["Mean"]
            filtered_epistemic_sd = ticker_df["EpistemicSD"]
            plt.figure(figsize=(8, 4))
            plt.plot(
                dates,
                true_ret,
                label="Actual Returns",
                color="black",
                alpha=0.5,
                linewidth=1,
            )
            plt.plot(
                dates,
                filtered_mean,
                label="Predicted Mean",
                color=colors["secondary"],
                linewidth=1,
            )
            plt.fill_between(
                dates,
                filtered_mean - filtered_epistemic_sd,
                filtered_mean + filtered_epistemic_sd,
                color=colors["primary"],
                alpha=0.8,
                label="Epistemic Uncertainty (67%)",
            )
            plt.fill_between(
                dates,
                filtered_mean - 2 * filtered_epistemic_sd,
                filtered_mean + 2 * filtered_epistemic_sd,
                color=colors["primary"],
                alpha=0.5,
                label="Epistemic Uncertainty (95%)",
            )
            plt.fill_between(
                dates,
                filtered_mean - 2.57 * filtered_epistemic_sd,
                filtered_mean + 2.57 * filtered_epistemic_sd,
                color=colors["primary"],
                alpha=0.3,
                label="Epistemic Uncertainty (99%)",
            )
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            plt.title(
                f"{display_name} return predictions with epistemic uncertainty ({ticker}, {TEST_SET} data)"
            )
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=3,
                fontsize=10,
                frameon=False,
            )
            fig, ax = plt.gcf(), plt.gca()

            # Create an inset
            axins = inset_axes(
                ax,
                width="35%",
                height="35%",
                loc="lower left",
                bbox_to_anchor=(0.1, 0.05, 1, 1),
                bbox_transform=ax.transAxes,
            )

            # Define the zoom region (adjust these index ranges as needed)
            start_idx, end_idx = 20, 70  # example range with visible variance
            zoom_dates = dates[start_idx:end_idx]
            zoom_mean = filtered_mean.iloc[start_idx:end_idx]
            zoom_sd = filtered_epistemic_sd.iloc[start_idx:end_idx]
            zoom_true_ret = true_ret.iloc[start_idx:end_idx]

            # Plot in the inset
            axins.plot(zoom_dates, zoom_true_ret, color="black", alpha=0.1, linewidth=1)
            axins.plot(zoom_dates, zoom_mean, color=colors["secondary"], linewidth=1)
            # Plot the zero line
            axins.axhline(0, color="black", linestyle="--", linewidth=0.5)

            plt.fill_between(
                zoom_dates,
                zoom_mean - zoom_sd,
                zoom_mean + zoom_sd,
                color=colors["primary"],
                alpha=0.8,
                label="Epistemic Uncertainty (67%)",
            )
            plt.fill_between(
                zoom_dates,
                zoom_mean - 2 * zoom_sd,
                zoom_mean + 2 * zoom_sd,
                color=colors["primary"],
                alpha=0.5,
                label="Epistemic Uncertainty (95%)",
            )
            plt.fill_between(
                zoom_dates,
                zoom_mean - 2.57 * zoom_sd,
                zoom_mean + 2.57 * zoom_sd,
                color=colors["primary"],
                alpha=0.3,
                label="Epistemic Uncertainty (99%)",
            )

            # Tighten y-limits for better vertical zoom
            mid = zoom_mean.mean()
            span = (3 * zoom_sd).max() * 3  # amplify focus around the epistemic band
            axins.set_ylim(mid - span, mid + span)

            # Hide x-axis ticks and labels
            axins.set_xticks([])
            axins.set_xticklabels([])

            # Optionally, hide y ticks too if minimalism is desired
            axins.set_yticks([])

            # Mark the zoom area
            mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="black")

            axins.set_frame_on(True)
            for spine in axins.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(1)

            plt.savefig(
                f"results/time_series/epistemic/{ticker}_{model_name}.pdf",
            )
            plt.show()

# %%
# Second epistemic variance plot: only predicted mean and epistemic variance
# without actual returns, and without a zoomed inset
for model_set in [our, traditional, ml_benchmarks]:
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            continue
        epistemic_var = entry.get("epistemic_var")
        if epistemic_var is None or pd.isnull(epistemic_var).all():
            continue
        log_df = pd.DataFrame(
            index=[entry["symbols"], entry["dates"]],
        )
        log_df.index.names = ["Symbol", "Date"]
        log_df["Mean"] = entry.get("mean_pred")
        log_df["EpistemicSD"] = np.sqrt(np.array(epistemic_var))
        df = np.exp(log_df) - 1
        for ticker in example_tickers:
            ticker_df = df.xs(ticker, level="Symbol")
            # Filter on first year
            ticker_df = ticker_df.loc[
                (ticker_df.index >= "2020-01-01") & (ticker_df.index < "2021-01-01")
            ]
            dates = ticker_df.index
            filtered_mean = ticker_df["Mean"]
            filtered_epistemic_sd = ticker_df["EpistemicSD"]
            plt.figure(figsize=(8, 5))
            plt.plot(
                dates,
                filtered_mean,
                label="Predicted Mean",
                color=colors["secondary"],
                linewidth=1,
            )
            plt.fill_between(
                dates,
                filtered_mean - filtered_epistemic_sd,
                filtered_mean + filtered_epistemic_sd,
                color=colors["primary"],
                alpha=0.8,
                label="Epistemic Uncertainty (67%)",
            )
            plt.fill_between(
                dates,
                filtered_mean - 2 * filtered_epistemic_sd,
                filtered_mean + 2 * filtered_epistemic_sd,
                color=colors["primary"],
                alpha=0.5,
                label="Epistemic Uncertainty (95%)",
            )
            plt.fill_between(
                dates,
                filtered_mean - 2.57 * filtered_epistemic_sd,
                filtered_mean + 2.57 * filtered_epistemic_sd,
                color=colors["primary"],
                alpha=0.3,
                label="Epistemic Uncertainty (99%)",
            )
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            plt.title(
                f"{display_name} return predictions with epistemic uncertainty ({ticker}, {TEST_SET} data)"
            )
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=3,
                fontsize=10,
                frameon=False,
            )
            plt.tight_layout()
            plt.savefig(
                f"results/time_series/epistemic_no_actual/{ticker}_{model_name}.pdf",
            )
            plt.show()
            plt.close()

# %%
# Plot VaR 97.5% and ES 97.5% per model
for model_set in [our, traditional, ml_benchmarks]:
    for display_name, model_name in model_set:
        entry = next(
            (entry for entry in preds_per_model if entry["name"] == model_name), None
        )
        if entry is None:
            continue
        log_df = pd.DataFrame(
            index=[entry["symbols"], entry["dates"]],
        )
        log_df.index.names = ["Symbol", "Date"]
        log_df["Mean"] = entry.get("mean_pred")
        var_key = "LB_95"
        log_df[var_key] = np.array(entry.get(var_key))
        es_key = "ES_97.5"
        log_df[es_key] = np.array(entry.get(es_key))
        df = np.exp(log_df) - 1
        for ticker in example_tickers:
            print(display_name, ":", ticker)
            true_log_ret = df_validation.xs(ticker, level="Symbol")["LogReturn"]
            # true_log_ret = true_log_ret.loc["2020":"2021"]
            true_ret = np.exp(true_log_ret) - 1
            ticker_df = df.xs(ticker, level="Symbol")
            # ticker_df = ticker_df.loc["2020":"2021"]
            plt.figure(figsize=(13, 2.5))
            plt.plot(
                true_ret,
                label="Actual Returns",
                color="black",
                alpha=0.5,
                linewidth=1,
            )
            if not ticker_df["Mean"].isnull().all():
                plt.plot(
                    ticker_df["Mean"],
                    label="Predicted Mean",
                    color=colors["secondary"],
                    linewidth=1,
                )
            plt.plot(
                ticker_df[var_key],
                label="VaR 97.5%",
                color=colors["primary"],
                linewidth=1,
                alpha=0.5,
            )
            plt.plot(
                ticker_df[es_key],
                label="ES 97.5%",
                color=colors["primary"],
                linewidth=1,
            )
            # Add y ticks per 5%
            plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.05))
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            plt.ylim(-0.27, 0.15)
            # plt.title(f"{display_name} predictions for {ticker} on test data")
            # Place legend in the right corner
            plt.legend(
                loc="lower right",
                fontsize=10,
                frameon=False,
            )
            # Make x-axis fill the whole width
            plt.xlim(ticker_df.index.min(), ticker_df.index.max())
            plt.tight_layout()
            plt.savefig(f"results/time_series/var_es/{ticker}_{model_name}.pdf")
            plt.show()

# %%
# Calculate p-value of outperformance in terms of PICP miss per stock
p_value_df_picp = pd.DataFrame(index=passing_model_names, columns=passing_model_names)
p_value_df_picp.index.name = "Benchmark"
p_value_df_picp.columns.name = "Challenger"
unique_symbols = set(df_validation.index.get_level_values("Symbol"))
if FILTER_ON_IMPORTANT_TICKERS:
    unique_symbols = unique_symbols.intersection(IMPORTANT_TICKERS)
unique_symbols = sorted(unique_symbols)

for benchmark in passing_model_names:
    benchmark_entry = next(
        entry for entry in passing_models if entry["name"] == benchmark
    )
    for challenger in passing_model_names:
        if benchmark == challenger:
            continue
        challenger_entry = next(
            entry for entry in passing_models if entry["name"] == challenger
        )
        benchmark_coverage_df = pd.DataFrame(
            index=unique_symbols, columns=CONFIDENCE_LEVELS
        )
        challenger_coverage_df = pd.DataFrame(
            index=unique_symbols, columns=CONFIDENCE_LEVELS
        )
        for cl in CONFIDENCE_LEVELS:
            if cl > 0.99:
                # This is too extreme to be useful on single stocks
                continue
            chr_key = f"chr_results_df_{format_cl(cl)}"
            if chr_key not in benchmark_entry or chr_key not in challenger_entry:
                continue
            benchmark_chr_results_df = benchmark_entry[chr_key]
            challenger_chr_results_df = challenger_entry[chr_key]
            benchmark_coverage_df[cl] = (
                benchmark_chr_results_df["Coverage"] - cl
            ).abs()
            challenger_coverage_df[cl] = (
                challenger_chr_results_df["Coverage"] - cl
            ).abs()

        mask = ~pd.isna(challenger_coverage_df) & ~pd.isna(benchmark_coverage_df)
        benchmark_coverage_df = benchmark_coverage_df[mask]
        challenger_coverage_df = challenger_coverage_df[mask]

        # Drop columns where all values are nan
        benchmark_coverage_df = benchmark_coverage_df.dropna(axis=1, how="all")
        challenger_coverage_df = challenger_coverage_df.dropna(axis=1, how="all")

        # Drop rows where all values are nan
        benchmark_coverage_df = benchmark_coverage_df.dropna(axis=0, how="all")
        challenger_coverage_df = challenger_coverage_df.dropna(axis=0, how="all")

        # Do a t-test on the sum of misses
        benchmark_sum_misses = benchmark_coverage_df.sum(axis=1)
        challenger_sum_misses = challenger_coverage_df.sum(axis=1)
        t_stat, p_value = ttest_rel(
            challenger_sum_misses, benchmark_sum_misses, alternative="less"
        )

        # Store the p-value in the dataframe
        p_value_df_picp.loc[benchmark, challenger] = p_value

p_value_df_picp

# %%
# Calculate winner based on p-values
p_value_df_picp.fillna(1).sum(axis=0).T.sort_values() - 1


# %%
# Calculate p-value of outperformance in terms of CRPS
p_value_df_crps = pd.DataFrame(index=passing_model_names, columns=passing_model_names)
p_value_df_crps.index.name = "Benchmark"
p_value_df_crps.columns.name = "Challenger"

for benchmark in passing_model_names:
    for challenger in passing_model_names:
        if benchmark == challenger:
            continue
        benchmark_entry = next(
            entry for entry in passing_models if entry["name"] == benchmark
        )
        challenger_entry = next(
            entry for entry in passing_models if entry["name"] == challenger
        )
        benchmark_crps = benchmark_entry.get("crps")
        challenger_crps = challenger_entry.get("crps")
        if benchmark_crps is None or challenger_crps is None:
            p_value_df_crps.loc[benchmark, challenger] = np.nan
            continue
        benchmark_crps = np.array(benchmark_crps)
        challenger_crps = np.array(challenger_crps)
        mask = ~np.isnan(benchmark_crps) & ~np.isnan(challenger_crps)
        benchmark_crps = benchmark_crps[mask]
        challenger_crps = challenger_crps[mask]

        # Paired one-sided t-test
        t_stat, p_value = ttest_rel(challenger_crps, benchmark_crps, alternative="less")

        # Store the p-value in the dataframe
        p_value_df_crps.loc[benchmark, challenger] = p_value

p_value_df_crps

# %%
# Examine correlation between probability of increase and actual increase
for entry in passing_models:
    p_up = entry.get("p_up")
    if p_up is None:
        continue
    df_copy = df_validation.copy()
    df_copy["p_up"] = p_up
    df_copy = df_copy.dropna()

    x = df_copy["p_up"]
    y = df_copy["LogReturn"]

    plt.figure(figsize=(10, 5))
    plt.scatter(df_copy["p_up"], y, alpha=0.5)
    plt.xlabel("Predicted probability of increase")
    plt.ylabel("Actual return")
    plt.title(entry["name"])

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    plt.plot(x, slope * x + intercept, color="red")
    plt.text(
        0.05,
        0.95,
        f"Correlation: {slope:.2f} ($p$ = {p_value:.4f})",
        transform=plt.gca().transAxes,
    )
    plt.show()

# %%
# Examine correlation between predicted mean and actual return
for entry in passing_models:
    mean_pred = entry.get("mean_pred")
    if mean_pred is None or np.isnan(mean_pred).all() or np.all(mean_pred == 0):
        continue
    df_copy = df_validation.copy()
    df_copy["mean_pred"] = mean_pred
    df_copy = df_copy.dropna()

    x = df_copy["mean_pred"]
    y = df_copy["LogReturn"]

    plt.figure(figsize=(10, 5))
    plt.scatter(df_copy["mean_pred"], y, alpha=0.5)
    plt.xlabel("Predicted mean return")
    plt.ylabel("Actual return")
    plt.title(entry["name"])

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    plt.plot(x, slope * x + intercept, color="red")
    plt.text(
        0.05,
        0.95,
        f"Correlation: {slope:.2f} ($p$ = {p_value:.4f})",
        transform=plt.gca().transAxes,
    )
    plt.show()

# %%
# Test trading strategy: for every model that estimates p_up, buy the 10% of stocks with the highest p_up
# and sell the 10% of stocks with the lowest p_up
for entry in passing_models:
    p_up = entry.get("p_up")
    if p_up is None:
        continue
    decisions_df = df_validation.copy()
    decisions_df["p_up"] = p_up
    decisions_df = decisions_df.dropna()
    decisions_df["p_up decile"] = decisions_df.groupby("Date")["p_up"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    decisions_df["p_up decile"] = decisions_df["p_up decile"].astype(int)
    decisions_df["Decision"] = 0
    decisions_df.loc[decisions_df["p_up decile"] == 0, "Decision"] = -1
    decisions_df.loc[decisions_df["p_up decile"] == 9, "Decision"] = 1
    decisions_df["Return"] = decisions_df["LogReturn"] * decisions_df["Decision"]
    returns_df = decisions_df.groupby("Date")["Return"].mean()
    returns_df = returns_df.to_frame()
    returns_df["Cumulative Return"] = (1 + returns_df["Return"]).cumprod() - 1
    entry["p_up_strat_returns_df"] = returns_df
    returns_df["Cumulative Return"].plot(title=entry["name"])
    plt.show()

# %%
# Test trading strategy: buy the 10% of stocks with the highest predicted mean
# and sell the 10% of stocks with the lowest
strat_results_df = pd.DataFrame(
    columns=["Mean return", "p-value (H1: mean return > 0)", "Cumulative return"]
)

for entry in passing_models:
    mean_pred = entry.get("mean_pred")
    if mean_pred is None or np.isnan(mean_pred).all() or np.all(mean_pred == 0):
        continue
    decisions_df = df_validation.copy()
    decisions_df["mean_pred"] = mean_pred
    decisions_df = decisions_df.dropna()
    decisions_df["mean_pred decile"] = decisions_df.groupby("Date")[
        "mean_pred"
    ].transform(lambda x: pd.qcut(x, 2, labels=False, duplicates="drop"))
    decisions_df["mean_pred decile"] = decisions_df["mean_pred decile"].astype(int)
    decisions_df["Decision"] = 0
    decisions_df.loc[decisions_df["mean_pred decile"] == 0, "Decision"] = -1
    decisions_df.loc[decisions_df["mean_pred decile"] == 1, "Decision"] = 1
    decisions_df["Return"] = decisions_df["LogReturn"] * decisions_df["Decision"]
    returns_df = decisions_df.groupby("Date")["Return"].mean()
    returns_df = returns_df.to_frame()
    returns_df["Cumulative Return"] = (1 + returns_df["Return"]).cumprod() - 1
    entry["mean_strat_returns_df"] = returns_df
    returns_df["Cumulative Return"].plot(title=entry["name"])
    plt.show()

    mean_return = returns_df["Return"].mean()
    cumulative_return = returns_df["Cumulative Return"].iloc[-1]

    t_stat, p_value = ttest_1samp(returns_df["Return"], 0)
    strat_results_df.loc[entry["name"]] = [mean_return, p_value, cumulative_return]

strat_results_df

# %%
# Plot all expected shortfalls
for cl in CONFIDENCE_LEVELS:
    alpha = 1 - (1 - cl) / 2
    es_estimates = []
    model_names = []
    for entry in passing_models:
        key = f"ES_{format_cl(alpha)}"
        vals = entry.get(key)
        if vals is None:
            continue
        es_estimates.append(np.array(vals).reshape(-1, 1))
        model_names.append(entry["name"])
    if not es_estimates:
        continue
    es_df = pd.DataFrame(
        np.hstack(es_estimates), index=df_validation.index, columns=model_names
    )
    for example_stock in ["AAPL", "WMT", "GS"]:
        es_df.xs(example_stock, level="Symbol").plot(
            title=f"Expected Shortfall ({cl * 100}%) for {example_stock}"
        )

# %%
# Plot ES for failing stocks for LSTM-IV
db_p_values = {
    "AAPL": 0,
    "AMGN": 0.601,
    "AMZN": 0.221,
    "AXP": 0.939,
    "BA": 0.693,
    "CAT": 0.721,
    "CRM": 0.257,
    "CSCO": 0.922,
    "CVX": 0.312,
    "DIS": 0.201,
    "GS": 0.45,
    "HD": 0.884,
    "HON": 0.156,
    "IBM": 0.976,
    "INTC": 0.926,
    "JNJ": 0.014,
    "JPM": 0.179,
    "KO": 0.658,
    "MCD": 0.014,
    "MMM": 0.765,
    "MRK": 0.957,
    "MSFT": 0.011,
    "NKE": 0,
    "PG": 0.976,
    "TRV": 0.96,
    "UNH": 0.541,
    "V": 0.908,
    "VZ": 0.695,
    "WMT": 0.937,
}
entry = next(
    entry for entry in passing_models if entry["name"] == "LSTM MDN ivol-final-rolling"
)
alpha = 0.99
es_df = pd.DataFrame(
    {
        "Date": entry["dates"],
        "Symbol": entry["symbols"],
        "ES": entry[f"ES_{format_cl(alpha)}"],
        "VaR": entry[f"LB_{format_cl(1-((1-alpha)*2))}"],
    }
).set_index(["Date", "Symbol"])
es_df["Actual"] = df_validation["LogReturn"]
es_df = np.exp(es_df) - 1
for stock in es_df.index.get_level_values("Symbol").unique():
    stock_df = es_df.xs(stock, level="Symbol")
    stock_df = stock_df.dropna()
    if stock_df.empty:
        continue
    stock_df[["ES", "VaR", "Actual"]].plot(linewidth=1)
    violations = stock_df["Actual"] < stock_df["VaR"]
    plt.scatter(
        stock_df[violations].index,
        stock_df[violations]["Actual"],
        marker="x",
        label="VaR Exceedances",
        color="red",
        s=50,
        zorder=10,
    )
    for i, d in enumerate(stock_df[violations].index):
        plt.axvline(d, color="red", linestyle="--", alpha=0.8, linewidth=1)
        plt.text(
            d,
            stock_df.loc[d, "Actual"],
            f"{i+1}",
            fontsize=16,
            ha="center",
            va="bottom",
        )
    p = db_p_values.get(stock, None)
    violations_df = stock_df[violations].copy()
    violations_df["Loss - ES"] = violations_df["Actual"] - violations_df["ES"]
    mean_diff = violations_df["Loss - ES"].mean()
    # Do a t-test on the mean difference to see if it is significantly different from 0
    t_stat, p_diff = ttest_1samp(violations_df["Loss - ES"], 0)
    # Test whether the rate of exceedances is significantly different from the expected rate
    p_rate = 1 - stats.binom.cdf(violations.sum(), len(stock_df), 1 - alpha)
    # Do a joint test of the mean difference and the rate of exceedances
    joint_stat = -2 * (np.log(p_rate) + np.log(p_diff))
    recreated_p = 1 - stats.chi2.cdf(joint_stat, df=2)

    plt.text(
        stock_df.index[-1] + pd.Timedelta(days=10),
        stock_df["ES"].min() * 1.05,
        f"Mean diff: {mean_diff:.3f}\n"
        f"$p$ for diff $\\ne$ 0: {p_diff:.3f}\n"
        f"Rate of exceedances: {violations.sum()/len(stock_df):.3f}\n"
        f"$p$ for rate $\\ne$ {1 - alpha:.2f}: {p_rate:.3f}\n"
        f"Joint $p$-value: {recreated_p:.3f}\n"
        f"Test Result: {'FAIL' if p < 0.05 else 'PASS'}\n"
        f"$p$-value: {p:.3f}",
    )
    plt.title(f"ES and VaR ({alpha * 100}%) for {stock}")
    plt.axhline(0, color="red", linestyle="--")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

    # For failing models, also plot the violations_df
    if p < 0.05:
        violations_df.plot(title=f"Violations for {stock}")
        plt.show()


# %%
# %%
# Analyze which sectors each model passes/fails for
for test_name, key_prefix in [
    ("Unconditional Coverage", "uc_"),
    ("Independence", "ind_"),
    ("Conditional Coverage", "cc_"),
]:
    for cl in CONFIDENCE_LEVELS:
        cl_str = format_cl(cl)
        sector_key = "GICS Sector"  # "GICS Sub-Industry" # "GICS Sector"
        meta_df = pd.read_csv("data/sp500_stocks_meta.csv")
        meta_df = meta_df.set_index("Symbol")
        unique_sectors = set()
        pass_pct_key = f"sector_pass_pct_{format_cl(cl)}"

        passing_models = [
            entry for entry in preds_per_model if entry["name"] in results_df.columns
        ]

        for entry in passing_models:
            chr_results_df = entry.get(f"chr_results_df_{cl_str}")
            if chr_results_df is None:
                continue
            chr_results_df = chr_results_df.join(meta_df, how="left")
            chr_results_df = chr_results_df.dropna(subset=f"{key_prefix}pass")
            passes = (
                chr_results_df[chr_results_df[f"{key_prefix}pass"].astype(bool)]
                .groupby(sector_key)[f"{key_prefix}pass"]
                .count()
                .sort_values(ascending=False)
            )
            fails = (
                chr_results_df[~chr_results_df[f"{key_prefix}pass"].astype(bool)]
                .groupby(sector_key)[f"{key_prefix}pass"]
                .count()
                .sort_values(ascending=False)
            )
            pass_pct = passes / (passes + fails)
            pass_pct = pass_pct.sort_values()
            entry[pass_pct_key] = pass_pct
            unique_sectors.update(passes.index)

        # Get the union of sectors from both datasets
        sectors = sorted(
            unique_sectors,
            key=lambda sector: sum(
                entry[pass_pct_key].get(sector, 0)
                for entry in passing_models
                if entry["name"] != "GARCH" and pass_pct_key in entry
            ),
        )
        y = np.arange(len(sectors))

        num_models = len(passing_models)
        group_height = 0.8  # total vertical space for each sector's bars
        bar_height = group_height / num_models
        # create evenly spaced offsets that place bars side by side
        offsets = np.linspace(
            -group_height / 2 + bar_height / 2,
            group_height / 2 - bar_height / 2,
            num_models,
        )

        plt.figure(figsize=(10, len(sectors) * 0.7))

        plt.title(f"Pass Rate ({test_name} at {cl_str}% interval) by Sector")

        for i, entry in enumerate(passing_models):
            if pass_pct_key not in entry:
                continue
            pass_pct = entry[pass_pct_key].reindex(sectors, fill_value=0)
            plt.barh(y + offsets[i], pass_pct, height=bar_height, label=entry["name"])

        plt.yticks(y, sectors)
        plt.gca().set_xticklabels(
            ["{:.1f}%".format(x * 100) for x in plt.gca().get_xticks()]
        )
        plt.legend()
        plt.show()

# %%
# Analyze coverage by sector
for cl in CONFIDENCE_LEVELS:
    sector_key = "GICS Sector"  # "GICS Sub-Industry" # "GICS Sector"
    meta_df = pd.read_csv("data/sp500_stocks_meta.csv")
    meta_df = meta_df.set_index("Symbol")
    unique_sectors = set()
    coverage_key = f"sector_coverage_{format_cl(cl)}"

    for entry in passing_models:
        if f"chr_results_df_{format_cl(cl)}" not in entry:
            print("Missing Christoffersen results for", entry["name"], "at", cl)
            continue
        chr_results_df = entry[f"chr_results_df_{format_cl(cl)}"]
        chr_results_df = chr_results_df.join(meta_df, how="left")
        chr_results_df = chr_results_df.dropna(subset="all_pass")
        entry[coverage_key] = (
            chr_results_df.groupby(sector_key)["Coverage"]
            .mean()
            .sort_values(ascending=False)
        )
        unique_sectors.update(entry[coverage_key].index)

    # Get the union of sectors from both datasets
    if not unique_sectors:
        print("Unique sectors is empty for", cl)
        continue  # No passes
    sectors = sorted(
        unique_sectors,
        key=lambda sector: sum(
            entry[coverage_key].get(sector, 0)
            for entry in passing_models
            if entry["name"] != "GARCH" and coverage_key in entry
        ),
    )
    y = np.arange(len(sectors))

    num_models = len(passing_models)
    group_height = 0.8  # total vertical space for each sector's bars
    bar_height = group_height / num_models
    # create evenly spaced offsets that place bars side by side
    offsets = np.linspace(
        -group_height / 2 + bar_height / 2,
        group_height / 2 - bar_height / 2,
        num_models,
    )

    plt.figure(figsize=(10, len(sectors) * 0.7))

    plt.title(f"PICP by Sector ({format_cl(cl)}%)")

    for i, entry in enumerate(passing_models):
        if coverage_key not in entry:
            continue
        pass_pct = entry[coverage_key].reindex(sectors, fill_value=0)
        plt.barh(y + offsets[i], pass_pct, height=bar_height, label=entry["name"])

    x_from = cl - 0.15
    x_to = min(cl + 0.15, 1)
    plt.xlim(x_from, x_to)
    plt.axvline(cl, color="black", linestyle="--", label="Target")
    plt.yticks(y, sectors)
    plt.gca().set_xticklabels(
        ["{:.1f}%".format(x * 100) for x in plt.gca().get_xticks()]
    )
    plt.legend()
    plt.show()

# %%
# Plot PICP for all the important tickers
include_models = {
    "GARCH",
    "LSTM MDN ffnn",
    "LSTM MDN tuned",
    "Transformer MDN tuned",
    "Transformer MDN time",
    "LSTM MDN ivol-only_ensemble",
    "LSTM MDN ivol-only-2_ensemble",
}
for cl in CONFIDENCE_LEVELS:
    existing_tickers = sorted(
        set(df_validation.index.get_level_values("Symbol")).intersection(
            IMPORTANT_TICKERS
        )
    )
    y = np.arange(len(existing_tickers))

    plt.figure(figsize=(10, len(existing_tickers) * 0.5))
    plt.title(f"PICP by ticker ({format_cl(cl)}% interval)")
    x_from = cl - 0.10
    x_to = cl + 0.10

    use_models = (
        [entry for entry in passing_models if entry["name"] in include_models]
        if include_models
        else passing_models
    )
    num_models = len(use_models)
    group_height = 0.8  # total vertical space for each sector's bars
    bar_height = group_height / num_models
    offsets = np.linspace(
        -group_height / 2 + bar_height / 2,
        group_height / 2 - bar_height / 2,
        num_models,
    )

    for i, entry in enumerate(use_models):
        chr_key = f"chr_results_df_{format_cl(cl)}"
        if chr_key not in entry:
            continue
        chr_results_df = entry[chr_key]
        chr_results_df = chr_results_df.loc[
            np.isin(chr_results_df.index.values, IMPORTANT_TICKERS)
        ]
        chr_results_df = chr_results_df.reindex(IMPORTANT_TICKERS, fill_value=0)
        plt.barh(
            y + offsets[i],
            chr_results_df["Coverage"],
            height=bar_height,
            label=entry["name"],
        )
        # Add a check mark or cross to indicate if the model passes or fails
        for idx, row in chr_results_df.iterrows():
            picp = row["Coverage"]
            if x_from < picp < x_to:
                plt.text(
                    row["Coverage"] - 0.002,
                    y[existing_tickers.index(idx)] + offsets[i],
                    "✓" if row["uc_pass"] else "✗",
                    verticalalignment="center",
                    color="white",
                )

    plt.xlim(x_from, x_to)
    plt.axvline(cl, color="black", linestyle="--", label="Target")
    plt.yticks(y, existing_tickers)
    plt.gca().set_xticklabels(
        ["{:.1f}%".format(x * 100) for x in plt.gca().get_xticks()]
    )
    plt.legend()
    plt.savefig(f"results/picp_by_ticker_{cl}.svg")
    plt.show()
