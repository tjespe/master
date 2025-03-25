# %%
from shared.adequacy import bayer_dimitriadis_test, christoffersen_test
from shared.mdn import calculate_es_for_quantile
from shared.conf_levels import format_cl
from shared.loss import (
    al_loss,
    crps_normal_univariate,
    ece_gaussian,
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
)
from data.tickers import IMPORTANT_TICKERS
from scipy.stats import ttest_rel
from scipy.stats import linregress
from scipy.stats import ttest_1samp
from arch.bootstrap import MCS


# %%
# Defined which confidence level to use for prediction intervals
CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.98]

# %%
# Select whether to only filter on important tickers
FILTER_ON_IMPORTANT_TICKERS = True

# %%
# Select wheter or not to print all of the christoffersen tests while testing
PRINT_CHRISTOFFERSEN_TESTS = False

# %%
# Exclude uninteresting models
EXCLUDE_MODELS = []

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
df

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
df
# %%
# Filter away data before 1990
df = df[df.index.get_level_values("Date") >= "1990-01-01"]
df

# %%
# Get validation part of df
dates = df.index.get_level_values("Date")
df_validation = df[(dates >= TRAIN_VALIDATION_SPLIT) & (dates < VALIDATION_TEST_SPLIT)]
df_validation

# %%
# Filter on important tickers
if FILTER_ON_IMPORTANT_TICKERS:
    df_validation = df_validation[
        df_validation.index.get_level_values("Symbol").isin(IMPORTANT_TICKERS)
    ]
    df_validation


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

        for cl in CONFIDENCE_LEVELS:
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
        (garch_t_dates >= TRAIN_VALIDATION_SPLIT)
        & (garch_t_dates < VALIDATION_TEST_SPLIT)
    ]
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

# AR GARCH Model
for version in ["AR(1)-GARCH(1,1)-normal"]:
    try: 
        ar_garch_preds = pd.read_csv(f"predictions/predictions_{version}.csv")
        ar_garch_preds["Date"] = pd.to_datetime(ar_garch_preds["Date"])
        ar_garch_preds = ar_garch_preds.set_index(["Date", "Symbol"])
        ar_garch_dates = ar_garch_preds.index.get_level_values("Date")
        ar_garch_preds = ar_garch_preds[
            (ar_garch_dates >= TRAIN_VALIDATION_SPLIT)
            & (ar_garch_dates < VALIDATION_TEST_SPLIT)
        ]
        combined_df = df_validation.join(ar_garch_preds, how="left", rsuffix="_AR_GARCH")
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
            "nll": student_t_nll(
                y_true,
                mus,
                ar_garch_vol_pred,
                nus,
            ) if model_dist == "t" else nll_loss_mean_and_vol(
                y_true,
                mus,
                ar_garch_vol_pred,
            ),
            "crps": crps,
            "ece": ece_student_t(y_true, mus, ar_garch_vol_pred, nus) if model_dist == "t" else ece_gaussian(y_true, mus, np.log(ar_garch_vol_pred**2)),
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
    # "python",
    "R",
]:
    try:
        har_preds = pd.read_csv(f"predictions/HAR_{version}.csv")
        har_preds["Date"] = pd.to_datetime(har_preds["Date"])
        har_preds = har_preds.set_index(["Date", "Symbol"])
        har_dates = har_preds.index.get_level_values("Date")
        har_preds = har_preds[
            (har_dates >= TRAIN_VALIDATION_SPLIT) & (har_dates < VALIDATION_TEST_SPLIT)
        ]
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

        for cl in CONFIDENCE_LEVELS:
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
for version in ["python", "R"]:
    try:
        harq_preds = pd.read_csv(f"predictions/HARQ_{version}.csv")
        harq_preds["Date"] = pd.to_datetime(harq_preds["Date"])
        harq_preds = harq_preds.set_index(["Date", "Symbol"])
        harq_dates = harq_preds.index.get_level_values("Date")
        harq_preds = harq_preds[
            (harq_dates >= TRAIN_VALIDATION_SPLIT)
            & (harq_dates < VALIDATION_TEST_SPLIT)
        ]
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

        for cl in CONFIDENCE_LEVELS:
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
            (realized_garch_dates >= TRAIN_VALIDATION_SPLIT)
            & (realized_garch_dates < VALIDATION_TEST_SPLIT)
        ]
        combined_df = df_validation.join(
            realized_garch_preds, how="left", rsuffix="_Realized_GARCH"
        )
        realized_garch_preds = combined_df["Forecast_Volatility"].values
        y_true = combined_df["LogReturn"].values
        mus = combined_df["Mean"].values

        entry = {
            "name": "Realized GARCH",
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

        for cl in CONFIDENCE_LEVELS:
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
            print(f"Realized GARCH has {nans} NaN predictions")
    except FileNotFoundError:
        print("Realized GARCH predictions not found")


# LSTM MDN
for version in [
    # "quick",
    # "fe",
    # "pireg",
    # "dynamic",
    # "dynamic-weighted",
    # "embedded",
    # "l2",
    # "embedded-2",
    # "embedded-small",
    # "crps",
    # "crps-2",
    # "nll-crps-mix",
    # 3,
    # "basic",
    # "basic-w-tickers",
    # "rv-data",
    "rv-data-2",
    # "rv-data-3",
    # "w-egarch",
    # "w-egarch-2",
    "ffnn",
    # "tuned",
    # "tuned-w-fred",
    "ivol-only",
    "rv-only",
    "rv-5-only",
    "rv-and-ivol",
    # Ensemble models
    "ivol-only_ensemble",
]:
    try:
        lstm_mdn_df = pd.read_csv(
            f"predictions/lstm_mdn_predictions{SUFFIX}_v{version}.csv"
        )
        lstm_mdn_df["Symbol"] = lstm_mdn_df["Symbol"].str.replace(".O", "")
        lstm_mdn_df["Date"] = pd.to_datetime(lstm_mdn_df["Date"])
        lstm_mdn_df = lstm_mdn_df.set_index(["Date", "Symbol"])
        lstm_mdn_dates = lstm_mdn_df.index.get_level_values("Date")
        lstm_mdn_df = lstm_mdn_df[
            (lstm_mdn_dates >= TRAIN_VALIDATION_SPLIT)
            & (lstm_mdn_dates < VALIDATION_TEST_SPLIT)
        ]
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
        }
        for cl in CONFIDENCE_LEVELS:
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

# Transformer MDN
for version in [
    # 3,
    "time",
    # "time-2",
    # "mini",
    # "tuned",
    # "tuned-2",
    # "time-step-attention",
    # "last-time-step",
    # "tuned-overridden",
    # "w-fred",
    "tuned-8-mixtures",
    "tuned-calibration",
]:
    try:
        transformer_df = pd.read_csv(
            f"predictions/transformer_mdn_predictions{SUFFIX}_v{version}.csv"
        )
        transformer_df["Symbol"] = transformer_df["Symbol"].str.replace(".O", "")
        transformer_df["Date"] = pd.to_datetime(transformer_df["Date"])
        transformer_df = transformer_df.set_index(["Date", "Symbol"])
        transformer_dates = transformer_df.index.get_level_values("Date")
        transformer_df = transformer_df[
            (transformer_dates >= TRAIN_VALIDATION_SPLIT)
            & (transformer_dates < VALIDATION_TEST_SPLIT)
        ]
        combined_df = df_validation.join(
            transformer_df, how="left", rsuffix="_Transformer_MDN"
        )
        ece_col = combined_df.get("ECE")
        entry = {
            "name": f"Transformer MDN {version}",
            "mean_pred": combined_df["Mean_SP"].values,
            "volatility_pred": combined_df["Vol_SP"].values,
            "nll": combined_df.get("NLL", combined_df.get("loss")).values,
            "dates": combined_df.index.get_level_values("Date"),
            "symbols": combined_df.index.get_level_values("Symbol"),
            "crps": (
                crps.values if (crps := combined_df.get("CRPS")) is not None else None
            ),
            "p_up": combined_df.get("Prob_Increase"),
            "ece": ece_col.median() if ece_col is not None else None,
        }
        for cl in CONFIDENCE_LEVELS:
            lb = combined_df.get(f"LB_{format_cl(cl)}")
            ub = combined_df.get(f"UB_{format_cl(cl)}")
            if lb is None or ub is None:
                print(
                    f"Missing {format_cl(cl)}% interval for Transformer MDN {version}"
                )
            entry[f"LB_{format_cl(cl)}"] = lb
            entry[f"UB_{format_cl(cl)}"] = ub
            alpha = 1 - (1 - cl) / 2
            entry[f"ES_{format_cl(alpha)}"] = combined_df.get(f"ES_{format_cl(alpha)}")
        preds_per_model.append(entry)
        nans = combined_df["Mean_SP"].isnull().sum()
        if nans > 0:
            print(f"Transformer MDN {version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"Transformer MDN {version} predictions not found")

# VAE based models
for version in [4]:
    for predictor in [
        "simple_regressor_on_means",
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
                (dates >= TRAIN_VALIDATION_SPLIT) & (dates < VALIDATION_TEST_SPLIT)
            ]
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
            for cl in CONFIDENCE_LEVELS:
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
            (lstm_maf_dates >= TRAIN_VALIDATION_SPLIT)
            & (lstm_maf_dates < VALIDATION_TEST_SPLIT)
        ]
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
            (vae_dates >= TRAIN_VALIDATION_SPLIT) & (vae_dates < VALIDATION_TEST_SPLIT)
        ]
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
try:
    catboost_preds = pd.read_csv(
        f"predictions/Benchmark_Catboost_Dynamic_ES{SUFFIX}.csv"
    )
    catboost_preds["Date"] = pd.to_datetime(catboost_preds["Date"])
    catboost_preds = catboost_preds.set_index(["Date", "Symbol"])
    catboost_dates = catboost_preds.index.get_level_values("Date")
    catboost_preds = catboost_preds[
        (catboost_dates >= TRAIN_VALIDATION_SPLIT)
        & (catboost_dates < VALIDATION_TEST_SPLIT)
    ]
    combined_df = df_validation.join(catboost_preds, how="left", rsuffix="_Catboost")
    # make a Mean_SP column full of 0s for now
    combined_df["Mean_SP"] = 0
    # same for Vol_SP
    combined_df["Vol_SP"] = 0
    # same for NLL
    combined_df["nll"] = np.nan

    preds_per_model.append(
        {
            "name": "Benchmark Catboost",
            "mean_pred": combined_df["Mean_SP"].values,
            "volatility_pred": combined_df["Vol_SP"].values,
            "nll": combined_df["nll"].values,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "LB_98": combined_df["Quantile_0.010"].values,
            "UB_98": combined_df["Quantile_0.990"].values,
            "LB_95": combined_df["Quantile_0.025"].values,
            "UB_95": combined_df["Quantile_0.975"].values,
            "LB_90": combined_df["Quantile_0.050"].values,
            "UB_90": combined_df["Quantile_0.950"].values,
            "ES_99": combined_df["ES_0.010"].values,
            "ES_97.5": combined_df["ES_0.025"].values,
            "ES_95": combined_df["ES_0.050"].values,
            "ES_0.05": combined_df["ES_0.950"].values,
            "ES_0.025": combined_df["ES_0.975"].values,
            "ES_0.01": combined_df["ES_0.990"].values,
        }
    )
    # nans = combined_df["Mean_SP"].isnull().sum()
    nans = 0
    if nans > 0:
        print(f"Catboost has {nans} NaN predictions")
except FileNotFoundError:
    print("Catboost predictions not found")

try:
    lightGBMpreds = pd.read_csv(
        f"predictions/Benchmark_LIGHTGBM_Dynamic_ES_stocks_RVdata.csv"
    )
    lightGBMpreds["Date"] = pd.to_datetime(lightGBMpreds["Date"])
    lightGBMpreds = lightGBMpreds.set_index(["Date", "Symbol"])
    lightGBM_dates = lightGBMpreds.index.get_level_values("Date")
    lightGBMpreds = lightGBMpreds[
        (lightGBM_dates >= TRAIN_VALIDATION_SPLIT)
        & (lightGBM_dates < VALIDATION_TEST_SPLIT)
    ]
    combined_df = df_validation.join(lightGBMpreds, how="left", rsuffix="_LIGHTGBM")
    # make a Mean_SP column full of 0s for now
    combined_df["Mean_SP"] = 0
    # same for Vol_SP
    combined_df["Vol_SP"] = 0
    # same for NLL
    combined_df["nll"] = np.nan

    preds_per_model.append(
        {
            "name": "Benchmark LightGBM RV_only",
            "mean_pred": combined_df["Mean_SP"].values,
            "volatility_pred": combined_df["Vol_SP"].values,
            "nll": combined_df["nll"].values,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "LB_98": combined_df["Quantile_0.010"].values,
            "UB_98": combined_df["Quantile_0.990"].values,
            "LB_95": combined_df["Quantile_0.025"].values,
            "UB_95": combined_df["Quantile_0.975"].values,
            "LB_90": combined_df["Quantile_0.050"].values,
            "UB_90": combined_df["Quantile_0.950"].values,
            "ES_99": combined_df["ES_0.010"].values,
            "ES_97.5": combined_df["ES_0.025"].values,
            "ES_95": combined_df["ES_0.050"].values,
            "ES_0.05": combined_df["ES_0.950"].values,
            "ES_0.025": combined_df["ES_0.975"].values,
            "ES_0.01": combined_df["ES_0.990"].values,
        }
    )
    # nans = combined_df["Mean_SP"].isnull().sum()
    nans = 0
    if nans > 0:
        print(f"LightGBM has {nans} NaN predictions")
except FileNotFoundError:
    print("LightGBM predictions not found")

try:
    xg_boost_preds = pd.read_csv(
        f"predictions/Benchmark_XGBoost_Dynamic_ES_stocks_RVdata.csv"
    )
    xg_boost_preds["Date"] = pd.to_datetime(xg_boost_preds["Date"])
    xg_boost_preds = xg_boost_preds.set_index(["Date", "Symbol"])
    xg_boost_dates = xg_boost_preds.index.get_level_values("Date")
    xg_boost_preds = xg_boost_preds[
        (xg_boost_dates >= TRAIN_VALIDATION_SPLIT)
        & (xg_boost_dates < VALIDATION_TEST_SPLIT)
    ]
    combined_df = df_validation.join(xg_boost_preds, how="left", rsuffix="_XGBoost")
    # make a Mean_SP column full of 0s for now
    combined_df["Mean_SP"] = 0
    # same for Vol_SP
    combined_df["Vol_SP"] = 0
    # same for NLL
    combined_df["nll"] = np.nan

    preds_per_model.append(
        {
            "name": "Benchmark XGBoost RV_only",
            "mean_pred": combined_df["Mean_SP"].values,
            "volatility_pred": combined_df["Vol_SP"].values,
            "nll": combined_df["nll"].values,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "LB_98": combined_df["Quantile_0.010"].values,
            "UB_98": combined_df["Quantile_0.990"].values,
            "LB_95": combined_df["Quantile_0.025"].values,
            "UB_95": combined_df["Quantile_0.975"].values,
            "LB_90": combined_df["Quantile_0.050"].values,
            "UB_90": combined_df["Quantile_0.950"].values,
            "ES_99": combined_df["ES_0.010"].values,
            "ES_97.5": combined_df["ES_0.025"].values,
            "ES_95": combined_df["ES_0.050"].values,
            "ES_0.05": combined_df["ES_0.950"].values,
            "ES_0.025": combined_df["ES_0.975"].values,
            "ES_0.01": combined_df["ES_0.990"].values,
        }
    )
    # nans = combined_df["Mean_SP"].isnull().sum()
    nans = 0
    if nans > 0:
        print(f"XGBoost has {nans} NaN predictions")
except FileNotFoundError:
    print("XGBoost predictions not found")

try:
    DB_preds = pd.read_csv(f"predictions/val_predictions_DB_all_tickers.csv")
    DB_preds["Date"] = pd.to_datetime(DB_preds["Date"])
    DB_preds = DB_preds.set_index(["Date", "Symbol"])
    DB_dates = DB_preds.index.get_level_values("Date")
    DB_preds = DB_preds[
        (DB_dates >= TRAIN_VALIDATION_SPLIT) & (DB_dates < VALIDATION_TEST_SPLIT)
    ]
    combined_df = df_validation.join(DB_preds, how="left", rsuffix="_DB")
    # make a Mean_SP column full of 0s for now
    combined_df["Mean_SP"] = 0
    # same for Vol_SP
    combined_df["Vol_SP"] = 0
    # same for NLL
    combined_df["nll"] = np.nan

    preds_per_model.append(
        {
            "name": "Benchmark DB RV_only",
            "mean_pred": combined_df["Mean_SP"].values,
            "volatility_pred": combined_df["Vol_SP"].values,
            "nll": combined_df["nll"].values,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "dates": combined_df.index.get_level_values("Date"),
            "LB_98": combined_df["DB_RV_set_0.01"].values,
            "UB_98": combined_df["DB_RV_set_0.99"].values,
            "LB_95": combined_df["DB_RV_set_0.025"].values,
            "UB_95": combined_df["DB_RV_set_0.975"].values,
            "LB_90": combined_df["DB_RV_set_0.05"].values,
            "UB_90": combined_df["DB_RV_set_0.95"].values,
            "ES_99": combined_df["DB_RV_set_ES_0.01"].values,
            "ES_97.5": combined_df["DB_RV_set_ES_0.025"].values,
            "ES_95": combined_df["DB_RV_set_ES_0.05"].values,
            "ES_0.05": combined_df["DB_RV_set_ES_0.95"].values,
            "ES_0.025": combined_df["DB_RV_set_ES_0.975"].values,
            "ES_0.01": combined_df["DB_RV_set_ES_0.99"].values,
        }
    )
    # nans = combined_df["Mean_SP"].isnull().sum()
    nans = 0
    if nans > 0:
        print(f"DB has {nans} NaN predictions")
except FileNotFoundError:
    print("DB predictions not found")
###########################################

# %%
# Remove excluded models
preds_per_model = [
    model for model in preds_per_model if model["name"] not in EXCLUDE_MODELS
]


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

        # Calculate Interval Scores
        interval_score = calculate_interval_score(
            y_test_actual, entry[f"LB_{cl_str}"], entry[f"UB_{cl_str}"], cl
        )
        entry[f"interval_score_{cl_str}"] = interval_score

        # Calculate quantile loss
        entry[f"quantile_loss_{cl_str}"] = np.nanmean(
            np.maximum(
                y_test_actual - entry[f"UB_{cl_str}"],
                entry[f"LB_{cl_str}"] - y_test_actual,
            )
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
                "Within Bounds": entry[f"within_bounds_{cl_str}"],
            }
        )
        exceedance_df = exceedance_df.dropna(subset=["Within Bounds"])

        pooled_exceedances_list = []
        reset_indices = []
        start_index = 0

        # Group by symbol in original order.
        exceedance_df.sort_values("Symbol", inplace=True)
        for symbol, group in exceedance_df.groupby("Symbol", sort=False):
            asset_exceedances = (
                ~group["Within Bounds"].astype(bool)
            ).values  # 1 if exceedance, 0 otherwise.
            pooled_exceedances_list.append(asset_exceedances)
            reset_indices.append(start_index)  # mark start of this asset's series.
            start_index += len(asset_exceedances)

        if len(pooled_exceedances_list) == 0:
            print("No valid data to run Christoffersen test.")
        else:
            pooled_exceedances = np.concatenate(pooled_exceedances_list)
            pooled_result = christoffersen_test(
                pooled_exceedances, 1 - cl, reset_indices=reset_indices
            )
            entry[f"christoffersen_test_{cl_str}"] = interpret_christoffersen_test(
                pooled_result
            )

        if PRINT_CHRISTOFFERSEN_TESTS:
            print(
                f"\n{entry['name']} Pooled Christoffersen's Test Results ({cl_str}%):"
            )
            try:
                display(entry[f"christoffersen_test_{cl_str}"])
            except NameError:
                print(entry[f"christoffersen_test_{cl_str}"])

        chr_results = []
        for symbol, within_bounds in exceedance_df.groupby("Symbol")["Within Bounds"]:
            if within_bounds.isna().any():
                print(f"Skipping {symbol} due to NaN values")
                continue
            exceedances = ~within_bounds.astype(bool)
            result = christoffersen_test(exceedances, 1 - cl)
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

        entry[f"chr_results_df_{cl_str}"] = chr_results_df
        entry[f"uc_passes_{cl_str}"] = int(chr_results_df["uc_pass"].sum())
        entry[f"uc_fails_{cl_str}"] = (chr_results_df["uc_pass"] == 0).sum()
        entry[f"uc_nans_{cl_str}"] = chr_results_df["uc_pass"].isna().sum()
        entry[f"ind_passes_{cl_str}"] = int(chr_results_df["ind_pass"].sum())
        entry[f"ind_fails_{cl_str}"] = (chr_results_df["ind_pass"] == 0).sum()
        entry[f"ind_nans_{cl_str}"] = chr_results_df["ind_pass"].isna().sum()
        entry[f"cc_passes_{cl_str}"] = int(chr_results_df["cc_pass"].sum())
        entry[f"cc_fails_{cl_str}"] = (chr_results_df["cc_pass"] == 0).sum()
        entry[f"cc_nans_{cl_str}"] = chr_results_df["cc_pass"].isna().sum()
        if PRINT_CHRISTOFFERSEN_TESTS:
            print(
                f"Unconditional Coverage:\t{entry[f'uc_passes_{cl_str}']} passes,\t{entry[f'uc_fails_{cl_str}']} fails,\t{entry[f'uc_nans_{cl_str}']} indeterminate\n"
                f"Independence:\t\t{entry[f'ind_passes_{cl_str}']} passes,\t{entry[f'ind_fails_{cl_str}']} fails,\t{entry[f'ind_nans_{cl_str}']} indeterminate\n"
                f"Conditional Coverage:\t{entry[f'cc_passes_{cl_str}']} passes,\t{entry[f'cc_fails_{cl_str}']} fails,\t{entry[f'cc_nans_{cl_str}']} indeterminate\n"
            )

        es_alpha = 1 - (1 - cl) / 2
        es_str = format_cl(es_alpha)
        es_pred = entry.get(f"ES_{es_str}")
        if es_pred is not None:
            # Code left in for now, but we have not found academic support for this approach.
            pooled_bayer_dimitriadis_result = bayer_dimitriadis_test(
                y_test_actual, entry[f"LB_{cl_str}"], es_pred, cl
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
                    "Date": y_test_actual,
                    "LB": np.array(entry[f"LB_{cl_str}"]),
                    "ES": np.array(es_pred),
                }
            )
            passes = 0
            fails = 0
            indeterminate = 0
            # We should use both 0.05 (since it's the default) and 0.10 (which is stricter) to show that GARCH underestimates risk
            pass_threshold = 0.05
            mean_violations = []
            for symbol, group in es_df.groupby("Symbol"):
                result = bayer_dimitriadis_test(
                    group["Date"].values, group["LB"].values, group["ES"].values, cl
                )
                mean_violations.append(result["mean_z"])
                p = result["p_value"]
                if not np.isfinite(p):
                    print(
                        "WARNING: nan p-value in Bayer-Dimitriadis test for ",
                        symbol,
                        "at",
                        cl_str,
                    )
                    indeterminate += 1
                elif p < pass_threshold:
                    fails += 1
                else:
                    passes += 1
            entry[f"bayer_dim_passes_{es_str}"] = passes
            entry[f"bayer_dim_fails_{es_str}"] = fails
            entry[f"bayer_dim_indeterminate_{es_str}"] = indeterminate
            entry[f"bayer_dim_mean_violation_{es_str}"] = np.mean(mean_violations)

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
    rmse = calculate_rmse(y_test_actual, entry["mean_pred"])
    entry["rmse"] = rmse

    rmse_RV = calculate_rmse(y_test_RV, entry["volatility_pred"] ** 2)
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
quantile_metric_keys = [
    "PICP",
    "PICP Miss",
    "Mean width (MPIW)",
    "Interval Score",
    "QL",
    "Lopez Loss",
    "Pooled UC p-value",
    "Pooled UC pass?",
    "Pooled Ind p-value",
    "Pooled Ind pass?",
    "Pooled CC p-value",
    "Pooled CC pass?",
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
es_metric_keys = [
    "Pooled Bayer-Dimitriadis pass?",
    "Pooled Bayer-Dimitriadis p-value",
    "Pooled Bayer-Dimitriadis mean violation",
    "Pooled Bayer-Dimitriadis violation SD",
    "Bayer-Dimitriadis passes",
    "Bayer-Dimitriadis fails",
    "Bayer-Dimitriadis indeterminate",
    "Bayer-Dimitriadis pass rate",
    "Bayer-Dimitriadis mean bias",
    "FZ Loss",
    "AL Loss",
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
    # Quantile based metrics
    **(
        {
            f"[{format_cl(cl)}] {key}": []
            for cl in CONFIDENCE_LEVELS
            for key in quantile_metric_keys
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
}

for entry in preds_per_model:
    results["Model"].append(entry["name"])
    results["NLL"].append(np.nanmean(entry["nll"]))
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
        if entry.get(f"picp_{cl_str}") is None:
            for key in quantile_metric_keys:
                results[f"[{cl_str}] {key}"].append(np.nan)
            continue
        picp_miss = entry[f"picp_{cl_str}"] - cl
        results[f"[{cl_str}] PICP"].append(entry[f"picp_{cl_str}"])
        results[f"[{cl_str}] PICP Miss"].append(picp_miss)
        results[f"[{cl_str}] Mean width (MPIW)"].append(entry[f"mpiw_{cl_str}"])
        results[f"[{cl_str}] Interval Score"].append(entry[f"interval_score_{cl_str}"])
        results[f"[{cl_str}] QL"].append(entry[f"quantile_loss_{cl_str}"])
        results[f"[{cl_str}] Lopez Loss"].append(entry[f"lopez_loss_{cl_str}"])
        pooled_results = entry[f"christoffersen_test_{cl_str}"]["p-value"]
        for i, test in enumerate(["UC", "Ind", "CC"]):
            results[f"[{cl_str}] Pooled {test} p-value"].append(pooled_results[i])
            results[f"[{cl_str}] Pooled {test} pass?"].append(
                interpret_christoffersen_stat(pooled_results[i])
            )
        uc_passes = entry[f"uc_passes_{cl_str}"]
        uc_fails = entry[f"uc_fails_{cl_str}"]
        uc_pass_pct = uc_passes / (uc_passes + uc_fails)
        results[f"[{cl_str}] UC pass pct"].append(uc_pass_pct)
        results[f"[{cl_str}] UC passes"].append(uc_passes)
        results[f"[{cl_str}] UC fails"].append(uc_fails)
        results[f"[{cl_str}] UC indeterminate"].append(entry[f"uc_nans_{cl_str}"])
        results[f"[{cl_str}] Ind passes"].append(entry[f"ind_passes_{cl_str}"])
        results[f"[{cl_str}] Ind fails"].append(entry[f"ind_fails_{cl_str}"])
        results[f"[{cl_str}] Ind indeterminate"].append(entry[f"ind_nans_{cl_str}"])
        results[f"[{cl_str}] CC passes"].append(entry[f"cc_passes_{cl_str}"])
        results[f"[{cl_str}] CC fails"].append(entry[f"cc_fails_{cl_str}"])
        results[f"[{cl_str}] CC indeterminate"].append(entry[f"cc_nans_{cl_str}"])
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
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis mean bias"].append(
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
                passes := entry[f"bayer_dim_passes_{es_str}"]
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis fails"].append(
                fails := entry[f"bayer_dim_fails_{es_str}"]
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis indeterminate"].append(
                entry[f"bayer_dim_indeterminate_{es_str}"]
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis pass rate"].append(
                passes / (passes + fails)
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis mean bias"].append(
                entry[f"bayer_dim_mean_violation_{es_str}"]
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
    # if "Benchmark" in model:
    #     continue
    # if "HAR" in model:
    #     continue
    # if "VAE MDN" in model:
    #     continue
    passes = 0
    fails = 0
    for cl in CONFIDENCE_LEVELS:
        passes += results_df.loc[model, f"[{format_cl(cl)}] CC passes"]
        fails += results_df.loc[model, f"[{format_cl(cl)}] CC fails"]
    if fails / (passes + fails) > 0.2:
        print(f"Removing {model} due to CC fails")
        results_df.drop(model, inplace=True)

# Identify winners
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
    results_df.loc["Winner", f"[{cl_str}] Mean width (MPIW)"] = results_df[
        f"[{cl_str}] Mean width (MPIW)"
    ].idxmin()
    results_df.loc["Winner", f"[{cl_str}] Interval Score"] = results_df[
        f"[{cl_str}] Interval Score"
    ].idxmin()
    results_df.loc["Winner", f"[{cl_str}] QL"] = results_df[f"[{cl_str}] QL"].idxmax()
    results_df.loc["Winner", f"[{cl_str}] Lopez Loss"] = results_df[
        f"[{cl_str}] Lopez Loss"
    ].idxmin()
    for chr_test in ["UC", "Ind", "CC"]:
        results_df.loc["Winner", f"[{cl_str}] Pooled {chr_test} p-value"] = results_df[
            f"[{cl_str}] Pooled {chr_test} p-value"
        ].idxmax()
        results_df.loc["Winner", f"[{cl_str}] {chr_test} passes"] = results_df[
            f"[{cl_str}] {chr_test} passes"
        ].idxmax()
        results_df.loc["Winner", f"[{cl_str}] {chr_test} fails"] = results_df[
            f"[{cl_str}] {chr_test} fails"
        ].idxmin()
    results_df.loc["Winner", f"[{cl_str}] UC pass pct"] = results_df[
        f"[{cl_str}] UC pass pct"
    ].idxmax()
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
    results_df.loc["Winner", f"[{es_str}] Bayer-Dimitriadis mean bias"] = (
        results_df[f"[{es_str}] Bayer-Dimitriadis mean bias"].abs().idxmin()
    )
    results_df.loc["Winner", f"[{es_str}] FZ Loss"] = results_df[
        f"[{es_str}] FZ Loss"
    ].idxmin()
    results_df.loc["Winner", f"[{es_str}] AL Loss"] = results_df[
        f"[{es_str}] AL Loss"
    ].idxmin()
results_df = results_df.T
results_df.to_csv(f"results/comp_results{SUFFIX}.csv")


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
rankings_df.to_csv(f"results/comp_ranking{SUFFIX}.csv")

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

# %%
# Define the most important loss functions
loss_fns = [
    "nll",
    "FZ0_95",
    "AL_95",
    "FZ0_97.5",
    "AL_97.5",
    "FZ0_99",
    "AL_99",
]

# %%
# Calculate p-value of outperformance in terms of NLL
for loss_fn in loss_fns:
    passing_model_names = [entry["name"] for entry in passing_models]
    p_value_df = pd.DataFrame(index=passing_model_names, columns=passing_model_names)
    p_value_df.index.name = "Benchmark"
    p_value_df.columns.name = "Challenger"

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
            benchmark_values = benchmark_entry.get(loss_fn)
            challenger_values = challenger_entry.get(loss_fn)
            if benchmark_values is None or challenger_values is None:
                p_value_df.loc[benchmark, challenger] = np.nan
                continue
            benchmark_values = np.array(benchmark_values)
            challenger_values = np.array(challenger_values)
            mask = ~np.isnan(benchmark_values) & ~np.isnan(challenger_values)
            benchmark_values = benchmark_values[mask]
            challenger_values = challenger_values[mask]

            # Paired one-sided t-test
            t_stat, p_value = ttest_rel(
                challenger_values, benchmark_values, alternative="less"
            )

            # Store the p-value in the dataframe
            p_value_df.loc[benchmark, challenger] = p_value

    print("\n\n===== Loss function:", loss_fn, "=====")
    total_ps = p_value_df.fillna(1).sum(axis=0).T.sort_values().reset_index()
    total_ps.columns = ["Model", "Sum of p-values"]
    total_ps = total_ps.T
    total_ps.columns.name = "Ranking"
    try:

        def color_cells(val):
            if val < 0.05:
                return "color: gold"
            else:
                return ""  # No styling for other values

        styled_df = p_value_df.style.applymap(color_cells)
        display(styled_df)
        print("Sum of p-values (for ranking):")
        display(total_ps)
    except Exception as e:
        print(p_value_df)

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
all_results = pd.DataFrame(index=passing_model_names, columns=loss_fns)
for metric in loss_fns:
    # Construct a DataFrame with columns = each model's losses for this metric, plus symbol
    df_losses = pd.DataFrame(index=df_validation.index)
    for entry in passing_models:
        entry_df = pd.DataFrame(
            index=[entry["symbols"], entry["dates"]],
            columns=[entry["name"]],
        )
        loss_vals = entry.get(metric)
        if loss_vals is None:
            continue
        if not isinstance(loss_vals, np.ndarray):
            loss_vals = np.array(loss_vals)
        if np.isnan(loss_vals).any():
            print(f"NaNs found in {entry['name']} for {metric}, excluding from MCS")
            continue
        entry_df[entry["name"]] = loss_vals
        df_losses = df_losses.join(entry_df, how="left")

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

    # Perform the MCS procedure at 5% significance (95% confidence)
    # Choose a block length for bootstrap (e.g., sqrt(T) or a value based on autocorrelation analysis)
    T = loss_matrix.shape[0]
    block_len = int(
        T**0.5
    )  # using sqrt(T) as a rule of thumb&#8203;:contentReference[oaicite:13]{index=13}
    mcs = MCS(
        loss_matrix,
        size=0.05,
        reps=1000,
        block_size=block_len,
        bootstrap="stationary",
        method="max",
    )
    mcs.compute()  # run the bootstrap elimination procedure

    # Get indices of models included in the MCS and their p-values
    included_indices = mcs.included  # list of column indices that remain
    pvals = mcs.pvalues.values  # array of p-values for each model
    included_models = [avg_losses_by_time.columns[i] for i in included_indices]

    # print(f"\nMCS results for {metric}:")
    # print("Included models (95% MCS):", included_models)
    for model in passing_model_names:
        if model in list(avg_losses_by_time.columns):
            all_results.loc[model, metric] = False
        if model in included_models:
            all_results.loc[model, metric] = True
    # Optionally, print p-values for reference
    # for model_name, pval in zip(avg_losses_by_time.columns, pvals):
    #     print(f"  p-value for {model_name}: {pval}")


def color_cells(val):
    if np.isnan(val):
        return "color: gray"
    elif val:
        return "color: gold"
    else:
        return ""


styled_df = all_results.style.applymap(color_cells)
styled_df

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
    if mean_pred is None or (mean_pred == 0).all():
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
    if mean_pred is None or (mean_pred == 0).all():
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
