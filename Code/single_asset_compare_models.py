# %%
# Define parameters
from shared.loss import nll_loss_mean_and_vol
from settings import (
    LOOKBACK_DAYS,
    TEST_ASSET,
    DATA_PATH,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)
from shared.crps import crps_loss_mean_and_vol

# %%
# Defined which confidence level to use for prediction intervals
CONFIDENCE_LEVEL = 0.05

# %%
# Exclude uninteresting models
EXCLUDE_MODELS = [
    "Linear Regression",
    "Yesterday's VIX",
    "Yesterday's RVOL",
    "Transformer MDN w MC Dropout",
    "MLP with MAF",
    "LSTM FFNN MDN w MC Dropout v2",
]

# %%
import numpy as np
from scipy.stats import chi2, norm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

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
    df.groupby("Symbol")["Close"].apply(lambda x: np.log(x / x.shift(1))).droplevel(0)
)

# Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
df = df[~df["LogReturn"].isnull()]

df["SquaredReturn"] = df["LogReturn"] ** 2

# Set date and symbol as index
df: pd.DataFrame = df.set_index(["Date", "Symbol"])
df

# %%
# Check if TEST_ASSET is in the data
if TEST_ASSET not in df.index.get_level_values("Symbol"):
    raise ValueError(f"TEST_ASSET '{TEST_ASSET}' not found in the data")

# %%
# Filter away data before 1990
df = df[df.index.get_level_values("Date") >= "1990-01-01"]
df

# %%
# Get test part of df
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
df_validation


# %%
# Collect models and their predictions (in order of model complexity)
preds_per_model = []

# Linear Regression Model
try:
    linreg_preds = (
        pd.read_csv(
            f"predictions/linreg_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "Linear Regression",
            "mean_pred": linreg_preds["Mean"].values,
            "volatility_pred": linreg_preds["Volatility"].values,
        }
    )
except FileNotFoundError:
    print("Linear Regression predictions not found")

# GARCH Model
try:
    garch_vol_pred = (
        pd.read_csv(
            f"predictions/garch_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]["Volatility"]
        .values
    )
    preds_per_model.append(
        {
            "name": "GARCH",
            "mean_pred": np.zeros_like(garch_vol_pred),
            "volatility_pred": garch_vol_pred,
        }
    )
except FileNotFoundError:
    print("GARCH predictions not found")

# GARCH-GJR Model
try:
    garch_gjr_vol_pred = (
        pd.read_csv(
            f"predictions/gjr_garch_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]["Volatility"]
        .values
    )
    preds_per_model.append(
        {
            "name": "GARCH-GJR",
            "mean_pred": np.zeros_like(garch_gjr_vol_pred),
            "volatility_pred": garch_gjr_vol_pred,
        }
    )
except FileNotFoundError:
    print("GARCH-GJR predictions not found")

# LSTM Model
try:
    lstm_preds = (
        pd.read_csv(
            f"predictions/lstm_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "LSTM",
            "mean_pred": lstm_preds["Mean"].values,
            "volatility_pred": lstm_preds["Volatility"].values,
        }
    )
except FileNotFoundError:
    print("LSTM predictions not found")

# LSTM with MC Dropout Model
try:
    lstm_mc_preds = (
        pd.read_csv(
            f"predictions/lstm_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "LSTM-MC",
            "mean_pred": lstm_mc_preds["Mean"].values,
            "volatility_pred": lstm_mc_preds["Volatility"].values,
            "epistemic_sd_vol": lstm_mc_preds[
                "Epistemic_Uncertainty_Volatility"
            ].values,
        }
    )
except FileNotFoundError:
    print("LSTM-MC predictions not found")

# LSTM with FFNN Model
try:
    lstm_w_ffnn_preds = (
        pd.read_csv(
            f"predictions/lstm_ffnn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "LSTM w FFNN",
            "mean_pred": lstm_w_ffnn_preds["Mean"].values,
            "volatility_pred": lstm_w_ffnn_preds["Volatility"].values,
        }
    )
except FileNotFoundError:
    print("LSTM w FFNN predictions not found")

# LSTM with FFNN and MC Dropout Model
try:
    lstm_w_ffnn_mc_preds = (
        pd.read_csv(
            f"predictions/lstm_ffnn_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "LSTM w FFNN and MC Dropout",
            "mean_pred": lstm_w_ffnn_mc_preds["Mean"].values,
            "volatility_pred": lstm_w_ffnn_mc_preds["Volatility"].values,
            "epistemic_sd_vol": lstm_w_ffnn_mc_preds[
                "Epistemic_Uncertainty_Volatility"
            ].values,
        }
    )
except FileNotFoundError:
    print("LSTM w FFNN and MC Dropout predictions not found")

# Mini LSTM
try:
    mini_lstm_preds = (
        pd.read_csv(
            f"predictions/lstm_mini_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "Mini LSTM",
            "mean_pred": mini_lstm_preds["Mean"].values,
            "volatility_pred": mini_lstm_preds["Volatility"].values,
        }
    )
except FileNotFoundError:
    print("Mini LSTM predictions not found")

# Mini LSTM w RVOL
try:
    mini_lstm_w_rvol_preds = (
        pd.read_csv(
            f"predictions/lstm_mini_w_rvol_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "Mini LSTM w RVOL",
            "mean_pred": mini_lstm_w_rvol_preds["Mean"].values,
            "volatility_pred": mini_lstm_w_rvol_preds["Volatility"].values,
        }
    )
except FileNotFoundError:
    print("Mini LSTM w RVOL predictions not found")

# Mini LSTM w RVOL and VIX
try:
    mini_lstm_w_rvol_and_vix_preds = (
        pd.read_csv(
            f"predictions/lstm_mini_w_rvol_and_vix_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "Mini LSTM w RVOL & VIX",
            "mean_pred": mini_lstm_w_rvol_and_vix_preds["Mean"].values,
            "volatility_pred": mini_lstm_w_rvol_and_vix_preds["Volatility"].values,
        }
    )
except FileNotFoundError:
    print("Mini LSTM w RVOL and VIX predictions not found")

# Monte Carlo LSTM w RVOL and VIX
try:
    mc_lstm_w_rvol_and_vix_preds = (
        pd.read_csv(
            f"predictions/lstm_mc_w_rvol_and_vix_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "Monte Carlo LSTM w RVOL & VIX",
            "mean_pred": mc_lstm_w_rvol_and_vix_preds["Mean"].values,
            "volatility_pred": mc_lstm_w_rvol_and_vix_preds["Volatility"].values,
            "epistemic_sd_vol": mc_lstm_w_rvol_and_vix_preds[
                "Epistemic_Uncertainty_Volatility"
            ].values,
        }
    )
except FileNotFoundError:
    print("Monte Carlo LSTM w RVOL and VIX predictions not found")

# Transformer MDN
for version in ["v1"]:
    try:
        transformer_mdn_preds = (
            pd.read_csv(
                f"predictions/transformer_mdn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_{version}.csv"
            )
            .set_index("Date")
            .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
        )
        preds_per_model.append(
            {
                "name": f"Transformer MDN {version}",
                "mean_pred": transformer_mdn_preds["Mean_SP"].values,
                "volatility_pred": transformer_mdn_preds["Vol_SP"].values,
                "LB_95": transformer_mdn_preds["LB_95"].values,
                "UB_95": transformer_mdn_preds["UB_95"].values,
                "nll": transformer_mdn_preds["NLL"].values.mean(),
            }
        )
    except FileNotFoundError:
        print(f"Transformer MDN {version} predictions not found")

# Transformer MDN w MC Dropout
try:
    transformer_mdn_mc_preds = (
        pd.read_csv(
            f"predictions/transformer_mdn_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
        )
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "Transformer MDN w MC Dropout",
            "mean_pred": transformer_mdn_mc_preds["Mean_MC"].values,
            "volatility_pred": transformer_mdn_mc_preds["Vol_MC"].values,
            "epistemic_sd_vol": transformer_mdn_mc_preds["Epistemic_Unc_Vol"].values,
        }
    )
except FileNotFoundError:
    print("Transformer MDN w MC Dropout predictions not found")

# MLP with MAF model
try:
    mlp_with_maf_preds = (
        pd.read_csv(f"predictions/mlp_maf_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "MLP with MAF",
            "mean_pred": mlp_with_maf_preds["Mean_SP"].values,
            "volatility_pred": mlp_with_maf_preds["Vol_SP"].values,
        }
    )
except FileNotFoundError:
    print("MLP with MAF predictions not found")

    # LSTM with MAF and LSTM feature extractor model
try:
    lstm_with_maf_non_linear_preds = (
        pd.read_csv(f"predictions/lstm_MAF_v2_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "LSTM with MAF non-linear flows",
            "mean_pred": lstm_with_maf_non_linear_preds["Mean_SP"].values,
            "volatility_pred": lstm_with_maf_non_linear_preds["Vol_SP"].values,
            "LB_95": lstm_with_maf_non_linear_preds["LB_95"].values,
            "UB_95": lstm_with_maf_non_linear_preds["UB_95"].values,
            "LB_99": lstm_with_maf_non_linear_preds["LB_99"].values,
            "UB_99": lstm_with_maf_non_linear_preds["UB_99"].values,
            "nll": lstm_with_maf_non_linear_preds["NLL"].values.mean(),
        }
    )
except FileNotFoundError:
    print("LSTM with MAF non-linear flows predictions not found")

    # LSTM with MAF and LSTM feature extractor model v3
try:
    lstm_with_maf_non_linear_preds_v3 = (
        pd.read_csv(f"predictions/lstm_MAF_v3_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "LSTM with MAF non-linear flows v3",
            "mean_pred": lstm_with_maf_non_linear_preds_v3["Mean_SP"].values,
            "volatility_pred": lstm_with_maf_non_linear_preds_v3["Vol_SP"].values,
            "LB_95": lstm_with_maf_non_linear_preds_v3["LB_95"].values,
            "UB_95": lstm_with_maf_non_linear_preds_v3["UB_95"].values,
            "LB_99": lstm_with_maf_non_linear_preds_v3["LB_99"].values,
            "UB_99": lstm_with_maf_non_linear_preds_v3["UB_99"].values,
            "nll": lstm_with_maf_non_linear_preds_v3["NLL"].values.mean(),
        }
    )
except FileNotFoundError:
    print("LSTM with MAF non-linear flows predictions not found")


# LSTM with MAF and LSTM v4
try:
    mlp_with_maf_non_linear_preds_v4 = (
        pd.read_csv(f"predictions/lstm_MAF_v4_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")
        .set_index("Date")
        .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    )
    preds_per_model.append(
        {
            "name": "LSTM with MAF non-linear flows v4",
            "mean_pred": mlp_with_maf_non_linear_preds_v4["Mean_SP"].values,
            "volatility_pred": mlp_with_maf_non_linear_preds_v4["Vol_SP"].values,
            "LB_95": mlp_with_maf_non_linear_preds_v4["LB_95"].values,
            "UB_95": mlp_with_maf_non_linear_preds_v4["UB_95"].values,
            "LB_99": mlp_with_maf_non_linear_preds_v4["LB_99"].values,
            "UB_99": mlp_with_maf_non_linear_preds_v4["UB_99"].values,
            "nll": mlp_with_maf_non_linear_preds_v4["NLL"].values.mean(),
        }
    )
except FileNotFoundError:
    print("LSTM with MAF non-linear flows v4 predictions not found")

if TEST_ASSET == "S&P":
    # VIX
    try:
        vix_preds = pd.read_csv(f"data/VIX.csv")
        vix_preds["Date"] = pd.to_datetime(vix_preds["Date"])
        vix_preds = vix_preds.set_index("Date")
        vix_preds = vix_preds.loc[df_validation.index]
        vix_vol_est = vix_preds["Close"].values / 100 / np.sqrt(252)
        preds_per_model.append(
            {
                "name": "Yesterday's VIX",
                "mean_pred": np.zeros_like(vix_vol_est),
                "volatility_pred": np.hstack(
                    [
                        # We cheat here and use the actual value for the first day just
                        # to get correct dimensions and a non-zero value that does not
                        # ruin calculations
                        vix_vol_est[0],
                        # The rest of the values, shifted by one day
                        vix_vol_est[:-1],
                    ]
                ),
                "linestyle": "--",
            }
        )
    except FileNotFoundError:
        print("VIX predictions not found")

    # RVOL
    try:
        rvol_preds = pd.read_csv(f"data/RVOL.csv")
        rvol_preds["Date"] = pd.to_datetime(rvol_preds["Date"]).dt.date
        rvol_preds = rvol_preds.set_index("Date")
        # Reindex to same index as df_validation and fill missing
        rvol_preds = rvol_preds.reindex(df_validation.index, method="ffill")
        rvol_vol_est = rvol_preds["Close"].values / 100 / np.sqrt(252)
        preds_per_model.append(
            {
                "name": "Yesterday's RVOL",
                "mean_pred": np.zeros_like(rvol_vol_est),
                # Shift by one day to make it a fair comparison
                "volatility_pred": np.hstack(
                    [
                        # We cheat here and use the actual value for the first day just
                        # to get correct dimensions and a non-zero value that does not
                        # ruin calculations
                        rvol_vol_est[0],
                        # The rest of the values, shifted by one day
                        rvol_vol_est[:-1],
                    ]
                ),
                "linestyle": "--",
            }
        )
    except FileNotFoundError:
        print("RVOL predictions not found")

# GARCH + LSTM-MC ensemble
try:
    preds_per_model.append(
        {
            "name": "GARCH + LSTM-MC ensemble",
            "mean_pred": np.mean(
                np.array(
                    [
                        lstm_mc_preds["Mean"].values,
                        np.zeros_like(lstm_mc_preds["Mean"].values),
                    ]
                ),
                axis=0,
            ),
            "volatility_pred": np.mean(
                np.array([garch_vol_pred, lstm_mc_preds["Volatility"]]), axis=0
            ),
        }
    )
except NameError:
    print("Ensemble cannot be created due to either GARCH or LSTM-MC missing")


# LSTM MDN
for version in ["v1", "v2", "v3", "vbig", "vbig2", "vbig3", "vbig-fng", "vquick"]:
    try:
        lstm_mdn_preds = (
            pd.read_csv(
                f"predictions/lstm_mdn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_{version}.csv"
            )
            .set_index("Date")
            .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
        )
        preds_per_model.append(
            {
                "name": f"LSTM MDN {version}",
                "mean_pred": lstm_mdn_preds["Mean_SP"].values,
                "volatility_pred": lstm_mdn_preds["Vol_SP"].values,
                "LB_95": lstm_mdn_preds["LB_95"].values,
                "UB_95": lstm_mdn_preds["UB_95"].values,
                "nll": lstm_mdn_preds["NLL"].values.mean(),
                "LB_99": lstm_mdn_preds["LB_99"].values,
                "UB_99": lstm_mdn_preds["UB_99"].values,
                "crps": lstm_mdn_preds["CRPS"].values.mean(),
            }
        )
    except FileNotFoundError:
        print(f"LSTM MDN {version} predictions not found")

# VAE + LSTM
for version in ["v1"]:
    try:
        vae_lstm_preds = pd.read_csv(
            f"predictions/vae_lstm_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_{version}.csv"
        )
        preds_per_model.append(
            {
                "name": "VAE + LSTM",
                "mean_pred": vae_lstm_preds["Pred_Mean"].values,
                "volatility_pred": vae_lstm_preds["Pred_Std"].values,
                "LB_95": vae_lstm_preds["LB_95"].values,
                "UB_95": vae_lstm_preds["UB_95"].values,
                "nll": vae_lstm_preds["NLL"].values.mean(),
                "LB_99": vae_lstm_preds["LB_99"].values,
                "UB_99": vae_lstm_preds["UB_99"].values,
                "crps": vae_lstm_preds["CRPS"].values.mean(),
            }
        )
    except FileNotFoundError:
        print("VAE + LSTM predictions not found")


# LSTM FFNN MDN
for version in ["v1", "v2"]:
    try:
        lstm_ffnn_mdn_preds = (
            pd.read_csv(
                f"predictions/lstm_ffnn_mdn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_{version}.csv"
            )
            .set_index("Date")
            .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
        )
        preds_per_model.append(
            {
                "name": f"LSTM FFNN MDN {version}",
                "mean_pred": lstm_ffnn_mdn_preds["Mean_SP"].values,
                "volatility_pred": lstm_ffnn_mdn_preds["Vol_SP"].values,
                "LB_95": lstm_ffnn_mdn_preds["LB_95"].values,
                "UB_95": lstm_ffnn_mdn_preds["UB_95"].values,
                "nll": lstm_ffnn_mdn_preds["NLL"].values.mean(),
                "LB_99": lstm_ffnn_mdn_preds["LB_99"].values,
                "UB_99": lstm_ffnn_mdn_preds["UB_99"].values,
                "crps": lstm_ffnn_mdn_preds["CRPS"].values.mean(),
            }
        )
    except FileNotFoundError:
        print(f"LSTM FFNN MDN {version} predictions not found")

# LSTM FFNN MDN w MC Dropout
for version in ["v2"]:
    try:
        lstm_ffnn_mdn_mc_preds = (
            pd.read_csv(
                f"predictions/lstm_ffnn_mdn_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_{version}.csv"
            )
            .set_index("Date")
            .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
        )
        preds_per_model.append(
            {
                "name": f"LSTM FFNN MDN w MC Dropout {version}",
                "mean_pred": lstm_ffnn_mdn_mc_preds["expected_returns"].values,
                "volatility_pred": lstm_ffnn_mdn_mc_preds[
                    "volatility_estimates"
                ].values,
                "epistemic_sd_vol": lstm_ffnn_mdn_mc_preds[
                    "epistemic_uncertainty_volatility_estimates"
                ].values,
                "epistemic_sd_mean": lstm_ffnn_mdn_mc_preds[
                    "epistemic_uncertainty_expected_returns"
                ].values,
            }
        )
    except FileNotFoundError:
        print("LSTM FFNN MDN w MC Dropout predictions not found")

# %%
# Remove excluded models
preds_per_model = [
    model for model in preds_per_model if model["name"] not in EXCLUDE_MODELS
]


# %%
# Functions for plotting and evaluation
def plot_volatility_prediction(model, df_validation, abs_returns_test):
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_validation.index, abs_returns_test, label="Absolute Returns", color="black"
    )
    plt.plot(
        df_validation.index,
        model["volatility_pred"],
        label=f"{model['name']} Volatility Prediction",
    )
    if "epistemic_sd_vol" in model:
        plt.fill_between(
            df_validation.index,
            model["volatility_pred"] - model["epistemic_sd_vol"],
            model["volatility_pred"] + model["epistemic_sd_vol"],
            alpha=0.5,
            label="67% confidence interval",
        )
    plt.title(f"Volatility Prediction with {model['name']}")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.savefig(
        f"results/{model['name'].replace(' ', '_').replace('/', ':').lower()}_{TEST_ASSET}.svg"
    )
    plt.show()


def plot_mean_returns_prediction(model, df_validation):
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_validation.index,
        model["mean_pred"],
        label=f"{model['name']} Mean Prediction",
    )
    plt.fill_between(
        df_validation.index,
        model["mean_pred"] - model["volatility_pred"],
        model["mean_pred"] + model["volatility_pred"],
        alpha=0.5,
        label="Volatility",
    )
    if "epistemic_sd_vol" in model:
        plt.fill_between(
            df_validation.index,
            model["mean_pred"] - model["volatility_pred"] - model["epistemic_sd_vol"],
            model["mean_pred"] + model["volatility_pred"] + model["epistemic_sd_vol"],
            alpha=0.3,
            label="Volatility w Epistemic Uncertainty",
        )
    plt.scatter(
        df_validation.index,
        df_validation["LogReturn"],
        label="True Log Returns",
        color="black",
        s=1,
    )
    plt.title(f"Mean Prediction with {model['name']}")
    plt.xlabel("Date")
    plt.ylabel("Log Returns")
    plt.legend()
    plt.show()


def calculate_prediction_intervals(model, alpha):
    cl = int((1 - alpha) * 100)
    if f"LB_{cl}" in model and f"UB_{cl}" in model:
        model["lower_bounds"] = model[f"LB_{cl}"]
        model["upper_bounds"] = model[f"UB_{cl}"]
        return
    print(f"Assuming normal distribution for {model['name']} prediction intervals")
    z_alpha = norm.ppf(1 - alpha / 2)
    model["lower_bounds"] = model["mean_pred"] - z_alpha * model["volatility_pred"]
    model["upper_bounds"] = model["mean_pred"] + z_alpha * model["volatility_pred"]


def calculate_picp(y_true, lower_bounds, upper_bounds):
    within_bounds = np.logical_and(y_true >= lower_bounds, y_true <= upper_bounds)
    picp = np.mean(within_bounds)
    return picp, within_bounds


def calculate_mpiw(lower_bounds, upper_bounds):
    mpiw = np.mean(upper_bounds - lower_bounds)
    return mpiw


def calculate_interval_score(y_true, lower_bounds, upper_bounds, alpha):
    """
    Calculates the Winkler Score for prediction intervals.
    Penalizes for both the width of the interval and the distance of the true value
    from the interval if the interval does not contain the true value.
    """
    interval_width = upper_bounds - lower_bounds
    penalties = (2 / alpha) * (
        (lower_bounds - y_true) * (y_true < lower_bounds)
        + (y_true - upper_bounds) * (y_true > upper_bounds)
    )
    interval_scores = interval_width + penalties
    mean_interval_score = np.mean(interval_scores)
    return mean_interval_score


def calculate_uncertainty_error_correlation(y_true, mean_pred, interval_width):
    prediction_errors = np.abs(y_true - mean_pred)
    correlation = np.corrcoef(interval_width, prediction_errors)[0, 1]
    return correlation


def christoffersen_test(exceedances, alpha):
    N = len(exceedances)
    x = np.sum(exceedances)
    pi_hat = x / N

    # Unconditional Coverage Test
    LR_uc = -2 * np.log(
        ((1 - alpha) ** (N - x) * alpha**x) / ((1 - pi_hat) ** (N - x) * pi_hat**x)
    )
    p_value_uc = 1 - chi2.cdf(LR_uc, df=1)

    # Handle cases where pi_hat is 0 or 1 to avoid division by zero
    if pi_hat == 0 or pi_hat == 1:
        LR_uc = np.nan
        p_value_uc = np.nan

    # Independence Test
    exceedances_shifted = np.roll(exceedances, 1)
    exceedances_shifted[0] = 0  # Set the first element to 0 (no prior observation)

    N_00 = np.sum((exceedances_shifted == 0) & (exceedances == 0))
    N_01 = np.sum((exceedances_shifted == 0) & (exceedances == 1))
    N_10 = np.sum((exceedances_shifted == 1) & (exceedances == 0))
    N_11 = np.sum((exceedances_shifted == 1) & (exceedances == 1))

    # Handle divisions by zero
    denom_0 = N_00 + N_01
    denom_1 = N_10 + N_11

    pi_0 = N_01 / denom_0 if denom_0 > 0 else 0
    pi_1 = N_11 / denom_1 if denom_1 > 0 else 0

    # Handle cases where pi_0 or pi_1 are 0 or 1
    if pi_0 == 0 or pi_0 == 1 or pi_1 == 0 or pi_1 == 1:
        LR_ind = np.nan
        p_value_ind = np.nan
        LR_cc = np.nan
        p_value_cc = np.nan
    else:
        L_null = ((1 - pi_hat) ** (N - x)) * (pi_hat**x)
        L_alt = (
            ((1 - pi_0) ** N_00) * (pi_0**N_01) * ((1 - pi_1) ** N_10) * (pi_1**N_11)
        )

        if L_null == 0 or L_alt == 0:
            LR_ind = np.nan
            p_value_ind = np.nan
        else:
            LR_ind = -2 * np.log(L_null / L_alt)
            p_value_ind = 1 - chi2.cdf(LR_ind, df=1)

        # Conditional Coverage Test
        if np.isnan(LR_uc) or np.isnan(LR_ind):
            LR_cc = np.nan
            p_value_cc = np.nan
        else:
            LR_cc = LR_uc + LR_ind
            p_value_cc = 1 - chi2.cdf(LR_cc, df=2)

    return {
        "LR_uc": LR_uc,
        "p_value_uc": p_value_uc,
        "LR_ind": LR_ind,
        "p_value_ind": p_value_ind,
        "LR_cc": LR_cc,
        "p_value_cc": p_value_cc,
    }


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


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# %%
# Evaluate models
y_test_actual = df_validation["LogReturn"].values
abs_returns_test = np.abs(y_test_actual)

for entry in preds_per_model:
    # Plot volatility predictions
    plot_volatility_prediction(entry, df_validation, abs_returns_test)

    # Plot mean return predictions
    plot_mean_returns_prediction(entry, df_validation)

    # Calculate prediction intervals
    calculate_prediction_intervals(entry, CONFIDENCE_LEVEL)

    # Calculate PICP and MPIW
    picp, within_bounds = calculate_picp(
        y_test_actual, entry["lower_bounds"], entry["upper_bounds"]
    )
    mpiw = calculate_mpiw(entry["lower_bounds"], entry["upper_bounds"])
    entry["picp"] = picp
    entry["mpiw"] = mpiw
    entry["within_bounds"] = within_bounds

    # Calculate Interval Scores
    interval_score = calculate_interval_score(
        y_test_actual, entry["lower_bounds"], entry["upper_bounds"], CONFIDENCE_LEVEL
    )
    entry["interval_score"] = interval_score

    # Calculate RMSE
    rmse = calculate_rmse(y_test_actual, entry["mean_pred"])
    entry["rmse"] = rmse
    # Calculate NLL
    if "nll" not in entry:
        entry["nll"] = nll_loss_mean_and_vol(
            y_test_actual, entry["mean_pred"], entry["volatility_pred"]
        ).mean()

    # Calculate quantile loss
    entry["quantile_loss"] = np.mean(
        np.maximum(
            y_test_actual - entry["upper_bounds"],
            entry["lower_bounds"] - y_test_actual,
        )
    )

    # Calculate Lopez loss function
    entry["lopez_loss"] = np.mean(
        np.where(
            y_test_actual < entry["lower_bounds"],
            1 + (entry["lower_bounds"] - y_test_actual) ** 2,
            0,
        )
    )

    # Uncertainty-Error Correlation
    interval_width = entry["upper_bounds"] - entry["lower_bounds"]
    correlation = calculate_uncertainty_error_correlation(
        y_test_actual, entry["mean_pred"], interval_width
    )
    entry["uncertainty_error_correlation"] = correlation

    # Calculate sign of return accuracy
    sign_accuracy = np.mean(np.sign(y_test_actual) == np.sign(entry["mean_pred"]))
    entry["sign_accuracy"] = sign_accuracy

    # Christoffersen's Test
    exceedances = ~entry["within_bounds"]
    christoffersen_result = christoffersen_test(exceedances, CONFIDENCE_LEVEL)
    entry["christoffersen_test"] = interpret_christoffersen_test(christoffersen_result)

    # Print Christoffersen's Test Results
    print(f"\n{entry['name']} Christoffersen's Test Results:")
    # Check if we are in a Jupyter notebook and can use display
    try:
        display(entry["christoffersen_test"])
    except NameError:
        print(entry["christoffersen_test"])

    # Calculate CRPS
    if "crps" not in entry:
        print("Calculating CRPS for", entry["name"])
        entry["crps"] = crps_loss_mean_and_vol(
            y_test_actual, entry["mean_pred"], entry["volatility_pred"]
        ).mean()

# %%
# Compile results into DataFrame
results = {
    "Model": [],
    "PICP": [],
    "PICP Miss": [],
    "Mean width (MPIW)": [],
    "Interval Score": [],
    "Correlation (vol. vs. errors)": [],
    # "PICP/MPIW": [],
    "NLL": [],
    "QL": [],
    "CRPS": [],
    "Lopez Loss": [],
    "RMSE": [],
    "Sign accuracy": [],
}

for entry in preds_per_model:
    picp_miss = 1 - CONFIDENCE_LEVEL - entry["picp"]
    results["Model"].append(entry["name"])
    results["PICP"].append(entry["picp"])
    results["PICP Miss"].append(picp_miss)
    results["Mean width (MPIW)"].append(entry["mpiw"])
    results["Interval Score"].append(entry["interval_score"])
    results["Correlation (vol. vs. errors)"].append(
        entry["uncertainty_error_correlation"]
    )
    # results["PICP/MPIW"].append(entry["picp"] / entry["mpiw"])
    results["NLL"].append(np.mean(entry["nll"]))
    results["QL"].append(entry["quantile_loss"])
    results["CRPS"].append(entry["crps"])
    results["Lopez Loss"].append(entry["lopez_loss"])
    results["RMSE"].append(entry["rmse"])
    results["Sign accuracy"].append(entry["sign_accuracy"])

results_df = pd.DataFrame(results)
results_df = results_df.set_index("Model")

# Identify winners
results_df.loc["Winner", "PICP"] = results_df["PICP Miss"].abs().idxmin()
results_df.loc["Winner", "PICP Miss"] = results_df["PICP Miss"].abs().idxmin()
results_df.loc["Winner", "Mean width (MPIW)"] = results_df["Mean width (MPIW)"].idxmin()
results_df.loc["Winner", "Interval Score"] = results_df["Interval Score"].idxmin()
results_df.loc["Winner", "Correlation (vol. vs. errors)"] = results_df[
    "Correlation (vol. vs. errors)"
].idxmax()
# results_df.loc["Winner", "PICP/MPIW"] = results_df["PICP/MPIW"].idxmax()
results_df.loc["Winner", "NLL"] = results_df["NLL"].idxmin()
results_df.loc["Winner", "QL"] = results_df["QL"].idxmax()
results_df.loc["Winner", "CRPS"] = results_df["CRPS"].idxmin()
results_df.loc["Winner", "Lopez Loss"] = results_df["Lopez Loss"].idxmin()
results_df.loc["Winner", "RMSE"] = results_df["RMSE"].idxmin()
results_df.loc["Winner", "Sign accuracy"] = results_df["Sign accuracy"].idxmax()
results_df = results_df.T
results_df.to_csv(f"results/comp_results_{TEST_ASSET}.csv")
results_df

# %%
# Calculate overall winner
winner_name = results_df["Winner"].mode()[0]
winner_name


# %%
# Calculate each model's rank in each metric, taking into account whether higher or lower is better
# Initialize a dictionary to hold ranking series for each metric.
# Each metric’s ranking is computed such that rank 1 is best.
rankings = {}
model_cols = results_df.columns.drop("Winner")
picp_target = (
    1 - CONFIDENCE_LEVEL
)  # e.g. if CONFIDENCE_LEVEL = 0.05, then target = 0.95

for metric in results_df.index:
    # Get the values for this metric for each model.
    values = results_df.loc[metric, model_cols]

    # Skip PICP Miss because it is redundant with PICP.
    if metric == "PICP Miss" or metric == "Sign accuracy":
        continue

    # Determine ranking rule for each metric.
    if metric == "PICP":
        # Rank by closeness to the target coverage.
        # Since picp_miss = target - picp, |picp - target| is equivalent.
        key = (values - picp_target).abs()
        ascending = True  # lower difference is better
    elif metric in ["Correlation (vol. vs. errors)", "QL", "Sign Accuracy"]:
        # For these, higher is better.
        key = values
        ascending = False
    else:
        # For all other metrics, lower is better.
        key = values
        ascending = True

    # Compute the ranking; ties get the minimum rank.
    rankings[metric] = key.rank(method="min", ascending=ascending)

# Create a DataFrame of rankings with models as the index and metrics as columns.
# Each cell shows the rank of that model for that metric (1 = best).
rankings_df = pd.DataFrame(rankings, index=model_cols).T
# Change data type to integer
rankings_df = rankings_df.astype(int)
rankings_df.loc["Rank Sum"] = rankings_df.sum()
rankings_df

# %%
# Save rankings to CSV
rankings_df.to_csv(f"results/comp_rankings_{TEST_ASSET}.csv")


# %%
# Plot volatility comparison
def plot_volatility_comparison(
    models, returns_test, abs_returns_test, lookback_days=30, steps=30
):
    from_idx = len(returns_test) - lookback_days
    to_idx = min(from_idx + steps, len(returns_test))
    plt.figure(figsize=(14, 8))
    plt.plot(
        returns_test.index[from_idx:to_idx],
        abs_returns_test[from_idx:to_idx],
        label="Absolute Returns",
        color="black",
    )
    # plt.plot(
    #     returns_test.index[from_idx:to_idx],
    #     rvol_vol_est[from_idx:to_idx],
    #     label="True RVOL",
    #     linestyle="--",
    #     color="black",
    #     linewidth=1,
    # )
    mean_vol_pred = np.mean([model["volatility_pred"] for model in models], axis=0)
    colors = ["blue", "green", "orange", "red", "purple", "brown", "#337ab7", "pink"]
    for idx, model in enumerate(models):
        # Exclude models with very high volatility predictions
        if model["volatility_pred"].mean() / mean_vol_pred.mean() > 1.2:
            continue
        linewidth = 1.7 if model["name"] == winner_name else 0.7
        plt.plot(
            returns_test.index[from_idx:to_idx],
            model["volatility_pred"][from_idx:to_idx],
            label=f"{model['name']} Volatility Prediction",
            color=colors[idx % len(colors)],
            linestyle=model.get("linestyle", "-"),
            linewidth=linewidth,
        )
        if "epistemic_sd_vol" in model:
            plt.fill_between(
                returns_test.index[from_idx:to_idx],
                model["volatility_pred"][from_idx:to_idx]
                - model["epistemic_sd_vol"][from_idx:to_idx],
                model["volatility_pred"][from_idx:to_idx]
                + model["epistemic_sd_vol"][from_idx:to_idx],
                alpha=0.3,
                label=f"{model['name']} 67% epistemic confidence interval",
                color=colors[idx % len(colors)],
            )
    plt.title("Volatility Prediction Comparison")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    for i in range(from_idx, to_idx):
        plt.axvline(
            returns_test.index[i],
            color="gray",
            alpha=0.5,
            linewidth=0.5,
        )
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2%}"))
    plt.legend()
    plt.show()


returns_test = df_validation["LogReturn"]

plot_volatility_comparison(
    preds_per_model, returns_test, abs_returns_test, lookback_days=50, steps=50
)


# %% Calculate Strategy Returns for Each Model
for entry in preds_per_model:
    # Generate trading signal: long (1) if predicted mean is positive, short (-1) if negative
    entry["signal"] = np.where(entry["mean_pred"] > 0, 1, -1)

    # Calculate daily strategy returns by multiplying the signal with the actual return
    entry["strategy_returns"] = entry["signal"] * df_validation["LogReturn"].shift(-1)

    # Calculate cumulative returns for the strategy
    entry["cumulative_strategy_returns"] = (1 + entry["strategy_returns"]).cumprod()

# %% Calculate Buy-and-Hold Cumulative Returns
df_validation["buy_hold_returns"] = (1 + df_validation["LogReturn"]).cumprod()

# %% Plot Cumulative Returns Comparison
plt.figure(figsize=(14, 8))
plt.plot(
    df_validation.index,
    df_validation["buy_hold_returns"],
    label="Buy-and-Hold",
    color="black",
    linestyle="--",
)

for entry in preds_per_model:
    plt.plot(
        df_validation.index,
        entry["cumulative_strategy_returns"],
        label=f"{entry['name']} Buy/Short Strategy",
        linewidth=1.2,
    )

plt.title(f"Cumulative Returns Comparison ({TEST_ASSET})")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

# Print final cumulative returns for all and the buy hold strategy, last entry for all is NaN so don't look at that
final_cumulative_returns = [
    entry["cumulative_strategy_returns"].iloc[-2] for entry in preds_per_model
]
final_cumulative_returns.append(df_validation["buy_hold_returns"].iloc[-2])
final_cumulative_returns = pd.Series(
    final_cumulative_returns,
    index=[entry["name"] for entry in preds_per_model] + ["Buy-and-Hold"],
)
final_cumulative_returns
# %%
