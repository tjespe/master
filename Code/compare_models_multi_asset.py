# %%
# Define parameters
from shared.mdn import calculate_es_for_quantile
from shared.conf_levels import format_cl
from shared.loss import crps_normal_univariate, nll_loss_mean_and_vol
from settings import (
    DATA_PATH,
    SUFFIX,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)
from data.tickers import IMPORTANT_TICKERS
from scipy.stats import ttest_rel
from scipy.stats import linregress
from scipy.stats import ttest_1samp


# %%
# Defined which confidence level to use for prediction intervals
CONFIDENCE_LEVELS = [0.67, 0.90, 0.95, 0.98, 0.99]

# %%
# Select whether to only filter on important tickers
FILTER_ON_IMPORTANT_TICKERS = True

# %%
# Exclude uninteresting models
EXCLUDE_MODELS = []

# %%
import numpy as np
from scipy.stats import chi2, norm
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

# Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
df = df[~df["LogReturn"].isnull()]

df["SquaredReturn"] = df["LogReturn"] ** 2

# Set date and symbol as index
df: pd.DataFrame = df.set_index(["Date", "Symbol"])
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

        entry = {
            "name": garch_type,
            "mean_pred": mus,
            "volatility_pred": garch_vol_pred,
            "symbols": df_validation.index.get_level_values("Symbol"),
            "nll": nll_loss_mean_and_vol(
                y_true,
                mus,
                garch_vol_pred,
            ),
            "crps": crps_normal_univariate(y_true, mus, garch_vol_pred),
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

# LSTM MDN
for version in [
    # "quick",
    # "fe",
    "pireg",
    "dynamic",
    # "dynamic-weighted",
    # "embedded",
    "l2",
    # "embedded-2",
    "embedded-small",
    # "crps",
    # "crps-2",
    # "nll-crps-mix",
    3,
    "rv-data",
    "rv-data-2",
    "rv-data-3",
    "w-egarch",
    "w-egarch-2",
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
        entry = {
            "name": f"LSTM MDN {version}",
            "mean_pred": combined_df["Mean_SP"].values,
            "volatility_pred": combined_df["Vol_SP"].values,
            "nll": combined_df.get("NLL", combined_df.get("loss")).values,
            "symbols": combined_df.index.get_level_values("Symbol"),
            "crps": (
                crps.values if (crps := combined_df.get("CRPS")) is not None else None
            ),
            "p_up": combined_df.get("Prob_Increase"),
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

# LSTM MAF
for version in ["v2", "v3", "v4"]:
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
                # "crps": lstm_mdn_preds["CRPS"].values.mean(),
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
                # "crps": lstm_mdn_preds["CRPS"].values.mean(),
            }
        )
        nans = combined_df["Mean_SP"].isnull().sum()
        if nans > 0:
            print(f"VAE {version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"VAE {version} predictions not found")

try:
    maf_entry = next(
        entry for entry in preds_per_model if entry["name"] == "LSTM MAF v2"
    )
    mdn_entry = next(
        entry for entry in preds_per_model if entry["name"] == "LSTM MDN rv-data-3"
    )
    preds_per_model.append(
        {
            "name": "LSTM MDN MAF Ensemble",
            "mean_pred": (maf_entry["mean_pred"] + mdn_entry["mean_pred"]) / 2,
            "volatility_pred": (
                maf_entry["volatility_pred"] + mdn_entry["volatility_pred"]
            )
            / 2,
            "LB_67": (maf_entry["LB_67"] + mdn_entry["LB_67"]) / 2,
            "UB_67": (maf_entry["UB_67"] + mdn_entry["UB_67"]) / 2,
            "LB_90": (maf_entry["LB_90"] + mdn_entry["LB_90"]) / 2,
            "UB_90": (maf_entry["UB_90"] + mdn_entry["UB_90"]) / 2,
            "LB_95": (maf_entry["LB_95"] + mdn_entry["LB_95"]) / 2,
            "UB_95": (maf_entry["UB_95"] + mdn_entry["UB_95"]) / 2,
            "LB_99": (maf_entry["LB_99"] + mdn_entry["LB_99"]) / 2,
            "UB_99": (maf_entry["UB_99"] + mdn_entry["UB_99"]) / 2,
            "nll": (maf_entry["nll"] + mdn_entry["nll"]) / 2,
            "symbols": maf_entry["symbols"],
        }
    )
except ValueError:
    print("Could not create ensemble: LSTM MAF V2 or LSTM MDN pireg not found")

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


def christoffersen_test(exceedances, alpha, reset_indices=None):
    """
    exceedances: array-like of 1 if exceedance, 0 otherwise.
    alpha: expected probability of exceedance (e.g. 0.05 for 95% VaR).
    reset_indices: positions in the pooled array that mark the start of a new asset.
    """
    # Ensure positional indexing.
    exceedances = np.asarray(exceedances)
    N = len(exceedances)
    x = np.sum(exceedances)
    pi_hat = x / N

    # Check for degenerate series.
    if x == 0 or x == N:
        return {
            "LR_uc": np.nan,
            "p_value_uc": np.nan,
            "LR_ind": np.nan,
            "p_value_ind": np.nan,
            "LR_cc": np.nan,
            "p_value_cc": np.nan,
        }

    # Unconditional Coverage Test using logs.
    try:
        logL0 = (N - x) * np.log(1 - alpha) + x * np.log(alpha)
        logL1 = (N - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
    except Exception:
        return {
            "LR_uc": np.nan,
            "p_value_uc": np.nan,
            "LR_ind": np.nan,
            "p_value_ind": np.nan,
            "LR_cc": np.nan,
            "p_value_cc": np.nan,
        }
    LR_uc = -2 * (logL0 - logL1)
    p_value_uc = 1 - chi2.cdf(LR_uc, df=1)

    # Independence Test: use only transitions within each asset.
    valid = np.ones(N, dtype=bool)
    valid[0] = False  # first observation has no predecessor.
    if reset_indices is not None:
        for idx in reset_indices:
            valid[idx] = False

    valid_indices = np.nonzero(valid)[0]
    if len(valid_indices) == 0:
        return {
            "LR_uc": LR_uc,
            "p_value_uc": p_value_uc,
            "LR_ind": np.nan,
            "p_value_ind": np.nan,
            "LR_cc": np.nan,
            "p_value_cc": np.nan,
        }

    # Form transition pairs for valid indices.
    prev = exceedances[valid_indices - 1]
    curr = exceedances[valid_indices]

    N_00 = np.sum((prev == 0) & (curr == 0))
    N_01 = np.sum((prev == 0) & (curr == 1))
    N_10 = np.sum((prev == 1) & (curr == 0))
    N_11 = np.sum((prev == 1) & (curr == 1))

    denom_0 = N_00 + N_01
    denom_1 = N_10 + N_11
    pi_0 = N_01 / denom_0 if denom_0 > 0 else 0
    pi_1 = N_11 / denom_1 if denom_1 > 0 else 0

    # If either conditional probability is degenerate, skip the independence test.
    if pi_0 in [0, 1] or pi_1 in [0, 1]:
        LR_ind = np.nan
        p_value_ind = np.nan
    else:
        try:
            logL_null = (N - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
            logL_alt = (
                N_00 * np.log(1 - pi_0)
                + N_01 * np.log(pi_0)
                + N_10 * np.log(1 - pi_1)
                + N_11 * np.log(pi_1)
            )
        except Exception:
            LR_ind = np.nan
            p_value_ind = np.nan
        else:
            LR_ind = -2 * (logL_null - logL_alt)
            p_value_ind = 1 - chi2.cdf(LR_ind, df=1)

    LR_cc = LR_uc + LR_ind if (not np.isnan(LR_uc) and not np.isnan(LR_ind)) else np.nan
    p_value_cc = 1 - chi2.cdf(LR_cc, df=2) if not np.isnan(LR_cc) else np.nan

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


def bayer_dimitriadis_test(y_true, var_pred, es_pred, alpha):
    """
    Test that the expected value of VaR exceedances matches the expected shortfall,
    i.e. that
    E[ I(y_t > VaR_t^alpha) * (y_t - ES_t^alpha) ] = 0
    where I is the indicator function.

    Returns:
      dict with:
        test_statistic : the standardized test statistic
        p_value        : two-sided p-value from the standard normal distribution
        mean_z         : average of the test variable (should be ~ 0 if well-calibrated)
        std_z          : standard deviation of the test variable
    """
    # Convert inputs to np.array for safety
    y_true = np.asarray(y_true)
    var_pred = np.asarray(var_pred)
    es_pred = np.asarray(es_pred)

    n = len(y_true)
    if any(len(arr) != n for arr in [var_pred, es_pred]):
        raise ValueError("y_true, var_pred, es_pred must have the same length.")

    # Indicator of exceedance: 1 if actual loss is bigger than the VaR threshold
    exceedances = y_true < var_pred

    # Define the test variable z_t
    # z_t = I(X_t > VaR_t^alpha) * [ (X_t - ES_t^alpha) / alpha ]
    z = np.where(exceedances, (y_true - es_pred) / alpha, 0.0)

    mean_z = np.nanmean(z)
    std_z = np.nanstd(z, ddof=1)

    # If the test variable is degenerate, return NaNs
    if std_z == 0:
        print("SD[Z] was 0 for alpha", alpha, "returning NaNs")
        print("mean_z", mean_z)
        print("std_z", std_z)
        print("exceedances", exceedances)
        print("y_true", y_true)
        print("var_pred", var_pred)
        print("es_pred", es_pred)
        return {
            "test_statistic": np.nan,
            "p_value": np.nan,
            "mean_z": mean_z,
            "std_z": std_z,
        }

    # Compute test statistic: T = sqrt(n) * (mean of z) / stdev(z)
    test_statistic = np.sqrt(n) * mean_z / std_z
    # Two-sided p-value
    p_value = 2.0 * (1.0 - norm.cdf(abs(test_statistic)))

    return {
        "test_statistic": test_statistic,
        "p_value": p_value,
        "mean_z": mean_z,
        "std_z": std_z,
    }


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
abs_returns_test = np.abs(y_test_actual)

for entry in preds_per_model:
    # Calculate prediction intervals
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

        print(f"\n{entry['name']} Pooled Christoffersen's Test Results ({cl_str}%):")
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
        print(f"\n{entry['name']} Average Christoffersen's Test Results ({cl_str}%):")

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
        print(
            f"Unconditional Coverage:\t{entry[f'uc_passes_{cl_str}']} passes,\t{entry[f'uc_fails_{cl_str}']} fails,\t{entry[f'uc_nans_{cl_str}']} indeterminate\n"
            f"Independence:\t\t{entry[f'ind_passes_{cl_str}']} passes,\t{entry[f'ind_fails_{cl_str}']} fails,\t{entry[f'ind_nans_{cl_str}']} indeterminate\n"
            f"Conditional Coverage:\t{entry[f'cc_passes_{cl_str}']} passes,\t{entry[f'cc_fails_{cl_str}']} fails,\t{entry[f'cc_nans_{cl_str}']} indeterminate\n"
        )

        es_alpha = 1 - (1 - cl) / 2
        es_str = format_cl(es_alpha)
        es_pred = entry.get(f"ES_{es_str}")
        if es_pred is not None:
            bayer_dimitriadis_result = bayer_dimitriadis_test(
                y_test_actual, entry[f"LB_{cl_str}"], es_pred, cl
            )
            entry[f"bayer_dimitriadis_{es_str}"] = bayer_dimitriadis_result
            entry[f"bd_p_value_{es_str}"] = bayer_dimitriadis_result["p_value"]
            entry[f"bd_mean_violation_{es_str}"] = bayer_dimitriadis_result["mean_z"]
            print(
                f"Bayer-Dimitriadis Test ({cl_str}%):\n"
                f"Test statistic: {bayer_dimitriadis_result['test_statistic']}\n"
                f"p-value: {bayer_dimitriadis_result['p_value']}\n"
            )
        else:
            print(f"No ES_{es_str} predictions available for Bayer-Dimitriadis test.")

    # Calculate RMSE
    rmse = calculate_rmse(y_test_actual, entry["mean_pred"])
    entry["rmse"] = rmse

    correlation = calculate_uncertainty_error_correlation(
        y_test_actual, entry["mean_pred"], interval_width
    )
    entry["uncertainty_error_correlation"] = correlation

    # Calculate sign of return accuracy
    sign_accuracy = np.nanmean(np.sign(y_test_actual) == np.sign(entry["mean_pred"]))
    entry["sign_accuracy"] = sign_accuracy

# %%
# Compile results into DataFrame
pd.set_option("display.max_rows", 500)
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
    "Bayer-Dimitriadis pass",
    "Bayer-Dimitriadis p-value",
    "Bayer-Dimitriadis mean violation",
    "Bayer-Dimitriadis violation SD",
]
results = {
    "Model": [],
    # Non-quantile-based metrics
    "NLL": [],
    "CRPS": [],
    "RMSE": [],
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
    results["RMSE"].append(entry["rmse"])
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
        if entry.get(f"bayer_dimitriadis_{es_str}") is None:
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis pass"].append(np.nan)
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis p-value"].append(np.nan)
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis mean violation"].append(
                np.nan
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis violation SD"].append(
                np.nan
            )
        else:
            bd_test_result = entry[f"bayer_dimitriadis_{es_str}"]
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis pass"].append(
                interpret_bayer_dimitriadis_stat(bd_test_result["p_value"])
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis p-value"].append(
                bd_test_result["p_value"]
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis mean violation"].append(
                bd_test_result["mean_z"]
            )
            results[f"[{format_cl(es_alpha)}] Bayer-Dimitriadis violation SD"].append(
                bd_test_result["std_z"]
            )


results_df = pd.DataFrame(results)
results_df = results_df.set_index("Model")

# %%
# Remove inadequate models
for model in results_df.index:
    if "GARCH" in model:
        continue
    passes = 0
    for cl in CONFIDENCE_LEVELS:
        if results_df.loc[model, f"[{format_cl(cl)}] Pooled CC p-value"] > 0.05:
            passes += 1
    if passes == 0:
        results_df.drop(model, inplace=True)

# %%
# Identify winners
results_df.loc["Winner", "NLL"] = results_df["NLL"].idxmin()
results_df.loc["Winner", "CRPS"] = results_df["CRPS"].idxmin()
results_df.loc["Winner", "Correlation (vol. vs. errors)"] = results_df[
    "Correlation (vol. vs. errors)"
].idxmax()
results_df.loc["Winner", "RMSE"] = results_df["RMSE"].idxmin()
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
    results_df.loc["Winner", f"[{es_str}] Bayer-Dimitriadis p-value"] = results_df[
        f"[{es_str}] Bayer-Dimitriadis p-value"
    ].idxmax()
    results_df.loc["Winner", f"[{es_str}] Bayer-Dimitriadis mean violation"] = (
        results_df[f"[{es_str}] Bayer-Dimitriadis mean violation"].abs().idxmin()
    )
results_df = results_df.T
results_df.to_csv(f"results/comp_results{SUFFIX}.csv")
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
        ascending = True  # lower difference is better
    elif "mean violation" in metric:
        # Rank by closeness to zero.
        key = values.abs()
        ascending = True
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
rankings_df = rankings_df.astype(pd.Int64Dtype)
rankings_df.loc["Rank Sum"] = rankings_df.sum()
rankings_df

# %%
print("Lowest total rank:", rankings_df.loc["Rank Sum"].idxmin())

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
            entry["sector_pass_pct"] = pass_pct
            unique_sectors.update(passes.index)

        # Get the union of sectors from both datasets
        sectors = sorted(
            unique_sectors,
            key=lambda sector: sum(
                entry["sector_pass_pct"].get(sector, 0)
                for entry in passing_models
                if entry["name"] != "GARCH"
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
            pass_pct = entry["sector_pass_pct"].reindex(sectors, fill_value=0)
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

    for entry in passing_models:
        chr_results_df = entry[f"chr_results_df_{format_cl(cl)}"]
        chr_results_df = chr_results_df.join(meta_df, how="left")
        chr_results_df = chr_results_df.dropna(subset="all_pass")
        entry["sector_coverage"] = (
            chr_results_df.groupby(sector_key)["Coverage"]
            .mean()
            .sort_values(ascending=False)
        )
        unique_sectors.update(passes.index)

    # Get the union of sectors from both datasets
    sectors = sorted(
        unique_sectors,
        key=lambda sector: sum(
            entry["sector_coverage"].get(sector, 0)
            for entry in passing_models
            if entry["name"] != "GARCH"
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
        pass_pct = entry["sector_coverage"].reindex(sectors, fill_value=0)
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
for cl in CONFIDENCE_LEVELS:
    existing_tickers = sorted(
        set(df_validation.index.get_level_values("Symbol")).intersection(
            IMPORTANT_TICKERS
        )
    )
    y = np.arange(len(existing_tickers))

    plt.figure(figsize=(10, len(existing_tickers) * 0.5))
    plt.title("PICP by ticker ({:.0f}% interval)".format(cl * 100))
    x_from = cl - 0.05
    x_to = cl + 0.05

    num_models = len(passing_models)
    offsets = np.linspace(
        -group_height / 2 + bar_height / 2,
        group_height / 2 - bar_height / 2,
        num_models,
    )

    for i, entry in enumerate(passing_models):
        chr_results_df = entry[f"chr_results_df_{format_cl(cl)}"]
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
# Calculate p-value of outperformance in terms of NLL
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
        benchmark_nll = benchmark_entry["nll"]
        challenger_nll = challenger_entry["nll"]
        mask = ~np.isnan(benchmark_nll) & ~np.isnan(challenger_nll)
        benchmark_nll = benchmark_nll[mask]
        challenger_nll = challenger_nll[mask]

        # Paired one-sided t-test
        t_stat, p_value = ttest_rel(challenger_nll, benchmark_nll, alternative="less")

        # Store the p-value in the dataframe
        p_value_df.loc[benchmark, challenger] = p_value

p_value_df

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
for entry in preds_per_model:
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
for entry in preds_per_model:
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
for entry in preds_per_model:
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

for entry in preds_per_model:
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
    for entry in preds_per_model:
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
