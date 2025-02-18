# %%
# Define parameters
from shared.loss import nll_loss_mean_and_vol
from settings import (
    DATA_PATH,
    SUFFIX,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)
from shared.crps import crps_loss_mean_and_vol

# %%
# Defined which confidence level to use for prediction intervals
CONFIDENCE_LEVEL = 0.05

# %%
# Exclude uninteresting models
EXCLUDE_MODELS = []

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
# Filter away data before 1990
df = df[df.index.get_level_values("Date") >= "1990-01-01"]
df

# %%
# Get validation part of df
dates = df.index.get_level_values("Date")
df_validation = df[(dates >= TRAIN_VALIDATION_SPLIT) & (dates < VALIDATION_TEST_SPLIT)]
df_validation


# %%
# Collect models and their predictions (in order of model complexity)
preds_per_model = []

# GARCH Model
try:
    garch_df = pd.read_csv(f"data/sp500_stocks_garch.csv")
    garch_df["Date"] = pd.to_datetime(garch_df["Date"])
    garch_df = garch_df.set_index(["Date", "Symbol"])
    garch_dates = garch_df.index.get_level_values("Date")
    garch_df = garch_df[
        (garch_dates >= TRAIN_VALIDATION_SPLIT) & (garch_dates < VALIDATION_TEST_SPLIT)
    ]
    combined_df = df_validation.join(garch_df, how="left", rsuffix="_GARCH")
    garch_vol_pred = combined_df["GARCH_Vol"].values
    preds_per_model.append(
        {
            "name": "GARCH",
            "mean_pred": np.zeros_like(garch_vol_pred),
            "volatility_pred": garch_vol_pred,
            "symbols": combined_df.index.get_level_values("Symbol"),
        }
    )
    nans = np.isnan(garch_vol_pred).sum()
    if nans > 0:
        print(f"GARCH has {nans} NaN predictions")
except FileNotFoundError:
    print("GARCH predictions not found")

# LSTM MDN
for version in ["vquick", "vfe"]:
    try:
        lstm_mdn_df = pd.read_csv(
            f"predictions/lstm_mdn_predictions{SUFFIX}_{version}.csv"
        )
        lstm_mdn_df["Date"] = pd.to_datetime(lstm_mdn_df["Date"])
        lstm_mdn_df = lstm_mdn_df.set_index(["Date", "Symbol"])
        lstm_mdn_dates = lstm_mdn_df.index.get_level_values("Date")
        lstm_mdn_df = lstm_mdn_df[
            (lstm_mdn_dates >= TRAIN_VALIDATION_SPLIT)
            & (lstm_mdn_dates < VALIDATION_TEST_SPLIT)
        ]
        combined_df = df_validation.join(lstm_mdn_df, how="left", rsuffix="_LSTM_MDN")
        preds_per_model.append(
            {
                "name": f"LSTM MDN {version}",
                "mean_pred": combined_df["Mean_SP"].values,
                "volatility_pred": combined_df["Vol_SP"].values,
                "LB_95": combined_df["LB_95"].values,
                "UB_95": combined_df["UB_95"].values,
                "nll": np.nanmean(combined_df["NLL"].values),
                "symbols": combined_df.index.get_level_values("Symbol"),
                # "crps": lstm_mdn_preds["CRPS"].values.mean(),
            }
        )
        nans = combined_df["Mean_SP"].isnull().sum()
        if nans > 0:
            print(f"LSTM MDN {version} has {nans} NaN predictions")
    except FileNotFoundError:
        print(f"LSTM MDN {version} predictions not found")


# %%
# Remove excluded models
preds_per_model = [
    model for model in preds_per_model if model["name"] not in EXCLUDE_MODELS
]


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
    correlation = pd.Series(prediction_errors).corr(pd.Series(interval_width))
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


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2))


# %%
# Evaluate models
y_test_actual = df_validation["LogReturn"].values
abs_returns_test = np.abs(y_test_actual)

for entry in preds_per_model:
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
        )

    # Calculate quantile loss
    entry["quantile_loss"] = np.nanmean(
        np.maximum(
            y_test_actual - entry["upper_bounds"],
            entry["lower_bounds"] - y_test_actual,
        )
    )

    # Calculate Lopez loss function
    entry["lopez_loss"] = np.nanmean(
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
    sign_accuracy = np.nanmean(np.sign(y_test_actual) == np.sign(entry["mean_pred"]))
    entry["sign_accuracy"] = sign_accuracy

    exceedance_df = pd.DataFrame(
        {
            "Symbol": entry["symbols"],
            "Within Bounds": entry["within_bounds"],
        }
    )
    exceedance_df = exceedance_df.dropna(subset=["Within Bounds"])

    pooled_exceedances_list = []
    reset_indices = []
    start_index = 0

    # Group by symbol in original order.
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
            pooled_exceedances, CONFIDENCE_LEVEL, reset_indices=reset_indices
        )
        entry["christoffersen_test"] = interpret_christoffersen_test(pooled_result)

    print(f"\n{entry['name']} Pooled Christoffersen's Test Results:")
    try:
        display(entry["christoffersen_test"])
    except NameError:
        print(entry["christoffersen_test"])

    chr_results = []
    for symbol, within_bounds in exceedance_df.groupby("Symbol")["Within Bounds"]:
        if within_bounds.isna().any():
            print(f"Skipping {symbol} due to NaN values")
            continue
        exceedances = ~within_bounds.astype(bool)
        result = christoffersen_test(exceedances, CONFIDENCE_LEVEL)
        chr_results.append(
            {**result, "Symbol": symbol, "Coverage": within_bounds.mean()}
        )
    chr_results_df = pd.DataFrame(chr_results).set_index("Symbol")
    chr_results_df["uc_pass"] = np.where(
        chr_results_df["p_value_uc"].isna(), np.nan, chr_results_df["p_value_uc"] > 0.05
    )
    chr_results_df["ind_pass"] = np.where(
        chr_results_df["p_value_ind"].isna(),
        np.nan,
        chr_results_df["p_value_ind"] > 0.05,
    )
    chr_results_df["cc_pass"] = np.where(
        chr_results_df["p_value_cc"].isna(), np.nan, chr_results_df["p_value_cc"] > 0.05
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
    entry["christoffersen_test"] = interpret_christoffersen_test(
        chr_results_df.replace([np.inf, -np.inf], np.nan).mean()
    )

    # Print Christoffersen's Test Results
    print(f"\n{entry['name']} Average Christoffersen's Test Results:")

    entry["chr_results_df"] = chr_results_df
    entry["uc_pass_pct"] = chr_results_df["uc_pass"].mean() * 100
    entry["ind_pass_pct"] = chr_results_df["ind_pass"].mean() * 100
    entry["cc_pass_pct"] = chr_results_df["cc_pass"].mean() * 100
    entry["chr_pass_pct"] = chr_results_df["all_pass"].mean() * 100
    print(
        f"Unconditional Coverage Pass Rate: {entry['uc_pass_pct']:.2f}%\n"
        f"Independence Pass Rate: {entry['ind_pass_pct']:.2f}%\n"
        f"Conditional Coverage Pass Rate: {entry['cc_pass_pct']:.2f}%\n"
        f"All Pass Rate: {entry['chr_pass_pct']:.2f}%"
    )

    # # Calculate CRPS
    # if "crps" not in entry:
    #     print("Calculating CRPS for", entry["name"])
    #     entry["crps"] = crps_loss_mean_and_vol(
    #         y_test_actual, entry["mean_pred"], entry["volatility_pred"]
    #     ).mean()

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
    "UC Pass %": [],
    "Ind Pass %": [],
    "CC Pass %": [],
    "CHR Pass %": [],
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
    results["CRPS"].append(entry.get("crps"))
    results["Lopez Loss"].append(entry["lopez_loss"])
    results["RMSE"].append(entry["rmse"])
    results["Sign accuracy"].append(entry["sign_accuracy"])
    results["UC Pass %"].append(entry["uc_pass_pct"])
    results["Ind Pass %"].append(entry["ind_pass_pct"])
    results["CC Pass %"].append(entry["cc_pass_pct"])
    results["CHR Pass %"].append(entry["chr_pass_pct"])

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
results_df.loc["Winner", "UC Pass %"] = results_df["UC Pass %"].idxmax()
results_df.loc["Winner", "Ind Pass %"] = results_df["Ind Pass %"].idxmax()
results_df.loc["Winner", "CC Pass %"] = results_df["CC Pass %"].idxmax()
results_df.loc["Winner", "CHR Pass %"] = results_df["CHR Pass %"].idxmax()
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
rankings_df = rankings_df.astype(pd.Int64Dtype)
rankings_df.loc["Rank Sum"] = rankings_df.sum()
rankings_df

# %%
# Save rankings to CSV
rankings_df.to_csv(f"results/comp_rankings{SUFFIX}.csv")
# %%
