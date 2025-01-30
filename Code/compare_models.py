# %%
# Define parameters
from settings import LOOKBACK_DAYS, TEST_ASSET, DATA_PATH, TRAIN_TEST_SPLIT

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
# Check if TEST_ASSET is in the data
if TEST_ASSET not in df.index.get_level_values("Symbol"):
    raise ValueError(f"TEST_ASSET '{TEST_ASSET}' not found in the data")

# %%
# Filter away data before 1990
df = df[df.index.get_level_values("Date") >= "1990-01-01"]
df

# %%
# Get test part of df
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test


# %%
# Collect models and their predictions (in order of model complexity)
preds_per_model = []

# Linear Regression Model
try:
    linreg_preds = pd.read_csv(
        f"predictions/linreg_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
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
    garch_vol_pred = pd.read_csv(
        f"predictions/garch_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
    )["Volatility"].values
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
    garch_gjr_vol_pred = pd.read_csv(
        f"predictions/gjr_garch_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
    )["Volatility"].values
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
    lstm_preds = pd.read_csv(
        f"predictions/lstm_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
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
    lstm_mc_preds = pd.read_csv(
        f"predictions/lstm_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
    )
    preds_per_model.append(
        {
            "name": "LSTM-MC",
            "mean_pred": lstm_mc_preds["Mean"].values,
            "volatility_pred": lstm_mc_preds["Volatility"].values,
            "epistemic_sd": lstm_mc_preds["Epistemic_Uncertainty_Volatility"].values,
        }
    )
except FileNotFoundError:
    print("LSTM-MC predictions not found")

# LSTM with FFNN Model
try:
    lstm_w_ffnn_preds = pd.read_csv(
        f"predictions/lstm_ffnn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
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
    lstm_w_ffnn_mc_preds = pd.read_csv(
        f"predictions/lstm_ffnn_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
    )
    preds_per_model.append(
        {
            "name": "LSTM w FFNN and MC Dropout",
            "mean_pred": lstm_w_ffnn_mc_preds["Mean"].values,
            "volatility_pred": lstm_w_ffnn_mc_preds["Volatility"].values,
            "epistemic_sd": lstm_w_ffnn_mc_preds[
                "Epistemic_Uncertainty_Volatility"
            ].values,
        }
    )
except FileNotFoundError:
    print("LSTM w FFNN and MC Dropout predictions not found")

# Mini LSTM
try:
    mini_lstm_preds = pd.read_csv(
        f"predictions/lstm_mini_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
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
    mini_lstm_w_rvol_preds = pd.read_csv(
        f"predictions/lstm_mini_w_rvol_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
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
    mini_lstm_w_rvol_and_vix_preds = pd.read_csv(
        f"predictions/lstm_mini_w_rvol_and_vix_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
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
    mc_lstm_w_rvol_and_vix_preds = pd.read_csv(
        f"predictions/lstm_mc_w_rvol_and_vix_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
    )
    preds_per_model.append(
        {
            "name": "Monte Carlo LSTM w RVOL & VIX",
            "mean_pred": mc_lstm_w_rvol_and_vix_preds["Mean"].values,
            "volatility_pred": mc_lstm_w_rvol_and_vix_preds["Volatility"].values,
            "epistemic_sd": mc_lstm_w_rvol_and_vix_preds[
                "Epistemic_Uncertainty_Volatility"
            ].values,
        }
    )
except FileNotFoundError:
    print("Monte Carlo LSTM w RVOL and VIX predictions not found")

# Transformer MDN
try:
    transformer_mdn_preds = pd.read_csv(
        f"predictions/transformer_mdn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
    )
    preds_per_model.append(
        {
            "name": "Transformer MDN",
            "mean_pred": transformer_mdn_preds["Mean_SP"].values,
            "volatility_pred": transformer_mdn_preds["Vol_SP"].values,
        }
    )
except FileNotFoundError:
    print("Transformer MDN predictions not found")

# Transformer MDN w MC Dropout
try:
    transformer_mdn_mc_preds = pd.read_csv(
        f"predictions/transformer_mdn_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
    )
    preds_per_model.append(
        {
            "name": "Transformer MDN w MC Dropout",
            "mean_pred": transformer_mdn_mc_preds["Mean_MC"].values,
            "volatility_pred": transformer_mdn_mc_preds["Vol_MC"].values,
            "epistemic_sd": transformer_mdn_mc_preds["Epistemic_Unc_Mean"].values,
        }
    )
except FileNotFoundError:
    print("Transformer MDN w MC Dropout predictions not found")

if TEST_ASSET == "S&P":
    # VIX
    try:
        vix_preds = pd.read_csv(f"data/VIX.csv")
        vix_preds["Date"] = pd.to_datetime(vix_preds["Date"])
        vix_preds = vix_preds.set_index("Date")
        vix_preds = vix_preds.loc[df_test.index]
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
        # Reindex to same index as df_test and fill missing
        rvol_preds = rvol_preds.reindex(df_test.index, method="ffill")
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


# %%
# Functions for plotting and evaluation
def plot_volatility_prediction(model, df_test, abs_returns_test):
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, abs_returns_test, label="Absolute Returns", color="black")
    plt.plot(
        df_test.index,
        model["volatility_pred"],
        label=f"{model['name']} Volatility Prediction",
    )
    if "epistemic_sd" in model:
        plt.fill_between(
            df_test.index,
            model["volatility_pred"] - model["epistemic_sd"],
            model["volatility_pred"] + model["epistemic_sd"],
            alpha=0.5,
            label="67% confidence interval",
        )
    plt.title(f"Volatility Prediction with {model['name']}")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.savefig(
        f"results/{model['name'].replace(' ', '_').replace('/', ':').lower()}.svg"
    )
    plt.show()


def plot_mean_returns_prediction(model, df_test):
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_test.index, model["mean_pred"], label=f"{model['name']} Mean Prediction"
    )
    plt.fill_between(
        df_test.index,
        model["mean_pred"] - model["volatility_pred"],
        model["mean_pred"] + model["volatility_pred"],
        alpha=0.5,
        label="Volatility",
    )
    if "epistemic_sd" in model:
        plt.fill_between(
            df_test.index,
            model["mean_pred"] - model["volatility_pred"] - model["epistemic_sd"],
            model["mean_pred"] + model["volatility_pred"] + model["epistemic_sd"],
            alpha=0.3,
            label="Volatility w Epistemic Uncertainty",
        )
    plt.scatter(
        df_test.index,
        df_test["LogReturn"],
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


def delta_sign_accuracy(y_true, vol_pred):
    y_true_diff = np.diff(np.abs(y_true))
    vol_pred_diff = np.diff(vol_pred)
    return np.mean(np.sign(y_true_diff) == np.sign(vol_pred_diff))


def calculate_nll(y_true, mean_pred, volatility_pred):
    sigma2 = volatility_pred**2

    term_1 = ((y_true - mean_pred) ** 2) / (2 * sigma2)
    term_2 = 0.5 * np.log(sigma2)

    return np.mean(term_1 + term_2)


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


# %%
# Evaluate models
alpha = 0.05
y_test_actual = df_test["LogReturn"].values
abs_returns_test = np.abs(y_test_actual)

for entry in preds_per_model:
    # Plot volatility predictions
    plot_volatility_prediction(entry, df_test, abs_returns_test)

    # Plot mean return predictions
    plot_mean_returns_prediction(entry, df_test)

    # Calculate prediction intervals
    calculate_prediction_intervals(entry, alpha)

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
        y_test_actual, entry["lower_bounds"], entry["upper_bounds"], alpha
    )
    entry["interval_score"] = interval_score

    # Calculate NLL
    nll = calculate_nll(y_test_actual, entry["mean_pred"], entry["volatility_pred"])
    entry["nll"] = nll

    # Uncertainty-Error Correlation
    interval_width = entry["upper_bounds"] - entry["lower_bounds"]
    correlation = calculate_uncertainty_error_correlation(
        y_test_actual, entry["mean_pred"], interval_width
    )
    entry["uncertainty_error_correlation"] = correlation

    # Delta Sign Accuracy
    delta_accuracy = delta_sign_accuracy(y_test_actual, entry["volatility_pred"])
    entry["delta_sign_accuracy"] = delta_accuracy

    # Christoffersen's Test
    exceedances = ~entry["within_bounds"]
    christoffersen_result = christoffersen_test(exceedances, alpha)
    entry["christoffersen_test"] = interpret_christoffersen_test(christoffersen_result)

    # Print Christoffersen's Test Results
    print(f"\n{entry['name']} Christoffersen's Test Results:")
    display(entry["christoffersen_test"])

# %%
# Compile results into DataFrame
results = {
    "Model": [],
    "PICP": [],
    "PICP Miss": [],
    "Mean width (MPIW)": [],
    "Interval Score": [],
    "Correlation (vol. vs. errors)": [],
    "PICP/MPIW": [],
    "NLL": [],
    "Direction Accuracy": [],
}

for entry in preds_per_model:
    picp_miss = 1 - alpha - entry["picp"]
    results["Model"].append(entry["name"])
    results["PICP"].append(entry["picp"])
    results["PICP Miss"].append(picp_miss)
    results["Mean width (MPIW)"].append(entry["mpiw"])
    results["Interval Score"].append(entry["interval_score"])
    results["Correlation (vol. vs. errors)"].append(
        entry["uncertainty_error_correlation"]
    )
    results["PICP/MPIW"].append(entry["picp"] / entry["mpiw"])
    results["Direction Accuracy"].append(entry["delta_sign_accuracy"])
    results["NLL"].append(np.mean(entry["nll"]))

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
results_df.loc["Winner", "PICP/MPIW"] = results_df["PICP/MPIW"].idxmax()
results_df.loc["Winner", "Direction Accuracy"] = results_df[
    "Direction Accuracy"
].idxmax()
results_df.loc["Winner", "NLL"] = results_df["NLL"].idxmin()
results_df = results_df.T
results_df.to_csv(f"results/comp_results.csv")
results_df

# %%
# Calculate overall winner
winner_name = results_df["Winner"].mode()[0]
winner_name


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
    plt.plot(
        returns_test.index[from_idx:to_idx],
        rvol_vol_est[from_idx:to_idx],
        label="True RVOL",
        linestyle="--",
        color="black",
        linewidth=1,
    )
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
        if "epistemic_sd" in model:
            plt.fill_between(
                returns_test.index[from_idx:to_idx],
                model["volatility_pred"][from_idx:to_idx]
                - model["epistemic_sd"][from_idx:to_idx],
                model["volatility_pred"][from_idx:to_idx]
                + model["epistemic_sd"][from_idx:to_idx],
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


returns_test = df_test["LogReturn"]

plot_volatility_comparison(
    preds_per_model, returns_test, abs_returns_test, lookback_days=60, steps=30
)

# %%
