# %%
# Define parameters (based on settings)
import subprocess
from typing import Optional
from transformer_mdn_ensemble import add_day_indices
from shared.ensemble import MDNEnsemble, ParallelProgressCallback
from shared.conf_levels import format_cl
from settings import LOOKBACK_DAYS, SUFFIX, TEST_SET
import multiprocessing as mp

VERSION = "rv-iv"
MODEL_NAME = f"mdn_ensemble_{VERSION}_{TEST_SET}"

# %%
# Model selection
models = [
    f"transformer_mdn_ensemble_rv-and-ivol_{TEST_SET}",
    f"lstm_mdn_ensemble{SUFFIX}_vrv-and-ivol-final_{TEST_SET}",
]

# %%
# Feature selection
MULTIPLY_MARKET_FEATURES_BY_BETA = False
PI_PENALTY = False
MU_PENALTY = False
SIGMA_PENALTY = False
INCLUDE_MARKET_FEATURES = False
INCLUDE_RETURNS = False
INCLUDE_FNG = False
INCLUDE_RETURNS = False
INCLUDE_INDUSTRY = False
INCLUDE_GARCH = False
INCLUDE_BETA = False
INCLUDE_OTHERS = False
INCLUDE_FRED_MD = False
INCLUDE_10_DAY_IVOL = True
INCLUDE_30_DAY_IVOL = True
INCLUDE_1MIN_RV = True
INCLUDE_5MIN_RV = True
INCLUDE_TICKERS = False

# %%
# Settings
N_ENSEMBLE_MEMBERS = len(models)
SUBMODEL_MIXTURES = 10
SUBMODEL_MEMBERS = 10
SUBMODEL_TOTAL_MIXTURES = SUBMODEL_MIXTURES * SUBMODEL_MEMBERS

# %%
# Imports from code shared across models
from shared.mdn import (
    calculate_es_for_quantile,
    calculate_intervals_vectorized,
    calculate_prob_above_zero_vectorized,
    get_mdn_bias_initializer,
    get_mdn_kernel_initializer,
    parse_mdn_output,
    plot_sample_days,
    univariate_mixture_mean_and_var_approx,
)
from shared.loss import (
    ece_mdn,
    mdn_crps_tf,
    mdn_nll_numpy,
    mean_mdn_crps_tf,
    mdn_nll_tf,
)
from shared.processing import get_lstm_train_test_new

# %%
# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import gc

# TensorFlow / Keras
import tensorflow as tf

warnings.filterwarnings("ignore")

# %%
# Begin script
if __name__ == "__main__":
    print(f"MDN Ensemble v{VERSION}")

    # %%
    # Load preprocessed data
    data = get_lstm_train_test_new(
        multiply_by_beta=MULTIPLY_MARKET_FEATURES_BY_BETA,
        include_returns=INCLUDE_RETURNS,
        include_spx_data=INCLUDE_MARKET_FEATURES,
        include_others=INCLUDE_OTHERS,
        include_beta=INCLUDE_BETA,
        include_fng=INCLUDE_FNG,
        include_garch=INCLUDE_GARCH,
        include_industry=INCLUDE_INDUSTRY,
        include_fred_md=INCLUDE_FRED_MD,
        include_1min_rv=INCLUDE_1MIN_RV,
        include_5min_rv=INCLUDE_5MIN_RV,
        include_ivol_cols=(["10 Day Call IVOL"] if INCLUDE_10_DAY_IVOL else [])
        + (["Historical Call IVOL"] if INCLUDE_30_DAY_IVOL else []),
    )

    train = data.get_training_set(test_set=TEST_SET)
    test = data.get_test_set(TEST_SET)

    # %%
    # Garbage collection
    gc.collect()

    # %%
    # 1) Inspect shapes
    print(f"X_train.shape: {train.X.shape}, y_train.shape: {train.y.shape}")
    print(f"Training dates: {train.dates.min()} to {train.dates.max()}")
    print(f"Validation set shape: {test.X.shape}, {test.y.shape}")
    print(f"Validation dates: {test.dates.min()} to {test.dates.max()}")

    # %%
    # 2) Load members
    members = []
    for model in models:
        print(f"Loading {model}...")
        member_model = tf.keras.models.load_model(
            f"models/{model}.keras",
            custom_objects={
                "loss_fn": mdn_nll_tf(SUBMODEL_TOTAL_MIXTURES),
                "mdn_kernel_initializer": get_mdn_kernel_initializer(SUBMODEL_MIXTURES),
                "mdn_bias_initializer": get_mdn_bias_initializer(SUBMODEL_MIXTURES),
                "MDNEnsemble": MDNEnsemble,
                "add_day_indices": add_day_indices,
            },
        )
        members.append(member_model)
    ensemble_model = MDNEnsemble(members, SUBMODEL_TOTAL_MIXTURES, name=MODEL_NAME)
    ensemble_model

    # %%
    # 8) Single-pass predictions
    y_pred_mdn, epistemic_var = ensemble_model.predict(
        [test.X, test_ticker_ids] if INCLUDE_TICKERS else test.X
    )
    pi_pred, mu_pred, sigma_pred = parse_mdn_output(
        y_pred_mdn, SUBMODEL_TOTAL_MIXTURES * N_ENSEMBLE_MEMBERS
    )

    # %%
    # 9) Plot 10 charts with the distributions for 10 random days
    example_tickers = ["AAPL", "WMT", "GS"]
    for ticker in example_tickers:
        s = test.sets[ticker]
        from_idx, to_idx = test.get_range(ticker)
        plot_sample_days(
            s.y_dates,
            s.y,
            pi_pred[from_idx:to_idx],
            mu_pred[from_idx:to_idx],
            sigma_pred[from_idx:to_idx],
            SUBMODEL_TOTAL_MIXTURES * N_ENSEMBLE_MEMBERS,
            ticker=ticker,
            save_to=f"results/distributions/{ticker}_ensemble_mdn_v{VERSION}_ensemble.svg",
        )

    # %%
    # 10) Plot weights over time to show how they change
    fig, axes = plt.subplots(
        nrows=len(example_tickers), figsize=(18, len(example_tickers) * 9)
    )

    # Dictionary to store union of legend entries
    legend_dict = {}

    for ax, ticker in zip(axes, example_tickers):
        s = test.sets[ticker]
        from_idx, to_idx = test.get_range(ticker)
        pi_pred_ticker = pi_pred[from_idx:to_idx]
        for j in range(SUBMODEL_TOTAL_MIXTURES * N_ENSEMBLE_MEMBERS):
            mean_over_time = np.mean(pi_pred_ticker[:, j], axis=0)
            if mean_over_time < 0.01:
                continue
            (line,) = ax.plot(s.y_dates, pi_pred_ticker[:, j], label=f"$\pi_{{{j}}}$")
            # Only add new labels
            if f"$\pi_{{{j}}}$" not in legend_dict:
                legend_dict[f"$\pi_{{{j}}}$"] = line
        ax.set_yticklabels(["{:.2f}%".format(x * 100) for x in ax.get_yticks()])
        ax.set_title(f"Evolution of Mixture Weights for {ticker}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Weight")

    # Create a combined legend using the union of entries
    handles = list(legend_dict.values())
    labels = list(legend_dict.keys())
    fig.legend(handles, labels, loc="center left")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results/ensemble_mdn_v{VERSION}_ensemble_mixture_weights.svg")

    # %%
    # 11) Calculate intervals for 67%, 95%, 97.5% and 99% confidence levels
    confidence_levels = [0.67, 0.90, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999]
    intervals = calculate_intervals_vectorized(
        pi_pred, mu_pred, sigma_pred, confidence_levels
    )

    # %%
    # 13) Plot predicted means with epistemic uncertainty for example tickers
    mean = (pi_pred * mu_pred).numpy().sum(axis=1)
    for ticker in example_tickers:
        from_idx, to_idx = test.get_range(ticker)
        ticker_mean = mean[from_idx:to_idx]
        filtered_mean = ticker_mean
        epistemic_sd = np.sqrt(epistemic_var[from_idx:to_idx])
        filtered_epistemic_sd = epistemic_sd
        s = test.sets[ticker]

        dates = s.y_dates
        actual_return = s.y
        plt.figure(figsize=(12, 6))
        plt.plot(
            dates,
            actual_return,
            label="Actual Returns",
            color="black",
            alpha=0.5,
        )
        plt.plot(dates, filtered_mean, label="Predicted Mean", color="red")
        plt.fill_between(
            dates,
            filtered_mean - filtered_epistemic_sd,
            filtered_mean + filtered_epistemic_sd,
            color="blue",
            alpha=0.8,
            label="Epistemic Uncertainty (67%)",
        )
        plt.fill_between(
            dates,
            filtered_mean - 2 * filtered_epistemic_sd,
            filtered_mean + 2 * filtered_epistemic_sd,
            color="blue",
            alpha=0.5,
            label="Epistemic Uncertainty (95%)",
        )
        plt.fill_between(
            dates,
            filtered_mean - 2.57 * filtered_epistemic_sd,
            filtered_mean + 2.57 * filtered_epistemic_sd,
            color="blue",
            alpha=0.3,
            label="Epistemic Uncertainty (99%)",
        )
        plt.fill_between(
            dates,
            filtered_mean - 3.29 * filtered_epistemic_sd,
            filtered_mean + 3.29 * filtered_epistemic_sd,
            color="blue",
            alpha=0.1,
            label="Epistemic Uncertainty (99.9%)",
        )
        plt.axhline(
            actual_return.mean(),
            color="red",
            linestyle="--",
            label="True mean return across time",
            alpha=0.5,
        )
        plt.gca().set_yticklabels(
            ["{:.1f}%".format(x * 100) for x in plt.gca().get_yticks()]
        )
        plt.title(f"Ensemble MDN predictions for {ticker}, {TEST_SET} data")
        plt.xlabel("Date")
        plt.ylabel("LogReturn")
        plt.legend()
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"results/time_series/epistemic/{ticker}_ensemble_mdn_v{VERSION}_ensemble.svg"
        )

    # %%
    # 13) Make data frame for signle pass predictions
    df_validation = pd.DataFrame(
        np.vstack([test.dates, test.tickers]).T,
        columns=["Date", "Symbol"],
    )
    # %%
    # For comparison to other models, compute mixture means & variances
    uni_mixture_mean_sp, uni_mixture_var_sp = univariate_mixture_mean_and_var_approx(
        pi_pred, mu_pred, sigma_pred
    )
    uni_mixture_mean_sp = uni_mixture_mean_sp.numpy()
    uni_mixture_std_sp = np.sqrt(uni_mixture_var_sp.numpy())
    df_validation["Mean_SP"] = uni_mixture_mean_sp
    df_validation["Vol_SP"] = uni_mixture_std_sp

    # %%
    # Calculate loss
    df_validation["NLL"] = mdn_nll_numpy(SUBMODEL_TOTAL_MIXTURES * N_ENSEMBLE_MEMBERS)(
        test.y, y_pred_mdn
    )

    # %%
    # Calculate CRPS
    crps = mdn_crps_tf(SUBMODEL_TOTAL_MIXTURES * N_ENSEMBLE_MEMBERS)
    df_validation["CRPS"] = crps(test.y, y_pred_mdn)

    # %%
    # Calculate ECE
    ece = ece_mdn(SUBMODEL_TOTAL_MIXTURES * N_ENSEMBLE_MEMBERS, test.y, y_pred_mdn)
    df_validation["ECE"] = ece
    ece

    # %%
    # Add confidence intervals
    for i, cl in enumerate(confidence_levels):
        df_validation[f"LB_{format_cl(cl)}"] = intervals[:, i, 0]
        df_validation[f"UB_{format_cl(cl)}"] = intervals[:, i, 1]

    # %%
    # Calculate expected shortfall
    for i, cl in enumerate(confidence_levels):
        alpha = (1 - cl) / 2  # The lower quantile of the confidence interval
        var_estimates = intervals[:, i, 0]
        es = calculate_es_for_quantile(pi_pred, mu_pred, sigma_pred, var_estimates)
        df_validation[f"ES_{format_cl(1-alpha)}"] = es

    # %%
    # Example plot of ES
    for ticker in example_tickers:
        df_validation.set_index(["Date", "Symbol"]).xs(
            ticker, level="Symbol"
        ).sort_index()[["LB_90", "ES_95", "LB_98", "ES_99"]].rename(
            columns={
                "LB_90": "95% VaR",
                "ES_95": "95% ES",
                "LB_98": "99% VaR",
                "ES_99": "99% ES",
            }
        ).plot(
            title=f"99% VaR and ES for {ticker}",
            # Color appropriately
            color=["#ffaaaa", "#ff0000", "#aaaaff", "#0000ff"],
            figsize=(12, 6),
        )
        plt.plot(
            test.sets[ticker].df.reset_index().set_index("Date")["ActualReturn"],
            label="Actual Returns",
            color="black",
        )
        plt.legend()

    # %%
    # Calculate probability of price increase
    df_validation["Prob_Increase"] = calculate_prob_above_zero_vectorized(
        pi_pred, mu_pred, sigma_pred
    )

    # %%
    # Add epistemic variance
    df_validation["EpistemicVarMean"] = epistemic_var

    # %%
    # Save
    df_validation.set_index(["Date", "Symbol"]).to_csv(
        f"predictions/ensemble_mdn_predictions{SUFFIX}_v{VERSION}_ensemble.csv"
    )
