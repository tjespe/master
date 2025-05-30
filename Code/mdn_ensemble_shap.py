# %%
#!/usr/bin/env python3
"""
Load an MDN ensemble model, compute volatility (variance), Value-at-Risk (VaR),
and Expected Shortfall (ES), then run SHAP analysis on each quantity.
"""

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from lstm_mdn_ensemble import build_lstm_mdn
from shared.conf_levels import format_cl, get_VaR_level
from shared.jupyter import is_notebook
from transformer_mdn_ensemble import build_transformer_mdn
from settings import SUFFIX
from shared.styling_guidelines_graphs import colors

# Model loading parameters
# %%
VERSION = "ivol-final-rolling"  # "rvol-ivol"  # "rv-and-ivol-final-rolling" for LSTM-MDN
# VERSION = "rvol-ivol"  # "rv-and-ivol-final-rolling" for LSTM-MDN
# SHAP takes a long time to run, so we only look at a subset of the data
ANALYSIS_START_DATE = "2024-02-27"
# Model name (used for storing results)
MODEL_NAME = f"lstm_mdn_ensemble{SUFFIX}_v{VERSION}_test"
# MODEL_NAME = f"transformer_mdn_ensemble_{VERSION}_test_expanding"  # f"lstm_mdn_ensemble{SUFFIX}_v{VERSION}_test"
# Filename for weights. Important that the loaded model is not trained on data after
# the ANALYSIS_START_DATE.
MODEL_FNAME = f"models/rolling/{MODEL_NAME}_{ANALYSIS_START_DATE}.h5"
# Function for building the model
# BUILD_FN = build_lstm_mdn  # build_transformer_mdn  # build_lstm_mdn
BUILD_FN = build_transformer_mdn  # build_lstm_mdn

# %%
# Feature name mapping
FEATURE_NAME_MAPPING = {
    "RV": "RV$_{1\\text{-}min}$",
    "RV_5": "RV$_{5\\text{-}min}$",
    "BPV": "BPV$_{1\\text{-}min}$",
    "BPV_5": "BPV$_{5\\text{-}min}$",
    "Good": "RV$^+_{1\\text{-}min}$",
    "Good_5": "RV$^+_{5\\text{-}min}$",
    "Bad": "RV$^-_{1\\text{-}min}$",
    "Bad_5": "RV$^-_{5\\text{-}min}$",
    "RQ": "RQ$_{1\\text{-}min}$",
    "RQ_5": "RQ$_{5\\text{-}min}$",
    "10 Day Call IVOL": "IV$_{10\\text{-}day}$",
    "Historical Call IVOL": "IV$_{20\\text{-}day}$"
}

# %%
# Structural parameters
N_ENSEMBLE_MEMBERS = 10
N_MIXTURES = 10
NUM_HIDDEN_LAYERS = 0
EMBEDDING_DIMENSIONS = None

# %%
# Data flags (must match training settings)
DATA_FLAGS = dict(
    multiply_by_beta=False,
    include_returns=False,
    include_spx_data=False,
    include_others=False,
    include_beta=False,
    include_fng=False,
    include_garch=False,
    include_industry=False,
    include_fred_md=False,
    include_1min_rv="rv" in VERSION,
    include_5min_rv="rv" in VERSION,
    include_ivol_cols=(
        ["10 Day Call IVOL", "Historical Call IVOL"] if "ivol" in VERSION else []
    ),
)

# %%
# Imports
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import shap

from shared.processing import get_lstm_train_test_new
from shared.ensemble import MDNEnsemble
from shared.mdn import (
    calculate_es_for_quantile,
    calculate_intervals_vectorized,
    parse_mdn_output,
    mdn_mean_and_var,
)
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    VALIDATION_TEST_SPLIT,
)


# %%
# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    """
    Load the latest LSTM-MDN ensemble model, compute volatility (variance), Value-at-Risk (VaR),
    and Expected Shortfall (ES), then run SHAP analysis on each quantity.
    This script assumes the model is trained on a dataset with the same features as specified
    in DATA_FLAGS.
    """
    # %%
    # Load data
    data = get_lstm_train_test_new(**DATA_FLAGS)
    first_test_date = pd.to_datetime(ANALYSIS_START_DATE)
    train = data.get_training_set_for_date(first_test_date)
    test = data.get_test_set_for_date(first_test_date)

    X_train = train.X
    X_test = test.X
    num_features = X_train.shape[2]
    cols = list(train.df.columns)[1:]

    # %%
    confidence_levels = [
        0.90,  # 95% VaR
        0.95,  # 97.5% VaR
        0.98,  # 99% VaR
    ]

    # %%
    # Load latest ensemble
    submodels = [BUILD_FN(num_features, None) for _ in range(N_ENSEMBLE_MEMBERS)]
    ensemble_model = MDNEnsemble(submodels, N_MIXTURES)
    ensemble_model.load_weights(MODEL_FNAME)

    # %%
    # Flatten utility
    def flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    # %%
    # Background and test samples for SHAP (adjust sizes as needed)
    # Pick 50 evenly spread samples from the training set for background
    Xb_idx = np.linspace(0, len(X_train) - 1, 50, dtype=int)
    Xb = flatten(X_train[Xb_idx])
    # Pick 20 evenly spread samples from the test set for analysis
    Xt_idx = np.linspace(0, len(X_test) - 1, 20, dtype=int)
    Xt = X_test[Xt_idx]
    Xtf = flatten(Xt)

    # %%
    def predict(x_flat: np.ndarray) -> np.ndarray:
        X = x_flat.reshape(-1, LOOKBACK_DAYS, num_features)
        raw, epistemic_var = ensemble_model.predict(X)
        pi, mu, sigma = parse_mdn_output(raw, N_MIXTURES * N_ENSEMBLE_MEMBERS)
        mean, var = mdn_mean_and_var(pi, mu, sigma)
        aleatoric_var = var - epistemic_var
        vol = np.sqrt(aleatoric_var)
        intervals = calculate_intervals_vectorized(pi, mu, sigma, confidence_levels)
        VaR_estimates = []
        ES_estimates = []
        for i, cl in enumerate(confidence_levels):
            VaR = intervals[:, i, 0]
            es = calculate_es_for_quantile(pi, mu, sigma, VaR)
            ES_estimates.append(es)
            VaR_estimates.append(VaR)
        return np.vstack((vol, *VaR_estimates, *ES_estimates)).T

    # %%
    # SHAP analysis
    expl_var = shap.KernelExplainer(predict, Xb)
    shap_values = expl_var.shap_values(Xtf)

    # %%
    # Save shap_values to file
    os.makedirs("results/xai/raw/", exist_ok=True)
    np.save(f"results/xai/raw/{MODEL_NAME}_{ANALYSIS_START_DATE}_shap.npy", shap_values)

    # %%
    # Load shap_values from file
    shap_values = np.load(
        f"results/xai/raw/{MODEL_NAME}_{ANALYSIS_START_DATE}_shap.npy"
    )

    # %%
    # Present results
    feature_names = [
        f"{FEATURE_NAME_MAPPING[feat]} ({LOOKBACK_DAYS - lag} days ago)".replace(
            "(1 days", "(1 day"
        )
        for lag in range(LOOKBACK_DAYS)
        for feat in cols
    ]
    output_names = [
        "Volatility",
        *[f"VaR {format_cl(get_VaR_level(cl))}%" for cl in confidence_levels],
        *[f"ES {format_cl(get_VaR_level(cl))}%" for cl in confidence_levels],
    ]
    bounds = {
        "Volatility": (-0.004, 0.002),
        "VaR 95%": (-0.01, 0.01),
        "VaR 97.5%": (-0.01, 0.01),
        "VaR 99%": (-0.01, 0.01),
        "ES 95%": (-0.004, 0.012),
        "ES 97.5%": (-0.01, 0.015),
        "ES 99%": (-0.01, 0.018),
    }
    top_n = 10  # Number of features to show in summary plot
    for i, metric in enumerate(output_names):
        shap.summary_plot(
            shap_values[:, :, i],
            Xtf,
            show=False,
            feature_names=feature_names,
            max_display=top_n,
        )
        model_base_name = "LSTM-MDN" if "lstm" in MODEL_NAME else "Transformer-MDN"
        version_expl = "-".join(
            (["RV"] if "rv" in VERSION else []) + (["IV"] if "iv" in VERSION else [])
        )
        model_display_name = f"{model_base_name}-{version_expl}"
        #plt.title(
        #    f"SHAP analysis of {model_display_name} {metric} estimates",
        #    fontsize=18,
        #    ha="center",
        #)
        plt.xlim(*bounds[metric])
        ax = plt.gca()
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)  # X-axis label
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        fig = plt.gcf()
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(
            f"SHAP analysis of {model_display_name} {metric} estimates",
            fontsize=18,
            ha="center",
            x=0.5  # Center of the figure
        )
        axes = fig.axes
        colorbar_ax = axes[1]  # Second axis is the color bar
        colorbar_ax.set_ylabel("Feature value", fontsize=14)
        colorbar_ax.tick_params(labelsize=12)
        plt.savefig(
            f"results/xai/shap_{metric}_{MODEL_NAME}_{ANALYSIS_START_DATE}.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        if is_notebook():
            plt.show()

    # %%
    # Calculate feature interactions
    # Compute mean absolute SHAP value across all samples and outputs
    metric = "ES 97.5%"
    output_i = output_names.index(metric)
    mean_abs_shap = np.abs(shap_values[:, :, output_i]).mean(axis=0)

    # Get indices of top 20 features
    top_k = 10
    top_idx = np.argsort(mean_abs_shap)[-top_k:]

    # Subset SHAP and input arrays
    shap_subset = shap_values[:, top_idx, output_i]
    Xtf_subset = Xtf[:, top_idx]
    feature_names_subset = [feature_names[i] for i in top_idx]

    # Calculate partial interaction effects: to what degree can the value of one feature
    # explain the SHAP value of another feature, adjusted for correlations.
    pairwise_interaction = np.zeros((top_k, top_k))
    pairwise_coefficients = np.zeros((top_k, top_k))

    for i in range(top_k):
        y = shap_subset[:, i]  # SHAP for feature i
        Xi = Xtf_subset[:, i].reshape(-1, 1)
        # Add a square term for the feature to capture non-linear effects
        # base_X = np.column_stack((Xi, Xi**2))
        base_X = Xi
        # fit base: SHAP_i ~ X_i
        base = LinearRegression().fit(base_X, y)
        R2_base = base.score(base_X, y)

        for j in range(top_k):
            if j == i:
                pairwise_interaction[i, j] = 0.0
                continue

            Xj = Xtf_subset[:, j].reshape(-1, 1)
            # full_X = np.column_stack((Xi, Xi**2, Xj, Xj**2, Xj * Xi))
            full_X = np.column_stack((Xi, Xj))
            full = LinearRegression().fit(full_X, y)
            # print(
            #     dict(
            #         zip(
            #             ["Xi", "Xi^2", "Xj", "Xj^2", "Xi*Xj"],
            #             [round(float(n), 5) for n in full.coef_],
            #         )
            #     )
            # )
            pairwise_coefficients[i, j] = full.coef_[1]

            R2_full = full.score(full_X, y)

            pairwise_interaction[i, j] = max(0, R2_full - R2_base)

    # %%
    # Plot the pairwise interaction heatmap
    # Define a fixed color range for better comparison
    vmin, vmax = 0.0, 0.15

    plt.figure(figsize=(12, 10))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_heatmap",
        [
            colors["primary"],
            colors["muted"],
            colors["accent"],
            colors["secondary"],
            colors["highlight"],
        ],
        N=256,
    )
    im = plt.imshow(pairwise_interaction, cmap=cmap, norm=norm, aspect="auto")

    # ticks
    plt.xticks(np.arange(top_k), feature_names_subset, rotation=90)
    plt.yticks(np.arange(top_k), feature_names_subset)

    # axis labels
    plt.xlabel("Input feature $j$ (values)", fontsize=14)
    plt.ylabel("SHAP value for feature $i$ (importances)", fontsize=14)

    # colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Partial $R^2$ of $X_j$ explaining $SHAP_i$", fontsize=12)

    # disable grid
    plt.grid(False)

    # in each cell, add a + or - sign if the coefficient is positive or negative
    for i in range(top_k):
        for j in range(top_k):
            if pairwise_coefficients[i, j] > 0:
                sign = "+"
            elif pairwise_coefficients[i, j] < 0:
                sign = "-"
            else:
                sign = ""
            plt.text(
                j,
                i,
                sign,
                ha="center",
                va="center",
                color="white" if abs(pairwise_coefficients[i, j]) > 0.1 else "black",
            )

    # title
    plt.title(f"SHAP Interaction Heatmap (Top {top_k} Features)", fontsize=16)

    plt.tight_layout()
    plt.savefig(
        f"results/xai/pairwise_r2_shap_{metric}_{MODEL_NAME}_{ANALYSIS_START_DATE}.pdf",
        dpi=300,
    )

    # %%
    # Plot a similar heatmap for the coefficients
    plt.figure(figsize=(12, 10))
    im = plt.imshow(pairwise_coefficients, cmap="viridis", aspect="auto")
    # ticks
    plt.xticks(np.arange(top_k), feature_names_subset, rotation=90)
    plt.yticks(np.arange(top_k), feature_names_subset)
    # axis labels
    plt.xlabel("Input feature j (values)", fontsize=14)
    plt.ylabel("SHAP value for feature i (importances)", fontsize=14)
    # colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Coefficient of $X_j$ in $SHAP_i$", fontsize=12)
    # title
    plt.title("SHAP Coefficient Heatmap (Top 20 Features)", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f"results/xai/pairwise_coeff_shap_{metric}_{MODEL_NAME}_{ANALYSIS_START_DATE}.pdf",
        dpi=300,
    )


# %%
