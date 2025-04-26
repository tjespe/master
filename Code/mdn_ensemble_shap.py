# %%
#!/usr/bin/env python3
"""
Load an MDN ensemble model, compute volatility (variance), Value-at-Risk (VaR),
and Expected Shortfall (ES), then run SHAP analysis on each quantity.
"""

from matplotlib import pyplot as plt
from lstm_mdn_ensemble import build_lstm_mdn
from shared.conf_levels import format_cl, get_VaR_level
from shared.jupyter import is_notebook
from transformer_mdn_ensemble import build_transformer_mdn
from settings import SUFFIX
import shared.styling_guidelines_graphs

# %%
# Model loading parameters
VERSION = "rv-and-ivol-final-rolling"
# SHAP takes a long time to run, so we only look at a subset of the data
ANALYSIS_START_DATE = "2024-02-27"
# Model name (used for storing results)
MODEL_NAME = f"lstm_mdn_ensemble{SUFFIX}_v{VERSION}_test"
# Filename for weights. Important that the loaded model is not trained on data after
# the ANALYSIS_START_DATE.
MODEL_FNAME = f"models/rolling/{MODEL_NAME}_{ANALYSIS_START_DATE}.h5"
# Function for building the model
BUILD_FN = build_lstm_mdn  # or build_transformer_mdn

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
    include_ivol_cols=["10 Day Call IVOL", "Historical Call IVOL"] if "ivol" in VERSION else [],
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
    univariate_mixture_mean_and_var_approx,
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
    confidence_levels = [
        0.90,  # 95% VaR
        0.95,  # 97.5% VaR
        0.98,  # 99% VaR
    ]

    def predict(x_flat: np.ndarray) -> np.ndarray:
        X = x_flat.reshape(-1, LOOKBACK_DAYS, num_features)
        raw, _ = ensemble_model.predict(X)
        pi, mu, sigma = parse_mdn_output(raw, N_MIXTURES * N_ENSEMBLE_MEMBERS)
        mean, var = univariate_mixture_mean_and_var_approx(pi, mu, sigma)
        vol = np.sqrt(var)
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
    # Present results
    feature_names = [
        f"{feat} ({LOOKBACK_DAYS - lag} days ago)".replace("(1 days", "(1 day")
        for lag in range(LOOKBACK_DAYS)
        for feat in cols
    ]
    output_names = [
        "Volatility",
        *[f"VaR {format_cl(get_VaR_level(cl))}%" for cl in confidence_levels],
        *[f"ES {format_cl(get_VaR_level(cl))}%" for cl in confidence_levels],
    ]
    for i, name in enumerate(output_names):
        shap.summary_plot(
            shap_values[:, :, i],
            Xtf,
            show=False,
            feature_names=feature_names,
        )
        plt.title(name)
        plt.savefig(
            f"results/xai/shap_{name}_{MODEL_NAME}_{ANALYSIS_START_DATE}.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        if is_notebook():
            plt.show()

# %%
