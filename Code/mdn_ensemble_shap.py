# %%
#!/usr/bin/env python3
"""
Load an MDN ensemble model, compute volatility (variance), Value-at-Risk (VaR),
and Expected Shortfall (ES), then run SHAP analysis on each quantity.
"""


# ---------------------------
# Model & Data Parameters
# ---------------------------
from lstm_mdn_ensemble import build_lstm_mdn
from transformer_mdn_ensemble import build_transformer_mdn
from settings import SUFFIX
from shared.expanding_window import find_latest_model

VERSION = "ivol-final-rolling"
# Which model to load, can be set to a string directly or the output of find_latest_model
MODEL_FNAME = find_latest_model(
    "models/rolling", f"lstm_mdn_ensemble{SUFFIX}_v{VERSION}_test_*.h5"
)
BUILD_FN = build_lstm_mdn  # or build_transformer_mdn
N_ENSEMBLE_MEMBERS = 10
N_MIXTURES = 10
HIDDEN_UNITS = 60
DROPOUT = 0.0
NUM_HIDDEN_LAYERS = 0
EMBEDDING_DIMENSIONS = None

# Data flags (match your training settings)
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
    include_1min_rv=False,
    include_5min_rv=False,
    include_ivol_cols=["10 Day Call IVOL", "Historical Call IVOL"],
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
    get_mdn_kernel_initializer,
    get_mdn_bias_initializer,
)
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    VALIDATION_TEST_SPLIT,
)
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LSTM,
    Embedding,
    Flatten,
    Concatenate,
    RepeatVector,
)
from tensorflow.keras.regularizers import l2


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
    first_test_date = pd.to_datetime(VALIDATION_TEST_SPLIT)
    train = data.get_training_set_for_date(first_test_date)
    # for illustration, use a 30-day test window
    test = data.get_test_set_for_date(
        first_test_date, first_test_date + pd.DateOffset(days=30)
    )

    X_train = train.X
    X_test = test.X
    num_features = X_train.shape[2]

    # %%
    # Load latest ensemble
    submodels = [BUILD_FN(num_features) for _ in range(N_ENSEMBLE_MEMBERS)]
    ensemble_model = MDNEnsemble(submodels, N_MIXTURES)
    ensemble_model.load_weights(MODEL_FNAME)

    # %%
    # Flatten utility
    def flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    # %%
    # Background and test samples for SHAP (adjust sizes as needed)
    Xb = flatten(X_train[:50])
    Xt = X_test[:20]
    Xtf = flatten(Xt)

    # %%
    # Prediction functions
    def predict_variance(x_flat: np.ndarray) -> np.ndarray:
        X = x_flat.reshape(-1, LOOKBACK_DAYS, num_features)
        raw, _ = ensemble_model.predict(X)
        pi, mu, sigma = parse_mdn_output(raw, N_MIXTURES * N_ENSEMBLE_MEMBERS)
        var = np.sum(pi * (sigma**2 + mu**2), axis=1) - np.sum(pi * mu, axis=1) ** 2
        return var

    def predict_var95(x_flat: np.ndarray) -> np.ndarray:
        X = x_flat.reshape(-1, LOOKBACK_DAYS, num_features)
        raw, _ = ensemble_model.predict(X)
        pi, mu, sigma = parse_mdn_output(raw, N_MIXTURES * N_ENSEMBLE_MEMBERS)
        intervals = calculate_intervals_vectorized(pi, mu, sigma, [0.95])
        return intervals[:, 0, 0]

    def predict_es95(x_flat: np.ndarray) -> np.ndarray:
        X = x_flat.reshape(-1, LOOKBACK_DAYS, num_features)
        raw, _ = ensemble_model.predict(X)
        pi, mu, sigma = parse_mdn_output(raw, N_MIXTURES * N_ENSEMBLE_MEMBERS)
        intervals = calculate_intervals_vectorized(pi, mu, sigma, [0.95])
        var95 = intervals[:, 0, 0]
        es = calculate_es_for_quantile(pi, mu, sigma, var95)
        return es

    # %%
    # SHAP analysis
    expl_var = shap.KernelExplainer(predict_variance, Xb)
    shap_values_var = expl_var.shap_values(Xtf)
    shap.summary_plot(shap_values_var, Xtf, show=True)

    expl_vaR = shap.KernelExplainer(predict_var95, Xb)
    shap_values_vaR = expl_vaR.shap_values(Xtf)
    shap.summary_plot(shap_values_vaR, Xtf, show=True)

    expl_es = shap.KernelExplainer(predict_es95, Xb)
    shap_values_es = expl_es.shap_values(Xtf)
    shap.summary_plot(shap_values_es, Xtf, show=True)

# %%
