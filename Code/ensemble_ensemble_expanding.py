# %%
# Expanding‐Window Ensemble Predictions
import subprocess
import sys
from datetime import date
from lstm_mdn_ensemble_expanding import build_lstm_mdn
from transformer_mdn_ensemble import add_day_indices, build_transformer_mdn
from shared.ensemble import MDNEnsemble
from shared.conf_levels import format_cl
from settings import LOOKBACK_DAYS, SUFFIX, TEST_SET, VALIDATION_TEST_SPLIT
import numpy as np
import pandas as pd
import warnings
import os
import gc
import tensorflow as tf

from shared.processing import get_lstm_train_test_new
from shared.mdn import (
    calculate_es_for_quantile,
    calculate_intervals_vectorized,
    calculate_prob_above_zero_vectorized,
    get_mdn_bias_initializer,
    get_mdn_kernel_initializer,
    parse_mdn_output,
    mdn_mean_and_var,
)
from shared.loss import (
    ece_mdn,
    mdn_crps_tf,
    mdn_nll_numpy,
    mdn_nll_tf,
)

warnings.filterwarnings("ignore")

# Ensemble parameters
VERSION = sys.argv[1] if len(sys.argv) > 1 else "rv-iv"
MODEL_NAME = f"mdn_ensemble_{VERSION}_{TEST_SET}_expanding"
WINDOW_DAYS = 30

# Mixture settings (each base-ensemble has these)
SUBMODEL_MIXTURES = 10
SUBMODEL_MEMBERS = 10
SUBMODEL_TOTAL_MIXTURES = SUBMODEL_MIXTURES * SUBMODEL_MEMBERS


# %%
# Helper functions for making untrained models (since the stored ones only have weights)
def make_transformer_ensemble(n_features):
    return MDNEnsemble(
        [
            build_transformer_mdn(
                num_features=n_features,
                ticker_ids_dim=None,
            )
            for _ in range(SUBMODEL_MEMBERS)
        ],
        SUBMODEL_MIXTURES,
    )


def make_lstm_ensemble(n_features):
    return MDNEnsemble(
        [
            build_lstm_mdn(
                num_features=n_features,
                ticker_ids_dim=None,
            )
            for _ in range(SUBMODEL_MEMBERS)
        ],
        SUBMODEL_MIXTURES,
    )


# %%
# Define contents of each version
VERSIONS = {
    "rv-iv": [
        (
            make_transformer_ensemble,
            f"transformer_mdn_ensemble_rvol-ivol_{TEST_SET}_expanding",
        ),
        (
            make_lstm_ensemble,
            f"lstm_mdn_ensemble{SUFFIX}_vrv-and-ivol-final-rolling_{TEST_SET}",
        ),
    ],
    "rv": [
        (
            make_transformer_ensemble,
            f"transformer_mdn_ensemble_rvol_{TEST_SET}_expanding",
        ),
        (make_lstm_ensemble, f"lstm_mdn_ensemble{SUFFIX}_vrv-final-rolling_{TEST_SET}"),
    ],
    "iv": [
        (
            make_transformer_ensemble,
            f"transformer_mdn_ensemble_ivol_{TEST_SET}_expanding",
        ),
        (
            make_lstm_ensemble,
            f"lstm_mdn_ensemble{SUFFIX}_vivol-final-rolling_{TEST_SET}",
        ),
    ],
}
if VERSION not in VERSIONS:
    raise ValueError(
        f"Invalid version: {VERSION}. You must choose one of {VERSIONS.keys()} or add a new entry to VERSIONS."
    )
BASE_MODELS = VERSIONS[VERSION]
N_BASE_MODELS = len(BASE_MODELS)


# %%
if __name__ == "__main__":
    print(f"Expanding MDN Ensemble v{VERSION}")

    # %%
    # Load data once
    data = get_lstm_train_test_new(
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
            ["10 Day Call IVOL", "Historical Call IVOL"] if "iv" in VERSION else []
        ),
    )

    # %%
    # Sliding window start
    first_date = pd.to_datetime(VALIDATION_TEST_SPLIT)
    y_pred_list = []
    epistemic_list = []
    true_y_list = []
    date_list = []
    symbol_list = []

    # Loop through expanding windows
    while True:
        end_date = first_date + pd.DateOffset(days=WINDOW_DAYS)
        test = data.get_test_set_for_date(first_date, end_date)
        if test.y.shape[0] == 0:
            break
        print(f"Predicting window {first_date.date()} to {end_date.date()}")

        # Load each base ensemble trained on this window
        members = []
        for build_fn, base_fname in BASE_MODELS:
            fname = f"models/rolling/{base_fname}_{first_date.date().isoformat()}.h5"
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Missing model file: {fname}")
            print(f"Loading {fname}")
            member = build_fn(test.X.shape[-1])
            member.load_weights(fname)
            members.append(member)

        # Top-level ensemble of the two base ensembles
        ensemble_model = MDNEnsemble(members, SUBMODEL_TOTAL_MIXTURES, name=MODEL_NAME)

        # Predict
        X_test = test.X
        y_pred_mdn, epistemic_var = ensemble_model.predict(X_test)
        pi_pred, mu_pred, sigma_pred = parse_mdn_output(
            y_pred_mdn, SUBMODEL_TOTAL_MIXTURES * N_BASE_MODELS
        )

        # Collect
        y_pred_list.append(y_pred_mdn)
        epistemic_list.append(epistemic_var)
        true_y_list.extend(test.y.tolist())
        date_list.extend(test.dates.tolist())
        symbol_list.extend(test.tickers.tolist())

        # Advance window
        first_date = end_date + pd.DateOffset(days=1)

    # %%
    # Concatenate
    y_pred_all = np.concatenate(y_pred_list, axis=0)
    epistemic_all = np.concatenate(epistemic_list, axis=0)

    # Decompose mixtures
    pi_all, mu_all, sigma_all = parse_mdn_output(
        y_pred_all, SUBMODEL_TOTAL_MIXTURES * N_BASE_MODELS
    )

    # %%
    # Build DataFrame
    df = pd.DataFrame({"Date": date_list, "Symbol": symbol_list})

    # %%
    # Mixture moments
    mixture_mean_sp, total_var = mdn_mean_and_var(pi_all, mu_all, sigma_all)
    aleatoric_var = total_var - epistemic_all
    mixture_mean_sp = mixture_mean_sp.numpy()
    mixture_vol = np.sqrt(aleatoric_var.numpy())
    df["Mean_SP"] = mixture_mean_sp
    df["Vol_SP"] = mixture_vol

    # %%
    # Calculate NLL
    # True-y array
    true_y = np.array(true_y_list).astype(np.float32)
    # Metrics
    df["NLL"] = mdn_nll_numpy(SUBMODEL_TOTAL_MIXTURES * N_BASE_MODELS)(
        true_y, y_pred_all
    )

    # %%
    # Calculate CRPS
    # Free all memory before calculating CRPS because it is very resource intensive
    del ensemble_model
    del members
    tf.keras.backend.clear_session()
    gc.collect()
    # Calculate per symbol to avoid OOM errors
    for symbol in df["Symbol"].unique():
        symbol_mask = df["Symbol"] == symbol
        true_y_symbol = true_y[symbol_mask]
        y_pred_symbol = y_pred_all[symbol_mask]
        df.loc[symbol_mask, "CRPS"] = mdn_crps_tf(
            SUBMODEL_TOTAL_MIXTURES * N_BASE_MODELS
        )(true_y_symbol, y_pred_symbol).numpy()

    # %%
    # Calculate ECE
    df["ECE"] = ece_mdn(SUBMODEL_TOTAL_MIXTURES * N_BASE_MODELS, true_y, y_pred_all)

    # %%
    # Confidence intervals
    cls = [0.67, 0.90, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999]
    intervals = calculate_intervals_vectorized(pi_all, mu_all, sigma_all, cls)
    for i, cl in enumerate(cls):
        df[f"LB_{format_cl(cl)}"] = intervals[:, i, 0]
        df[f"UB_{format_cl(cl)}"] = intervals[:, i, 1]

    # %%
    # Expected shortfall
    for i, cl in enumerate(cls):
        alpha = (1 - cl) / 2
        var_est = intervals[:, i, 0]
        df[f"ES_{format_cl(1-alpha)}"] = calculate_es_for_quantile(
            pi_all, mu_all, sigma_all, var_est
        )

    # %%
    # Probability of increase & epistemic var
    df["Prob_Increase"] = calculate_prob_above_zero_vectorized(
        pi_all, mu_all, sigma_all
    )
    df["EpistemicVarMean"] = epistemic_all

    # %%
    # Save
    out = f"predictions/{MODEL_NAME}.csv"
    df.set_index(["Date", "Symbol"]).to_csv(out)
    print(f"Saved expanding predictions to {out}")

    # %%
    # Git commit
    try:
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(["git", "add", out], check=True)
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"Add expanding predictions for MDN Ensemble {VERSION}",
            ],
            check=True,
        )
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")
