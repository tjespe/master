# optuna_lstm_mdn.py

import os
import gc
import numpy as np
import pandas as pd
import warnings
import subprocess

import tensorflow as tf
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import optuna

from shared.conf_levels import format_cl
from shared.mdn import (
    get_mdn_bias_initializer,
    get_mdn_kernel_initializer,
)
from shared.loss import mdn_nll_tf
from shared.processing import get_lstm_train_test_new
from settings import SUFFIX, LOOKBACK_DAYS

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# Global constants and fixed settings
VERSION = "seq_tickers"

# Feature inclusion flags (fixed for this experiment)
MULTIPLY_MARKET_FEATURES_BY_BETA = False
PI_PENALTY = False
INCLUDE_MARKET_FEATURES = False
INCLUDE_RETURNS = False
INCLUDE_FNG = False
INCLUDE_INDUSTRY = False
INCLUDE_GARCH = False
INCLUDE_BETA = False
INCLUDE_OTHERS = False
INCLUDE_TICKERS = True

# -------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------
print("Loading preprocessed data...")
data = get_lstm_train_test_new(
    multiply_by_beta=MULTIPLY_MARKET_FEATURES_BY_BETA,
    include_returns=INCLUDE_RETURNS,
    include_spx_data=INCLUDE_MARKET_FEATURES,
    include_others=INCLUDE_OTHERS,
    include_beta=INCLUDE_BETA,
    include_fng=INCLUDE_FNG,
    include_garch=INCLUDE_GARCH,
    include_industry=INCLUDE_INDUSTRY,
)
gc.collect()

print(f"X_train.shape: {data.train.X.shape}, y_train.shape: {data.train.y.shape}")
print(f"Validation set shape: {data.validation.X.shape}, {data.validation.y.shape}")
if INCLUDE_TICKERS:
    print(f"Ticker IDs dimension: {data.ticker_ids_dim}")


# -------------------------------------------------------------------------
# Model Building Function
# -------------------------------------------------------------------------
def build_lstm_mdn(
    num_features: int,
    ticker_ids_dim: int,
    hidden_units: int,
    dropout: float,
    num_hidden_layers: int,
    n_mixtures: int,
    l2_reg: float,
    ticker_d: int,
):
    """
    Creates an LSTM-based encoder for sequences of shape (LOOKBACK_DAYS, num_features)
    and outputs 3 * n_mixtures for the MDN.
    """
    # Sequence input
    seq_input = Input(shape=(LOOKBACK_DAYS, num_features), name="seq_input")
    inputs = [seq_input]

    # If tickers are included, add ticker input and embedding
    if ticker_ids_dim is not None:
        ticker_input = Input(shape=(), dtype="int32", name="ticker_input")
        inputs.append(ticker_input)

        ticker_embed = Embedding(
            input_dim=ticker_ids_dim,
            output_dim=ticker_d,
            name="ticker_embedding",
        )(ticker_input)
        ticker_embed = Flatten()(ticker_embed)
        ticker_embed = RepeatVector(LOOKBACK_DAYS)(ticker_embed)

        x = Concatenate(axis=-1, name="concat_seq_ticker")([seq_input, ticker_embed])
    else:
        x = seq_input

    # LSTM layer
    x = LSTM(
        units=hidden_units,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="lstm_layer",
    )(x)

    if dropout > 0:
        x = Dropout(dropout, name="dropout_lstm")(x)

    # Additional Dense layers
    for i in range(num_hidden_layers):
        x = Dense(
            units=hidden_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name=f"dense_layer_{i}",
        )(x)
        if dropout > 0:
            x = Dropout(dropout, name=f"dropout_layer_{i}")(x)

    # MDN output layer: 3 * n_mixtures (for [logits_pi, mu, log_sigma])
    mdn_kernel_init = get_mdn_kernel_initializer(n_mixtures)
    mdn_bias_init = get_mdn_bias_initializer(n_mixtures)
    mdn_output = Dense(
        3 * n_mixtures,
        activation=None,
        kernel_initializer=mdn_kernel_init,
        bias_initializer=mdn_bias_init,
        name="mdn_output",
    )(x)

    model = Model(inputs=inputs, outputs=mdn_output)
    return model


# -------------------------------------------------------------------------
# Objective Function for Hyperparameter Optimization
# -------------------------------------------------------------------------
def objective(trial):
    # Suggest hyperparameters to optimize
    hidden_units = trial.suggest_int("HIDDEN_UNITS", 1, 80, step=4)
    num_hidden_layers = trial.suggest_int("NUM_HIDDEN_LAYERS", 0, 4)
    n_mixtures = trial.suggest_int("N_MIXTURES", 1, 30, step=5)
    dropout = trial.suggest_float("DROPOUT", 0, 0.7, step=0.1)
    learning_rate = trial.suggest_float("LEARNING_RATE", 1e-5, 1e-3, log=True)
    l2_reg = trial.suggest_float("L2_REG", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_int("BATCH_SIZE", 16, 128, log=True)
    ticker_embedding_dimensions = trial.suggest_int(
        "TICKER_EMBEDDING_DIM", 1, 30, step=1, log=True
    )
    weight_decay = trial.suggest_float("WEIGHT_DECAY", 1e-6, 0.1, log=True)

    # Build and compile the LSTM MDN model with these hyperparameters
    lstm_model = build_lstm_mdn(
        num_features=data.train.X.shape[2],
        ticker_ids_dim=(data.ticker_ids_dim if INCLUDE_TICKERS else None),
        hidden_units=hidden_units,
        dropout=dropout,
        num_hidden_layers=num_hidden_layers,
        n_mixtures=n_mixtures,
        l2_reg=l2_reg,
        ticker_d=ticker_embedding_dimensions,
    )

    optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    loss_fn = mdn_nll_tf(n_mixtures, PI_PENALTY)
    lstm_model.compile(optimizer=optimizer, loss=loss_fn)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )

    history = lstm_model.fit(
        [data.train.X, data.train_ticker_ids] if INCLUDE_TICKERS else data.train.X,
        data.train.y,
        validation_data=(
            (
                [data.validation.X, data.validation_ticker_ids]
                if INCLUDE_TICKERS
                else data.validation.X
            ),
            data.validation.y,
        ),
        epochs=30,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    val_loss = min(history.history["val_loss"])

    # Cleanup
    del lstm_model
    gc.collect()
    tf.keras.backend.clear_session()

    return val_loss


# -------------------------------------------------------------------------
# Optional Git Commit Callback (commits study DB after each trial)
# -------------------------------------------------------------------------
def git_commit_callback(study: optuna.Study, trial: optuna.Trial):
    print(
        f"Trial {trial.number} finished with value: {trial.value}. Committing study DB to git."
    )
    try:
        subprocess.run(["git", "pull", "--no-edit"], check=True)
        subprocess.run(["git", "add", "lstm_mdn_study.db"], check=True)
        commit_header = f"Trial {trial.number} - LSTM MDN study DB update"
        commit_body = (
            f"Trial {trial.number} finished with value: {trial.value}\n"
            f"Hyperparameters: {trial.params}\n"
            f"Study Best Value: {study.best_value}\n"
            f"Study Best Params: {study.best_trial.params}\n"
        )
        subprocess.run(
            ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
        )
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")


# -------------------------------------------------------------------------
# Run the Study
# -------------------------------------------------------------------------
if __name__ == "__main__":
    study_name = "lstm_mdn_hyperparam_search"
    storage = "sqlite:///optuna/optuna.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
    )

    n_trials = 1000  # Set the number of trials to run
    study.optimize(objective, n_trials=n_trials, callbacks=[git_commit_callback])

    # Display the best trial
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value (val_loss): {best_trial.value}")
    print("  Params:")
    for key, val in best_trial.params.items():
        print(f"    {key}: {val}")

    # Save trial results to CSV for future reference
    df = study.trials_dataframe()
    df.to_csv("lstm_study_results.csv", index=False)
    print("\nAll trial results have been saved to 'lstm_study_results.csv'.")
