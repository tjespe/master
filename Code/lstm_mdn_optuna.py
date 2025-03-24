# %%
import os
import gc
import sys
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
from optuna.study import StudyDirection

from shared.adequacy import christoffersen_test
from shared.conf_levels import format_cl
from shared.mdn import (
    calculate_intervals_vectorized,
    get_mdn_bias_initializer,
    get_mdn_kernel_initializer,
    parse_mdn_output,
)
from shared.loss import mdn_nll_tf
from shared.processing import get_lstm_train_test_new
from settings import SUFFIX, LOOKBACK_DAYS


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


class OptunaEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        # Save as user attributes so you can inspect them later:
        self.trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)
        self.trial.set_user_attr(f"val_loss_epoch_{epoch}", val_loss)

        # This is not possible with multi objectives
        # self.trial.report(val_loss, step=epoch)

        # We can choose to prune here if performance is bad:
        # if self.trial.should_prune():
        #     raise optuna.TrialPruned()


# -------------------------------------------------------------------------
# Setup Optuna objective
# We will run multiple trials, each trying out different hyperparameters.
# We keep your defaults as the "preferred" or "default" values in the search.
# -------------------------------------------------------------------------
def objective(trial: optuna.Trial):
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

    MULTIPLY_MARKET_FEATURES_BY_BETA = False
    PI_PENALTY = False
    INCLUDE_MARKET_FEATURES = (
        False  # trial.suggest_categorical("INCLUDE_MARKET_FEATURES", [True, False])
    )
    INCLUDE_RETURNS = (
        False  # trial.suggest_categorical("INCLUDE_RETURNS", [True, False])
    )
    INCLUDE_FNG = False  # trial.suggest_categorical("INCLUDE_FNG", [True, False])
    INCLUDE_INDUSTRY = (
        False  # trial.suggest_categorical("INCLUDE_INDUSTRY", [True, False])
    )
    INCLUDE_GARCH = False  # trial.suggest_categorical("INCLUDE_GARCH", [True, False])
    INCLUDE_BETA = False  # trial.suggest_categorical("INCLUDE_BETA", [True, False])
    INCLUDE_OTHERS = False  # trial.suggest_categorical("INCLUDE_OTHERS", [True, False])
    INCLUDE_TICKERS = trial.suggest_categorical("INCLUDE_TICKERS", [True, False])
    INCLUDE_10_DAY_IVOL = trial.suggest_categorical(
        "INCLUDE_10_DAY_IVOL", [True, False]
    )
    INCLUDE_30_DAY_IVOL = trial.suggest_categorical(
        "INCLUDE_30_DAY_IVOL", [True, False]
    )
    INCLUDE_1MIN_RV = trial.suggest_categorical("INCLUDE_1MIN_RVOL", [True, False])
    INCLUDE_5MIN_RV = trial.suggest_categorical("INCLUDE_5MIN_RVOL", [True, False])

    if (
        not INCLUDE_10_DAY_IVOL
        and not INCLUDE_30_DAY_IVOL
        and not INCLUDE_1MIN_RV
        and not INCLUDE_5MIN_RV
    ):
        raise optuna.exceptions.TrialPruned()

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    print("Loading preprocessed data...")

    # Get data
    try:
        data = get_lstm_train_test_new(
            multiply_by_beta=MULTIPLY_MARKET_FEATURES_BY_BETA,
            include_returns=INCLUDE_RETURNS,
            include_spx_data=INCLUDE_MARKET_FEATURES,
            include_others=INCLUDE_OTHERS,
            include_beta=INCLUDE_BETA,
            include_fng=INCLUDE_FNG,
            include_garch=INCLUDE_GARCH,
            include_industry=INCLUDE_INDUSTRY,
            include_1min_rv=INCLUDE_1MIN_RV,
            include_5min_rv=INCLUDE_5MIN_RV,
            include_ivol_cols=(["10 Day Call IVOL"] if INCLUDE_10_DAY_IVOL else [])
            + (["Historical Call IVOL"] if INCLUDE_30_DAY_IVOL else []),
        )
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    gc.collect()

    print(f"X_train.shape: {data.train.X.shape}, y_train.shape: {data.train.y.shape}")
    print(
        f"X_val.shape: {data.validation.X.shape}, y_val.shape: {data.validation.y.shape}"
    )
    if INCLUDE_TICKERS:
        print(f"Ticker IDs dimension: {data.ticker_ids_dim}")

    # Build the model with these hyperparams
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
    optuna_callback = OptunaEpochCallback(trial)

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
        epochs=50,
        batch_size=batch_size,
        callbacks=[early_stop, optuna_callback],
        verbose=1,
    )

    # Store how many epochs we actually trained for as trial attributes
    trial.set_user_attr("epochs_trained", len(history.history["loss"]))

    # Evaluate
    val_loss = min(history.history["val_loss"])  # best validation loss from this run

    # Calculate confidence intervals and PICPs
    y_val_pred = lstm_model.predict(
        [data.validation.X, data.validation_ticker_ids]
        if INCLUDE_TICKERS
        else data.validation.X
    )
    pis, mus, sigmas = parse_mdn_output(y_val_pred, n_mixtures)
    confidence_levels = [0.67, 0.90, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999]
    intervals = calculate_intervals_vectorized(pis, mus, sigmas, confidence_levels)

    picps = {
        cl: np.mean(
            np.logical_and(
                data.validation.y > intervals[:, i, 0],
                data.validation.y < intervals[:, i, 1],
            )
        )
        for i, cl in enumerate(confidence_levels)
    }
    picp_miss = {cl: picp - cl for cl, picp in picps.items()}
    total_picp_miss = np.sum(np.abs(list(picp_miss.values())))

    chr_results = []
    chr_results_indices = []
    for i, cl in enumerate(confidence_levels):
        df = pd.DataFrame(
            index=[data.validation.tickers, data.validation.dates],
            columns=["y_true", "interval_low", "interval_high"],
        )
        df.index.names = ["Symbol", "Date"]
        df["y_true"] = data.validation.y
        df["interval_low"] = intervals[:, i, 0]
        df["interval_high"] = intervals[:, i, 1]
        df["exceeded"] = (df["y_true"] < df["interval_low"]) | (
            df["y_true"] > df["interval_high"]
        )

        # Group by symbol in original order.
        df.sort_index(inplace=True)
        for symbol, group in df.groupby(
            df.index.get_level_values("Symbol"), sort=False
        ):
            exceedances = (group["exceeded"].astype(bool)).values
            result = christoffersen_test(exceedances, 1 - cl)
            chr_results.append(result)
            chr_results_indices.append((symbol, cl))

    chr_results = pd.DataFrame(chr_results, index=chr_results_indices)
    print(chr_results)
    total_fails = np.nansum(
        (chr_results["p_value_uc"] < 0.05).astype(int)
        + (chr_results["p_value_ind"] < 0.05).astype(int)
        + (chr_results["p_value_cc"] < 0.05).astype(int)
    )
    total_passes = np.nansum(
        (chr_results["p_value_uc"] > 0.05).astype(int)
        + (chr_results["p_value_ind"] > 0.05).astype(int)
        + (chr_results["p_value_cc"] > 0.05).astype(int)
    )

    # Clean up GPU memory
    del lstm_model
    gc.collect()
    tf.keras.backend.clear_session()

    return val_loss, total_picp_miss, total_passes, total_fails


def git_commit_callback(study: optuna.Study, trial: optuna.Trial):
    print(
        f"Trial {trial.number} finished with value: {trial.values}. Committing DB to git."
    )
    try:
        # Update the local repo
        subprocess.run(["git", "pull", "--no-edit"], check=True)
        # Stage the sqlite DB file
        subprocess.run(["git", "add", "optuna"], check=True)
        # Build a detailed commit message
        commit_header = f"Trial {trial.number} - Updated study DB"
        commit_body = (
            f"Trial {trial.number} finished with objective values: {trial.values}\n"
            f"Hyperparameters: {trial.params}\n"
        )
        # Commit with the constructed message
        subprocess.run(
            ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
        )
        # Push the commit
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")


# -------------------------------------------------------------------------
# Running the Optuna study
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Create or load a study
    study_name = "lstm_mdn_hyperparam_and_feature_search_christoffersen_per_asset"
    storage = "sqlite:///optuna/optuna.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=[
            StudyDirection.MINIMIZE,
            StudyDirection.MINIMIZE,
            StudyDirection.MAXIMIZE,
            StudyDirection.MINIMIZE,
        ],
    )

    # Optimize
    try:
        n_trials = int(sys.argv[1])
    except:
        n_trials = 1000
    study.optimize(objective, n_trials=n_trials, callbacks=[git_commit_callback])

    # Print best result
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value (val_loss): {best_trial.value}")
    print("  Params:")
    for key, val in best_trial.params.items():
        print(f"    {key}: {val}")

    # Optionally save all trials to a CSV for your records
    df = study.trials_dataframe()
    df.to_csv("optuna/lstm_mdn_study_results.csv", index=False)
    print("\nAll trial results have been saved to 'lstm_mdn_study_results.csv'.")
