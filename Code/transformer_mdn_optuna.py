# %%
import os
import gc
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys

# For type hints
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    Embedding,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    Concatenate,
    RepeatVector,
    Lambda,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from shared.adequacy import christoffersen_test
from settings import LOOKBACK_DAYS
import optuna
from optuna.study import StudyDirection

# -------------------------------------------------------------------------
# Imports from your shared modules (these should match your original project)
# -------------------------------------------------------------------------
from shared.conf_levels import format_cl
from shared.mdn import (
    calculate_es_for_quantile,
    calculate_intervals_vectorized,
    calculate_prob_above_zero_vectorized,
    get_mdn_bias_initializer,
    get_mdn_kernel_initializer,
    parse_mdn_output,
    predict_with_mc_dropout_mdn,
)
from shared.loss import (
    mdn_crps_tf,
    mdn_nll_numpy,
    mean_mdn_loss_numpy,
    mean_mdn_crps_tf,
    mdn_nll_tf,
)
from shared.crps import crps_mdn_numpy
from shared.processing import get_lstm_train_test_new

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# Original "defaults" or "preferred" values
# We’ll use these inside Optuna as the default hyperparameters
# -------------------------------------------------------------------------
DEFAULTS = {
    "D_MODEL": 32,
    "HIDDEN_UNITS_FF": 32 * 4,  # 128
    "N_MIXTURES": 8,
    "DROPOUT": 0.5,
    "L2_REGULARIZATION": 1e-5,
    "NUM_ENCODERS": 2,
    "NUM_HEADS": 4,
    "D_TICKER_EMBEDDING": 4,
    "PATIENCE": 20,
    "REDUCE_LR_PATIENCE": 5,
    "LEARNING_RATE": 5e-5,
    "WEIGHT_DECAY": 1e-2,
    "EPOCHS": 20,
    "BATCH_SIZE": 32,
}


# -------------------------------------------------------------------------
# Helper layers / methods (same as your code)
# -------------------------------------------------------------------------
def add_day_indices(tensor):
    """
    Adds day indices to sequential input for the Transformer.
    tensor shape: (batch, LOOKBACK_DAYS, ?)
    """
    batch_size = tf.shape(tensor)[0]
    indices = tf.range(LOOKBACK_DAYS, dtype=tf.float32)
    indices = tf.expand_dims(indices, axis=0)
    indices = tf.tile(indices, [batch_size, 1])
    indices = tf.expand_dims(indices, axis=-1)
    return tf.concat([tensor, indices], axis=-1)


def transformer_encoder(inputs, d_model, hidden_units_ff, num_heads, dropout, l2_reg):
    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(
        inputs, inputs
    )
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Feed-forward
    ffn = Dense(
        hidden_units_ff,
        activation="relu",
        kernel_regularizer=l2(l2_reg),
    )(out1)
    ffn = Dropout(dropout)(ffn)
    ffn = Dense(d_model, kernel_regularizer=l2(l2_reg))(ffn)
    ffn = Dropout(dropout)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2


def build_transformer_mdn(
    num_features: int,
    ticker_ids_dim: Optional[int],
    lookback_days: int,
    d_model: int,
    hidden_units_ff: int,
    n_mixtures: int,
    dropout: float,
    l2_reg: float,
    num_encoders: int,
    num_heads: int,
    d_ticker_embedding: int,
):
    """
    Creates a Transformer-based encoder for sequences of shape:
      (lookback_days, num_features)
    Then outputs 3*n_mixtures for the MDN (univariate).
    """
    # Input layers
    feature_inputs = Input(shape=(lookback_days, num_features), name="feature_inputs")

    if ticker_ids_dim is not None:
        ticker_inputs = Input(shape=(), dtype=tf.int32, name="ticker_inputs")
        # Ticker embedding
        ticker_embedding = Embedding(
            input_dim=ticker_ids_dim,
            output_dim=d_ticker_embedding,
            name="ticker_embedding",
        )(ticker_inputs)
        ticker_embedding_broadcast = RepeatVector(lookback_days)(ticker_embedding)
        x = Concatenate(axis=-1)([feature_inputs, ticker_embedding_broadcast])
    else:
        x = feature_inputs

    # Append day indices automatically
    x = Lambda(add_day_indices)(x)

    # Project inputs to d_model
    x = Dense(d_model, activation=None)(x)

    # Stack multiple Transformer encoder blocks
    for _ in range(num_encoders):
        x = transformer_encoder(x, d_model, hidden_units_ff, num_heads, dropout, l2_reg)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)

    # MDN output layer (3 * n_mixtures => [logits_pi, mu, log_sigma])
    mdn_kernel_init = get_mdn_kernel_initializer(n_mixtures)
    mdn_bias_init = get_mdn_bias_initializer(n_mixtures)
    mdn_output = Dense(
        3 * n_mixtures,
        activation=None,
        name="mdn_output",
        kernel_initializer=mdn_kernel_init,
        bias_initializer=mdn_bias_init,
    )(x)

    inputs = (
        [feature_inputs, ticker_inputs]
        if ticker_ids_dim is not None
        else feature_inputs
    )
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
    # Instead of searching over all hyperparameters, pick the ones that matter most:
    d_model = trial.suggest_int("D_MODEL", 16, 64, step=8)
    n_mixtures = trial.suggest_int("N_MIXTURES", 1, 40, step=2)
    dropout = trial.suggest_float("DROPOUT", 0, 0.7, step=0.1)
    l2_reg = trial.suggest_float("L2_REGULARIZATION", 1e-7, 1e-3, log=True)
    num_encoders = trial.suggest_int("NUM_ENCODERS", 1, 4)
    num_heads = trial.suggest_int("NUM_HEADS", 2, 8, step=2)
    batch_size = trial.suggest_int("BATCH_SIZE", 8, 64, step=8)

    # The feed-forward size is typically 4 * d_model. We can keep or let it vary:
    hidden_units_ff = trial.suggest_int("HIDDEN_UNITS_FF", 16, 1000, step=32)

    MULTIPLY_MARKET_FEATURES_BY_BETA = False
    PI_PENALTY = False
    MU_PENALTY = False
    SIGMA_PENALTY = False
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
    if INCLUDE_TICKERS:
        d_ticker_embedding = trial.suggest_int(
            "TICKER_EMBEDDING_DIM", 1, 30, step=1, log=True
        )
    else:
        d_ticker_embedding = None
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
        raise ValueError("At least one of the IVOL or RV features must be included.")

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    print("Loading preprocessed data...")

    # Get data
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

    gc.collect()

    print(f"X_train.shape: {data.train.X.shape}, y_train.shape: {data.train.y.shape}")
    print(
        f"X_val.shape: {data.validation.X.shape}, y_val.shape: {data.validation.y.shape}"
    )
    if INCLUDE_TICKERS:
        print(f"Ticker IDs dimension: {data.ticker_ids_dim}")

    # Build the model with these hyperparams
    transformer_model = build_transformer_mdn(
        num_features=data.train.X.shape[2],
        ticker_ids_dim=(data.ticker_ids_dim if INCLUDE_TICKERS else None),
        lookback_days=LOOKBACK_DAYS,
        d_model=d_model,
        hidden_units_ff=hidden_units_ff,
        n_mixtures=n_mixtures,
        dropout=dropout,
        l2_reg=l2_reg,
        num_encoders=num_encoders,
        num_heads=num_heads,
        d_ticker_embedding=d_ticker_embedding,
    )

    # Compile
    # You could try searching for learning rate or penalty (PI_PENALTY, etc.) as well
    optimizer = Adam(
        learning_rate=DEFAULTS["LEARNING_RATE"],
        weight_decay=DEFAULTS["WEIGHT_DECAY"],
    )
    loss_fn = mdn_nll_tf(n_mixtures, PI_PENALTY)  # or mean_mdn_crps_tf, etc.
    transformer_model.compile(optimizer=optimizer, loss=loss_fn)

    # We do a short training for the sake of hyperparameter tuning
    early_stop = EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-7
    )
    # optuna_callback = OptunaEpochCallback(trial)

    # Fit the model
    history = transformer_model.fit(
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
        epochs=DEFAULTS["EPOCHS"],
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],  # , optuna_callback],
        verbose=1,
    )

    # Store how many epochs we actually trained for as trial attributes
    trial.set_user_attr("epochs_trained", len(history.history["loss"]))
    for epoch in range(len(history.history["loss"])):
        trial.set_user_attr(f"train_loss_epoch_{epoch}", history.history["loss"][epoch])
        trial.set_user_attr(
            f"val_loss_epoch_{epoch}", history.history["val_loss"][epoch]
        )

    # Evaluate
    val_loss = min(history.history["val_loss"])  # best validation loss from this run

    # Calculate confidence intervals and PICPs
    y_val_pred = transformer_model.predict(
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
    del transformer_model
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
    study_name = (
        "transformer_mdn_hyperparam_and_feature_search_christoffersen_per_asset"
    )
    try:
        storage = sys.argv[2]
    except:
        storage = "sqlite:///optuna/optuna.db"
    print(f"Using storage: {storage}")
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
    print(f"Running {n_trials} trials...")
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
    df.to_csv("optuna/transformer_mdn_study_results.csv", index=False)
    print("\nAll trial results have been saved to 'study_results.csv'.")
