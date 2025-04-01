# %%
# Define parameters (based on settings)
import subprocess
from typing import Optional
from shared.ensemble import MDNEnsemble, ParallelProgressCallback
from shared.conf_levels import format_cl
from settings import LOOKBACK_DAYS, SUFFIX, TEST_SET
import multiprocessing as mp

VERSION = "rv"
MODEL_NAME = f"transformer_mdn_ensemble_{VERSION}_{TEST_SET}"

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
INCLUDE_10_DAY_IVOL = False
INCLUDE_30_DAY_IVOL = False
INCLUDE_1MIN_RV = True
INCLUDE_5MIN_RV = True
INCLUDE_TICKERS = False

# %%
# Model settings
D_MODEL = 32
HIDDEN_UNITS_FF = 300
N_MIXTURES = 10
DROPOUT = 0
L2_REGULARIZATION = 1e-6
NUM_ENCODERS = 1
NUM_HEADS = 2
D_TICKER_EMBEDDING = None

# %%
# Settings for training
WEIGHT_DECAY = 1e-2
LEARNING_RATE = 5e-5
BATCH_SIZE = 40
REDUCE_LR_PATIENCE = 3  # Patience before halving learning rate
PATIENCE = 10  # Early stopping patience
N_ENSEMBLE_MEMBERS = 10
PARALLELLIZE = True

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

warnings.filterwarnings("ignore")


# %%
def transformer_encoder(inputs):
    """
    A single block of Transformer encoder:
      1) MHA + residual + LayerNorm
      2) FFN + residual + LayerNorm
    """
    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL)(
        inputs, inputs
    )
    attn_output = Dropout(DROPOUT)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Feed-forward
    ffn = Dense(
        HIDDEN_UNITS_FF,
        activation="relu",
        kernel_regularizer=l2(L2_REGULARIZATION),
    )(out1)
    ffn = Dropout(DROPOUT)(ffn)
    ffn = Dense(
        D_MODEL,
        kernel_regularizer=l2(L2_REGULARIZATION),
    )(ffn)
    ffn = Dropout(DROPOUT)(ffn)

    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2


# %%
def add_day_indices(tensor):
    """
    Adds day indices to sequential input.
    tensor: shape (batch, LOOKBACK_DAYS, ?)
    We'll create a new dimension with tf.range(LOOKBACK_DAYS) and concatenate it.
    """
    batch_size = tf.shape(tensor)[0]  # dynamic batch size
    indices = tf.range(LOOKBACK_DAYS, dtype=tf.float32)  # shape=(LOOKBACK_DAYS,)
    indices = tf.expand_dims(indices, axis=0)  # shape=(1, LOOKBACK_DAYS)
    indices = tf.tile(indices, [batch_size, 1])  # shape=(batch, LOOKBACK_DAYS)
    indices = tf.expand_dims(indices, axis=-1)  # shape=(batch, LOOKBACK_DAYS, 1)

    # Concatenate to existing features: now we have one extra dimension for day index
    return tf.concat([tensor, indices], axis=-1)


# %%
def build_transformer_mdn(
    num_features: int,
    ticker_ids_dim: Optional[int],
):
    """
    Creates a Transformer-based encoder for sequences of shape:
      (lookback_days, num_features)
    Then outputs 3*n_mixtures for MDN (univariate).
    """
    # Input layers
    feature_inputs = Input(shape=(LOOKBACK_DAYS, num_features), name="feature_inputs")

    if ticker_ids_dim is not None:
        ticker_inputs = Input(
            shape=(), dtype=tf.int32, name="ticker_inputs"
        )  # Scalar ID

        # Ticker embedding
        ticker_embedding = Embedding(
            input_dim=ticker_ids_dim,  # Number of unique tickers
            output_dim=D_TICKER_EMBEDDING,  # Size of the embedding vector
            name="ticker_embedding",
        )(ticker_inputs)

        # Expand and concatenate with input features
        ticker_embedding_broadcast = RepeatVector(LOOKBACK_DAYS)(ticker_embedding)
        x = Concatenate(axis=-1)([feature_inputs, ticker_embedding_broadcast])
    else:
        x = feature_inputs

    # Append day indices automatically via a Lambda layer
    x = Lambda(add_day_indices)(x)

    # Project inputs to d_model
    x = Dense(D_MODEL, activation=None)(x)

    # Stack multiple Transformer encoder blocks
    for _ in range(NUM_ENCODERS):
        x = transformer_encoder(x)

    # Global average pooling (or take last time step)
    x = GlobalAveragePooling1D()(x)

    # Create initializers for MDN output layer
    mdn_kernel_init = get_mdn_kernel_initializer(N_MIXTURES)
    mdn_bias_init = get_mdn_bias_initializer(N_MIXTURES)

    # Output layer: 3*n_mixtures => [logits_pi, mu, log_var]
    mdn_output = Dense(
        3 * N_MIXTURES,
        activation=None,
        name="mdn_output",
        kernel_initializer=mdn_kernel_init,
        bias_initializer=mdn_bias_init,
    )(x)

    model = Model(
        inputs=(
            [feature_inputs, ticker_inputs]
            if ticker_ids_dim is not None
            else feature_inputs
        ),
        outputs=mdn_output,
    )
    return model


class FixedLearningRateSchedule(tf.keras.callbacks.Callback):
    """
    Fixed learning rate based on results from validation.
    This is only used for the final training phase.
    """

    def on_epoch_begin(self, epoch, logs=None):
        # Piecewise schedule
        if epoch < 10:
            lr = 5e-5
        elif epoch < 20:
            lr = 2.5e-5
        elif epoch < 25:
            lr = 1e-5
        else:
            lr = 5e-6  # 0.5e-5

        self.model.optimizer.learning_rate.assign(lr)
        print(f"Epoch {epoch + 1} - Setting learning rate to {lr:.6g}")


def _train_single_member(args):
    (
        i,
        build_kwargs,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size,
        patience,
        lr,
        weight_decay,
        n_mixtures,
        pi_penalty,
        pre_trained_weights,
    ) = args

    # Build/compile model
    model = build_transformer_mdn(**build_kwargs)
    if pre_trained_weights is not None:
        model.load_weights(pre_trained_weights)
    model.compile(
        optimizer=Adam(learning_rate=lr, weight_decay=weight_decay),
        loss=mdn_nll_tf(n_mixtures, pi_penalty),
    )

    # Always have the progress callback
    progress_cb = ParallelProgressCallback(worker_id=i)

    # -----------------------------------------------------------
    # Decide which callbacks + how many epochs based on TEST_SET
    # -----------------------------------------------------------
    if TEST_SET == "validation":
        # Dynamic schedule & early stopping
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=REDUCE_LR_PATIENCE, min_lr=1e-6
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        )
        callbacks = [reduce_lr, early_stop, progress_cb]
        n_epochs = 50
    else:
        # Fixed piecewise schedule with no early stopping
        fixed_schedule_cb = FixedLearningRateSchedule()
        callbacks = [progress_cb, fixed_schedule_cb]
        # Total of 10 + 10 + 5 + 5 = 30 epochs
        n_epochs = 30

    print(f"[Parallel] Fitting model {i} with TEST_SET={TEST_SET} ...")

    history = model.fit(
        X_train,
        y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=not PARALLELLIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    val_loss = float(np.min(history.history["val_loss"]))
    best_epoch = int(np.argmin(history.history["val_loss"]))
    print(f"[Parallel] Model {i} validation loss: {val_loss} (epoch {best_epoch})")

    return i, model.get_weights(), history.history, val_loss, best_epoch


# %%
# Begin script
if __name__ == "__main__":
    print(f"Training Transformer MDN Ensemble v{VERSION}")
    mp.set_start_method("spawn", force=True)

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
    # 2) Build model
    ensemble_model = MDNEnsemble(
        [
            build_transformer_mdn(
                num_features=train.X.shape[2],
                ticker_ids_dim=data.ticker_ids_dim if INCLUDE_TICKERS else None,
            )
            for _ in range(N_ENSEMBLE_MEMBERS)
        ],
        N_MIXTURES,
    )

    # %%
    # 4) Load existing model if it exists
    model_fname = f"models/{MODEL_NAME}.keras"
    already_trained = False
    if os.path.exists(model_fname):
        mdn_kernel_initializer = get_mdn_kernel_initializer(N_MIXTURES)
        mdn_bias_initializer = get_mdn_bias_initializer(N_MIXTURES)
        ensemble_model = tf.keras.models.load_model(
            model_fname,
            custom_objects={
                "loss_fn": mean_mdn_crps_tf(N_MIXTURES, PI_PENALTY),
                "mdn_kernel_initializer": mdn_kernel_initializer,
                "mdn_bias_initializer": mdn_bias_initializer,
                "MDNEnsemble": MDNEnsemble,
                "add_day_indices": add_day_indices,
            },
        )
        already_trained = True
        print("Loaded pre-trained model from disk.")

    # %%
    # 5) Train model
    # Train each member in parallel until val loss converges
    X_train = [train.X, train_ticker_ids] if INCLUDE_TICKERS else train.X
    y_train = train.y
    X_val = [test.X, test_ticker_ids] if INCLUDE_TICKERS else test.X
    y_val = test.y

    # Create argument tuples for each submodel
    job_args = []
    for i in range(N_ENSEMBLE_MEMBERS):
        build_kwargs = {
            "num_features": train.X.shape[2],
            "ticker_ids_dim": data.ticker_ids_dim if INCLUDE_TICKERS else None,
        }
        pre_trained_weights = (
            ensemble_model.submodels[i].get_weights() if already_trained else None
        )
        job_args.append(
            (
                i,
                build_kwargs,
                X_train,
                y_train,
                X_val,
                y_val,
                BATCH_SIZE,
                PATIENCE,
                LEARNING_RATE,
                WEIGHT_DECAY,
                N_MIXTURES,
                PI_PENALTY,
                pre_trained_weights,
            )
        )

    histories = [None] * N_ENSEMBLE_MEMBERS
    val_losses = [None] * N_ENSEMBLE_MEMBERS
    optimal_epochs = [None] * N_ENSEMBLE_MEMBERS

    if PARALLELLIZE:
        with mp.Pool(processes=N_ENSEMBLE_MEMBERS) as pool:
            results = pool.map(_train_single_member, job_args)
    else:
        results = [_train_single_member(args) for args in job_args]

    # results is a list of (i, trained_model, history_dict, val_loss).
    # Sort by i so we can store them in order:
    results.sort(key=lambda x: x[0])

    # Store trained submodels back into the ensemble, plus record histories/losses
    for i, weights, hist_dict, best_loss, best_epoch in results:
        ensemble_model.submodels[i].set_weights(weights)
        histories[i] = hist_dict
        val_losses[i] = best_loss
        optimal_epochs[i] = best_epoch
        print(f"Model {i} done (best val_loss={best_loss} [epoch {best_epoch}]).")

    # %%
    # 6) Save model
    ensemble_model.save(model_fname)

    # Store details from training
    with open(f"models/{MODEL_NAME}_training_details.txt", "w") as f:
        f.write(f"Training details for {MODEL_NAME}\n")
        f.write(f"VERSION: {VERSION}\n")
        f.write(f"LOOKBACK_DAYS: {LOOKBACK_DAYS}\n")
        f.write(f"SUFFIX: {SUFFIX}\n")
        f.write(f"\n\nFeatures:\n")
        f.write(
            f"MULTIPLY_MARKET_FEATURES_BY_BETA: {MULTIPLY_MARKET_FEATURES_BY_BETA}\n"
        )
        f.write(f"PI_PENALTY: {PI_PENALTY}\n")
        f.write(f"MU_PENALTY: {MU_PENALTY}\n")
        f.write(f"SIGMA_PENALTY: {SIGMA_PENALTY}\n")
        f.write(f"INCLUDE_MARKET_FEATURES: {INCLUDE_MARKET_FEATURES}\n")
        f.write(f"INCLUDE_FNG: {INCLUDE_FNG}\n")
        f.write(f"INCLUDE_RETURNS: {INCLUDE_RETURNS}\n")
        f.write(f"INCLUDE_INDUSTRY: {INCLUDE_INDUSTRY}\n")
        f.write(f"INCLUDE_GARCH: {INCLUDE_GARCH}\n")
        f.write(f"INCLUDE_BETA: {INCLUDE_BETA}\n")
        f.write(f"INCLUDE_OTHERS: {INCLUDE_OTHERS}\n")
        f.write(f"INCLUDE_TICKERS: {INCLUDE_TICKERS}\n")
        f.write(f"INCLDUE_FRED_MD: {INCLUDE_FRED_MD}\n")
        f.write(f"INCLUDE_10_DAY_IVOL: {INCLUDE_10_DAY_IVOL}\n")
        f.write(f"INCLUDE_30_DAY_IVOL: {INCLUDE_30_DAY_IVOL}\n")
        f.write(f"INCLUDE_1MIN_RV: {INCLUDE_1MIN_RV}\n")
        f.write(f"INCLUDE_5MIN_RV: {INCLUDE_5MIN_RV}\n")
        f.write(f"\n\nModel settings:\n")
        f.write(f"D_MODEL: {D_MODEL}\n")
        f.write(f"HIDDEN_UNITS_FF: {HIDDEN_UNITS_FF}\n")
        f.write(f"N_MIXTURES: {N_MIXTURES}\n")
        f.write(f"DROPOUT: {DROPOUT}\n")
        f.write(f"L2_REGULARIZATION: {L2_REGULARIZATION}\n")
        f.write(f"NUM_ENCODERS: {NUM_ENCODERS}\n")
        f.write(f"NUM_HEADS: {NUM_HEADS}\n")
        f.write(f"D_TICKER_EMBEDDING: {D_TICKER_EMBEDDING}\n")
        f.write(f"\n\nTraining settings:\n")
        f.write(f"PATIENCE: {PATIENCE}\n")
        f.write(f"WEIGHT_DECAY: {WEIGHT_DECAY}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"N_ENSEMBLE_MEMBERS: {N_ENSEMBLE_MEMBERS}\n")
        f.write(f"\n\nTraining results:\n")
        f.write(f"Optimal number of epochs: {[int(n) for n in optimal_epochs]}\n")
        f.write(f"Validation losses: {val_losses}\n")
        f.write(f"\n\nTraining loss histories:\n")
        training_loss_df = pd.DataFrame(
            [h["loss"] for h in histories], index=range(N_ENSEMBLE_MEMBERS)
        ).round(4)
        training_loss_df.index.name = "Member"
        training_loss_df.columns = [
            f"Epoch {i}" for i in range(1, training_loss_df.shape[1] + 1)
        ]
        training_loss_df.to_csv(f, sep="\t", mode="a")
        f.write("\n\nValidation loss histories:\n")
        validation_loss_df = pd.DataFrame(
            [h["val_loss"] for h in histories], index=range(N_ENSEMBLE_MEMBERS)
        ).round(4)
        validation_loss_df.index.name = "Member"
        validation_loss_df.columns = [
            f"Epoch {i}" for i in range(1, validation_loss_df.shape[1] + 1)
        ]
        validation_loss_df.to_csv(f, sep="\t", mode="a")
        learning_rate_df = pd.DataFrame(
            [h["learning_rate"] for h in histories],
            index=range(N_ENSEMBLE_MEMBERS),
        )
        learning_rate_df = learning_rate_df.applymap(lambda x: f"{x:.3e}")
        learning_rate_df.index.name = "Member"
        learning_rate_df.columns = [
            f"Epoch {i}" for i in range(1, learning_rate_df.shape[1] + 1)
        ]
        learning_rate_df.to_csv(f, sep="\t", mode="a")

    # %%
    # 7) Commit and push
    try:
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(["git", "add", f"models/*{MODEL_NAME}*"], check=True)

        commit_header = f"Train Transformer MDN Ensemble {VERSION}"
        commit_body = f"Training history:\n" + "\n".join([str(h) for h in histories])

        subprocess.run(
            ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
        )
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")

    # %%
    # 8) Single-pass predictions
    y_pred_mdn, epistemic_var = ensemble_model.predict(
        [test.X, test_ticker_ids] if INCLUDE_TICKERS else test.X
    )
    pi_pred, mu_pred, sigma_pred = parse_mdn_output(
        y_pred_mdn, N_MIXTURES * N_ENSEMBLE_MEMBERS
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
            N_MIXTURES * N_ENSEMBLE_MEMBERS,
            ticker=ticker,
            save_to=f"results/distributions/{ticker}_transformer_mdn_v{VERSION}_ensemble.svg",
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
        for j in range(N_MIXTURES * N_ENSEMBLE_MEMBERS):
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
    plt.savefig(f"results/transformer_mdn_v{VERSION}_ensemble_mixture_weights.svg")

    # %%
    # 11) Calculate intervals for 67%, 95%, 97.5% and 99% confidence levels
    confidence_levels = [0.67, 0.90, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999]
    intervals = calculate_intervals_vectorized(
        pi_pred, mu_pred, sigma_pred, confidence_levels
    )

    # %%
    # 12) Plot time series with mean, volatility and actual returns for last X days
    mean = (pi_pred * mu_pred).numpy().sum(axis=1)
    for ticker in example_tickers:
        from_idx, to_idx = test.get_range(ticker)
        ticker_mean = mean[from_idx:to_idx]
        filtered_mean = ticker_mean
        ticker_intervals = intervals[from_idx:to_idx]
        filtered_intervals = ticker_intervals
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
        median = filtered_intervals[:, 0, 0]
        plt.plot(dates, median, label="Median", color="green")
        for i, cl in enumerate(confidence_levels):
            if cl == 0:
                continue
            plt.fill_between(
                dates,
                filtered_intervals[:, i, 0],
                filtered_intervals[:, i, 1],
                color="blue",
                alpha=0.7 - i * 0.07,
                label=f"{100*cl:.1f}% Interval",
            )
            # Mark violations
            violations = np.logical_or(
                actual_return < filtered_intervals[:, i, 0],
                actual_return > filtered_intervals[:, i, 1],
            )
            plt.scatter(
                np.array(dates)[violations],
                actual_return[violations],
                marker="x",
                label=f"Violations ({100*cl:.1f}%)",
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
        plt.title(f"Transformer MDN predictions for {ticker}, {TEST_SET} data")
        plt.xlabel("Date")
        plt.ylabel("LogReturn")
        plt.legend()
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"results/time_series/{ticker}_transformer_mdn_v{VERSION}_ensemble.svg"
        )

    # %%
    # 13) Plot predicted means with epistemic uncertainty for example tickers
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
        plt.title(f"Transformer MDN predictions for {ticker}, {TEST_SET} data")
        plt.xlabel("Date")
        plt.ylabel("LogReturn")
        plt.legend()
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"results/time_series/{ticker}_transformer_mdn_v{VERSION}_ensemble_epistemic.svg"
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
    df_validation["NLL"] = mdn_nll_numpy(N_MIXTURES * N_ENSEMBLE_MEMBERS)(
        test.y, y_pred_mdn
    )

    # %%
    # Calculate CRPS
    crps = mdn_crps_tf(N_MIXTURES * N_ENSEMBLE_MEMBERS)
    df_validation["CRPS"] = crps(test.y, y_pred_mdn)

    # %%
    # Calculate ECE
    ece = ece_mdn(N_MIXTURES * N_ENSEMBLE_MEMBERS, test.y, y_pred_mdn)
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
        f"predictions/transformer_mdn_predictions{SUFFIX}_v{VERSION}_ensemble.csv"
    )

    # %%
    # Commit predictions
    try:
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(
            ["git", "add", f"predictions/*transformer_mdn*{SUFFIX}*"], check=True
        )
        commit_header = f"Add predictions for Transformer MDN Ensemble {VERSION}"
        commit_body = f"Validation loss: {df_validation['NLL'].mean()}"
        subprocess.run(
            ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
        )
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")

# %%
