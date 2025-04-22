# %%
# Define parameters (based on settings)
import subprocess
from typing import Optional
from shared.ensemble import MDNEnsemble, ParallelProgressCallback
from shared.conf_levels import format_cl
from settings import LOOKBACK_DAYS, SUFFIX, VALIDATION_TEST_SPLIT, TEST_SET
import multiprocessing as mp


VERSION = "ivol"
MODEL_NAME = f"transformer_mdn_ensemble_{VERSION}_{TEST_SET}_expanding"

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
INCLUDE_1MIN_RV = False
INCLUDE_5MIN_RV = False
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
N_ENSEMBLE_MEMBERS = 10
PARALLELLIZE = True
RETRAINING_INTERVAL = 30  # Monthly retraining interval

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


# %%
# Define code for training each member
def _train_single_member(args):
    """
    Worker function for parallel training of one submodel.
    Returns (submodel_index, trained_model, history_dict, best_val_loss).
    """
    (
        i,
        build_kwargs,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        lr,
        weight_decay,
        n_mixtures,
        pi_penalty,
    ) = args

    # Rebuild callbacks/optimizer/loss in each process
    model = build_transformer_mdn(**build_kwargs)
    model.compile(
        optimizer=Adam(learning_rate=lr, weight_decay=weight_decay),
        loss=mdn_nll_tf(n_mixtures, pi_penalty),
    )

    progress_cb = ParallelProgressCallback(worker_id=i)

    # Fixed piecewise schedule with no early stopping
    fixed_schedule_cb = FixedLearningRateSchedule()

    print(f"[Worker {i}] Fitting model {i}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=30,  # Correponds to the 10 + 10 + 5 + 5 epochs in the FixedLearningRateSchedule
        batch_size=batch_size,
        verbose=not PARALLELLIZE,
        validation_data=(X_test, y_test),
        callbacks=[progress_cb, fixed_schedule_cb],
    )
    val_loss = float(np.min(history.history["val_loss"]))
    best_epoch = np.argmin(history.history["val_loss"])
    print(f"[Worker {i}] Model {i} validation loss: {val_loss} (epoch {best_epoch})")
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

    # %%
    # Garbage collection
    gc.collect()

    # %%
    # Train model (rolling if ROLLING_INTERVAL is set)
    # Get correct train and test
    first_test_date = pd.to_datetime(VALIDATION_TEST_SPLIT)
    if RETRAINING_INTERVAL is None or RETRAINING_INTERVAL == 0:
        RETRAINING_INTERVAL = np.inf
    end_date = lambda: first_test_date + pd.DateOffset(days=RETRAINING_INTERVAL)
    y_pred_results = []
    epistemic_var_results = []
    symbols = []
    dates = []
    true_y = []
    while (test := data.get_test_set_for_date(first_test_date, end_date())).y.shape[0]:
        train = data.get_training_set_for_date(first_test_date)

        # 1) Inspect shapes
        print("Making rolling predictions from", first_test_date.date().isoformat())
        print(f"X_train.shape: {train.X.shape}, y_train.shape: {train.y.shape}")
        print(
            f"Training dates: {train.dates.min().date()} to {train.dates.max().date()}"
        )
        print(f"Test set shape: {test.X.shape}, {test.y.shape}")
        print(f"Test dates: {test.dates.min().date()} to {test.dates.max().date()}")

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

        # 3) Extract relevant data
        X_train = [train.X, train_ticker_ids] if INCLUDE_TICKERS else train.X
        y_train = train.y
        X_test = [test.X, test_ticker_ids] if INCLUDE_TICKERS else test.X
        y_test = test.y

        # 4) Load if exists
        model_fname = (
            f"models/rolling/{MODEL_NAME}_{first_test_date.date().isoformat()}.h5"
        )
        if os.path.exists(model_fname):
            ensemble_model.load_weights(model_fname)
            print("Loaded pre-trained model from disk.")
        else:
            print("Could not find pre-trained model", model_fname)

            # 5) Train each member in parallel until val loss converges
            job_args = []
            for i in range(N_ENSEMBLE_MEMBERS):
                build_kwargs = {
                    "num_features": train.X.shape[2],
                    "ticker_ids_dim": data.ticker_ids_dim if INCLUDE_TICKERS else None,
                }
                job_args.append(
                    (
                        i,
                        build_kwargs,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        BATCH_SIZE,
                        LEARNING_RATE,
                        WEIGHT_DECAY,
                        N_MIXTURES,
                        PI_PENALTY,
                    )
                )

            if PARALLELLIZE:
                with mp.Pool(processes=N_ENSEMBLE_MEMBERS) as pool:
                    results = pool.map(_train_single_member, job_args)
            else:
                results = [_train_single_member(args) for args in job_args]

            # Store trained submodels back into the ensemble, plus record histories/losses
            for i, weights, hist_dict, best_loss, best_epoch in results:
                ensemble_model.submodels[i].set_weights(weights)
                print(
                    f"Model {i} done (best val_loss={best_loss} [epoch {best_epoch}])."
                )

            # Save model in case we need it
            ensemble_model.save(model_fname)

        # 6) Make predictions
        y_pred_mdn, epistemic_var = ensemble_model.predict(X_test)
        y_pred_results.append(y_pred_mdn)
        epistemic_var_results.append(epistemic_var)
        symbols += list(test.tickers)
        dates += list(test.dates)
        true_y += list(test.y)

        # 7) Move to next date
        first_test_date = end_date() + pd.DateOffset(days=1)

    # %%
    # 5) Concatenate results
    y_pred_mdn = np.concatenate(y_pred_results, axis=0)
    epistemic_var = np.concatenate(epistemic_var_results, axis=0)
    pi_pred, mu_pred, sigma_pred = parse_mdn_output(
        y_pred_mdn, N_MIXTURES * N_ENSEMBLE_MEMBERS
    )
    symbols_arr = np.array(symbols)
    filter_ndarray = lambda ticker, ndarr: np.array(ndarr)[symbols_arr == ticker]

    # %%
    # 6) Plot 10 charts with the distributions for 10 random days
    example_tickers = ["AAPL", "WMT", "GS"]
    for ticker in example_tickers:
        plot_sample_days(
            filter_ndarray(ticker, dates),
            filter_ndarray(ticker, true_y),
            filter_ndarray(ticker, pi_pred),
            filter_ndarray(ticker, mu_pred),
            filter_ndarray(ticker, sigma_pred),
            N_MIXTURES * N_ENSEMBLE_MEMBERS,
            ticker=ticker,
            save_to=f"results/distributions/{ticker}_{MODEL_NAME}.svg",
        )

    # %%
    # 7) Plot weights over time to show how they change
    fig, axes = plt.subplots(
        nrows=len(example_tickers), figsize=(18, len(example_tickers) * 9)
    )

    # Dictionary to store union of legend entries
    legend_dict = {}

    for ax, ticker in zip(axes, example_tickers):
        pi_pred_ticker = filter_ndarray(ticker, pi_pred)
        for j in range(N_MIXTURES * N_ENSEMBLE_MEMBERS):
            mean_over_time = np.mean(pi_pred_ticker[:, j], axis=0)
            if mean_over_time < 0.01:
                continue
            (line,) = ax.plot(
                filter_ndarray(ticker, dates),
                pi_pred_ticker[:, j],
                label=f"$\pi_{{{j}}}$",
            )
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
    plt.savefig(f"results/{MODEL_NAME}_mixture_weights.svg")

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
        ticker_mean = filter_ndarray(ticker, mean)
        ticker_intervals = filter_ndarray(ticker, intervals)
        ticker_dates = filter_ndarray(ticker, dates)
        actual_return = filter_ndarray(ticker, true_y)

        plt.figure(figsize=(12, 6))
        plt.plot(
            ticker_dates,
            actual_return,
            label="Actual Returns",
            color="black",
            alpha=0.5,
        )
        plt.plot(ticker_dates, ticker_mean, label="Predicted Mean", color="red")
        median = ticker_intervals[:, 0, 0]
        plt.plot(ticker_dates, median, label="Median", color="green")
        for i, cl in enumerate(confidence_levels):
            if cl == 0:
                continue
            plt.fill_between(
                ticker_dates,
                ticker_intervals[:, i, 0],
                ticker_intervals[:, i, 1],
                color="blue",
                alpha=0.7 - i * 0.07,
                label=f"{100*cl:.1f}% Interval",
            )
            # Mark violations
            violations = np.logical_or(
                actual_return < ticker_intervals[:, i, 0],
                actual_return > ticker_intervals[:, i, 1],
            )
            plt.scatter(
                np.array(ticker_dates)[violations],
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
        plt.title(f"Transformer w MDN predictions for {ticker}, test data")
        plt.xlabel("Date")
        plt.ylabel("LogReturn")
        # Place legend outside of plot
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(f"results/time_series/{ticker}_{MODEL_NAME}.svg")

    # %%
    # 13) Make data frame for signle pass predictions
    df_validation = pd.DataFrame(
        np.vstack([dates, symbols]).T,
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
        true_y, y_pred_mdn
    )

    # %%
    # Calculate CRPS
    crps = mdn_crps_tf(N_MIXTURES * N_ENSEMBLE_MEMBERS)
    df_validation["CRPS"] = crps(true_y, y_pred_mdn)

    # %%
    # Calculate ECE
    ece = ece_mdn(N_MIXTURES * N_ENSEMBLE_MEMBERS, np.array(true_y), y_pred_mdn)
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
    df_validation.set_index(["Date", "Symbol"]).xs("AAPL", level="Symbol").sort_index()[
        ["LB_90", "ES_95", "LB_98", "ES_99"]
    ].rename(
        columns={
            "LB_90": "95% VaR",
            "ES_95": "95% ES",
            "LB_98": "99% VaR",
            "ES_99": "99% ES",
        }
    ).plot(
        title="99% VaR and ES for AAPL",
        # Color appropriately
        color=["#ffaaaa", "#ff0000", "#aaaaff", "#0000ff"],
        figsize=(12, 6),
    )

    # %%
    # Calculate probability of price increase
    df_validation["Prob_Increase"] = calculate_prob_above_zero_vectorized(
        pi_pred, mu_pred, sigma_pred
    )

    # %%
    # Save
    df_validation.set_index(["Date", "Symbol"]).to_csv(f"predictions/{MODEL_NAME}.csv")

    # %%
    # Commit predictions
    try:
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(["git", "add", f"predictions/*transformer_mdn*"], check=True)
        commit_header = (
            f"Add predictions for expanding Transformer MDN Ensemble {VERSION}"
        )
        commit_body = f"Loss (test set): {df_validation['NLL'].mean()}"
        subprocess.run(
            ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
        )
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
