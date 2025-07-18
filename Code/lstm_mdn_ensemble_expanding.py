# %%
# Define parameters (based on settings)
import subprocess
from typing import Optional
from shared.jupyter import is_notebook
from shared.ensemble import MDNEnsemble, ParallelProgressCallback
from shared.conf_levels import format_cl
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    VALIDATION_TEST_SPLIT,
)
import multiprocessing as mp
import shared.styling_guidelines_graphs
import sys


VERSION = sys.argv[1] if len(sys.argv) > 1 else "rv-and-ivol-final-rolling"

# %%
# Feature selection
MULTIPLY_MARKET_FEATURES_BY_BETA = False
PI_PENALTY = False
MU_PENALTY = False
SIGMA_PENALTY = False
INCLUDE_MARKET_FEATURES = False
INCLUDE_FNG = False
INCLUDE_RETURNS = False
INCLUDE_INDUSTRY = False
INCLUDE_GARCH = False
INCLUDE_BETA = False
INCLUDE_OTHERS = False
INCLUDE_TICKERS = False
INCLDUE_FRED_MD = False
INCLUDE_10_DAY_IVOL = "ivol" in VERSION
INCLUDE_30_DAY_IVOL = "ivol" in VERSION
INCLUDE_1MIN_RV = "rv" in VERSION
INCLUDE_5MIN_RV = "rv" in VERSION

# %%
# Model settings
HIDDEN_UNITS = 60
N_MIXTURES = 10
DROPOUT = 0.0
L2_REG = 1e-4
NUM_HIDDEN_LAYERS = 0
EMBEDDING_DIMENSIONS = None
MODEL_NAME = f"lstm_mdn_ensemble{SUFFIX}_v{VERSION}_test"

# %%
# Settings for training
WEIGHT_DECAY = 1e-4  # from optuna
LEARNING_RATE = 0.00015  # from optuna
BATCH_SIZE = 32
N_ENSEMBLE_MEMBERS = 10
EPOCHS = 15
PARALLELLIZE = True
ROLLING_INTERVAL = 30  # Monthly retraining interval

# %%
# Imports from code shared across models
from shared.mdn import (
    calculate_es_for_quantile,
    calculate_intervals_vectorized,
    calculate_prob_above_zero_vectorized,
    get_mdn_bias_initializer,
    get_mdn_kernel_initializer,
    parse_mdn_output,
    plot_sample_day,
    plot_sample_days,
    mdn_mean_and_var,
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
    LSTM,
    Embedding,
    Flatten,
    Concatenate,
    RepeatVector,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")


# %%
def build_lstm_mdn(
    num_features: int,
    ticker_ids_dim: Optional[int],
):
    """
    Creates a lstm-based encoder for sequences of shape:
      (lookback_days, num_features)
    Then outputs 3*n_mixtures for MDN (univariate).
    """
    # Sequence input (time series)
    seq_input = Input(shape=(LOOKBACK_DAYS, num_features), name="seq_input")

    inputs = [seq_input]  # We'll append ticker_input if we have tickers

    # If we have ticker IDs, embed them
    if ticker_ids_dim is not None:
        ticker_input = Input(shape=(), dtype="int32", name="ticker_input")
        inputs.append(ticker_input)

        # 1) Embed the ticker ID -> shape: (batch, embed_dim)
        ticker_embed = Embedding(
            input_dim=ticker_ids_dim,
            output_dim=EMBEDDING_DIMENSIONS,
            name="ticker_embedding",
        )(
            ticker_input
        )  # shape: (batch, 1, embed_dim)

        ticker_embed = Flatten()(ticker_embed)  # shape: (batch, embed_dim)

        # 2) Repeat the embedding across time -> shape: (batch, lookback_days, embed_dim)
        ticker_embed = RepeatVector(LOOKBACK_DAYS)(ticker_embed)

        # 3) Concatenate with the input features -> shape: (batch, lookback_days, num_features + embed_dim)
        x = Concatenate(axis=-1, name="concat_seq_ticker")([seq_input, ticker_embed])
    else:
        # No ticker input
        x = seq_input

    # 4) Pass through LSTM
    x = LSTM(
        units=HIDDEN_UNITS,
        activation="tanh",
        kernel_regularizer=l2(1e-3),
        name="lstm_layer",
    )(x)

    if DROPOUT > 0:
        x = Dropout(DROPOUT, name="dropout_lstm")(x)

    # 5) Optionally pass through additional Dense layers
    for i in range(NUM_HIDDEN_LAYERS):
        x = Dense(
            units=HIDDEN_UNITS,
            activation="relu",
            kernel_regularizer=l2(1e-3),
            name=f"dense_layer_{i}",
        )(x)
        if DROPOUT > 0:
            x = Dropout(DROPOUT, name=f"dropout_layer_{i}")(x)

    # MDN output layer: 3 * n_mixtures (for [logits_pi, mu, log_var])
    mdn_kernel_init = get_mdn_kernel_initializer(N_MIXTURES)
    mdn_bias_init = get_mdn_bias_initializer(N_MIXTURES)
    mdn_output = Dense(
        3 * N_MIXTURES,
        activation=None,
        kernel_initializer=mdn_kernel_init,
        bias_initializer=mdn_bias_init,
        name="mdn_output",
    )(x)

    model = Model(
        inputs=[seq_input, ticker_input] if ticker_ids_dim is not None else [seq_input],
        outputs=mdn_output,
    )
    return model


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
    model = build_lstm_mdn(**build_kwargs)
    model.compile(
        optimizer=Adam(learning_rate=lr, weight_decay=weight_decay),
        loss=mdn_nll_tf(n_mixtures, pi_penalty),
    )

    # custom callback
    progress_cb = ParallelProgressCallback(worker_id=i)

    print(f"[Worker {i}] Fitting model {i}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=batch_size,
        verbose=not PARALLELLIZE,
        validation_data=(X_test, y_test),
        callbacks=[progress_cb],
    )
    val_loss = float(np.min(history.history["val_loss"]))
    best_epoch = np.argmin(history.history["val_loss"])
    print(f"[Worker {i}] Model {i} validation loss: {val_loss} (epoch {best_epoch})")
    return i, model.get_weights(), history.history, val_loss, best_epoch


# %%
# Begin script
if __name__ == "__main__":
    print(f"Training LSTM MDN Ensemble v{VERSION}")
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
        include_fred_md=INCLDUE_FRED_MD,
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
    if ROLLING_INTERVAL is None or ROLLING_INTERVAL == 0:
        ROLLING_INTERVAL = np.inf
    end_date = lambda: first_test_date + pd.DateOffset(days=ROLLING_INTERVAL)
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
                build_lstm_mdn(
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

    # %%
    # Define function to filter arrays by ticker
    filter_cache = {}

    def filter_ndarray(ticker, ndarr):
        cache_key = (ticker, id(ndarr))
        if cache_key in filter_cache:
            return filter_cache[cache_key]
        val = np.array([val for val, t in zip(ndarr, symbols) if t == ticker])
        filter_cache[cache_key] = val
        return val

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
    # 6b) Make plot for paper: 2x2 grid with 2 tickers and 2 random days
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    paper_name = "-".join(
        (["RV"] if INCLUDE_1MIN_RV else []) + (["IV"] if INCLUDE_30_DAY_IVOL else [])
    )
    for i, (ticker, date) in enumerate(
        [
            ("AAPL", "2020-04-15"),
            ("AAPL", "2023-06-05"),
            ("WMT", "2020-04-15"),
            ("WMT", "2023-06-05"),
        ]
    ):
        date = pd.to_datetime(date)
        ax = axes[i // 2, i % 2]
        ticker_dates = filter_ndarray(ticker, dates)
        day = len(ticker_dates) - list(ticker_dates).index(date) - 1
        plot_sample_day(
            ticker_dates,
            filter_ndarray(ticker, true_y),
            filter_ndarray(ticker, pi_pred),
            filter_ndarray(ticker, mu_pred),
            filter_ndarray(ticker, sigma_pred),
            N_MIXTURES * N_ENSEMBLE_MEMBERS,
            ax,
            ticker,
            day,
        )
    plt.tight_layout()
    plt.savefig(f"results/distributions/{MODEL_NAME}_comparison.pdf")
    if is_notebook():
        plt.show()
    plt.close()

    # %%
    # 6c) Make plot for paper: 2x2 grid with 2 tickers and 2 random days - SAVE AS INDUVIDUAL FILES
    # Create and save individual plots (7x4 inches each)
    paper_name = "-".join(
        (["RV"] if INCLUDE_1MIN_RV else []) + (["IV"] if INCLUDE_30_DAY_IVOL else [])
    )

    # Define tickers and dates to loop through
    examples = [
        ("AAPL", "2020-04-16"),
        ("AAPL", "2023-06-06"),
        ("WMT", "2020-04-16"),
        ("WMT", "2023-06-06"),
    ]

    for i, (ticker, date) in enumerate(examples):
        date = pd.to_datetime(date)
        ticker_dates = filter_ndarray(ticker, dates)
        day = len(ticker_dates) - list(ticker_dates).index(date)

        # Create a new figure for each sample day
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.set_xlim(-0.1, 0.1)
        plot_sample_day(
            ticker_dates,
            filter_ndarray(ticker, true_y),
            filter_ndarray(ticker, pi_pred),
            filter_ndarray(ticker, mu_pred),
            filter_ndarray(ticker, sigma_pred),
            N_MIXTURES * N_ENSEMBLE_MEMBERS,
            ax,
            ticker,
            day,
        )
        ax.set_title(
            f"{date.strftime('%Y-%m-%d')} - LSTM-MDN-{paper_name} predicted return distribution for {ticker}",
            pad=15,
        )
        ax.set_xlabel("Return")
        plt.tight_layout()
        filename = (
            f"results/distributions/{MODEL_NAME}_{ticker}_{date.date()}_single.pdf"
        )
        plt.savefig(filename)

        if is_notebook():
            plt.show()

        plt.close()

    # %%
    # 7) Plot weights over time to show how they change
    # Dictionary to store union of legend entries
    for ticker in example_tickers:
        legend_dict = {}
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
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
        ax.set_xlim(filter_ndarray(ticker, dates)[0], filter_ndarray(ticker, dates)[-1])
        ax.set_ylim(0, 0.1)
        ax.set_yticklabels(["{:.1f}%".format(x * 100) for x in ax.get_yticks()])
        ax.set_title(f"Evolution of LSTM-MDN-{paper_name} Mixture Weights for {ticker}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Weight")

        # Create a combined legend using the union of entries
        handles = list(legend_dict.values())
        labels = list(legend_dict.keys())
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(
            f"results/mixture_weights/{ticker}_{MODEL_NAME}_mixture_weights.pdf"
        )
        if is_notebook():
            plt.show()
        plt.close()

    # %%
    # 11) Calculate intervals for 67%, 95%, 97.5% and 99% confidence levels
    confidence_levels = [0.67, 0.90, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999]
    intervals = calculate_intervals_vectorized(
        pi_pred, mu_pred, sigma_pred, confidence_levels
    )

    # %%
    # 13) Make data frame for signle pass predictions
    df_validation = pd.DataFrame(
        np.vstack([dates, symbols]).T,
        columns=["Date", "Symbol"],
    )
    # %%
    # For comparison to other models, compute mixture means & variances
    mixture_mean_sp, total_var = mdn_mean_and_var(pi_pred, mu_pred, sigma_pred)
    aleatoric_var = total_var - epistemic_var
    mixture_mean_sp = mixture_mean_sp.numpy()
    mixture_vol = np.sqrt(aleatoric_var.numpy())
    df_validation["Mean_SP"] = mixture_mean_sp
    df_validation["Vol_SP"] = mixture_vol

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
    # Calculate probability of price increase
    df_validation["Prob_Increase"] = calculate_prob_above_zero_vectorized(
        pi_pred, mu_pred, sigma_pred
    )

    # %%
    # Add epistemic variance
    df_validation["EpistemicVarMean"] = epistemic_var

    # %%
    # Save
    df_validation.set_index(["Date", "Symbol"]).to_csv(f"predictions/{MODEL_NAME}.csv")

    # %%
    # Commit predictions
    try:
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(["git", "add", f"predictions/*lstm_mdn*{SUFFIX}*"], check=True)
        commit_header = f"Add predictions for LSTM MDN Ensemble {VERSION}"
        commit_body = f"Loss (test set): {df_validation['NLL'].mean()}"
        subprocess.run(
            ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
        )
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
