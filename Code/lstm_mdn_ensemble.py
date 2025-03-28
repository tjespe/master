# %%
# Define parameters (based on settings)
import subprocess
from typing import Optional
from shared.ensemble import MDNEnsemble, ParallelProgressCallback
from shared.conf_levels import format_cl
from settings import LOOKBACK_DAYS, SUFFIX, TEST_SET
import multiprocessing as mp

VERSION = "rv-and-ivol-final"

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
INCLUDE_10_DAY_IVOL = True
INCLUDE_30_DAY_IVOL = True
INCLUDE_1MIN_RV = True
INCLUDE_5MIN_RV = True

# %%
# Model settings
HIDDEN_UNITS = 60
N_MIXTURES = 10
DROPOUT = 0.0
L2_REG = 1e-4
NUM_HIDDEN_LAYERS = 0
EMBEDDING_DIMENSIONS = None
MODEL_NAME = f"lstm_mdn_ensemble{SUFFIX}_v{VERSION}"

# %%
# Settings for training
TRAIN = True
PATIENCE = 3  # Early stopping patience
WEIGHT_DECAY = 1e-4  # from optuna
LEARNING_RATE = 0.00015  # from optuna
BATCH_SIZE = 32
N_ENSEMBLE_MEMBERS = 10
OPTIMAL_EPOCHS = 7
USE_EARLY_STOPPING = TEST_SET == "validation"
EPOCHS = 50 if USE_EARLY_STOPPING else OPTIMAL_EPOCHS
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

    # Rebuild callbacks/optimizer/loss in each process
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    model = build_lstm_mdn(**build_kwargs)
    if pre_trained_weights is not None:
        model.load_weights(pre_trained_weights)
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
        validation_data=(X_val, y_val),
        callbacks=[early_stop, progress_cb] if USE_EARLY_STOPPING else [progress_cb],
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
    # Extract correct train and test data based on settings
    train = data.get_training_set(test_set=TEST_SET)
    test = data.get_test_set(test_set=TEST_SET)

    # %%
    # 1) Inspect shapes
    print(f"X_train.shape: {train.X.shape}, y_train.shape: {train.y.shape}")
    print(f"Training dates: {train.dates.min().date()} to {train.dates.max().date()}")
    print(f"Test set shape: {test.X.shape}, {test.y.shape}")
    print(f"Test dates: {test.dates.min().date()} to {test.dates.max().date()}")

    # %%
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
    already_trained = False

    # %%
    # 4) Load existing model if it exists
    model_fname = f"models/{MODEL_NAME}.keras"
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
            },
        )
        print("Loaded pre-trained model from disk.")
        already_trained = True

    # %%
    # 5) Train model
    # Train each member in parallel until val loss converges
    X_train = [train.X, train_ticker_ids] if INCLUDE_TICKERS else train.X
    y_train = train.y
    X_test = [test.X, test_ticker_ids] if INCLUDE_TICKERS else test.X
    y_test = test.y

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
                X_test,
                y_test,
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

    if TRAIN:
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
            f.write(f"INCLDUE_FRED_MD: {INCLDUE_FRED_MD}\n")
            f.write(f"INCLUDE_10_DAY_IVOL: {INCLUDE_10_DAY_IVOL}\n")
            f.write(f"INCLUDE_30_DAY_IVOL: {INCLUDE_30_DAY_IVOL}\n")
            f.write(f"INCLUDE_1MIN_RV: {INCLUDE_1MIN_RV}\n")
            f.write(f"INCLUDE_5MIN_RV: {INCLUDE_5MIN_RV}\n")
            f.write(f"\n\nModel settings:\n")
            f.write(f"HIDDEN_UNITS: {HIDDEN_UNITS}\n")
            f.write(f"N_MIXTURES: {N_MIXTURES}\n")
            f.write(f"DROPOUT: {DROPOUT}\n")
            f.write(f"L2_REG: {L2_REG}\n")
            f.write(f"NUM_HIDDEN_LAYERS: {NUM_HIDDEN_LAYERS}\n")
            f.write(f"EMBEDDING_DIMENSIONS: {EMBEDDING_DIMENSIONS}\n")
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
            )
            training_loss_df.index.name = "Member"
            training_loss_df.columns = [
                f"Epoch {i}" for i in range(1, training_loss_df.shape[1] + 1)
            ]
            training_loss_df.to_csv(f, sep="\t", mode="a")
            f.write("\n\nValidation loss histories:\n")
            validation_loss_df = pd.DataFrame(
                [h["val_loss"] for h in histories], index=range(N_ENSEMBLE_MEMBERS)
            )
            validation_loss_df.index.name = "Member"
            validation_loss_df.columns = [
                f"Epoch {i}" for i in range(1, validation_loss_df.shape[1] + 1)
            ]
            validation_loss_df.to_csv(f, sep="\t", mode="a")

        # %%
        # 7) Commit and push
        try:
            subprocess.run(["git", "pull"], check=True)
            subprocess.run(["git", "add", f"models/*{MODEL_NAME}*"], check=True)

            commit_header = f"Train LSTM MDN Ensemble {VERSION}"
            commit_body = f"Training history:\n" + "\n".join(
                [str(h.history) for h in histories]
            )

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
            save_to=f"results/distributions/{ticker}_lstm_mdn_v{VERSION}_ensemble.svg",
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
    plt.savefig(f"results/lstm_mdn_v{VERSION}_ensemble_mixture_weights.svg")

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
        plt.title(f"LSTM w MDN predictions for {ticker}, {TEST_SET} data")
        plt.xlabel("Date")
        plt.ylabel("LogReturn")
        # Place legend outside of plot
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(f"results/time_series/{ticker}_lstm_mdn_v{VERSION}_ensemble.svg")

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
    df_validation.set_index(["Date", "Symbol"]).to_csv(
        f"predictions/lstm_mdn_predictions{SUFFIX}_v{VERSION}_ensemble.csv"
    )

    # %%
    # Commit predictions
    try:
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(["git", "add", f"predictions/*lstm_mdn*{SUFFIX}*"], check=True)
        commit_header = f"Add predictions for LSTM MDN Ensemble {VERSION}"
        commit_body = f"Loss ({TEST_SET}): {df_validation['NLL'].mean()}"
        subprocess.run(
            ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
        )
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
