# %%
# Define parameters (based on settings)
import subprocess
from settings import LOOKBACK_DAYS, SUFFIX

VERSION = "rv-data-2"
MULTIPLY_MARKET_FEATURES_BY_BETA = False
PI_PENALTY = False
MU_PENALTY = False
SIGMA_PENALTY = False
INCLUDE_MARKET_FEATURES = True
INCLUDE_RETURNS = True
HIDDEN_UNITS = 20
N_MIXTURES = 5
DROPOUT = 0.4
EMBEDDING_DIMENSIONS = 4
MODEL_NAME = f"lstm_mdn_{LOOKBACK_DAYS}_days{SUFFIX}_v{VERSION}"

# %%
# Imports from code shared across models
from shared.mdn import (
    calculate_intervals_vectorized,
    get_mdn_bias_initializer,
    get_mdn_kernel_initializer,
    parse_mdn_output,
    plot_sample_days,
    predict_with_mc_dropout_mdn,
    univariate_mixture_mean_and_var_approx,
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
    StringLookup,
    concatenate,
    Flatten,
    Lambda,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")


# %%
# Load preprocessed data
data = get_lstm_train_test_new(
    multiply_by_beta=MULTIPLY_MARKET_FEATURES_BY_BETA,
    include_returns=INCLUDE_RETURNS,
    include_spx_data=INCLUDE_MARKET_FEATURES,
)

# %%
# Garbage collection
gc.collect()


# %%
def build_lstm_mdn(
    lookback_days,
    num_features: int,
    dropout: float,
    n_mixtures: int,
    hidden_units: int,
    embed_dim: int,
    ticker_ids_dim: int,
):
    """
    Creates a lstm-based encoder for sequences of shape:
      (lookback_days, num_features)
    Then outputs 3*n_mixtures for MDN (univariate).
    """
    # Sequence input (time series)
    seq_input = Input(shape=(lookback_days, num_features), name="seq_input")

    # Ticker input (integer-encoded)
    ticker_input = Input(shape=(1,), dtype="int32", name="ticker_input")
    ticker_embed = Embedding(
        input_dim=ticker_ids_dim, output_dim=embed_dim, name="ticker_embedding"
    )(ticker_input)
    ticker_embed = Flatten()(ticker_embed)  # now shape: (None, embed_dim)

    # Process the sequence with LSTM
    x = LSTM(
        units=hidden_units,
        activation="tanh",
        kernel_regularizer=l2(1e-3),
        name="lstm_layer",
    )(seq_input)
    if dropout > 0:
        x = Dropout(dropout, name="dropout_layer")(x)

    # Combine LSTM output and ticker embedding
    x = concatenate([x, ticker_embed], name="concat_layer")

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

    model = Model(inputs=[seq_input, ticker_input], outputs=mdn_output)
    return model


# %%
# 1) Inspect shapes
print(f"X_train.shape: {data.train.X.shape}, y_train.shape: {data.train.y.shape}")
print(f"Validation set shape: {data.validation.X.shape}, {data.validation.y.shape}")

# %%
# 2) Build model
lstm_mdn_model = build_lstm_mdn(
    lookback_days=LOOKBACK_DAYS,
    num_features=data.train.X.shape[2],
    dropout=DROPOUT,
    n_mixtures=N_MIXTURES,
    hidden_units=HIDDEN_UNITS,
    embed_dim=EMBEDDING_DIMENSIONS,
    ticker_ids_dim=data.ticker_ids_dim,
)

# %%
# 4) Load existing model if it exists
load_best_val_loss_model = True
best_val_suffix = "_best_val_loss" if load_best_val_loss_model else ""
model_fname = f"models/{MODEL_NAME}{best_val_suffix}.keras"
already_trained = False
if os.path.exists(model_fname):
    mdn_kernel_initializer = get_mdn_kernel_initializer(N_MIXTURES)
    mdn_bias_initializer = get_mdn_bias_initializer(N_MIXTURES)
    lstm_mdn_model = tf.keras.models.load_model(
        model_fname,
        custom_objects={
            "loss_fn": mean_mdn_crps_tf(N_MIXTURES, PI_PENALTY),
            "mdn_kernel_initializer": mdn_kernel_initializer,
            "mdn_bias_initializer": mdn_bias_initializer,
        },
    )
    # Re-compile
    lstm_mdn_model.compile(
        optimizer=Adam(learning_rate=1e-4, weight_decay=1e-2),
        loss=mean_mdn_crps_tf(N_MIXTURES, PI_PENALTY),
    )
    print("Loaded pre-trained model from disk.")
    already_trained = True

# %%
# 5) Train
val_loss = (
    lstm_mdn_model.evaluate(
        [data.validation.X, data.validation_ticker_ids], data.validation.y, verbose=0
    )
    if already_trained
    else np.inf
)
histories = []
weight_per_ticker = pd.Series(index=np.unique(data.train.tickers)).fillna(1)


# %%
def calculate_weight_per_ticker() -> pd.Series:
    """
    Use the training loss of each ticker to give more weights to series with poor loss.
    Also calculate quantiles for some of the stocks with the worst loss and give even
    higher weight to samples where the coverage does not match the confidence level.
    """
    print("Collecting garbage...")
    gc.collect()  # We need a lot of memory to make a prediction on the entire training set
    ## Calculate loss per ticker
    print("Calculating loss per ticker...")
    y_train_pred = lstm_mdn_model.predict([data.train.X, data.train_ticker_ids])
    nlls = mdn_nll_numpy(N_MIXTURES)(data.train.y, y_train_pred)
    train_dates = np.array(data.train.dates)
    train_tickers = np.array(data.train.tickers)
    loss_df = pd.DataFrame(
        {
            "Date": train_dates,
            "Symbol": train_tickers,
            "loss": nlls,
        }
    )
    loss_per_ticker = loss_df.groupby("Symbol")["loss"].mean().sort_values()
    try:
        display(loss_per_ticker)
    except:
        print(loss_per_ticker)

    ## Calculate weights based on loss
    print("Calculating weights based on loss...")
    weight_best_nll = 0.1
    weight_worst_nll = 1
    best_nll = loss_per_ticker.min()
    underperformance_per_ticker = loss_per_ticker - best_nll
    underperformance_per_ticker = (
        underperformance_per_ticker / underperformance_per_ticker.max()
    )
    weight_per_ticker = (
        weight_best_nll
        + (weight_worst_nll - weight_best_nll) * underperformance_per_ticker
    )
    weight_per_ticker = weight_per_ticker.clip(0, 1)
    n_tickers = len(weight_per_ticker)

    ## Add extra weight to the worst performing tickers with the worst coverage
    print("Calculating coverage for worst tickers...")
    worst_tickers = loss_per_ticker.tail(int(n_tickers / 10)).index
    worst_ticker_mask = np.isin(np.array(data.train.tickers), worst_tickers)
    filtered_y_train_pred = y_train_pred[worst_ticker_mask]
    pi_pred, mu_pred, sigma_pred = parse_mdn_output(filtered_y_train_pred, N_MIXTURES)
    confidence_levels = [0.95, 0.99]
    intervals = calculate_intervals_vectorized(
        pi_pred, mu_pred, sigma_pred, confidence_levels
    )
    actuals = data.train.y[worst_ticker_mask]
    within_bounds = {
        cl: np.logical_and(
            actuals > intervals[:, i, 0],
            actuals < intervals[:, i, 1],
        )
        for i, cl in enumerate(confidence_levels)
    }
    within_bounds_df = pd.DataFrame(
        {
            "Date": train_dates[worst_ticker_mask],
            "Symbol": train_tickers[worst_ticker_mask],
            "WithinBounds-0.95": within_bounds[0.95],
            "WithinBounds-0.99": within_bounds[0.99],
        }
    ).set_index(["Date", "Symbol"])
    coverage = within_bounds_df.groupby("Symbol").mean()
    coverage_miss = pd.DataFrame(
        {
            "CoverageMiss-0.95": np.abs(0.95 - coverage["WithinBounds-0.95"]),
            "CoverageMiss-0.99": np.abs(0.99 - coverage["WithinBounds-0.99"]),
        },
        index=coverage.index,
    )
    mean_miss = coverage_miss.mean(axis=1)
    worst_miss = mean_miss.max()
    extra_weight_worst_miss = 1
    picp_based_weight = extra_weight_worst_miss * mean_miss / worst_miss

    ## Combine weights
    weight_per_ticker.loc[worst_tickers] += picp_based_weight

    ## Plot weights for inspection if we are running in a notebook
    try:
        display(weight_per_ticker)
        weight_per_ticker.hist(bins=50)
        plt.title("Distribution of weights per ticker")
        plt.show()
    except:
        print(weight_per_ticker)

    return weight_per_ticker


# %%
if already_trained:
    weight_per_ticker = calculate_weight_per_ticker()

# %%
# Train until validation loss stops decreasing
increases_since_best = 0
max_increases_since_best = 0
best_model_weights = lstm_mdn_model.get_weights()
best_val_loss = val_loss
while True:
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=0,  # number of epochs with no improvement to wait
        restore_best_weights=True,
    )

    print("Aligning ticker weights...")
    ticker_weights = weight_per_ticker.loc[data.train.tickers].values
    print("Compiling model...", flush=True)
    lstm_mdn_model.compile(
        optimizer=Adam(learning_rate=1e-4, weight_decay=1e-2),
        loss=mdn_nll_tf(N_MIXTURES, PI_PENALTY),
    )
    print("Fitting model...", flush=True)
    history = lstm_mdn_model.fit(
        [data.train.X, data.train_ticker_ids],
        data.train.y,
        epochs=50,
        batch_size=32,
        verbose=1,
        validation_data=(
            [data.validation.X, data.validation_ticker_ids],
            data.validation.y,
        ),
        callbacks=[early_stop],
        sample_weight=ticker_weights,
    )
    val_loss = np.array(history.history["val_loss"]).min()
    if val_loss >= best_val_loss:
        increases_since_best += 1
        if increases_since_best >= max_increases_since_best:
            print(
                f"Validation loss has not decreased for {max_increases_since_best} iterations. "
                "Stopping training. \n"
                "If you want to restore the best model, type:\n"
                "lstm_mdn_model.set_weights(best_model_weights)"
            )
            break
    else:
        best_model_weights = lstm_mdn_model.get_weights()
        best_val_loss = val_loss
        increases_since_best = 0
    histories.append(history)
    weight_per_ticker = calculate_weight_per_ticker()

# %%
# Train one epoch with CRPS loss
print("Training one epoch with CRPS loss...")
lstm_mdn_model.compile(
    optimizer=Adam(learning_rate=1e-7, weight_decay=1e-7),
    loss=mdn_crps_tf(N_MIXTURES, PI_PENALTY, MU_PENALTY, SIGMA_PENALTY, npts=64),
)
history = lstm_mdn_model.fit(
    [data.train.X, data.train_ticker_ids],
    data.train.y,
    epochs=1,
    batch_size=32,
    verbose=1,
    validation_data=(
        [data.validation.X, data.validation_ticker_ids],
        data.validation.y,
    ),
)
histories.append(history)

# %%
# 6) Save both current model and the model with the best validation loss
lstm_mdn_model.save(model_fname)
best_model = build_lstm_mdn(
    lookback_days=LOOKBACK_DAYS,
    num_features=data.train.X.shape[2],
    dropout=DROPOUT,
    n_mixtures=N_MIXTURES,
    hidden_units=HIDDEN_UNITS,
    embed_dim=EMBEDDING_DIMENSIONS,
    ticker_ids_dim=data.ticker_ids_dim,
)
best_model.set_weights(best_model_weights)
best_model.save(f"models/{MODEL_NAME}_best_val_loss.keras")

# %%
# 7) Commit and push
try:
    subprocess.run(["git", "pull"], check=True)
    subprocess.run(["git", "add", f"models/*{MODEL_NAME}*"], check=True)

    commit_header = f"Train LSTM MDN {VERSION}"
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
y_pred_mdn = lstm_mdn_model.predict([data.validation.X, data.validation_ticker_ids])
pi_pred, mu_pred, sigma_pred = parse_mdn_output(y_pred_mdn, N_MIXTURES)

# %%
# 9) Plot 10 charts with the distributions for 10 random days
example_tickers = ["AAPL", "WMT", "GS"]
for ticker in example_tickers:
    s = data.validation.sets[ticker]
    from_idx, to_idx = data.validation.get_range(ticker)
    plot_sample_days(
        s.y_dates,
        s.y,
        pi_pred[from_idx:to_idx],
        mu_pred[from_idx:to_idx],
        sigma_pred[from_idx:to_idx],
        N_MIXTURES,
        ticker=ticker,
    )

# %%
# 10) Plot weights over time to show how they change
fig, axes = plt.subplots(
    nrows=len(example_tickers), figsize=(18, len(example_tickers) * 9)
)

# Dictionary to store union of legend entries
legend_dict = {}

for ax, ticker in zip(axes, example_tickers):
    s = data.validation.sets[ticker]
    from_idx, to_idx = data.validation.get_range(ticker)
    pi_pred_ticker = pi_pred[from_idx:to_idx]
    for j in range(N_MIXTURES):
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
plt.show()


# %%
# 11) Calculate intervals for 67%, 95%, 97.5% and 99% confidence levels
confidence_levels = [0.67, 0.95, 0.99]
intervals = calculate_intervals_vectorized(
    pi_pred, mu_pred, sigma_pred, confidence_levels
)

# %%
# 12) Plot time series with mean, volatility and actual returns for last X days
days = 150
shift = 1
mean = (pi_pred * mu_pred).numpy().sum(axis=1)
for ticker in example_tickers:
    from_idx, to_idx = data.validation.get_range(ticker)
    ticker_mean = mean[from_idx:to_idx]
    filtered_mean = ticker_mean[-days - shift : -shift]
    ticker_intervals = intervals[from_idx:to_idx]
    filtered_intervals = ticker_intervals[-days - shift : -shift]
    s = data.validation.sets[ticker]
    dates = s.y_dates[-days - shift : -shift]
    actual_return = s.y[-days - shift : -shift]

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
    plt.plot(dates, median, label="Median", color="blue")
    for i, cl in enumerate(confidence_levels):
        if cl == 0:
            continue
        plt.fill_between(
            dates,
            filtered_intervals[:, i, 0],
            filtered_intervals[:, i, 1],
            color="blue",
            alpha=0.7 - i * 0.1,
            label=f"{int(100*cl)}% Interval",
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
    plt.title(f"LSTM w MDN predictions for {ticker}, {days} days")
    plt.xlabel("Date")
    plt.ylabel("LogReturn")
    plt.legend()
    plt.show()

# %%
# 13) Make data frame for signle pass predictions
df_validation = pd.DataFrame(
    np.vstack([data.validation.dates, data.validation.tickers]).T,
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
df_validation["NLL"] = mdn_nll_numpy(N_MIXTURES)(data.validation.y, y_pred_mdn)

# %%
# Calculate CRPS
crps = mdn_crps_tf(N_MIXTURES)
df_validation["CRPS"] = crps(data.validation.y, y_pred_mdn)

# %%
# Add confidence intervals
for i, cl in enumerate(confidence_levels):
    df_validation[f"LB_{int(100*cl)}"] = intervals[:, i, 0]
    df_validation[f"UB_{int(100*cl)}"] = intervals[:, i, 1]

# %%
# Save
df_validation.set_index(["Date", "Symbol"]).to_csv(
    f"predictions/lstm_mdn_predictions{SUFFIX}_v{VERSION}.csv"
)

# %%
# Commit predictions
try:
    subprocess.run(["git", "pull"], check=True)
    subprocess.run(["git", "add", f"predictions/*lstm_mdn*{SUFFIX}*"], check=True)
    commit_header = f"Add predictions for LSTM MDN {VERSION}"
    commit_body = f"Validation loss: {val_loss}"
    subprocess.run(
        ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
    )
    subprocess.run(["git", "push"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Git command failed: {e}")

# %%
# 9) MC Dropout predictions
mc_results = predict_with_mc_dropout_mdn(
    lstm_mdn_model, data.validation.y, T=100, n_mixtures=N_MIXTURES
)

df_validation["Mean_MC"] = mc_results["expected_returns"]
df_validation["Vol_MC"] = mc_results["volatility_estimates"]
df_validation["Epistemic_Unc_Vol"] = mc_results[
    "epistemic_uncertainty_volatility_estimates"
]
df_validation["Epistemic_Unc_Mean"] = mc_results[
    "epistemic_uncertainty_expected_returns"
]

df_validation.to_csv(f"predictions/lstm_mdn_mc_predictions{SUFFIX}_days_v{VERSION}.csv")
