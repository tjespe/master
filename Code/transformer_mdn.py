# %%
# Define parameters (based on settings)
import subprocess
from typing import Optional
from shared.conf_levels import format_cl
from settings import LOOKBACK_DAYS, SUFFIX

VERSION = "ffnn"
MULTIPLY_MARKET_FEATURES_BY_BETA = False
PI_PENALTY = False
MU_PENALTY = False
SIGMA_PENALTY = False
INCLUDE_MARKET_FEATURES = True
INCLUDE_RETURNS = True
INCLUDE_FNG = False
INCLUDE_RETURNS = True
INCLUDE_INDUSTRY = False
INCLUDE_GARCH = True
INCLUDE_BETA = True
INCLUDE_OTHERS = True
INCLUDE_TICKERS = True
D_MODEL = 64
HIDDEN_UNITS_FF = 256
N_MIXTURES = 10
DROPOUT = 0.5
NUM_ENCODERS = 2
NUM_HEADS = 8
EMBEDDING_DIMENSIONS = 4
MODEL_NAME = f"lstm_mdn_{LOOKBACK_DAYS}_days{SUFFIX}_v{VERSION}"

# %%
# Settings for training
PATIENCE = 3  # Early stopping patience
REWEIGHT_WORST_PERFORMERS = True
REWEIGHT_WORST_PERFORMERS_EPOCHS = 0

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
    LayerNormalization,
    Embedding,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    Concatenate,
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
    include_others=INCLUDE_OTHERS,
    include_beta=INCLUDE_BETA,
    include_fng=INCLUDE_FNG,
    include_garch=INCLUDE_GARCH,
    include_industry=INCLUDE_INDUSTRY,
)

# %%
# Garbage collection
gc.collect()


# %%
def transformer_encoder(inputs, d_model, num_heads, ff_dim, rate):
    """
    A single block of Transformer encoder:
      1) MHA + residual + LayerNorm
      2) FFN + residual + LayerNorm
    """
    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(
        inputs, inputs
    )
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Feed-forward
    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dropout(rate)(ffn)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(rate)(ffn)

    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2


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
    ticker_inputs = Input(shape=(), dtype=tf.int32, name="ticker_inputs")  # Scalar ID

    # Ticker embedding
    ticker_embedding = Embedding(
        input_dim=ticker_ids_dim,  # Number of unique tickers
        output_dim=EMBEDDING_DIMENSIONS,  # Size of the embedding vector
        name="ticker_embedding",
    )(ticker_inputs)

    # Expand and concatenate with input features
    ticker_embedding_expanded = tf.expand_dims(
        ticker_embedding, axis=1
    )  # (batch, 1, embed_dim)
    ticker_embedding_expanded = tf.repeat(
        ticker_embedding_expanded, LOOKBACK_DAYS, axis=1
    )  # (batch, lookback_days, embed_dim)

    x = Concatenate(axis=-1)([feature_inputs, ticker_embedding_expanded])

    # Project inputs to d_model
    x = Dense(D_MODEL, activation=None)(x)

    # Stack multiple Transformer encoder blocks
    for _ in range(NUM_ENCODERS):
        x = transformer_encoder(x, D_MODEL, NUM_HEADS, HIDDEN_UNITS_FF, rate=DROPOUT)

    # Global average pooling (or take last time step)
    x = GlobalAveragePooling1D()(x)

    # Create initializers for MDN output layer
    mdn_kernel_init = get_mdn_kernel_initializer(N_MIXTURES)
    mdn_bias_init = get_mdn_bias_initializer(N_MIXTURES)

    # Output layer: 3*n_mixtures => [logits_pi, mu, log_sigma]
    mdn_output = Dense(
        3 * N_MIXTURES,
        activation=None,
        name="mdn_output",
        kernel_initializer=mdn_kernel_init,
        bias_initializer=mdn_bias_init,
    )(x)

    model = Model(inputs=[feature_inputs, ticker_inputs], outputs=mdn_output)
    return model


# %%
# 4) Load existing model if it exists
load_best_val_loss_model = False
best_val_suffix = "_best_val_loss" if load_best_val_loss_model else ""
model_fname = f"models/{MODEL_NAME}{best_val_suffix}.keras"
if os.path.exists(model_fname):
    mdn_kernel_initializer = get_mdn_kernel_initializer(N_MIXTURES)
    mdn_bias_initializer = get_mdn_bias_initializer(N_MIXTURES)
    transformer_model = tf.keras.models.load_model(
        model_fname,
        custom_objects={
            "loss_fn": mean_mdn_crps_tf(N_MIXTURES, PI_PENALTY),
            "mdn_kernel_initializer": mdn_kernel_initializer,
            "mdn_bias_initializer": mdn_bias_initializer,
        },
    )
    # Re-compile
    transformer_model.compile(
        optimizer=Adam(learning_rate=1e-4, weight_decay=1e-2),
        loss=mean_mdn_crps_tf(N_MIXTURES, PI_PENALTY),
    )
    print("Loaded pre-trained model from disk.")
    already_trained = True

# %%
# 5) Train
val_loss = (
    transformer_model.evaluate(
        [data.validation.X, data.validation_ticker_ids] if INCLUDE_TICKERS else None,
        data.validation.y,
        verbose=0,
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
    y_train_pred = transformer_model.predict(
        [data.train.X, data.train_ticker_ids] if INCLUDE_TICKERS else data.train.X
    )
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
    confidence_levels = [0.67, 0.90, 0.95, 0.99]
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
best_model_weights = transformer_model.get_weights()
best_val_loss = val_loss
while True:
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,  # number of epochs with no improvement to wait
        restore_best_weights=True,
    )

    print("Aligning ticker weights...")
    ticker_weights = weight_per_ticker.loc[data.train.tickers].values
    print("Compiling model...", flush=True)
    transformer_model.compile(
        optimizer=Adam(learning_rate=1e-4, weight_decay=1e-2),
        loss=mdn_nll_tf(N_MIXTURES, PI_PENALTY),
    )
    print("Fitting model...", flush=True)
    history = transformer_model.fit(
        [data.train.X, data.train_ticker_ids] if INCLUDE_TICKERS else data.train.X,
        data.train.y,
        epochs=50,
        batch_size=32,
        verbose=1,
        validation_data=(
            (
                [data.validation.X, data.validation_ticker_ids]
                if INCLUDE_TICKERS
                else data.validation.X
            ),
            data.validation.y,
        ),
        callbacks=[early_stop],
        sample_weight=ticker_weights,
    )
    val_loss = np.array(history.history["val_loss"]).min()
    if not REWEIGHT_WORST_PERFORMERS:
        break
    if val_loss >= best_val_loss:
        increases_since_best += 1
        if increases_since_best >= REWEIGHT_WORST_PERFORMERS_EPOCHS:
            print(
                f"Validation loss has not decreased for {REWEIGHT_WORST_PERFORMERS_EPOCHS} iterations. "
                "Stopping training. \n"
                # "If you want to restore the best model, type:\n"
                # "lstm_mdn_model.set_weights(best_model_weights)"
            )
            transformer_model.set_weights(best_model_weights)
            break
    else:
        best_model_weights = transformer_model.get_weights()
        best_val_loss = val_loss
        increases_since_best = 0
    histories.append(history)
    weight_per_ticker = calculate_weight_per_ticker()

# %%
# 6) Save both current model and the model with the best validation loss
transformer_model.save(model_fname)

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
y_pred_mdn = transformer_model.predict(
    [data.validation.X, data.validation_ticker_ids]
    if INCLUDE_TICKERS
    else data.validation.X
)
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
        save_to=f"results/distributions/{ticker}_lstm_mdn_v{VERSION}.svg",
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
plt.savefig(f"results/lstm_mdn_v{VERSION}_mixture_weights.svg")
plt.show()


# %%
# 11) Calculate intervals for 67%, 95%, 97.5% and 99% confidence levels
confidence_levels = [0.67, 0.90, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999]
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
    plt.savefig(f"results/time_series/{ticker}_lstm_mdn_v{VERSION}.svg")
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
