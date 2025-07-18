# %%
# Define parameters (based on settings)
import subprocess
from typing import Optional
from shared.conf_levels import format_cl
from settings import LOOKBACK_DAYS, SUFFIX

VERSION = "rv-5-only"
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
INCLUDE_10_DAY_IVOL = False
INCLUDE_30_DAY_IVOL = False
INCLUDE_1MIN_RV = False
INCLUDE_5MIN_RV = True
HIDDEN_UNITS = 61
N_MIXTURES = 26
DROPOUT = 0.0
L2_REG = 0.00015
NUM_HIDDEN_LAYERS = 0
EMBEDDING_DIMENSIONS = None
MODEL_NAME = f"lstm_mdn_{LOOKBACK_DAYS}_days{SUFFIX}_v{VERSION}"

# %%
# Settings for training
PATIENCE = 3  # Early stopping patience
WEIGHT_DECAY = 0.05  # from optuna
LEARNING_RATE = 0.00015  # from optuna
BATCH_SIZE = 39

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
    mdn_mean_and_var,
)
from shared.loss import (
    ece_mdn,
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
    Concatenate,
    RepeatVector,
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

    # MDN output layer: 3 * n_mixtures (for [logits_pi, mu, log_sigma])
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
# 1) Inspect shapes
print(f"X_train.shape: {data.train.X.shape}, y_train.shape: {data.train.y.shape}")
print(f"Validation set shape: {data.validation.X.shape}, {data.validation.y.shape}")

# %%
# 2) Build model
lstm_mdn_model = build_lstm_mdn(
    num_features=data.train.X.shape[2],
    ticker_ids_dim=data.ticker_ids_dim if INCLUDE_TICKERS else None,
)

# %%
# 4) Load existing model if it exists
load_best_val_loss_model = False
best_val_suffix = "_best_val_loss" if load_best_val_loss_model else ""
model_fname = f"models/{MODEL_NAME}{best_val_suffix}.keras"
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
        optimizer=Adam(learning_rate=1e-4, weight_decay=WEIGHT_DECAY),
        loss=mean_mdn_crps_tf(N_MIXTURES, PI_PENALTY),
    )
    print("Loaded pre-trained model from disk.")

# %%
# 5) Train model
# Train until validation loss stops decreasing
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,  # number of epochs with no improvement to wait
    restore_best_weights=True,
)
print("Compiling model...", flush=True)
lstm_mdn_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    loss=mdn_nll_tf(N_MIXTURES, PI_PENALTY),
)
print("Fitting model...", flush=True)
history = lstm_mdn_model.fit(
    [data.train.X, data.train_ticker_ids] if INCLUDE_TICKERS else data.train.X,
    data.train.y,
    epochs=50,
    batch_size=BATCH_SIZE,
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
)
val_loss = np.array(history.history["val_loss"]).min()
# %%
# 6) Save model
lstm_mdn_model.save(model_fname)

# %%
# 7) Commit and push
try:
    subprocess.run(["git", "pull"], check=True)
    subprocess.run(["git", "add", f"models/*{MODEL_NAME}*"], check=True)

    commit_header = f"Train LSTM MDN {VERSION}"
    commit_body = f"Training history:\n{history.history}"

    subprocess.run(
        ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
    )
    subprocess.run(["git", "push"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Git command failed: {e}")

# %%
# 8) Single-pass predictions
y_pred_mdn = lstm_mdn_model.predict(
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
# 13) Make data frame for signle pass predictions
df_validation = pd.DataFrame(
    np.vstack([data.validation.dates, data.validation.tickers]).T,
    columns=["Date", "Symbol"],
)
# %%
# For comparison to other models, compute mixture means & variances
uni_mixture_mean_sp, uni_mixture_var_sp = mdn_mean_and_var(pi_pred, mu_pred, sigma_pred)
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
# Calculate ECE
ece = ece_mdn(N_MIXTURES, data.validation.y, y_pred_mdn)
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

# %%
