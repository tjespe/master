# %%
# Define parameters (based on settings)
import subprocess
from settings import LOOKBACK_DAYS, SUFFIX

VERSION = "pireg"
MULTIPLY_MARKET_FEATURES_BY_BETA = False
PI_PENALTY = True
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
from shared.numerical_mixture_moments import numerical_mixture_moments
from shared.loss import mdn_loss_numpy, mdn_loss_tf
from shared.crps import crps_mdn_numpy
from shared.processing import get_lstm_train_test_new

# %%
# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LSTM,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")


# %%
# Load preprocessed data
data = get_lstm_train_test_new(multiply_by_beta=MULTIPLY_MARKET_FEATURES_BY_BETA)


# %%
def build_lstm_mdn(
    lookback_days,
    num_features: int,
    dropout=0.1,
    n_mixtures=5,
    hidden_units=10,
):
    """
    Creates a lstm-based encoder for sequences of shape:
      (lookback_days, num_features)
    Then outputs 3*n_mixtures for MDN (univariate).
    """
    inputs = Input(shape=(lookback_days, num_features))

    # Add LSTM layer
    x = LSTM(
        units=hidden_units,
        activation="tanh",
        kernel_regularizer=l2(1e-4),
    )(inputs)

    # Add dropout
    if dropout > 0:
        x = Dropout(dropout)(x)

    # Create the custom kernel initializer for the Dense layer.
    mdn_kernel_init = get_mdn_kernel_initializer(n_mixtures)
    mdn_bias_init = get_mdn_bias_initializer(n_mixtures)

    # Output layer: 3*n_mixtures => [logits_pi, mu, log_sigma]
    mdn_output = Dense(
        3 * n_mixtures,
        activation=None,
        kernel_initializer=mdn_kernel_init,
        bias_initializer=mdn_bias_init,
        name="mdn_output",
    )(x)

    model = Model(inputs=inputs, outputs=mdn_output)
    return model


# %%
# 1) Inspect shapes
print(f"X_train.shape: {data.X_train.shape}, y_train.shape: {data.y_train.shape}")
print(f"Validation set shape: {data.X_val_combined.shape}, {data.y_val_combined.shape}")

# %%
# 2) Build model
N_MIXTURES = 100
lstm_mdn_model = build_lstm_mdn(
    lookback_days=LOOKBACK_DAYS,
    num_features=data.X_train.shape[2],
    dropout=0.1,
    n_mixtures=N_MIXTURES,
    hidden_units=20,
)

# %%
# 4) Load existing model if it exists
model_fname = f"models/{MODEL_NAME}.keras"
if os.path.exists(model_fname):
    mdn_kernel_initializer = get_mdn_kernel_initializer(N_MIXTURES)
    mdn_bias_initializer = get_mdn_bias_initializer(N_MIXTURES)
    lstm_mdn_model = tf.keras.models.load_model(
        model_fname,
        custom_objects={
            "loss_fn": mdn_loss_tf(N_MIXTURES, PI_PENALTY),
            "mdn_kernel_initializer": mdn_kernel_initializer,
            "mdn_bias_initializer": mdn_bias_initializer,
        },
    )
    # Re-compile
    lstm_mdn_model.compile(
        optimizer=Adam(learning_rate=1e-3), loss=mdn_loss_tf(N_MIXTURES, PI_PENALTY)
    )
    print("Loaded pre-trained model from disk.")

# %%
# 5) Train
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,  # number of epochs with no improvement to wait
    restore_best_weights=True,
)

# Start with one learning rate, then reduce
# lstm_mdn_model.compile(optimizer=Adam(learning_rate=1e-3), loss=mdn_loss_tf(N_MIXTURES))
# history = lstm_mdn_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# %%
# Reduce learning rate
lstm_mdn_model.compile(
    optimizer=Adam(learning_rate=1e-4, weight_decay=1e-2),
    loss=mdn_loss_tf(N_MIXTURES, PI_PENALTY),
)
history = lstm_mdn_model.fit(
    data.X_train,
    data.y_train,
    epochs=100,
    batch_size=32,
    verbose=1,
    validation_data=(data.X_val_combined, data.y_val_combined),
    callbacks=[early_stop],
)

# %%
# 6) Save
lstm_mdn_model.save(model_fname)

# %%
# 7) Commit and push
try:
    subprocess.run(["git", "pull"], check=True)
    subprocess.run(["git", "add", f"models/*{MODEL_NAME}*"], check=True)

    commit_header = f"Train LSTM MDN {VERSION}"
    commit_body = f"Training history: {history.history}"

    subprocess.run(
        ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
    )
    subprocess.run(["git", "push"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Git command failed: {e}")

# %%
# 8) Single-pass predictions
y_pred_mdn = lstm_mdn_model.predict(data.X_val_combined)  # shape: (batch, 3*N_MIXTURES)
pi_pred, mu_pred, sigma_pred = parse_mdn_output(y_pred_mdn, N_MIXTURES)

# %%
# 9) Plot 10 charts with the distributions for 10 random days
example_tickers = ["GOOG", "AON", "WMT"]
for ticker in example_tickers:
    s = data.validation_sets[ticker]
    from_idx, to_idx = data.get_validation_range(ticker)
    plot_sample_days(
        s.df,
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
    s = data.validation_sets[ticker]
    dates = s.df.index.get_level_values("Date")
    from_idx, to_idx = data.get_validation_range(ticker)
    pi_pred_ticker = pi_pred[from_idx:to_idx]
    for j in range(N_MIXTURES):
        mean_over_time = np.mean(pi_pred_ticker[:, j], axis=0)
        if mean_over_time < 0.01:
            continue
        (line,) = ax.plot(dates, pi_pred_ticker[:, j], label=f"$\pi_{{{j}}}$")
        # Only add new labels
        if f"$\pi_{{{j}}}$" not in legend_dict:
            legend_dict[f"$\pi_{{{j}}}$"] = line
    ax.set_yticklabels(["{:.0f}%".format(x * 100) for x in ax.get_yticks()])
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
    filtered_df = data.validation_sets[ticker].df.iloc[-days - shift : -shift]
    from_idx, to_idx = data.get_validation_range(ticker)
    ticker_mean = mean[from_idx:to_idx]
    filtered_mean = ticker_mean[-days - shift : -shift]
    ticker_intervals = intervals[from_idx:to_idx]
    filtered_intervals = ticker_intervals[-days - shift : -shift]
    dates = filtered_df.index.get_level_values("Date")

    plt.figure(figsize=(12, 6))
    plt.plot(
        dates,
        filtered_df["LogReturn"],
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
        filtered_df["LogReturn"].mean(),
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
    np.vstack([data.validation_dates, data.validation_tickers]).T,
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
# Calculate NLL
df_validation["NLL"] = mdn_loss_numpy(N_MIXTURES)(data.y_val_combined, y_pred_mdn)

# %%
# Calculate CRPS (slow!)
# crps = crps_mdn_numpy(N_MIXTURES)
# df_validation["CRPS"] = crps(data.y_val_combined, y_pred_mdn)

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
# 9) MC Dropout predictions
mc_results = predict_with_mc_dropout_mdn(
    lstm_mdn_model, data.y_val_combined, T=100, n_mixtures=N_MIXTURES
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
