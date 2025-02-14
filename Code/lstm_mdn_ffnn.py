# %%
# Define parameters (based on settings)
import subprocess
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    TEST_ASSET,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)

VERSION = 2
MODEL_NAME = f"lstm_ffnn_mdn_{LOOKBACK_DAYS}_days{SUFFIX}_v{VERSION}"

# %%
# Imports from code shared across models
from shared.mdn import (
    calculate_intervals,
    get_mdn_kernel_initializer,
    get_mdn_bias_initializer,
    parse_mdn_output,
    plot_sample_days,
    univariate_mixture_mean_and_var_approx,
)
from shared.numerical_mixture_moments import numerical_mixture_moments
from shared.loss import mdn_loss_numpy, mdn_loss_tf
from shared.crps import crps_mdn_numpy
from shared.processing import get_lstm_train_test

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

warnings.filterwarnings("ignore")


# %%
# Load preprocessed data
df, X_train, X_test, y_train, y_test = get_lstm_train_test(include_log_returns=True)


# %%
N_MIXTURES = 30


def build_model(
    lookback_days,
    num_features: int,
):
    """
    Creates a lstm-based encoder for sequences of shape:
      (lookback_days, num_features)
    Then outputs 3*n_mixtures for MDN (univariate).
    """
    inputs = Input(shape=(lookback_days, num_features))

    # Add LSTM layer
    x = LSTM(units=30, activation="tanh")(inputs)

    # Add dropout
    x = Dropout(0.1)(x)

    # Add feed-forward layers with dropouts
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)

    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)

    # Create the custom kernel initializer for the Dense layer.
    mdn_kernel_init = get_mdn_kernel_initializer(N_MIXTURES)
    mdn_bias_init = get_mdn_bias_initializer(N_MIXTURES)

    # Output layer: 3*n_mixtures => [logits_pi, mu, log_sigma]
    mdn_output = Dense(
        3 * N_MIXTURES,
        activation=None,
        kernel_initializer=mdn_kernel_init,
        bias_initializer=mdn_bias_init,
        name="mdn_output",
    )(x)

    model = Model(inputs=inputs, outputs=mdn_output)
    return model


# %%
# 1) Inspect shapes
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape},   y_test.shape: {y_test.shape}")

# %%
# 2) Build model
lstm_mdn_model = build_model(
    lookback_days=LOOKBACK_DAYS,
    num_features=X_train.shape[2],  # 2 features in our example
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
            "loss_fn": mdn_loss_tf(N_MIXTURES),
            "mdn_kernel_initializer": mdn_kernel_initializer,
            "mdn_bias_initializer": mdn_bias_initializer,
        },
    )
    # Re-compile
    lstm_mdn_model.compile(
        optimizer=Adam(learning_rate=1e-3), loss=mdn_loss_tf(N_MIXTURES)
    )
    print("Loaded pre-trained model from disk.")

# %%
# 5) Train
# Start with one learning rate, then reduce
lstm_mdn_model.compile(optimizer=Adam(learning_rate=1e-3), loss=mdn_loss_tf(N_MIXTURES))
history = lstm_mdn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# %%
# Reduce learning rate
# lstm_mdn_model.compile(optimizer=Adam(learning_rate=1e-4), loss=mdn_loss_tf(N_MIXTURES))
# history = lstm_mdn_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# %%
# 6) Save
lstm_mdn_model.save(model_fname)

# %%
# 7) Commit and push
try:
    subprocess.run(["git", "pull"], check=True)
    subprocess.run(["git", "add", f"models/*{MODEL_NAME}*"], check=True)

    commit_header = f"Train {MODEL_NAME}"
    commit_body = f"Training history: {history.history}"

    subprocess.run(
        ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
    )
    subprocess.run(["git", "push"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Git command failed: {e}")

# %%
# 8) Single-pass predictions
y_pred_mdn = lstm_mdn_model.predict(X_test)  # shape: (batch, 3*N_MIXTURES)
pi_pred, mu_pred, sigma_pred = parse_mdn_output(y_pred_mdn, N_MIXTURES)

# %%
# 9) Plot 10 charts with the distributions for 10 random days
plot_sample_days(
    df,
    y_test,
    pi_pred,
    mu_pred,
    sigma_pred,
    N_MIXTURES,
    f"results/{MODEL_NAME}_sample_days.svg",
)

# %%
# 10) Plot weights over time to show how they change
plt.figure(figsize=(18, 8))
dates = (
    df.xs(TEST_ASSET, level="Symbol")
    .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    .index.get_level_values("Date")
)
for j in range(N_MIXTURES):
    mean_over_time = np.mean(pi_pred[:, j], axis=0)
    if mean_over_time < 0.01:
        continue
    plt.plot(dates, pi_pred[:, j], label=f"$\pi_{{{j}}}$")
plt.gca().set_yticklabels(["{:.0f}%".format(x * 100) for x in plt.gca().get_yticks()])
plt.title(f"Evolution of Mixture Weights for {TEST_ASSET}")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.legend()
plt.show()


# %%
# 11) Calculate intervals for 67%, 95%, 97.5% and 99% confidence levels
confidence_levels = [0, 0.5, 0.67, 0.90, 0.95, 0.975, 0.99]
intervals = calculate_intervals(pi_pred, mu_pred, sigma_pred, confidence_levels)

# %%
# 12) Plot time series with mean, volatility and actual returns for last X days
days = 150
shift = 500
filtered_df = (
    df.xs(TEST_ASSET, level="Symbol")
    .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    .tail(days + shift)
    .head(days)
)
filtered_intervals = intervals[-days - shift : -shift]
mean = (pi_pred * mu_pred).numpy().sum(axis=1)
filtered_mean = mean[-days - shift : -shift]

plt.figure(figsize=(12, 6))
plt.plot(
    filtered_df.index,
    filtered_df["LogReturn"],
    label="Actual Returns",
    color="black",
    alpha=0.5,
)
plt.plot(filtered_df.index, filtered_mean, label="Predicted Mean", color="red")
median = filtered_intervals[:, 0, 0]
plt.plot(filtered_df.index, median, label="Median", color="blue")
for i, cl in enumerate(confidence_levels):
    if cl == 0:
        continue
    plt.fill_between(
        filtered_df.index,
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
plt.gca().set_yticklabels(["{:.1f}%".format(x * 100) for x in plt.gca().get_yticks()])
plt.title(f"LSTM w MDN predictions for {TEST_ASSET}, {days} days")
plt.xlabel("Date")
plt.ylabel("LogReturn")
plt.legend()
plt.show()

# %%
# 13) Store single-pass predictions
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
# For reference, compute mixture means & variances
uni_mixture_mean_sp, uni_mixture_var_sp = univariate_mixture_mean_and_var_approx(
    pi_pred, mu_pred, sigma_pred
)
uni_mixture_mean_sp = uni_mixture_mean_sp.numpy()
uni_mixture_std_sp = np.sqrt(uni_mixture_var_sp.numpy())
df_validation["Mean_SP"] = uni_mixture_mean_sp
df_validation["Vol_SP"] = uni_mixture_std_sp
df_validation["NLL"] = mdn_loss_numpy(N_MIXTURES)(y_test, y_pred_mdn)
crps = crps_mdn_numpy(N_MIXTURES)
df_validation["CRPS"] = crps(y_test, y_pred_mdn)

for i, cl in enumerate(confidence_levels):
    df_validation[f"LB_{int(100*cl)}"] = intervals[:, i, 0]
    df_validation[f"UB_{int(100*cl)}"] = intervals[:, i, 1]

os.makedirs("predictions", exist_ok=True)
df_validation.to_csv(
    f"predictions/lstm_ffnn_mdn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_v{VERSION}.csv"
)
