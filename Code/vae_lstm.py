# https://chatgpt.com/share/67a20c84-7930-8001-b8fe-e74113a1a935
# %% [markdown]
# # 0) Imports & Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Dropout,
    Lambda,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam
import os
import warnings

from shared.processing import get_lstm_train_test

warnings.filterwarnings("ignore")

# For root-finding in intervals (if you adapt from your mixture approach)
from scipy.optimize import brentq
from scipy.stats import norm

# Import parameters
from settings import LOOKBACK_DAYS, TEST_ASSET, SUFFIX

VERSION = 1
MODEL_NAME = f"vae_lstm_{LOOKBACK_DAYS}_days{SUFFIX}_v{VERSION}"


# %% [markdown]
# # 1) Load and preprocess data (abbreviated)

# In practice, reuse your own code to produce:
# X_train, X_test, y_train, y_test

df, X_train, X_test, y_train, y_test = get_lstm_train_test()

# %% [markdown]
# # 2) Variational Autoencoder (VAE)

latent_dim = 8  # dimension of the latent space


# 2.1.1) Encoder
def build_vae_encoder(lookback_days, num_features, latent_dim):
    """
    LSTM-based encoder that produces mean and log-variance for the latent space.
    """
    encoder_inputs = Input(shape=(lookback_days, num_features))
    x = LSTM(16, activation="tanh")(encoder_inputs)
    # You can add Dropout or more layers, etc.
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, name="z")([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


# 2.1.2) Decoder
def build_vae_decoder(lookback_days, num_features, latent_dim):
    """
    LSTM-based decoder. For a sequence reconstruction, we can:
      1) Repeat 'z' for 'lookback_days' timesteps
      2) LSTM -> TimeDistributed Dense (if you want a sequence output)
    """
    latent_inputs = Input(shape=(latent_dim,), name="z_decoder_input")
    # Expand to (batch, 1, latent_dim), then repeat
    repeated_z = RepeatVector(lookback_days)(latent_inputs)
    x = LSTM(16, activation="tanh", return_sequences=True)(repeated_z)
    x = TimeDistributed(Dense(num_features, activation=None))(x)

    decoder = Model(latent_inputs, x, name="decoder")
    return decoder


# 2.1.3) VAE model with custom loss
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Return only one output so Keras doesn't expect multiple targets
        return reconstructed

    def train_step(self, data):
        # data is your input; in an autoencoder context, y = x
        x = data

        with tf.GradientTape() as tape:
            # Instead of relying on self(...) to get z_mean etc.,
            # call the encoder directly for them:
            z_mean, z_log_var, z = self.encoder(x, training=True)
            reconstructed = self.decoder(z, training=True)

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(x - reconstructed), axis=[1, 2])
            )

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
                )
            )

            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Track these losses for metrics
        self.compiled_metrics.update_state(x, reconstructed)
        return {
            "loss": total_loss,
            "recon_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


# %% [markdown]
# # 2.2) Build & train the VAE
num_features = X_train.shape[2]


def dummy_loss(y_true, y_pred):
    # Return a scalar tensor, e.g. always zero.
    return tf.constant(0.0)


encoder = build_vae_encoder(LOOKBACK_DAYS, num_features, latent_dim)
decoder = build_vae_decoder(LOOKBACK_DAYS, num_features, latent_dim)
vae = VAE(encoder, decoder, beta=1.0)
vae.compile(optimizer=Adam(learning_rate=1e-3), loss=dummy_loss)

# Fit (reconstruct input -> input)
vae.fit(X_train, epochs=10, batch_size=32, validation_data=(X_test, X_test), verbose=1)


# %% [markdown]
# # 3) Get latent embeddings from VAE


def get_embeddings_from_vae(encoder, X):
    """
    Returns mean (z_mean) for each input in X.
    If you prefer sampling, change to return z (the third output).
    """
    z_mean, z_log_var, z_sample = encoder.predict(X, batch_size=32)
    return z_sample  # shape: (batch_size, latent_dim)


Z_train = get_embeddings_from_vae(encoder, X_train)
Z_test = get_embeddings_from_vae(encoder, X_test)

print("Latent embeddings shape:")
print(f"  Z_train: {Z_train.shape}")
print(f"  Z_test:  {Z_test.shape}")


# %% [markdown]
# # 4) Build LSTM predictor that takes latent samples (1 step)


def build_latent_lstm_predictor(latent_dim):
    """
    Input shape: (batch, 1, latent_dim)
    Output: single scalar prediction
    """
    inputs = Input(shape=(1, latent_dim))
    x = LSTM(16, activation="tanh")(inputs)  # single-step LSTM
    x = Dropout(0.1)(x)
    x = Dense(8, activation="relu")(x)
    outputs = Dense(1, activation=None)(x)
    model = Model(inputs, outputs, name="latent_lstm_predictor")
    return model


latent_lstm = build_latent_lstm_predictor(latent_dim)
latent_lstm.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

# For training data, we can do a "deterministic" approach:
#  - use z_mean as input
#  - shape it as (N, 1, latent_dim)
Z_train_reshaped = Z_train.reshape(-1, 1, latent_dim)
Z_test_reshaped = Z_test.reshape(-1, 1, latent_dim)

latent_lstm.fit(
    Z_train_reshaped,
    y_train,
    validation_data=(Z_test_reshaped, y_test),
    epochs=10,
    batch_size=32,
    verbose=1,
)

# %% [markdown]
# # 5) Sample-based distribution from VAE + LSTM


def sample_latent_posterior(encoder, x, n_samples=100):
    """
    For a *single* input x of shape (lookback_days, num_features),
    sample 'n_samples' from q(z|x).

    Returns array of shape (n_samples, latent_dim).
    """
    # Expand dims to (1, lookback_days, num_features)
    x_tf = tf.expand_dims(tf.constant(x, dtype=tf.float32), axis=0)

    # Get posterior parameters (z_mean, z_log_var)
    z_mean, z_log_var, _ = encoder(x_tf, training=False)  # shape: (1, latent_dim)
    z_mean = z_mean[0].numpy()
    z_log_var = z_log_var[0].numpy()

    # Sample from N(z_mean, exp(z_log_var))
    eps = np.random.randn(n_samples, latent_dim)
    z_samples = z_mean + np.exp(0.5 * z_log_var) * eps
    return z_samples


def predict_distribution(encoder, predictor, X, n_samples=100):
    """
    For each sample X[i], sample the posterior 'n_samples' times,
    feed each sample into the LSTM -> distribution of predictions.

    Returns a 2D np.array of shape (len(X), n_samples).
    """
    all_preds = []
    for i in range(len(X)):
        x = X[i]  # shape: (lookback_days, num_features)

        # Sample from posterior
        z_samples = sample_latent_posterior(encoder, x, n_samples=n_samples)

        # Reshape for LSTM: (n_samples, 1, latent_dim)
        z_samples_reshaped = z_samples.reshape(n_samples, 1, latent_dim)

        # Predict
        preds = predictor.predict(z_samples_reshaped, verbose=0).ravel()
        all_preds.append(preds)
    return np.array(all_preds)


# Let's do it for X_test
n_samples = 200
y_dist_test = predict_distribution(encoder, latent_lstm, X_test, n_samples=n_samples)
print("y_dist_test shape:", y_dist_test.shape)  # (num_test_samples, n_samples)

# Some summary stats
y_mean_test = y_dist_test.mean(axis=1)
y_std_test = y_dist_test.std(axis=1)

# %% [markdown]
# # 6) Save predictions in a DataFrame

# In your real code, you'll have your actual test date index.
test_dates = pd.date_range(start="2020-01-01", periods=len(X_test), freq="B")
df_validation = pd.DataFrame(index=test_dates)

df_validation["True_Return"] = y_test
df_validation["Pred_Mean"] = y_mean_test
df_validation["Pred_Std"] = y_std_test

# Example: 95% interval
lower_q = 0.025
upper_q = 0.975
df_validation["Pred_LB_95"] = np.quantile(y_dist_test, lower_q, axis=1)
df_validation["Pred_UB_95"] = np.quantile(y_dist_test, upper_q, axis=1)

os.makedirs("predictions", exist_ok=True)
csv_fname = (
    f"predictions/vae_lstm_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_v{VERSION}.csv"
)
df_validation.to_csv(csv_fname)

df_validation.head()

# %% [markdown]
# # 7) Plot predictions vs. actual

import matplotlib.dates as mdates

plt.figure(figsize=(12, 6))
plt.plot(
    df_validation.index, df_validation["True_Return"], label="Actual", color="black"
)
plt.plot(
    df_validation.index, df_validation["Pred_Mean"], label="Predicted Mean", color="red"
)
plt.fill_between(
    df_validation.index,
    df_validation["Pred_LB_95"],
    df_validation["Pred_UB_95"],
    color="red",
    alpha=0.2,
    label="95% Interval",
)
plt.title("VAE + LSTM/MLP Non-Parametric Forecast Distribution")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.show()
# %%

# %%

# 8) make a plot showing the standard deviation of the predictions vs the true returns
plt.figure(figsize=(12, 6))
plt.plot(
    df_validation.index, df_validation["True_Return"], label="Actual", color="black"
)
plt.plot(
    df_validation.index, df_validation["Pred_Mean"], label="Predicted Mean", color="red"
)
plt.fill_between(
    df_validation.index,
    df_validation["Pred_LB_95"],
    df_validation["Pred_UB_95"],
    color="red",
    alpha=0.2,
    label="95% Interval",
)
plt.fill_between(
    df_validation.index,
    df_validation["Pred_Mean"] - df_validation["Pred_Std"],
    df_validation["Pred_Mean"] + df_validation["Pred_Std"],
    color="blue",
    alpha=0.2,
    label="1 Std Dev",
)
plt.fill_between(
    df_validation.index,
    df_validation["Pred_Mean"] - 2 * df_validation["Pred_Std"],
    df_validation["Pred_Mean"] + 2 * df_validation["Pred_Std"],
    color="green",
    alpha=0.2,
    label="2 Std Dev",
)
plt.title("VAE + LSTM Non-Parametric Forecast Distribution")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.show()
# %%

# %%
# the same but only for last 100 days
plt.figure(figsize=(12, 6))
plt.plot(
    df_validation.index[-100:],
    df_validation["True_Return"].iloc[-100:],
    label="Actual",
    color="black",
)
plt.plot(
    df_validation.index[-100:],
    df_validation["Pred_Mean"].iloc[-100:],
    label="Predicted Mean",
    color="red",
)
plt.fill_between(
    df_validation.index[-100:],
    df_validation["Pred_LB_95"].iloc[-100:],
    df_validation["Pred_UB_95"].iloc[-100:],
    color="red",
    alpha=0.2,
    label="95% Interval",
)
plt.fill_between(
    df_validation.index[-100:],
    df_validation["Pred_Mean"].iloc[-100:] - df_validation["Pred_Std"].iloc[-100:],
    df_validation["Pred_Mean"].iloc[-100:] + df_validation["Pred_Std"].iloc[-100:],
    color="blue",
    alpha=0.2,
    label="1 Std Dev",
)
plt.fill_between(
    df_validation.index[-100:],
    df_validation["Pred_Mean"].iloc[-100:] - 2 * df_validation["Pred_Std"].iloc[-100:],
    df_validation["Pred_Mean"].iloc[-100:] + 2 * df_validation["Pred_Std"].iloc[-100:],
    color="green",
    alpha=0.2,
    label="2 Std Dev",
)
plt.title("VAE + LSTM Non-Parametric Forecast Distribution")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# We'll assume:
#   y_dist_test: shape (num_test_samples, n_samples)
#   df_validation: DataFrame with index = test_dates, columns including "True_Return"
#   TEST_ASSET: name of the asset, for the title or label
#   n_samples: how many posterior samples per day (e.g. 200)

plt.figure(figsize=(8, 36))  # Make tall subplots
np.random.seed(0)

for subplot_i in range(10):
    plt.subplot(10, 1, subplot_i + 1)

    # Pick a random index among test samples
    idx = np.random.randint(0, len(df_validation))

    # The distribution of predictions for this day
    samples = y_dist_test[idx, :]  # shape (n_samples,)

    # The actual return
    actual_return = df_validation["True_Return"].iloc[idx]
    timestamp = df_validation.index[idx]

    # 1) Build a kernel density estimate or histogram for the samples
    # Option A: histogram
    # counts, bin_edges = np.histogram(samples, bins=50, density=True)
    # bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    # plt.fill_between(bin_centers, counts, color="blue", alpha=0.5, label="Empirical Dist.")

    # Option B: kernel density
    kde = gaussian_kde(samples)
    x_min = np.min(samples) - 0.01
    x_max = np.max(samples) + 0.01
    x_vals = np.linspace(x_min, x_max, 200)
    pdf_vals = kde(x_vals)
    plt.fill_between(x_vals, pdf_vals, color="blue", alpha=0.5, label="Empirical KDE")

    # 2) Plot vertical line at the true (actual) return
    plt.axvline(actual_return, color="red", linestyle="--", label="Actual")

    # 3) Plot vertical line at predicted mean
    pred_mean = np.mean(samples)
    plt.axvline(pred_mean, color="black", linestyle="--", label="Pred Mean")

    # 4) Optionally, add text with stats
    pred_std = np.std(samples)
    pred_min, pred_max = np.min(samples), np.max(samples)
    text_stats = (
        f"Mean: {pred_mean:.4f}\n"
        f"Std:  {pred_std:.4f}\n"
        f"Min:  {pred_min:.4f}\n"
        f"Max:  {pred_max:.4f}"
    )
    # position the text inside the subplot
    plt.text(
        x_min + 0.01 * (x_max - x_min),
        0.6 * np.max(pdf_vals),
        text_stats,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6),
    )

    plt.title(f"{timestamp.strftime('%Y-%m-%d')} - Distribution for {TEST_ASSET}")
    plt.ylabel("Density")
    if subplot_i == 9:
        plt.xlabel("Predicted Return")
    plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
# %%
