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

from shared.processing import get_lstm_train_test_new

warnings.filterwarnings("ignore")

# For root-finding in intervals (if you adapt from your mixture approach)
from scipy.optimize import brentq
from scipy.stats import norm

# Example placeholders (replace with your real paths)
# improt paprameters from settings.py
from settings import LOOKBACK_DAYS, TEST_ASSET, SUFFIX

VERSION = 2
MODEL_NAME = f"vae_lstm_{LOOKBACK_DAYS}_days{SUFFIX}_v{VERSION}"


# %% [markdown]
# # 1) Load and preprocess data (abbreviated)

# Load and preprocess data
processed_data = get_lstm_train_test_new()

# Extract train, validation, and test datasets
train_data = processed_data.train
validation_data = processed_data.validation
test_data = processed_data.test

# Extract features (X) and labels (y)
X_train, y_train = train_data.X, train_data.y
X_test, y_test = test_data.X, test_data.y

# Debugging - print shapes
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")

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
    x = Dropout(0.2)(x)  # TODO added afterwards to prevent overfitting (remove?)
    x = TimeDistributed(Dense(num_features, activation=None))(x)

    decoder = Model(latent_inputs, x, name="decoder")
    return decoder


# 2.1.3) VAE model with custom loss
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=0.01, **kwargs): # TODO beta=1.0 earlier
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

            # Reconstruction loss (OLD) # TODO: check if this is correct (original code new below)
            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(tf.square(x - reconstructed), axis=[1,2]) # TODO should this be axis=[1,2] or axis=1?
            # )

            # Reconstruction loss (NEW) TODO: check if this is correct
            # Use mean squared error (MSE)
            reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))

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
    

    def test_step(self, data): # TODO: check if this is correct (added afterwards)
        if isinstance(data, tuple):  # If a tuple (X, X) is passed, extract X
            x, _ = data
        else:
            x = data  # Otherwise, use it directly
        # x = data
        z_mean, z_log_var, z = self.encoder(x, training=False)
        reconstructed = self.decoder(z, training=False)

        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        total_loss = reconstruction_loss + self.beta * kl_loss
        return {"loss": total_loss, "recon_loss": reconstruction_loss, "kl_loss": kl_loss}

# %% [markdown]
# # 2.2) Build & train the VAE
num_features = X_train.shape[2]


def dummy_loss(y_true, y_pred):
    # Return a scalar tensor, e.g. always zero.
    return tf.constant(0.0)


encoder = build_vae_encoder(LOOKBACK_DAYS, num_features, latent_dim)
decoder = build_vae_decoder(LOOKBACK_DAYS, num_features, latent_dim)
vae = VAE(encoder, decoder, beta=0.01) # TODO beta=1.0 earlier
vae.compile(optimizer=Adam(learning_rate=5e-4), loss=dummy_loss) # TODO: check if this is correct (earlier: learning_rate=1e-3)  

# Fit (reconstruct input -> input)
vae.fit(X_train, epochs=3, batch_size=32, validation_data=(X_test, X_test), verbose=1)


# %% [markdown] # TODO onbly for testing (REMOVE AFTER TESTING)
from vae_lstm_HELPERS import plot_loss, visualize_latent_space, plot_reconstruction, compare_real_vs_generated

# Use after training
plot_loss(vae.history)  # Plot loss curves
visualize_latent_space(encoder, X_test, method="PCA")  # Check latent space
plot_reconstruction(vae, X_test, num_samples=5)  # Check reconstructions
compare_real_vs_generated(encoder, decoder, X_test)  # Compare distributions


# %% [markdown]
# # 3) Get latent embeddings from VAE


def get_embeddings_from_vae(encoder, X):
    z_mean, z_log_var, z_sample = encoder.predict(X, batch_size=32)
    return z_mean  # shape: (batch_size, latent_dim)


Z_train = get_embeddings_from_vae(encoder, X_train)
Z_test = get_embeddings_from_vae(encoder, X_test)

print("Latent embeddings shape:")
print(f"  Z_train: {Z_train.shape}")
print(f"  Z_test:  {Z_test.shape}")

#% % [markdown]
# TDDO? 


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
    epochs=3, # 10
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


def predict_distribution(encoder, predictor, X, n_samples=1000):
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


# %% [markdown]
# Calulacte loss function values for the total distribution sampled from output of the model
# =================================================================================================
# ============================= MUST BE MOVED TO loss.py FILE ===================================
# =================================================================================================

from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde
import numpy as np

def compute_nll(y_true, y_samples, eps=1e-12):
    """
    Compute Negative Log Likelihood (NLL) for VAE + LSTM.
    Uses Kernel Density Estimation (KDE) to approximate the probability density.
    
    Arguments:
    - y_true: True values (shape: (batch,))
    - y_samples: Sampled predictions from LSTM (shape: (batch, n_samples))
    
    Returns:
    - Mean NLL across all samples
    """
    B = y_true.shape[0]  # Batch size
    nlls = np.zeros(B)

    for i in range(B):
        sample_var = np.var(y_samples[i])  # Check variance

        # If variance is too low, assign a fallback probability
        if sample_var < 1e-8:  # Threshold to detect near-identical samples
            p_y = eps  # Assign a small probability to avoid log(0)
        else:
            try:
                kde = gaussian_kde(y_samples[i])  # Estimate density from samples
                p_y = kde(y_true[i])  # Evaluate density at the true value
            except np.linalg.LinAlgError:
                p_y = eps  # In case KDE still fails, use a fallback small probability

        p_y = np.maximum(p_y, eps)  # Prevent log(0)
        nlls[i] = -np.log(p_y)  # Compute NLL

    return np.mean(nlls)  # Return mean NLL across batch

def compute_crps(y_true, y_samples):
    """
    Compute CRPS (Continuous Ranked Probability Score) for VAE + LSTM.
    
    Arguments:
    - y_true: True values (shape: (batch,))
    - y_samples: Sampled predictions from LSTM (shape: (batch, n_samples))
    
    Returns:
    - CRPS values (shape: (batch,))
    """
    B = y_true.shape[0]  # Batch size
    crps_values = np.zeros(B)

    for i in range(B):
        s = np.sort(y_samples[i])  # Sort the sample predictions
        n = len(s)

        # Compute empirical CDF of the samples
        empirical_cdf = np.arange(1, n + 1) / n

        # Find where the true value falls in the sorted sample set
        indicator = (y_true[i] <= s).astype(float)

        # Compute CRPS using the empirical formula
        crps_values[i] = np.mean((empirical_cdf - indicator) ** 2)

    return crps_values  # Returns array of shape (batch,)

def compute_confidence_intervals(y_samples, confidence_levels):
    """
    Compute confidence intervals for different confidence levels from sampled predictions.

    Arguments:
    - y_samples: Sampled predictions from LSTM (shape: (batch, n_samples))
    - confidence_levels: List of confidence levels (e.g., [0.67, 0.95, 0.975, 0.99])

    Returns:
    - Array of shape (batch, len(confidence_levels), 2) containing lower and upper bounds.
    """
    num_levels = len(confidence_levels)
    num_samples = y_samples.shape[0]

    intervals = np.zeros((num_samples, num_levels, 2))

    for i, cl in enumerate(confidence_levels):
        lower_q = (1 - cl) / 2  # Lower bound quantile
        upper_q = 1 - lower_q   # Upper bound quantile

        # Compute lower and upper bound for each sample
        intervals[:, i, 0] = np.quantile(y_samples, lower_q, axis=1)
        intervals[:, i, 1] = np.quantile(y_samples, upper_q, axis=1)

    return intervals

df_test = pd.DataFrame(index=test_dates)  # Ensure test_dates is correctly defined


# Compute NLL for the test set
df_test["NLL"] = compute_nll(y_test, y_dist_test) 
df_test["CRPS"] = compute_crps(y_test, y_dist_test)

# Define confidence levels and compute confidence intervals from sampled predictions
confidence_levels = [0.5, 0.67, 0.90, 0.95, 0.975, 0.99]
intervals = compute_confidence_intervals(y_dist_test, confidence_levels)

# Store in DataFrame
for i, cl in enumerate(confidence_levels):
    df_test[f"LB_{int(100*cl)}"] = intervals[:, i, 0]  # Lower bound
    df_test[f"UB_{int(100*cl)}"] = intervals[:, i, 1]  # Upper bound


# %% [markdown]
# # 7) Save predictions to CSV and plot
os.makedirs("predictions", exist_ok=True)
csv_fname = (
    f"predictions/vae_lstm_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_v{VERSION}.csv"
)
df_validation.to_csv(csv_fname)

df_validation.head()




# %% [markdown]
# ================================== GRAPHS ==================================
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
