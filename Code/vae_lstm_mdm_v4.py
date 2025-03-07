# https://chatgpt.com/share/67a20c84-7930-8001-b8fe-e74113a1a935
# %% [markdown]
# # 0) Imports & Setup

import shutil
import sys
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

from shared.loss import mdn_crps_tf, mdn_nll_numpy, mdn_nll_tf
from shared.mdn import calculate_es_for_quantile, calculate_intervals_vectorized, calculate_prob_above_zero_vectorized, get_mdn_bias_initializer, get_mdn_kernel_initializer, parse_mdn_output, plot_sample_days, univariate_mixture_mean_and_var_approx
from shared.conf_levels import format_cl
from shared.processing import get_lstm_train_test_new

warnings.filterwarnings("ignore")

# For root-finding in intervals (if you adapt from your mixture approach)
from scipy.optimize import brentq
from scipy.stats import norm

# Example placeholders (replace with your real paths)
# improt paprameters from settings.py
from settings import LOOKBACK_DAYS, TEST_ASSET, SUFFIX

VERSION = 4
MODEL_NAME = f"vae_lstm_mdm_{LOOKBACK_DAYS}_days{SUFFIX}_v{VERSION}"


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
X_val, y_val = validation_data.X, validation_data.y

# Debugging - print shapes
print(f"X_train.shape: {X_train.shape}")
print(f"X_val.shape: {X_val.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_val.shape: {y_val.shape}")

# %% [markdown]
# # 2) Variational Autoencoder (VAE)

latent_dim = 8  # dimension of the latent space


# 2.1.1) Encoder
def build_vae_encoder(lookback_days, num_features, latent_dim):
    """
    LSTM-based encoder that produces mean and log-variance for the latent space.
    """
    encoder_inputs = Input(shape=(lookback_days, num_features))
    x = LSTM(32, activation="tanh")(encoder_inputs)
    # You can add Dropout or more layers, etc.
    x = Dropout(0.2)(x) # TODO added afterwards to prevent overfitting (remove?)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(
        sampling,
        name="z",
        output_shape=(latent_dim,)
    )([z_mean, z_log_var])
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
    x = LSTM(32, activation="tanh", return_sequences=True)(repeated_z)
    x = Dropout(0.2)(x)  # TODO added afterwards to prevent overfitting (remove?)
    x = TimeDistributed(Dense(num_features, activation=None))(x)

    decoder = Model(latent_inputs, x, name="decoder")
    return decoder


# 2.1.3) VAE model with custom loss
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs): # TODO beta=1.0 earlier
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
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(x - reconstructed), axis=[1,2]) # TODO should this be axis=[1,2] or axis=1?
            )

            # Reconstruction loss (NEW) TODO: check if this is correct
            # Use mean squared error (MSE)
            # reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))

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

        # reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))
        reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(x - reconstructed), axis=[1,2]) # TODO should this be axis=[1,2] or axis=1?
            )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        total_loss = reconstruction_loss + self.beta * kl_loss
        return {"loss": total_loss, "recon_loss": reconstruction_loss, "kl_loss": kl_loss}


    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Hard-code the architecture that was used originally,
        so that load_model() can rebuild a matching VAE.
        """
        # Rebuild sub-models with the same shapes/dims that were used before.
        encoder = build_vae_encoder(LOOKBACK_DAYS, num_features, latent_dim)
        decoder = build_vae_decoder(LOOKBACK_DAYS, num_features, latent_dim)
        return cls(encoder, decoder, beta=1.0)
    
# %% [markdown]
# # 2.2) Build & train the VAE
num_features = X_train.shape[2]


def dummy_loss(y_true, y_pred):
    # Return a scalar tensor, e.g. always zero.
    return tf.constant(0.0)


encoder = build_vae_encoder(LOOKBACK_DAYS, num_features, latent_dim)
decoder = build_vae_decoder(LOOKBACK_DAYS, num_features, latent_dim)
vae = VAE(encoder, decoder, beta=1.0) # TODO beta=1.0 earlier
vae.compile(optimizer=Adam(learning_rate=1e-4), loss=dummy_loss) # TODO: check if this is correct (earlier: learning_rate=1e-3)  

# %%
# Load vae from file if it exists
if os.path.exists("models/"+MODEL_NAME + "_vae.keras"):
    vae = tf.keras.models.load_model(
        "models/"+MODEL_NAME + "_vae.keras",
        custom_objects={"VAE": VAE, "dummy_loss": dummy_loss},
    )

    print("Loaded model from file")

# %%
# Fit (reconstruct input -> input)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
vae.fit(X_train, epochs=20, batch_size=32, validation_data=(X_val, X_val), verbose=1, callbacks=[early_stopping])
#%%
vae.save("models/"+MODEL_NAME + "_vae.keras")
# %% [markdown] # TODO onbly for testing (REMOVE AFTER TESTING)
from vae_lstm_HELPERS import plot_loss, plot_reconstruction_distributions, visualize_latent_space, plot_reconstruction, compare_real_vs_generated

# Use after training
plot_loss(vae.history)  # Plot loss curves
visualize_latent_space(encoder, X_val, method="PCA")  # Check latent space
plot_reconstruction(vae, X_val, num_samples=5)  # Check reconstructions
compare_real_vs_generated(encoder, decoder, X_val)  # Compare distributions
plot_reconstruction_distributions(vae, X_val)  # Check reconstructions


# %% [markdown]
# # 3) Get latent embeddings from VAE

Z_train_means, Z_train_log_vars, _ = encoder.predict(X_train, batch_size=32)
Z_val_means, Z_val_log_vars, _ = encoder.predict(X_val, batch_size=32)

# %%
# Get samples from the latent space
def get_embeddings_from_vae(encoder, X):
    z_mean, z_log_var, z_sample = encoder.predict(X, batch_size=32)
    return z_sample  # shape: (batch_size, latent_dim)

Z_trains = []
Z_vals = []
y_trains = []
y_vals = []
n_samples = 100
for i in range(n_samples):
    Z_trains.append(get_embeddings_from_vae(encoder, X_train))
    y_trains.append(y_train)
    Z_vals.append(get_embeddings_from_vae(encoder, X_val))
    y_vals.append(y_val)
Z_train_samples = np.vstack(Z_trains)
y_train_broadcasted = np.hstack(y_trains)
Z_val_samples = np.vstack(Z_vals)
y_val_broadcasted = np.hstack(y_vals)

print("Latent embeddings shape:")
print(f"  Z_train: {Z_train_samples.shape}")
print(f"  Z_val:  {Z_val_samples.shape}")


# %%
# Save samples
np.save("data/"+MODEL_NAME + "_Z_train_samples.npy", Z_train_samples)
np.save("data/"+MODEL_NAME + "_Z_val_samples.npy", Z_val_samples)

# %%
# Load samples if found
Z_train_samples = np.load("data/"+MODEL_NAME + "_Z_train_samples.npy")
Z_val_samples = np.load("data/"+MODEL_NAME + "_Z_val_samples.npy")
n_samples = Z_train_samples.shape[0] // Z_train_means.shape[0]
y_train_broadcasted = np.tile(y_train, n_samples)
y_val_broadcasted = np.tile(y_val, n_samples)
n_samples, "samples loaded"

# %% [markdown]
# # 4) Build predictors that takes latent samples (1 step)

N_MIXTURES = 10

def build_latent_linear_regression_predictor(latent_dim):
    """
    Input shape: (batch, latent_dim)
    Output: single scalar prediction
    """
    inputs = Input(shape=(latent_dim,))

    # MDN output layer: 3 * n_mixtures (for [logits_pi, mu, log_sigma])
    mdn_kernel_init = get_mdn_kernel_initializer(N_MIXTURES)
    mdn_bias_init = get_mdn_bias_initializer(N_MIXTURES)
    mdn_output = Dense(
        3 * N_MIXTURES,
        activation=None,
        kernel_initializer=mdn_kernel_init,
        bias_initializer=mdn_bias_init,
        name="mdn_output",
    )(inputs)

    model = Model(
        inputs=inputs,
        outputs=mdn_output,
    )
    return model


def build_latent_mean_and_std_predictor(latent_dim):
    """
    Input shape: (batch, latent_dim, 2) # 2 for mean and std
    Output: single scalar prediction
    """
    inputs = Input(shape=(latent_dim, 2))

    # Flatten the input
    x = tf.keras.layers.Flatten()(inputs)

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
        inputs=inputs,
        outputs=mdn_output,
    )
    return model

def build_latent_ffnn_predictor(latent_dim):
    """
    Input shape: (batch, latent_dim)
    Output: single scalar prediction
    """
    inputs = Input(shape=(latent_dim,))
    x = Dense(64, activation="relu")(inputs)
    x = Dropout(0.4)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.4)(x)

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
        inputs=inputs,
        outputs=mdn_output,
    )
    return model

def build_latent_mean_and_std_fnn_predictor(latent_dim):
    """
    Input shape: (batch, latent_dim, 2) # 2 for mean and std
    Output: single scalar prediction
    """
    inputs = Input(shape=(latent_dim, 2))
    x = tf.keras.layers.Flatten()(inputs)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.4)(x)

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
        inputs=inputs,
        outputs=mdn_output,
    )
    return model



def build_latent_lstm_predictor(latent_dim):
    """
    Input shape: (batch, 1, latent_dim)
    Output: single scalar prediction
    """
    inputs = Input(shape=(1, latent_dim))
    x = LSTM(64, activation="tanh")(inputs)  # single-step LSTM
    x = Dropout(0.4)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.4)(x)

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
        inputs=inputs,
        outputs=mdn_output,
    )
    return model

# %%
def make_predictions_and_stats(preds, n_mixtures=N_MIXTURES):
    df_validation = pd.DataFrame(index=[validation_data.dates, validation_data.tickers])
    df_validation.index.names = ["Date", "Symbol"]
    pis, mus, sigmas = parse_mdn_output(preds, n_mixtures)
    uni_mixture_mean_sp, uni_mixture_var_sp = univariate_mixture_mean_and_var_approx(
        pis, mus, sigmas
    )
    df_validation["NLL"] = mdn_nll_numpy(n_mixtures)(validation_data.y, preds)
    # df_validation["CRPS"] = mdn_crps_tf(n_mixtures)(validation_data.y, preds)
    df_validation["Pred_Mean"] = uni_mixture_mean_sp
    uni_mixture_mean_sp = uni_mixture_mean_sp.numpy()
    uni_mixture_std_sp = np.sqrt(uni_mixture_var_sp.numpy())
    df_validation["Pred_Std"] = uni_mixture_std_sp
    confidence_levels = [0.67, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    intervals = calculate_intervals_vectorized(
        pis, mus, sigmas, confidence_levels
    )
    df_validation["Prob_Increase"] = calculate_prob_above_zero_vectorized(
        pis, mus, sigmas
    )
    for i, cl in enumerate(confidence_levels):
        df_validation[f"LB_{format_cl(cl)}"] = intervals[:, i, 0]  # Lower bound
        df_validation[f"UB_{format_cl(cl)}"] = intervals[:, i, 1]  # Upper bound
        es_alpha = 1-(1-cl)/2
        df_validation[f"ES_{format_cl(es_alpha)}"] = calculate_es_for_quantile(pis,mus,sigmas, intervals[:, i, 0])
    return df_validation

def merge_sample_preds(preds, n_samples):
    # Divide preds into n_samples parts
    preds = np.split(preds, n_samples)
    all_pis = []
    all_mus = []
    all_sigmas = []
    # Divide pis by the number of samples to correct for the split
    for i in range(n_samples):
        pis, mus, sigmas = np.split(preds[i], 3, axis=1)
        pis = tf.nn.softmax(preds[i][:, :N_MIXTURES]).numpy()
        pis /= n_samples
        all_pis.append(pis)
        all_mus.append(mus)
        all_sigmas.append(sigmas)

    # Concatenate the predictions for each sample (day + ticker combo), placing pis together, mus together, sigmas together
    c_pis = np.concatenate(all_pis, axis=1)
    c_mus = np.concatenate(all_mus, axis=1)
    c_sigmas = np.concatenate(all_sigmas, axis=1)
    c_pis = np.log(c_pis)
    preds = np.concatenate([c_pis, c_mus, c_sigmas], axis=1)
    return preds

# %%
simple_regressor_on_means = build_latent_linear_regression_predictor(latent_dim)
simple_regressor_on_means.compile(optimizer=Adam(learning_rate=1e-3), loss=mdn_nll_tf(N_MIXTURES))

# %%
# Load regressor from file if it exists
if os.path.exists("models/"+MODEL_NAME + "_simple_regressor_on_means.keras"):
    simple_regressor_on_means = tf.keras.models.load_model(
        "models/"+MODEL_NAME + "_simple_regressor_on_means.keras",
        custom_objects={"loss_fn": mdn_nll_tf(N_MIXTURES), "mdn_kernel_initializer": get_mdn_kernel_initializer(N_MIXTURES), "mdn_bias_initializer": get_mdn_bias_initializer(N_MIXTURES)},
    )

    print("Loaded model from file")

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)
simple_regressor_on_means.fit(
    Z_train_means,
    y_train,
    validation_data=(Z_val_means, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler],
)

# %%
# Save the model
extended_model_name = MODEL_NAME + "_simple_regressor_on_means"
simple_regressor_on_means.save("models/"+extended_model_name+".keras")

# %%
# Make predictions and calculate stats
preds = simple_regressor_on_means.predict(Z_val_means)
df_validation = make_predictions_and_stats(preds)
df_validation.head()

# %%
# Save predictions to CSV
pred_fname = "predictions/" + extended_model_name + ".csv"
df_validation.to_csv(pred_fname)
print("Saved predictions to:", pred_fname)

# %%
# Plot 10 charts with the distributions for 10 random days
example_tickers = ["AAPL", "WMT", "GS"]
pi_pred, mu_pred, sigma_pred = parse_mdn_output(preds, N_MIXTURES)
for ticker in example_tickers:
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
    plot_sample_days(
        s.y_dates,
        s.y,
        pi_pred[from_idx:to_idx],
        mu_pred[from_idx:to_idx],
        sigma_pred[from_idx:to_idx],
        N_MIXTURES,
        ticker=ticker,
        save_to=f"results/distributions/{ticker}_{extended_model_name}.svg",
    )

# %%
# Create simple regressor on samples model
simple_regressor_on_samples = build_latent_linear_regression_predictor(latent_dim)
simple_regressor_on_samples.compile(optimizer=Adam(learning_rate=1e-3), loss=mdn_nll_tf(N_MIXTURES))

# %%
simple_regressor_on_samples_model_name = MODEL_NAME + "_simple_regressor_on_samples"

# %%
# Load regressor from file if it exists
if os.path.exists("models/"+simple_regressor_on_samples_model_name + ".keras"):
    simple_regressor_on_samples = tf.keras.models.load_model(
        "models/"+simple_regressor_on_samples_model_name + ".keras",
        custom_objects={"loss_fn": mdn_nll_tf(N_MIXTURES), "mdn_kernel_initializer": get_mdn_kernel_initializer(N_MIXTURES), "mdn_bias_initializer": get_mdn_bias_initializer(N_MIXTURES)},
    )

    print("Loaded model from file")

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1
)
simple_regressor_on_samples.fit(
    Z_train_samples,
    y_train_broadcasted,
    validation_data=(Z_val_samples, y_val_broadcasted),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler],
)

# %%
# Save the model
simple_regressor_on_samples.save("models/"+simple_regressor_on_samples_model_name + ".keras")

# %%
# Make predictions
preds = merge_sample_preds(simple_regressor_on_samples.predict(Z_val_samples), n_samples)

# %%
# Calculate stats
df_validation = make_predictions_and_stats(preds, N_MIXTURES*n_samples)
df_validation.head()

# %%
# Save predictions to CSV
pred_fname = "predictions/" + simple_regressor_on_samples_model_name + ".csv"
df_validation.to_csv(pred_fname)
print("Saved predictions to:", pred_fname)

# %%
# Plot 10 charts with the distributions for 10 random days
example_tickers = ["AAPL", "WMT", "GS"]
pi_pred, mu_pred, sigma_pred = parse_mdn_output(preds, N_MIXTURES*n_samples)
for ticker in example_tickers:
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
    plot_sample_days(
        s.y_dates,
        s.y,
        pi_pred[from_idx:to_idx],
        mu_pred[from_idx:to_idx],
        sigma_pred[from_idx:to_idx],
        N_MIXTURES*n_samples,
        ticker=ticker,
        save_to=f"results/distributions/{ticker}_{simple_regressor_on_samples_model_name}.svg",
    )

# %%
# Plot weights over time to show how they change
fig, axes = plt.subplots(
    nrows=len(example_tickers), figsize=(18, len(example_tickers) * 9)
)

# Dictionary to store union of legend entries
legend_dict = {}

for ax, ticker in zip(axes, example_tickers):
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
    pi_pred_ticker = pi_pred[from_idx:to_idx]
    for j in range(N_MIXTURES*n_samples):
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
# Create simple regressor on mean and standard deviations of latent space
simple_regressor_on_means_and_std = build_latent_mean_and_std_predictor(latent_dim)
simple_regressor_on_means_and_std.compile(optimizer=Adam(learning_rate=1e-3), loss=mdn_nll_tf(N_MIXTURES))

# %%
simple_regressor_on_means_and_std_model_name = MODEL_NAME + "_simple_regressor_on_means_and_std"

# %%
# Load regressor from file if it exists
if os.path.exists("models/"+simple_regressor_on_means_and_std_model_name + ".keras"):
    simple_regressor_on_means_and_std = tf.keras.models.load_model(
        "models/"+simple_regressor_on_means_and_std_model_name + ".keras",
        custom_objects={"loss_fn": mdn_nll_tf(N_MIXTURES), "mdn_kernel_initializer": get_mdn_kernel_initializer(N_MIXTURES), "mdn_bias_initializer": get_mdn_bias_initializer(N_MIXTURES)},
    )

    print("Loaded model from file")

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)
input_data = np.array([Z_train_means, Z_train_log_vars]).reshape(-1, latent_dim, 2)
val_input_data = np.array([Z_val_means, Z_val_log_vars]).reshape(-1, latent_dim, 2)
simple_regressor_on_means_and_std.fit(
    input_data,
    y_train,
    validation_data=(val_input_data, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler],
)

# %%
# Save the model
simple_regressor_on_means_and_std.save("models/"+simple_regressor_on_means_and_std_model_name + ".keras")

# %%
# Make predictions
preds = simple_regressor_on_means_and_std.predict(val_input_data)

# %%
# Calculate stats
df_validation = make_predictions_and_stats(preds)
df_validation.head()

# %%
# Save predictions to CSV
pred_fname = "predictions/" + simple_regressor_on_means_and_std_model_name + ".csv"
df_validation.to_csv(pred_fname)
print("Saved predictions to:", pred_fname)

# %%
# Plot 10 charts with the distributions for 10 random days
example_tickers = ["AAPL", "WMT", "GS"]
pi_pred, mu_pred, sigma_pred = parse_mdn_output(preds, N_MIXTURES)
for ticker in example_tickers:
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
    plot_sample_days(
        s.y_dates,
        s.y,
        pi_pred[from_idx:to_idx],
        mu_pred[from_idx:to_idx],
        sigma_pred[from_idx:to_idx],
        N_MIXTURES,
        ticker=ticker,
        save_to=f"results/distributions/{ticker}_{simple_regressor_on_means_and_std_model_name}.svg",
    )

# %%
# Plot weights over time to show how they change
fig, axes = plt.subplots(
    nrows=len(example_tickers), figsize=(18, len(example_tickers) * 9)
)

# Dictionary to store union of legend entries
legend_dict = {}

for ax, ticker in zip(axes, example_tickers):
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
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
# Create FFNN predictor based on latent means
ffnn_on_latent_means = build_latent_ffnn_predictor(latent_dim)
ffnn_on_latent_means.compile(optimizer=Adam(learning_rate=1e-3), loss=mdn_nll_tf(N_MIXTURES))

# %%
ffnn_on_latent_means_model_name = MODEL_NAME + "_ffnn_on_latent_means"

# %%
# Load regressor from file if it exists
if os.path.exists("models/"+ffnn_on_latent_means_model_name + ".keras"):
    ffnn_on_latent_means = tf.keras.models.load_model(
        "models/"+ffnn_on_latent_means_model_name + ".keras",
        custom_objects={"loss_fn": mdn_nll_tf(N_MIXTURES), "mdn_kernel_initializer": get_mdn_kernel_initializer(N_MIXTURES), "mdn_bias_initializer": get_mdn_bias_initializer(N_MIXTURES)},
    )

    print("Loaded model from file")

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)
ffnn_on_latent_means.fit(
    Z_train_means,
    y_train,
    validation_data=(Z_val_means, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler],
)

# %%
# Save the model
ffnn_on_latent_means.save("models/"+ffnn_on_latent_means_model_name + ".keras")

# %%
# Make predictions
preds = ffnn_on_latent_means.predict(Z_val_means)

# %%
# Calculate stats
df_validation = make_predictions_and_stats(preds)
df_validation.head()

# %%
# Save predictions to CSV
pred_fname = "predictions/" + ffnn_on_latent_means_model_name + ".csv"
df_validation.to_csv(pred_fname)
print("Saved predictions to:", pred_fname)

# %%
# Plot 10 charts with the distributions for 10 random days
example_tickers = ["AAPL", "WMT", "GS"]
pi_pred, mu_pred, sigma_pred = parse_mdn_output(preds, N_MIXTURES)
for ticker in example_tickers:
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
    plot_sample_days(
        s.y_dates,
        s.y,
        pi_pred[from_idx:to_idx],
        mu_pred[from_idx:to_idx],
        sigma_pred[from_idx:to_idx],
        N_MIXTURES,
        ticker=ticker,
        save_to=f"results/distributions/{ticker}_{ffnn_on_latent_means_model_name}.svg",
    )

# %%
# Plot weights over time to show how they change
fig, axes = plt.subplots(
    nrows=len(example_tickers), figsize=(18, len(example_tickers) * 9)
)

# Dictionary to store union of legend entries
legend_dict = {}

for ax, ticker in zip(axes, example_tickers):
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
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
# Create FFNN predictor based on latent means and standard deviations
ffnn_on_latent_means_and_std = build_latent_mean_and_std_fnn_predictor(latent_dim)
ffnn_on_latent_means_and_std.compile(optimizer=Adam(learning_rate=1e-3), loss=mdn_nll_tf(N_MIXTURES))

# %%
ffnn_on_latent_means_and_std_model_name = MODEL_NAME + "_ffnn_on_latent_means_and_std"

# %%
# Load regressor from file if it exists
if os.path.exists("models/"+ffnn_on_latent_means_and_std_model_name + ".keras"):
    ffnn_on_latent_means_and_std = tf.keras.models.load_model(
        "models/"+ffnn_on_latent_means_and_std_model_name + ".keras",
        custom_objects={"loss_fn": mdn_nll_tf(N_MIXTURES), "mdn_kernel_initializer": get_mdn_kernel_initializer(N_MIXTURES), "mdn_bias_initializer": get_mdn_bias_initializer(N_MIXTURES)},
    )

    print("Loaded model from file")

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)
input_data = np.array([Z_train_means, Z_train_log_vars]).reshape(-1, latent_dim, 2)
val_input_data = np.array([Z_val_means, Z_val_log_vars]).reshape(-1, latent_dim, 2)
ffnn_on_latent_means_and_std.fit(
    input_data,
    y_train,
    validation_data=(val_input_data, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler],
)

# %%
# Save the model
ffnn_on_latent_means_and_std.save("models/"+ffnn_on_latent_means_and_std_model_name + ".keras")

# %%
# Make predictions
preds = ffnn_on_latent_means_and_std.predict(val_input_data)

# %%
# Calculate stats
df_validation = make_predictions_and_stats(preds)
df_validation.head()

# %%
# Save predictions to CSV
pred_fname = "predictions/" + ffnn_on_latent_means_and_std_model_name + ".csv"

# %%
df_validation.to_csv(pred_fname)
print("Saved predictions to:", pred_fname)

# %%
# Plot 10 charts with the distributions for 10 random days
example_tickers = ["AAPL", "WMT", "GS"]
pi_pred, mu_pred, sigma_pred = parse_mdn_output(preds, N_MIXTURES)
for ticker in example_tickers:
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
    plot_sample_days(
        s.y_dates,
        s.y,
        pi_pred[from_idx:to_idx],
        mu_pred[from_idx:to_idx],
        sigma_pred[from_idx:to_idx],
        N_MIXTURES,
        ticker=ticker,
        save_to=f"results/distributions/{ticker}_{ffnn_on_latent_means_and_std_model_name}.svg",
    )

# %%
# Plot weights over time to show how they change
fig, axes = plt.subplots(
    nrows=len(example_tickers), figsize=(18, len(example_tickers) * 9)
)

# Dictionary to store union of legend entries
legend_dict = {}

for ax, ticker in zip(axes, example_tickers):
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
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
# Create FFNN for predicting based on samples
ffnn_on_latent_samples = build_latent_ffnn_predictor(latent_dim)
ffnn_on_latent_samples.compile(optimizer=Adam(learning_rate=1e-3), loss=mdn_nll_tf(N_MIXTURES))

# %%
ffnn_on_latent_samples_model_name = MODEL_NAME + "_ffnn_on_latent_samples"

# %%
# Load regressor from file if it exists
if os.path.exists("models/"+ffnn_on_latent_samples_model_name + ".keras"):
    ffnn_on_latent_samples = tf.keras.models.load_model(
        "models/"+ffnn_on_latent_samples_model_name + ".keras",
        custom_objects={"loss_fn": mdn_nll_tf(N_MIXTURES), "mdn_kernel_initializer": get_mdn_kernel_initializer(N_MIXTURES), "mdn_bias_initializer": get_mdn_bias_initializer(N_MIXTURES)},
    )

    print("Loaded model from file")

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)
ffnn_on_latent_samples.fit(
    Z_train_samples,
    y_train_broadcasted,
    validation_data=(Z_val_samples, y_val_broadcasted),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler],
)

# %%
# Save the model
ffnn_on_latent_samples.save("models/"+ffnn_on_latent_samples_model_name + ".keras")

# %%
# Make predictions
preds = merge_sample_preds(ffnn_on_latent_samples.predict(Z_val_samples), n_samples)

# %%
# Calculate stats
df_validation = make_predictions_and_stats(preds, N_MIXTURES*n_samples)
df_validation.head()

# %%
# Save predictions to CSV
pred_fname = "predictions/" + ffnn_on_latent_samples_model_name + ".csv"
df_validation.to_csv(pred_fname)
print("Saved predictions to:", pred_fname)

# %%
# Plot 10 charts with the distributions for 10 random days
example_tickers = ["AAPL", "WMT", "GS"]

pi_pred, mu_pred, sigma_pred = parse_mdn_output(preds, N_MIXTURES*n_samples)
for ticker in example_tickers:
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
    plot_sample_days(
        s.y_dates,
        s.y,
        pi_pred[from_idx:to_idx],
        mu_pred[from_idx:to_idx],
        sigma_pred[from_idx:to_idx],
        N_MIXTURES*n_samples,
        ticker=ticker,
        save_to=f"results/distributions/{ticker}_{ffnn_on_latent_samples_model_name}.svg",
    )

# %%
# Plot weights over time to show how they change
fig, axes = plt.subplots(
    nrows=len(example_tickers), figsize=(18, len(example_tickers) * 9)
)

# Dictionary to store union of legend entries
legend_dict = {}

for ax, ticker in zip(axes, example_tickers):
    s = validation_data.sets[ticker]
    from_idx, to_idx = validation_data.get_range(ticker)
    pi_pred_ticker = pi_pred[from_idx:to_idx]
    for j in range(N_MIXTURES*n_samples):
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
