#https://chatgpt.com/share/67bd9d7a-d6a0-8008-86d6-157ffb922f08

# %%
# Define parameters
from settings import LOOKBACK_DAYS, SUFFIX, DATA_PATH, TRAIN_VALIDATION_SPLIT, VALIDATION_TEST_SPLIT, TEST_ASSET
MODEL_NAME = f"cVAE_v1_{LOOKBACK_DAYS}_days{SUFFIX}"
# %%
# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import seaborn as sns
from tqdm import tqdm
import math


from shared.processing import get_lstm_train_test_new

# %% 
# Load data
all_data = get_lstm_train_test_new(multiply_by_beta=False, include_fng=False, include_spx_data=True, include_returns=True)
X_train = all_data.train.X
y_train = all_data.train.y
X_val = all_data.validation.X
y_val = all_data.validation.y

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# %%
LATENT_DIM = 16      # dimension of the latent variable z
LSTM_UNITS_ENC = [32, 32, 32, 32]  # LSTM units for the encoder
LSTM_UNITS_DEC = [32, 32, 32, 32]  # LSTM units for the decoder
LSTM_UNITS_PRIOR = [32, 32,32, 32 ]  # LSTM units for the prior
DENSE_UNITS = 32     # size of dense layer before producing z params
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
TIME_STEPS = 30
FEATURE_DIM = 39 # this has to be the number of features in the input data
BETA = 10             # weight for the KL


# %%
# Define a helper function to build stacked LSTM layers
# 1) Helper function to build a stack of LSTM layers
def build_lstm_stack(
    input_tensor,
    lstm_units_list,
    name_prefix,
    final_return_sequences=False
):
    """
    Builds a stack of LSTM layers. The final layer will have return_sequences=False
    by default, unless final_return_sequences=True is specified.

    Args:
        input_tensor: The Keras input or intermediate tensor to feed in.
        lstm_units_list: List of ints, e.g. [64, 32].
        name_prefix: String prefix for layer names, e.g. "encoder" or "decoder".
        final_return_sequences: Whether the last LSTM in the stack should return sequences.

    Returns:
        The output tensor after applying all LSTM layers.
    """
    x = input_tensor
    for i, units in enumerate(lstm_units_list):
        is_last = (i == len(lstm_units_list) - 1)
        # If it's the last layer and we don't want to return sequences, set return_sequences=False
        # Otherwise, set return_sequences=True
        return_seq = (not is_last) or final_return_sequences
        x = layers.LSTM(
            units, 
            return_sequences=return_seq, 
            name=f"{name_prefix}_lstm_{i+1}"
        )(x)
    return x

#################################### ENCNNODER ####################################
# %%
# Define the encoder architecture
# Encoder inputs
encoder_input_x = keras.Input(shape=(TIME_STEPS, FEATURE_DIM), name="encoder_x")  # (None, 30, 36)
encoder_input_y = keras.Input(shape=(1,), name="encoder_y")                       # (None, 1)

# LSTM encodes X
# Stacked LSTM over X
x_encoded = build_lstm_stack(
    input_tensor=encoder_input_x,
    lstm_units_list=LSTM_UNITS_ENC,
    name_prefix="encoder",
    final_return_sequences=False  # we want a final hidden vector
)  # shape (None, last LSTM size)

# Concat the final LSTM state with the scalar y
xy_concat = layers.Concatenate(name="xy_concat")([x_encoded, encoder_input_y])  # shape (None, LSTM_UNITS_ENC + 1)

# Dense layer for further processing
h_enc = layers.Dense(DENSE_UNITS, activation="relu", name="encoder_dense")(xy_concat)

# z_mean and z_log_var
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(h_enc)         # shape (None, LATENT_DIM)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(h_enc)   # shape (None, LATENT_DIM)

# Build the encoder model
encoder = keras.Model(inputs=[encoder_input_x, encoder_input_y],
                      outputs=[z_mean, z_log_var],
                      name="Encoder")
encoder.summary()

# ################################## DECODER ##################################
# %%
# Define the decoder architecture
# Decoder Inputs:
#  - X again (30,36)
#  - z latent vector (LATENT_DIM)
decoder_input_x = keras.Input(shape=(TIME_STEPS, FEATURE_DIM), name="decoder_x")
decoder_input_z = keras.Input(shape=(LATENT_DIM,), name="decoder_z")

# Step A: Tile z over time dimension
#        shape (None, TIME_STEPS, LATENT_DIM)
def repeat_z(x):
    # x is (batch_size, LATENT_DIM)
    z_expanded = tf.expand_dims(x, axis=1)           # (batch, 1, latent_dim)
    z_tiled = tf.tile(z_expanded, [1, TIME_STEPS, 1])# (batch, TIME_STEPS, latent_dim)
    return z_tiled

z_tiled = layers.Lambda(repeat_z, name="repeat_z")(decoder_input_z) 
# shape => (None, 30, LATENT_DIM)

# Step B: Concatenate X and repeated z => shape (None, 30, FEATURE_DIM + LATENT_DIM)
xz_concat = layers.Concatenate(axis=-1, name="xz_concat")([decoder_input_x, z_tiled])

# Step C Stacked LSTM over (X + z)
dec_hidden = build_lstm_stack(
    input_tensor=xz_concat,
    lstm_units_list=LSTM_UNITS_DEC,
    name_prefix="decoder",
    final_return_sequences=False  # produce final hidden state
)

# Step D: Dense to produce y_mean, y_log_var
h_dec = layers.Dense(DENSE_UNITS, activation="relu", name="decoder_dense")(dec_hidden)
y_mean = layers.Dense(1, name="y_mean")(h_dec)
y_log_var = layers.Dense(1, name="y_log_var")(h_dec)

# Build the decoder model
decoder = keras.Model(inputs=[decoder_input_x, decoder_input_z],
                      outputs=[y_mean, y_log_var],
                      name="Decoder")
decoder.summary()

####################################### CONDITIONAL PRIOR #######################################
# %%
# 1) Input for the prior network
prior_input_x = keras.Input(shape=(TIME_STEPS, FEATURE_DIM), name="prior_input_x")

# 2) You can reuse your existing build_lstm_stack(...) if you like:
h_prior = build_lstm_stack(
    input_tensor=prior_input_x,
    lstm_units_list=LSTM_UNITS_PRIOR,  # or some other config
    name_prefix="prior",
    final_return_sequences=False
)

# 3) Dense layer for final aggregator
h_prior = layers.Dense(DENSE_UNITS, activation="relu", name="prior_dense")(h_prior)

# 4) Produce z_mean_prior and z_log_var_prior
z_mean_prior = layers.Dense(LATENT_DIM, name="z_mean_prior")(h_prior)
z_log_var_prior = layers.Dense(LATENT_DIM, name="z_log_var_prior")(h_prior)

# Build the prior model
prior_model = keras.Model(inputs=prior_input_x,
                          outputs=[z_mean_prior, z_log_var_prior],
                          name="ConditionalPrior")
prior_model.summary()
# %% 
# Define the sampling layer
class Sampling(layers.Layer):
    """Reparameterization trick: sample z ~ N(z_mean, exp(z_log_var))"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# %%
# Define the VAE as a Model 
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, prior_model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.prior_model = prior_model  # p(z | x)
        self.sampling = Sampling()
    
    def call(self, inputs, training=False):
        # inputs is [X, y]
        x, y = inputs 
        z_mean_q, z_log_var_q = self.encoder([x, y], training=training)
        z = self.sampling([z_mean_q, z_log_var_q])
        y_mean, y_log_var = self.decoder([x, z], training=training)
        return z_mean_q, z_log_var_q, y_mean, y_log_var
    
    def compute_losses(self, x_batch, y_batch, training):
        """Compute total_loss, recon_loss, kl_loss given a batch."""

        #1) Forward pass
        z_mean_q, z_log_var_q, y_mean, y_log_var = self((x_batch, y_batch), training=training)

        # 2) Reconstruction loss: -log p(y|z, X) (Gaussian assumption)
        var = K.exp(y_log_var)  # shape (batch_size, 1)
        recon_loss = 0.5 * (y_log_var + K.square(y_batch - y_mean) / (var + 1e-8))
        recon_loss = K.mean(recon_loss)  # average over the batch

        # 3) Get the prior distribution p(z|x)
        z_mean_p, z_log_var_p = self.prior_model(x_batch, training=training)

        # 2) KL divergence: D_kl( Normal(q_mu, q_var) || Normal(p_mu, p_var) ) per sample
        q_var = K.exp(z_log_var_q)
        p_var = K.exp(z_log_var_p)

        kl_per_dim = 0.5 * (
            z_log_var_p - z_log_var_q
            + (q_var + K.square(z_mean_q - z_mean_p)) / (p_var + 1e-8)
            - 1.
        )
        # sum across latent dim
        kl_per_sample = K.sum(kl_per_dim, axis=1)
        # average across batch
        kl_loss = K.mean(kl_per_sample)
        # 5) Combine losses
        total_loss = recon_loss + BETA * kl_loss
        return total_loss, recon_loss, kl_loss
        
    
    def train_step(self, data):
        """
        Custom training step. data == ((X_batch, y_batch), ) for typical Keras usage.
        Returns a dictionary of losses that Keras logs each batch and averages per epoch.
        """
        (x_batch, y_batch) = data
        
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self.compute_losses(x_batch, y_batch, training=True)
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Return a dict mapping metric names to current batch values
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }

    def test_step(self, data):
        """
        Custom validation step. This runs on the validation dataset
        at the end of each epoch. 
        Returns a dictionary of losses that Keras will log as 'val_loss', 
        'val_recon_loss', and 'val_kl_loss'.
        """
        (x_batch, y_batch) = data
        total_loss, recon_loss, kl_loss = self.compute_losses(x_batch, y_batch, training=False)
        
        return {
            "loss": total_loss,  # Keras will call this 'val_loss'
            "recon_loss": recon_loss,  # becomes 'val_recon_loss'
            "kl_loss": kl_loss  # becomes 'val_kl_loss'
        }

# %%
# Instantiate the cVAE with the LSTM encoder and decoder
cvae = CVAE(encoder=encoder, decoder=decoder, prior_model=prior_model)

# Optimizer
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Compile
cvae.compile(optimizer=optimizer)

# Suppose we have: 
# X_train shape => (133306, 30, 36)
# y_train shape => (133306,)
# X_val  shape => (14558, 30, 36)
# y_val  shape => (14558,)

# Convert y to shape (None, 1) if needed
y_train_2d = np.expand_dims(y_train, axis=-1)  # shape (133306, 1)
y_val_2d  = np.expand_dims(y_val, axis=-1)   # shape (14558, 1)


# %%
# Implement early stopping
# 7) Early Stopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",       # watch validation loss
    patience=3,              # stop after 3 epochs with no improvement
    restore_best_weights=True
)
# %%
# Train
history = cvae.fit(
    x=X_train, 
    y=y_train_2d,
    validation_data=(X_val, y_val_2d),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping_cb],
    verbose=1  # ensure we see detailed logs
)

# %%
# Plot training history
plt.figure(figsize=(8,6))
plt.plot(history.history["loss"], label="train_total_loss")
plt.plot(history.history["val_loss"], label="val_total_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
# %% 
# Function to sample based on the learned prior
def sample_y_distribution_cVAE(cvae, X_new, num_samples=100):
    """
    Generate multiple plausible y samples given X_new by sampling z from the prior.
    """
    # X_new shape => (30,36) for a single sample
    X_new = np.expand_dims(X_new, axis=0).astype("float32")  # (1, 30, 36)

    # 1) Get prior parameters from cVAE.prior_model
    z_mean_prior, z_log_var_prior = cvae.prior_model(X_new, training=False)
    z_mean_prior = z_mean_prior.numpy()[0]      # shape (LATENT_DIM,)
    z_log_var_prior = z_log_var_prior.numpy()[0] # shape (LATENT_DIM,)

    # 2) Sample multiple z from that Gaussian
    z_samples = []
    for _ in range(num_samples):
        eps = np.random.normal(size=z_mean_prior.shape)
        z = z_mean_prior + np.exp(0.5 * z_log_var_prior) * eps
        z_samples.append(z)
    z_samples = np.array(z_samples).astype("float32")  # shape (num_samples, LATENT_DIM)

    # 3) Pass each z (plus X_new) to decoder
    X_tiled = np.repeat(X_new, repeats=num_samples, axis=0) # (num_samples, TIME_STEPS, FEATURE_DIM)
    y_mean, y_log_var = cvae.decoder([X_tiled, z_samples], training=False)
    y_mean = y_mean.numpy().ravel()
    y_log_var = y_log_var.numpy().ravel()
    
    # 4) For each sample, we can either take the mean or sample from the predicted Gaussian (CURRENTLY WE THUS ASSUME Y GAUSSIAN, THIS IS NOT NON-PARAMETRIC)
    y_draws = []
    for i in range(num_samples):
        mean_i = y_mean[i]
        log_var_i = y_log_var[i]
        std_i = np.exp(0.5 * log_var_i)
        # sample
        y_i = np.random.normal(loc=mean_i, scale=std_i)
        y_draws.append(y_i)

    return np.array(y_draws)

# %%
# Defining stuff for printing
example_tickers = ["AAPL", "GS", "WMT"]

# %%
# Try to predict the next day's return for a sample
sample_idx = 0
X_sample = X_val[sample_idx]
samples = sample_y_distribution_cVAE(cvae, X_sample, num_samples=1000)
mean_est = np.mean(samples)
std_est = np.std(samples)
lower_95 = np.percentile(samples, 2.5)
upper_95 = np.percentile(samples, 97.5)
# plotting the distribution of the predicted returns
plt.hist(samples, bins=30, alpha=0.5, color="b", label="Predicted y samples")
plt.axvline(y_val[sample_idx], color="r", label="True y")
plt.axvline(mean_est, color="g", label="Mean pred")
plt.axvline(lower_95, color="g", linestyle="--", label="95% CI")
plt.axvline(upper_95, color="g", linestyle="--")
plt.legend()
plt.title("Distribution of predicted returns for a sample")
plt.show()
# %%
# Print smoothed distributions for a few example tickers for 5 random days
for ticker in example_tickers:
    from_idx, to_idx = all_data.validation.get_range(ticker)
    random_indices = np.random.choice(range(from_idx, to_idx), 5)
    for idx in random_indices:
        specific_sample = X_val[idx]  # Select a random test sample
        actual_return = y_val[idx].item()  # Actual next-day return
        samples = sample_y_distribution_cVAE(cvae, specific_sample, num_samples=1000)
        pred_mean = np.mean(samples)
        print(f"Ticker: {ticker}, Actual Return: {actual_return:.4f}, Predicted Mean: {pred_mean:.4f}")

        # Plot KDE with proper empirical representation
        plt.figure(figsize=(8, 4))
        sns.kdeplot(
            samples,
            bw_adjust=0.5,  # Adjust bandwidth for smoothing; lower values make it follow the data more closely
            fill=True,
            alpha=0.6,
            label="Predicted Distribution",
        )
        
        # Add vertical lines for actual and predicted return
        plt.axvline(
            pred_mean,
            color="blue",
            linestyle="dashed",
            linewidth=2,
            label="Predicted Mean",
        )
        plt.axvline(
            actual_return,
            color="red",
            linestyle="solid",
            linewidth=2,
            label="Actual Return",
        )

        plt.title(f"Predicted Return Distribution for {ticker} Test Point {idx}")
        plt.legend()
        plt.show()


# %%
# Save the model
cvae.save(f"models/{MODEL_NAME}.keras")

# %%
# Predict the conditional distribution for every sample in the validation set

# Function to predict the entire validation set
def predict_entire_val_set(cvae, X_val, n_draws=100):
    """
    Use the 'sample_y_distribution_cVAE' function on each item of X_val
    to get a distribution of y-values. Returns an array of shape (n_val, n_draws).
    
    Args:
        cvae: a trained cVAE model (with prior_model)
        X_val: shape (n_val, TIME_STEPS, FEATURE_DIM)
        n_draws: number of samples to draw per X_val[i]

    Returns:
        y_samples_2d: shape (n_val, n_draws), where each row i
                      is the set of y-draws for X_val[i].
    """
    n_val = X_val.shape[0]
    y_samples_2d = np.empty((n_val, n_draws), dtype=np.float32)

    # Loop over each sample in X_val
    for i in tqdm(range(n_val), desc="Predicting entire validation set"):
        X_single = X_val[i]  # shape (TIME_STEPS, FEATURE_DIM)
        # Use your sampling function
        y_draws = sample_y_distribution_cVAE(cvae, X_single, num_samples=n_draws)
        # Store the results in one row
        y_samples_2d[i, :] = y_draws

    return y_samples_2d

# VECTORIZED VERSION
def predict_distributions_vectorized_with_prior_draws(
    cvae,
    X_val,
    n_draws=100,
    chunk_size=2000
):
    """
    Vectorized approach that uses the LEARNED PRIOR MODEL to sample z for each X,
    then decodes, and FINALLY samples from the predicted Gaussian to get actual y draws.
    
    Args:
        cvae: your trained cVAE model, which has:
              - cvae.prior_model(x) => (z_mean_p, z_log_var_p)
              - cvae.decoder([x_tiled, z_samples_flat])
        X_val: shape (n_val, TIME_STEPS, FEATURE_DIM)
        n_draws: number of final draws (z -> decode -> y) per X
        chunk_size: how many X_val items to process at once (to limit memory usage)
    
    Returns:
        y_draws_2d:  shape (n_val, n_draws), each row i are the n_draws final samples
                     from p(y|x_i).
        y_mean_2d:   shape (n_val, n_draws), raw predicted means from the decoder
        y_log_var_2d: shape (n_val, n_draws), raw log-variances from the decoder
    """
    n_val = X_val.shape[0]
    latent_dim = cvae.decoder.input_shape[1][1]  # or known from your code
    y_draws_2d = np.empty((n_val, n_draws), dtype=np.float32)
    y_means_big = np.empty((n_val * n_draws,), dtype=np.float32)
    y_log_vars_big = np.empty((n_val * n_draws,), dtype=np.float32)

    current_position = 0
    
    start_idx = 0
    # implement tqdm progress bar

    with tqdm(total=n_val, desc="Predicting entire validation set") as pbar:

        while start_idx < n_val:
            end_idx = min(start_idx + chunk_size, n_val)
            X_chunk = X_val[start_idx:end_idx]  # shape (B, TIME_STEPS, FEATURE_DIM)
            B = X_chunk.shape[0]

            # --- 1) PRIOR p(z|x) -> means and log_vars for each sample in chunk
            z_mean_p, z_log_var_p = cvae.prior_model(X_chunk, training=False)
            z_mean_p = z_mean_p.numpy()         # shape (B, latent_dim)
            z_log_var_p = z_log_var_p.numpy()   # shape (B, latent_dim)

            # --- 2) For each x_i in chunk, sample n_draws for z
            eps_z = np.random.normal(size=(B, n_draws, latent_dim)).astype(np.float32)
            z_std_p = np.exp(0.5 * z_log_var_p)  # shape (B, latent_dim)
            # Expand dims so shape => (B, 1, latent_dim)
            z_std_p = np.expand_dims(z_std_p, axis=1)
            z_mean_p = np.expand_dims(z_mean_p, axis=1)

            # shape => (B, n_draws, latent_dim)
            z_samples = z_mean_p + z_std_p * eps_z

            # Flatten => (B * n_draws, latent_dim)
            z_samples_flat = z_samples.reshape(-1, latent_dim)

            # --- 3) Also tile X_chunk
            # shape => (B*n_draws, TIME_STEPS, FEATURE_DIM)
            X_chunk_expanded = np.repeat(X_chunk, repeats=n_draws, axis=0)

            # --- 4) Decode => (y_mean, y_log_var)
            y_mean_chunk, y_log_var_chunk = cvae.decoder([X_chunk_expanded, z_samples_flat], training=False)
            # shape => (B*n_draws, 1)
            y_mean_chunk = y_mean_chunk.numpy().ravel()
            y_log_var_chunk = y_log_var_chunk.numpy().ravel()

            # Store them in big arrays
            batch_size_times_draws = B * n_draws
            y_means_big[current_position : current_position + batch_size_times_draws] = y_mean_chunk
            y_log_vars_big[current_position : current_position + batch_size_times_draws] = y_log_var_chunk

            current_position += batch_size_times_draws
            start_idx = end_idx

            pbar.update(B)

    # --- Reshape to (n_val, n_draws)
    y_mean_2d = y_means_big.reshape(n_val, n_draws)
    y_log_var_2d = y_log_vars_big.reshape(n_val, n_draws)

    # --- 5) Final sample from Gaussian for each cell
    eps_y = np.random.normal(size=(n_val, n_draws)).astype(np.float32)
    std_2d = np.exp(0.5 * y_log_var_2d)
    y_draws_2d = y_mean_2d + std_2d * eps_y  # shape (n_val, n_draws)

    return y_draws_2d, y_mean_2d, y_log_var_2d

# %%
# Predict the distributions
# 1) Vectorized distribution forecast:
n_draws = 100
chunk_size = 2000
y_samples_2d, y_mean_2d, y_log_var_2d = predict_distributions_vectorized_with_prior_draws(
    cvae,
    X_val,        # shape (n_val, 30, 36)
    n_draws=n_draws,
    chunk_size=chunk_size
)
# y_samples_2d shape => (n_val, n_draws)
# %%
# 2) Compute your desired stats
confidence_levels = [0, 0.5, 0.67, 0.90, 0.95, 0.975, 0.99]
intervals = np.zeros((len(y_val), len(confidence_levels), 2), dtype=np.float32)
predicted_returns = y_samples_2d.mean(axis=1)  # shape (n_val,)
predicted_stds = y_samples_2d.std(axis=1)     # shape (n_val,)

# 3) Confidence intervals
for j, level in enumerate(confidence_levels):
    lower_percentile = 50 - level * 50
    upper_percentile = 50 + level * 50
    # shape => (n_val,) after applying percentile across axis=1 (the draws)
    intervals[:, j, 0] = np.percentile(y_samples_2d, lower_percentile, axis=1)
    intervals[:, j, 1] = np.percentile(y_samples_2d, upper_percentile, axis=1)

# %%
# plotting the distribution of the predicted returns for example tickers
for ticker in example_tickers:
    from_idx, to_idx = all_data.validation.get_range(ticker)

    plt.figure(figsize=(12, 5))
    
    # Actual vs Predicted Returns
    plt.plot(range(from_idx, to_idx), y_val[from_idx:to_idx], label="Actual Returns", color="red", linestyle="solid")
    plt.plot(range(from_idx, to_idx), predicted_returns[from_idx:to_idx], label="Predicted Mean Returns", color="blue", linestyle="dashed")

    # Standard Deviation
    plt.fill_between(
        range(from_idx, to_idx),
        np.array(predicted_returns[from_idx:to_idx]) - np.array(predicted_stds[from_idx:to_idx]),
        np.array(predicted_returns[from_idx:to_idx]) + np.array(predicted_stds[from_idx:to_idx]),
        color="blue",
        alpha=0.2,
        label="Predicted Return Std",
    )
    # Confidence Interval (e.g., 95%)
    plt.fill_between(
        range(from_idx, to_idx),
        intervals[from_idx:to_idx, 4, 0],  # Lower bound (95% confidence)
        intervals[from_idx:to_idx, 4, 1],  # Upper bound (95% confidence)
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval"
    )

    plt.title(f"Predicted Return Series for {ticker}")
    plt.legend()
    plt.show()

    # plot only the last 100 points
    plt.figure(figsize=(12, 5))
    plt.plot(range(to_idx-100, to_idx), y_val[to_idx-100:to_idx], label="Actual Returns", color="red", linestyle="solid")
    plt.plot(range(to_idx-100, to_idx), predicted_returns[to_idx-100:to_idx], label="Predicted Mean Returns", color="blue", linestyle="dashed")
    plt.fill_between(
        range(to_idx-100, to_idx),
        np.array(predicted_returns[to_idx-100:to_idx]) - np.array(predicted_stds[to_idx-100:to_idx]),
        np.array(predicted_returns[to_idx-100:to_idx]) + np.array(predicted_stds[to_idx-100:to_idx]),
        color="blue",
        alpha=0.2,
        label="Predicted Return Std",
    )
    plt.fill_between(
        range(to_idx-100, to_idx),
        intervals[to_idx-100:to_idx, 4, 0],  # Lower bound (95% confidence)
        intervals[to_idx-100:to_idx, 4, 1],  # Upper bound (95% confidence)
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval"
    )
    plt.title(f"Predicted Return Series for {ticker} (Last 100 Time Steps)")
    plt.legend()
    plt.show()

    # plot only the last 100 points
    plt.figure(figsize=(12, 5))
    plt.plot(range(to_idx-100, to_idx), y_val[to_idx-100:to_idx], label="Actual Returns", color="red", linestyle="solid")
    plt.plot(range(to_idx-100, to_idx), predicted_returns[to_idx-100:to_idx], label="Predicted Mean Returns", color="blue", linestyle="dashed")
    plt.fill_between(
        range(to_idx-100, to_idx),
        np.array(predicted_returns[to_idx-100:to_idx]) - np.array(predicted_stds[to_idx-100:to_idx]),
        np.array(predicted_returns[to_idx-100:to_idx]) + np.array(predicted_stds[to_idx-100:to_idx]),
        color="blue",
        alpha=0.2,
        label="Predicted Return Std",
    )
    plt.fill_between(
        range(to_idx-100, to_idx),
        intervals[to_idx-100:to_idx, 4, 0],  # Lower bound (95% confidence)
        intervals[to_idx-100:to_idx, 4, 1],  # Upper bound (95% confidence)
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval"
    )
    plt.title(f"Predicted Return Series for {ticker} (Last 100 Time Steps)")
    plt.legend()
    plt.show()

# %%
# Plotting example distributions for 5 random days for example tickers

for ticker in example_tickers:
    from_idx, to_idx = all_data.validation.get_range(ticker)
    random_indices = np.random.choice(range(from_idx, to_idx), 5)
    for idx in random_indices:
        specific_sample = X_val[idx]  # Select a random test sample
        actual_return = y_val[idx].item()  # Actual next-day return
        samples = sample_y_distribution_cVAE(cvae, specific_sample, num_samples=1000)
        pred_mean = np.mean(samples)
        print(f"Ticker: {ticker}, Actual Return: {actual_return:.4f}, Predicted Mean: {pred_mean:.4f}")

        # Plot KDE with proper empirical representation
        plt.figure(figsize=(8, 4))
        sns.kdeplot(
            samples,
            bw_adjust=0.5,  # Adjust bandwidth for smoothing; lower values make it follow the data more closely
            fill=True,
            alpha=0.6,
            label="Predicted Distribution",
        )
        
        # Add vertical lines for actual and predicted return
        plt.axvline(
            pred_mean,
            color="blue",
            linestyle="dashed",
            linewidth=2,
            label="Predicted Mean",
        )
        plt.axvline(
            actual_return,
            color="red",
            linestyle="solid",
            linewidth=2,
            label="Actual Return",
        )

        plt.title(f"Predicted Return Distribution for {ticker} Test Point {idx}")
        plt.legend()
        plt.show()


# %%
# Check for nan values in the predicted returns
nan_values = np.isnan(predicted_returns)
# print the number of nan values
print("Number of NaN values in predicted returns:", nan_values.sum())
# print the last predicted return
print("Last predicted return:", predicted_returns[-1])

# %%
# Check lenghts of the predicted returns and stds
print("predicted returns lenght:", len(predicted_returns))
print("predicted stds lenght:", len(predicted_stds))

# %%
# 8) Store single-pass predictions
df_validation = pd.DataFrame(
    np.vstack([all_data.validation.dates, all_data.validation.tickers]).T,
    columns=["Date", "Symbol"],
)
df_validation["Mean_SP"] = predicted_returns
df_validation["Vol_SP"] = predicted_stds
# df_validation["NLL"] = nll_loss

# Store the intervals in the DataFrame.
for j, cl in enumerate(confidence_levels):
    df_validation[f"LB_{int(100*cl)}"] = intervals[:, j, 0]
    df_validation[f"UB_{int(100*cl)}"] = intervals[:, j, 1]

df_validation

# %%
# Save the predictions to a CSV file
df_validation.set_index(["Date", "Symbol"]).to_csv(
    f"predictions/vae_lstm_v1{SUFFIX}.csv"
)
# %%
