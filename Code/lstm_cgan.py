# %%
# Define parameters (imported from your settings)
from shared.processing import get_cgan_train_test
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    TEST_ASSET,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)

VERSION = 1
MODEL_NAME = f"lstm_cgan_{LOOKBACK_DAYS}_days{SUFFIX}_v{VERSION}"

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam

import os
from scipy.stats import norm  # might be useful for optional tasks

# %%
# Hyperparameters
NOISE_DIM = 16  # Dimension of random noise input to the generator
G_LR = 1e-4  # Generator learning rate
D_LR = 1e-4  # Discriminator learning rate
BATCH_SIZE = 32
EPOCHS = 50

# %%
# 0) Get data
df, X_train, X_test, y_train, y_test, scaling_mean, scaling_std = get_cgan_train_test()


# %%
# 1) Build the Generator
#    The generator takes in (X, noise) and produces next-day return y_fake.
def build_generator(
    lookback_days, num_features, noise_dim, hidden_units=2, dropout=0.2
):
    """
    A simple LSTM-based generator for cGAN.
    Inputs:
      - X: shape (batch, lookback_days, num_features)
      - noise z: shape (batch, noise_dim)
    Output:
      - y_fake: shape (batch, 1), the “generated” next-day return
    """
    # Conditional input: X
    x_input = Input(shape=(lookback_days, num_features), name="generator_X_input")
    # Random noise input
    z_input = Input(shape=(noise_dim,), name="generator_noise_input")

    # LSTM or anything you want
    x = LSTM(hidden_units, activation="tanh")(x_input)
    x = Dropout(dropout)(x)

    # Combine LSTM output with noise
    combined = Concatenate()([x, z_input])
    # Optionally pass combined through a small MLP
    x = Dense(hidden_units, activation="relu")(combined)
    x = Dropout(dropout)(x)

    # Output a single scalar (the predicted next-day return)
    y_fake = Dense(1, activation="linear")(x)

    generator = Model(inputs=[x_input, z_input], outputs=y_fake, name="Generator")
    return generator


# %%
# 2) Build the Discriminator
#    The discriminator sees (X, y) and outputs a single logit or probability
#    indicating whether (X, y) is real or fake.
def build_discriminator(
    lookback_days, num_features, hidden_units_lstm=4, hidden_units_mlp=10, dropout=0.2
):
    """
    A simple LSTM-based discriminator for cGAN.
    Inputs:
      - X: shape (batch, lookback_days, num_features)
      - y: shape (batch, 1)
    Output:
      - logit or probability that input is real (vs. fake)
    """
    # Conditional input: X
    x_input = Input(shape=(lookback_days, num_features), name="disc_X_input")
    # Next-day return
    y_input = Input(shape=(1,), name="disc_y_input")

    x = LSTM(hidden_units_lstm, activation="tanh")(x_input)
    x = Dropout(dropout)(x)

    # Concatenate X encoding with the candidate y
    combined = Concatenate()([x, y_input])
    x = Dense(hidden_units_mlp, activation="relu")(combined)
    x = Dropout(dropout)(x)

    # Output logit (or probability)
    # For standard cGAN, a single output logit with sigmoid activation
    out = Dense(1, activation="sigmoid")(x)

    discriminator = Model(inputs=[x_input, y_input], outputs=out, name="Discriminator")
    return discriminator


# %%
# 3) Assemble the cGAN
#    - We'll keep generator and discriminator separate but also define a "combined" model
#      that is used to train the generator in a typical adversarial fashion.
generator = build_generator(
    lookback_days=LOOKBACK_DAYS,
    num_features=X_train.shape[2],
    noise_dim=NOISE_DIM,
)

# %%
# Load generator from file
generator_fname = f"models/{MODEL_NAME}_generator.keras"
if os.path.exists(generator_fname):
    generator = tf.keras.models.load_model(generator_fname)

# %%
discriminator = build_discriminator(
    lookback_days=LOOKBACK_DAYS,
    num_features=X_train.shape[2],
)

# %%
# Load discriminator from file
discriminator_fname = f"models/{MODEL_NAME}_discriminator.keras"
if os.path.exists(discriminator_fname):
    discriminator = tf.keras.models.load_model(discriminator_fname)

# %%
# Compile discriminator (trained stand-alone)
discriminator.compile(
    loss="binary_crossentropy", optimizer=Adam(learning_rate=D_LR), metrics=["accuracy"]
)

# %%
# Build combined model:
#   We input X + noise to generator => y_fake
#   Then feed (X, y_fake) to the discriminator
#   We want to train generator to fool the discriminator (so label=1 for fake).
#   So the generator sees a compiled model with frozen discriminator.

# Freeze discriminator weights in the combined model
discriminator.trainable = False

# Gan input: X_input, z_input
gan_X_input = Input(shape=(LOOKBACK_DAYS, X_train.shape[2]))
gan_noise_input = Input(shape=(NOISE_DIM,))

y_fake = generator([gan_X_input, gan_noise_input])
disc_out = discriminator([gan_X_input, y_fake])

# The combined model goes from (X, noise) -> disc_out
combined_model = Model([gan_X_input, gan_noise_input], disc_out, name="Combined_cGAN")
combined_model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=G_LR))

# Unfreeze discriminator (for direct usage)
discriminator.trainable = True


# %%
# 4) cGAN Training Loop
#    We'll define a simple training loop that goes through mini-batches.
X_disc_debug = None
y_disc_debug = None
labels_disc_debug = None


def train_cgan(
    generator,
    discriminator,
    combined_model,
    X_train,
    y_train,
    noise_dim=NOISE_DIM,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
):
    global X_disc_debug, y_disc_debug, labels_disc_debug
    # Number of training samples
    n_samples = X_train.shape[0]
    # Number of steps per epoch
    steps_per_epoch = n_samples // batch_size

    for epoch in range(epochs):
        # Shuffle data indices
        idxs = np.arange(n_samples)
        np.random.shuffle(idxs)

        for step in range(steps_per_epoch):
            # Get a batch of real data
            batch_idxs = idxs[step * batch_size : (step + 1) * batch_size]
            real_X = X_train[batch_idxs]
            real_y = y_train[batch_idxs].reshape(-1, 1)

            # Train Discriminator
            # --------------------
            # 1) Real pairs (X, y)
            real_labels = np.ones((batch_size, 1), dtype=np.float32)

            # 2) Fake pairs (X, y_fake)
            noise = np.random.normal(0, 1, (batch_size, noise_dim)).astype(np.float32)
            y_fake = generator.predict([real_X, noise], verbose=0)
            fake_labels = np.zeros((batch_size, 1), dtype=np.float32)

            # Combine real+fake for a single update to D
            X_disc = np.concatenate([real_X, real_X], axis=0)
            y_disc = np.concatenate([real_y, y_fake], axis=0)
            labels_disc = np.concatenate([real_labels, fake_labels], axis=0)
            X_disc_debug = X_disc
            y_disc_debug = y_disc
            labels_disc_debug = labels_disc

            # Perform D step
            d_loss, d_acc = discriminator.train_on_batch([X_disc, y_disc], labels_disc)

            # Train Generator
            # ---------------
            # We want G to produce y_fake that makes D output "1"
            # So we flip the labels to all 1 for fake
            noise = np.random.normal(0, 1, (batch_size, noise_dim)).astype(np.float32)
            valid_labels = np.ones((batch_size, 1), dtype=np.float32)

            # The combined model input is [X, noise], target is 1.0 for tricking D
            g_loss = combined_model.train_on_batch([real_X, noise], valid_labels)

        # Print progress
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Discriminator loss: {d_loss:.4f}, Discriminator accuracy: {d_acc:.4f} | Generator loss: {g_loss:.4f}"
        )


# %%
# 5) Run the training
train_cgan(generator, discriminator, combined_model, X_train, y_train)

# %%
# 6) Save models
generator.save(generator_fname)
discriminator.save(discriminator_fname)


# %%
# 7) Use the trained generator to create an empirical distribution for test data
#    We'll sample multiple times from the generator for each X_test row.
def cgan_predict_distribution(generator, X, T=100, noise_dim=NOISE_DIM):
    """
    Generate T samples for each item in X, returning an array of shape (len(X), T).
    """
    n_test = X.shape[0]
    all_samples = np.zeros((n_test, T), dtype=np.float32)
    for t in range(T):
        noise = np.random.normal(0, 1, (n_test, noise_dim)).astype(np.float32)
        y_fake = generator.predict([X, noise], verbose=0)
        all_samples[:, t] = y_fake[:, 0]  # shape (batch,)
    return all_samples


# %%
def compute_empirical_intervals(samples, confidence_levels=[0.5, 0.67, 0.95, 0.99]):
    """
    Given samples of shape (n_samples, T), compute intervals for each row.
    We return an array intervals of shape (n_samples, len(confidence_levels), 2).
    intervals[i, j, 0] = lower bound
    intervals[i, j, 1] = upper bound
    at confidence level confidence_levels[j].
    """
    n_samples = samples.shape[0]
    n_conf = len(confidence_levels)
    intervals = np.zeros((n_samples, n_conf, 2), dtype=np.float32)

    # Sort each row’s samples
    sorted_samples = np.sort(samples, axis=1)
    T = samples.shape[1]

    for i, cl in enumerate(confidence_levels):
        alpha_lower = 0.5 * (1 - cl)
        alpha_upper = 1 - alpha_lower

        lower_idx = int(np.floor(alpha_lower * T))
        upper_idx = int(np.floor(alpha_upper * T))
        lower_idx = max(0, min(lower_idx, T - 1))
        upper_idx = max(0, min(upper_idx, T - 1))

        intervals[:, i, 0] = sorted_samples[:, lower_idx]
        intervals[:, i, 1] = sorted_samples[:, upper_idx]

    return intervals


# %%
# 8) Predict on X_test
T_SAMPLES = 1000  # number of draws per test sample, choose as you like
output_samples = cgan_predict_distribution(generator, X_test, T=T_SAMPLES)
# Scaled back to original scale
samples_test = output_samples / 100

# %%
# Check how the discriminator classifies the generated samples
# (should be close to 0.5 for good generator)
fool_pcts = []
for i in range(T_SAMPLES):
    y_fake = output_samples.reshape(-1, 1)
    noise = np.random.normal(0, 1, (len(y_fake), NOISE_DIM)).astype(np.float32)
    labels = np.ones((len(y_fake), 1), dtype=np.float32)
    d_loss, d_acc = discriminator.evaluate([X_test, y_fake], labels)
    fool_pcts.append(d_acc)
print(f"Mean fooling percentage: {np.mean(fool_pcts):.4f}")


# %%
# Plot empirical distributions for 10 random test samples along with the true return
plt.figure(figsize=(10, 40))
np.random.seed(0)
days = np.random.randint(0, len(y_test), 10)
days = np.sort(days)[::-1]
for i, day in enumerate(days):
    plt.subplot(10, 1, i + 1)
    timestamp = df.index[-day][0]
    plt.hist(
        samples_test[-day],
        bins=50,
        density=True,
        alpha=0.7,
        label="cGAN Empirical Dist",
    )
    true_val = y_test[-day] / 100
    plt.axvline(true_val, color="red", linestyle="--", label="Actual return")
    plt.axvline(
        np.mean(samples_test[-day]),
        color="blue",
        linestyle="--",
        label="Predcited mean",
    )
    # Plot quantiles
    for q in [0.025, 0.975]:
        q_val = np.quantile(samples_test[-day], q)
        plt.axvline(q_val, color="black", linestyle="--", alpha=0.7)
        plt.text(
            q_val,
            0,
            f"{q*100:.1f}%",
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="right",
            color="black",
            alpha=0.7,
        )
    plt.title(f"Sample cGAN distribution ({timestamp.strftime('%Y-%m-%d')})")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.xlim(-0.05, 0.05)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.legend()
plt.tight_layout()
plt.show()

# %%
# Compute cGAN mean and volatility
mean_cgan = np.mean(samples_test, axis=1)
std_cgan = np.std(samples_test, axis=1)

# %%
# Confidence intervals
confidence_levels = [0.5, 0.67, 0.95, 0.975, 0.99]
intervals_cgan = compute_empirical_intervals(samples_test, confidence_levels)

# %%
# 9) Store predictions in a DataFrame
df_validation = (
    df.xs(TEST_ASSET, level="Symbol")
    .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    .copy()
)
df_validation = df_validation.iloc[-len(X_test) :]  # align shapes if needed

df_validation["Mean_cGAN"] = mean_cgan
df_validation["Vol_cGAN"] = std_cgan

# Add intervals
for i, cl in enumerate(confidence_levels):
    df_validation[f"LB_{int(100*cl)}"] = intervals_cgan[:, i, 0]
    df_validation[f"UB_{int(100*cl)}"] = intervals_cgan[:, i, 1]

# %%
# Save
df_validation.to_csv(
    f"predictions/lstm_cgan_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_v{VERSION}.csv"
)

# %%
# Plot the mean and volatility of the cGAN predictions over time, along with the true return
lookback = 50
shift = 1
df_validation_filtered = df_validation.iloc[-lookback - shift : -shift]
plt.figure(figsize=(15, 5))
plt.plot(
    df_validation_filtered.index,
    df_validation_filtered["Mean_cGAN"],
    label="cGAN Mean",
    color="black",
)
for i, cl in enumerate(confidence_levels[1:]):
    plt.fill_between(
        df_validation_filtered.index,
        df_validation_filtered[f"LB_{int(100*cl)}"],
        df_validation_filtered[f"UB_{int(100*cl)}"],
        alpha=0.5 - i * 0.1,
        label=f"{int(100*cl)}% CI",
        color="C0",
    )
plt.plot(
    df_validation_filtered.index,
    df_validation_filtered["LogReturn"],
    label="True Return",
    linestyle="--",
    color="red",
)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.title(f"cGAN Predictions ({TEST_ASSET})")
plt.ylabel("Return")
plt.legend()
plt.show()

# %%
# 10) (Optional) Example plot of cGAN distribution for a random day
import random

random_idx = random.randint(0, len(X_test) - 1)
n_points = 200
hist_range = (-0.1, 0.1)
plt.figure(figsize=(7, 4))
plt.hist(
    samples_test[random_idx],
    bins=n_points,
    range=hist_range,
    density=True,
    alpha=0.7,
    label="cGAN Empirical Dist",
)
true_val = y_test[random_idx]
plt.axvline(true_val, color="red", linestyle="--", label="Actual return")
plt.title(f"Sample cGAN distribution (Test idx={random_idx})")
plt.xlabel("Return")
plt.ylabel("Density")
plt.legend()
plt.show()
