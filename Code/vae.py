#https://chatgpt.com/share/67bd9d7a-d6a0-8008-86d6-157ffb922f08

# %%
# Define parameters
from settings import LOOKBACK_DAYS, SUFFIX, DATA_PATH, TRAIN_VALIDATION_SPLIT, VALIDATION_TEST_SPLIT, TEST_ASSET
MODEL_NAME = f"VAE_v1_{LOOKBACK_DAYS}_days{SUFFIX}"
# %%
# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from shared.processing import get_lstm_train_test_new

# %% 
# Load data
data = get_lstm_train_test_new(multiply_by_beta=False, include_fng=False, include_spx_data=True, include_returns=True)
X_train = data.train.X
y_train = data.train.y
X_val = data.validation.X
y_val = data.validation.y

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_val.shape)
print("y_test shape:", y_val.shape)

# %%
# Hyperparameters (you can tune these)
LATENT_DIM = 16      # dimension of the latent space 'z'
LSTM_UNITS = 32      # number of units in LSTM layers
DENSE_UNITS = 32     # hidden dense layer size after LSTM
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
BETA = 1             # weight for the KL

# %%
# Define the encoder architecture
# Encoder inputs
encoder_input_x = keras.Input(shape=(30, 36), name="encoder_x")   # (None, 30, 36)
encoder_input_y = keras.Input(shape=(1,), name="encoder_y")       # (None, 1)

# LSTM extracts a representation of X
encoder_lstm = layers.LSTM(LSTM_UNITS, return_sequences=False, name="encoder_lstm")
x_encoded = encoder_lstm(encoder_input_x)  # (None, LSTM_UNITS)

# Concatenate the LSTM output with y
xy_concat = layers.Concatenate(name="xy_concat")([x_encoded, encoder_input_y])  # shape (None, LSTM_UNITS+1)

# Dense layer for further processing
h_enc = layers.Dense(DENSE_UNITS, activation="relu")(xy_concat)  # (None, DENSE_UNITS)

# Two separate dense layers for z_mean and z_log_var
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(h_enc)      # (None, LATENT_DIM)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(h_enc)  # (None, LATENT_DIM)

# Build encoder model (we won't train it standalone, but it's nice to have a handle)
encoder = keras.Model(inputs=[encoder_input_x, encoder_input_y],
                      outputs=[z_mean, z_log_var],
                      name="encoder")
encoder.summary()

# %% 
# Define the sampleing layer
class Sampling(layers.Layer):
    """Reparameterization trick: sample z ~ N(z_mean, exp(z_log_var))"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# %% 
# Define the decoder architecture
decoder_input_x = keras.Input(shape=(30, 36), name="decoder_x")   # (None, 30, 36)
decoder_input_z = keras.Input(shape=(LATENT_DIM,), name="decoder_z")   # (None, LATENT_DIM)

decoder_lstm = layers.LSTM(LSTM_UNITS, return_sequences=False, name="decoder_lstm")
x_decoded = decoder_lstm(decoder_input_x)  # (None, LSTM_UNITS)

xz_concat = layers.Concatenate(name="xz_concat")([x_decoded, decoder_input_z])  # (None, LSTM_UNITS + LATENT_DIM)

h_dec = layers.Dense(DENSE_UNITS, activation="relu")(xz_concat)

# Output distribution parameters for y
y_mean = layers.Dense(1, name="y_mean")(h_dec)
y_log_var = layers.Dense(1, name="y_log_var")(h_dec)

decoder = keras.Model(inputs=[decoder_input_x, decoder_input_z],
                      outputs=[y_mean, y_log_var],
                      name="decoder")
decoder.summary()

# %%
# Define the VAE as a Model 
import tensorflow.keras.backend as K

class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()
    
    def call(self, inputs, training=False):
        """
        Forward pass: returns intermediate results needed for the loss.
        """
        x, y = inputs
        z_mean, z_log_var = self.encoder([x, y], training=training)
        z = self.sampling([z_mean, z_log_var])
        y_mean, y_log_var = self.decoder([x, z], training=training)
        return z_mean, z_log_var, y_mean, y_log_var
    
    def train_step(self, data):
        # data is ( (X_batch, y_batch), ) in typical Keras usage
        (x_batch, y_batch) = data
        
        with tf.GradientTape() as tape:
            z_mean, z_log_var, y_mean, y_log_var = self((x_batch, y_batch), training=True)
            
            # 1) Reconstruction loss: negative log p(y|x,z)
            #    Gaussian log-likelihood with mean=y_mean, var=exp(y_log_var)
            var = K.exp(y_log_var)  # shape (batch_size, 1)
            recon_loss = 0.5 * (y_log_var + K.square(y_batch - y_mean) / var)
            # We can sum over the batch or mean; typically we do mean
            recon_loss = K.mean(recon_loss)
            
            # 2) KL divergence: q(z|x,y) vs p(z)=N(0,I)
            kl_loss = -0.5 * K.sum(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
                axis=1
            )
            kl_loss = K.mean(kl_loss)
            
            total_loss = recon_loss + BETA*kl_loss
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Track losses
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

# %%
# Instantiate the model
# Instantiate cVAE
cvae = CVAE(encoder, decoder)

# Choose an optimizer (hyperparameter)
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

cvae.compile(optimizer=optimizer)

# %%
# Train the model
history = cvae.fit([X_train, y_train],
                   y=None,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   validation_data=([X_val, y_val], None))


# %% 
# Sample from prior
def sample_y_distribution(cvae, X_new, num_samples=100):
    """
    Return an array of shape (num_samples,) with plausible y values
    given X_new by sampling from the latent prior and decoding.
    """
    # Expand dims so X_new is shape (1, 30, 36)
    X_new = np.expand_dims(X_new, axis=0).astype("float32")

    # Sample multiple z's from N(0, I)
    z_dim = LATENT_DIM
    z_samples = np.random.normal(size=(num_samples, z_dim)).astype("float32")
    
    y_samples = []
    for i in range(num_samples):
        z_i = np.expand_dims(z_samples[i], axis=0)  # shape (1, LATENT_DIM)
        
        # Pass (X_new, z_i) to the decoder
        y_mean, y_log_var = cvae.decoder([X_new, z_i], training=False)
        # Convert Tensors to numpy
        mean_val = y_mean.numpy()[0, 0]
        log_var_val = y_log_var.numpy()[0, 0]
        std_val = np.exp(0.5 * log_var_val)
        
        # We can draw from the predicted Gaussian
        y_sample = np.random.normal(loc=mean_val, scale=std_val)
        y_samples.append(y_sample)
    
    return np.array(y_samples)
