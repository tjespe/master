# %%
# Define parameters
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    TEST_ASSET,
    DATA_PATH,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)

MODEL_NAME = f"lstm_mini_log_var_{LOOKBACK_DAYS}_days{SUFFIX}"

# %%
import numpy as np
from scipy.stats import chi2, norm
from scipy.special import erfinv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
    LayerNormalization,
    Flatten,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf
import warnings
import os
from tensorflow.keras.initializers import Constant, RandomNormal

warnings.filterwarnings("ignore")

# %%
df = pd.read_csv(DATA_PATH)

if not "Symbol" in df.columns:
    df["Symbol"] = TEST_ASSET

# Ensure the Date column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort the dataframe by both Date and Symbol
df = df.sort_values(["Symbol", "Date"])

# Calculate log returns for each instrument separately using groupby
df["LogReturn"] = (
    df.groupby("Symbol")["Close"]
    .apply(lambda x: np.log(x / x.shift(1)))
    .reset_index()["Close"]
)

# Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
df = df[~df["LogReturn"].isnull()]

df["SquaredReturn"] = df["LogReturn"] ** 2

# Set date and symbol as index
df: pd.DataFrame = df.set_index(["Date", "Symbol"])
df

# %%
# Check if TEST_ASSET is in the data
if TEST_ASSET not in df.index.get_level_values("Symbol"):
    raise ValueError(f"TEST_ASSET '{TEST_ASSET}' not found in the data")

# %%
# Filter away data before 1990
df = df[df.index.get_level_values("Date") >= "1990-01-01"]
df

# %%
# Prepare data for LSTM
X_train = []
X_test = []
y_train = []
y_test = []

# Group by symbol to handle each instrument separately
for symbol, group in df.groupby(level="Symbol"):
    # Extract the log returns and squared returns for the current group
    returns = group["LogReturn"].values.reshape(-1, 1)

    # Use the log of squared returns as the second input feature, because the
    # NN should output the log of variance
    log_sq_returns = np.log(
        group["SquaredReturn"].values.reshape(-1, 1)
        # For numerical stability: add a small constant to avoid log(0) equal to 0.1% squared
        + (0.1 / 100) ** 2
    )

    # Find date to split on
    TRAIN_VALIDATION_SPLIT_index = len(
        group[group.index.get_level_values("Date") < TRAIN_VALIDATION_SPLIT]
    )
    VALIDATION_TEST_SPLIT_index = len(
        group[group.index.get_level_values("Date") < VALIDATION_TEST_SPLIT]
    )

    # # Stack returns and squared returns together
    # data = np.hstack((returns, log_sq_returns))
    # Only use log squared returns for now
    data = log_sq_returns

    # Create training sequences of length 'sequence_length'
    for i in range(LOOKBACK_DAYS, TRAIN_VALIDATION_SPLIT_index):
        X_train.append(data[i - LOOKBACK_DAYS : i])
        y_train.append(returns[i, 0])

    # Add the test data
    if symbol == TEST_ASSET:
        for i in range(TRAIN_VALIDATION_SPLIT_index, VALIDATION_TEST_SPLIT_index):
            X_test.append(data[i - LOOKBACK_DAYS : i])
            y_test.append(returns[i, 0])

# Convert X and y to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Ensure float32 for X and y
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Print shapes
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")


# %%
# Custom loss function: NLL loss for Gaussian distribution where we ignore the mean prediction and only predict the variance
def nll_loss_variance_only(y_true, y_pred):
    mu = 0  # Assume mean is 0
    log_sigma2 = y_pred
    sigma2 = tf.exp(log_sigma2)
    epsilon = 1e-6
    sigma2 = tf.maximum(sigma2, epsilon)

    term_1 = tf.square(y_true - mu) / (2 * sigma2)
    term_2 = 0.5 * tf.math.log(sigma2)

    loss = tf.reduce_mean(term_1 + term_2)
    return loss


# %%
# Build the LSTM model
inputs = Input(shape=(X_train.shape[1], 1))

# LSTM layer
lstm_out = LSTM(
    units=1,
    activation="tanh",
)(inputs)

# Variance output with softplus activation
variance_out = Dense(1, activation="linear")(lstm_out)

# Define the model
model = Model(inputs=inputs, outputs=variance_out)


# %%
# Compile model
model.compile(optimizer=Adam(), loss=nll_loss_variance_only)

# %%
# Load the model (if already trained)
MODEL_FNAME = f"models/{MODEL_NAME}.h5"
if os.path.exists(MODEL_FNAME):
    model = tf.keras.models.load_model(MODEL_FNAME, compile=False)
    model.compile(optimizer=Adam(), loss=nll_loss_variance_only)

# %%
# Fit the model (can be repeated several times to train further)
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# %%
# Save the model
model.save(MODEL_FNAME)

# %%
# Predict expected return and volatility using the LSTM
y_pred = model.predict(X_test)
variance_pred = tf.exp(y_pred[:, 0])
mean_pred = np.zeros_like(variance_pred)  # Assume mean is 0
volatility_pred = np.sqrt(variance_pred)

# %%
# Save predictions to file
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
df_validation["Volatility"] = volatility_pred
df_validation["Mean"] = mean_pred
df_validation.to_csv(
    f"predictions/lstm_mini_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)

# %%
# Plot results of only LSTM
plt.figure(figsize=(12, 6))
plt.plot(
    df_validation.index, volatility_pred, label="Volatility Prediction", color="black"
)
plt.title("Volatility Prediction with LSTM")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

# %%
