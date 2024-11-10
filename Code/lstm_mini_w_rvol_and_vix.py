# %%
# Define parameters
from settings import LOOKBACK_DAYS, SUFFIX, TEST_ASSET, DATA_PATH, TRAIN_TEST_SPLIT

MODEL_NAME = f"lstm_mini_w_rvol_and_vix_log_var_{LOOKBACK_DAYS}_days{SUFFIX}"
RVOL_DATA_PATH = "data/RVOL.csv"
VIX_DATA_PATH = "data/VIX.csv"

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
# Read RVOL data
rvol_df = pd.read_csv(RVOL_DATA_PATH)
rvol_df["Date"] = pd.to_datetime(rvol_df["Date"])
rvol_df = rvol_df.set_index("Date")
rvol_df

# %%
# Read VIX data
vix_df = pd.read_csv(VIX_DATA_PATH)
vix_df["Date"] = pd.to_datetime(vix_df["Date"])
vix_df = vix_df.set_index("Date")
vix_df

# %%
# Check if TEST_ASSET is in the data
if TEST_ASSET not in df.index.get_level_values("Symbol"):
    raise ValueError(f"TEST_ASSET '{TEST_ASSET}' not found in the data")

# %%
# Filter away data we don't have RVOL data for
df = df[df.index.get_level_values("Date") >= rvol_df.index[0]]
df = df[df.index.get_level_values("Date") <= rvol_df.index[-1]]
df

# %%
# Filter away data we don't have VIX data for
df = df[df.index.get_level_values("Date") >= vix_df.index[0]]
df = df[df.index.get_level_values("Date") <= vix_df.index[-1]]
df

# %%
# Add RVOL data to the dataframe
df = df.join(rvol_df, how="left", rsuffix="_RVOL")
df

# %%
# Add VIX data to the dataframe
df = df.join(vix_df, how="left", rsuffix="_VIX")

# %%
# Check for NaN values
nan_mask = df[["LogReturn", "Close_RVOL", "Close_VIX"]].isnull().sum(axis=1).gt(0)
df[nan_mask]

# %%
# Impute missing RVOL values using the last available value
df["Close_RVOL"] = df["Close_RVOL"].fillna(method="ffill")
df[nan_mask]

# %%
# Impute missing VIX values using the last available value
df["Close_VIX"] = df["Close_VIX"].fillna(method="ffill")
df[nan_mask]

# %%
# Check for NaN values
df[df[["LogReturn", "Close_RVOL", "Close_VIX"]].isnull().sum(axis=1).gt(0)]

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

    # Extract realized volatility and transform it to a similar scale
    rvol_annualized = group["Close_RVOL"].values.reshape(-1, 1) / 100
    rvol_daily = rvol_annualized / np.sqrt(252)
    rvol = np.log(rvol_daily**2 + (0.1 / 100) ** 2)

    # Extract VIX and transform it to a similar scale
    vix_annualized = group["Close_VIX"].values.reshape(-1, 1) / 100
    vix_daily = vix_annualized / np.sqrt(252)
    vix = np.log(vix_daily**2 + (0.1 / 100) ** 2)

    # Find date to split on
    train_test_split_index = len(
        group[group.index.get_level_values("Date") < TRAIN_TEST_SPLIT]
    )

    # Stack returns and squared returns together
    data = np.hstack((log_sq_returns, rvol))

    # Create training sequences of length 'sequence_length'
    for i in range(LOOKBACK_DAYS, train_test_split_index):
        X_train.append(data[i - LOOKBACK_DAYS : i])
        y_train.append(returns[i, 0])

    # Add the test data
    if symbol == TEST_ASSET:
        for i in range(train_test_split_index, len(data)):
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
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

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
# Load the model (if already trained)
MODEL_FNAME = f"models/{MODEL_NAME}.h5"
if os.path.exists(MODEL_FNAME):
    model = tf.keras.models.load_model(MODEL_FNAME, compile=False)

# %%
# Fit the model (can be repeated several times to train further)
# First fit with high learning rate to quickly get close to the optimal solution
model.compile(optimizer=Adam(learning_rate=1e-2), loss=nll_loss_variance_only)
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# %%
# Then fit with lower learning rate to fine-tune the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss=nll_loss_variance_only)
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

# %%
# Then fit with even lower learning rate to fine-tune the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss=nll_loss_variance_only)
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

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
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Volatility"] = volatility_pred
df_test["Mean"] = mean_pred
df_test.to_csv(
    f"predictions/lstm_mini_w_rvol_and_vix_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)


# %%
# Get test part of df
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test

# %%
# Plot results of only LSTM
plt.figure(figsize=(12, 6))
plt.plot(df_test.index, volatility_pred, label="Volatility Prediction", color="black")
plt.title("Volatility Prediction with LSTM")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

# %%
