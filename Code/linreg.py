# %%
# Define parameters
LOOKBACK_DAYS = 10
SUFFIX = "_stocks"  # Use "_stocks" for the single stocks or "" for S&P500 only
MODEL_NAME = f"lstm_log_var_{LOOKBACK_DAYS}_days{SUFFIX}"
TEST_ASSET = "GOOG"
DATA_PATH = "data/sp500_stocks.csv"
TRAIN_TEST_SPLIT = "2020-06-30"

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
    train_test_split_index = len(
        group[group.index.get_level_values("Date") < TRAIN_TEST_SPLIT]
    )

    # # Stack returns and squared returns together
    data = np.hstack((returns, log_sq_returns))

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
# Custom loss function: NLL loss for Gaussian distribution where y_pred is [mu, log(sigma^2)]
def nll_loss(y_true, y_pred):
    mu = y_pred[:, 0]
    log_sigma2 = y_pred[:, 1]
    sigma2 = tf.exp(log_sigma2)
    epsilon = 1e-6
    sigma2 = tf.maximum(sigma2, epsilon)

    term_1 = tf.square(y_true - mu) / (2 * sigma2)
    term_2 = 0.5 * tf.math.log(sigma2)

    loss = tf.reduce_mean(term_1 + term_2)
    return loss


# %%
# Custom loss function: NLL loss for Gaussian distribution where we ignore the mean prediction and only predict the variance
def nll_loss_variance_only(y_true, y_pred):
    # mu = y_pred[:, 0]
    mu = 0  # Assume mean is 0
    log_sigma2 = y_pred[:, 1]
    sigma2 = tf.exp(log_sigma2)
    epsilon = 1e-6
    sigma2 = tf.maximum(sigma2, epsilon)

    term_1 = tf.square(y_true - mu) / (2 * sigma2)
    term_2 = 0.5 * tf.math.log(sigma2)

    loss = tf.reduce_mean(term_1 + term_2)
    return loss


# %%
# Train and test a linear regression model (using NN modules from tensorflow to be able to output two values)
# This is to get a baseline for the LSTM model
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# Flatten the input so that we can directly connect it to the Dense layers
flattened = Flatten()(inputs)

# Regularization strength: choose a very small value, since this is just to stabilize the model
regularization_strength = 0  # 1e-7

# Mean output
mean_out = Dense(
    1,
    activation="linear",
    use_bias=True,
    kernel_regularizer=l2(regularization_strength),
    name="mean_out",
)(flattened)

# Output the log of variance (log(sigma^2)) so that we can use a linear activation
log_variance_out = Dense(
    1,
    activation="linear",
    use_bias=True,
    kernel_initializer="zeros",
    name="log_variance_out",
    kernel_regularizer=l2(regularization_strength),
)(flattened)

# Concatenate outputs
outputs = Concatenate()([mean_out, log_variance_out])

# Define and compile the model
linreg_model = Model(inputs=inputs, outputs=outputs)
linreg_model.compile(optimizer=Adam(learning_rate=0.001), loss=nll_loss)

# %%
# Set initial weights for log variance layer
var_weights = linreg_model.get_layer("log_variance_out").get_weights()
input_dim = var_weights[0].shape[0]  # Should be LOOKBACK_DAYS * num_features
num_time_steps = LOOKBACK_DAYS
num_features = 2  # 'returns' and 'log_sq_returns'

weights_matrix = var_weights[0].reshape((num_time_steps, num_features))
# Set weights for returns close to 0
weights_matrix[:, 0] = np.random.normal(0, 0.01, num_time_steps)  # 'returns' weights
# Set weights for log squared returns to a decaying pattern
log_sq_return_weights = 0.8 ** np.arange(num_time_steps, 0, -1)
log_sq_return_weights = log_sq_return_weights / np.sum(log_sq_return_weights)
weights_matrix[:, 1] = log_sq_return_weights

var_weights[0] = weights_matrix.reshape((input_dim, 1))
linreg_model.get_layer("log_variance_out").set_weights(var_weights)

# %%
# Set initial weights for mean layer to all zeros
mean_weights = linreg_model.get_layer("mean_out").get_weights()
mean_weights[0] = np.zeros_like(mean_weights[0])
linreg_model.get_layer("mean_out").set_weights(mean_weights)


# %%
# Load the model (if already trained)
linreg_model_fname = f"models/linreg_{LOOKBACK_DAYS}_days{SUFFIX}.h5"
if os.path.exists(linreg_model_fname):
    linreg_model = tf.keras.models.load_model(linreg_model_fname, compile=False)
    linreg_model.compile(optimizer=Adam(), loss=nll_loss)
    print("Loaded from disk")

# %%
# Fit the model
linreg_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# %%
# Save the model
linreg_model.save(linreg_model_fname)

# %%
# Predict expected return and volatility using the linear regression model
y_pred_linreg = linreg_model.predict(X_test)
log_variance_pred_linreg = y_pred_linreg[:, 1]
mean_pred_linreg = np.zeros_like(
    log_variance_pred_linreg
)  # Ignore mean pred for now # y_pred_linreg[:, 0]
variance_pred_linreg = np.exp(log_variance_pred_linreg)
volatility_pred_linreg = np.sqrt(variance_pred_linreg)

# %%
# Save predictions to file
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Mean"] = mean_pred_linreg
df_test["Volatility"] = volatility_pred_linreg
df_test.to_csv(f"predictions/linreg_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")

# %%
# Print the weights for the variance output with labels
input_types = []
time_lags = []
for time_index in range(LOOKBACK_DAYS):
    time_lag = LOOKBACK_DAYS - time_index - 1
    input_types.append("returns")
    time_lags.append(time_lag)
    input_types.append("log_sq_returns")
    time_lags.append(time_lag)

weights_matrix = linreg_model.get_layer("log_variance_out").get_weights()[0]

# Print weights with labels
weight_df = pd.DataFrame(
    {
        "Input Type": input_types,
        "Time Lag": time_lags,
        "Weight": weights_matrix.flatten(),
    }
)

# Plot log sq return weights
plt.figure(figsize=(12, 6))
plt.bar(
    weight_df[weight_df["Input Type"] == "log_sq_returns"]["Time Lag"],
    weight_df[weight_df["Input Type"] == "log_sq_returns"]["Weight"],
)
plt.title("Weights for log squared returns")
plt.xlabel("Time Lag")
plt.ylabel("Weight")
plt.show()

# Plot return weights
plt.figure(figsize=(12, 6))
plt.bar(
    weight_df[weight_df["Input Type"] == "returns"]["Time Lag"],
    weight_df[weight_df["Input Type"] == "returns"]["Weight"],
)
plt.title("Weights for returns")
plt.xlabel("Time Lag")
plt.ylabel("Weight")
plt.show()

weight_df
