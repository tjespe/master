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

MODEL_NAME = f"lstm_log_var_{LOOKBACK_DAYS}_days{SUFFIX}"

# %%
from shared.mc_dropout import predict_with_mc_dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
    LayerNormalization,
)
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf
import warnings
import os

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
    df.groupby("Symbol")["Close"].apply(lambda x: np.log(x / x.shift(1))).droplevel(0)
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
    data = np.hstack((returns, log_sq_returns))

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
# Scale data for NN

# Scale the input data using one scaler for the first column and another for the second
scaler_1 = StandardScaler()
scaler_2 = StandardScaler()

# Fit the scalers
scaler_1.fit(X_train[:, :, 0])
scaler_2.fit(X_train[:, :, 1])

# Transform the data
scaled_X_train = X_train.copy()
scaled_X_train[:, :, 0] = scaler_1.transform(scaled_X_train[:, :, 0])
# Don't scale log squared returns for now
# scaled_X_train[:, :, 1] = scaler_2.transform(scaled_X_train[:, :, 1])

scaled_X_test = X_test.copy()
scaled_X_test[:, :, 0] = scaler_1.transform(scaled_X_test[:, :, 0])
# Don't scale log squared returns for now
# scaled_X_test[:, :, 1] = scaler_2.transform(scaled_X_test[:, :, 1])

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")

# %%
scaled_X_train = scaled_X_train.astype(np.float32)
scaled_X_test = scaled_X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# %%
# Build the LSTM model

inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# LSTM layer
lstm_out = LSTM(
    units=200,
    activation="tanh",
)(inputs)
lstm_out = LayerNormalization()(lstm_out)
lstm_out = Dropout(0.1)(lstm_out)

# Mean output
mean_out = Dense(1, activation="linear")(lstm_out)

# Output the log of variance (log(sigma^2)) so that we can use a linear activation
log_variance_out = Dense(1, activation="linear")(lstm_out)

# Concatenate outputs
outputs = Concatenate()([mean_out, log_variance_out])

# Define the model
model = Model(inputs=inputs, outputs=outputs)


# %%
# Compile model
lr = 0.0001
model.compile(optimizer=Adam(learning_rate=lr), loss=nll_loss)

# %%
# Load the model (if already trained)
model_fname = f"models/{MODEL_NAME}.h5"
if os.path.exists(model_fname):
    model = tf.keras.models.load_model(model_fname, compile=False)
    model.compile(optimizer=Adam(learning_rate=lr), loss=nll_loss)
    print("Loaded from disk")

# %%
# Fit the model (can be repeated several times to train further)
model.fit(scaled_X_train, y_train, epochs=15, batch_size=32, verbose=1)

# %%
# Save the model
model.save(model_fname)

# %%
# Predict expected return and volatility using the LSTM
y_pred = model.predict(scaled_X_test)
mean_pred = y_pred[:, 0]
log_variance_pred = y_pred[:, 1]
variance_pred = np.exp(log_variance_pred)
variance_pred = np.maximum(variance_pred, 1e-6)
volatility_pred = np.sqrt(variance_pred)

# %%
# Save predictions to file
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
df_validation["Mean"] = mean_pred
df_validation["Volatility"] = volatility_pred
df_validation.to_csv(
    f"predictions/lstm_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)


# %%
# Make predictions for regular LSTM with MC Dropout
mc_results = predict_with_mc_dropout(model, scaled_X_test, T=100)

# %%
# Save predictions to file
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
df_validation["Mean"] = mc_results["expected_returns"]
df_validation["Volatility"] = mc_results["volatility_estimates"]
df_validation["Epistemic_Uncertainty_Volatility"] = mc_results[
    "epistemic_uncertainty_volatility_estimates"
]
df_validation["Epistemic_Uncertainty_Mean"] = mc_results[
    "epistemic_uncertainty_expected_returns"
]
df_validation.to_csv(
    f"predictions/lstm_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)

# %%
# Plot the distribution of expected returns for the first day
plt.figure(figsize=(12, 6))
plt.hist(mc_results["preds"][:, 0, 0], bins=20, alpha=0.5, label="Expected Returns")
plt.axvline(y_test[0], color="red", label="True Return")
plt.title("Distribution of Expected Returns for the First Day")
plt.xlabel("Expected Return")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %%
# Plot the distribution of estimated volatilities for the first day
plt.figure(figsize=(12, 6))
plt.hist(
    np.sqrt(np.exp(mc_results["preds"][:, 0, 1])),
    bins=20,
    alpha=0.5,
    label="Estimated Volatilities",
)
plt.axvline(np.abs(y_test[0]), color="red", label="Actual absolute return")
plt.title("Distribution of Estimated Volatilities for the First Day")
plt.xlabel("Estimated Volatility")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %%
# Plot the distribution of estimated volatilities across all days
plt.figure(figsize=(12, 6))
plt.hist(
    np.sqrt(np.exp(mc_results["preds"][:, :, 1])).flatten(),
    bins=30,
    alpha=0.5,
    label="Estimated Volatilities",
)
plt.title("Distribution of Estimated Volatilities Across All Days")
plt.xlabel("Estimated Volatility")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %%
# Load GARCH(1, 1) predictions from file
garch_vol_pred = pd.read_csv(
    f"predictions/garch_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)["Volatility"]

# %%
# Get test part of df
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
abs_returns_test = df_validation["LogReturn"].abs()

# Training and test data
returns_train = (
    df.xs(TEST_ASSET, level="Symbol")["LogReturn"].loc[:TRAIN_VALIDATION_SPLIT] * 100
)  # Scale to percentages
returns_test = df_validation["LogReturn"]

# %%
# Plot bounds of MC against actual returns last X days
lookback_days = 100
idx = len(returns_test) - lookback_days
plt.figure(figsize=(12, 6))
plt.plot(
    returns_test.index[idx:],
    returns_test[idx:],
    label="Actual Returns",
    color="black",
)
plt.plot(
    returns_test.index[idx:],
    mc_results["expected_returns"][idx:],
    label="MC Dropout Expected Returns",
    color="blue",
)
plt.fill_between(
    returns_test.index[idx:],
    mc_results["expected_returns"][idx:] - mc_results["volatility_estimates"][idx:],
    mc_results["expected_returns"][idx:] + mc_results["volatility_estimates"][idx:],
    color="blue",
    alpha=0.6,
    label="67% Prediction Interval",
)
plt.fill_between(
    returns_test.index[idx:],
    mc_results["expected_returns"][idx:] - 2 * mc_results["volatility_estimates"][idx:],
    mc_results["expected_returns"][idx:] + 2 * mc_results["volatility_estimates"][idx:],
    color="blue",
    alpha=0.3,
    label="95% Prediction Interval",
)
plt.title(
    f"MC Dropout Prediction Interval without epistemic uncertainty (Last {lookback_days} days)"
)
plt.xlabel("Date")
plt.ylabel("Returns")
plt.legend()
plt.show()
cov_67 = np.mean(
    (
        returns_test[idx:]
        >= mc_results["expected_returns"][idx:]
        - mc_results["volatility_estimates"][idx:]
    )
    & (
        returns_test[idx:]
        <= mc_results["expected_returns"][idx:]
        + mc_results["volatility_estimates"][idx:]
    )
)
cov_67_garch = np.mean(
    (returns_test[idx:] >= -garch_vol_pred[idx:])
    & (returns_test[idx:] <= garch_vol_pred[idx:])
)
cov_95 = np.mean(
    (
        returns_test[idx:]
        >= mc_results["expected_returns"][idx:]
        - 1.96 * mc_results["volatility_estimates"][idx:]
    )
    & (
        returns_test[idx:]
        <= mc_results["expected_returns"][idx:]
        + 1.96 * mc_results["volatility_estimates"][idx:]
    )
)
cov_95_garch = np.mean(
    (returns_test[idx:] >= -1.96 * garch_vol_pred[idx:])
    & (returns_test[idx:] <= 1.96 * garch_vol_pred[idx:])
)
mean_width_67 = np.mean(mc_results["volatility_estimates"][idx:])
mean_width_67_garch = np.mean(garch_vol_pred[idx:])
mean_width_95 = np.mean(1.96 * mc_results["volatility_estimates"][idx:] * 2)
mean_width_95_garch = np.mean(1.96 * garch_vol_pred[idx:] * 2)

print(f"Stats for last {lookback_days} days:")
pd.DataFrame(
    {
        "Model": ["MC Dropout", "GARCH"],
        "PICP (67%)": [cov_67, cov_67_garch],
        "PICP (95%)": [cov_95, cov_95_garch],
        "Width (67%)": [mean_width_67, mean_width_67_garch],
        "Width (95%)": [mean_width_95, mean_width_95_garch],
        "PICP/Width (67%)": [
            cov_67 / mean_width_67,
            cov_67_garch / mean_width_67_garch,
        ],
        "PICP/Width (95%)": [
            cov_95 / mean_width_95,
            cov_95_garch / mean_width_95_garch,
        ],
    }
)

# %%
