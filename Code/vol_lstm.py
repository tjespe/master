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
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Mean"] = mean_pred
df_test["Volatility"] = volatility_pred
df_test.to_csv(f"predictions/lstm_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")

# %%
# Build an LSTM w/ FFNN layers and MC Dropout
# Define the model
lstm_w_ffnn_inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# LSTM layer
lstm_out = LSTM(units=200, activation="tanh")(lstm_w_ffnn_inputs)
lstm_out = LayerNormalization()(lstm_out)
lstm_out = Dropout(0.1)(lstm_out)

# FFNN layers
ffnn_out = Dense(100, activation="relu")(lstm_out)
ffnn_out = Dropout(0.1)(ffnn_out)
ffnn_out = Dense(50, activation="relu")(ffnn_out)
ffnn_out = Dropout(0.1)(ffnn_out)

# Mean output
lstm_w_ffnn_mean_out = Dense(
    1,
    activation="linear",
    # Initialize to very small values because we assume it is difficult to predict the mean return and we want to avoid large predictions
    kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
    name="mean_out",
)(ffnn_out)

# Output the log of variance (log(sigma^2)) so that we can use a linear activation
lstm_w_ffnn_log_variance_out = Dense(
    1,
    activation="linear",
    # Add a small regularizer to not over-estimate the variance
    kernel_regularizer=l2(1e-6),
    name="log_variance_out",
)(ffnn_out)

# Concatenate outputs
lstm_w_ffnn_outputs = Concatenate()(
    [lstm_w_ffnn_mean_out, lstm_w_ffnn_log_variance_out]
)

# Define the model
lstm_w_ffnn_model = Model(inputs=lstm_w_ffnn_inputs, outputs=lstm_w_ffnn_outputs)

# Compile model
lstm_w_ffnn_model.compile(optimizer=Adam(), loss=nll_loss)


# %%
# Load the model (if already trained)
lstm_w_ffnn_model_fname = f"models/{MODEL_NAME}_w_ffnn.h5"
if os.path.exists(lstm_w_ffnn_model_fname):
    lstm_w_ffnn_model = tf.keras.models.load_model(
        lstm_w_ffnn_model_fname, compile=False
    )
    lstm_w_ffnn_model.compile(optimizer=Adam(), loss=nll_loss)
    print("Loaded from disk")

# %%
# Fit the model (can be repeated several times to train further)
lstm_w_ffnn_model.fit(scaled_X_train, y_train, epochs=50, batch_size=32, verbose=1)

# %%
# Save the model
lstm_w_ffnn_model.save(lstm_w_ffnn_model_fname)

# %%
# Predict expected return and volatility using the LSTM w/ FFNN
y_pred_lstm_w_ffnn = lstm_w_ffnn_model.predict(scaled_X_test)
log_variance_pred_lstm_w_ffnn = y_pred_lstm_w_ffnn[:, 1]
mean_pred_lstm_w_ffnn = y_pred_lstm_w_ffnn[:, 0]
variance_pred_lstm_w_ffnn = np.exp(log_variance_pred_lstm_w_ffnn)
volatility_pred_lstm_w_ffnn = np.sqrt(variance_pred_lstm_w_ffnn)

# %%
# Save predictions to file
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Mean"] = mean_pred_lstm_w_ffnn
df_test["Volatility"] = volatility_pred_lstm_w_ffnn
df_test.to_csv(
    f"predictions/lstm_ffnn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)


# %%
# Define function for making predictions with MC Dropout
@tf.function
def predict_with_mc_dropout_tf(model, X, T=100):
    preds = []
    for i in range(T):
        y_p = model(X, training=True)  # Keep dropout active
        preds.append(y_p)

    # Stack predictions into a tensor
    preds = tf.stack(preds)

    # Calculate statistics using TensorFlow operations
    expected_returns = tf.reduce_mean(preds[:, :, 0], axis=0)
    epistemic_uncertainty_expected_returns = tf.math.reduce_std(preds[:, :, 0], axis=0)
    log_variance_preds = preds[:, :, 1]
    variance_estimates = tf.math.exp(log_variance_preds)
    variance_estimates = tf.maximum(variance_estimates, 1e-6)
    volatility_estimates = tf.math.sqrt(variance_estimates)
    volatility_estimate_per_day = tf.reduce_mean(volatility_estimates, axis=0)
    epistemic_uncertainty_volatility_estimates = tf.math.reduce_std(
        volatility_estimates, axis=0
    )
    total_uncertainty = tf.math.sqrt(
        tf.square(epistemic_uncertainty_expected_returns)
        + variance_estimates
        + tf.square(epistemic_uncertainty_volatility_estimates)
    )

    return {
        "expected_returns": expected_returns,
        "epistemic_uncertainty_expected_returns": epistemic_uncertainty_expected_returns,
        "volatility_estimates": volatility_estimate_per_day,
        "epistemic_uncertainty_volatility_estimates": epistemic_uncertainty_volatility_estimates,
        "total_uncertainty": total_uncertainty,
        "preds": preds,
    }


def predict_with_mc_dropout(model, X, T=100):
    # Call the @tf.function-decorated function
    results = predict_with_mc_dropout_tf(model, X, T)

    # Convert TensorFlow tensors to NumPy arrays
    return {key: value.numpy() for key, value in results.items()}


# %%
# Make predictions for regular LSTM with MC Dropout
mc_results = predict_with_mc_dropout(model, scaled_X_test, T=100)

# %%
# Save predictions to file
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Mean"] = mc_results["expected_returns"]
df_test["Volatility"] = mc_results["volatility_estimates"]
df_test["Epistemic_Uncertainty_Volatility"] = mc_results[
    "epistemic_uncertainty_volatility_estimates"
]
df_test["Epistemic_Uncertainty_Mean"] = mc_results[
    "epistemic_uncertainty_expected_returns"
]
df_test.to_csv(f"predictions/lstm_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")

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
# Make predictions for LSTM w/ FFNN with MC Dropout
mc_results_lstm_w_ffnn = predict_with_mc_dropout(
    lstm_w_ffnn_model, scaled_X_test, T=100
)

# %%
# Save predictions to file
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Mean"] = mc_results_lstm_w_ffnn["expected_returns"]
df_test["Volatility"] = mc_results_lstm_w_ffnn["volatility_estimates"]
df_test["Epistemic_Uncertainty_Volatility"] = mc_results_lstm_w_ffnn[
    "epistemic_uncertainty_volatility_estimates"
]
df_test["Epistemic_Uncertainty_Mean"] = mc_results_lstm_w_ffnn[
    "epistemic_uncertainty_expected_returns"
]
df_test.to_csv(
    f"predictions/lstm_ffnn_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)

# %%
# Fit a GARCH(1,1) model and make predictions
# Filter the data for the selected symbol
df_filtered = df.xs(TEST_ASSET, level="Symbol")

# Training data
returns_train = df_filtered["LogReturn"].loc[:TRAIN_TEST_SPLIT]
returns_train = returns_train * 100  # Scale to percentages

# Test data
returns_test = df_filtered["LogReturn"].loc[TRAIN_TEST_SPLIT:]
scaled_returns_test = returns_test * 100  # Scale to percentages

# Initialize an empty list to store forecasts
garch_vol_pred = []

# Combine the training and test data
returns_combined = pd.concat([returns_train, scaled_returns_test])

# Perform rolling forecasts
for i in range(len(scaled_returns_test)):
    if i % 20 == 0:
        print(f"Progress: {i/len(scaled_returns_test):.2%}", end="\r")

    # Update the model with data up to the current point in time
    end = len(returns_train) + i
    returns_sample = returns_combined.iloc[:end]

    # Fit the GARCH model
    am = arch_model(returns_sample, vol="GARCH", p=1, q=1, mean="Zero")
    res = am.fit(disp="off")

    # Forecast the next time point
    forecast = res.forecast(horizon=1)
    forecast_var = forecast.variance.iloc[-1].values[0]
    forecast_vol = np.sqrt(forecast_var) / 100  # Adjust scaling
    garch_vol_pred.append(forecast_vol)

# Convert the list to a numpy array
garch_vol_pred = np.array(garch_vol_pred)

# %%
# Save GARCH predictions to file
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Volatility"] = garch_vol_pred
df_test["Mean"] = 0  # Assume mean is 0
df_test.to_csv(f"predictions/garch_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")

# %%
# Get test part of df
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
abs_returns_test = df_test["LogReturn"].abs()

# Training and test data
returns_train = (
    df.xs(TEST_ASSET, level="Symbol")["LogReturn"].loc[:TRAIN_TEST_SPLIT] * 100
)  # Scale to percentages
returns_test = df_test["LogReturn"]

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
