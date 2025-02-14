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

MODEL_NAME = f"lstm_mc_w_rvol_and_vix_log_var_{LOOKBACK_DAYS}_days{SUFFIX}"
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
# Add feature: is next day trading day or not
df["NextDayNotTradingDay"] = ~(
    df.index.get_level_values("Date")
    .shift(1, freq="D")
    .isin(df.index.get_level_values("Date"))
)
df["NextDayNotTradingDay"]

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

    # Whether the next day is a trading day or not
    next_day_not_trading_day = group["NextDayNotTradingDay"].values.reshape(-1, 1)

    # Sign of return to capture the direction of the return
    sign_return = np.sign(returns)

    # Extract realized volatility and transform it to a similar scale
    rvol_annualized = group["Close_RVOL"].values.reshape(-1, 1) / 100
    rvol_daily = rvol_annualized / np.sqrt(252)
    rvol = np.log(rvol_daily**2 + (0.1 / 100) ** 2)

    # Calculate one day change in RVOL
    rvol_change_1d = np.diff(rvol, axis=0, prepend=rvol[0, 0])
    rvol_change_2d = rvol - np.vstack([rvol[:2], rvol[:-2]])
    rvol_change_7d = rvol - np.vstack([rvol[:7], rvol[:-7]])

    # Extract VIX and transform it to a similar scale
    vix_annualized = group["Close_VIX"].values.reshape(-1, 1) / 100
    vix_daily = vix_annualized / np.sqrt(252)
    vix = np.log(vix_daily**2 + (0.1 / 100) ** 2)

    # Calculate one day change in VIX
    vix_change_1d = np.diff(vix, axis=0, prepend=vix[0, 0])
    vix_change_2d = vix - np.vstack([vix[:2], vix[:-2]])
    vix_change_7d = vix - np.vstack([vix[:7], vix[:-7]])

    # Find date to split on
    TRAIN_VALIDATION_SPLIT_index = len(
        group[group.index.get_level_values("Date") < TRAIN_VALIDATION_SPLIT]
    )
    VALIDATION_TEST_SPLIT_index = len(
        group[group.index.get_level_values("Date") < VALIDATION_TEST_SPLIT]
    )

    # Stack returns and squared returns together
    data = np.hstack(
        (
            log_sq_returns,
            sign_return,
            rvol,
            rvol_change_1d,
            rvol_change_2d,
            rvol_change_7d,
            vix_change_1d,
            vix_change_2d,
            vix_change_7d,
            next_day_not_trading_day,
        )
    )

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
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# LSTM layer
lstm_out = LSTM(
    units=100,
    activation="tanh",
)(inputs)

# Dropout layer
lstm_out = Dropout(0.2)(lstm_out)

# Variance output with softplus activation
variance_out = Dense(1, activation="linear")(lstm_out)

# Define the model
model = Model(inputs=inputs, outputs=variance_out)

# %%
# Load the model (if already trained)
MODEL_FNAME = f"models/{MODEL_NAME}.h5"
if os.path.exists(MODEL_FNAME):
    model = tf.keras.models.load_model(MODEL_FNAME, compile=False)
    model.compile(optimizer=Adam(), loss=nll_loss_variance_only)
    print("Model loaded from disk")

# %%
# Fit the model (can be repeated several times to train further)
# First fit with high learning rate to quickly get close to the optimal solution
model.compile(optimizer=Adam(learning_rate=1e-2), loss=nll_loss_variance_only)
model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

# %%
# Then fit with lower learning rate to fine-tune the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss=nll_loss_variance_only)
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

# %%
# Then fit with even lower learning rate to fine-tune the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss=nll_loss_variance_only)
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

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
# Calculate NLL for the test set
nll_test = nll_loss_variance_only(y_test, y_pred).numpy()
print(f"NLL on test set: {nll_test}")

# %%
# Get test part of df
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
df_validation

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
# Make predictions with epistemic uncertainty
from shared.mc_dropout import predict_with_mc_dropout

preds = predict_with_mc_dropout(model, X_test, 100, mean_i=None, var_i=0)

# %%
vol_preds = preds["volatility_estimates"]
epistemic_uncertainty = preds["epistemic_uncertainty_volatility_estimates"]

# %%
# Plot results of LSTM with epistemic uncertainty
from_i = np.argmax(df_validation.index > "2024-05")
plt.figure(figsize=(12, 6))
plt.plot(
    df_validation.index[from_i:],
    df_validation["LogReturn"].abs().values[from_i:],
    label="Absolute Return",
    color="red",
    linewidth=1,
)
plt.plot(
    df_validation.index[from_i:],
    volatility_pred[from_i:],
    label="Volatility Prediction",
    color="black",
)
plt.fill_between(
    df_validation.index[from_i:],
    volatility_pred[from_i:] - epistemic_uncertainty[from_i:],
    volatility_pred[from_i:] + epistemic_uncertainty[from_i:],
    color="black",
    alpha=0.4,
    label="67% confidence interval",
)
plt.fill_between(
    df_validation.index[from_i:],
    volatility_pred[from_i:] - 2 * epistemic_uncertainty[from_i:],
    volatility_pred[from_i:] + 2 * epistemic_uncertainty[from_i:],
    color="black",
    alpha=0.2,
    label="95% confidence interval",
)
plt.title("Volatility Prediction with LSTM and Epistemic Uncertainty")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

# %%
# Plot distribution of volatility predictions for a random day
random_day = np.random.randint(0, len(vol_preds))
plt.figure(figsize=(12, 6))
samples = preds["preds"][:, random_day, 0]
samples_vol = np.sqrt(np.exp(samples))
plt.hist(samples_vol, bins=50)
plt.title("Distribution of Volatility Predictions")
plt.xlabel("Volatility")
plt.ylabel("Frequency")
vals = plt.gca().get_xticks()
plt.gca().set_xticklabels(["{:.2f}%".format(x * 100) for x in vals])
plt.axvline(volatility_pred[random_day], color="black", label="Mean Prediction")
plt.axvline(
    df_validation["LogReturn"].abs().values[random_day],
    color="red",
    label="Absolute Return",
)
plt.legend()
plt.show()

# %%
# Plot distribution of volatility predictions across all days
plt.figure(figsize=(12, 6))
samples = preds["preds"][:, :, 0]
samples_vol = np.sqrt(np.exp(samples))
plt.hist(samples_vol.flatten(), bins=100)
plt.title("Distribution of Volatility Predictions")
plt.xlabel("Volatility")
plt.ylabel("Frequency")
vals = plt.gca().get_xticks()
plt.gca().set_xticklabels(["{:.2f}%".format(x * 100) for x in vals])
plt.show()

# %%
# Plot distribution of log variances across all days
plt.figure(figsize=(12, 6))
plt.hist(samples.flatten(), bins=100)
plt.title("Distribution of Volatility Predictions")
plt.xlabel("Log variance (aleatoric)")
plt.ylabel("Frequency")
plt.show()

# %%
# Plot distribution of epistemic uncertainty for all days
plt.figure(figsize=(12, 6))
plt.hist(epistemic_uncertainty, bins=100)
plt.title("Distribution of Epistemic Uncertainty")
plt.xlabel("Epistemic Uncertainty")
plt.ylabel("Frequency")
vals = plt.gca().get_xticks()
plt.gca().set_xticklabels(["{:.2f}%".format(x * 100) for x in vals])
plt.show()

# %%
# Save predictions to file
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
df_validation["Volatility"] = volatility_pred
df_validation["Mean"] = mean_pred
df_validation["Epistemic_Uncertainty_Volatility"] = epistemic_uncertainty
df_validation.to_csv(
    f"predictions/lstm_mc_w_rvol_and_vix_predicitons_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)

# %%
