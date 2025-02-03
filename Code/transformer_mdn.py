# %%
# Define parameters (imported from your settings)
from shared.numerical_mixture_moments import numerical_mixture_moments
from settings import LOOKBACK_DAYS, SUFFIX, TEST_ASSET, DATA_PATH, TRAIN_TEST_SPLIT

MODEL_NAME = f"transformer_mdn_{LOOKBACK_DAYS}_days{SUFFIX}"
RVOL_DATA_PATH = "data/RVOL.csv"
VIX_DATA_PATH = "data/VIX.csv"

# %%
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    Embedding,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam

# We will use MultiHeadAttention from TF >= 2.4
from tensorflow.keras.layers import MultiHeadAttention

# For old TF versions, you'd need a custom MHA layer or upgrade TF
# from tensorflow.keras.layers import MultiHeadAttention

# External
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq
from scipy.stats import norm

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
df["NextDayTradingDay"] = (
    df.index.get_level_values("Date")
    .shift(1, freq="D")
    .isin(df.index.get_level_values("Date"))
)
df["NextDayTradingDay"]

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

    # Sign of return to capture the direction of the return
    sign_return = np.sign(returns)

    # Use whether the next day is a trading day or not as a feature
    next_day_trading_day = group["NextDayTradingDay"].values.reshape(-1, 1)

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
    train_test_split_index = len(
        group[group.index.get_level_values("Date") < TRAIN_TEST_SPLIT]
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
            next_day_trading_day,
        )
    )

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
# Mixture Density Network loss for univariate mixtures
def mdn_loss(num_mixtures):
    """
    Negative log-likelihood for a mixture of Gaussians (univariate).
    Output shape: (batch_size, 3*num_mixtures)
      => we parse [logits_pi, mu, log_var].
    """

    def loss_fn(y_true, y_pred):
        # parse
        logits_pi = y_pred[:, :num_mixtures]
        mu = y_pred[:, num_mixtures : 2 * num_mixtures]
        log_var = y_pred[:, 2 * num_mixtures :]

        pi = tf.nn.softmax(logits_pi, axis=-1)
        sigma = tf.exp(0.5 * log_var)  # interpret as log-variance

        # expand dims for broadcast (B,) -> (B,1)
        y_true = tf.expand_dims(y_true, axis=-1)

        # gaussian pdf for each mixture component
        normal_dist = (1.0 / (sigma * tf.sqrt(2.0 * np.pi))) * tf.exp(
            -0.5 * ((y_true - mu) / sigma) ** 2
        )
        weighted_pdf = pi * normal_dist
        pdf_sum = tf.reduce_sum(weighted_pdf, axis=1) + 1e-12  # avoid log(0)

        nll = -tf.math.log(pdf_sum)
        return tf.reduce_mean(nll)

    return loss_fn


# %%
def transformer_encoder(inputs, d_model, num_heads, ff_dim, rate=0.1):
    """
    A single block of Transformer encoder:
      1) MHA + residual + LayerNorm
      2) FFN + residual + LayerNorm
    """
    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(
        inputs, inputs
    )
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Feed-forward
    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dropout(rate)(ffn)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(rate)(ffn)

    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2


# %%
def build_transformer_mdn(
    lookback_days,
    num_features=2,  # e.g. [LogReturn, log(SquaredReturn)]
    d_model=64,
    num_heads=4,
    num_layers=2,
    ff_dim=128,
    dropout=0.1,
    n_mixtures=5,
):
    """
    Creates a Transformer-based encoder for sequences of shape:
      (lookback_days, num_features)
    Then outputs 3*n_mixtures for MDN (univariate).
    """
    inputs = Input(shape=(lookback_days, num_features))

    # Project inputs to d_model
    x = Dense(d_model, activation=None)(inputs)

    # Stack multiple Transformer encoder blocks
    for _ in range(num_layers):
        x = transformer_encoder(x, d_model, num_heads, ff_dim, rate=dropout)

    # Global average pooling (or take last time step)
    x = GlobalAveragePooling1D()(x)

    # Output layer: 3*n_mixtures => [logits_pi, mu, log_sigma]
    mdn_output = Dense(3 * n_mixtures, activation=None, name="mdn_output")(x)

    model = Model(inputs=inputs, outputs=mdn_output)
    return model


# %%
def parse_mdn_output(mdn_out, n_mixtures):
    """
    Given y_pred from the model with shape (batch, 3*n_mixtures),
    parse out pi, mu, sigma. Returns (pi, mu, sigma) each shape = (batch, n_mixtures).
    """
    logits_pi = mdn_out[:, :n_mixtures]
    mu = mdn_out[:, n_mixtures : 2 * n_mixtures]
    log_var = mdn_out[:, 2 * n_mixtures :]

    pi = tf.nn.softmax(logits_pi, axis=-1)
    sigma = tf.exp(0.5 * log_var)

    return pi, mu, sigma


def mixture_mean_and_var(pi, mu, sigma):
    """
    For univariate mixture:
      mixture_mean = sum(pi_k * mu_k)
      mixture_var  = sum(pi_k * (sigma_k^2 + mu_k^2)) - mixture_mean^2
    """
    mixture_mean = tf.reduce_sum(pi * mu, axis=1)
    mixture_mean_sq = tf.square(mixture_mean)
    # E[sigma^2 + mu^2] = sum_k(pi_k * (sigma_k^2 + mu_k^2))
    e_sigma2_mu2 = tf.reduce_sum(pi * (tf.square(sigma) + tf.square(mu)), axis=1)
    mixture_var = e_sigma2_mu2 - mixture_mean_sq
    return mixture_mean, mixture_var


# %%
# Example MC Dropout approach specialized for MDN outputs
def predict_with_mc_dropout_mdn(model, X, T=100, n_mixtures=5):
    """
    Performs T stochastic forward passes with model(X, training=True).
    Aggregates mixture distribution => splits aleatoric vs. epistemic.

    Returns a dict with:
      "expected_returns" : mean of mixture means across MC samples
      "volatility_estimates" : sqrt of total variance (aleatoric + epistemic)
      "epistemic_uncertainty_expected_returns" : std dev of mixture means across MC
      "epistemic_uncertainty_volatility_estimates": see notes
      ...
    Feel free to adapt as needed.
    """
    # Storage for mixture means, mixture variances across T samples
    mc_means = []
    mc_vars = []

    # Run T forward passes
    for _ in range(T):
        mdn_out = model(X, training=True)  # (batch, 3*n_mixtures)
        pi, mu, sigma = parse_mdn_output(mdn_out, n_mixtures)

        # compute mixture mean & var for each sample
        mean_s, var_s = mixture_mean_and_var(pi, mu, sigma)
        mc_means.append(mean_s.numpy())
        mc_vars.append(var_s.numpy())

    mc_means = np.array(mc_means)  # shape: (T, batch)
    mc_vars = np.array(mc_vars)  # shape: (T, batch)

    # Aleatoric = average of variances
    aleatoric_var = np.mean(mc_vars, axis=0)  # (batch,)

    # Epistemic = variance of means across MC
    epistemic_var = np.var(mc_means, axis=0)  # (batch,)

    # Final predicted mean = average of mixture means
    final_means = np.mean(mc_means, axis=0)

    # total variance = aleatoric + epistemic
    total_var = aleatoric_var + epistemic_var

    results = {
        "expected_returns": final_means,  # shape (batch,)
        "volatility_estimates": np.sqrt(total_var),
        "epistemic_uncertainty_expected_returns": np.sqrt(epistemic_var),
        "epistemic_uncertainty_volatility_estimates": np.sqrt(epistemic_var),
        # or define your own decomposition
    }
    return results


# %%
# 1) Inspect shapes
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape},   y_test.shape: {y_test.shape}")

# %%
# 2) Build model
N_MIXTURES = 5
transformer_mdn_model = build_transformer_mdn(
    lookback_days=LOOKBACK_DAYS,
    num_features=X_train.shape[2],  # 2 features in our example
    d_model=64,
    num_heads=4,
    num_layers=2,
    ff_dim=128,
    dropout=0.1,
    n_mixtures=N_MIXTURES,
)

# %%
# 4) Load existing model if it exists
model_fname = f"models/{MODEL_NAME}.keras"
if os.path.exists(model_fname):
    transformer_mdn_model = tf.keras.models.load_model(
        model_fname, custom_objects={"loss_fn": mdn_loss(N_MIXTURES)}, compile=False
    )
    # Re-compile
    transformer_mdn_model.compile(
        optimizer=Adam(learning_rate=1e-3), loss=mdn_loss(N_MIXTURES)
    )
    print("Loaded pre-trained model from disk.")

# %%
# 5) Train
# Start with a high learning rate, then reduce
transformer_mdn_model.compile(
    optimizer=Adam(learning_rate=1e-2), loss=mdn_loss(N_MIXTURES)
)
transformer_mdn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# %%
# Reduce learning rate
transformer_mdn_model.compile(
    optimizer=Adam(learning_rate=1e-3), loss=mdn_loss(N_MIXTURES)
)
transformer_mdn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# %%
# Reduce learning rate again
transformer_mdn_model.compile(
    optimizer=Adam(learning_rate=1e-4), loss=mdn_loss(N_MIXTURES)
)
transformer_mdn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# %%
# 6) Save
transformer_mdn_model.save(model_fname)

# %%
# 7) Single-pass predictions
y_pred_mdn = transformer_mdn_model.predict(X_test)  # shape: (batch, 3*N_MIXTURES)
pi_pred, mu_pred, sigma_pred = parse_mdn_output(y_pred_mdn, N_MIXTURES)
# For reference, compute mixture means & variances
mixture_mean_sp, mixture_var_sp = mixture_mean_and_var(pi_pred, mu_pred, sigma_pred)
mixture_mean_sp = mixture_mean_sp.numpy()
mixture_std_sp = np.sqrt(mixture_var_sp.numpy())

# %%
# Plot 10 charts with the distributions for 10 random days
plt.figure(figsize=(10, 40))
np.random.seed(0)
for i in range(10):
    plt.subplot(10, 1, i + 1)
    i = np.random.randint(0, len(y_test))
    timestamp = df.index[-i][0]
    mu_m = mixture_mean_sp[-i]
    sigma_m = mixture_std_sp[-i]
    x_min = -0.1
    x_max = 0.1
    x_vals = np.linspace(x_min, x_max, 1000)
    mixture_pdf = (1 / (sigma_m * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_vals - mu_m) / sigma_m) ** 2
    )
    plt.fill_between(
        x_vals,
        np.zeros_like(x_vals),
        mixture_pdf,
        color="blue",
        label="Mixture",
        alpha=0.5,
    )
    for j in range(N_MIXTURES):
        weight = pi_pred[-i, j].numpy()
        mu = mu_pred[-i, j]
        sigma = sigma_pred[-i, j]
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_vals - mu) / sigma) ** 2
        )
        plt.plot(x_vals, pdf, label=f"$\pi$ = {weight*100:.2f}%")
    plt.axvline(y_test[-i], color="red", linestyle="--", label="Actual")
    plt.axvline(
        mixture_mean_sp[-i], color="black", linestyle="--", label="Predicted Mean"
    )
    moment_estimates = numerical_mixture_moments(
        np.array(pi_pred[-i]),
        np.array(mu_pred[-i]),
        np.array(sigma_pred[-i]),
        range_factor=3,
    )
    plt.text(
        x_min + 0.01,
        5,
        f"Mean: {mu_m*100:.2f}%\n"
        f"Std: {sigma_m*100:.2f}%\n"
        f"Skewness: {moment_estimates['skewness']:.4f}*\n"
        f"Excess kurtosis: {moment_estimates['excess_kurtosis']:.4f}*\n"
        f"* Numerically estimated",
        fontsize=10,
    )
    plt.gca().set_xticklabels(
        ["{:.1f}%".format(x * 100) for x in plt.gca().get_xticks()]
    )
    plt.title(f"{timestamp.strftime('%Y-%m-%d')} - Predicted Distribution")
    plt.legend()
    plt.ylabel("Density")
plt.xlabel("LogReturn")
plt.tight_layout()
plt.show()

# %%
# Plot time series with mean, volatility and actual returns for last 100 days
days = 200
shift = 500
filtered_df = (
    df.xs(TEST_ASSET, level="Symbol")
    .loc[TRAIN_TEST_SPLIT:]
    .tail(days + shift)
    .head(days)
)
filtered_mix_mean = mixture_mean_sp[-days - shift : -shift]
filtered_mix_std = mixture_std_sp[-days - shift : -shift]

plt.figure(figsize=(12, 6))
plt.plot(
    filtered_df.index,
    filtered_df["LogReturn"],
    label="Actual Returns",
    color="black",
    alpha=0.5,
)
plt.plot(filtered_df.index, filtered_mix_mean, label="Predicted Mean", color="blue")
plt.fill_between(
    filtered_df.index,
    filtered_mix_mean - filtered_mix_std,
    filtered_mix_mean + filtered_mix_std,
    color="blue",
    alpha=0.5,
    label="±1 aleatoric SD Interval",
)
plt.fill_between(
    filtered_df.index,
    filtered_mix_mean - 2 * filtered_mix_std,
    filtered_mix_mean + 2 * filtered_mix_std,
    color="blue",
    alpha=0.2,
    label="±2 aleatoric SD Interval",
)
plt.axhline(
    filtered_df["LogReturn"].mean(),
    color="red",
    linestyle="--",
    label="True mean return across time",
    alpha=0.5,
)
plt.gca().set_yticklabels(["{:.1f}%".format(x * 100) for x in plt.gca().get_yticks()])
plt.title(f"Transformer + MDN predictions, {days} days")
plt.xlabel("Date")
plt.ylabel("LogReturn")
plt.legend()
plt.show()


# %%
def calculate_interval(pis, mus, sigmas, confidence_levels):
    """
    Calculate the (lower, upper) quantile intervals for a Gaussian mixture for several
    confidence levels simultaneously.

    For each sample i, the mixture CDF is defined as:
        F_i(x) = sum_m pis[i, m] * Phi((x - mus[i, m]) / sigmas[i, m])
    where Phi is the standard normal CDF. For each confidence level cl, the interval is
    determined by:
        lower_i  such that F_i(lower_i) = (1 - cl)/2
        upper_i  such that F_i(upper_i) = 1 - (1 - cl)/2

    Args:
        pis: Mixture weights, shape (n_samples, n_mixtures)
        mus: Mixture means, shape (n_samples, n_mixtures)
        sigmas: Mixture standard deviations, shape (n_samples, n_mixtures)
        confidence_levels: Iterable of confidence levels (floats), e.g., [0.95, 0.5]

    Returns:
        intervals: Array of shape (n_samples, n_confidence_levels, 2) where
            intervals[i, j, 0] is the lower bound and intervals[i, j, 1] is the upper bound
            for sample i at confidence level confidence_levels[j].
    """
    pis = np.asarray(pis)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    conf_levels = np.asarray(confidence_levels)

    n_samples, n_mixtures = pis.shape
    n_conf = conf_levels.size
    intervals = np.empty((n_samples, n_conf, 2))

    # To bracket all quantile roots for every cl, we use the most extreme targets,
    # which come from the highest confidence level.
    max_conf = np.max(conf_levels)
    global_alpha = (1 - max_conf) / 2  # smallest quantile target (further left)
    global_beta = 1 - global_alpha  # largest quantile target (further right)

    for i in range(n_samples):
        weights = pis[i, :]
        means = mus[i, :]
        stds = sigmas[i, :]

        # Mixture CDF for sample i.
        def mixture_cdf(x):
            return np.sum(weights * norm.cdf((x - means) / stds))

        # Set an initial search interval.
        s = np.max(stds)
        low = np.min(means - 10 * stds)
        high = np.max(means + 10 * stds)

        # Expand search interval if necessary.
        while mixture_cdf(low) > global_alpha:
            low -= 10 * s
        while mixture_cdf(high) < global_beta:
            high += 10 * s

        # Compute intervals for each confidence level.
        for j, cl in enumerate(conf_levels):
            alpha = (1 - cl) / 2
            beta = 1 - alpha  # equivalently, (1 + cl) / 2
            lower_bound = brentq(lambda x: mixture_cdf(x) - alpha, low, high)
            upper_bound = brentq(lambda x: mixture_cdf(x) - beta, low, high)
            intervals[i, j, 0] = lower_bound
            intervals[i, j, 1] = upper_bound

    return intervals


# %%
# Calculate intervals for 67%, 95%, 97.5% and 99% confidence levels
confidence_levels = [0.67, 0.95, 0.975, 0.99]
intervals = calculate_interval(pi_pred, mu_pred, sigma_pred, confidence_levels)


# %%
# 8) Store single-pass predictions
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Mean_SP"] = mixture_mean_sp
df_test["Vol_SP"] = mixture_std_sp
df_test["NLL"] = mdn_loss(N_MIXTURES)(y_test, y_pred_mdn).numpy()

for i, cl in enumerate(confidence_levels):
    df_test[f"LB_{int(100*cl)}"] = intervals[:, i, 0]
    df_test[f"UB_{int(100*cl)}"] = intervals[:, i, 1]

os.makedirs("predictions", exist_ok=True)
df_test.to_csv(
    f"predictions/transformer_mdn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)

# %%
# 9) MC Dropout predictions
mc_results = predict_with_mc_dropout_mdn(
    transformer_mdn_model, X_test, T=100, n_mixtures=N_MIXTURES
)

df_test["Mean_MC"] = mc_results["expected_returns"]
df_test["Vol_MC"] = mc_results["volatility_estimates"]
df_test["Epistemic_Unc_Vol"] = mc_results["epistemic_uncertainty_volatility_estimates"]
df_test["Epistemic_Unc_Mean"] = mc_results["epistemic_uncertainty_expected_returns"]

df_test.to_csv(
    f"predictions/transformer_mdn_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
)

# %%
# 10) (Optional) Compare coverage vs. GARCH predictions
garch_path = f"predictions/garch_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"
if os.path.exists(garch_path):
    garch_vol_pred = pd.read_csv(garch_path, index_col="Date")["Volatility"]
else:
    garch_vol_pred = None

# %%
# We'll do a short coverage example on the last 100 points
lookback_days_plot = 100
shift = 200
idx_start = max(0, len(df_test) - lookback_days_plot - shift)
idx_end = idx_start + lookback_days_plot

actual_returns = df_test["LogReturn"].iloc[idx_start:idx_end]
mc_means = df_test["Mean_MC"].iloc[idx_start:idx_end]
mc_vols = df_test["Vol_MC"].iloc[idx_start:idx_end]

# Chart with intervals
plt.figure(figsize=(12, 6))
plt.plot(actual_returns.index, actual_returns, label="Actual Returns", color="black")
plt.plot(mc_means.index, mc_means, label="MC Dropout Mean", color="blue")
plt.fill_between(
    mc_means.index,
    mc_means - mc_vols,
    mc_means + mc_vols,
    color="blue",
    alpha=0.5,
    label="±1 SD Interval (MC)",
)
plt.fill_between(
    mc_means.index,
    mc_means - 2 * mc_vols,
    mc_means + 2 * mc_vols,
    color="blue",
    alpha=0.2,
    label="±2 SD Interval (MC)",
)
# Plot epistemc uncertainty as a bar chart on a secondary y-axis
plt.bar(
    mc_means.index,
    df_test["Epistemic_Unc_Mean"].iloc[idx_start:idx_end],
    alpha=0.3,
    color="red",
    label="Epistemic Uncertainty (MC)",
)
plt.title(f"Transformer + MDN (MC Dropout) intervals, last {lookback_days_plot} points")
plt.xlabel("Date")
plt.ylabel("LogReturn")
plt.legend()
plt.show()

# %%
# Coverage stats
# 67% ~ 1 std
cov_67 = np.mean(
    (actual_returns >= mc_means - mc_vols) & (actual_returns <= mc_means + mc_vols)
)
# 95% ~ 2 std
cov_95 = np.mean(
    (actual_returns >= mc_means - 2 * mc_vols)
    & (actual_returns <= mc_means + 2 * mc_vols)
)

# %%
# Compare with GARCH if available
if garch_vol_pred is not None:
    # align garch_vol_pred with same date range
    garch_vol_plot = garch_vol_pred.loc[mc_means.index]
    cov_67_garch = np.mean(
        (actual_returns >= -garch_vol_plot) & (actual_returns <= garch_vol_plot)
    )
    cov_95_garch = np.mean(
        (actual_returns >= -1.96 * garch_vol_plot)
        & (actual_returns <= 1.96 * garch_vol_plot)
    )

    mean_width_67_garch = garch_vol_plot.mean()
    mean_width_95_garch = 1.96 * 2 * garch_vol_plot.mean()

    mean_width_67_mc = mc_vols.mean()
    mean_width_95_mc = 2.0 * 2 * mc_vols.mean()  # 2 stdev => 95% in normal approx

    print(f"Stats for last {lookback_days_plot} points:")
    stats_df = pd.DataFrame(
        {
            "Model": ["MC Dropout", "GARCH"],
            "PICP (67%)": [cov_67, cov_67_garch],
            "PICP (95%)": [cov_95, cov_95_garch],
            "Width (67%)": [mean_width_67_mc, mean_width_67_garch],
            "Width (95%)": [mean_width_95_mc, mean_width_95_garch],
            "PICP/Width (67%)": [
                cov_67 / mean_width_67_mc if mean_width_67_mc != 0 else np.nan,
                (
                    cov_67_garch / mean_width_67_garch
                    if mean_width_67_garch != 0
                    else np.nan
                ),
            ],
            "PICP/Width (95%)": [
                cov_95 / mean_width_95_mc if mean_width_95_mc != 0 else np.nan,
                (
                    cov_95_garch / mean_width_95_garch
                    if mean_width_95_garch != 0
                    else np.nan
                ),
            ],
        }
    )
    print(stats_df)
else:
    print(f"Coverage (67%): {cov_67:.3f}, Coverage (95%): {cov_95:.3f}")

print("Done.")

# %%
