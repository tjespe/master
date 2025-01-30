# %%
# Define parameters (imported from your settings)
from settings import LOOKBACK_DAYS, SUFFIX, TEST_ASSET, DATA_PATH, TRAIN_TEST_SPLIT

MODEL_NAME = f"transformer_mdn_{LOOKBACK_DAYS}_days{SUFFIX}"

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

# We'll assume you have a separate GARCH or other predictions to compare with
# We'll read them later for coverage stats.

warnings.filterwarnings("ignore")


# %%
def load_data_and_preprocess():
    """
    Reads DATA_PATH, computes log-returns, builds
    overlapping windows of length LOOKBACK_DAYS for the
    chosen TEST_ASSET. Returns:
      scaled_X_train, scaled_X_test, y_train, y_test
      plus the original DataFrame (for indexing)
    """
    # 1) Read data
    df = pd.read_csv(DATA_PATH)
    if "Symbol" not in df.columns:
        df["Symbol"] = TEST_ASSET

    # Ensure Date is datetime, then sort
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])

    # 2) Compute log-return by Symbol
    df["LogReturn"] = (
        df.groupby("Symbol")["Close"]
        .apply(lambda x: np.log(x / x.shift(1)))
        .reset_index(drop=True)
    )
    df = df[~df["LogReturn"].isnull()]  # drop NaNs

    # For a second feature: log of squared returns, offset by a small constant
    df["SquaredReturn"] = df["LogReturn"] ** 2

    # 3) Filter date >= 1990
    df = df[df["Date"] >= "1990-01-01"]

    # 4) Set index
    df = df.set_index(["Date", "Symbol"])

    # 5) Create X, y for each symbol, focusing on next-day log-return
    X_train, X_test, y_train, y_test = [], [], [], []

    for symbol, group in df.groupby(level="Symbol"):
        returns = group["LogReturn"].values.reshape(-1, 1)
        log_sq_returns = np.log(
            group["SquaredReturn"].values.reshape(-1, 1)
            + (0.1 / 100) ** 2  # small offset for numerical stability
        )

        # find train/test split
        # count how many rows before TRAIN_TEST_SPLIT date
        train_test_split_index = len(
            group[group.index.get_level_values("Date") < TRAIN_TEST_SPLIT]
        )

        # combine features: [LogReturn, log(SquaredReturn + const)]
        data = np.hstack((returns, log_sq_returns))

        # build train sequences
        for i in range(LOOKBACK_DAYS, train_test_split_index):
            X_train.append(data[i - LOOKBACK_DAYS : i])
            y_train.append(returns[i, 0])  # next-day log-return

        # build test sequences only for TEST_ASSET
        if symbol == TEST_ASSET:
            for i in range(train_test_split_index, len(data)):
                if i - LOOKBACK_DAYS >= 0:  # ensure valid slice
                    X_test.append(data[i - LOOKBACK_DAYS : i])
                    y_test.append(returns[i, 0])

    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Scale the first column (log-returns) if desired
    scaler_1 = StandardScaler()
    # We won't scale the second column (log of squared returns) here,
    # but you could if you like.

    scaler_1.fit(X_train[:, :, 0])
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[:, :, 0] = scaler_1.transform(X_train_scaled[:, :, 0])
    X_test_scaled[:, :, 0] = scaler_1.transform(X_test_scaled[:, :, 0])

    return X_train_scaled, X_test_scaled, y_train, y_test, df


# %%
# Mixture Density Network loss for univariate mixtures
def mdn_loss(num_mixtures):
    """
    Negative log-likelihood for a mixture of Gaussians (univariate).
    Output shape: (batch_size, 3*num_mixtures)
      => we parse [logits_pi, mu, log_sigma].
    """

    def loss_fn(y_true, y_pred):
        # parse
        logits_pi = y_pred[:, :num_mixtures]
        mu = y_pred[:, num_mixtures : 2 * num_mixtures]
        log_sigma = y_pred[:, 2 * num_mixtures :]

        pi = tf.nn.softmax(logits_pi, axis=-1)
        sigma = tf.exp(log_sigma)

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
    log_sigma = mdn_out[:, 2 * n_mixtures :]

    pi = tf.nn.softmax(logits_pi, axis=-1)
    sigma = tf.exp(log_sigma)
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
# 1) Load data
X_train, X_test, y_train, y_test, df = load_data_and_preprocess()
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
# 3) Compile model
transformer_mdn_model.compile(
    optimizer=Adam(learning_rate=1e-2), loss=mdn_loss(N_MIXTURES)
)

# %%
# 4) Load existing model if it exists
model_fname = f"models/{MODEL_NAME}.h5"
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
os.makedirs("models", exist_ok=True)
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
# Plot 10 charts with the distributions for the last 10 points
plt.figure(figsize=(12, 12))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.hist(np.random.normal(mixture_mean_sp[-i], mixture_std_sp[-i], 1000), bins=50)
    plt.axvline(y_test[-i], color="red", linestyle="--", label="Actual")
    plt.axvline(
        mixture_mean_sp[-i], color="black", linestyle="--", label="Predicted Mean"
    )
    plt.title(f"Point {i+1} - Predicted Distribution")
plt.tight_layout()
plt.legend()
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
# 8) Store single-pass predictions
df_test = df.xs(TEST_ASSET, level="Symbol").loc[TRAIN_TEST_SPLIT:]
df_test["Mean_SP"] = mixture_mean_sp
df_test["Vol_SP"] = mixture_std_sp

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
