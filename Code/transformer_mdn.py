# %%
# Define parameters (imported from your settings)
from shared.mdn import (
    calculate_intervals,
    compute_mixture_pdf,
    get_mdn_bias_initializer,
    get_mdn_kernel_initializer,
    parse_mdn_output,
    predict_with_mc_dropout_mdn,
    univariate_mixture_mean_and_var_approx,
)
from shared.processing import get_lstm_train_test
from shared.numerical_mixture_moments import numerical_mixture_moments
from shared.loss import mdn_loss_numpy, mdn_loss_tf
from shared.crps import crps_mdn_numpy
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    TEST_ASSET,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT,
)

RVOL_DATA_PATH = "data/RVOL.csv"
VIX_DATA_PATH = "data/VIX.csv"
VERSION = 2
MODEL_NAME = f"transformer_mdn_{LOOKBACK_DAYS}_days{SUFFIX}_v{VERSION}"

# %%
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import subprocess
import gc

# For old TF versions, you'd need a custom MHA layer or upgrade TF
# from tensorflow.keras.layers import MultiHeadAttention

# External
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")


# %%
# Load preprocessed data
df, X_train, X_val, y_train, y_val = get_lstm_train_test(
    include_log_returns=True, include_fng=True
)
gc.collect()

# %%
# Import tensorflow and keras
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

    # Create initializers for MDN output layer
    mdn_kernel_init = get_mdn_kernel_initializer(n_mixtures)
    mdn_bias_init = get_mdn_bias_initializer(n_mixtures)

    # Output layer: 3*n_mixtures => [logits_pi, mu, log_sigma]
    mdn_output = Dense(
        3 * n_mixtures,
        activation=None,
        name="mdn_output",
        kernel_initializer=mdn_kernel_init,
        bias_initializer=mdn_bias_init,
    )(x)

    model = Model(inputs=inputs, outputs=mdn_output)
    return model


# %%
# 1) Inspect shapes
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_val.shape},   y_val.shape: {y_val.shape}")

# %%
# 2) Build model
N_MIXTURES = 20
transformer_mdn_model = build_transformer_mdn(
    lookback_days=LOOKBACK_DAYS,
    num_features=X_train.shape[2],  # 2 features in our example
    dropout=0.1,
    n_mixtures=N_MIXTURES,
)

# %%
# 4) Load existing model if it exists
model_fname = f"models/{MODEL_NAME}.keras"
if os.path.exists(model_fname):
    mdn_kernel_initializer = get_mdn_kernel_initializer(N_MIXTURES)
    mdn_bias_initializer = get_mdn_bias_initializer(N_MIXTURES)
    transformer_mdn_model = tf.keras.models.load_model(
        model_fname,
        custom_objects={
            "loss_fn": mdn_loss_tf(N_MIXTURES),
            "mdn_kernel_initializer": mdn_kernel_initializer,
            "mdn_bias_initializer": mdn_bias_initializer,
        },
    )
    # Re-compile
    transformer_mdn_model.compile(
        optimizer=Adam(learning_rate=1e-3), loss=mdn_loss_tf(N_MIXTURES)
    )
    print("Loaded pre-trained model from disk.")

# %%
# 5) Train
transformer_mdn_model.compile(
    optimizer=Adam(learning_rate=1e-3), loss=mdn_loss_tf(N_MIXTURES)
)
history = transformer_mdn_model.fit(
    X_train, y_train, epochs=10, batch_size=32, verbose=1
)

# %%
# Reduce learning rate
# transformer_mdn_model.compile(
#     optimizer=Adam(learning_rate=1e-4), loss=mdn_loss_tf(N_MIXTURES)
# )
# history = transformer_mdn_model.fit(
#     X_train, y_train, epochs=5, batch_size=32, verbose=1
# )

# %%
# 6) Save
transformer_mdn_model.save(model_fname)

# %%
# 6b) Commit and push
try:
    subprocess.run(["git", "pull"], check=True)
    subprocess.run(["git", "add", "models/transformer_mdn_*"], check=True)

    commit_header = "Train transformer MDN model."
    commit_body = f"Training history: {history.history}"

    subprocess.run(
        ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
    )
    subprocess.run(["git", "push"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Git command failed: {e}")

# %%
# 7) Single-pass predictions
y_pred_mdn = transformer_mdn_model.predict(X_val)  # shape: (batch, 3*N_MIXTURES)
pi_pred, mu_pred, sigma_pred = parse_mdn_output(y_pred_mdn, N_MIXTURES)

# %%
# Plot 10 charts with the distributions for 10 random days
plt.figure(figsize=(10, 40))
np.random.seed(0)
days = np.random.randint(0, len(y_val), 10)
days = np.sort(days)[::-1]
for i, day in enumerate(days):
    plt.subplot(10, 1, i + 1)
    timestamp = df.index[-day][0]
    x_min = -0.1
    x_max = 0.1
    x_vals = np.linspace(x_min, x_max, 1000)
    mixture_pdf = compute_mixture_pdf(
        x_vals, pi_pred[-day], mu_pred[-day], sigma_pred[-day]
    )
    plt.fill_between(
        x_vals,
        np.zeros_like(x_vals),
        mixture_pdf,
        color="blue",
        label="Mixture",
        alpha=0.5,
    )
    plotted_mixtures = 0
    top_weights = np.argsort(pi_pred[-day])[-7:][::-1]
    for j in range(N_MIXTURES):
        weight = pi_pred[-day, j].numpy()
        if weight < 0.001:
            continue
        plotted_mixtures += 1
        mu = mu_pred[-day, j]
        sigma = sigma_pred[-day, j]
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_vals - mu) / sigma) ** 2
        )
        legend = f"$\pi_{{{j}}}$ = {weight*100:.2f}%" if j in top_weights else None
        plt.plot(x_vals, pdf, label=legend, alpha=min(10 * weight, 1))
    plt.axvline(y_val[-day], color="red", linestyle="--", label="Actual")
    moment_estimates = numerical_mixture_moments(
        np.array(pi_pred[-day]),
        np.array(mu_pred[-day]),
        np.array(sigma_pred[-day]),
        range_factor=3,
    )
    plt.axvline(
        moment_estimates["mean"], color="black", linestyle="--", label="Predicted Mean"
    )
    plt.text(
        x_min + 0.01,
        5,
        f"Mean: {moment_estimates['mean']*100:.2f}%\n"
        f"Std: {moment_estimates['std']*100:.2f}%\n"
        f"Skewness: {moment_estimates['skewness']:.4f}*\n"
        f"Excess kurtosis: {moment_estimates['excess_kurtosis']:.4f}*\n"
        f"* Numerically estimated",
        fontsize=10,
    )
    plt.gca().set_xticklabels(
        ["{:.1f}%".format(x * 100) for x in plt.gca().get_xticks()]
    )
    plt.title(
        f"{timestamp.strftime('%Y-%m-%d')} - Predicted Return Distribution for {TEST_ASSET}"
    )
    plt.ylim(0, 50)
    plt.legend()
    plt.ylabel("Density")
plt.xlabel("LogReturn")
plt.tight_layout()
plt.savefig(
    f"results/transformer_mdn_distributions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_v{VERSION}.svg"
)
plt.show()


# %%
# Calculate intervals for 67%, 95%, 97.5% and 99% confidence levels
confidence_levels = [0, 0.5, 0.67, 0.90, 0.95, 0.975, 0.99]
intervals = calculate_intervals(pi_pred, mu_pred, sigma_pred, confidence_levels)

# %%
# Plot time series with mean, volatility and actual returns for last 100 days
days = 500
shift = 1
filtered_df = (
    df.xs(TEST_ASSET, level="Symbol")
    .loc[TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT]
    .tail(days + shift)
    .head(days)
)
filtered_intervals = intervals[-days - shift : -shift]
mean = (pi_pred * mu_pred).numpy().sum(axis=1)
filtered_mean = mean[-days - shift : -shift]

plt.figure(figsize=(12, 6))
plt.plot(
    filtered_df.index,
    filtered_df["LogReturn"],
    label="Actual Returns",
    color="black",
    alpha=0.5,
)
plt.plot(filtered_df.index, filtered_mean, label="Predicted Mean", color="red")
median = filtered_intervals[:, 0, 0]
plt.plot(filtered_df.index, median, label="Median", color="blue")
for i, cl in enumerate(confidence_levels):
    if cl == 0:
        continue
    plt.fill_between(
        filtered_df.index,
        filtered_intervals[:, i, 0],
        filtered_intervals[:, i, 1],
        color="blue",
        alpha=0.7 - i * 0.1,
        label=f"{int(100*cl)}% Interval",
    )
plt.axhline(
    filtered_df["LogReturn"].mean(),
    color="red",
    linestyle="--",
    label="True mean return across time",
    alpha=0.5,
)
plt.gca().set_yticklabels(["{:.1f}%".format(x * 100) for x in plt.gca().get_yticks()])
plt.title(f"Lstm w MDN predictions, {days} days")
plt.xlabel("Date")
plt.ylabel("LogReturn")
plt.legend()
plt.show()

# %%
# 8) Store single-pass predictions
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
# For reference, compute mixture means & variances
uni_mixture_mean_sp, uni_mixture_var_sp = univariate_mixture_mean_and_var_approx(
    pi_pred, mu_pred, sigma_pred
)
uni_mixture_mean_sp = uni_mixture_mean_sp.numpy()
uni_mixture_std_sp = np.sqrt(uni_mixture_var_sp.numpy())
df_validation["Mean_SP"] = uni_mixture_mean_sp
df_validation["Vol_SP"] = uni_mixture_std_sp
df_validation["NLL"] = mdn_loss_numpy(N_MIXTURES)(y_val, y_pred_mdn)
crps = crps_mdn_numpy(N_MIXTURES)
df_validation["CRPS"] = crps(y_val, y_pred_mdn)

for i, cl in enumerate(confidence_levels):
    df_validation[f"LB_{int(100*cl)}"] = intervals[:, i, 0]
    df_validation[f"UB_{int(100*cl)}"] = intervals[:, i, 1]

# %%
df_validation.to_csv(
    f"predictions/transformer_mdn_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_v{VERSION}.csv"
)

# %%
# 9) MC Dropout predictions
mc_results = predict_with_mc_dropout_mdn(
    transformer_mdn_model, X_val, T=100, n_mixtures=N_MIXTURES
)

df_validation["Mean_MC"] = mc_results["expected_returns"]
df_validation["Vol_MC"] = mc_results["volatility_estimates"]
df_validation["Epistemic_Unc_Vol"] = mc_results[
    "epistemic_uncertainty_volatility_estimates"
]
df_validation["Epistemic_Unc_Mean"] = mc_results[
    "epistemic_uncertainty_expected_returns"
]

df_validation.to_csv(
    f"predictions/transformer_mdn_mc_predictions_{TEST_ASSET}_{LOOKBACK_DAYS}_days_v{VERSION}.csv"
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
idx_start = max(0, len(df_validation) - lookback_days_plot - shift)
idx_end = idx_start + lookback_days_plot

actual_returns = df_validation["LogReturn"].iloc[idx_start:idx_end]
mc_means = df_validation["Mean_MC"].iloc[idx_start:idx_end]
mc_vols = df_validation["Vol_MC"].iloc[idx_start:idx_end]

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
    df_validation["Epistemic_Unc_Mean"].iloc[idx_start:idx_end],
    alpha=0.3,
    color="red",
    label="Epistemic Uncertainty (MC)",
)
plt.title(f"transformer + MDN (MC Dropout) intervals, last {lookback_days_plot} points")
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
