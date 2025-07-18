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

MODEL_NAME = f"lstm_normalizing_non_linear_flows_{LOOKBACK_DAYS}_days{SUFFIX}"
RVOL_DATA_PATH = "data/RVOL.csv"
VIX_DATA_PATH = "data/VIX.csv"

# %%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from shared.processing import get_lstm_train_test_old


import os
import warnings

# %%
# Load data
df, X_train, X_test, y_train, y_test = get_lstm_train_test_old(
    include_log_returns=False
)
df

# %%
# Convert to tensors for model training
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# print tensor shapes
X_train.shape, y_train.shape

# %%
# Define window size
window_size = LOOKBACK_DAYS


# %%
# Define the LSTM-Based Masked Autoregressive Flow
class LSTMConditioningNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers=10):
        super(LSTMConditioningNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # Output mu and log_sigma

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # Get last hidden state
        params = self.fc(h_n[-1])  # Pass through linear layer
        return params.chunk(2, dim=-1)  # Split into mu and log_sigma


# %%
# Define LSTM flow
class LSTMFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMFlow, self).__init__()
        self.conditioner = LSTMConditioningNetwork(input_dim, hidden_dim)

    def forward(self, y, x):
        x = x.view(x.size(0), -1)
        mu, log_sigma = self.conditioner(x)
        log_sigma = torch.clamp(log_sigma, min=-3, max=3)  # Tighter range
        sigma = torch.exp(log_sigma)

        z = (y - mu) / sigma
        log_det_jacobian = -log_sigma.sum(dim=-1)  # Corrected dimension
        return z, log_det_jacobian

    def inverse(self, z, x):
        x = x.view(x.size(0), -1)
        mu, log_sigma = self.conditioner(x)
        sigma = torch.exp(log_sigma)
        z = torch.tanh(z)
        z = 0.5 * torch.log((1 + z) / (1 - z + 1e-6))
        y = z * sigma + mu
        return y


# %%
# Define the masked autoregressive flow network (MAF)
class LSTMMaskedAutoregressiveFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_flows):
        super(LSTMMaskedAutoregressiveFlow, self).__init__()
        self.flows = nn.ModuleList(
            [LSTMFlow(input_dim, hidden_dim) for _ in range(n_flows)]
        )
        self.base_dist = Normal(0, 1)

    def log_prob(self, y, x):
        log_det_jacobian = 0
        z = y.unsqueeze(1)
        for flow in self.flows:
            z, log_det = flow(z, x)
            log_det_jacobian += log_det
        log_prob = self.base_dist.log_prob(z).sum(dim=-1) + log_det_jacobian
        return log_prob

    def sample(self, x, num_samples=1000):
        z = self.base_dist.sample((num_samples, 1))
        for flow in reversed(self.flows):
            z = flow.inverse(z, x)
        return z


# %%
# Define the model
hidden_dim = 64
n_flows = 20
input_dim = 300  # 10 LSTM layers with 30 days lookback
print("Input dimension:", input_dim)
# %%
# Initialize the model
model = LSTMMaskedAutoregressiveFlow(input_dim, hidden_dim, n_flows)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True
)


# %%
# Train the model
# Learning rate schedule
epochs = 45
# Training Loop with Learning Rate Decay
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    log_prob = model.log_prob(y_train, X_train)
    loss = -log_prob.mean()  # Negative log-likelihood
    loss.backward()
    optimizer.step()

    # Adjust learning rate
    scheduler.step(loss.item())

    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# evaluate the model on the test set
model.eval()
# %%
# Predicting the Distribution for 10 Random Test Samples
np.random.seed(42)
random_indices = np.random.choice(len(X_test), 10, replace=False)

# make test set tensors
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

for idx in random_indices:
    specific_sample = X_test[idx].unsqueeze(0)  # Select a random test sample
    actual_return = y_test[idx].item()  # Actual next-day return

    # Generate predicted return distribution
    samples = model.sample(x=specific_sample, num_samples=10000)
    predicted_return = samples.mean().item()  # Mean of predicted distribution

    print("Predicted Return:", predicted_return, "Actual Return:", actual_return)
    # Analyzing Results
    plt.figure(figsize=(8, 4))
    plt.hist(
        samples.detach().numpy().flatten(),
        bins=50,
        density=True,
        alpha=0.6,
        label="Predicted Distribution",
    )
    plt.axvline(
        predicted_return,
        color="blue",
        linestyle="dashed",
        linewidth=2,
        label="Predicted Mean",
    )
    plt.axvline(
        actual_return,
        color="red",
        linestyle="solid",
        linewidth=2,
        label="Actual Return",
    )
    plt.title(f"Predicted Return Distribution for Test Point {idx}")
    plt.legend()
    plt.show()

# %%
# Plot the predicted distribution for the last test sample
specific_sample = X_test[-1].unsqueeze(0)  # Select the last test sample
actual_return = y_test[-1].item()  # Actual next-day return

# Generate predicted return distribution
samples = model.sample(x=specific_sample, num_samples=5000)
predicted_return = samples.mean().item()  # Mean of predicted distribution

# Analyzing Results
plt.figure(figsize=(8, 4))
plt.hist(
    samples.detach().numpy().flatten(),
    bins=50,
    density=True,
    alpha=0.6,
    label="Predicted Distribution",
)
plt.axvline(
    predicted_return,
    color="blue",
    linestyle="dashed",
    linewidth=2,
    label="Predicted Mean",
)
plt.axvline(
    actual_return, color="red", linestyle="solid", linewidth=2, label="Actual Return"
)
plt.title("Predicted Return Distribution for Last Test Point")
# %%
# Predicting the Distribution for the Entire Test Period

predicted_returns = []
predicted_stds = []
actual_returns = y_test.numpy().flatten()

for i in range(len(X_test)):
    specific_sample = X_test[i].unsqueeze(0)
    samples = model.sample(x=specific_sample, num_samples=5000)
    predicted_return = samples.mean().item()
    predicted_std = samples.std().item()
    predicted_stds.append(predicted_std)
    predicted_returns.append(predicted_return)

# Plotting Predicted Returns vs Actual Returns
plt.figure(figsize=(14, 6))
# plt.plot(predicted_returns, label='Predicted Returns', color='blue')
plt.plot(actual_returns, label="Actual Returns", color="red", alpha=0.7)
# add the standard deviation with label
plt.fill_between(
    range(len(predicted_returns)),
    np.array(predicted_returns) - np.array(predicted_stds),
    np.array(predicted_returns) + np.array(predicted_stds),
    color="blue",
    alpha=0.2,
    label="Predicted Return Std",
)
plt.title("Predicted Returns vs Actual Returns (Test Period)")
plt.xlabel("Time Step")
plt.ylabel("Return")
plt.legend()
plt.show()

# %%
# Plot only for the last 100 points
plt.figure(figsize=(14, 6))
plt.plot(predicted_returns[-100:], label="Predicted Returns", color="blue")
plt.plot(actual_returns[-100:], label="Actual Returns", color="red", alpha=0.7)
plt.fill_between(
    range(len(predicted_returns[-100:])),
    np.array(predicted_returns[-100:]) - np.array(predicted_stds[-100:]),
    np.array(predicted_returns[-100:]) + np.array(predicted_stds[-100:]),
    color="blue",
    alpha=0.2,
    label="Predicted Return Std",
)
plt.title("Predicted Returns vs Actual Returns (Last 100 Time Steps)")
plt.xlabel("Time Step")
plt.ylabel("Return")
plt.legend()
plt.show()

# %%
# Store predicted returns and standard deviations in a csv
print("predicted returns lenght:", len(predicted_returns))
# %%
# 8) Store single-pass predictions
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]
df_validation["Mean_SP"] = predicted_returns
df_validation["Vol_SP"] = predicted_stds

df_validation

# %%
# Save the predictions to a CSV file
os.makedirs("predictions", exist_ok=True)
df_validation.to_csv(f"predictions/lstm_maf_v1_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")
# %%
