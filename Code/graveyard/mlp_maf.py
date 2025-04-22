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

MODEL_NAME = f"mlp_normalizing_flows_{LOOKBACK_DAYS}_days{SUFFIX}"
RVOL_DATA_PATH = "data/RVOL.csv"
VIX_DATA_PATH = "data/VIX.csv"

# %%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from arch import arch_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import os
import warnings

# %%
# Load data
df = pd.read_csv(DATA_PATH)

if not "Symbol" in df.columns:
    df["Symbol"] = TEST_ASSET

# Ensure the Date column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])
df

# %%

# Sort the dataframe by both Date and Symbol
df = df.sort_values(["Symbol", "Date"])

df

# %%
# Calculate log returns for each instrument separately using groupby
df["LogReturn"] = (
    df.groupby("Symbol")["Close"].apply(lambda x: np.log(x / x.shift(1))).droplevel(0)
)
# Drop rows where LogReturn is NaN (i.e., the first row for each instrument)
df = df[~df["LogReturn"].isnull()]

# Making a squared return column
df["SquaredReturn"] = df["LogReturn"] ** 2
# Set date and symbol as index
df: pd.DataFrame = df.set_index(["Date", "Symbol"])

df

# %%
# Remove data before 1990
df = df.loc[df.index.get_level_values("Date") >= "1990-01-01"]
df

# %%
# Remove test data (we only use train and validation at this point)
df = df.loc[df.index.get_level_values("Date") < VALIDATION_TEST_SPLIT]
df

# %%
# Check if TEST_ASSET is in the data
if TEST_ASSET not in df.index.get_level_values("Symbol"):
    raise ValueError(f"{TEST_ASSET} not in data")

# %%
# Define window size
window_size = LOOKBACK_DAYS


# %%
# Prepare features
def create_sequence(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


# create the sequences
X, y = create_sequence(df["LogReturn"].values, window_size)
X.shape, y.shape

# %%
print("X:")
X

# %%
print("y:")
y
# %%
# Convert to tensors for model training
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# split the data into training and test sets based on TRAIN_VALIDATION_SPLIT which is a date
split = df.index.get_level_values("Date") < TRAIN_VALIDATION_SPLIT
split_index = split.sum() - 30  # FIX THIS -30 STUFF HERE # Number of training samples
# print the number of test points we will have
print("Number of test points:", len(y) - split_index)
print("Split index:", split_index)

# Split the data into training and test sets
X_train, X_test = X_tensor[:split_index], X_tensor[split_index:]
y_train, y_test = y_tensor[:split_index], y_tensor[split_index:]

X_test.shape, y_test.shape


# X_train, X_test = X_tensor[:split], X_tensor[split:]
# y_train, y_test = y_tensor[:split], y_tensor[split:]
# %%
# Define the masked autoregressive flow network (MAF)
class MaskedAutoregressiveFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_flows):
        super(MaskedAutoregressiveFlow, self).__init__()
        self.n_flows = n_flows
        self.flows = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_dim, 2 * input_dim
                    ),  # Outputting both mu and log_sigma
                )
                for _ in range(n_flows)
            ]
        )
        self.base_dist = Normal(0, 1)

    def forward(self, x):
        log_det_jacobian = 0
        for flow in self.flows:
            params = flow(x)
            mu, log_sigma = params.chunk(2, dim=-1)
            sigma = torch.exp(log_sigma)
            x = (x - mu) / sigma  # Inverse transformation
            log_det_jacobian -= log_sigma.sum(dim=-1)
        return x, log_det_jacobian

    def inverse(self, z, x):
        for flow in reversed(self.flows):
            params = flow(x)
            mu, log_sigma = params.chunk(2, dim=-1)
            sigma = torch.exp(log_sigma)
            z = z * sigma + mu  # Forward transformation
        return z

    def log_prob(self, y, x):
        log_det_jacobian = 0
        z = y.unsqueeze(1)  # Ensure z has the correct shape (batch_size, 1)

        # Apply each flow conditioned on x
        for flow in self.flows:
            params = flow(x)  # Conditioning on features x
            mu, log_sigma = params.chunk(2, dim=-1)
            sigma = torch.exp(log_sigma)

            # Transform y into z using mu and sigma
            z = (z - mu) / sigma
            log_det_jacobian -= log_sigma.sum(dim=-1)

        # Compute log-probability of z under the base distribution
        log_prob = self.base_dist.log_prob(z).sum(dim=-1) + log_det_jacobian
        return log_prob

    def sample(self, x, num_samples=1000):
        z = self.base_dist.sample((num_samples, x.shape[1]))
        return self.inverse(z, x)


# %%
# Define the model
hidden_dim = 64
n_flows = 5
input_dim = X_train.shape[-1]
print("Input dimension:", input_dim)
# %%
# Initialize the model
model = MaskedAutoregressiveFlow(input_dim, hidden_dim, n_flows)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# %%
# Train the model
# Learning rate schedule
learning_rates = [0.01, 0.005, 0.001]  # Starting high, then reducing
epochs = [100, 500, 1000]  # Number of epochs for each learning rate

# Training Loop with Learning Rate Decay
for i in range(len(learning_rates)):
    optimizer = optim.Adam(model.parameters(), lr=learning_rates[i])
    for epoch in range(epochs[i]):
        model.train()
        optimizer.zero_grad()
        log_prob = model.log_prob(y_train, X_train)
        loss = -log_prob.mean()  # Negative log-likelihood
        loss.backward()
        optimizer.step()
        print(
            f"Learning Rate: {learning_rates[i]}, Epoch: {epoch + 1}, Loss: {loss.item()}"
        )

# evaluate the model on the test set
model.eval()
# with torch.no_grad():
#     test_log_prob = model.log_prob(y_test, X_test)
#     test_loss = -test_log_prob.mean()
#     print(f'Test Loss: {test_loss.item()}')

# %%
# Predicting the Distribution for 10 Random Test Samples
np.random.seed(42)
random_indices = np.random.choice(len(X_test), 10, replace=False)

for idx in random_indices:
    specific_sample = X_test[idx].unsqueeze(0)  # Select a random test sample
    actual_return = y_test[idx].item()  # Actual next-day return

    # Generate predicted return distribution
    samples = model.sample(x=specific_sample, num_samples=1000)
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
samples = model.sample(x=specific_sample, num_samples=1000)
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
    samples = model.sample(x=specific_sample, num_samples=1000)
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
df_validation.to_csv(f"predictions/mlp_maf_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv")
# %%
