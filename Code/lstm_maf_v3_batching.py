# %%
# Define parameters
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    TEST_ASSET,
    DATA_PATH,
    TRAIN_VALIDATION_SPLIT,
    VALIDATION_TEST_SPLIT
)
from scipy.stats import ks_2samp


MODEL_NAME = f"LSTM_MAF_v3_{LOOKBACK_DAYS}_days{SUFFIX}"
RVOL_DATA_PATH = "data/RVOL.csv"
VIX_DATA_PATH = "data/VIX.csv"
SPX_DATA_PATH = "data/SPX.csv"

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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


from shared.processing import get_lstm_train_test
from shared.loss import nll_loss_maf
from tqdm import tqdm

import os
import warnings


# %%
# Load preprocessed data
df, X_train, X_test, y_train, y_test = get_lstm_train_test(include_log_returns=False)
df


# remove the following features: 
# %%
# Define window size
window_size = LOOKBACK_DAYS

# %%
# Convert to tensors for model training
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# setup batch size
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# print tensor shapes
X_train.shape, y_train.shape

y_dim = 1  # Dimension of the target variable


# %%
# Define a LSTMFeature extractor model
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout_rate=0.2):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # Use the final hidden state as the feature vector
        return hn[-1]  # Shape: (batch_size, hidden_dim)


# %%
# Define the Masked Autoregressive Network with Non-Linear Flows
class NonLinearFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(NonLinearFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout Layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout Layer
            nn.Linear(hidden_dim, y_dim * 2),  # Outputting both mu and log_sigma
        )

    def forward(self, y, x):
        x = x.view(x.size(0), -1)
        params = self.net(x)
        mu, log_sigma = params.chunk(2, dim=-1)

        sigma = torch.exp(log_sigma)

        # Non-linear transformation
        y = y.reshape(-1, 1)  # Reshape y to match mu and sigma

        # Non-linear transformation
        z = (y - mu) / sigma
        z = torch.tanh(z)  # Adding non-linearity

        log_det_jacobian = -log_sigma.sum(dim=-1) + torch.log(1 - z**2 + 1e-6).sum(
            dim=-1
        )
        return z, log_det_jacobian

    def inverse(self, z, x):
        x = x.view(x.size(0), -1)  # Flatten (30, 10) -> (1, 300)
        params = self.net(x)
        mu, log_sigma = params.chunk(2, dim=-1)

        sigma = torch.exp(log_sigma)
        z = torch.tanh(z)
        z = 0.5 * torch.log((1 + z) / (1 - z + 1e-6))

        y = z * sigma + mu
        return y


# %%
# Define a new type of model
class LSTMMAFModel(nn.Module):
    def __init__(
        self,
        feature_dim,
        input_dim,
        lstm_hidden_dim,
        maf_hidden_dim,
        n_flows,
        num_layers,
        extractor_dropout,
        flow_dropout,
    ):
        super(LSTMMAFModel, self).__init__()
        self.lstm_feature_extractor = LSTMFeatureExtractor(
            feature_dim, lstm_hidden_dim, num_layers, extractor_dropout
        )
        self.maf = MaskedAutoregressiveFlow(
            lstm_hidden_dim, maf_hidden_dim, n_flows, flow_dropout
        )

    def log_prob(self, y, x):
        feature_vector = self.lstm_feature_extractor(x)
        return self.maf.log_prob(y, feature_vector)

    def sample(self, x, num_samples=1000):
        feature_vector = self.lstm_feature_extractor(x)
        return self.maf.sample(feature_vector, num_samples)


# %%
# Define the masked autoregressive flow network (MAF)
class MaskedAutoregressiveFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_flows, flow_dropout):
        super(MaskedAutoregressiveFlow, self).__init__()
        self.flows = nn.ModuleList(
            [NonLinearFlow(input_dim, hidden_dim, flow_dropout) for _ in range(n_flows)]
        )
        self.base_dist = Normal(0, 1)

    def log_prob(self, y, x):
        log_det_jacobian = 0
        # z = y.unsqueeze(1)
        z = y.clone().unsqueeze(1)

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
lstm_hidden_dim = 128  # Adjust based on sequence length and features
maf_hidden_dim = 64  # 64
n_flows = 5  # 20
input_dim = 300  # 10 features * 30 lookback days
# number of features
feature_dim = X_train.shape[-1]
extractor_num_layers = 1
extractor_dropout = 0.15
flow_dropout = 0.15
print("Input dimension:", input_dim)


# %%
# Initialize the model
model = LSTMMAFModel(
    input_dim=input_dim,
    feature_dim=feature_dim,
    lstm_hidden_dim=lstm_hidden_dim,
    maf_hidden_dim=maf_hidden_dim,
    n_flows=n_flows,
    num_layers=extractor_num_layers,
    extractor_dropout=extractor_dropout,
    flow_dropout=flow_dropout,
)

# %%
optimizer = optim.Adam(model.parameters(), lr=1e-4) # weight_decay=1e-4
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True
)


# %%
# Train the model
epochs = 1
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    # Iterate over batches
    for X_batch, y_batch in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
    ):
        optimizer.zero_grad()

        # Compute the log probability for the current batch
        log_prob = model.log_prob(y_batch, X_batch)

        # Calculate the negative log-likelihood loss
        loss = -log_prob.mean()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate loss (multiplied by batch size for averaging later)
        epoch_loss += loss.item() * X_batch.size(0)

    # Calculate average loss for the epoch
    avg_loss = epoch_loss / len(train_dataset)

    # Update the learning rate scheduler
    scheduler.step(avg_loss)

    # Print the loss every 10 epochs
    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}")


# %%
# Predicting the Distribution for 10 Random Test Samples
model.eval()
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
confidence_levels = [0, 0.5, 0.67, 0.90, 0.95, 0.975, 0.99]
intervals = np.zeros((len(X_test), len(confidence_levels), 2))
predicted_returns = []
predicted_stds = []
actual_returns = y_test.numpy().flatten()

for i in range(len(X_test)):
    specific_sample = X_test[i].unsqueeze(0)
    samples = model.sample(x=specific_sample, num_samples=5000)
    samples_np = samples.detach().cpu().numpy().flatten()
    predicted_return = samples.mean().item()
    predicted_std = samples.std().item()
    predicted_stds.append(predicted_std)
    predicted_returns.append(predicted_return)
    # For each confidence level, calculate the lower and upper quantiles.
    for j, cl in enumerate(confidence_levels):
        # For cl=0, you may simply take the median (50th percentile) for both bounds.
        if cl == 0:
            lower_q = 50
            upper_q = 50
        else:
            lower_q = (1 - cl) / 2 * 100
            upper_q = 100 - lower_q
        
        lower_bound = np.percentile(samples_np, lower_q)
        upper_bound = np.percentile(samples_np, upper_q)
        intervals[i, j, 0] = lower_bound
        intervals[i, j, 1] = upper_bound

# Plotting Predicted Returns vs Actual Returns
plt.figure(figsize=(14, 6))
plt.plot(predicted_returns, label="Predicted Returns", color="blue")
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
# Check for nan values in the predicted returns
nan_values = np.isnan(predicted_returns)
# print the number of nan values
print("Number of NaN values in predicted returns:", nan_values.sum())
# print the last predicted return
print("Last predicted return:", predicted_returns[-1])
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
# Calculate the KS statistic between the predicted returns and actual returns, add check for p-value
ks_statistic, p_value = ks_2samp(predicted_returns, actual_returns)
# print the KS statistic and p-value + add checkmark if passed and X if failed
print("KS Statistic:", ks_statistic, "P-Value:", p_value)
if p_value > 0.05:
    print("Passed")
else:
    print("Failed")


print("NLL Loss:", nll_loss_maf(model, X_test, y_test))

# %%
# Store predicted returns and standard deviations in a csv
print("predicted returns lenght:", len(predicted_returns))
print("predicted stds lenght:", len(predicted_stds))

# %%
# 8) Store single-pass predictions
df_validation = df.xs(TEST_ASSET, level="Symbol").loc[
    TRAIN_VALIDATION_SPLIT:VALIDATION_TEST_SPLIT
]  # TEST_ASSET
df_validation["Mean_SP"] = predicted_returns
df_validation["Vol_SP"] = predicted_stds
df_validation["NLL"] = nll_loss_maf(model, X_test, y_test)

# Store the intervals in the DataFrame.
for j, cl in enumerate(confidence_levels):
    df_validation[f"LB_{int(100*cl)}"] = intervals[:, j, 0]
    df_validation[f"UB_{int(100*cl)}"] = intervals[:, j, 1]

df_validation

# %%
# Save the predictions to a CSV file
os.makedirs("predictions", exist_ok=True)
df_validation.to_csv(
    f"predictions/lstm_MAF_v3_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"  # TEST_ASSET
)


# %%
# Define MC-dropout for uncertainty estimation
def mc_dropout_sample(model, X, num_samples=1000):
    """Perform MC-Dropout for uncertainty estimation."""
    model.train()  # Keep dropout active during test-time sampling
    predictions = []

    for _ in range(num_samples):
        pred = model.sample(X)  # Sample prediction from the model
        predictions.append(pred.detach().cpu().numpy())

    predictions = np.array(predictions)
    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(
        axis=0
    )  # Uncertainty estimate (standard deviation)

    return mean_prediction, std_prediction
