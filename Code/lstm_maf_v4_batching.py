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
from scipy.stats import ks_2samp


MODEL_NAME = f"LSTM_MAF_v4{LOOKBACK_DAYS}_days{SUFFIX}"
RVOL_DATA_PATH = "data/RVOL.csv"
VIX_DATA_PATH = "data/VIX.csv"
SPX_DATA_PATH = "data/SPX.csv"

# %%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from shared.processing import get_lstm_train_test_new
from shared.loss import nll_loss_maf
from shared.n_flows import compute_confidence_intervals
from tqdm import tqdm

import os
import warnings


# %%
# Load preprocessed data
data = get_lstm_train_test_new()
data


# remove the following features: 
# %%
# Define window size
X_train = data.train.X
y_train = data.train.y
X_val = data.validation.X
y_val = data.validation.y

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_val.shape)
print("y_test shape:", y_val.shape)
window_size = LOOKBACK_DAYS
# %%
# Convert to tensors for model training
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# setup batch size
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


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

        log_det_jacobian = -log_sigma.sum(dim=-1) + torch.log(1 - z**2 + 1e-6).sum(dim=-1)
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
        """
        Generates samples from the flow-based distribution given input features x.
        
        Args:
            x: Tensor of shape (batch_size, feature_dim)
            num_samples: Number of Monte Carlo samples per test sample.

        Returns:
            - Shape (batch_size, num_samples) for inference
        """
        batch_size = x.shape[0]
        
        # Sample from base distribution (Standard Normal)
        z = self.base_dist.sample((batch_size, num_samples, 1)).squeeze(-1)  # Shape: (batch_size, num_samples)
        
        # Pass through the normalizing flows
        for flow in reversed(self.flows):
            z = flow.inverse(z, x)

        return z  # Shape: (batch_size, num_samples)


# %%
# Define the model
hidden_dim = 64
lstm_hidden_dim = 128  # Adjust based on sequence length and features
maf_hidden_dim = 64  # 64
n_flows = 5  # 20
input_dim = 300  # 10 features * 30 lookback days
# number of features
feature_dim = X_train.shape[-1]
extractor_num_layers = 2
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
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True
)

# Convert validation set to tensors
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()

# Create DataLoader for validation set
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
train_losses = []
val_losses = []

# %%
# Train the model
epochs = 3
l2_lambda = 1e-4  # Regularization strength
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0.0

    # Iterate over batches
    for X_batch, y_batch in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        optimizer.zero_grad()

        # Compute the log probability for the current batch
        log_prob = model.log_prob(y_batch, X_batch)

        # Calculate the negative log-likelihood loss
        base_loss = -log_prob.mean()

        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = base_loss + l2_lambda * l2_norm
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate loss (multiplied by batch size for averaging later)
        epoch_train_loss += loss.item() * X_batch.size(0)


    avg_train_loss = epoch_train_loss / len(train_dataset)

    # Update the learning rate scheduler
    scheduler.step(avg_train_loss)

    # ====== VALIDATION PHASE ======
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():  # No gradient computation for validation
        for X_batch, y_batch in val_loader:
            log_prob = model.log_prob(y_batch, X_batch)
            base_loss = -log_prob.mean()
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            val_loss = base_loss + l2_lambda * l2_norm

            epoch_val_loss += val_loss.item() * X_batch.size(0)

    avg_val_loss = epoch_val_loss / len(val_dataset)
    val_losses.append(avg_val_loss)

    # Update the learning rate scheduler based on validation loss
    scheduler.step(avg_val_loss)

    # Print loss per epoch
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")


# %%
# Predicting the Distribution for 10 Random Test Samples
model.eval()
np.random.seed(42)

# make test set tensors
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()
example_tickers = ["GOOG", "AON", "WMT"]

# %%
# Print smooted distributions
for ticker in example_tickers:
    from_idx, to_idx = data.validation.get_range(ticker)
    random_indices = np.random.choice(range(from_idx, to_idx), 10)

    for idx in random_indices:
        specific_sample = X_val[idx].unsqueeze(0)  # Select a random test sample
        actual_return = y_val[idx].item()  # Actual next-day return

        # Generate predicted return distribution
        samples = model.sample(x=specific_sample, num_samples=25000).detach().numpy().flatten()
        predicted_return = np.mean(samples)  # Mean of predicted distribution

        print("Predicted Return:", predicted_return, "Actual Return:", actual_return)

        # Plot KDE with proper empirical representation
        plt.figure(figsize=(8, 4))
        sns.kdeplot(
            samples,
            bw_adjust=0.5,  # Adjust bandwidth for smoothing; lower values make it follow the data more closely
            fill=True,
            alpha=0.6,
            label="Predicted Distribution",
        )
        
        # Add vertical lines for actual and predicted return
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

        plt.title(f"Predicted Return Distribution for {ticker} Test Point {idx}")
        plt.legend()
        plt.show()

# %%
# Predicting the Distribution for the Entire Validation Period
batch_size = 64
# Move tensors to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_val = X_val.to(device)
y_val = y_val.to(device)
model.to(device)
X_val_dataset = TensorDataset(X_val)
X_val_loader = DataLoader(X_val_dataset, batch_size=batch_size, shuffle=False)

confidence_levels = np.array([0, 0.5, 0.67, 0.90, 0.95, 0.975, 0.99], dtype=np.float64)
num_test_samples = len(X_val)
num_conf_levels = len(confidence_levels)

intervals = np.zeros((len(X_val), len(confidence_levels), 2))
predicted_returns = np.zeros(num_test_samples)
predicted_stds = np.zeros(num_test_samples)
nll_loss = np.zeros(num_test_samples)
actual_returns = y_val.cpu().numpy().flatten()

# batched prediction
with torch.no_grad():
    batch_start = 0
    for batch in tqdm(X_val_loader, desc="Processing samples", unit="batch"):
        X_batch = batch[0]
        batch_size = X_batch.size(0)

        # generate samples
        samples = model.sample(x=X_batch, num_samples=10000)
        samples_np = samples.detach().cpu().numpy()

        # calculate predicted return and std
        predicted_return = samples.mean(dim=1).numpy().flatten()
        predicted_std = samples.std(dim=1).numpy().flatten()

        # calculate nll loss
        nll_loss[batch_start:batch_start + batch_size] = model.log_prob(y_val[batch_start:batch_start + batch_size], X_batch).numpy()

        # store predicted return and std
        predicted_returns[batch_start:batch_start + batch_size] = predicted_return
        predicted_stds[batch_start:batch_start + batch_size] = predicted_std

        # 🚀 **Numba-Optimized Confidence Intervals**
        intervals[batch_start:batch_start + batch_size] = compute_confidence_intervals(samples_np, confidence_levels)

        # update batch start
        batch_start += batch_size

# %%
# Plotting the Predicted Returns and Confidence Intervals for example tickers
for ticker in example_tickers:
    from_idx, to_idx = data.validation.get_range(ticker)

    plt.figure(figsize=(12, 5))
    
    # Actual vs Predicted Returns
    plt.plot(range(from_idx, to_idx), actual_returns[from_idx:to_idx], label="Actual Returns", color="red", linestyle="solid")
    plt.plot(range(from_idx, to_idx), predicted_returns[from_idx:to_idx], label="Predicted Mean Returns", color="blue", linestyle="dashed")

    # Standard Deviation
    plt.fill_between(
        range(from_idx, to_idx),
        np.array(predicted_returns[from_idx:to_idx]) - np.array(predicted_stds[from_idx:to_idx]),
        np.array(predicted_returns[from_idx:to_idx]) + np.array(predicted_stds[from_idx:to_idx]),
        color="blue",
        alpha=0.2,
        label="Predicted Return Std",
    )
    # Confidence Interval (e.g., 95%)
    plt.fill_between(
        range(from_idx, to_idx),
        intervals[from_idx:to_idx, 4, 0],  # Lower bound (95% confidence)
        intervals[from_idx:to_idx, 4, 1],  # Upper bound (95% confidence)
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval"
    )

    plt.title(f"Predicted Return Series for {ticker}")
    plt.legend()
    plt.show()

    # plot only the last 100 points
    plt.figure(figsize=(12, 5))
    plt.plot(range(to_idx-100, to_idx), actual_returns[to_idx-100:to_idx], label="Actual Returns", color="red", linestyle="solid")
    plt.plot(range(to_idx-100, to_idx), predicted_returns[to_idx-100:to_idx], label="Predicted Mean Returns", color="blue", linestyle="dashed")
    plt.fill_between(
        range(to_idx-100, to_idx),
        np.array(predicted_returns[to_idx-100:to_idx]) - np.array(predicted_stds[to_idx-100:to_idx]),
        np.array(predicted_returns[to_idx-100:to_idx]) + np.array(predicted_stds[to_idx-100:to_idx]),
        color="blue",
        alpha=0.2,
        label="Predicted Return Std",
    )
    plt.fill_between(
        range(to_idx-100, to_idx),
        intervals[to_idx-100:to_idx, 4, 0],  # Lower bound (95% confidence)
        intervals[to_idx-100:to_idx, 4, 1],  # Upper bound (95% confidence)
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval"
    )
    plt.title(f"Predicted Return Series for {ticker} (Last 100 Time Steps)")
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
# Calculate NLL - this value should be the same as in the on in compare models
# print("NLL Loss:", nll_loss_maf(model, X_val, y_val))

# %%
# Check lenghts of the predicted returns and stds
print("predicted returns lenght:", len(predicted_returns))
print("predicted stds lenght:", len(predicted_stds))

# %%
# 8) Store single-pass predictions
df_validation = pd.DataFrame(
    np.vstack([data.validation.dates, data.validation.tickers]).T,
    columns=["Date", "Symbol"],
)
df_validation["Mean_SP"] = predicted_returns
df_validation["Vol_SP"] = predicted_stds
df_validation["NLL"] = nll_loss

# Store the intervals in the DataFrame.
for j, cl in enumerate(confidence_levels):
    df_validation[f"LB_{int(100*cl)}"] = intervals[:, j, 0]
    df_validation[f"UB_{int(100*cl)}"] = intervals[:, j, 1]

df_validation

# %%
# Save the predictions to a CSV file
df_validation.set_index(["Date", "Symbol"]).to_csv(
    f"predictions/lstm_MAF_v4{SUFFIX}.csv"
)


# %%
# Define MC-dropout for uncertainty estimation (NOT WORKING CURRENTLY)
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
