# %%
# Define parameters
from settings import (
    LOOKBACK_DAYS,
    SUFFIX,
    TEST_ASSET,
    DATA_PATH,
    TRAIN_VALIDATION_SPLIT,
)
from scipy.stats import ks_2samp


MODEL_NAME = f"LSTM_MAF_{LOOKBACK_DAYS}_days{SUFFIX}"
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

from shared.processing import get_lstm_train_test_old
from shared.loss import nll_loss_maf

import os
import warnings


# %%
# Load preprocessed data
df, X_train, X_test, y_train, y_test = get_lstm_train_test_old(
    include_log_returns=False
)
df

# %%
# Define window size
window_size = LOOKBACK_DAYS

# %%
# Convert to tensors for model training
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# print tensor shapes
X_train.shape, y_train.shape

y_dim = 1  # Dimension of the target variable


# %%
# ====================================================
# 1. LSTM Feature Extractor
# ====================================================
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout_rate=0.2):
        """
        An LSTM-based feature extractor. Given a time series input, it outputs the final hidden state.

        Args:
            input_dim (int): Number of features in the input.
            hidden_dim (int): Hidden dimension of the LSTM.
            num_layers (int): Number of LSTM layers.
            dropout_rate (float): Dropout rate between LSTM layers.
        """
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        _, (hn, _) = self.lstm(x)
        # Return the last layer's hidden state: shape (batch_size, hidden_dim)
        return hn[-1]


# %%
# ====================================================
# 2. Expanded Flow Block (as used in the flow network)
# ====================================================
class ExpandedFlowBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.2):
        """
        A deep feed-forward network with residual blocks for computing flow parameters.

        Args:
            input_dim (int): Dimensionality of the input (e.g. the flattened context).
            hidden_dim (int): Number of units for hidden layers.
            output_dim (int): Output dimension (typically 2*y_dim, for mu and log_sigma).
            num_layers (int): Number of residual layers.
            dropout_rate (float): Dropout probability.
        """
        super(ExpandedFlowBlock, self).__init__()
        # Initial transformation of the input
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_act = nn.ReLU()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout_rate)

        # Build residual layers
        self.residual_layers = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate),
            )
            self.residual_layers.append(block)

        # Output layer that produces parameters for mu and log_sigma
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the expanded flow block.

        Args:
            x (Tensor): Input of shape (batch_size, input_dim)

        Returns:
            Tensor: Output of shape (batch_size, output_dim)
        """
        out = self.input_layer(x)
        out = self.input_act(out)
        out = self.input_norm(out)
        out = self.input_dropout(out)

        for block in self.residual_layers:
            residual = out  # Save for the skip connection
            out = block(out)
            out = out + residual  # Residual (skip) connection

        out = self.output_layer(out)
        return out


# %%
# ====================================================
# 3. Expanded Non-Linear Flow
# ====================================================
class ExpandedNonLinearFlow(nn.Module):
    def __init__(self, context_dim, hidden_dim, num_layers, dropout_rate=0.2, y_dim=1):
        """
        A flow layer that uses an expanded network to condition on context and transform y.

        Args:
            context_dim (int): Dimension of the context vector (e.g. from the LSTM).
            hidden_dim (int): Hidden dimension for the flow network.
            num_layers (int): Number of residual layers in the flow block.
            dropout_rate (float): Dropout rate.
            y_dim (int): Dimensionality of the target variable.
        """
        super(ExpandedNonLinearFlow, self).__init__()
        # The network takes the flattened context and produces parameters for transforming y.
        self.context_net = ExpandedFlowBlock(
            input_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=y_dim * 2,  # To produce mu and log_sigma
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
        self.y_dim = y_dim

    def forward(self, y, context):
        """
        Forward pass for the flow layer.

        Args:
            y (Tensor): Target variable of shape (batch_size, y_dim)
            context (Tensor): Conditioning context of shape (batch_size, context_dim)

        Returns:
            z (Tensor): Transformed variable.
            log_det (Tensor): Log-determinant of the Jacobian.
        """
        params = self.context_net(context)  # (batch_size, 2*y_dim)
        mu, log_sigma = params.chunk(2, dim=-1)

        # Prevent sigma from collapsing to zero
        sigma = torch.exp(log_sigma).clamp(min=1e-6)

        # Ensure y has shape (batch_size, y_dim)
        y = y.view(-1, self.y_dim)

        # Affine transformation: standardize y
        u = (y - mu) / sigma

        # Non-linear transformation for flexibility
        z = torch.tanh(u)

        # Compute the log-determinant of the Jacobian.
        # Derivative of tanh(u) is (1 - tanh(u)^2), then adjust for the division by sigma.
        log_det = -torch.log(sigma).sum(dim=-1) + torch.log(1 - z**2 + 1e-6).sum(dim=-1)
        return z, log_det

    def inverse(self, z, context):
        """
        Inverse transformation for sampling.

        Args:
            z (Tensor): Latent variable of shape (batch_size, y_dim)
            context (Tensor): Conditioning context of shape (batch_size, context_dim)

        Returns:
            y (Tensor): Inverted sample (original space).
        """
        params = self.context_net(context)
        mu, log_sigma = params.chunk(2, dim=-1)
        sigma = torch.exp(log_sigma).clamp(min=1e-6)

        # Invert the tanh using arctanh, ensuring numerical stability
        z = torch.clamp(z, -0.999, 0.999)
        u = 0.5 * torch.log((1 + z) / (1 - z + 1e-6))

        # Reverse the affine transformation
        y = u * sigma + mu
        return y


# %%
# ====================================================
# 4. Expanded Masked Autoregressive Flow
# ====================================================
class ExpandedMaskedAutoregressiveFlow(nn.Module):
    def __init__(
        self, context_dim, hidden_dim, num_layers, num_flows, dropout_rate=0.2, y_dim=1
    ):
        """
        Stack multiple flow layers to build a flexible autoregressive flow.

        Args:
            context_dim (int): Dimension of the context vector.
            hidden_dim (int): Hidden dimension for each flow block.
            num_layers (int): Number of residual layers in each flow block.
            num_flows (int): Number of flow blocks to stack.
            dropout_rate (float): Dropout rate.
            y_dim (int): Dimensionality of the target variable.
        """
        super(ExpandedMaskedAutoregressiveFlow, self).__init__()
        self.flows = nn.ModuleList(
            [
                ExpandedNonLinearFlow(
                    context_dim, hidden_dim, num_layers, dropout_rate, y_dim
                )
                for _ in range(num_flows)
            ]
        )
        self.base_dist = torch.distributions.Normal(0, 1)
        self.y_dim = y_dim

    def log_prob(self, y, context):
        """
        Compute the log probability of y given context.

        Args:
            y (Tensor): Target variable (batch_size, y_dim)
            context (Tensor): Conditioning context (batch_size, context_dim)

        Returns:
            Tensor: Log probability for each sample in the batch.
        """
        log_det_jacobian = 0.0
        z = y
        for flow in self.flows:
            z, log_det = flow(z, context)
            log_det_jacobian += log_det
        base_log_prob = self.base_dist.log_prob(z).sum(dim=-1)
        return base_log_prob + log_det_jacobian

    def sample(self, context, num_samples=1000):
        """
        Generate samples for each element in the batch given a context.

        Args:
            context (Tensor): Conditioning context (batch_size, context_dim)
            num_samples (int): Number of samples to generate per batch element.

        Returns:
            Tensor: Samples with shape (num_samples, batch_size, y_dim)
        """
        base_samples = self.base_dist.sample((num_samples, context.size(0), self.y_dim))
        z = base_samples
        # Inverse the flow transformation
        for flow in reversed(self.flows):
            new_z = []
            for i in range(z.size(0)):
                sample = z[i]  # shape: (batch_size, y_dim)
                sample_inv = flow.inverse(sample, context)
                new_z.append(sample_inv)
            z = torch.stack(new_z, dim=0)
        return z


# %%
# ====================================================
# 5. Full Model: LSTM Integrated with the Flow Network
# ====================================================
class LSTMFlowModel(nn.Module):
    def __init__(
        self,
        lstm_input_dim,
        lstm_hidden_dim,
        lstm_num_layers,
        flow_context_dim,
        flow_hidden_dim,
        flow_num_layers,
        num_flows,
        dropout_rate=0.2,
        y_dim=1,
    ):
        """
        A full model that integrates an LSTM feature extractor with an expanded flow network.

        Args:
            lstm_input_dim (int): Number of features per time step.
            lstm_hidden_dim (int): Hidden dimension of the LSTM.
            lstm_num_layers (int): Number of LSTM layers.
            flow_context_dim (int): Dimension of the context used by the flow network.
            flow_hidden_dim (int): Hidden dimension within the flow network.
            flow_num_layers (int): Number of residual layers per flow block.
            num_flows (int): Number of flow blocks to stack.
            dropout_rate (float): Dropout rate (used both in LSTM and flow network).
            y_dim (int): Dimensionality of the target variable.
        """
        super(LSTMFlowModel, self).__init__()
        self.lstm_feature_extractor = LSTMFeatureExtractor(
            lstm_input_dim, lstm_hidden_dim, lstm_num_layers, dropout_rate
        )

        # Optionally project LSTM output to desired flow context dimension if needed.
        if lstm_hidden_dim != flow_context_dim:
            self.context_proj = nn.Linear(lstm_hidden_dim, flow_context_dim)
        else:
            self.context_proj = None

        self.flow = ExpandedMaskedAutoregressiveFlow(
            context_dim=flow_context_dim,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_num_layers,
            num_flows=num_flows,
            dropout_rate=dropout_rate,
            y_dim=y_dim,
        )

    def forward(self, x, y):
        """
        Compute the log probability (NLL) of y given input sequence x.

        Args:
            x (Tensor): Input time series (batch_size, sequence_length, lstm_input_dim)
            y (Tensor): Target variable (batch_size, y_dim)

        Returns:
            Tensor: Log probability for each sample in the batch.
        """
        # Extract features from the time series using the LSTM.
        context = self.lstm_feature_extractor(x)  # (batch_size, lstm_hidden_dim)
        if self.context_proj is not None:
            context = self.context_proj(context)  # (batch_size, flow_context_dim)
        # Compute and return the log probability using the flow network.
        return self.flow.log_prob(y, context)

    def sample(self, x, num_samples=1000):
        """
        Generate samples of y given the input sequence x.

        Args:
            x (Tensor): Input time series (batch_size, sequence_length, lstm_input_dim)
            num_samples (int): Number of samples to generate per input.

        Returns:
            Tensor: Generated samples with shape (num_samples, batch_size, y_dim)
        """
        context = self.lstm_feature_extractor(x)
        if self.context_proj is not None:
            context = self.context_proj(context)
        return self.flow.sample(context, num_samples=num_samples)


# %%
# Define the model

# Example hyperparameters.
lstm_input_dim = X_train.shape[-1]  # e.g., 10 features per time step.
sequence_length = LOOKBACK_DAYS  # e.g., lookback period of 30 days.
lstm_hidden_dim = 256
lstm_num_layers = 1

# Flow network hyperparameters.
flow_context_dim = 256  # We can set this equal to lstm_hidden_dim.
flow_hidden_dim = 128
flow_num_layers = 3  # Number of residual layers in each flow block.
num_flows = 10
dropout_rate = 0.1
y_dim = 1  # Target variable dimension (e.g., one return value)


# %%
# Initialize the model
model = LSTMFlowModel(
    lstm_input_dim=lstm_input_dim,
    lstm_hidden_dim=lstm_hidden_dim,
    lstm_num_layers=lstm_num_layers,
    flow_context_dim=flow_context_dim,
    flow_hidden_dim=flow_hidden_dim,
    flow_num_layers=flow_num_layers,
    num_flows=num_flows,
    dropout_rate=dropout_rate,
    y_dim=y_dim,
)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True
)


# %%
# Train the model
epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    log_prob = model.log_prob(y_train, X_train)
    loss = -log_prob.mean()  # Negative log-likelihood
    loss.backward()
    optimizer.step()

    # Adjust learning rate
    scheduler.step(loss.item())

    # Print the loss every 10 epochs
    epoch += 1

    if (epoch) % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


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
    TRAIN_VALIDATION_SPLIT:
]  # TEST_ASSET
df_validation["Mean_SP"] = predicted_returns
df_validation["Vol_SP"] = predicted_stds
df_validation["NLL"] = nll_loss_maf(model, X_test, y_test)

df_validation

# %%
# Save the predictions to a CSV file
os.makedirs("predictions", exist_ok=True)
df_validation.to_csv(
    f"predictions/lstm_MAF_v2_{TEST_ASSET}_{LOOKBACK_DAYS}_days.csv"  # TEST_ASSET
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
