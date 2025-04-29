# %%
import json
import os

# Change to the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Change to Code folder so that imports and paths work correctly
os.chdir("../..")

import optuna
import pandas as pd
import numpy as np
import subprocess
from xgboost import XGBRegressor
import sys
from xgboost.callback import EarlyStopping

# Add shared path and import
from shared.processing import get_lstm_train_test_new

# %%
# =============================================================================
# 1. Load Data
# =============================================================================
INCLUDE_RV = True
INCLUDE_IV = True

print("Loading data...")
data = get_lstm_train_test_new(
    include_1min_rv=INCLUDE_RV,
    include_5min_rv=INCLUDE_RV,
    include_ivol_cols=(
        ["10 Day Call IVOL", "Historical Call IVOL"] if INCLUDE_IV else []
    ),
)

# %%
# Convert to tabular
# Extract train and validation sets
X_train = data.train.X[:, -1, :]
y_train = data.train.y
X_val = data.validation.X[:, -1, :]
y_val = data.validation.y

# Add tickers to X_train and X_val
X_train = np.concatenate([X_train, data.train.tickers.reshape(-1, 1)], axis=1)
X_val = np.concatenate([X_val, data.validation.tickers.reshape(-1, 1)], axis=1)

# Create feature column names based on actual names
feature_cols = list(data.train.df.columns.drop("ActualReturn")) + ["Symbol"]

# Create DataFrames
X_train = pd.DataFrame(X_train, columns=feature_cols)
X_val = pd.DataFrame(X_val, columns=feature_cols)

dtypes = ["float64"] * (X_train.shape[1] - 1) + ["category"]
for col, dtype in zip(X_train.columns, dtypes):
    X_train[col] = X_train[col].astype(dtype)
    X_val[col] = X_val[col].astype(dtype)

print("Data loaded.")

# %%
# Define the confidence levels and quantiles of interest
CONFIDENCE_LEVELS = [0.90, 0.95, 0.98]
lower_quantiles = [np.round((1 - cl) / 2, 5) for cl in CONFIDENCE_LEVELS]
upper_quantiles = [1 - lq for lq in lower_quantiles]
n_es_quantiles = 5  # We use 5 quantiles to approximate the expected shortfall
es_quantiles = [
    np.round(small_q, 5)
    for q in lower_quantiles
    for small_q in np.linspace(0, q, n_es_quantiles + 1)[1:]
]
all_quantiles = [
    float(n) for n in sorted(set(lower_quantiles + upper_quantiles + es_quantiles))
]
print(f"Quantiles: {all_quantiles}")


# %%
# =============================================================================
# 2. Define Quantile Loss
# =============================================================================
def quantile_loss(y_true, y_pred, alpha):
    residual = y_true - y_pred
    return np.maximum(alpha * residual, (alpha - 1) * residual).mean()


# %%
# =============================================================================
# 3. Define Optuna Objective
# =============================================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
        "max_depth": trial.suggest_int("max_depth", 2, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 100.0, log=True),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.1, 300.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "max_bin": trial.suggest_int("max_bin", 128, 1024),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "early_stopping_rounds": 20,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 72,
    }

    print(f"Hyperparameters: {json.dumps(params, indent=2)}")

    quantile_losses = {}

    for alpha in all_quantiles:
        print(f"Training for quantile: {alpha:.5f}")

        model = XGBRegressor(
            **params,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        loss = quantile_loss(y_val, preds, alpha)
        quantile_losses[alpha] = loss

    # Store each quantile loss as a user attribute
    for alpha, loss in quantile_losses.items():
        trial.set_user_attr(f"QL_{alpha:.3f}", loss)

    avg_loss = np.mean(list(quantile_losses.values()))
    return avg_loss


# %%
# =============================================================================
# 4. Git Commit Callback
# =============================================================================
def git_commit_callback(study: optuna.Study, trial: optuna.Trial):
    print(
        f"Trial {trial.number} finished with value: {trial.value}. Committing DB to git."
    )
    try:
        subprocess.run(["git", "pull", "--no-edit"], check=True)
        subprocess.run(["git", "add", "optuna"], check=True)
        commit_header = f"XGB tuning trial {trial.number} - Updated study DB"
        commit_body = (
            f"Trial {trial.number} finished with objective value: {trial.value}\n"
            f"Hyperparameters: {trial.params}\n"
        )

        # Include per-quantile losses
        for key, val in trial.user_attrs.items():
            commit_body += f"{key}: {val}\n"

        subprocess.run(
            ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
        )
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")


# %%
# =============================================================================
# 5. Run Optuna Study with SQLite Storage
# =============================================================================
storage_url = "sqlite:///optuna/optuna.db"
study = optuna.create_study(
    direction="minimize",
    study_name="xgb_tuning",
    storage=storage_url,
    load_if_exists=True,
)
try:
    n_trials = int(sys.argv[1])
except:
    n_trials = 100
study.optimize(objective, n_trials=n_trials, callbacks=[git_commit_callback])

# %%
