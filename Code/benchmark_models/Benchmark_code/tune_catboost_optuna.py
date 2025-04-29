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
import sys
import os
from catboost import CatBoostRegressor, Pool

from shared.processing import get_lstm_train_test_new

# =============================================================================
# 1. Setup
# =============================================================================
# Ensure correct working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("../..")

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

X_train = data.train.X[:, -1, :]
y_train = data.train.y
X_val = data.validation.X[:, -1, :]
y_val = data.validation.y

# Add tickers
X_train = np.concatenate([X_train, data.train.tickers.reshape(-1, 1)], axis=1)
X_val = np.concatenate([X_val, data.validation.tickers.reshape(-1, 1)], axis=1)

feature_cols = list(data.train.df.columns.drop("ActualReturn")) + ["Symbol"]

X_train = pd.DataFrame(X_train, columns=feature_cols)
X_val = pd.DataFrame(X_val, columns=feature_cols)

# Fix dtypes
cat_feature = "Symbol"
dtypes = ["float64"] * (len(feature_cols) - 1) + ["category"]
for col, dtype in zip(X_train.columns, dtypes):
    X_train[col] = X_train[col].astype(dtype)
    X_val[col] = X_val[col].astype(dtype)

cat_feature_index = [X_train.columns.get_loc(cat_feature)]

print("Data loaded.")

CONFIDENCE_LEVELS = [0.90, 0.95, 0.98]
lower_quantiles = [np.round((1 - cl) / 2, 5) for cl in CONFIDENCE_LEVELS]
upper_quantiles = [1 - lq for lq in lower_quantiles]
n_es_quantiles = 5
es_quantiles = [
    np.round(small_q, 5)
    for q in lower_quantiles
    for small_q in np.linspace(0, q, n_es_quantiles + 1)[1:]
]
all_quantiles = [
    float(n) for n in sorted(set(lower_quantiles + upper_quantiles + es_quantiles))
]
print(f"Quantiles: {all_quantiles}")


# =============================================================================
# 2. Define Quantile Loss
# =============================================================================
def quantile_loss(y_true, y_pred, alpha):
    residual = y_true - y_pred
    return np.maximum(alpha * residual, (alpha - 1) * residual).mean()


# =============================================================================
# 3. Define Optuna Objective
# =============================================================================
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "depth": trial.suggest_int("depth", 2, 16),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 100.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 128, 512),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 20
        ),
        "random_seed": 72,
        "early_stopping_rounds": 50,
        "verbose": False,
    }

    quantile_losses = {}

    for alpha in all_quantiles:
        print(f"Training for quantile: {alpha:.5f}")

        model = CatBoostRegressor(
            **params,
            loss_function=f"Quantile:alpha={alpha}",
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_feature_index,
        )

        preds = model.predict(X_val)
        loss = quantile_loss(y_val, preds, alpha)
        quantile_losses[alpha] = loss

    for alpha, loss in quantile_losses.items():
        trial.set_user_attr(f"QL_{alpha:.5f}", loss)

    avg_loss = np.mean(list(quantile_losses.values()))
    return avg_loss


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
        commit_header = f"Trial {trial.number} - Updated study DB"
        commit_body = (
            f"Trial {trial.number} finished with objective value: {trial.value}\n"
            f"Hyperparameters: {trial.params}\n"
        )
        for key, val in trial.user_attrs.items():
            commit_body += f"{key}: {val}\n"
        subprocess.run(
            ["git", "commit", "-m", commit_header, "-m", commit_body], check=True
        )
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")


# =============================================================================
# 5. Run Optuna Study
# =============================================================================
if not os.path.exists("optuna"):
    os.makedirs("optuna")

storage_url = "sqlite:///optuna/optuna.db"
study = optuna.create_study(
    direction="minimize",
    study_name="catboost_tuning",
    storage=storage_url,
    load_if_exists=True,
)

try:
    n_trials = int(sys.argv[1])
except:
    n_trials = 1000

study.optimize(objective, n_trials=n_trials, callbacks=[git_commit_callback])

# =============================================================================
# 6. Save Results
# =============================================================================
print("Best parameters:")
print(study.best_params)

best_params_df = pd.DataFrame([study.best_params])
best_params_df.to_csv("results/best_catboost_params.csv", index=False)

print("Done.")


# %%
