# %%
import os

# Change to the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Change to Code folder so that imports and paths work correctly
os.chdir("../..")

# Add shared path and import
from shared.processing import get_lstm_train_test_new
import sys
import json
import optuna
import pandas as pd
import numpy as np
import subprocess
from lightgbm import LGBMRegressor, early_stopping


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

X_train = data.train.X[:, -1, :]
y_train = data.train.y
X_val = data.validation.X[:, -1, :]
y_val = data.validation.y

# Add ticker as feature
X_train = np.concatenate([X_train, data.train.tickers.reshape(-1, 1)], axis=1)
X_val = np.concatenate([X_val, data.validation.tickers.reshape(-1, 1)], axis=1)

feature_cols = list(data.train.df.columns.drop("ActualReturn")) + ["Symbol"]

X_train = pd.DataFrame(X_train, columns=feature_cols)
X_val = pd.DataFrame(X_val, columns=feature_cols)

# Set correct types
dtypes = ["float64"] * (X_train.shape[1] - 1) + ["category"]
for col, dtype in zip(X_train.columns, dtypes):
    X_train[col] = X_train[col].astype(dtype)
    X_val[col] = X_val[col].astype(dtype)

print("Data loaded.")

# %%
# =============================================================================
# 2. Quantile setup
# =============================================================================
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
# 3. Define Quantile Loss
# =============================================================================
def quantile_loss(y_true, y_pred, alpha):
    residual = y_true - y_pred
    return np.maximum(alpha * residual, (alpha - 1) * residual).mean()


# =============================================================================
# 4. Optuna Objective
# =============================================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 511),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 100.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 10),
        "max_bin": trial.suggest_int("max_bin", 128, 1024),
        "cat_l2": trial.suggest_float("cat_l2", 1e-4, 100.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        "random_state": 72,
        "boosting_type": "gbdt",
        "objective": "quantile",
        "verbose": -1,
    }

    print(f"Trial {trial.number}: {json.dumps(params, indent=2)}")

    quantile_losses = {}

    for alpha in all_quantiles:
        print(f"Training for quantile: {alpha:.5f}")
        model = LGBMRegressor(**params, alpha=alpha)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
            categorical_feature=[X_train.columns.get_loc("Symbol")],
        )
        preds = model.predict(X_val)
        loss = quantile_loss(y_val, preds, alpha)
        quantile_losses[alpha] = loss

    for alpha, loss in quantile_losses.items():
        trial.set_user_attr(f"QL_{alpha:.3f}", loss)

    avg_loss = np.mean(list(quantile_losses.values()))
    return avg_loss


# =============================================================================
# 5. Git Commit Callback
# =============================================================================
def git_commit_callback(study: optuna.Study, trial: optuna.Trial):
    print(f"Trial {trial.number} finished. Committing to Git.")
    try:
        subprocess.run(["git", "pull", "--no-edit"], check=True)
        subprocess.run(["git", "add", "optuna"], check=True)
        commit_header = f"LGB tuning trial {trial.number} - Updated study DB"
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
# 6. Run Optuna Study
# =============================================================================
if not os.path.exists("optuna"):
    os.makedirs("optuna")

storage_url = "sqlite:///optuna/optuna.db"
study = optuna.create_study(
    direction="minimize",
    study_name="lgb_tuning",
    storage=storage_url,
    load_if_exists=True,
)

try:
    n_trials = int(sys.argv[1])
except:
    n_trials = 100

study.optimize(objective, n_trials=n_trials, callbacks=[git_commit_callback])

# %%
