# =============================================================================
# ========================= XGBOOST MODEL=====================================
# =============================================================================

# %%
# define what version to run
INCLUDE_RV = True
INCLUDE_IV = False

# version is RV if INCLUDE_RV is True, IV if INCLUDE_IV is True, RV_IV if both are True
VERSION = (
    "RV"
    if INCLUDE_RV and not INCLUDE_IV
    else (
        "IV"
        if INCLUDE_IV and not INCLUDE_RV
        else "RV_IV" if INCLUDE_RV and INCLUDE_IV else "None"
    )
)

print(f"Running version: {VERSION}")
# %%
from sklearn.calibration import LabelEncoder
import xgboost

print(xgboost.__version__)
import pandas as pd
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import sys
from tqdm import tqdm


# %%
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from processing import get_lstm_train_test_new

# %%
# %%
# =============================================================================
# 1 Functions
# =============================================================================
import pandas as pd

run_models = []

# ES quantiles of interest
ES_quantiles = [0.01, 0.025, 0.05, 0.165, 0.835, 0.95, 0.975, 0.99]
p = 5  # 'p' defined as per requirement

# Create a list to store the quantiles
quantiles = []

for es in ES_quantiles:
    # For ES values less than 0.5, we calculate and append in descending order
    if es < 0.5:
        new_quantiles = [es - (es * i / p) for i in range(p)]
    # For ES values greater than or equal to 0.5, we calculate and append in ascending order
    else:
        new_quantiles = [es] + [es + ((1 - es) * i / p) for i in range(1, p)]
    quantiles.extend(new_quantiles)

# Format to 3 decimal places
quantiles = [f"{q:.3f}" for q in quantiles]


# Function to clean and normalize column names
def remove_suffix(input_string, suffix):
    # Check if the string ends with the specified suffix
    if input_string.endswith(suffix):
        # Slice the string to remove the suffix
        return input_string[: -len(suffix)]
    if input_string.startswith(suffix):
        return input_string[len(suffix) :]
    return input_string


# Function to ensure no quantile crossing
# Function to ensure no quantile crossing - adapted for multi-asset # FIX THIS TO DO THE RIGHT THING
def ensure_non_crossing_unified(df: pd.DataFrame) -> pd.DataFrame:
    quantile_columns = [col for col in df.columns if col.startswith("Quantile_")]

    def get_alpha(col_name):
        return float(col_name.split("_")[-1])

    # Sort columns by numeric alpha
    quantile_columns_sorted = sorted(quantile_columns, key=get_alpha)

    # Single forward pass for each row
    for idx in df.index:
        for i in range(len(quantile_columns_sorted) - 1):
            col_cur = quantile_columns_sorted[i]
            col_nxt = quantile_columns_sorted[i + 1]
            val_cur = df.at[idx, col_cur]
            val_nxt = df.at[idx, col_nxt]

            # if Q_cur > Q_nxt => crossing => fix
            if val_cur > val_nxt:
                df.at[idx, col_nxt] = (
                    val_cur  # or val_cur + small_epsilon #MAYBE add small epsilon
                )

    return df


# Function to get a df from the preprocessed data
def combine_processed_data_into_df(window_size=1500):
    data = get_lstm_train_test_new(
        include_1min_rv=INCLUDE_RV,
        include_5min_rv=INCLUDE_RV,
        include_ivol_cols=(["10 Day Call IVOL"] if INCLUDE_IV else [])
        + (["Historical Call IVOL"] if INCLUDE_IV else []),
    )

    X_train = data.train.X
    y_train = data.train.y
    X_val = data.validation.X
    y_val = data.validation.y
    X_test = data.test.X
    y_test = data.test.y
    tickers_train = data.train.tickers
    tickers_val = data.validation.tickers
    tickers_test = data.test.tickers
    dates_train = data.train.dates
    dates_val = data.validation.dates
    dates_test = data.test.dates

    # Changing shape to not take in lags as features
    X_train = X_train[:, -1, :]
    X_val = X_val[:, -1, :]
    X_test = X_test[:, -1, :]

    # make a training set as df
    feat_cols = [f"feat_{i}" for i in range(X_train.shape[1])]
    df_train = pd.DataFrame(X_train, columns=feat_cols)
    df_train["Date"] = dates_train
    df_train["Symbol"] = tickers_train
    df_train["TrueY"] = y_train

    print("df_train shape: ", df_train.shape)

    # Do the same for the validation set but keep all data
    feat_cols = [f"feat_{i}" for i in range(X_val.shape[1])]
    df_val = pd.DataFrame(X_val, columns=feat_cols)
    df_val["Date"] = dates_val
    df_val["Symbol"] = tickers_val
    df_val["TrueY"] = y_val
    print("df_val shape: ", df_val.shape)

    # count how many entries there are per ticker in the data
    print(df_val["Symbol"].value_counts())

    # merge the training and validation data since we now are going to predict the test set
    df_train = pd.concat([df_train, df_val], axis=0)
    print("df_train after merge shape: ", df_train.shape)

    # remove all elements from the training set except the last window_size dates for each ticker
    df_train = df_train.groupby("Symbol").tail(window_size)
    print("df_train after removing all except tail shape shape: ", df_train.shape)

    # count how many entries there are per ticker in the data
    print(df_train["Symbol"].value_counts())

    # count how many unique dates there are in the data
    print("Unique dates:", len(np.unique(df_train["Date"])))

    # print last and first date in the data
    print(f"Last date: {df_train['Date'].max()}")
    print(f"First date: {df_train['Date'].min()}")

    # do the same for the test set but keep all data
    feat_cols = [f"feat_{i}" for i in range(X_test.shape[1])]
    df_test = pd.DataFrame(X_test, columns=feat_cols)
    df_test["TrueY"] = y_test
    df_test["Date"] = dates_test
    df_test["Symbol"] = tickers_test
    print("df_test shape: ", df_test.shape)

    # merge the training and test data since we now are going to predict the test set
    df_big = pd.concat([df_train, df_test], axis=0)
    print("df_big shape: ", df_big.shape)

    # count how many entries there are per ticker in the data
    print(df_big["Symbol"].value_counts())

    # Sort by date, then ticker
    df_big.sort_values(by=["Date", "Symbol"], inplace=True)

    # Specify the symbol column as categorical
    df_big["Symbol"] = df_big["Symbol"].astype("category")

    # Return the big DF plus some info about which columns to use as features
    # Return the big DF plus some info about which columns to use as features
    feature_cols = feat_cols + ["Symbol"]
    cat_feature_index = [
        df_big.columns.get_loc("Symbol")
    ]  # We'll pass this to XGBoost to indicate that Symbol is a categorical feature
    return df_big, feature_cols, cat_feature_index


# %%
# =============================================================================
# 2 Model Specification and Forecasting
# =============================================================================
def train_and_predict_xgb(
    X_train, y_train, X_val, y_val, X_test, quantile_alpha, label=None
):
    """Trains an XGBoost model for a specific quantile and predicts."""
    model = XGBRegressor(
        quantile_alpha=quantile_alpha,
        objective="reg:quantileerror",
        verbosity=0,
        random_state=72,
        early_stopping_rounds=20,
        enable_categorical=True,  # Enable categorical features,
        tree_method="hist",
        n_estimators=169,
        max_depth=2,
        learning_rate=0.21714914917425182,
        reg_lambda=5.431533853816789,
        min_child_weight=256.3743336159067,
        subsample=0.9786341633008059,
        colsample_bytree=0.9352284063335753,
        gamma=0.0017916989650043633,
        max_bin=128,
        grow_policy="lossguide",
    )

    # Early stopping for better generalization
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Save model if label is provided
    if label is not None:
        model.save_model(f"trained/xgb_{VERSION}_{label}_{quantile_alpha}.json")

    return model.predict(X_test)


def run_quantile_regression_rolling_window(
    df_big, feature_cols, window_size=1500, horizon=1, step=1, quantiles=quantiles
):
    """
    Function to perform quantile regression on a given dataset using a rolling window approach.

    Args:
        file_path (str): Path to the Excel file containing the data.
        sheet_name (str): Name of the sheet in the Excel file.
        date_column (str): Name of the column containing dates.
        target_column (str): Name of the column to be predicted (the target variable).
        feature_columns (list): List of column names to be used as features.
    """
    # 1) Gather the sorted unique dates and praper the predictions list
    unique_dates = df_big["Date"].sort_values().unique()
    predictions_list = []

    # 2) Iterate over the dates
    for i in tqdm(
        range(window_size, len(unique_dates) - horizon + 1, step),
        desc="Rolling Window Steps",
    ):
        train_start_idx = i - window_size
        train_end_idx = i
        train_dates = unique_dates[train_start_idx:train_end_idx]

        # test data
        test_date_idx = i + horizon - 1
        if test_date_idx >= len(unique_dates):
            break
        test_date = unique_dates[test_date_idx]

        # 3) Build train/val subsets
        df_window = df_big[df_big["Date"].isin(train_dates)].copy()
        if len(df_window) < 2:
            continue

        df_window.sort_values("Date", inplace=True)

        split_idx = int(len(df_window) * 0.8)
        df_train = df_window.iloc[:split_idx]
        df_val = df_window.iloc[split_idx:]

        # 4) Build test subsets: all rows whose Date == test_date
        df_test = df_big[df_big["Date"] == test_date].copy()
        if len(df_test) == 0:
            continue

        # 5) Extract X/y for train/val/test
        X_train = df_train[feature_cols]
        y_train = df_train["TrueY"].values
        X_val = df_val[feature_cols]
        y_val = df_val["TrueY"].values
        X_test = df_test[feature_cols]
        y_test = df_test["TrueY"].values  # for reference
        test_tickers = df_test["Symbol"].values
        test_dates = df_test["Date"].values

        # 6) For each quantile, train and predict on test
        pred_quantiles = {}
        for q in tqdm(quantiles, desc="Quantile Predictions", leave=False):
            pred = train_and_predict_xgb(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                quantile_alpha=q,
                label=test_date.strftime("%Y%m%d"),
            )
            pred_quantiles[q] = pred

        # 7) For each row in df_test, we build a dict with Ticker, Date, TrueY, and predicted quantiles
        for row_idx in range(len(df_test)):
            row_dict = {
                "Date": test_dates[row_idx],
                "Symbol": test_tickers[row_idx],
                "TrueY": y_test[row_idx],
            }
            for q in quantiles:
                row_dict[f"Quantile_{q}"] = pred_quantiles[q][row_idx]
            predictions_list.append(row_dict)

    # 8) Build the final DataFrame with predictions
    df_predictions = pd.DataFrame(predictions_list)
    # reorder columns
    q_cols = [c for c in df_predictions.columns if c.startswith("Quantile_")]
    final_cols = ["Symbol", "Date", "TrueY"] + sorted(q_cols)
    df_predictions = df_predictions[final_cols].sort_values(["Date", "Symbol"])
    return df_predictions


# %%
# =============================================================================
# 3 Run all model variants
# =============================================================================
# Define a list to store the output filenames
def main_global_rolling_preds():
    # 1) Load the data
    df_big, feature_cols, cat_feature_index = combine_processed_data_into_df(
        window_size=1500
    )
    print("Data loaded and processed")
    print(df_big.head())

    # 2) Run the quantile regression with the current feature set
    df_preds = run_quantile_regression_rolling_window(
        df_big=df_big,
        feature_cols=feature_cols,
        window_size=1500,
        horizon=1,
        step=1,
        quantiles=quantiles,
    )

    # 3) Print sample predictions
    print("Sample predictions:")
    print(df_preds.head(20))

    # 4 Ensure non-crossing
    df_no_cross = ensure_non_crossing_unified(df_preds)

    return df_no_cross


# %%
# =============================================================================
# Estimate ES
# =============================================================================
def estimate_es_from_predictions(
    df_preds: pd.DataFrame, es_quantiles: list, p: int = 5
) -> pd.DataFrame:

    df_out = df_preds.copy()
    print(es_quantiles)
    for alpha in es_quantiles:
        print(alpha)
        alpha_subs = []
        if alpha < 0.5:
            # E.g. alpha=0.01 => [0.01, 0.008, 0.006, 0.004, 0.002] etc.
            alpha_subs = [alpha - (alpha * i / p) for i in range(p)]
        else:
            # E.g. alpha=0.95 => [0.95, 0.96, 0.97, 0.98, 0.99] etc.
            alpha_subs = [alpha + ((1 - alpha) * i / p) for i in range(p)]

        alpha_subs_3sf = [f"{a:.3f}" for a in alpha_subs]
        sub_quantile_cols = [f"Quantile_{a}" for a in alpha_subs_3sf]
        existing_cols = [c for c in sub_quantile_cols if c in df_out.columns]

        if not existing_cols:
            print(f"WARNING: No columns found for alpha={alpha}")
            continue

        # Average them row-wise => ES for alpha
        df_out[f"ES_{alpha:.3f}"] = df_out[existing_cols].mean(axis=1)

    return df_out


# %%
# Make predictions
if __name__ == "__main__":
    final_df = main_global_rolling_preds()
    final_df

    # %%
    es_df = estimate_es_from_predictions(final_df, ES_quantiles, p)
    es_df

    # %%
    # Write to csv
    es_df.to_csv(f"../../predictions/XGBoost_{VERSION}_4y.csv", index=False)
