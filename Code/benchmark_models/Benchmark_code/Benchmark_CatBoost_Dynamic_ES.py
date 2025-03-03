# =============================================================================
# ========================= CATBOOST MODEL ==============================
# =============================================================================

# %%
# =============================================================================
# 1 Functions
# =============================================================================
# Importing required libraries
from typing import List
import pandas as pd
from sklearn.calibration import LabelEncoder    
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import sys
from tqdm import tqdm


# %%
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'shared')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from processing import get_lstm_train_test_new


# %%
# Define all ES Quantiles and sub-quantiles of interest

# ES quantiles of interest
ES_quantiles = [0.01, 0.025, 0.05, 0.95, 0.975, 0.99]
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
        return input_string[:-len(suffix)]
    if input_string.startswith(suffix):
        return input_string[len(suffix):]
    return input_string


# Function to ensure no quantile crossing - adapted for multi-asset
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
            col_nxt = quantile_columns_sorted[i+1]
            val_cur = df.at[idx, col_cur]
            val_nxt = df.at[idx, col_nxt]

            # if Q_cur > Q_nxt => crossing => fix
            if val_cur > val_nxt:
                df.at[idx, col_nxt] = val_cur  # or val_cur + small_epsilon

    return df

#Function to get a df from the preprocessed data
def combine_processed_data_into_df(window_size=1500):
    data = get_lstm_train_test_new(multiply_by_beta=False, include_fng=False, include_spx_data=False, include_returns=True)

    # load data anf covert to sensibe format
    X_train = data.train.X
    y_train = data.train.y
    X_val = data.validation.X
    y_val = data.validation.y
    tickers_train = data.train.tickers
    tickers_val = data.validation.tickers
    dates_train = data.train.dates
    dates_val = data.validation.dates

    print()
    print("Shapes after loading data:")
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_val shape: ", X_val.shape)
    print("y_val shape: ", y_val.shape)
    print("tickers_train shape: ", tickers_train.shape)
    print("tickers_val shape: ", tickers_val.shape)
    print("dates_train shape: ", dates_train.shape)
    print("dates_val shape: ", dates_val.shape)

    # Changing shape to not take in lags as features
    X_train = X_train[:, -1, : ]
    X_val = X_val[:, -1, : ]
    # remove all elements from X_train and Y_train except the last window_size entries
    X_train = X_train[-window_size:]
    y_train = y_train[-window_size:]
    tickers_train = tickers_train[-window_size:]
    dates_train = dates_train[-window_size:]

    # print the shapes of the data
    print()
    print("Shapes after changing:")
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_val shape: ", X_val.shape)
    print("y_val shape: ", y_val.shape)
    print("tickers_train shape: ", tickers_train.shape)
    print("tickers_val shape: ", tickers_val.shape)
    print("dates_train shape: ", dates_train.shape)
    print("dates_val shape: ", dates_val.shape)

    X_all = np.concatenate((X_train, X_val), axis=0)
    y_all = np.concatenate((y_train, y_val), axis=0)
    tickers_all = np.concatenate((tickers_train, tickers_val), axis=0)
    dates_all = np.concatenate((dates_train, dates_val), axis=0)

    # print the shapes of the data
    print()
    print("Final shapes:")
    print("X_all shape: ", X_all.shape)
    print("y_all shape: ", y_all.shape)
    print("tickers_all shape: ", tickers_all.shape)
    print("dates_all shape: ", dates_all.shape)

    
    

    # Build a DataFrame
    feat_cols = [f"feat_{i}" for i in range(X_all.shape[1])]
    df_big = pd.DataFrame(X_all, columns=feat_cols)
    df_big["Date"] = dates_all
    df_big["Symbol"] = tickers_all
    df_big["TrueY"] = y_all

    # Sort by date, then ticker
    df_big.sort_values(by=["Date", "Symbol"], inplace=True)

    # Encode Ticker as categorical (LabelEncoder) or leave as string category
    le = LabelEncoder()
    le.fit(df_big["Symbol"])
    df_big["Ticker_Cat"] = le.transform(df_big["Symbol"])

    # Return the big DF plus some info about which columns to use as features
    feature_cols = feat_cols + ["Ticker_Cat"]  # We'll pass these to CatBoost
    cat_feature_index = [len(feature_cols) - 1]  # Ticker_Cat is last => cat col
    return df_big, feature_cols, cat_feature_index


# %%
# =============================================================================
# 2. Model specification and forecasting
# =============================================================================

def train_and_predict_catboost(X_train, y_train, X_val, y_val, X_test, quantile_alpha, cat_features_indices: list[int]):
    """Trains a CatBoost model for a specific quantile and predicts on test data."""
    model = CatBoostRegressor(
        iterations=400,
        learning_rate=0.08,
        depth=4,
        l2_leaf_reg=3,
        loss_function=f'Quantile:alpha={quantile_alpha}',
        #loss_function='Quantile',
        #alpha=quantile_alpha,
        verbose=False,
        random_seed=72
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, cat_features=cat_features_indices)
    preds = model.predict(X_test)
    preds
    return preds

#    loss_function='Quantile',
#    alpha=quantile_alpha,

def run_quantile_regression_rolling_window(df_big: pd.DataFrame,
                                            feature_cols: List[str],
                                            cat_feature_index: List[int],
                                            window_size=1500,
                                            horizon=1,
                                            step=1,
                                            quantiles=[0.01, 0.025, 0.05, 0.95, 0.975, 0.99]
                                            ):
    """
    Function to perform quantile regression with CatBoost on a given dataset using a rolling window approach.

    Args:
        file_path (str): Path to the Excel file containing the data.
        sheet_name (str): Name of the sheet in the Excel file.
        date_column (str): Name of the column containing dates.
        target_column (str): Name of the column to be predicted (the target variable).
        feature_columns (list): List of column names to be used as features.
    """

    # 1) Gather the sorted unique dates
    unique_dates = df_big["Date"].sort_values().unique()

    # 2) Prepare the list to store the predictions
    predictions_list = []

    # 3) Iterate over the dates
    for i in tqdm(range(window_size, len(unique_dates) - horizon + 1, step), desc="Rolling Window Steps"):
        # The training window covers unique_dates in the range [i-window_size, i)
        train_start_idx = i - window_size  # inclusive
        train_end_idx = i  # exclusive
        train_dates_range = unique_dates[train_start_idx:train_end_idx]

        # The test date is unique_dates[i + horizon - 1]
        test_date_idx = i + horizon - 1
        if test_date_idx >= len(unique_dates):
            break
        test_date_val = unique_dates[test_date_idx]

        # 3) Build train/val subsets: all rows whose date is in train_dates_range
        df_window = df_big[df_big["Date"].isin(train_dates_range)].copy()
        if len(df_window) < 2:
            continue  # skip if not enough data

        # Sort window by date so we can do a time-based split
        df_window.sort_values("Date", inplace=True)
        # We'll do 80/20
        split_idx = int(0.8 * len(df_window))
        df_train = df_window.iloc[:split_idx]
        df_val = df_window.iloc[split_idx:]

        # 4) Build test subset: all rows whose Date == test_date_val
        df_test = df_big[df_big["Date"] == test_date_val].copy()
        if len(df_test) == 0:
            # Possibly no data for that date
            continue

        # 5) Extract X,y for train/val/test
        X_train = df_train[feature_cols]
        y_train = df_train["TrueY"].values
        X_val = df_val[feature_cols]
        y_val = df_val["TrueY"].values
        X_test = df_test[feature_cols]
        y_test = df_test["TrueY"].values  # for reference
        test_tickers = df_test["Symbol"].values
        test_dates = df_test["Date"].values

        # 6) For each quantile alpha, train a model, predict on test
        #    (Or you could build a single multi-quantile model, but typically we do one per alpha.)
        pred_quantiles = {}
        for alpha in tqdm(quantiles, desc="Quantiles", leave=False):
            y_pred = train_and_predict_catboost(
                X_train=X_train, 
                y_train=y_train, 
                X_val=X_val, 
                y_val=y_val, 
                X_test=X_test,
                quantile_alpha=alpha,
                cat_features_indices=cat_feature_index
            )
            pred_quantiles[alpha] = y_pred

        # 7) For each row in df_test, we build a dict with Ticker, Date, TrueY, and predicted quantiles
        for row_idx in range(len(df_test)):
            row_dict = {
                "Symbol": test_tickers[row_idx],
                "Date": test_dates[row_idx],
                "TrueY": y_test[row_idx],
            }
            for alpha in quantiles:
                row_dict[f"Quantile_{alpha:.3f}"] = pred_quantiles[alpha][row_idx]
            predictions_list.append(row_dict)

    # 8) Build final DataFrame with predictions
    df_preds = pd.DataFrame(predictions_list)
    # reorder columns
    q_cols = [c for c in df_preds.columns if c.startswith("Quantile_")]
    final_cols = ["Symbol", "Date", "TrueY"] + sorted(q_cols)
    df_preds = df_preds[final_cols].sort_values(["Date", "Symbol"])
    return df_preds


# %%
# =============================================================================
# 3. Running all model variants
# =============================================================================
def main_global_rolling_example_preds():
    # 1) Obtain your ProcessedData object with all your tickers/time series

    # 2) Combine into a single DataFrame
    df_big, feature_cols, cat_feature_idx = combine_processed_data_into_df(window_size=1500)
    print(df_big.head())

    # 3) Define desired quantiles
    ES_quantiles = [0.01, 0.025, 0.05, 0.95, 0.975, 0.99]

    # 4) Run the rolling-window approach
    df_predictions = run_quantile_regression_rolling_window(
        df_big=df_big,
        feature_cols=feature_cols,
        cat_feature_index=cat_feature_idx,
        window_size=1500,  # in *dates*, not rows
        horizon=1,
        step=1,
        quantiles=ES_quantiles,
    )

    print("Sample predictions:")
    print(df_predictions.head(20))

    # 5) If you have ensure_non_crossing or ES calculation, apply it now:
    df_no_cross = ensure_non_crossing_unified(df_predictions)
    # ...
    # df_no_cross.to_excel("global_rolling_predictions.xlsx", index=False)

    return df_no_cross


final_df = main_global_rolling_example_preds()
# %%
final_df

# %%
# Write the quantile predictions to a csv file for storage
predictions_copy = final_df.copy()
final_df.to_csv("Benchmark_Catboost_Dynamic_ES_quantiles.csv", index=False)

# %%
# =============================================================================
# 5. Estimate ES
# =============================================================================

def estimate_es_from_predictions(
    df_preds: pd.DataFrame,
    es_alphas=[0.01, 0.025, 0.05, 0.95, 0.975, 0.99],
    p=5
) -> pd.DataFrame:
    """
    Given a DataFrame df_preds that has columns:
      [Ticker, Date, TrueY, Quantile_0.xxx, Quantile_0.yyy, ...]
    for multiple sub-quantiles,
    compute the ES for each alpha in es_alphas by averaging
    the sub-quantile columns that the old logic used.

    The old logic:
      - For alpha < 0.5:
          new_quantiles = [alpha - (alpha * i / p) for i in range(p)]
        (descending in value)
      - For alpha >= 0.5:
          new_quantiles = [alpha + ((1 - alpha) * i / p) for i in range(p)]
        (ascending in value)

    We then average the columns "Quantile_xxx" matching those sub-quantiles
    to get a single column ES_alpha.

    Returns a new DataFrame with the same rows as df_preds,
    plus extra columns 'ES_0.xx' for each alpha in es_alphas.
    """

    # Work on a copy so we don't mutate the original
    df_out = df_preds.copy()

    for alpha in es_alphas:
        alpha_subs = []
        if alpha < 0.5:
            # E.g. alpha=0.01 => [0.01, 0.008, 0.006, 0.004, 0.002] etc.
            alpha_subs = [alpha - (alpha * i / p) for i in range(p)]
        else:
            # E.g. alpha=0.95 => [0.95, 0.96, 0.97, 0.98, 0.99] etc.
            alpha_subs = [alpha + ((1 - alpha) * i / p) for i in range(p)]

        # Round to 3 decimals to match columns like "Quantile_0.010"
        alpha_subs_3d = [f"{q:.3f}" for q in alpha_subs]

        # Build a list of the column names for these sub-quantiles
        sub_quantile_cols = [f"Quantile_{q3}" for q3 in alpha_subs_3d]

        # Check which of those columns exist in df_preds
        existing_cols = [c for c in sub_quantile_cols if c in df_out.columns]

        if not existing_cols:
            # If we didn't find any, we skip
            print(f"WARNING: No matching sub-quantile columns found for alpha={alpha}")
            continue

        # Average them row-wise => ES for alpha
        df_out[f"ES_{alpha:.3f}"] = df_out[existing_cols].mean(axis=1)

    return df_out

es_df = estimate_es_from_predictions(final_df)
es_df
# %%
# Write the ES predictions to a csv file for storage
es_df.to_csv("../../predictions/Benchmark_Catboost_Dynamic_ES_stocks.csv", index=False)
# %%
