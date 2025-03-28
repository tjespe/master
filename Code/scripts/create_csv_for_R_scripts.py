# This codes generates CSV files for the R scripts to use.
# Based on the processing functions in the shared folder


# %%
# Define what to include, RV and/or IV
INCLUDE_RV = True
INCLUDE_IV = False

# version name is "RV" if only RV is included, "IV" if only IV is included, and "RV_IV" if both are included
VERSION = "RV" if INCLUDE_RV and not INCLUDE_IV else "IV" if not INCLUDE_RV and INCLUDE_IV else "RV_IV"
print(f"Version name: {VERSION}")
# %%
# Import the necessary libraries
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shared')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from processing import get_lstm_train_test_new

WINDOW = 1500

# %%
#######################################
# GET ONLY RV DATA - SET 1
#######################################

data = get_lstm_train_test_new(
        include_1min_rv = INCLUDE_RV,
        include_5min_rv = INCLUDE_RV,
        include_ivol_cols=(["10 Day Call IVOL"] if INCLUDE_IV else [])
        + (["Historical Call IVOL"] if INCLUDE_IV else [])
    )
data

# %%
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

# %%
# Changing shape to not take in lags as features
X_train = X_train[:, -1, : ]
X_val = X_val[:, -1, : ]
X_test = X_test[:, -1, : ]

# make a training set as df
feat_cols = [f"feat_{i}" for i in range(X_train.shape[1])]
df_train = pd.DataFrame(X_train, columns=feat_cols)
df_train["Date"] = dates_train
df_train["Symbol"] = tickers_train
df_train["Return"] = y_train

df_train

#%%
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
# %%
# remove all elements from the training set except the last window_size dates for each ticker
df_train = df_train.groupby("Symbol").tail(WINDOW)
df_train

# %%
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
# %%
# save to csv
df_big.to_csv(f"../data/processed_data_DB_{VERSION}.csv")