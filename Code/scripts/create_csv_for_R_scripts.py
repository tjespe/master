# This codes generates CSV files for the R scripts to use.
# Based on the processing functions in the shared folder

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

data = get_lstm_train_test_new(multiply_by_beta=False,
                                    include_fng=False,
                                    include_spx_data=False,
                                    include_returns=False,
                                    include_industry=False,
                                    include_garch=False,
                                    include_beta=False,
                                    include_others=False)
data
# %%
X_train = data.train.X
y_train = data.train.y
X_val = data.validation.X
y_val = data.validation.y
tickers_train = data.train.tickers
tickers_val = data.validation.tickers
dates_train = data.train.dates
dates_val = data.validation.dates

# %%
# Changing shape to not take in lags as features
X_train = X_train[:, -1, : ]
X_val = X_val[:, -1, : ]
# remove all elements from X_train and Y_train except the last window_size entries
X_train = X_train[-WINDOW:]
y_train = y_train[-WINDOW:]
tickers_train = tickers_train[-WINDOW:]
dates_train = dates_train[-WINDOW:]

# %%
X_all = np.concatenate((X_train, X_val), axis=0)
y_all = np.concatenate((y_train, y_val), axis=0)
tickers_all = np.concatenate((tickers_train, tickers_val), axis=0)
dates_all = np.concatenate((dates_train, dates_val), axis=0)
#%%
feat_cols = [f"feat_{i}" for i in range(X_all.shape[1])]
df_big = pd.DataFrame(X_all, columns=feat_cols)
df_big["Date"] = dates_all
df_big["Ticker"] = tickers_all
df_big["Return"] = y_all

df_big

# %%
data = df_big
# %%
# save to csv
data.to_csv("../data/processed_data_RV_only_for_DB.csv")

# %%
# create a set with only the AAPL ticker to test the R script
data_aapl = data[data["Ticker"] == "AAPL"]
data_aapl.to_csv("../data/processed_data_RV_only_for_DB_AAPL.csv")
# %%
#######################################
# GET ONLY RV + MAKRO DATA - SET 2
#######################################