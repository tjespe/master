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

# make a training set as df
feat_cols = [f"feat_{i}" for i in range(X_train.shape[1])]
df_train = pd.DataFrame(X_train, columns=feat_cols)
df_train["Date"] = dates_train
df_train["Ticker"] = tickers_train
df_train["Return"] = y_train

df_train
# %%
# remove all elements from the training set except the last window_size dates for each ticker
df_train = df_train.groupby("Ticker").tail(WINDOW)
df_train

# %%
# count how many entries there are per ticker in the data
df_train["Ticker"].value_counts()
# %%
# count how many unique dates there are in the data
len(np.unique(df_train["Date"]))

# %%
# print last and first date in the data
print(f"Last date: {df_train['Date'].max()}")
print(f"First date: {df_train['Date'].min()}")


# %%
# Do the same for the validation set but keep all data
feat_cols = [f"feat_{i}" for i in range(X_val.shape[1])]
df_val = pd.DataFrame(X_val, columns=feat_cols)
df_val["Date"] = dates_val
df_val["Ticker"] = tickers_val
df_val["Return"] = y_val
df_val

# %%
# count how many entries there are per ticker in the data
df_val["Ticker"].value_counts()

# %%
# merge the training and validation data
df_big = pd.concat([df_train, df_val], axis=0)
df_big

# %%
# count how many entries there are per ticker in the data
df_big["Ticker"].value_counts()

# %%
# save to csv
df_big.to_csv("../data/processed_data_RV_only_for_DB.csv")



# %%
#######################################
# GET ONLY AAPL data to test the model
#######################################

# filter the df:big to only include AAPL data
df_big = df_big[df_big["Ticker"] == "AAPL"]
df_big

# %%
# count how many entries there are per ticker in the data
df_big["Ticker"].value_counts()
# %%
# save to csv
df_big.to_csv("../data/processed_data_AAPL_only_for_DB.csv")
# %%
