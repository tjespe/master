# %% 1) Import packages
import time
import numpy as np
import pandas as pd

# %% 2) Read in the data and load the data
df_stock_data = pd.read_csv("data/dow_jones/raw_data/dow_jones_stocks_19022025v2.csv")
df_total_return = pd.read_csv("data/dow_jones/raw_data/dow_jones_stocks_total_return_1990_to_today_19022025.csv")
print("✅ Data loaded successfully.")
print("Stocks Data Shape:", df_stock_data.shape)
print("Total Return Data Shape:", df_total_return.shape)

# %% 2) Dropp all rows where the date is before 1990-01-01
df_stock_data = df_stock_data[df_stock_data["Date"] >= "1990-01-01"]
df_total_return = df_total_return[df_total_return["Date"] >= "1990-01-01"]
print("✅ Rows before 1990-01-01 dropped successfully.")
print("Stocks Data Shape:", df_stock_data.shape)
print("Total Return Data Shape:", df_total_return.shape)

# %% Drop all dubplicated rows in each data frame
df_stock_data = df_stock_data.drop_duplicates()
df_total_return = df_total_return.drop_duplicates()
print("✅ Duplicated rows dropped successfully.")

# print how many rows where dropepd in each data frame
print("Stocks Data Shape:", df_stock_data.shape)
print("Total Return Data Shape:", df_total_return.shape)


# %%  3) Display the first few rows of each DataFrame
print("Stocks Data:")
df_stock_data

# %% Display the first few rows of the total return DataFrame|
print("\nTotal Return Data:")
df_total_return

# %% 
df_total_return[df_total_return["Total Return"].isnull()]

# %% find all rows that are not in both dataframes
# Find rows in df_stock_data that are not in df_total_return
stocks_not_in_total_return = df_stock_data[~df_stock_data.set_index(['Date', 'Symbol']).index.isin(df_total_return.set_index(['Date', 'Symbol']).index)]

# Find rows in df_total_return that are not in df_stock_data
total_return_not_in_stocks = df_total_return[~df_total_return.set_index(['Date', 'Symbol']).index.isin(df_stock_data.set_index(['Date', 'Symbol']).index)]

#  %% Display the first few rows of each DataFrame
print("Rows in Stocks Data not in Total Return Data:")
stocks_not_in_total_return

# %% Display the first few rows of each DataFrame
print("\nRows in Total Return Data not in Stocks Data:")
total_return_not_in_stocks

# %% Print checkmark if dataframs have ram numer of rows and an cross if not
if stocks_not_in_total_return.shape[0] == 0 and total_return_not_in_stocks.shape[0] == 0:
    print("✅ Dataframes have the same number of time entreis (rows).")
else:
    print("❌ Dataframes do not have the same number of rows.")
    print("Stocks Data Length:", len(df_stock_data))
    print("Total Return Data Length:", len(df_total_return))
    print("Stocks Data Length - Total Return Data Length:", len(df_stock_data) - len(df_total_return))
# %% For each of teh rows in the data frame that are not in the total return data frame, retrive the missing data induvidually and store it in a data frame
# Define the missing data DataFrame
missing_data = pd.DataFrame(columns=["Date", "Symbol", "TR.TotalReturn"])

import refinitiv.data as rd
from datetime import datetime

rd.open_session()

# Define the missing data retrieval function
def retrieve_missing_data(row):

    # Define the request parameters
    kwargs = {
        "universe": row["Symbol"],
        "fields": ["TR.TotalReturn"],
        "start": datetime.strptime(str(row["Date"]), "%Y-%m-%d"), 
        "end": datetime.strptime(str(row["Date"]), "%Y-%m-%d"),
        "interval": "daily"
    }

    # Retry mechanism: Try fetching twice before skipping
    for attempt in range(2):  # Try up to 2 times
        try:
            data = rd.get_history(**kwargs)
            if data is not None and not data.empty:
                missing_data.loc[len(missing_data)] = [row["Date"], row["Symbol"], data.iloc[0]["Total Return"]]
                print(f"    ✅ Data retrieved for {row['Symbol']} on {row['Date']}")
            else:
                print(f"    ⚠️ No data returned for {row['Symbol']} on {row['Date']}")
            break  # Exit retry loop if successful

        except Exception as e:
            print(f"    ❌ Attempt {attempt + 1}: Error fetching data: {e}")
            if attempt < 1:  # Only sleep before retry, not after last attempt
                time.sleep(10)  # Wait before retrying

    else:  # If both attempts fail, skip this period
        print(f"    ❌ Skipping {row['Symbol']} on {row['Date']} due to repeated errors.")

# Loop over each row in stocks_not_in_total_return
for index, row in stocks_not_in_total_return.iterrows():
    retrieve_missing_data(row)  # Retrieve missing data

# %% Display the first few rows of the missing data DataFrame
print("\nMissing Data:")
missing_data


# %% 3) Merge the two DataFrames on 'Date' and 'Symbol'
df_merged = pd.merge(df_stock_data, df_total_return, on=['Date', 'Symbol'], how='inner')

# Display the first few rows of the merged DataFrame
print("Merged Data:")
df_merged

# %% find all rows in each of the date frams that is not in the merged dateframe
# Find rows in df_stock_data that are not in df_merged
stocks_not_in_merged = df_stock_data[~df_stock_data.set_index(['Date', 'Symbol']).index.isin(df_merged.set_index(['Date', 'Symbol']).index)]

# Find rows in df_total_return that are not in df_merged
total_return_not_in_merged = df_total_return[~df_total_return.set_index(['Date', 'Symbol']).index.isin(df_merged.set_index(['Date', 'Symbol']).index)]



# %%
stocks_not_in_merged

# %% For stock_no_in_merge merge the data from the missing data data frame into stock not in merge n and then into main datafreame
# Merge the missing data into stocks_not_in_merged TODO SOMTHING WRONG HERE?
stocks_not_in_merged = pd.merge(stocks_not_in_merged, missing_data, on=['Date', 'Symbol'], how='inner')
# Rename the column to match the merged DataFrame
stocks_not_in_merged.rename(columns={"TR.TotalReturn": "Total Return"}, inplace=True)


# Display the first few rows of the merged DataFrame
stocks_not_in_merged

# %%Merge the missing data into the main merged DataFrame
df_merged = df_merged.reset_index(drop=True)
stocks_not_in_merged = stocks_not_in_merged.reset_index(drop=True)

df_final = pd.concat([df_merged, stocks_not_in_merged], ignore_index=True)
df_final

# %% Rename colums to be called Date, Symbol, Close actual, High, Low, Open, Volume, TotalReturn
df_final.rename(columns={"CLOSE": "Close Actual", "HIGH": "High", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume", "Total Return": "Total Return"}, inplace=True)
df_final.sort_values(["Symbol", "Date"], inplace=True)
df_final

# %%  drop 2003-10-25 for symbol AMGN.O as it is not a trading day

df_final = df_final.drop(df_final[(df_final["Date"] == "2003-10-25") & (df_final["Symbol"] == "AMGN.O")].index)
df_final
# %% displauy all rows where column "TotalReturn" is null
df_final[df_final["Total Return"].isnull()]

# %%
# Sort the dataframe by Symbol and Date in descending order
df_final = df_final.sort_values(by=["Symbol", "Date"], ascending=[True, False])

# Initialize Close column
df_final["Close"] = np.nan

# Process each asset separately
for symbol in df_final["Symbol"].unique():
    asset_df = df_final[df_final["Symbol"] == symbol].copy()
    
    # Start from the most recent date where Close = Close Actual
    asset_df.loc[asset_df.index[0], "Close"] = asset_df.iloc[0]["Close Actual"]
    
    # Compute Close recursively moving backward in time
    for i in range(1, len(asset_df)):
        prev_adj_close = asset_df.iloc[i-1]["Close"]
        total_return = asset_df.iloc[i-1]["Total Return"]  # Use the TotalReturn of the next date
        
        asset_df.loc[asset_df.index[i], "Close"] = prev_adj_close / (1 + (total_return /100))
    
    # Update the main dataframe
    df_final.loc[asset_df.index, "Close"] = asset_df["Close"]
    print(f"✅ {symbol} processed successfully.")
# Sort back to original order (ascending date)
df_final = df_final.sort_values(by=["Symbol", "Date"], ascending=[True, True])

# %%
df_final
# %% Save the final DataFrame to a CSV file
df_final.to_csv("data/dow_jones/processed_data/dow_jones_stocks_1990_to_today_19022025_cleaned.csv", index=False)

# %%
